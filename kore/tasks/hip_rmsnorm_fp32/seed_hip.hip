// KORE HIP seed. Correct, deliberately unoptimised: see kore/tasks/hip_ops.py.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

#define KORE_CHECK_INPUT(t)                                                   \
  TORCH_CHECK((t).is_cuda(), #t " must be a device tensor");                   \
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")

namespace {

inline int64_t kore_ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline hipStream_t kore_stream() {
  return c10::hip::getCurrentHIPStream().stream();
}

}  // namespace

namespace {

constexpr int KORE_ROW_THREADS = 256;

// Block reduction over KORE_ROW_THREADS lanes. Plain shared-memory tree: no
// cross-lane intrinsics, no packed math -- exactly the kind of lowering the
// model is being asked to improve.
// ``value`` must already be the reduction identity on lanes with no work, which
// is what the callers' initialisers guarantee (-INFINITY for max, 0 for sum).
template <typename Op>
__device__ float kore_block_reduce(float value, Op op) {
  __shared__ float scratch[KORE_ROW_THREADS];
  int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int span = KORE_ROW_THREADS / 2; span > 0; span >>= 1) {
    if (tid < span) scratch[tid] = op(scratch[tid], scratch[tid + span]);
    __syncthreads();
  }
  float out = scratch[0];
  __syncthreads();
  return out;
}

struct KoreAdd { __device__ float operator()(float a, float b) const { return a + b; } };
struct KoreMax { __device__ float operator()(float a, float b) const { return a > b ? a : b; } };

}  // namespace

namespace {

template <typename scalar_t>
__global__ void kore_rmsnorm_kernel(const scalar_t* __restrict__ x,
                                    const scalar_t* __restrict__ w,
                                    scalar_t* __restrict__ y,
                                    int64_t n_cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * n_cols;
  scalar_t* yr = y + row * n_cols;

  float local = 0.0f;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    float v = static_cast<float>(xr[c]);
    local += v * v;
  }
  const float sumsq = kore_block_reduce(local, KoreAdd{});
  const float scale = rsqrtf(sumsq / static_cast<float>(n_cols) + eps);

  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    yr[c] = static_cast<scalar_t>(static_cast<float>(xr[c]) * scale *
                                  static_cast<float>(w[c]));
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor x, torch::Tensor w, double eps) {
  KORE_CHECK_INPUT(x);
  KORE_CHECK_INPUT(w);
  TORCH_CHECK(x.dim() == 2, "expected a 2-D [rows, cols] tensor");
  TORCH_CHECK(w.numel() == x.size(1), "weight must be one value per column");
  torch::Tensor y = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  if (rows == 0 || cols == 0) return y;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "kore_rmsnorm", ([&] {
        hipLaunchKernelGGL((kore_rmsnorm_kernel<scalar_t>),
                           dim3(static_cast<int>(rows)),
                           dim3(KORE_ROW_THREADS), 0, kore_stream(),
                           x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                           y.data_ptr<scalar_t>(), cols,
                           static_cast<float>(eps));
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP RMSNorm seed");
}
