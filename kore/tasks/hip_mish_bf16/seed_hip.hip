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

template <typename scalar_t>
__global__ void kore_unary_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ y, int64_t n) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    float v = static_cast<float>(x[i]);
    y[i] = static_cast<scalar_t>(v * tanhf(v > 20.0f ? v : log1pf(expf(v))));
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor x) {
  KORE_CHECK_INPUT(x);
  torch::Tensor y = torch::empty_like(x);
  int64_t n = x.numel();
  if (n == 0) return y;
  const int threads = 256;
  int blocks = static_cast<int>(kore_ceil_div(n, threads));
  if (blocks > 8192) blocks = 8192;
  if (blocks < 1) blocks = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "kore_unary", ([&] {
        hipLaunchKernelGGL((kore_unary_kernel<scalar_t>), dim3(blocks),
                           dim3(threads), 0, kore_stream(),
                           x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP elementwise seed");
}
