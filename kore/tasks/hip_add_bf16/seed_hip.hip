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
__global__ void kore_binary_kernel(const scalar_t* __restrict__ a,
                                   const scalar_t* __restrict__ b,
                                   scalar_t* __restrict__ out, int64_t n) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const float x = static_cast<float>(a[i]);
    const float y = static_cast<float>(b[i]);
    out[i] = static_cast<scalar_t>(x + y);
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
  KORE_CHECK_INPUT(a);
  KORE_CHECK_INPUT(b);
  TORCH_CHECK(a.sizes() == b.sizes(), "operands must have equal shape");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "operand dtype mismatch");
  torch::Tensor out = torch::empty_like(a);
  int64_t n = a.numel();
  if (n == 0) return out;
  const int threads = 256;
  int blocks = static_cast<int>(kore_ceil_div(n, threads));
  if (blocks > 8192) blocks = 8192;
  if (blocks < 1) blocks = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, a.scalar_type(), "kore_binary", ([&] {
        hipLaunchKernelGGL((kore_binary_kernel<scalar_t>), dim3(blocks),
                           dim3(threads), 0, kore_stream(),
                           a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                           out.data_ptr<scalar_t>(), n);
      }));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP binary pointwise seed");
}
