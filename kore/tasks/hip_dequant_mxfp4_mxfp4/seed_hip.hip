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

constexpr int KORE_MX_BLOCK = 32;
constexpr int KORE_DEQ_THREADS = 256;

// OCP E2M1 magnitudes for codes 0..7; bit 3 carries the sign.
__device__ __constant__ float kore_e2m1[8] =
    {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// MXFP4 -> bf16/fp16/fp32. Exact: an integer nibble unpack and a power-of-two
// scale, so any disagreement with the oracle is a real bug and not rounding.
template <typename scalar_t>
__global__ void kore_dequant_mxfp4_kernel(const uint8_t* __restrict__ packed,
                                          const uint8_t* __restrict__ e8m0,
                                          scalar_t* __restrict__ y,
                                          int64_t n_cols) {
  const int64_t row = blockIdx.x;
  const uint8_t* pr = packed + row * (n_cols / 2);
  const uint8_t* sr = e8m0 + row * (n_cols / KORE_MX_BLOCK);
  scalar_t* yr = y + row * n_cols;

  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_DEQ_THREADS) {
    const uint8_t byte = pr[c >> 1];
    const uint8_t code = (c & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    const float mag = kore_e2m1[code & 0x7];
    const float sign = (code & 0x8) ? -1.0f : 1.0f;
    const float scale = exp2f(static_cast<float>(sr[c / KORE_MX_BLOCK]) - 127.0f);
    yr[c] = static_cast<scalar_t>(sign * mag * scale);
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor packed, torch::Tensor e8m0,
                      c10::string_view out_dtype) {
  KORE_CHECK_INPUT(packed);
  KORE_CHECK_INPUT(e8m0);
  TORCH_CHECK(packed.dim() == 2 && e8m0.dim() == 2, "expected 2-D operands");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "packed codes must be uint8");
  TORCH_CHECK(e8m0.scalar_type() == at::kByte, "E8M0 exponents must be uint8");
  const int64_t rows = packed.size(0);
  const int64_t cols = packed.size(1) * 2;
  TORCH_CHECK(cols % KORE_MX_BLOCK == 0,
              "MXFP4 requires a column count that is a multiple of 32");
  TORCH_CHECK(e8m0.size(0) == rows && e8m0.size(1) == cols / KORE_MX_BLOCK,
              "one E8M0 exponent per 32 columns is required");

  at::ScalarType st = at::kBFloat16;
  if (out_dtype == "fp16") st = at::kHalf;
  else if (out_dtype == "fp32") st = at::kFloat;
  else TORCH_CHECK(out_dtype == "bf16", "out_dtype must be bf16, fp16 or fp32");

  torch::Tensor y = torch::empty({rows, cols}, packed.options().dtype(st));
  if (rows == 0 || cols == 0) return y;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, st, "kore_dequant_mxfp4", ([&] {
        hipLaunchKernelGGL((kore_dequant_mxfp4_kernel<scalar_t>),
                           dim3(static_cast<int>(rows)),
                           dim3(KORE_DEQ_THREADS), 0, kore_stream(),
                           packed.data_ptr<uint8_t>(), e8m0.data_ptr<uint8_t>(),
                           y.data_ptr<scalar_t>(), cols);
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP MXFP4 dequantize seed");
}
