"""HIP C++ task definitions: oracles, production baselines, and compiling seeds.

Why these tasks exist
---------------------
Triton has a measured codegen ceiling on AMD.  HipKittens (MLSys 2026,
arXiv 2511.08083) reports Triton at 1.3-3.0x slower than C++ tile primitives on
BF16 GEMM, and attributes it to ROCm/Triton "struggling with register lifetime
tracking and lowering memory accesses to the most performant intrinsics".  A
model that only emits Triton cannot cross that ceiling no matter how good it
gets, and both HIP bars in AgentKernelArena (``hip2hip`` 6.69x, ``torch2hip``
6.89x) are C++ targets.  So the task pool needs HIP C++.

What a HIP task is
------------------
Structurally identical to a Triton task -- same ``task.yaml`` schema, same
Python ``reference.py`` oracle, same ``driver.py`` shim into
:func:`kore.tasks._genops.driver_main`, so it inherits the full publication
timing protocol and the whole anti-hack battery.  The only difference is
``backend: hip``, which makes the environment stage the candidate as
``kernel.hip`` and compile it with hipcc (see :mod:`kore.env.hip_toolchain`).

The candidate ABI is the one AgentKernelArena uses, because it is already
validated on this hardware and it is what the published Opus bars were measured
against: a ``.hip`` file that binds a ``forward`` entry point with
``PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`` and takes/returns ``torch::Tensor``.

Provenance and benchmark hygiene
--------------------------------
These kernels are written here, from the operator definitions.  No source is
copied from AgentKernelArena, and the seeds deliberately do not reuse its
reference implementations, because AKA is the benchmark this project scores
against to compare with Opus -- see :mod:`kore.tasks.hip_provenance` for the
operator-level overlap disclosure that keeps that comparison honest.

Seeds are correct but deliberately unoptimised: a grid-stride elementwise loop,
a one-block-per-row reduction, a naive shared-memory GEMM tile.  That is the
point -- the seed has to compile and pass its own gate so the task is runnable,
while leaving the tiling, vectorisation and intrinsic selection that HipKittens
shows to be worth 1.3-3.0x as the thing the model has to discover.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F

#: Storage dtype IDs -> torch dtypes.  Keys match the ``dtype`` field in
#: ``task.yaml`` and must be members of ``kore.tasks.taxonomy.TRAIN_DTYPES`` for
#: a task to be trainable rather than eval-only.
TORCH_DTYPES: Mapping[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    # Low-precision IDs name the task's DEFINING format, which is not always the
    # tensor dtype an op's inputs carry.  ``fp8_e4m3fn`` is the OCP fp8 that
    # gfx950 implements natively (verified: a bf16 -> fp8_e4m3fn round trip on
    # this device is exact to 0.125, its own eps).  ``mxfp4`` names OCP MXFP4,
    # whose storage is packed uint8 nibbles plus E8M0 exponents -- see
    # MXFP4_NOTE for why it cannot be a torch dtype here -- so the value below is
    # the dtype the dequantized OUTPUT carries.
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "mxfp4": torch.bfloat16,
}

#: Why MXFP4 is carried as packed uint8 rather than a torch dtype, and why there
#: are no MXFP6 tasks.  Measured on this host (torch 2.10.0+rocm7.0, HIP 7.0.51831,
#: gfx950):
#:
#: * ``torch.float4_e2m1fn_x2`` exists as a dtype but cannot be converted:
#:   ``x.to(torch.float4_e2m1fn_x2)`` raises "copy_() does not support casting
#:   Float4_e2m1fn_x2 to different types".  It is storage-only in this build, so a
#:   task built on it could not construct its own inputs.  MXFP4 therefore uses
#:   the same representation the existing ``gemm_mxfp4`` task uses: E2M1 codes
#:   packed two-per-byte plus a biased-127 E8M0 exponent per 32 elements.
#: * ``torch.float6_e2m3fn`` and ``torch.float6_e3m2fn`` do not exist at all.
#:   MI355X's 10.1 PFLOP figure covers MXFP6 as well as MXFP4, but there is no way
#:   to express an MXFP6 tensor in this stack, so no MXFP6 task is added.  Adding
#:   one would mean hand-rolling a 6-bit packing whose oracle no library can
#:   corroborate -- a task that looks like coverage and verifies nothing.
MXFP4_NOTE = "OCP MXFP4: E2M1 codes packed 2/byte + biased-127 E8M0 per 32 elements"

#: OCP E2M1 magnitudes for codes 0..7; bit 3 is the sign.
E2M1_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
MX_BLOCK = 32
#: The fp8 e4m3 dynamic range the per-row scale targets.
FP8_E4M3_MAX = 448.0

#: Compute-capability floor.  Every seed below is plain HIP C++ plus the torch
#: C++ extension ABI, so it builds on any CDNA target; the tasks declare gfx950
#: because that is the product target and the timing baseline.
GPU_TARGET = "gfx950"

# --------------------------------------------------------------------------- #
# Shape lanes
#
# Every declared shape is BENCHED and must clear the publication timing gate, so
# shapes are not free -- they were measured on gfx950 (MI350X) and chosen against
# two constraints, not guessed:
#
# * **Timeable.** Candidate, baseline and paired-ratio coefficients of variation
#   all stay under the 3% publication threshold.  The first shape set tried here
#   failed 11 of 21 tasks on exactly this gate.
# * **Informative.** Both implementations must sit well above kernel-launch
#   latency, or the ratio measures the launch floor rather than the kernel.  The
#   first attempt included ``{M: 1, N: 14336}``, where candidate and baseline both
#   measured 6.9 us and the speedup was a flat 1.000x for *any* kernel -- a shape
#   that cannot teach anything and would have looked like a passing task.
#
# Measured CVs for these lanes are all <= 2.7% (most under 1%); see
# scripts/verify_hip_tasks_e2e.py for the reproduction.
# --------------------------------------------------------------------------- #

#: Memory-bound elementwise / gated-activation lanes.  Every lane is ~58-67M
#: elements: at that size the vendor baseline runs 75-90 us, which is where its
#: own median stops flickering.  Smaller lanes were tried and measured 3.1-4.6%
#: paired-ratio CV -- over the gate -- because a ~15-25 us baseline is measured
#: while the GPU is still recovering from the 256 MiB L2 flush that precedes every
#: timed iteration.
_ELEMENTWISE_SHAPES: Mapping[str, Any] = {
    "minimal": {"M": 8192, "N": 8192},
    "primary": {"M": 4096, "N": 14336},        # Llama-class MLP intermediate
    "validation": [
        {"M": 16384, "N": 4096},               # tall/narrow
        {"M": 2048, "N": 32768},               # short/wide
        {"M": 8191, "N": 8193},                # both dims non-power-of-two
    ],
}

#: Row-reduction lanes (softmax / RMSNorm / LayerNorm), sized on the same rule.
_ROW_REDUCE_SHAPES: Mapping[str, Any] = {
    "minimal": {"M": 4096, "N": 16384},
    "primary": {"M": 8192, "N": 8192},
    "validation": [
        {"M": 2048, "N": 32768},               # wide rows, few of them
        {"M": 32768, "N": 2048},               # many rows, narrow
        {"M": 8191, "N": 8193},                # both dims non-power-of-two
    ],
}

#: MXFP4 lanes.  Identical to the row-reduction lanes except that EVERY column
#: count is a multiple of 32, because an MX block spans 32 elements and a row that
#: does not divide evenly has no valid packing.  The first attempt reused
#: ``_ROW_REDUCE_SHAPES``, whose non-power-of-two lane is N=8193; the seed check
#: (which only exercised minimal and primary) passed it and the full end-to-end run
#: caught it -- which is the argument for gating on all declared shapes, not a
#: sample.  8160 = 255 * 32 keeps a non-power-of-two column count that is still a
#: legal MX row.
_MXFP4_SHAPES: Mapping[str, Any] = {
    "minimal": {"M": 4096, "N": 16384},
    "primary": {"M": 8192, "N": 8192},
    "validation": [
        {"M": 2048, "N": 32768},
        {"M": 32768, "N": 2048},
        {"M": 8191, "N": 8160},                # both dims non-power-of-two, N % 32 == 0
    ],
}

#: GEMM lanes.  Every one of these was measured individually inside the
#: environment's own timing pattern (all lanes in one process, 5 paired repeats,
#: interleaved AB/BA, L2 flush per iteration) and clears all three CV gates.
#:
#: Two measured facts shaped this set, and both cost several rejected attempts:
#:
#: * **Square K=M=N lanes are not admissible here.**  At 2048^3 the seed is ~12x
#:   slower than hipBLASLt, so each 28 us baseline median is taken immediately
#:   after ~14 ms of sustained candidate work; the baseline's CV measures 4.5-5.6%
#:   across three separate runs.  Shallower K keeps the same output size while
#:   shrinking that disturbance.
#: * **The FIRST lane benched in a process is the noisiest** (9.8% candidate CV for
#:   a 512^3 lane in position 1, versus 0.2% for a comparable lane in position 5),
#:   and shapes are benched in declared order.  So ``minimal`` here is the cheapest
#:   lane that is still reliably *timeable* -- not a smoke-test lane.  Every
#:   declared shape is benched and gated, so a tiny lane is not free.
_GEMM_SHAPES: Mapping[str, Any] = {
    "minimal": {"M": 2048, "N": 8192, "K": 512},
    "primary": {"M": 4096, "N": 4096, "K": 1024},
    "validation": [
        {"M": 4096, "N": 4096, "K": 512},      # shallow K, square output
        {"M": 1024, "N": 5120, "K": 1792},     # Qwen3-ish non-square
        {"M": 1023, "N": 4097, "K": 511},      # all three dims non-power-of-two
    ],
}

_PREAMBLE = """// KORE HIP seed. Correct, deliberately unoptimised: see kore/tasks/hip_ops.py.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

#define KORE_CHECK_INPUT(t)                                                   \\
  TORCH_CHECK((t).is_cuda(), #t " must be a device tensor");                   \\
  TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")

namespace {

inline int64_t kore_ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline hipStream_t kore_stream() {
  return c10::hip::getCurrentHIPStream().stream();
}

}  // namespace
"""

# --------------------------------------------------------------------------- #
# Elementwise / fused seeds
# --------------------------------------------------------------------------- #
_ELEMENTWISE_SEED = _PREAMBLE + """
namespace {

template <typename scalar_t>
__global__ void kore_unary_kernel(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ y, int64_t n) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    float v = static_cast<float>(x[i]);
    y[i] = static_cast<scalar_t>(%(EXPR)s);
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
"""

_GATED_SEED = _PREAMBLE + """
namespace {

template <typename scalar_t>
__global__ void kore_gated_kernel(const scalar_t* __restrict__ a,
                                  const scalar_t* __restrict__ b,
                                  scalar_t* __restrict__ y, int64_t n) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    float g = static_cast<float>(a[i]);
    float u = static_cast<float>(b[i]);
    y[i] = static_cast<scalar_t>((%(EXPR)s) * u);
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
  KORE_CHECK_INPUT(a);
  KORE_CHECK_INPUT(b);
  TORCH_CHECK(a.sizes() == b.sizes(), "gate and value must have equal shape");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "gate/value dtype mismatch");
  torch::Tensor y = torch::empty_like(a);
  int64_t n = a.numel();
  if (n == 0) return y;
  const int threads = 256;
  int blocks = static_cast<int>(kore_ceil_div(n, threads));
  if (blocks > 8192) blocks = 8192;
  if (blocks < 1) blocks = 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, a.scalar_type(), "kore_gated", ([&] {
        hipLaunchKernelGGL((kore_gated_kernel<scalar_t>), dim3(blocks),
                           dim3(threads), 0, kore_stream(),
                           a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                           y.data_ptr<scalar_t>(), n);
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP gated-activation seed");
}
"""

# --------------------------------------------------------------------------- #
# Row-reduction seeds (softmax / rmsnorm / layernorm)
# --------------------------------------------------------------------------- #
_ROW_REDUCE_PREFIX = _PREAMBLE + """
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
"""

_SOFTMAX_SEED = _ROW_REDUCE_PREFIX + """
namespace {

template <typename scalar_t>
__global__ void kore_softmax_kernel(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ y, int64_t n_cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * n_cols;
  scalar_t* yr = y + row * n_cols;

  float local_max = -INFINITY;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    float v = static_cast<float>(xr[c]);
    local_max = v > local_max ? v : local_max;
  }
  const float row_max = kore_block_reduce(local_max, KoreMax{});

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    local_sum += __expf(static_cast<float>(xr[c]) - row_max);
  }
  const float row_sum = kore_block_reduce(local_sum, KoreAdd{});
  const float inv = 1.0f / row_sum;

  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    yr[c] = static_cast<scalar_t>(
        __expf(static_cast<float>(xr[c]) - row_max) * inv);
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor x) {
  KORE_CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 2, "expected a 2-D [rows, cols] tensor");
  torch::Tensor y = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  if (rows == 0 || cols == 0) return y;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "kore_softmax", ([&] {
        hipLaunchKernelGGL((kore_softmax_kernel<scalar_t>),
                           dim3(static_cast<int>(rows)),
                           dim3(KORE_ROW_THREADS), 0, kore_stream(),
                           x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), cols);
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP row-softmax seed");
}
"""

_RMSNORM_SEED = _ROW_REDUCE_PREFIX + """
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
"""

_LAYERNORM_SEED = _ROW_REDUCE_PREFIX + """
namespace {

template <typename scalar_t>
__global__ void kore_layernorm_kernel(const scalar_t* __restrict__ x,
                                      const scalar_t* __restrict__ w,
                                      const scalar_t* __restrict__ b,
                                      scalar_t* __restrict__ y,
                                      int64_t n_cols, float eps) {
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * n_cols;
  scalar_t* yr = y + row * n_cols;
  const float inv_n = 1.0f / static_cast<float>(n_cols);

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    local_sum += static_cast<float>(xr[c]);
  }
  const float mean = kore_block_reduce(local_sum, KoreAdd{}) * inv_n;

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    float d = static_cast<float>(xr[c]) - mean;
    local_var += d * d;
  }
  const float var = kore_block_reduce(local_var, KoreAdd{}) * inv_n;
  const float inv_std = rsqrtf(var + eps);

  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    float norm = (static_cast<float>(xr[c]) - mean) * inv_std;
    yr[c] = static_cast<scalar_t>(norm * static_cast<float>(w[c]) +
                                  static_cast<float>(b[c]));
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      double eps) {
  KORE_CHECK_INPUT(x);
  KORE_CHECK_INPUT(w);
  KORE_CHECK_INPUT(b);
  TORCH_CHECK(x.dim() == 2, "expected a 2-D [rows, cols] tensor");
  TORCH_CHECK(w.numel() == x.size(1) && b.numel() == x.size(1),
              "weight/bias must be one value per column");
  torch::Tensor y = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  if (rows == 0 || cols == 0) return y;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "kore_layernorm", ([&] {
        hipLaunchKernelGGL((kore_layernorm_kernel<scalar_t>),
                           dim3(static_cast<int>(rows)),
                           dim3(KORE_ROW_THREADS), 0, kore_stream(),
                           x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                           b.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                           cols, static_cast<float>(eps));
      }));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP LayerNorm seed");
}
"""

# --------------------------------------------------------------------------- #
# GEMM seed
# --------------------------------------------------------------------------- #
_GEMM_SEED = _PREAMBLE + """
namespace {

constexpr int KORE_BM = 64, KORE_BN = 64, KORE_BK = 16;
constexpr int KORE_TM = 4, KORE_TN = 4;
constexpr int KORE_GEMM_THREADS = (KORE_BM / KORE_TM) * (KORE_BN / KORE_TN);

// 64x64x16 block tile, 4x4 register micro-tile per thread, fp32 accumulation.
// A reasonable starting point rather than a naive one, deliberately: the HIP bars
// worth beating (AgentKernelArena hip2hip, 6.69x for Opus) start from a working
// HIP kernel and ask for improvement, and a seed 70x off the vendor library
// disturbs its own baseline measurement badly enough to break the timing gate.
// What is still missing is everything CDNA-specific -- MFMA, double buffering,
// wide/async loads, swizzled layouts -- which is what HipKittens measures as
// worth 1.3-3.0x over Triton.
template <typename scalar_t>
__global__ void kore_gemm_kernel(const scalar_t* __restrict__ a,
                                 const scalar_t* __restrict__ b,
                                 scalar_t* __restrict__ c,
                                 int64_t M, int64_t N, int64_t K) {
  // A is staged K-major so the inner product reads it with unit stride.
  __shared__ float as[KORE_BK][KORE_BM];
  __shared__ float bs[KORE_BK][KORE_BN];

  const int tx = threadIdx.x;                 // along N
  const int ty = threadIdx.y;                 // along M
  const int tid = ty * (KORE_BN / KORE_TN) + tx;
  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * KORE_BM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * KORE_BN;

  float acc[KORE_TM][KORE_TN];
#pragma unroll
  for (int i = 0; i < KORE_TM; ++i) {
#pragma unroll
    for (int j = 0; j < KORE_TN; ++j) acc[i][j] = 0.0f;
  }

  constexpr int kTileElems = KORE_BM * KORE_BK;              // == BK * BN
  constexpr int kLoadsPerThread = kTileElems / KORE_GEMM_THREADS;

  for (int64_t k0 = 0; k0 < K; k0 += KORE_BK) {
#pragma unroll
    for (int l = 0; l < kLoadsPerThread; ++l) {
      const int idx = tid + l * KORE_GEMM_THREADS;
      const int am = idx / KORE_BK, ak = idx % KORE_BK;
      const int64_t gm = block_m + am, gk = k0 + ak;
      as[ak][am] = (gm < M && gk < K)
                       ? static_cast<float>(a[gm * K + gk]) : 0.0f;
    }
#pragma unroll
    for (int l = 0; l < kLoadsPerThread; ++l) {
      const int idx = tid + l * KORE_GEMM_THREADS;
      const int bk = idx / KORE_BN, bn = idx % KORE_BN;
      const int64_t gk = k0 + bk, gn = block_n + bn;
      bs[bk][bn] = (gk < K && gn < N)
                       ? static_cast<float>(b[gk * N + gn]) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < KORE_BK; ++kk) {
      float av[KORE_TM], bv[KORE_TN];
#pragma unroll
      for (int i = 0; i < KORE_TM; ++i) av[i] = as[kk][ty * KORE_TM + i];
#pragma unroll
      for (int j = 0; j < KORE_TN; ++j) bv[j] = bs[kk][tx * KORE_TN + j];
#pragma unroll
      for (int i = 0; i < KORE_TM; ++i) {
#pragma unroll
        for (int j = 0; j < KORE_TN; ++j) acc[i][j] += av[i] * bv[j];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < KORE_TM; ++i) {
    const int64_t gm = block_m + ty * KORE_TM + i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < KORE_TN; ++j) {
      const int64_t gn = block_n + tx * KORE_TN + j;
      if (gn < N) c[gm * N + gn] = static_cast<scalar_t>(acc[i][j]);
    }
  }
}

}  // namespace

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
  KORE_CHECK_INPUT(a);
  KORE_CHECK_INPUT(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2-D operands");
  TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must agree");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "operand dtype mismatch");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  torch::Tensor c = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0) return c;
  dim3 threads(KORE_BN / KORE_TN, KORE_BM / KORE_TM);
  dim3 blocks(static_cast<unsigned>(kore_ceil_div(N, KORE_BN)),
              static_cast<unsigned>(kore_ceil_div(M, KORE_BM)));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, a.scalar_type(), "kore_gemm", ([&] {
        hipLaunchKernelGGL((kore_gemm_kernel<scalar_t>), blocks, threads, 0,
                           kore_stream(), a.data_ptr<scalar_t>(),
                           b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
                           M, N, K);
      }));
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP tiled GEMM seed");
}
"""


# --------------------------------------------------------------------------- #
# Low-precision seeds (fp8 quantize, MXFP4 dequantize)
#
# This is where MI355X's spec lead is largest -- 10.1 PFLOPs on MXFP4/MXFP6
# against 2.5 on BF16, versus 9.0 MXFP4 on B200 -- so it is the most defensible
# place for the model to be strong, and the registry had two fp4 tasks.
#
# Both kernels are rounding-critical, and they land in different places, measured
# rather than assumed:
#
# * MXFP4 dequantization is an integer nibble unpack plus a power-of-two scale, so
#   it is exact by construction, and it measures exact: 999 dB (zero error) on
#   every lane.
# * fp8 quantization builds its output through ``c10::Float8_e4m3fn``, the same
#   round-to-nearest-even conversion torch uses, and still measures 56.8-57.2 dB
#   rather than exact.  That corresponds to roughly 1.3e-4 of elements differing
#   by one fp8 step (its own resolution is 12.5%), i.e. tie-breaking at rounding
#   boundaries where torch's bf16 -> fp8 path uses a different intermediate
#   precision than an explicit bf16 -> fp32 -> fp8 chain.  The gate is set at 20 dB
#   with that measurement recorded, not at a value that assumes exactness.
# --------------------------------------------------------------------------- #
_QUANT_FP8_SEED = _ROW_REDUCE_PREFIX + """
#include <c10/util/Float8_e4m3fn.h>

namespace {

constexpr float KORE_FP8_MAX = 448.0f;

// Per-row (per-token) fp8 e4m3 quantization: scale = max(|x|) / 448, then
// y = x / scale. The division (rather than a reciprocal multiply) is deliberate:
// it is what the fp32 oracle does, and a reciprocal would round differently and
// show up as a correctness failure rather than the arithmetic choice it is.
template <typename scalar_t>
__global__ void kore_quant_fp8_kernel(const scalar_t* __restrict__ x,
                                      c10::Float8_e4m3fn* __restrict__ y,
                                      float* __restrict__ scale,
                                      int64_t n_cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * n_cols;
  c10::Float8_e4m3fn* yr = y + row * n_cols;

  float local = 0.0f;
  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    const float v = fabsf(static_cast<float>(xr[c]));
    local = v > local ? v : local;
  }
  const float amax = kore_block_reduce(local, KoreMax{});
  const float s = fmaxf(amax, 1.0e-12f) / KORE_FP8_MAX;
  if (threadIdx.x == 0) scale[row] = s;

  for (int64_t c = threadIdx.x; c < n_cols; c += KORE_ROW_THREADS) {
    yr[c] = c10::Float8_e4m3fn(static_cast<float>(xr[c]) / s);
  }
}

}  // namespace

std::vector<torch::Tensor> forward(torch::Tensor x) {
  KORE_CHECK_INPUT(x);
  TORCH_CHECK(x.dim() == 2, "expected a 2-D [rows, cols] tensor");
  const int64_t rows = x.size(0), cols = x.size(1);
  torch::Tensor y = torch::empty(
      {rows, cols}, x.options().dtype(at::kFloat8_e4m3fn));
  torch::Tensor scale = torch::empty({rows}, x.options().dtype(at::kFloat));
  if (rows == 0 || cols == 0) return {y, scale};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "kore_quant_fp8", ([&] {
        hipLaunchKernelGGL((kore_quant_fp8_kernel<scalar_t>),
                           dim3(static_cast<int>(rows)),
                           dim3(KORE_ROW_THREADS), 0, kore_stream(),
                           x.data_ptr<scalar_t>(),
                           reinterpret_cast<c10::Float8_e4m3fn*>(
                               y.data_ptr<at::Float8_e4m3fn>()),
                           scale.data_ptr<float>(), cols);
      }));
  return {y, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KORE HIP per-token fp8 e4m3 quantize seed");
}
"""

_DEQUANT_MXFP4_SEED = _PREAMBLE + """
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
"""


# --------------------------------------------------------------------------- #
# Operator specifications
# --------------------------------------------------------------------------- #
def _parse_shape_factory(defaults: Mapping[str, int]) -> Callable[[str], dict]:
    def parse_shape(shape_str: str) -> dict:
        if not shape_str or shape_str == "default":
            return dict(defaults)
        out: dict[str, int] = {}
        for kv in shape_str.split(","):
            key, _, value = kv.partition("=")
            out[key.strip()] = int(value)
        return out or dict(defaults)

    return parse_shape


class HipOpSpec:
    """One HIP operator: its oracle, baseline, inputs, and compiling seed."""

    def __init__(
        self,
        op_id: str,
        *,
        product_family: str,
        source_family: str,
        seed: str,
        shape_defaults: Mapping[str, int],
        shapes: Mapping[str, Any],
        snr_db: float,
        baseline_kind: str,
        baseline_note: str,
        make_inputs: Callable[..., tuple],
        oracle: Callable[..., Any],
        baseline: Callable[..., Any],
        dtypes: tuple[str, ...] = ("bf16", "fp16", "fp32"),
        description: str = "",
        timing_admissible: bool = True,
        timing_note: str = "",
        dim_multiples: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.timing_admissible = bool(timing_admissible)
        self.timing_note = timing_note
        # Divisibility the op's own representation requires (MXFP4 needs N % 32 == 0).
        # Enforced by kore.tasks.generate_hip over EVERY declared shape, so an
        # illegal lane cannot reach the registry and fail at datagen time.
        self.dim_multiples = dict(dim_multiples or {})
        self.op_id = op_id
        self.product_family = product_family
        self.source_family = source_family
        self.seed = seed
        self.shape_defaults = dict(shape_defaults)
        self.shapes = shapes
        self.snr_db = float(snr_db)
        self.baseline_kind = baseline_kind
        self.baseline_note = baseline_note
        self.make_inputs = make_inputs
        self.oracle = oracle
        self.baseline = baseline
        self.dtypes = dtypes
        self.description = description


def _randn(shape, dtype: torch.dtype, device, seed: int, scale: float = 1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
    return (t * scale).to(dtype)


# ---- elementwise -------------------------------------------------------- #
def _unary_inputs(shape, dtype, device, seed):
    return (_randn((shape["M"], shape["N"]), dtype, device, seed),)


def _gelu_oracle(x):
    return F.gelu(x.float(), approximate="tanh").to(x.dtype)


def _gelu_baseline(x):
    return F.gelu(x, approximate="tanh")


def _silu_oracle(x):
    return F.silu(x.float()).to(x.dtype)


def _silu_baseline(x):
    return F.silu(x)


# ---- gated fusion ------------------------------------------------------- #
def _gated_inputs(shape, dtype, device, seed):
    return (
        _randn((shape["M"], shape["N"]), dtype, device, seed),
        _randn((shape["M"], shape["N"]), dtype, device, seed + 991),
    )


def _silu_mul_oracle(a, b):
    return (F.silu(a.float()) * b.float()).to(a.dtype)


def _silu_mul_baseline(a, b):
    return F.silu(a) * b


# ---- softmax ------------------------------------------------------------ #
def _softmax_oracle(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _softmax_baseline(x):
    return torch.softmax(x, dim=-1)


# ---- rmsnorm ------------------------------------------------------------ #
_EPS = 1.0e-6


def _rmsnorm_inputs(shape, dtype, device, seed):
    return (
        _randn((shape["M"], shape["N"]), dtype, device, seed),
        _randn((shape["N"],), dtype, device, seed + 7717, scale=0.5),
        _EPS,
    )


def _rmsnorm_oracle(x, w, eps):
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * w.float()).to(x.dtype)


def _rmsnorm_baseline(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)


# ---- layernorm ---------------------------------------------------------- #
def _layernorm_inputs(shape, dtype, device, seed):
    return (
        _randn((shape["M"], shape["N"]), dtype, device, seed),
        _randn((shape["N"],), dtype, device, seed + 3313, scale=0.5),
        _randn((shape["N"],), dtype, device, seed + 5519, scale=0.1),
        _EPS,
    )


def _layernorm_oracle(x, w, b, eps):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (((xf - mean) * torch.rsqrt(var + eps)) * w.float() + b.float()).to(x.dtype)


def _layernorm_baseline(x, w, b, eps):
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


# ---- gemm --------------------------------------------------------------- #
def _gemm_inputs(shape, dtype, device, seed):
    # K**-0.25 per operand puts the output at unit scale (std = sqrt(K)*s**2), so
    # the SNR gate measures accumulation fidelity rather than output over/underflow.
    k = shape["K"]
    scale = float(k) ** -0.25
    return (
        _randn((shape["M"], k), dtype, device, seed, scale=scale),
        _randn((k, shape["N"]), dtype, device, seed + 1279, scale=scale),
    )


def _gemm_oracle(a, b):
    return (a.float() @ b.float()).to(a.dtype)


def _gemm_baseline(a, b):
    return a @ b


# ---- per-token fp8 quantization ----------------------------------------- #
def _quant_fp8_inputs(shape, dtype, device, seed):
    # The activation being quantized is bf16 regardless of the task's dtype ID:
    # ``fp8_e4m3fn`` names the OUTPUT format, which is what the task is about.
    return (_randn((shape["M"], shape["N"]), torch.bfloat16, device, seed),)


def _quant_fp8(x):
    """Reference per-token fp8 e4m3 quantization: ``(y_fp8[M,N], scale[M])``."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
    scale = amax / FP8_E4M3_MAX
    return (xf / scale).to(torch.float8_e4m3fn), scale.squeeze(-1).contiguous()


# ---- MXFP4 dequantization ------------------------------------------------ #
def _mxfp4_pack(w: "torch.Tensor") -> tuple:
    """fp32 ``w[M,N]`` -> ``(packed[M,N//2] uint8, e8m0[M,N//32] uint8)``.

    Same representation as the existing ``gemm_mxfp4`` task, so the two agree on
    what MXFP4 means: per-32-element power-of-two shared exponent, biased by 127,
    with E2M1 codes packed low-nibble-first along the last axis.
    """
    rows, cols = w.shape
    blocks = w.reshape(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2.0,   # E2M1_MAX == 6 ~ 2^2
        torch.zeros_like(amax),
    ).clamp(-127.0, 127.0)
    scale = torch.exp2(exponent)
    normalized = (blocks / scale).clamp(-E2M1_MAX, E2M1_MAX)

    magnitudes = torch.tensor(E2M1_MAGNITUDES, device=w.device, dtype=torch.float32)
    midpoints = (magnitudes[1:] + magnitudes[:-1]) / 2.0
    absolute = normalized.abs()
    codes = torch.bucketize(absolute, midpoints).to(torch.uint8)
    codes = codes | (normalized < 0).to(torch.uint8) * 8
    codes = codes.reshape(rows, cols)

    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    e8m0 = (exponent + 127.0).to(torch.uint8).reshape(rows, cols // MX_BLOCK)
    return packed, e8m0.contiguous()


def _mxfp4_inputs(shape, dtype, device, seed):
    w = _randn((shape["M"], shape["N"]), torch.float32, device, seed)
    packed, e8m0 = _mxfp4_pack(w)
    return (packed, e8m0, "bf16")


def _dequant_mxfp4(packed, e8m0, out_dtype):
    """Reference MXFP4 dequantization; exact integer unpack + power-of-two scale."""
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[out_dtype]
    rows, half = packed.shape
    cols = half * 2
    codes = torch.empty((rows, cols), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    magnitudes = torch.tensor(E2M1_MAGNITUDES, device=packed.device,
                              dtype=torch.float32)
    values = magnitudes[(codes & 0x7).long()]
    values = torch.where(codes & 0x8 > 0, -values, values)
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(MX_BLOCK, dim=1)
    return (values * scale).to(torch_dtype)


HIP_OPS: Mapping[str, HipOpSpec] = {
    spec.op_id: spec
    for spec in (
        HipOpSpec(
            "gelu_tanh",
            product_family="activation",
            source_family="unary",
            seed=_ELEMENTWISE_SEED % {
                "EXPR": "0.5f * v * (1.0f + tanhf(0.7978845608028654f * "
                        "(v + 0.044715f * v * v * v)))",
            },
            shape_defaults={"M": 4096, "N": 14336},
            shapes=_ELEMENTWISE_SHAPES,
            snr_db=35.0,
            baseline_kind="torch_gelu_tanh",
            baseline_note="F.gelu(approximate='tanh'), the fused ROCm elementwise kernel",
            make_inputs=_unary_inputs,
            oracle=_gelu_oracle,
            baseline=_gelu_baseline,
            description="tanh-approximation GELU activation",
        ),
        HipOpSpec(
            "silu",
            product_family="activation",
            source_family="unary",
            seed=_ELEMENTWISE_SEED % {"EXPR": "v / (1.0f + __expf(-v))"},
            shape_defaults={"M": 4096, "N": 14336},
            shapes=_ELEMENTWISE_SHAPES,
            snr_db=35.0,
            baseline_kind="torch_silu",
            baseline_note="F.silu, the fused ROCm elementwise kernel",
            make_inputs=_unary_inputs,
            oracle=_silu_oracle,
            baseline=_silu_baseline,
            description="SiLU / swish activation",
        ),
        HipOpSpec(
            "silu_mul",
            product_family="activation",
            source_family="binary",
            seed=_GATED_SEED % {"EXPR": "g / (1.0f + __expf(-g))"},
            shape_defaults={"M": 4096, "N": 14336},
            shapes=_ELEMENTWISE_SHAPES,
            snr_db=35.0,
            baseline_kind="torch_silu_mul",
            baseline_note="F.silu(gate) * value, two fused ROCm elementwise kernels",
            make_inputs=_gated_inputs,
            oracle=_silu_mul_oracle,
            baseline=_silu_mul_baseline,
            description="SwiGLU gated activation (silu(gate) * value)",
        ),
        HipOpSpec(
            "softmax_rows",
            product_family="reduction",
            source_family="reduce",
            seed=_SOFTMAX_SEED,
            shape_defaults={"M": 8192, "N": 8192},
            shapes=_ROW_REDUCE_SHAPES,
            snr_db=30.0,
            baseline_kind="torch_softmax",
            baseline_note="torch.softmax(dim=-1), the vendor fused softmax",
            make_inputs=_unary_inputs,
            oracle=_softmax_oracle,
            baseline=_softmax_baseline,
            description="numerically-stable row softmax",
        ),
        HipOpSpec(
            "rmsnorm",
            product_family="normalization",
            source_family="reduce",
            seed=_RMSNORM_SEED,
            shape_defaults={"M": 8192, "N": 8192},
            shapes=_ROW_REDUCE_SHAPES,
            snr_db=30.0,
            baseline_kind="torch_rms_norm",
            baseline_note="F.rms_norm, the vendor fused RMSNorm",
            make_inputs=_rmsnorm_inputs,
            oracle=_rmsnorm_oracle,
            baseline=_rmsnorm_baseline,
            description="RMSNorm with a per-column weight",
        ),
        HipOpSpec(
            "layernorm",
            product_family="normalization",
            source_family="reduce",
            seed=_LAYERNORM_SEED,
            shape_defaults={"M": 8192, "N": 8192},
            shapes=_ROW_REDUCE_SHAPES,
            snr_db=30.0,
            baseline_kind="torch_layer_norm",
            baseline_note="F.layer_norm, the vendor fused LayerNorm",
            make_inputs=_layernorm_inputs,
            oracle=_layernorm_oracle,
            baseline=_layernorm_baseline,
            description="LayerNorm with per-column weight and bias",
        ),
        HipOpSpec(
            "gemm",
            product_family="gemm",
            source_family="gemm",
            seed=_GEMM_SEED,
            shape_defaults={"M": 4096, "N": 4096, "K": 1024},
            shapes=_GEMM_SHAPES,
            snr_db=25.0,
            baseline_kind="hipblaslt_matmul",
            baseline_note="a @ b, which lowers to the hipBLASLt vendor GEMM",
            make_inputs=_gemm_inputs,
            oracle=_gemm_oracle,
            baseline=_gemm_baseline,
            description="dense matrix multiply",
            timing_admissible=False,
            timing_note=(
                "Compiles and verifies (85-130 dB against a 25 dB gate), but is NOT "
                "timing-admissible on this host and so is not generated by default. "
                "Measured across four end-to-end runs and three different shape "
                "lane sets: the hipBLASLt baseline's own coefficient of variation "
                "lands at 3.4-5.6% against the 3% publication gate. The mechanism "
                "is the candidate/baseline asymmetry -- this seed is 10-16x slower "
                "than hipBLASLt, so every ~30 us baseline median is measured "
                "immediately after milliseconds of sustained candidate work, and "
                "with only 5 paired repeats the CV estimate crosses the gate more "
                "often than not. Raising the gate is not an option: it would "
                "corrupt every speedup this project reports. What would fix it is "
                "either more paired repeats for high-asymmetry tasks, or an "
                "MFMA-class seed that narrows the 10x gap."
            ),
        ),
        HipOpSpec(
            "quant_fp8_pertoken",
            product_family="quantization",
            source_family="unary",
            seed=_QUANT_FP8_SEED,
            shape_defaults={"M": 8192, "N": 8192},
            shapes=_ROW_REDUCE_SHAPES,
            snr_db=20.0,
            baseline_kind="torch_quant_fp8_pertoken",
            baseline_note=(
                "the eager torch path (absmax reduction, divide, cast), which is "
                "three separate kernel launches against the seed's one"),
            make_inputs=_quant_fp8_inputs,
            oracle=_quant_fp8,
            baseline=_quant_fp8,
            dtypes=("fp8_e4m3fn",),
            description=(
                "per-token fp8 e4m3 quantization, emitting codes and per-row scale"),
        ),
        HipOpSpec(
            "dequant_mxfp4",
            product_family="quantization",
            source_family="quant",
            seed=_DEQUANT_MXFP4_SEED,
            shape_defaults={"M": 8192, "N": 8192},
            shapes=_MXFP4_SHAPES,
            dim_multiples={"N": MX_BLOCK},
            snr_db=40.0,
            baseline_kind="torch_dequant_mxfp4",
            baseline_note=(
                "the eager torch path (nibble unpack, gather, exp2 scale, "
                "repeat_interleave), which materializes several full-size "
                "intermediates the seed never writes"),
            make_inputs=_mxfp4_inputs,
            oracle=_dequant_mxfp4,
            baseline=_dequant_mxfp4,
            dtypes=("mxfp4",),
            description=f"MXFP4 dequantization ({MXFP4_NOTE})",
        ),
    )
}


def seed_source(op_id: str) -> str:
    """The compiling HIP C++ seed for ``op_id``."""
    return HIP_OPS[op_id].seed


def make_reference(op_id: str, dtype_id: str) -> dict:
    """Build the ``reference.py`` namespace for one (op, dtype) HIP task.

    Mirrors the generated-Triton reference contract exactly -- ``entry_name``,
    ``mutates_input``, ``family``, ``baseline_kind``, ``parse_shape``,
    ``get_inputs``, ``ref_fn``, ``baseline_fn`` -- so
    :func:`kore.tasks._genops.driver_main` drives a HIP task with no special
    casing and the task is publication-eligible for timing.
    """
    spec = HIP_OPS[op_id]
    if dtype_id not in TORCH_DTYPES:
        raise KeyError(f"unknown HIP task dtype {dtype_id!r}")
    torch_dtype = TORCH_DTYPES[dtype_id]

    def get_inputs(shape: dict, device="cuda", seed: int = 0, dtype=None):
        return spec.make_inputs(shape, dtype or torch_dtype, device, seed)

    return {
        # The candidate ABI: a .hip file binding `forward` through
        # PYBIND11_MODULE(TORCH_EXTENSION_NAME, m).
        "entry_name": "forward",
        "mutates_input": False,
        "family": spec.source_family,
        "baseline_kind": spec.baseline_kind,
        "parse_shape": _parse_shape_factory(spec.shape_defaults),
        "get_inputs": get_inputs,
        "ref_fn": spec.oracle,
        "baseline_fn": spec.baseline,
        "dtype_id": dtype_id,
        "op_id": op_id,
    }


__all__ = [
    "GPU_TARGET",
    "HIP_OPS",
    "HipOpSpec",
    "TORCH_DTYPES",
    "make_reference",
    "seed_source",
]
