"""Tests for the mid-train corpus quality filter (:mod:`kore.data.corpus_quality`).

Pure-CPU, no heavy deps. We check that:

  * low-value files are DROPPED with the expected, actionable reason — minified
    one-liners, ``@generated`` / protobuf stubs, vendored/third-party paths,
    lockfiles, whole-file ``clang-format off``, punctuation/binary blobs,
    trivial stubs, and self-repetitive dumps;
  * license-only / badge-only / too-short markdown docs are DROPPED;
  * and — the property that matters most for a kernel corpus — REAL dense
    domain code is KEPT: a flash-attention Triton kernel, a HIP kernel, a
    Composable-Kernel fp8 GEMM ``.cu`` (SPDX header + a *local*
    ``clang-format off``/``on`` block), and a substantive ROCm perf doc.

The kept-fixtures below are faithful, self-contained kernels; an additional
(guarded) test also runs the gate over the real local repo files when present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kore.data import corpus_quality as cq
from kore.data.corpus_quality import (
    CODE_DROP_REASONS,
    DOC_DROP_REASONS,
    code_quality_ok,
    code_quality_reason,
    doc_quality_ok,
    doc_quality_reason,
    quality_filter,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Real-code fixtures (must be KEPT)
# --------------------------------------------------------------------------- #
# A real fused (flash) attention forward Triton kernel: long-ish lines, low
# comment ratio, online-softmax rescaling — exactly the dense code a naive
# comment-ratio or duplicate-line filter would wrongly discard.
FLASH_ATTN_TRITON = '''\
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out, L, M,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + start_n * stride_vk)
        acc += tl.dot(p.to(v.dtype), v)
        l_i = l_i_new
        m_i = m_i_new
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))
'''

# A second, DIFFERENT Triton kernel (rmsnorm). It shares many short lines with
# the attention kernel above (``import``/``@triton.jit``/``tl.*`` idioms/arg
# fragments) yet has a distinct body — the exact case the char-weighted
# repetition guard must keep.
RMSNORM_TRITON = '''\
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(X, W, Y, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = x * rstd * w
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)
'''

# A real HIP kernel: block-tiled reduction using shared (LDS) memory. Dense C++,
# comment-sparse, long-ish lines — must be KEPT.
HIP_KERNEL = '''\
#include <hip/hip_runtime.h>

#define BLOCK 256

__global__ void softmax_rows(const float* __restrict__ x,
                             float* __restrict__ y, int n) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * n;
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += BLOCK) {
        local_max = fmaxf(local_max, xr[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += BLOCK) {
        local_sum += expf(xr[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int i = tid; i < n; i += BLOCK) {
        y[row * n + i] = expf(xr[i] - row_max) * inv;
    }
}
'''

# A real Composable-Kernel fp8 GEMM .cu excerpt. Note two robustness traps that
# must NOT cause a drop: (1) an SPDX/copyright header at the very top, and
# (2) a *local* ``clang-format off``/``on`` block (only whole-file off is a
# generated-table signal).
CK_FP8_GEMM_CU = '''\
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType = FP8;
using B0DataType = FP8;
using AccDataType = F32;
using CShuffleDataType = F32;
using EDataType = F16;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
    // clang-format off
        <Row, Col, ck::Tuple<>, Row,
         A0DataType, B0DataType, ck::Tuple<>, EDataType, AccDataType, CShuffleDataType,
         PassThrough, PassThrough, CDEElementOp, GemmSpec, 256,
         128, 128, 64, 16, 16, 16, 16, 4, 4>;
// clang-format on

int main(int argc, char* argv[]) {
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;
    auto device_op = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();
    auto argument = device_op.MakeArgument(nullptr, nullptr, {}, nullptr,
                                           M, N, K, K, K, {}, N,
                                           PassThrough{}, PassThrough{}, CDEElementOp{});
    if (!device_op.IsSupportedArgument(argument)) {
        std::cerr << "unsupported argument\\n";
        return 1;
    }
    float ms = invoker.Run(argument, StreamConfig{nullptr, true});
    std::cout << "fp8 gemm: " << ms << " ms" << std::endl;
    return 0;
}
'''


# --------------------------------------------------------------------------- #
# Real code is KEPT (the core requirement)
# --------------------------------------------------------------------------- #
def test_real_flash_attn_triton_kernel_is_kept():
    assert code_quality_reason(FLASH_ATTN_TRITON, Path("flash_attn.py")) is None
    assert code_quality_ok(FLASH_ATTN_TRITON, Path("flash_attn.py")) is True


def test_real_hip_kernel_is_kept():
    assert code_quality_reason(HIP_KERNEL, Path("softmax.cu")) is None
    assert code_quality_ok(HIP_KERNEL, Path("softmax.hip")) is True


def test_real_ck_fp8_gemm_cu_is_kept():
    # SPDX header + local clang-format off/on block must not trigger a drop.
    assert code_quality_reason(CK_FP8_GEMM_CU, Path("gemm_fp8.cu")) is None
    assert code_quality_ok(CK_FP8_GEMM_CU, Path("gemm_fp8.cpp")) is True


def test_spdx_license_header_in_code_is_kept():
    # A license/SPDX header is NORMAL at the top of source files (unlike docs).
    src = ("// SPDX-License-Identifier: MIT\n"
           "// Copyright (c) 2024 AMD\n\n" + HIP_KERNEL)
    assert code_quality_ok(src, Path("k.cu")) is True


def test_local_clang_format_block_kept_but_whole_file_off_dropped():
    body = ("int compute(int a, int b) {\n"
            "    int c = a * b + a - b;\n"
            "    return c * 2 + 1;\n"
            "}\n")
    local = "// clang-format off\n" + body + "// clang-format on\n"
    whole = "// clang-format off\n" + body  # no matching 'on' -> whole file
    assert code_quality_ok(local, Path("a.cpp")) is True
    assert code_quality_reason(whole, Path("b.cpp")) == "generated_marker"


# --------------------------------------------------------------------------- #
# Low-value code is DROPPED (with the right reason)
# --------------------------------------------------------------------------- #
def test_minified_one_liner_dropped():
    minified = "var f=function(a,b){return a+b;};" + "x=x+1;" * 400
    assert code_quality_reason(minified, Path("bundle.min.js")) == "long_lines"
    assert code_quality_ok(minified, Path("bundle.min.js")) is False


def test_mean_line_length_minified_dropped():
    # Every line < 1000 chars, but the mean exceeds MAX_MEAN_LINE_LENGTH.
    text = "\n".join("a = " + "z" * 200 for _ in range(10))
    assert code_quality_reason(text, Path("gen.py")) == "long_lines"


def test_generated_marker_at_top_dropped():
    for marker in ("// @generated by protoc\n", "# DO NOT EDIT\n",
                   "/* This file was automatically generated. */\n",
                   "// Code generated by mockgen. DO NOT EDIT.\n"):
        src = marker + "\nint value = 1;\nint other = 2;\nint third = 3;\n"
        assert code_quality_reason(src, Path("x.c")) == "generated_marker", marker


def test_protobuf_pb2_path_dropped():
    src = "class Foo(object):\n    pass\n\n\ndef bar():\n    return 123\n\n\nX = 7\n"
    assert code_quality_reason(src, Path("proto/service_pb2.py")) == "generated_path"
    assert code_quality_reason(src, Path("gen/msg.pb.h")) == "generated_path"
    assert code_quality_reason(src, Path("a/b/thing.generated.cpp")) == "generated_path"


def test_vendored_third_party_path_dropped():
    src = "int main() {\n    int a = 1;\n    int b = 2;\n    return a + b;\n}\n"
    assert code_quality_reason(src, Path("repo/third_party/lib/x.cpp")) == "vendored_path"
    for part in ("node_modules", "external", "vendor", "site-packages",
                 "googletest", "pybind11", "__pycache__"):
        p = Path("root") / part / "nested" / "f.cpp"
        assert code_quality_reason(src, p) == "vendored_path", part


def test_lockfile_dropped():
    text = "\n".join(f'package-{i} = "1.2.{i}"' for i in range(60))
    assert code_quality_reason(text, Path("poetry.lock")) == "lockfile"
    assert code_quality_reason(text, Path("dir/Cargo.lock")) == "lockfile"


def test_trivial_stub_dropped():
    assert code_quality_reason("def todo():\n    pass\n", Path("s.py")) == "too_trivial"
    assert code_quality_reason("pass\n", Path("s.py")) == "too_trivial"
    # A file whose only content is comments has no real code -> trivial.
    only_comments = "# a comment\n# another comment\n# yet another one here\n"
    assert code_quality_reason(only_comments, Path("c.py")) == "too_trivial"
    assert code_quality_reason("", Path("empty.py")) == "too_trivial"


def test_low_alnum_punctuation_blob_dropped():
    blob = "\n".join("{}[]() ;:,.<>|&" for _ in range(20))
    assert code_quality_reason(blob, Path("weird.txt")) == "low_alnum"


def test_self_repetitive_dump_dropped():
    dump = "const int T[] = {\n" + "    0x00, 0x00, 0x00, 0x00, 0x00,\n" * 50 + "};\n"
    assert code_quality_reason(dump, Path("table.h")) == "repetitive"
    boiler = "\n".join(["self.register_buffer('x', torch.zeros(1))"] * 30)
    assert code_quality_reason(boiler, Path("m.py")) == "repetitive"


def test_multi_kernel_file_not_flagged_repetitive():
    # Two DIFFERENT kernels in one file repeat many SHORT lines (high
    # duplicate-LINE fraction) but their character mass is unique -> KEEP.
    two_kernels = FLASH_ATTN_TRITON + "\n\n" + RMSNORM_TRITON
    assert code_quality_reason(two_kernels, Path("attn.py")) is None
    # Boundary: duplicating the SAME kernel body verbatim IS repetitive (its
    # bulk of characters is an exact duplicate) and is correctly dropped.
    duplicated = FLASH_ATTN_TRITON + "\n\n" + FLASH_ATTN_TRITON
    assert code_quality_reason(duplicated, Path("dup.py")) == "repetitive"


# --------------------------------------------------------------------------- #
# Docs
# --------------------------------------------------------------------------- #
LICENSE_ONLY_MD = '''\
# License

SPDX-License-Identifier: MIT

Copyright (c) 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

BADGE_ONLY_MD = '''\
# my-project

[![Build](https://img.shields.io/badge/build-passing.svg)](https://ci.example.com)
[![License](https://img.shields.io/badge/license-MIT.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/x.svg)](https://pypi.org/project/x)

[Home](/) | [Docs](/docs) | [API](/api) | [Changelog](/changelog)
'''

PERF_DOC_MD = '''\
<!-- SPDX-License-Identifier: MIT -->
# MI300X GEMM Tuning Guide

This guide explains how to tune fp8 and bf16 GEMM kernels on the AMD MI300X
(gfx942) architecture. Achieving peak throughput requires balancing occupancy,
LDS usage, and the matrix-core (MFMA) instruction mix.

## Occupancy and waves

Each compute unit on gfx942 has four SIMDs. To hide global-memory latency you
should aim for at least four waves per SIMD. Increasing the block tile raises
register pressure, which lowers occupancy, so measure with rocprof and iterate.

## Recommended tile sizes

For a 4096x4096x4096 problem a 256x256 macro tile with a 64-deep K loop
delivered the best measured throughput in our experiments on the accelerator.
'''


def test_license_only_doc_dropped():
    assert doc_quality_reason(LICENSE_ONLY_MD, Path("LICENSE.md")) == "license_only"
    assert doc_quality_ok(LICENSE_ONLY_MD, Path("LICENSE.md")) is False


def test_badge_nav_only_doc_dropped():
    assert doc_quality_reason(BADGE_ONLY_MD, Path("README.md")) == "low_prose"


def test_too_short_doc_dropped():
    assert doc_quality_reason("# Title\n\nHi.\n", Path("x.md")) == "doc_too_short"
    assert doc_quality_reason("", Path("x.md")) == "doc_too_short"


def test_substantive_perf_doc_kept():
    assert doc_quality_reason(PERF_DOC_MD, Path("tuning.md")) is None
    assert doc_quality_ok(PERF_DOC_MD, Path("tuning.md")) is True


def test_doc_with_spdx_header_but_real_prose_is_kept():
    # SPDX marker present, but there is plenty of non-license prose -> KEEP.
    assert doc_quality_ok(PERF_DOC_MD, Path("tuning.md")) is True


# --------------------------------------------------------------------------- #
# quality_filter batch API + stats
# --------------------------------------------------------------------------- #
def test_quality_filter_code_stats_and_reasons():
    files = [
        (Path("good_attn.py"), FLASH_ATTN_TRITON),   # keep
        (Path("good_gemm.cu"), CK_FP8_GEMM_CU),       # keep
        (Path("bundle.min.js"), "x=x+1;" * 500),      # long_lines
        (Path("stub.py"), "pass\n"),                   # too_trivial
        (Path("svc_pb2.py"), "X = 1\nY = 2\nZ = 3\n"), # generated_path
        (Path("vendor/dep.cpp"), "int a=1;\nint b=2;\nint c=3;\n"),  # vendored_path
    ]
    kept, stats = quality_filter(files, is_doc=False)
    assert stats["in"] == 6
    assert stats["kept"] == 2
    assert stats["dropped"] == 4
    assert stats["drop_reasons"] == {
        "long_lines": 1,
        "too_trivial": 1,
        "generated_path": 1,
        "vendored_path": 1,
    }
    # kept preserves the original items (order + identity of tuples).
    assert kept == [files[0], files[1]]


def test_quality_filter_doc_mode():
    files = [
        (Path("tuning.md"), PERF_DOC_MD),        # keep
        (Path("LICENSE.md"), LICENSE_ONLY_MD),   # license_only
        (Path("nav.md"), BADGE_ONLY_MD),          # low_prose
        (Path("tiny.md"), "# Hi\n"),              # doc_too_short
    ]
    kept, stats = quality_filter(files, is_doc=True)
    assert stats["in"] == 4 and stats["kept"] == 1 and stats["dropped"] == 3
    assert stats["drop_reasons"] == {
        "license_only": 1, "low_prose": 1, "doc_too_short": 1,
    }
    assert kept == [files[0]]


def test_quality_filter_empty_input():
    kept, stats = quality_filter([])
    assert kept == []
    assert stats == {"in": 0, "kept": 0, "dropped": 0, "drop_reasons": {}}


def test_quality_filter_accepts_none_path():
    # A path is optional; content-only items still work.
    kept, stats = quality_filter([(None, FLASH_ATTN_TRITON), (None, "pass\n")])
    assert stats["kept"] == 1 and stats["drop_reasons"] == {"too_trivial": 1}


# --------------------------------------------------------------------------- #
# Reason vocabulary + tunability
# --------------------------------------------------------------------------- #
def test_reasons_are_in_published_vocabulary():
    assert code_quality_reason("x=x+1;" * 500, Path("a.min.js")) in CODE_DROP_REASONS
    assert doc_quality_reason("# Hi\n", Path("a.md")) in DOC_DROP_REASONS
    # every reason the code gate can emit is documented
    for r in ("lockfile", "vendored_path", "generated_path", "generated_marker",
              "long_lines", "low_alnum", "too_trivial", "repetitive"):
        assert r in CODE_DROP_REASONS


def test_thresholds_are_tunable_constants(monkeypatch):
    # Predicates read the module-level constants at call time, so a corpus
    # builder can retune them without touching the logic.
    dense = "\n".join("a = " + "z" * 80 for _ in range(6))  # mean line ~84
    assert code_quality_reason(dense, Path("k.py")) is None
    monkeypatch.setattr(cq, "MAX_MEAN_LINE_LENGTH", 50.0)
    assert code_quality_reason(dense, Path("k.py")) == "long_lines"


def test_module_import_is_side_effect_free():
    # Execute the module file in an ISOLATED namespace (no sys.modules mutation).
    # It must load standalone with only stdlib deps and define no logger / heavy
    # state, so ``from kore.data.corpus_quality import quality_filter`` is cheap.
    import importlib.util

    path = _REPO_ROOT / "kore" / "data" / "corpus_quality.py"
    spec = importlib.util.spec_from_file_location("_cq_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise / emit / touch the fs
    assert callable(mod.quality_filter)
    assert not hasattr(mod, "log") and not hasattr(mod, "get_logger")


# --------------------------------------------------------------------------- #
# Guarded: run the gate over the REAL local repo files if present
# --------------------------------------------------------------------------- #
def test_real_repo_kernels_are_kept_if_present():
    candidates = [
        "repos/flash-attention/flash_attn/flash_attn_triton_og.py",
        "repos/flash-attention/flash_attn/flash_attn_triton.py",
        "repos/composable_kernel/example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp",
        "repos/composable_kernel/example/67_gemm_microscaling/gemm_mx_fp8.cpp",
        "repos/aiter/csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu",
    ]
    present = [_REPO_ROOT / c for c in candidates if (_REPO_ROOT / c).is_file()]
    if not present:
        pytest.skip("local source repos not present on this box")
    for p in present:
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert code_quality_reason(text, p) is None, f"real kernel dropped: {p}"
