"""CPU sanity check for the DRAFT quantized-GEMM oracles (reference-correctness only).

Runs each task's ``reference.reference_output`` on CPU against a SMALL random input and
compares it to a fully INDEPENDENT fp32 dequant-matmul computed here with a DIFFERENT
code path than ``_quant_common`` (``torch.einsum`` with the scales applied on the
accumulator, ``repeat_interleave`` scale expansion, an ARITHMETIC e2m1 decode instead of
the reference LUT, and an explicit zero-point). If the two agree, the oracle math (which
scale on which axis, block/group indexing, zero-point order, mxfp4 e8m0 exponent, fp8
requant) is corroborated -- quant scale application is the classic bug, so per-token vs
per-tensor vs block vs group scaling is each checked explicitly.

It also verifies the K-alignment GUARDS (K=4095 fp8/MX/int4-illegal) raise, and that the
oracle is layout-agnostic for a transposed (non-contiguous) activation (edge L1).

This proves REFERENCE CORRECTNESS ON CPU ONLY. It does NOT compile the Triton seeds, run
the vendor baselines, or measure anything on gfx950 -- see VERIFICATION_CHECKLIST.md. Run:

    ~/kore-venv/bin/python kore/tasks/_drafts/quant/_cpu_sanity_check.py
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BLK = 128
MX = 32


def _load_ref(task_id):
    path = os.path.join(HERE, task_id, "reference.py")
    spec = importlib.util.spec_from_file_location(f"draft_qref_{task_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snr_db(o, r):
    o, r = o.float(), r.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _fp8():
    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    return FP8_DTYPE, FP8_MAX


# --- independent nibble unpack + arithmetic e2m1 decode (NOT the reference path) ----- #
def _unpack_nibbles(packed, K):
    R = packed.shape[0]
    codes = torch.empty((R, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = (packed & 0xF).to(torch.int32)
    codes[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32)
    return codes


def _e2m1_decode_arith(code):
    """Arithmetic e2m1 decode (independent of the reference LUT). code int -> fp32."""
    idx = code & 0x7
    sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
    exp = torch.div(idx, 2, rounding_mode="floor")     # 0,0,1,1,2,2,3,3
    mant = (idx & 1).float()
    mag_sub = mant * 0.5                                # exp==0 subnormal: 0 or 0.5
    mag_norm = (1.0 + 0.5 * mant) * torch.exp2((exp - 1).float())
    mag = torch.where(exp == 0, mag_sub, mag_norm)
    return sign * mag


# --- independent oracle per task (einsum + scales on the accumulator) --------------- #
def _independent(task_id, shape, inputs):
    if task_id in ("gemm_fp8_a8w8_pertoken", "gemm_fp8_a8w8_pertensor", "gemm_int8_a8w8"):
        xq, wq, x_scale, w_scale = inputs
        raw = torch.einsum("mk,nk->mn", xq.float(), wq.float())     # [M,N]
        y = raw * x_scale.float() * w_scale.float()                 # [M,1]*[1,N] on accum
        return y.to(torch.bfloat16)

    if task_id == "gemm_fp8_a8w8_blockscale":
        xq, wq, xs, ws = inputs
        M, K = xq.shape
        N = wq.shape[0]
        xd = xq.float() * xs.float().repeat_interleave(BLK, dim=1)                 # [M,K]
        wd = wq.float() * ws.float().repeat_interleave(BLK, dim=0).repeat_interleave(BLK, dim=1)
        return torch.einsum("mk,nk->mn", xd, wd).to(torch.bfloat16)

    if task_id == "gemm_mxfp4_a4w4":
        a_packed, a_e8m0, w_packed, w_e8m0 = inputs
        K = shape["K"]
        a_vals = _e2m1_decode_arith(_unpack_nibbles(a_packed, K))
        w_vals = _e2m1_decode_arith(_unpack_nibbles(w_packed, K))
        a_sc = torch.exp2(a_e8m0.float() - 127.0).repeat_interleave(MX, dim=1)     # [M,K]
        w_sc = torch.exp2(w_e8m0.float() - 127.0).repeat_interleave(MX, dim=1)     # [N,K]
        a_deq = a_vals * a_sc
        w_deq = w_vals * w_sc
        return torch.einsum("mk,nk->mn", a_deq, w_deq).to(torch.bfloat16)

    if task_id == "gemm_w4a16_g128":
        a, w_packed, scale, zero = inputs
        K = shape["K"]
        G = K // scale.shape[1]
        codes = _unpack_nibbles(w_packed, K).float()
        z = zero.to(torch.int32).repeat_interleave(G, dim=1).float()               # [N,K]
        s = scale.float().repeat_interleave(G, dim=1)                              # [N,K]
        w_deq = (codes - z) * s
        return torch.einsum("mk,nk->mn", a.float(), w_deq).to(torch.bfloat16)

    if task_id == "gemm_w4a8_fp8":
        xq, x_scale, w_packed, w_scale = inputs
        K = shape["K"]
        codes = _unpack_nibbles(w_packed, K).float() - 8.0
        w_deq = codes * w_scale.float()                                            # [N,K]
        raw = torch.einsum("mk,nk->mn", xq.float(), w_deq)                         # [M,N]
        return (raw * x_scale.float()).to(torch.bfloat16)

    if task_id == "gemm_fp8_requant_epilogue":
        xq, wq, x_scale, w_scale, bias, out_scale = inputs
        fp8, fmax = _fp8()
        raw = torch.einsum("mk,nk->mn", xq.float(), wq.float())
        acc = raw * x_scale.float() * w_scale.float() + bias.float().reshape(1, -1)
        yq = (acc / out_scale.float()).clamp(-fmax, fmax).to(fp8)                  # shared fp8 requant
        return (yq.float() * out_scale.float()).to(torch.bfloat16)

    raise KeyError(task_id)


# tiny primary shapes (each satisfies the task's alignment guard).
TINY = {
    "gemm_fp8_a8w8_pertoken": {"M": 4, "N": 6, "K": 8},
    "gemm_fp8_a8w8_pertensor": {"M": 4, "N": 6, "K": 8},
    "gemm_fp8_a8w8_blockscale": {"M": 3, "N": 128, "K": 128},
    "gemm_mxfp4_a4w4": {"M": 4, "N": 6, "K": 64},
    "gemm_int8_a8w8": {"M": 4, "N": 6, "K": 8},
    "gemm_w4a16_g128": {"M": 4, "N": 6, "K": 128},
    "gemm_w4a8_fp8": {"M": 4, "N": 6, "K": 8},
    "gemm_fp8_requant_epilogue": {"M": 4, "N": 6, "K": 8},
}

# extra edges: multi-block/group indexing, non-pow2 tails, transposed (non-contiguous) A.
EXTRA = {
    "gemm_fp8_a8w8_pertoken": {"M": 5, "N": 7, "K": 10, "TA": 1},   # transposed A (L1)
    "gemm_fp8_a8w8_blockscale": {"M": 5, "N": 256, "K": 256},       # 2x2 block indexing
    "gemm_mxfp4_a4w4": {"M": 3, "N": 5, "K": 96},                   # 3 MX groups
    "gemm_w4a16_g128": {"M": 4, "N": 6, "K": 256, "TA": 1},         # 2 groups + transposed A
    "gemm_int8_a8w8": {"M": 7, "N": 5, "K": 9},                     # non-pow2 tail
}

# get_inputs must GUARD these illegal K (raise), per the fp8/MX/int4 alignment rules.
GUARD = {
    "gemm_fp8_a8w8_blockscale": {"M": 8, "N": 512, "K": 4095},     # K%128 != 0
    "gemm_mxfp4_a4w4": {"M": 8, "N": 512, "K": 4095},              # K%32  != 0
    "gemm_w4a16_g128": {"M": 8, "N": 512, "K": 4095},              # K%128 != 0
    "gemm_w4a8_fp8": {"M": 8, "N": 512, "K": 4095},                # K odd (nibble packing)
}


def _run_one(task_id, shape):
    ref = _load_ref(task_id)
    inputs = ref.get_inputs(dict(shape), device="cpu", seed=0)
    got = ref.reference_output(dict(shape), inputs)
    exp = _independent(task_id, dict(shape), inputs)
    assert got.shape == exp.shape, f"{task_id}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    assert got.dtype == exp.dtype, f"{task_id}: dtype {got.dtype} != {exp.dtype}"
    snr = _snr_db(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    ac = torch.allclose(got.float(), exp.float(), atol=2e-2, rtol=2e-2)
    return snr, md, ac


def _run_guard(task_id, shape):
    ref = _load_ref(task_id)
    try:
        ref.get_inputs(dict(shape), device="cpu", seed=0)
        return False   # no raise -> guard MISSING
    except AssertionError:
        return True    # correctly rejected the illegal K


def main():
    all_ok = True
    print(f"{'task_id':<28} {'shape':<7} {'SNR(dB)':>9} {'max_diff':>10} {'allclose':>9}")
    print("-" * 74)
    for tid in sorted(TINY):
        for label, table in (("tiny", TINY), ("extra", EXTRA)):
            if tid not in table:
                continue
            snr, md, ac = _run_one(tid, table[tid])
            ok = (snr > 40.0) and ac
            all_ok = all_ok and ok
            print(f"{tid:<28} {label:<7} {snr:>9.2f} {md:>10.6f} {str(ac):>9}  {'OK' if ok else 'FAIL'}")
    print("-" * 74)
    print("K-alignment guards (K=4095 must raise):")
    for tid in sorted(GUARD):
        guarded = _run_guard(tid, GUARD[tid])
        all_ok = all_ok and guarded
        print(f"  {tid:<26} K=4095 -> {'REJECTED (OK)' if guarded else 'ACCEPTED (FAIL)'}")
    print("-" * 74)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
