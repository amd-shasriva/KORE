"""CPU sanity check for the DRAFT attention oracles (reference-correctness only).

Runs each task's ``reference.reference_output`` on CPU against a SMALL random input and
compares it to a fully INDEPENDENT brute-force attention computed here with
``torch.softmax`` + ``torch.einsum`` (a different code path than the hand-rolled
``_attn_common.sdpa_fp32`` the references use), plus independently-constructed masks. If
the two agree to within bf16 precision, the oracle math (softmax numerics, causal /
sliding / bottom-right masks, GQA/MQA broadcast, fp8 dequant, ragged per-sequence
bounds, gpt-oss sink) is corroborated.

This proves REFERENCE CORRECTNESS ON CPU ONLY. It does NOT compile the Triton seeds, run
the AITER vendor baselines, or measure anything on gfx950 -- see VERIFICATION_CHECKLIST.md
for what still must be verified on-GPU before promotion. Run from the repo root:

    python kore/tasks/_drafts/attention/_cpu_sanity_check.py
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ref(task_id):
    path = os.path.join(HERE, task_id, "reference.py")
    spec = importlib.util.spec_from_file_location(f"draft_ref_{task_id}", path)
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


def _brute(qf, kf, vf, scale, mask=None, sink=None):
    """Independent attention: torch.softmax over q.k^T (+mask), optional gpt-oss sink.

    qf [B,H,Sq,D], kf/vf [B,H,Sk,D] (KV already expanded). Returns [B,H,Sq,D] fp32."""
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    if mask is not None:
        scores = scores + mask
    if sink is not None:
        B, H, Sq, Sk = scores.shape
        sink_col = sink.view(1, H, 1, 1).expand(B, H, Sq, 1).float()
        combined = torch.cat([scores, sink_col], dim=-1)      # append sink logit column
        p = torch.softmax(combined, dim=-1)[..., :Sk]         # drop the sink column
    else:
        p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, vf)


def _expand(t, H):  # independent GQA/MQA expand
    KV = t.shape[1]
    return t if KV == H else t.repeat_interleave(H // KV, dim=1)


def _causal(Sq, Sk, q_off=0):
    i = torch.arange(Sq)[:, None] + q_off
    j = torch.arange(Sk)[None, :]
    return torch.where(j <= i, 0.0, float("-inf"))


def _sliding(Sq, Sk, W, q_off=0):
    i = torch.arange(Sq)[:, None] + q_off
    j = torch.arange(Sk)[None, :]
    return torch.where((j <= i) & (j > i - W), 0.0, float("-inf"))


# task_id -> (tiny shape, independent-oracle builder). Each builder returns a bf16 tensor
# in the SAME layout/dtype as reference_output, computed independently of _attn_common.
def _independent(task_id, shape, inputs):
    if task_id in ("flash_attn_mha_prefill_bf16", "flash_attn_mqa_prefill_bf16",
                   "flash_attn_headdim_prefill_bf16"):
        q, k, v = inputs
        B, S, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=_causal(S, S))
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_noncausal_prefill_bf16":
        q, k, v = inputs
        B, S, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=None)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_noncausal_fp8":
        q, k, v, sq, sk, sv = inputs
        B, S, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute((q.float() * float(sq)).transpose(1, 2),
                   _expand((k.float() * float(sk)).transpose(1, 2), H),
                   _expand((v.float() * float(sv)).transpose(1, 2), H), sc, mask=None)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_mqa_decode_bf16":
        q, k, v = inputs
        B, Sq, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=None)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_decode_fp8":
        q, k, v, sq, sk, sv = inputs
        B, Sq, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute((q.float() * float(sq)).transpose(1, 2),
                   _expand((k.float() * float(sk)).transpose(1, 2), H),
                   _expand((v.float() * float(sv)).transpose(1, 2), H), sc, mask=None)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_sliding_decode_bf16":
        q, k, v = inputs
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        W = int(shape["W"])
        sc = 1.0 / (D ** 0.5)
        mask = _sliding(1, Skv, W, q_off=Skv - 1)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=mask)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_sink_prefill_bf16":
        q, k, v, sink = inputs
        B, S, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=_causal(S, S),
                   sink=sink.float())
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_chunked_prefill_bf16":
        q, k, v = inputs
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        sc = 1.0 / (D ** 0.5)
        mask = _causal(Sq, Skv, q_off=Skv - Sq)
        o = _brute(q.float().transpose(1, 2), _expand(k.float().transpose(1, 2), H),
                   _expand(v.float().transpose(1, 2), H), sc, mask=mask)
        return o.transpose(1, 2).to(torch.bfloat16)
    if task_id == "flash_attn_varlen_noncausal_bf16":
        q, k, v, cu = inputs
        total, H, D = q.shape
        sc = 1.0 / (D ** 0.5)
        out = torch.empty((total, H, D), dtype=torch.bfloat16)
        cul = cu.tolist()
        for s in range(len(cul) - 1):
            a, b = cul[s], cul[s + 1]
            L = b - a
            if L <= 0:
                continue
            qf = q[a:b].float().transpose(0, 1).unsqueeze(0)
            kf = _expand(k[a:b].float().transpose(0, 1).unsqueeze(0), H)
            vf = _expand(v[a:b].float().transpose(0, 1).unsqueeze(0), H)
            o = _brute(qf, kf, vf, sc, mask=None)   # non-causal (bidirectional)
            out[a:b] = o.squeeze(0).transpose(0, 1).to(torch.bfloat16)
        return out
    raise KeyError(task_id)


TINY = {
    "flash_attn_mha_prefill_bf16": {"B": 1, "H": 4, "KV": 4, "S": 16, "D": 32},
    "flash_attn_noncausal_prefill_bf16": {"B": 1, "H": 4, "KV": 2, "S": 16, "D": 32},
    "flash_attn_mqa_prefill_bf16": {"B": 1, "H": 4, "KV": 1, "S": 16, "D": 32},
    "flash_attn_headdim_prefill_bf16": {"B": 1, "H": 4, "KV": 2, "S": 16, "D": 48},
    "flash_attn_noncausal_fp8": {"B": 1, "H": 4, "KV": 2, "S": 16, "D": 32},
    "flash_attn_mqa_decode_bf16": {"B": 2, "H": 4, "KV": 1, "Skv": 20, "D": 32},
    "flash_attn_decode_fp8": {"B": 2, "H": 4, "KV": 2, "Skv": 20, "D": 32},
    "flash_attn_varlen_noncausal_bf16": {"B": 3, "H": 4, "KV": 2, "S": 8, "D": 32},
    "flash_attn_sliding_decode_bf16": {"B": 2, "H": 4, "KV": 2, "Skv": 20, "D": 32, "W": 5},
    "flash_attn_sink_prefill_bf16": {"B": 1, "H": 4, "KV": 2, "S": 12, "D": 32},
    "flash_attn_chunked_prefill_bf16": {"B": 1, "H": 4, "KV": 2, "Sq": 4, "Skv": 12, "D": 32},
}

# Also exercise a NON-power-of-2 seqlen edge and W>=Skv (full) / Sq==Skv (ordinary causal).
EXTRA = {
    "flash_attn_mha_prefill_bf16": {"B": 1, "H": 4, "KV": 4, "S": 13, "D": 32},
    "flash_attn_sliding_decode_bf16": {"B": 1, "H": 4, "KV": 2, "Skv": 7, "D": 32, "W": 999},
    "flash_attn_chunked_prefill_bf16": {"B": 1, "H": 4, "KV": 2, "Sq": 8, "Skv": 8, "D": 32},
    "flash_attn_mqa_prefill_bf16": {"B": 1, "H": 6, "KV": 1, "S": 9, "D": 64},
}


def _run_one(task_id, shape):
    ref = _load_ref(task_id)
    inputs = ref.get_inputs(shape, device="cpu", seed=0)
    got = ref.reference_output(shape, inputs)
    exp = _independent(task_id, shape, inputs)
    assert got.shape == exp.shape, f"{task_id}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    assert got.dtype == exp.dtype, f"{task_id}: dtype {got.dtype} != {exp.dtype}"
    snr = _snr_db(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    ac = torch.allclose(got.float(), exp.float(), atol=2e-2, rtol=2e-2)
    return snr, md, ac


def main():
    tasks = sorted(TINY)
    all_ok = True
    print(f"{'task_id':<38} {'shape':<7} {'SNR(dB)':>9} {'max_diff':>10} {'allclose':>9}")
    print("-" * 78)
    for tid in tasks:
        for label, table in (("tiny", TINY), ("extra", EXTRA)):
            if tid not in table:
                continue
            snr, md, ac = _run_one(tid, dict(table[tid]))
            ok = (snr > 40.0) and ac
            all_ok = all_ok and ok
            flag = "OK" if ok else "FAIL"
            print(f"{tid:<38} {label:<7} {snr:>9.2f} {md:>10.5f} {str(ac):>9}  {flag}")
    print("-" * 78)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
