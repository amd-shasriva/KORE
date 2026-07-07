"""On-device calibration of the gfx950 roofline peaks (KORE P0, Phase 2).

The roofline lower bound ``T_min`` uses per-arch peak constants. The curated
defaults in ``rooflines.PEAKS`` are approximate datasheet numbers; this module
replaces them with *measured achievable* peaks so absolute SOL-attainment
(``eta``) is defensible:

  * HBM bandwidth  -> a STREAM-triad micro-benchmark ``a = b + q*c`` over large
    device arrays (traffic = 2 reads + 1 write = 3 * N * elem_bytes per pass).
  * bf16 / fp8 matrix peak -> a large SQUARE matmul (``2 N^3`` FLOPs), sized to
    be firmly compute-bound; we take the sustained achievable FLOP/s.

Measured values are piped through the EXISTING override mechanism
(``KORE_PEAK_BF16`` / ``KORE_PEAK_FP8`` / ``KORE_PEAK_HBM_BW`` read by
``rooflines.resolve_peaks``) -- this module never edits the PEAKS table. It
writes ``calibration.json`` (measured vs datasheet + ready-to-source exports).

Usage:
    python -m kore.analysis.calibrate_peaks --out data/calibration.json
    # then, to use the calibrated peaks:
    source <(python -m kore.analysis.calibrate_peaks --print-exports)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from kore.analysis.rooflines import DEFAULT_ARCH, PEAKS, detect_arch


def _batched_time(fn, iters: int, warmup: int, batches: int = 5) -> float:
    """Seconds-per-call from back-to-back calls timed with CUDA events.

    Peaks require the GPU to stay saturated (clocks boosted, no per-call sync
    bubble), so we enqueue ``iters`` calls between one start/stop event pair and
    take the fastest of ``batches`` such windows (the least-perturbed = closest to
    the achievable ceiling). Returns seconds per single call."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(batches):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3 / iters)
    return best


def measure_hbm_bw(n_bytes: int = 512 * 1024 * 1024, iters: int = 50,
                   warmup: int = 10) -> float:
    """STREAM triad achievable HBM bandwidth in bytes/s.

    ``a = b + q*c`` over three float32 arrays; traffic = 3 * N * 4 bytes (2 read,
    1 write) per pass. Returns bytes/s (achievable, typically 70-90% of datasheet).
    """
    import torch
    n = n_bytes // 4  # float32 elements per array
    a = torch.empty(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.randn(n, device="cuda", dtype=torch.float32)
    q = 3.0

    def triad():
        torch.add(b, c, alpha=q, out=a)

    t = _batched_time(triad, iters, warmup)
    traffic = 3.0 * n * 4.0
    return traffic / t


def measure_matmul_peak(n: int, dtype_str: str, iters: int = 30,
                        warmup: int = 10) -> Optional[float]:
    """Sustained achievable matrix FLOP/s from an ``n x n`` square matmul.

    ``2 n^3`` FLOPs; returns FLOP/s, or None if the dtype path is unavailable
    (e.g. fp8 ``_scaled_mm`` not present). bf16 uses ``torch.matmul`` (hipBLASLt);
    fp8 uses ``torch._scaled_mm`` (e4m3) with per-tensor scales.
    """
    import torch
    flops = 2.0 * (n ** 3)
    if "fp8" in dtype_str:
        fp8 = getattr(torch, "float8_e4m3fnuz", None) or getattr(torch, "float8_e4m3fn", None)
        smm = getattr(torch, "_scaled_mm", None)
        if fp8 is None or smm is None:
            return None
        a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(fp8)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(fp8).t().contiguous().t()
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        try:
            def mm():
                smm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            t = _batched_time(mm, iters, warmup)
        except Exception:  # noqa: BLE001 - fp8 gemm path unsupported on this stack
            return None
        return flops / t
    dt = torch.bfloat16 if "bf16" in dtype_str else torch.float16
    a = torch.randn(n, n, device="cuda", dtype=dt)
    b = torch.randn(n, n, device="cuda", dtype=dt)

    def mm():
        torch.matmul(a, b)

    t = _batched_time(mm, iters, warmup)
    return flops / t


def calibrate(arch: str, matmul_n: int = 8192, hbm_mb: int = 512,
              iters: int = 30, warmup: int = 10) -> dict:
    ds = PEAKS.get(arch, PEAKS[DEFAULT_ARCH])
    hbm = measure_hbm_bw(hbm_mb * 1024 * 1024, iters=max(iters, 50), warmup=warmup)
    bf16 = measure_matmul_peak(matmul_n, "bf16", iters=iters, warmup=warmup)
    fp8 = measure_matmul_peak(matmul_n, "fp8", iters=iters, warmup=warmup)
    exports = {"KORE_PEAK_HBM_BW": f"{hbm:.6e}"}
    if bf16:
        exports["KORE_PEAK_BF16"] = f"{bf16:.6e}"
    if fp8:
        exports["KORE_PEAK_FP8"] = f"{fp8:.6e}"
    return {
        "arch": arch,
        "matmul_n": matmul_n,
        "hbm_triad_mb": hbm_mb,
        "measured": {
            "hbm_bytes_per_s": hbm,
            "bf16_flops_per_s": bf16,
            "fp8_flops_per_s": fp8,
        },
        "datasheet": {
            "hbm_bytes_per_s": ds["hbm_bytes_per_s"],
            "bf16_flops_per_s": ds["bf16_flops_per_s"],
            "fp8_flops_per_s": ds["fp8_flops_per_s"],
        },
        "measured_over_datasheet": {
            "hbm": hbm / ds["hbm_bytes_per_s"],
            "bf16": (bf16 / ds["bf16_flops_per_s"]) if bf16 else None,
            "fp8": (fp8 / ds["fp8_flops_per_s"]) if fp8 else None,
        },
        "env_exports": exports,
    }


def _print_report(cal: dict) -> None:
    m, d = cal["measured"], cal["datasheet"]
    r = cal["measured_over_datasheet"]
    print(f"# gfx950 peak calibration  arch={cal['arch']}  (matmul n={cal['matmul_n']})")
    print(f"HBM  triad : {m['hbm_bytes_per_s']/1e12:6.2f} TB/s   "
          f"(datasheet {d['hbm_bytes_per_s']/1e12:.2f} TB/s, {r['hbm']*100:.0f}%)")
    if m["bf16_flops_per_s"]:
        print(f"bf16 matmul: {m['bf16_flops_per_s']/1e15:6.2f} PF/s   "
              f"(datasheet {d['bf16_flops_per_s']/1e15:.2f} PF/s, {r['bf16']*100:.0f}%)")
    if m["fp8_flops_per_s"]:
        print(f"fp8  matmul: {m['fp8_flops_per_s']/1e15:6.2f} PF/s   "
              f"(datasheet {d['fp8_flops_per_s']/1e15:.2f} PF/s, {r['fp8']*100:.0f}%)")
    else:
        print("fp8  matmul: (unavailable on this stack; keeping datasheet)")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate gfx950 roofline peaks (STREAM + matmul SOL)")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--hbm-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--print-exports", action="store_true",
                    help="print only `export KORE_PEAK_*=...` lines (for `source <(...)`)")
    args = ap.parse_args(argv)

    arch = args.arch or detect_arch()
    cal = calibrate(arch, matmul_n=args.matmul_n, hbm_mb=args.hbm_mb,
                    iters=args.iters, warmup=args.warmup)
    if args.print_exports:
        for k, v in cal["env_exports"].items():
            print(f"export {k}={v}")
        return 0
    _print_report(cal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2))
    print(f"\n[calibrate_peaks] wrote {out}")
    print("[calibrate_peaks] to apply: " + " ".join(f"{k}={v}" for k, v in cal["env_exports"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
