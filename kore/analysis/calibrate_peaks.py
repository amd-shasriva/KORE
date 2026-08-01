"""On-device calibration of the gfx950 roofline peaks (KORE P0, Phase 2).

The roofline lower bound ``T_min`` uses per-SKU peak constants. The curated
defaults in :mod:`kore.analysis.roofline` are datasheet numbers; this module
replaces them with *measured achievable* peaks so absolute SOL-attainment
(``eta``) is defensible:

  * HBM bandwidth  -> a STREAM-triad micro-benchmark ``a = b + q*c`` over large
    device arrays (traffic = 2 reads + 1 write = 3 * N * elem_bytes per pass).
  * bf16 / fp8 matrix peak -> a large SQUARE matmul (``2 N^3`` FLOPs), sized to
    be firmly compute-bound; we take the sustained achievable FLOP/s.

Output contract: a ``kore.runtime-calibration.v1`` DOCUMENT (the only format
:func:`kore.analysis.roofline.make_physical_model` accepts) plus its model
fingerprint. The document is validated by building the model from it before it
is written, so an artifact that cannot be applied never reaches disk. This
module never edits the datasheet table.

The previous output contract -- ``export KORE_PEAK_*`` lines -- was a no-op:
nothing has read those variables since process-global peak overrides were
removed for being unfingerprinted, so a reproduce path that sourced them
silently ran on datasheet peaks (roughly a 2x shift in ``eta``). The supported
replacement is the calibration path plus its fingerprint pin.

Usage:
    # 1. measure and write the calibration document
    python -m kore.analysis.calibrate_peaks --sku mi350x --out data/calibration.json

    # 2. apply it (fingerprint-pinned; fails closed on any mismatch)
    source <(python -m kore.analysis.calibrate_peaks \
                 --calibration data/calibration.json --print-exports)

    # 3. or verify/inspect an existing document without a GPU
    python -m kore.analysis.calibrate_peaks --calibration data/calibration.json --verify
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Optional

from kore.analysis.roofline import ModelError, hardware_spec
from kore.analysis.rooflines import (
    CALIBRATION_ENV_VAR,
    FINGERPRINT_ENV_VAR,
    calibration_document,
    load_calibration,
    resolve_model,
    resolve_sku,
)

# dtypes we actually measure; every other supported dtype keeps its datasheet
# peak so a calibration never silently deletes an operator's roofline.
MEASURED_DTYPES: tuple[str, ...] = ("bf16", "fp8")


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


def runtime_identity() -> dict[str, str]:
    """Stack identity recorded in the calibration document.

    ``make_physical_model`` rejects a calibration without runtime metadata: peaks
    are only meaningful together with the ROCm/torch/driver stack that produced
    them, and the fingerprint must change when the stack does.
    """
    identity = {"host": platform.node(), "python": platform.python_version()}
    try:
        import torch
        identity["torch"] = str(torch.__version__)
        identity["hip"] = str(getattr(torch.version, "hip", "") or "none")
        if torch.cuda.is_available():
            identity["device_name"] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            identity["gcn_arch"] = str(getattr(props, "gcnArchName", "") or "")
    except Exception:  # noqa: BLE001 - torch missing/unusable is not fatal here
        identity.setdefault("torch", "unavailable")
    return {key: value for key, value in identity.items() if value}


def calibrate(sku: str, matmul_n: int = 8192, hbm_mb: int = 512,
              iters: int = 30, warmup: int = 10,
              calibration_id: Optional[str] = None) -> dict:
    """Measure achievable peaks and return a ``kore.runtime-calibration.v1`` doc.

    Measured dtypes override the datasheet; every other dtype the SKU supports
    keeps its datasheet peak (dropping them would delete those operators'
    rooflines entirely). ``measured_dtypes`` / ``datasheet_dtypes`` record which
    is which, and ``measured_over_datasheet`` carries the achievable fractions.
    """
    spec = hardware_spec(sku)
    datasheet = dict(spec.compute_flops_per_s)
    hbm = measure_hbm_bw(hbm_mb * 1024 * 1024, iters=max(iters, 50), warmup=warmup)
    measured: dict[str, Optional[float]] = {}
    for dtype in MEASURED_DTYPES:
        if dtype in datasheet:
            measured[dtype] = measure_matmul_peak(matmul_n, dtype, iters=iters, warmup=warmup)

    compute = dict(datasheet)
    applied: list[str] = []
    for dtype, value in measured.items():
        if value and value > 0.0:
            compute[dtype] = float(value)
            applied.append(dtype)

    ratios = {
        "hbm": hbm / spec.hbm_bytes_per_s,
        **{
            dtype: (value / datasheet[dtype]) if value else None
            for dtype, value in measured.items()
        },
    }
    identifier = calibration_id or (
        f"{spec.sku.lower()}-stream-matmul-n{matmul_n}-hbm{hbm_mb}mb"
    )
    return calibration_document(
        spec.sku.lower(),
        hbm_bytes_per_s=hbm,
        compute_flops_per_s=compute,
        calibration_id=identifier,
        runtime=runtime_identity(),
        source="runtime-measured-stream-matmul",
        extra={
            "matmul_n": matmul_n,
            "hbm_triad_mb": hbm_mb,
            "measured_dtypes": sorted(applied),
            "datasheet_dtypes": sorted(set(datasheet) - set(applied)),
            "measured": {"hbm_bytes_per_s": hbm,
                         **{f"{d}_flops_per_s": v for d, v in measured.items()}},
            "datasheet": {"hbm_bytes_per_s": spec.hbm_bytes_per_s,
                          **{f"{d}_flops_per_s": v for d, v in datasheet.items()}},
            "measured_over_datasheet": ratios,
        },
    )


def _print_report(cal: dict) -> None:
    measured = cal.get("measured") or {}
    datasheet = cal.get("datasheet") or {}
    ratios = cal.get("measured_over_datasheet") or {}
    print(f"# peak calibration  sku={cal['sku']}  arch={cal['architecture']}  "
          f"(matmul n={cal.get('matmul_n')})")
    hbm = measured.get("hbm_bytes_per_s")
    if hbm:
        print(f"HBM  triad : {hbm/1e12:6.2f} TB/s   "
              f"(datasheet {datasheet.get('hbm_bytes_per_s', 0.0)/1e12:.2f} TB/s, "
              f"{(ratios.get('hbm') or 0.0)*100:.0f}%)")
    for dtype in MEASURED_DTYPES:
        value = measured.get(f"{dtype}_flops_per_s")
        if value:
            print(f"{dtype:<4} matmul: {value/1e15:6.2f} PF/s   "
                  f"(datasheet {datasheet.get(f'{dtype}_flops_per_s', 0.0)/1e15:.2f} PF/s, "
                  f"{(ratios.get(dtype) or 0.0)*100:.0f}%)")
        else:
            print(f"{dtype:<4} matmul: (unavailable on this stack; keeping datasheet)")
    print(f"applied measured dtypes : {cal.get('measured_dtypes')}")
    print(f"datasheet-kept dtypes   : {cal.get('datasheet_dtypes')}")
    print(f"model fingerprint       : {cal.get('model_fingerprint')}")


def _exports(path: Path, fingerprint: str) -> list[str]:
    """The SUPPORTED apply contract: an explicit, fingerprint-pinned calibration."""
    return [
        f"export {CALIBRATION_ENV_VAR}={path}",
        f"export {FINGERPRINT_ENV_VAR}={fingerprint}",
    ]


def verify(path: str, *, sku: Optional[str] = None,
           expected_fingerprint: Optional[str] = None) -> dict:
    """Load a calibration document and prove it still applies to its SKU.

    Fails closed: a document whose ``model_fingerprint`` does not reproduce (or
    that disagrees with ``expected_fingerprint``) raises ``ModelError`` rather
    than being applied.
    """
    document = load_calibration(path)
    selected = sku or document.get("sku")
    if not selected:
        raise ModelError("calibration document has no SKU; it cannot be applied safely")
    pin = expected_fingerprint or document.get("model_fingerprint")
    model = resolve_model(sku=str(selected).lower(), calibration=document,
                          expected_fingerprint=pin)
    return {
        "path": str(path),
        "sku": model.sku,
        "architecture": model.architecture,
        "calibration_id": model.calibration_id,
        "calibration_source": model.calibration_source,
        "hbm_bytes_per_s": model.hbm_bytes_per_s,
        "compute_flops_per_s": dict(sorted(model.compute_flops_per_s.items())),
        "model_fingerprint": model.fingerprint,
        "runtime": dict(model.runtime),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate gfx950 roofline peaks (STREAM + matmul SOL) into a "
                    "kore.runtime-calibration.v1 document")
    ap.add_argument("--sku", default=None,
                    help="explicit hardware SKU (mi350x|mi355x|mi300x); observed from the "
                         "device when omitted")
    ap.add_argument("--arch", default=None, help="architecture used to resolve --sku")
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--hbm-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--calibration", default=None,
                    help="apply/verify an EXISTING calibration document instead of measuring")
    ap.add_argument("--expect-fingerprint", default=None,
                    help="require this model fingerprint (fails closed on mismatch)")
    ap.add_argument("--verify", action="store_true",
                    help="with --calibration: print the resolved peaks and fingerprint")
    ap.add_argument("--print-exports", action="store_true",
                    help=f"print only `export {CALIBRATION_ENV_VAR}/{FINGERPRINT_ENV_VAR}` "
                         "lines (for `source <(...)`)")
    args = ap.parse_args(argv)

    # Apply/verify path: no GPU needed, and it never measures.
    if args.calibration:
        try:
            info = verify(args.calibration, sku=args.sku,
                          expected_fingerprint=args.expect_fingerprint)
        except ModelError as exc:
            # Fail closed with an actionable message: a calibration that cannot be
            # verified must never be applied, and a pre-v1 artifact (no SKU/runtime
            # identity) has to be re-measured rather than patched.
            print(f"[calibrate_peaks] REFUSED {args.calibration}: {exc}", file=sys.stderr)
            print("[calibrate_peaks] re-measure with: python -m kore.analysis.calibrate_peaks "
                  "--sku <mi350x|mi355x> --out <path>", file=sys.stderr)
            return 2
        if args.print_exports:
            for line in _exports(Path(args.calibration).resolve(), info["model_fingerprint"]):
                print(line)
            return 0
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    sku, source = resolve_sku(args.arch, args.sku)
    cal = calibrate(sku, matmul_n=args.matmul_n, hbm_mb=args.hbm_mb,
                    iters=args.iters, warmup=args.warmup)
    out = Path(args.out)
    if args.print_exports:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cal, indent=2, sort_keys=True))
        for line in _exports(out.resolve(), cal["model_fingerprint"]):
            print(line)
        return 0
    print(f"[calibrate_peaks] sku={sku} (source={source})")
    _print_report(cal)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2, sort_keys=True))
    print(f"\n[calibrate_peaks] wrote {out}")
    print("[calibrate_peaks] to apply: source <(python -m kore.analysis.calibrate_peaks "
          f"--calibration {out} --print-exports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
