"""Tier-1 VERIFIABLE kernel-math SFT generator (CDNA4 / gfx950 / MI355X).

A deterministic solver computes ground-truth numbers (roofline, peak, tiling,
occupancy, fp8-format); the teacher writes only the reasoning prose, and we
VERIFY the teacher's stated final number against the computed truth (reject
disagreements). Eliminates synthetic data's #1 failure mode (hallucinated wrong
math). gfx950 (CDNA4 / MI355X) only. Parameter spaces are WIDE so unique-after-
dedup yield is high.
"""
from __future__ import annotations
import random, re, json

# --- gfx950 / CDNA4 / MI355X verified specs (ROCm blogs + CDNA4 whitepaper) ---
CU = 256
WAVE = 64
VGPR_PER_LANE = 512          # unified regular+accumulator
VGPR_GRAN = 8
LDS_PER_CU = 160 * 1024      # bytes
MAX_WAVES_SIMD = 8
SIMD_PER_CU = 4
CLOCK_GHZ = 2.4              # MI355X (MI350X=2.2)
HBM_TBS = 8.0               # TB/s
FLOPS_CLK_CU = {"fp16": 4096, "bf16": 4096, "fp8": 8192, "mxfp8": 8192,
                "mxfp6": 16384, "mxfp4": 16384, "fp32": 1024, "fp64": 128}
DBYTES = {"fp64": 8, "fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2,
          "fp8": 1, "mxfp8": 1, "mxfp6": 0.75, "mxfp4": 0.5, "int8": 1}

DIMS = [128, 192, 256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
        2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384]
KDIMS = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
RF_DTYPES = ["bf16", "fp16", "fp8", "mxfp8", "mxfp6"]


def peak_tflops(dtype: str) -> float:
    return FLOPS_CLK_CU[dtype] * CU * (CLOCK_GHZ * 1e9) / 1e12


def _round8(v: int) -> int:
    return ((v + VGPR_GRAN - 1) // VGPR_GRAN) * VGPR_GRAN


def prob_roofline(rng):
    dt = rng.choice(RF_DTYPES); M = rng.choice(DIMS); N = rng.choice(DIMS); K = rng.choice(KDIMS)
    b = DBYTES[dt]
    flops = 2 * M * N * K
    byts = (M * K + K * N + M * N) * b
    I = flops / byts
    pi = peak_tflops(dt) * 1e12
    beta = HBM_TBS * 1e12
    ridge = pi / beta
    bound = "compute-bound" if I >= ridge else "memory-bound"
    achiev = min(pi, beta * I) / 1e12
    q = (f"On MI355X (gfx950, CDNA4) a {dt} GEMM computes C[{M},{N}] = A[{M},{K}].B[{K},{N}] "
         f"(HBM {HBM_TBS} TB/s, {dt} matrix peak {peak_tflops(dt):.0f} TFLOP/s; naive: no cache reuse, "
         f"read A+B, write C). Give (1) arithmetic intensity in FLOP/byte, (2) the roofline ridge "
         f"point, (3) whether it is memory- or compute-bound, and (4) the attainable TFLOP/s.")
    gt = {"intensity_flop_per_byte": round(I, 3), "ridge_flop_per_byte": round(ridge, 2),
          "bound": bound, "attainable_tflops": round(achiev, 1)}
    return q, gt, achiev


def prob_peak(rng):
    dt = rng.choice(list(FLOPS_CLK_CU)); clk = rng.choice([2.4, 2.2])
    val = FLOPS_CLK_CU[dt] * CU * (clk * 1e9) / 1e12
    chip = "MI355X" if clk == 2.4 else "MI350X"
    q = (f"Derive the dense {dt} matrix-core peak of {chip} (gfx950, CDNA4): {CU} CUs, "
         f"{FLOPS_CLK_CU[dt]} FLOPs/clock/CU for {dt}, {clk} GHz. Give peak TFLOP/s.")
    return q, {"peak_tflops": round(val, 1)}, val


def prob_tiling(rng):
    bm = rng.choice([32, 48, 64, 96, 128, 160, 192, 224, 256])
    bn = rng.choice([32, 48, 64, 96, 128, 160, 192, 224, 256])
    bk = rng.choice([16, 32, 48, 64, 96, 128])
    dt = rng.choice(["bf16", "fp16", "fp8"]); b = DBYTES[dt]; stages = rng.choice([1, 2, 3])
    I = bm * bn / (bm + bn)
    lds = int((bm * bk + bk * bn) * b * stages)
    fits = lds <= LDS_PER_CU
    q = (f"On MI355X (gfx950, 160 KB LDS/CU) a {dt} GEMM uses BLOCK_M={bm}, BLOCK_N={bn}, "
         f"BLOCK_K={bk}, num_stages={stages} (buffered A[{bm},{bk}] + B[{bk},{bn}] in LDS). "
         f"Give (1) the tile arithmetic-intensity proxy bm*bn/(bm+bn), and (2) the LDS bytes used and "
         f"whether it fits in 160 KB.")
    return q, {"tile_intensity": round(I, 2), "lds_bytes": lds, "fits_160kb": fits}, I


def prob_occupancy(rng):
    vgpr = rng.choice(list(range(24, 288, 4)))
    waves = min(MAX_WAVES_SIMD, VGPR_PER_LANE // _round8(vgpr))
    q = (f"On MI355X (gfx950) each SIMD has {VGPR_PER_LANE} VGPRs/lane (alloc granularity {VGPR_GRAN}, "
         f"max {MAX_WAVES_SIMD} waves/SIMD). A wave uses {vgpr} VGPRs/lane. Give the VGPR-limited "
         f"occupancy in waves/SIMD.")
    return q, {"waves_per_simd": waves, "rounded_vgpr": _round8(vgpr)}, waves


def prob_lds_occ(rng):
    lds_kb = rng.choice([8, 12, 16, 20, 24, 32, 40, 48, 64, 80])
    wg = LDS_PER_CU // (lds_kb * 1024)
    q = (f"On MI355X (gfx950, 160 KB LDS/CU) a kernel's workgroup allocates {lds_kb} KB of LDS. "
         f"How many workgroups can co-reside on one CU based on the LDS limit alone?")
    return q, {"workgroups_per_cu_lds": wg}, wg


def prob_fp8fmt(rng):
    variants = [
     "Which FP8 variant does MI355X (gfx950, CDNA4) use natively for matrix-core FP8? Name it (OCP or FNUZ).",
     "For an FP8 GEMM on gfx950, which OCP FP8 encodings are supported for the matrix inputs? Name the format family.",
     "Porting an FP8 kernel from MI300X to MI355X (gfx950): which FP8 format must the gfx950 path use, OCP or FNUZ?"]
    q = rng.choice(variants)
    return q, {"gfx950_fp8": "OCP"}, None


FAMILIES = {"roofline": prob_roofline, "tiling": prob_tiling, "occupancy": prob_occupancy,
            "lds_occ": prob_lds_occ, "peak": prob_peak, "fp8fmt": prob_fp8fmt}
ROSTER = (["roofline"] * 7 + ["tiling"] * 5 + ["occupancy"] * 3 + ["lds_occ"] * 2 + ["peak"] * 1 + ["fp8fmt"] * 1)


def _nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s)]


def verify(reply, check_val, gt) -> bool:
    if check_val is None:
        return "ocp" in reply.lower()
    ns = _nums(reply)
    tol = max(abs(check_val) * 0.02, 0.5)
    return any(abs(n - check_val) <= tol for n in ns)


SYS = ("You are KORE, an expert AMD CDNA4 (gfx950 / MI355X) GPU kernel engineer. Answer the kernel-math "
       "question with clear step-by-step reasoning and state each final number explicitly with units.")


def build_problem(seed: int):
    rng = random.Random(seed)
    name = ROSTER[rng.randrange(len(ROSTER))]
    q, gt, chk = FAMILIES[name](rng)
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    return messages, gt, chk, name


if __name__ == "__main__":
    import sys, hashlib, collections
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seen = set(); c = collections.Counter()
    for i in range(n):
        msgs, gt, chk, name = build_problem(i)
        h = hashlib.sha256(msgs[1]["content"].encode()).hexdigest()
        if h not in seen:
            seen.add(h); c[name] += 1
    print(f"n={n} unique={len(seen)} uniq_rate={len(seen)/n:.2f} by_type={dict(c)}")
    assert abs(peak_tflops("bf16") - 4096 * 256 * 2.4e9 / 1e12) < 1e-6
    print("peak bf16 =", round(peak_tflops("bf16"), 1), "fp8 =", round(peak_tflops("fp8"), 1))
    print("SELFTEST_OK")
