import os, sys, json, random, time, glob
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout
sys.path.insert(0, "/home/shasriva/Kore-RL/KORE")
from kore.data.teacher import make_teacher

OUTDIR = "/home/shasriva/kore_offline/gen"; os.makedirs(OUTDIR, exist_ok=True)
WORKERS = int(os.environ.get("GEN_WORKERS", "128"))
t = make_teacher("claude", resilient=True)

GFX = ("Target: MI355X / gfx950 / CDNA4. Facts: 256 CUs, wave64, 512 VGPR/lane (unified regular+"
 "accumulator, granularity 8), 160KB LDS/CU, max 8 waves/SIMD; BF16 matrix 4096 FLOP/clk/CU @2.4GHz "
 "(~2.5 PFLOP BF16, ~5 PFLOP FP8, ~10 PFLOP MXFP6/FP4 dense); 288GB HBM3E @8.0TB/s; OCP FP8 "
 "(E4M3FN/E5M2, NOT FNUZ); scaled-MFMA v_mfma_scale_f32_16x16x128_f8f6f4 for MXFP; FP32 accumulator "
 "for all <=16-bit; buffer_load_to_lds saves ~100 VGPR/wave; mfma_16x16 usually beats 32x32; bf16>f16; "
 "num_stages: single-GEMM=2, fused-two-GEMM (flash-attn)=1.")
SYS = ("You are KORE, an elite AMD CDNA4 (gfx950/MI355X) GPU kernel engineer. Be precise, correct, and "
       "concrete. Ground every claim in the real gfx950 hardware. " + GFX)


def gen(sys_p, user, source, meta=None):
    try:
        r = t.generate([{"role": "system", "content": sys_p}, {"role": "user", "content": user}])
    except Exception:
        return None
    if not r or len(r) < 60:
        return None
    row = {"messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user},
                        {"role": "assistant", "content": r}], "_source": source, "_arch": "gfx950"}
    if meta:
        row.update(meta)
    return row


QA_ASKS = [
 "Explain what this kernel computes, then identify its single biggest performance bottleneck on gfx950 and the first optimization you would apply.",
 "Rewrite the memory-access pattern of this kernel for coalesced HBM loads on gfx950 and explain why it helps.",
 "How would you tile this kernel to exploit the 160KB LDS on gfx950? Give concrete BLOCK_M/N/K and the LDS budget.",
 "Would this kernel benefit from buffer_load_to_lds on gfx950? Explain the VGPR-pressure effect.",
 "Port the key idea of this kernel to an efficient Triton kernel for gfx950; explain the tl.dot / MFMA mapping.",
 "What num_stages and num_warps would you pick for this kernel on gfx950, and why?",
 "If this ran in FP8 on gfx950, which OCP format and scaled-MFMA instruction applies, and what changes vs bf16?",
 "Diagnose likely LDS bank conflicts or wavefront divergence in this kernel on gfx950 and how to fix them.",
 "Estimate whether this kernel is memory- or compute-bound on gfx950 and justify with a roofline argument.",
 "Which AMD vendor library (rocBLAS/hipBLASLt/CK/aiter) would you compare against for this op on gfx950, and when would a custom kernel win?",
]


def tier2(kb, n, seed=0):
    rng = random.Random(seed); specs = []
    for i in range(n):
        k = rng.choice(kb); ask = rng.choice(QA_ASKS)
        snip = k.get("text", "")[:2400]
        u = "Here is a real GPU kernel / PyTorch module:\n\n```\n" + snip + "\n```\n\n" + ask
        specs.append((SYS, u, "kernel_qa", {"_qa_type": "grounded"}))
    return specs


TOPICS = ["the CDNA4 memory hierarchy (registers, LDS, L2, HBM) and their bandwidths",
 "what a wavefront is and why gfx950 uses wave64", "occupancy on gfx950 and the four limiters (VGPR/SGPR/LDS/workgroup)",
 "the roofline model and arithmetic intensity for GPU kernels", "memory coalescing and why it matters on HBM",
 "LDS bank conflicts and how to avoid them", "tiling / blocking for GEMM and the LDS reuse argument",
 "the MFMA matrix cores on CDNA4 and supported dtypes", "FP32 accumulation for <=16-bit MFMA and why",
 "OCP vs FNUZ FP8 and the gfx950 change", "MXFP8/MXFP6/MXFP4 microscaling and scaled-MFMA on gfx950",
 "num_stages / software pipelining in Triton on AMD", "buffer_load_to_lds direct global-to-LDS async copy",
 "flash-attention kernel structure and why num_stages=1", "reduction and scan parallel patterns on GPUs",
 "control divergence and its cost on wave64", "how rocprofv3 counters (VALUUtilization, LDSBankConflict, MemUnitBusy) guide tuning",
 "the ridge point and how to move a kernel from memory- to compute-bound", "register spilling and its performance cliff on gfx950",
 "why bf16 matrix throughput beats f16 on CDNA4", "the 512-VGPR budget and its effect on GEMM tile size",
 "structured 2:4 sparsity on CDNA4 matrix cores", "grid/block launch configuration heuristics for gfx950",
 "vectorized (128-bit) loads and their effect on memory throughput", "kernel fusion tradeoffs (compute vs memory savings)"]


def tier3(n, seed=0):
    rng = random.Random(seed); specs = []
    styles = ["Explain to a strong engineer, with a concrete gfx950 example and the key formula.",
              "Give a rigorous but concise tutorial, including when it matters and a pitfall.",
              "Teach this as if writing a textbook section for CDNA4 kernel engineers."]
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]; st = rng.choice(styles)
        u = "Teach: " + topic + ". " + st
        specs.append((SYS, u, "kernel_concept", {"_topic": topic}))
    return specs


def tier4(wins, n, seed=0):
    rng = random.Random(seed); specs = []
    pool = [w for w in wins if w.get("final_source") and (w.get("speedup") or 0) > 1.05]
    if not pool:
        return specs
    for i in range(n):
        w = rng.choice(pool); src = str(w.get("final_source"))[:2600]
        op = w.get("operation") or w.get("task_id") or "kernel"; sp = w.get("speedup") or 1.0
        u = ("This Triton kernel for '" + str(op) + "' on gfx950 was measured at " + ("%.2f" % sp) +
             "x over the baseline on real MI355X silicon:\n\n```python\n" + src + "\n```\n\n"
             "As a kernel engineer, give the CONCISE reasoning trace explaining WHY it is fast on gfx950 "
             "(the specific optimizations and hardware features it exploits) and what you would try next to push it further.")
        specs.append((SYS, u, "kernel_reasoning", {"_op": op, "_speedup": sp}))
    return specs


SCEN = ["make it work and stay fast in FP8 (OCP) on gfx950", "add varlen/ragged sequence support without losing MFMA utilization",
 "fuse a bias+activation epilogue without extra HBM round-trips", "retune tiles for the 160KB LDS to raise arithmetic intensity",
 "handle a K dimension not a multiple of 64 while keeping MFMA efficient", "convert a scalar FMA inner loop to tl.dot / MFMA",
 "reduce VGPR pressure with buffer_load_to_lds to raise occupancy", "pick num_stages/num_warps for a fused two-GEMM flash-attention"]


def tier5(n, seed=0):
    rng = random.Random(seed); specs = []
    ops = ["gemm", "flash-attention", "rmsnorm", "layernorm", "moe grouped-gemm", "fp8 gemm", "softmax", "rope", "gelu-gemm"]
    for i in range(n):
        op = rng.choice(ops); sc = rng.choice(SCEN)
        u = ("You have a working " + op + " Triton kernel for gfx950. Task: " + sc + ". Give the concrete "
             "kernel-engineering plan (what to change and why, with gfx950 specifics) and the key code changes.")
        specs.append((SYS, u, "kernel_evol", {"_op": op}))
    return specs


WRONG = ["remove the fp32 accumulator and accumulate in bf16 to save registers",
 "hardcode the expected output shape/values to pass the correctness check faster",
 "use mfma_32x32 everywhere because bigger tiles are always faster on gfx950",
 "skip the mask on boundary tiles to remove branches", "use FNUZ FP8 on gfx950 for portability with MI300",
 "raise num_stages to 6 on a single-GEMM to hide all latency", "drop the barriers/__syncthreads to reduce overhead",
 "assume 64KB LDS like CDNA3 when sizing tiles on gfx950", "use non-multiple-of-64 block sizes to fit more waves"]


def tier6(n, seed=0):
    rng = random.Random(seed); specs = []
    for i in range(n):
        w = rng.choice(WRONG)
        u = ("A teammate proposes this optimization for a gfx950 kernel: \"" + w + "\". Is this correct? "
             "Explain precisely why it is wrong (or harmful) on gfx950 and what the correct approach is.")
        specs.append((SYS, u, "kernel_distractor", {}))
    return specs


def run(specs, out, budget_s):
    """Generate with a HARD wall-clock budget: never blocks past budget_s, always
    writes whatever completed, and force-cancels any still-pending calls on exit."""
    t0 = time.time(); kept = 0; deadline = t0 + budget_s
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futs = {ex.submit(gen, *s): s for s in specs}
    try:
        with open(out, "w") as f:
            it = as_completed(futs, timeout=budget_s)
            n_done = 0
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(out + ": DEADLINE reached, kept=" + str(kept), flush=True)
                    break
                try:
                    fu = next(it)
                except (StopIteration, FutTimeout):
                    break
                n_done += 1
                row = fu.result()
                if row:
                    f.write(json.dumps(row) + "\n"); kept += 1
                if n_done % 250 == 0:
                    print(out + ": progress done=" + str(n_done) + "/" + str(len(specs)) +
                          " kept=" + str(kept) + " elapsed=" + ("%.0f" % (time.time() - t0)) + "s", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print(out + ": kept=" + str(kept) + "/" + str(len(specs)) + " " + ("%.0f" % (time.time() - t0)) + "s FINAL", flush=True)
    return kept


def rd(p):
    return [json.loads(l) for l in open(p) if l.strip()]


if __name__ == "__main__":
    kb = rd("/home/shasriva/kore_offline/kb.jsonl")
    wins = []
    for p in glob.glob("/home/shasriva/Kore-RL/KORE/data/b05factory/wins/*.jsonl"):
        try:
            wins += rd(p)
        except Exception:
            pass
    print("grounding: kb=" + str(len(kb)) + " wins=" + str(len(wins)), flush=True)
    # per-tier hard wall-clock budgets (seconds) - bounded total ~40 min worst case
    B2 = int(os.environ.get("BUDGET_T2", "600"))
    B3 = int(os.environ.get("BUDGET_T3", "480"))
    B4 = int(os.environ.get("BUDGET_T4", "600"))
    B5 = int(os.environ.get("BUDGET_T5", "480"))
    B6 = int(os.environ.get("BUDGET_T6", "480"))
    run(tier2(kb, 5000, 2), OUTDIR + "/kernel_qa.jsonl", B2)
    run(tier3(2000, 3), OUTDIR + "/kernel_concept.jsonl", B3)
    run(tier4(wins, 4000, 4), OUTDIR + "/kernel_reasoning.jsonl", B4)
    run(tier5(3000, 5), OUTDIR + "/kernel_evol.jsonl", B5)
    run(tier6(3000, 6), OUTDIR + "/kernel_distractor.jsonl", B6)
    print("CURRICULUM_DONE", flush=True)
