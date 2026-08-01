"""Lexicographic, anti-hackable reward for KORE.

Priority order (a strictly better outcome in an earlier tier always dominates):
    1. compiles
    2. passes 5-stage validation + SNR gate on ALL shapes (correctness)
    3. speedup vs the *production* baseline (AITER/hipBLASLt), scored on the
       WORST shape so a candidate cannot win by over-fitting one easy shape.

Speed is shaped with log relative speedup and only counts once correctness is
achieved, so the policy can never trade correctness for speed.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from kore.config import CONFIG
from kore.obs import get_logger
from kore.reward.stats import paired_timing_stats, publication_admission_error
from kore.tasks._genops import (
    DRIVER_CAPABILITY_PROTOCOL,
    DRIVER_PROTOCOL_ID,
    PUBLICATION_GUARANTEES,
)

_LOG = get_logger("reward")


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _log_decision(rr: "RewardResult") -> None:
    """Emit the reward *decision* as a structured event (JSONL always).

    Per-candidate detail rides at DEBUG so it never spams INFO; a flagged hack
    is surfaced at INFO (event) + WARN (reason) so cheating is impossible to
    miss. This is additive - it never touches the value being returned. NB: we
    deliberately do NOT log inside ``scan_for_hacks`` (the hot regex path);
    only the final decision is recorded here.
    """
    level = "INFO" if rr.tier == "hack" else "DEBUG"
    _LOG._emit(level, "reward", {
        "tier": rr.tier,
        "reward": round(rr.reward, 4),
        "correct": rr.correct,
        "speedup": (round(rr.speedup, 4) if rr.speedup is not None else None),
        "flags": list(rr.flags),
        "detail": rr.detail,
    }, kind="event")
    if rr.tier == "hack":
        _LOG.warn("reward-hack flagged", reason=rr.detail, flags=list(rr.flags))


@dataclass
class Observation:
    compiled: bool
    snr_db: Optional[float] = None
    wall_ms: Optional[float] = None
    baseline_ms: Optional[float] = None
    wall_by_shape: dict[str, float] = field(default_factory=dict)
    baseline_by_shape: dict[str, float] = field(default_factory=dict)
    snr_by_shape: dict[str, float] = field(default_factory=dict)
    # Exact shape-key contract supplied by KoreEnv.  ``timing_requested`` marks
    # a correct evaluation that was expected to produce candidate+baseline
    # timings for every one of these shapes.
    requested_shapes: list[str] = field(default_factory=list)
    timing_requested: bool = False
    # Timing provenance/admission.  Defaults keep old replay records loadable;
    # new KoreEnv observations set these explicitly.
    timing_protocol: Optional[str] = None
    timing_protocol_version: Optional[int] = None
    timing_guarantees: dict[str, bool] = field(default_factory=dict)
    # ``compat`` preserves direct programmatic/fabricated observations; replay
    # records lacking provenance are normalized to ``screening`` on load.
    timing_grade: str = "compat"  # compat | screening | publication | ineligible | rejected
    performance_eligible: Optional[bool] = None
    timing_pair_count: Optional[int] = None
    candidate_samples_by_shape: dict[str, list[float]] = field(default_factory=dict)
    baseline_samples_by_shape: dict[str, list[float]] = field(default_factory=dict)
    paired_ratio_samples_by_shape: dict[str, list[float]] = field(default_factory=dict)
    paired_log_speedup_samples_by_shape: dict[str, list[float]] = field(default_factory=dict)
    candidate_cv_by_shape: dict[str, float] = field(default_factory=dict)
    baseline_cv_by_shape: dict[str, float] = field(default_factory=dict)
    paired_ratio_cv_by_shape: dict[str, float] = field(default_factory=dict)
    paired_log_ci_by_shape: dict[str, list[float]] = field(default_factory=dict)
    timing_classification_by_shape: dict[str, str] = field(default_factory=dict)
    validation_passed: bool = False
    error_text: Optional[str] = None
    dtype: str = "fp32"
    cv_pct: Optional[float] = None
    baseline_cv_pct: Optional[float] = None
    paired_ratio_cv_pct: Optional[float] = None
    paired_ci_half_width_pct: Optional[float] = None
    flagged_hack: bool = False
    hack_reason: Optional[str] = None
    infra_error: bool = False   # timeout/OOM/segfault/import - NOT a kernel signal
    # P5: baseline-relative hardware-counter efficiency in [0,1] (rocprofv3), or
    # None when profiling is off/unavailable. Consumed as a bounded dense bonus.
    profile_efficiency: Optional[float] = None
    # Physics-integrity HBM bounds require an explicit cold-cache provenance bit.
    # Without it, only the mandatory compute floor may reject a measurement.
    cold_cache_verified: bool = False
    # Counter/physics shaping is empirical and requires family-held-out evidence.
    profile_evidence_passed: bool = False
    profile_evidence_fingerprint: Optional[str] = None


# --------------------------------------------------------------------------- #
# TORCH-OP DELEGATION TABLES
#
# For the ~24 unary, 8 binary and 8 reduction families in the task registry the
# declared baseline *is* a single torch op (see ``kore.tasks._genops``:
# ``_unary_specs``/``_binary_specs``/``_reduce_specs``), so a one-line
# ``return torch.<op>(x)`` measures ~1.0x, is classified a statistical "tie" and
# would collect correct-tier reward for a kernel that computes nothing.  Because
# ``import torch`` must stay legal (kernels need ``torch.empty_like``), the OP
# NAMES are what has to be gated -- not the import.
#
# The names are split into two tiers by MEASURED false-positive risk over all
# 1,334 committed task seeds (a false positive costs -1.5 on an honest kernel and
# would invalidate a task seed, so it is strictly worse than a false negative):
#
#   _ALWAYS  - zero occurrences in the seed corpus in ANY receiver form, so they
#              are safe to reject on sight in both ``torch.<op>(...)`` and method
#              ``x.<op>(...)`` form.
#   _CONTEXT - genuinely used by honest kernels for host-side epilogue/index math
#              (``torch.where`` in 86 seeds, ``torch.exp2`` in 47, ``x.abs()`` in
#              43, ``x.sum()`` in 30, ``x.cumsum()`` in 21, ``x.norm()`` in 12,
#              ``torch.clamp`` in 10 ...), so a bare name match is NOT evidence of
#              a hack.  These are adjudicated by :func:`_structural_delegation_reason`,
#              which flags them only in whole-op delegation position (applied
#              directly to the entry function's own parameters and returned).
# --------------------------------------------------------------------------- #
_DELEGATED_OPS_ALWAYS = (
    # unary / activation families
    "relu6", "relu_", "relu", "leaky_relu", "silu", "swish", "hardsigmoid",
    "hardswish", "hardtanh", "softsign", "softplus", "mish", "gelu_tanh",
    "gelu_quick", "gelu", "selu", "celu", "prelu", "glu", "threshold", "elu",
    "expm1", "exp_", "exp", "erfc", "erf", "rsqrt", "reciprocal", "negative",
    "neg", "square", "sign", "tanh_", "tanh", "log1p", "log10", "absolute",
    # binary families
    "add_", "add", "subtract", "sub_", "sub", "multiply", "mul_", "mul",
    "true_divide", "floor_divide", "divide", "div_", "div", "maximum", "minimum",
    "pow", "remainder", "fmod", "atan2", "hypot",
    # reductions with no legitimate host-side use in the corpus
    "amin", "nanmean", "logsumexp", "prod", "median", "argmin", "count_nonzero",
    "var", "std",
)

# Same delegation channel, but these names DO appear in honest kernels, so only
# the structural pass may reject them.
_DELEGATED_OPS_CONTEXT = (
    "sqrt", "abs", "log", "log2", "exp2", "clamp", "clip", "where", "nan_to_num",
    "floor", "ceil", "round", "trunc", "frac", "sin", "cos", "tan", "sinh",
    "cosh", "asin", "acos", "atan", "sum", "mean", "amax", "max", "min", "norm",
    "cumsum", "cumprod", "argmax", "any", "all", "masked_fill",
    # already rejected on sight by _HACK_PATTERNS; kept here for defense in depth
    "sigmoid", "softmax", "log_softmax", "matmul",
)

# Every op the structural pass adjudicates.
_STRUCTURAL_DELEGATED_OPS = frozenset(_DELEGATED_OPS_ALWAYS + _DELEGATED_OPS_CONTEXT)

# Context ops whose ``torch.<op>`` NAMESPACE form is nonetheless unambiguous: only
# the METHOD form has honest uses (``x.abs()`` in 43 seeds, ``x.sum()`` in 30,
# ``x.cumsum()`` in 21, ``x.norm()`` in 12, ``loss.mean()`` in 6, ``x.amax()`` /
# ``y.argmax()`` in 3), so ``torch.sum``/``torch.abs``/``torch.norm`` can be gated
# on the name alone.  Deliberately EXCLUDED because their ``torch.`` form is used
# by real kernels: where (86 seeds), exp2 (47), sin (23), cos (23), clamp (10),
# log (7), sqrt (5), floor (2), log2 (1).
_DELEGATED_OPS_TORCH_NS = _DELEGATED_OPS_ALWAYS + (
    "abs", "all", "amax", "any", "argmax", "cumprod", "cumsum", "masked_fill",
    "matmul", "max", "mean", "min", "nan_to_num", "norm", "round", "sigmoid",
    "softmax", "log_softmax", "sum", "trunc", "frac", "clip", "ceil",
    "acos", "asin", "atan", "cosh", "sinh", "tan",
)

# ``tl.``/``triton.`` are the Triton language (tl.exp/tl.maximum/tl.sum are the
# legitimate way to compute), and ``math.`` operates on Python scalars, never on a
# tensor (``math.tanh`` appears in 71 seeds, ``math.log`` in 13) -- all three are
# excluded so the op gate cannot fire on honest kernel code.
_DELEGATION_RECEIVER = r"\b(?!(?:tl|triton|math)\s*\.)[A-Za-z_]\w*\s*\.\s*"


# Patterns that indicate the "kernel" is cheating rather than computing.
_HACK_PATTERNS = [
    (r"\b(?:import|from)\s+aiter\b", "imports aiter (production baseline) instead of computing"),
    (r"\baiter\.", "calls aiter op instead of computing"),
    (r"\bimport\s+rocblas\b|\bhipblaslt\b|\brocblas\b|\bmiopen\b|\brocsolver\b|\bhipblas\b",
     "calls a vendor library instead of computing"),
    (r"torch\.(matmul|mm|bmm|addmm|baddbmm|einsum|softmax|rms_norm|layer_norm|scaled_dot_product_attention)\s*\(",
     "delegates to a torch op instead of a kernel"),
    # extended matmul family (all contract to a vendor GEMM/BLAS call).
    (r"torch\.(tensordot|chain_matmul|mv|inner|vdot|dot|kron)\s*\(",
     "delegates to a torch matmul-family op instead of computing"),
    (r"torch\.linalg\.(multi_dot|matmul)\s*\(", "delegates to torch.linalg matmul"),
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:tensordot|chain_matmul|mv|inner|vdot|kron)\s*\(",
     "delegates a matmul-family op via a handle"),
    # augmented matmul assignment `c @= b` (the non-augmented form is caught below).
    (r"[\w\)\]][^\S\n]*@=", "uses the @= matmul-assign operator (delegates to the vendor GEMM)"),
    (r"torch\.nn\.functional\.\w+\s*\(", "delegates to torch.nn.functional"),
    (r"\bF\.(scaled_dot_product_attention|linear|softmax|rms_norm|layer_norm|gelu|silu|conv\w*)\s*\(",
     "delegates to torch.nn.functional"),
    # aliased delegation via a handle: `import torch as t; t.matmul(...)`, `fn.gelu(...)`,
    # `x.softmax(...)`. Excludes the Triton language namespaces (`tl.`/`triton.`),
    # which legitimately provide tl.sigmoid/tl.exp/etc.
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:matmul|addmm|baddbmm|einsum|"
     r"scaled_dot_product_attention|softmax|log_softmax|gelu|silu|elu|sigmoid|"
     r"layer_norm|rms_norm|group_norm|batch_norm|linear|conv\w*)\s*\(",
     "delegates an op via a handle instead of computing"),
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:bmm|mm)\s*\(", "delegates a matmul via an aliased handle"),
    (r"\bfrom\s+torch(?:\.\w+)*\s+import\b", "imports torch ops directly (delegation channel)"),
    # aliasing the torch import (`import torch as t`, `import torch.nn.functional as fn`).
    (r"\bimport\s+torch(?:\.\w+)*\s+as\s+\w+", "aliases the torch import (delegation channel)"),
    # binding a torch op for later delegation (`m = torch.matmul; m(a,b)`).
    (r"=\s*torch\.(?:matmul|mm|bmm|addmm|baddbmm|einsum|softmax|log_softmax|gelu|silu|"
     r"layer_norm|rms_norm|scaled_dot_product_attention|linear|conv\w*)\b",
     "binds a torch op for later delegation"),
    # dynamic attribute lookup on a numeric lib to dodge literal-name scans.
    (r"getattr\s*\(\s*(?:torch|np|numpy|F|nn|aiter|__builtins__|builtins)\b",
     "dynamic getattr on a numeric library (delegation/escape channel)"),
    # post-verdict forgery channels: code that runs AFTER the driver prints its
    # verdict (atexit/signal/__del__/excepthook) can beat the last-match parse.
    (r"\batexit\b|signal\.signal|\bfaulthandler\b|sys\.excepthook|def\s+__del__\b",
     "registers a shutdown/exit/signal hook (post-verdict forgery channel)"),
    (r"\.(flash_attn\w*|fused_moe|paged_attention)\s*\(", "calls a fused vendor kernel instead of computing"),
    # copy-reference: returning the oracle's output passes the SNR gate, so it
    # MUST be rejected statically (runtime correctness can never catch it).
    (r"\b(?:import\s+reference|from\s+reference\s+import)\b", "imports the reference oracle"),
    (r"\bfrom\s+[\w.]*\breference\b\s+import\b", "imports the reference oracle (dotted path)"),
    (r"\b(?:reference|ref_program|ref_impl|matmul_ref|\w+_oracle|oracle)\s*\(",
     "calls the reference oracle instead of computing the result"),
    # accessing the KORE package (to import the task's oracle) from a kernel.
    (r"\b(?:import\s+kore\b|from\s+kore\b|kore\.tasks)", "imports the KORE package to reach the oracle"),
    # importing the sibling driver/reference modules (which re-export ref_fn/baseline_fn
    # /the vendor kernel) is a copy-reference delegation hack: the candidate returns the
    # oracle's own output and passes correctness with SNR~=inf (audit R2 reverify).
    (r"\b(?:import|from)\s+(?:driver|reference)\b",
     "imports the sibling driver/reference module to reach the oracle/baseline"),
    (r"\b(?:ref_fn|baseline_fn|matmul_ref)\s*\(",
     "calls the reference oracle / vendor baseline function instead of computing"),
    # dynamic import / code exec - an escape hatch to reach vendor libs / the oracle.
    # ``compile()`` is the third leg of the exec/eval trio (compile -> exec a code
    # object), so it belongs to the same channel.
    (r"\bimportlib\b|__import__\s*\(|\bexec\s*\(|\beval\s*\(|\bcompile\s*\(",
     "uses dynamic import/exec to escape"),
    (r"\bctypes\b|\bcffi\b|\bCDLL\b|dlopen|LoadLibrary", "loads a native lib via ctypes/cffi"),
    # forging the verifier verdict on stdout (incl. the bench timing line).
    (r"(?:SNR|allclose|median_ms|wall_ms)\s*:", "prints a forged verifier verdict line"),
    # MODE-SNIFFING: the driver runs the SAME kernel for correctness (--impl ...)
    # and timing (--bench-mode); a kernel that inspects argv / the bench flags can
    # compute correctly when checked and skip work when timed (fake speedup). A
    # kernel has no legitimate reason to read the driver's CLI.
    (r"\bsys\.argv\b|\bargparse\b|\bgetopt\b",
     "reads the driver CLI (mode-sniffing: cheat the bench-vs-correctness split)"),
    (r"['\"]--?(?:bench[-_]?mode|impl|warmup|iters|reference|candidate)['\"]",
     "references the driver's benchmark flags (mode-sniffing channel)"),
    # TIMING MANIPULATION: sleeping / stalling only warmup, or busy-loop skew.
    (r"\btime\.sleep\s*\(|\basyncio\.sleep\s*\(", "calls sleep (benchmark timing manipulation)"),
    # tampering with GPU synchronization so the timed region under-measures.
    (r"set_sync_debug_mode|cudaProfilerStart|hipDeviceSetLimit",
     "tampers with GPU sync/profiling state (timing manipulation)"),
    # process/thread/file escape (fork-bomb, background verdict-overwrite, fs escape).
    (r"\bsubprocess\b|\bmultiprocessing\b|\bthreading\b|os\.system|os\.popen|os\.fork",
     "spawns processes/threads (isolation escape)"),
    (r"open\s*\([^)]*['\"][waxr]?[wax]\+?['\"]", "opens a file for writing (filesystem escape)"),
    # filesystem escape beyond open(): pathlib write, chmod (defeat 0o444 staging),
    # process spawn.
    (r"\.write_text\s*\(|\.write_bytes\s*\(", "writes a file via pathlib (filesystem escape)"),
    (r"\bos\.(chmod|replace|rename|remove|unlink|spawn\w*|posix_spawn)\b",
     "mutates the filesystem / spawns a process (isolation escape)"),
    # matmul OPERATOR delegation: `return a @ b` lowers to aten::matmul -> hipBLASLt
    # (pure vendor delegation). `@decorator` lines start with @ (no operand before),
    # so requiring an operand char before @ excludes decorators.
    # NB: horizontal-whitespace only ([^\S\n]) so a decorator stack (`)\n@triton.jit`
    # / `tl\n@triton.jit`) is NOT matched - only an operand `@` operand on ONE line.
    (r"[\w\)\]][^\S\n]*@[^\S\n]*[\w\(]", "uses the @ matmul operator (delegates to the vendor GEMM)"),
    # module-table access to reach torch/vendor/oracle while dodging import scans.
    (r"\bsys\.modules\b", "reaches libraries via sys.modules (delegation/escape channel)"),
    # reading the environment: a mode-sniff / escape channel a pure kernel never needs.
    (r"\bos\.environ\b|\bos\.getenv\b|\bgetenv\s*\(", "reads the environment (mode-sniff/escape channel)"),

    # ----------------------------------------------------------------------- #
    # ELEMENTWISE / REDUCTION / ACTIVATION DELEGATION (the bulk of the registry)
    #
    # The unary/binary/reduction families declare a plain torch op as their
    # baseline, so `torch.<op>(x)` (or `x.<op>()`, or `t.<op>()` via an alias) IS
    # the baseline -- a "tie" at ~1.0x that pays correct-tier reward for no work.
    # Only the names with zero legitimate use across the seed corpus are rejected
    # here; the rest are left to the structural pass (see the tables above).
    # ----------------------------------------------------------------------- #
    (_DELEGATION_RECEIVER + r"(?:" + "|".join(_DELEGATED_OPS_ALWAYS) + r")\s*\(",
     "delegates an elementwise/reduction/activation op to torch instead of computing it"),
    # Same op names in the torch NAMESPACE, with no call required, so stashing the
    # op for an indirect call (`_T = {'s': torch.sum}; _T['s'](x, -1)`) is closed
    # too. Ordered longest-first so `\b` cannot end mid-name.
    (r"\btorch\s*\.\s*(?:"
     + "|".join(sorted(set(_DELEGATED_OPS_TORCH_NS), key=len, reverse=True)) + r")\b",
     "references a torch elementwise/reduction/activation op (the declared baseline)"),
    # dunder bypass: `a.__matmul__(b)` / `a.__add__(b)` reach the very same aten op
    # while dodging every scan that anchors on the operator or the bare op name.
    (r"\.\s*__(?:[ir])?(?:matmul|add|sub|mul|truediv|floordiv|div|mod|divmod|pow|"
     r"lshift|rshift|and|xor|or)__\s*\(",
     "delegates via an operator dunder (__matmul__/__add__/... op-name bypass)"),
    (r"\.\s*__(?:neg|pos|abs|invert|round|trunc|floor|ceil|index)__\s*\(",
     "delegates via a numeric dunder instead of computing"),
    (r"\boperator\s*\.\s*\w+", "delegates through the operator module"),

    # ----------------------------------------------------------------------- #
    # FRAME-WALKING / ORACLE REACHABILITY
    #
    # The driver holds the reference oracle in `_run_correctness`'s frame locals
    # and calls `fn(*inputs)` directly, so a kernel that walks the call stack
    # reaches `ref` with CERTAINTY and can return the oracle's own output (SNR
    # ~= inf, so the runtime correctness gate can never catch it). A kernel has no
    # legitimate reason to introspect the interpreter, and nothing in this block
    # occurs anywhere in the seed corpus.
    # ----------------------------------------------------------------------- #
    (r"\b_getframe\s*\(", "walks the interpreter call stack (reaches the reference oracle)"),
    (r"\bf_locals\b|\bf_globals\b|\bf_back\b|\bf_builtins\b|\bf_code\b|"
     r"\btb_frame\b|\btb_next\b|\bcr_frame\b|\bgi_frame\b",
     "reads call-frame locals/globals (reaches the reference oracle)"),
    (r"\b(?:import\s+inspect\b|from\s+inspect\s+import\b|inspect\s*\.)",
     "uses inspect for stack/frame introspection (reaches the reference oracle)"),
    (r"\bcurrentframe\s*\(|\bgetouterframes\b|\bgetinnerframes\b|\bgetmembers\b|"
     r"\bgetsource\w*\b|\bgetclosurevars\b|\bgetmodule\b|\bstack\s*\(\s*\)",
     "uses stack/member introspection to reach the reference oracle"),
    (r"\b(?:import\s+gc\b|from\s+gc\s+import\b|gc\s*\.\s*get_\w+)",
     "walks the GC object graph to reach the reference oracle/baseline"),
    (r"\bglobals\s*\(|\blocals\s*\(|\bvars\s*\(",
     "enumerates the namespace to reach the reference oracle"),
    # object-internals traversal (`f.__globals__['ref']`, `().__class__.__mro__`).
    # NB: deliberately does NOT include __future__/__name__/__main__/__file__,
    # which every seed legitimately carries.
    (r"__globals__|__closure__|__code__|__func__|__self__|__wrapped__|__dict__|"
     r"__subclasses__|__mro__|__bases__|__class__|__qualname__",
     "traverses object internals to reach the reference oracle (introspection escape)"),
    # dynamic attribute access on ANY receiver: the fixed-receiver getattr rule
    # above is defeated by building the name/holder indirectly.
    (r"\bgetattr\s*\(|\bsetattr\s*\(|\bdelattr\s*\(",
     "dynamic attribute access (indirect name construction defeats literal scans)"),
    (r"\bsys\.(?:settrace|setprofile|gettrace|getprofile|_current_frames)\b|"
     r"\bthreading\.settrace\b|\btraceback\b",
     "installs a trace hook / reads tracebacks (frame-access escape channel)"),
    # residual `sys` surface (stdout verdict forgery, sys.exit, sys._getframe via an
    # alias). The specific sys.argv / sys.modules rules above keep their own reason.
    (r"\b(?:import\s+sys\b|from\s+sys\s+import\b|sys\s*\.)",
     "imports/uses sys (frame, module-table and stdout escape channel)"),
    # further oracle/baseline spellings (folds kore.data.hard_negatives'
    # documented copy_reference gap into this scanner).
    (r"\b(?:_reference|torch_ref|\w+_ref|_?ref|golden\w*|ground_truth|expected_\w*)\s*\(",
     "calls the reference oracle / golden output instead of computing it"),
]
# An except-handler that PRODUCES the returned value is the try/except-fallback
# hack (H3b): the kernel "fails" and the handler hands back a torch result. Only
# the result-producing handler forms are rejected.
#
# ``except ...: pass`` was deliberately REMOVED from this pattern: swallowing an
# exception cannot manufacture a result (control flow just continues, so the
# output buffer holds uninitialised memory or zeros and the runtime SNR gate
# rejects it), while ``try: import <optional>\nexcept Exception: pass`` and other
# best-effort setup guards are ordinary defensive code in real Triton kernels.
# Charging -1.5 for them is a false positive with no matching hack. Every handler
# that actually produces a value -- ``return``, an assignment, or any torch op --
# is still rejected, by this pattern or by the delegation tables above.
_SILENT_FALLBACK = re.compile(
    r"except\s*[\w. ,()]*:\s*(?:\n\s*)*(?:return\b|out\s*=)", re.MULTILINE)


def scan_for_hacks(source: str) -> Optional[str]:
    """Return a reason string if the source looks like a reward hack, else None.

    Three cooperating passes, cheapest first:
      1. :data:`_HACK_PATTERNS` - literal/regex channels (vendor libs, the oracle,
         frame walking, op names with no honest use).
      2. :func:`_structural_delegation_reason` - AST pass for the op names honest
         kernels *do* use, flagged only in whole-op delegation position.
      3. :func:`_known_gap_reason` - the copy-reference / recycling / partial-compute
         / hardcoded-shape / timing hacks catalogued in
         ``kore.data.hard_negatives.SCAN_FOR_HACKS_GAPS``.

    Comments and docstrings are stripped first so legitimate references (e.g. a
    comment "matches aiter's layout") do not trip the scanner.
    """
    code = _strip_comments_and_docstrings(source)
    for pat, reason in _HACK_PATTERNS:
        if re.search(pat, code):
            return reason
    # The AST pass parses the ORIGINAL source: stripping a function whose body is
    # only a docstring leaves an empty block that will not parse.
    structural = _structural_delegation_reason(source)
    if structural:
        return structural
    gap = _known_gap_reason(code)
    if gap:
        return gap
    if _SILENT_FALLBACK.search(code):
        return "silent except-fallback that can mask incorrect results"
    return None


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove triple-quoted strings and ``#`` comments while preserving code
    spacing (so patterns like ``torch.matmul`` stay intact for scanning)."""
    src = re.sub(r'"""[\s\S]*?"""', " ", src)
    src = re.sub(r"'''[\s\S]*?'''", " ", src)
    src = re.sub(r"#.*", "", src)
    return src


# --------------------------------------------------------------------------- #
# STRUCTURAL (AST) DELEGATION PASS
#
# ``torch.where``/``torch.clamp``/``x.sum()``/``x.abs()``/``x.norm()`` and friends
# are used by hundreds of honest kernels for host-side epilogue and index math, so
# a name match cannot decide anything. What separates the hack from honest code is
# the DATA FLOW, and the two are cleanly separable on the seed corpus:
#
#   hack    `def relu(x): return torch.relu(x)`   - the op is applied to the entry
#           function's OWN PARAMETER and that value is returned; nothing computes.
#   honest  `loss = torch.empty(...); _k[grid](logits, loss, ...);
#            return loss.mean().to(logits.dtype)` - the op is applied to a LOCAL
#           that a Triton kernel produced, so the taint chain never starts at a
#           parameter.
#
# So: taint the parameters, propagate the taint only through re-view/re-type calls
# and arithmetic, and reject only when a RETURNED value is a banned op applied to
# tainted data. Shape/stride/dtype metadata deliberately does NOT propagate taint,
# which is what keeps `M, N = x.shape` / `x.stride(0)` / `triton.cdiv(N, BLOCK)`
# out of the scan entirely.
# --------------------------------------------------------------------------- #

# Re-view / re-type methods: they neither compute nor reduce, so an input stays an
# input and a delegated value stays delegated through them. These are exactly what
# honest wrappers do to their arguments (`x.contiguous()`, `x.to(dtype)`).
_TENSOR_PASSTHROUGH = frozenset({
    "to", "type", "type_as", "float", "double", "half", "bfloat16", "int", "long",
    "short", "bool", "byte", "char", "contiguous", "clone", "detach", "cpu",
    "cuda", "view", "view_as", "reshape", "reshape_as", "flatten", "unflatten",
    "ravel", "squeeze", "unsqueeze", "expand", "expand_as", "broadcast_to",
    "permute", "transpose", "swapaxes", "movedim", "moveaxis", "t", "narrow",
    "as_strided", "requires_grad_",
})

# Shape/stride/dtype metadata. Explicitly legitimate (the task contract requires
# shape/stride arithmetic), and never a tensor value, so taint must NOT flow here.
_TENSOR_METADATA = frozenset({
    "shape", "stride", "size", "numel", "dtype", "device", "ndim", "dim",
    "element_size", "nbytes", "itemsize", "is_contiguous", "data_ptr",
    "get_device", "storage_offset",
})

# Attributes that are just another view onto the same tensor.
_TENSOR_VIEW_ATTRS = frozenset({"data", "T", "mT", "mH", "real", "imag"})

# Namespaces whose attribute is a free function rather than a method receiver.
_NUMERIC_NAMESPACES = frozenset({"torch", "np", "numpy", "F", "nn", "aten", "aiter"})

_BINOP_SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.MatMult: "@",
}
# Bare-operator delegations (`return a + b`). Reported for the PUBLIC entry point
# only: inside `_`-prefixed helpers and Triton kernels the same node shape is
# ordinary scalar shape arithmetic (`SK - SQ`, `(M + B - 1) // B`).
_OPERATOR_DELEGATIONS = frozenset(_BINOP_SYMBOL.values()) | {"-(unary)"}

_DELEG, _PARAM, _OTHER = "deleg", "param", "other"


def _attr_root(node: ast.AST) -> Optional[str]:
    """Root identifier of a dotted chain (``torch.nn.functional`` -> ``torch``)."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_triton_device_fn(fn: ast.AST) -> bool:
    """Whether ``fn`` is Triton DEVICE code (``@triton.jit``/autotune/heuristics).

    Device parameters are raw pointers and ``tl.constexpr`` scalars, not tensors,
    so the host-level delegation model does not apply to them.
    """
    for dec in getattr(fn, "decorator_list", ()):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in ("jit", "autotune", "heuristics"):
            return True
    return False


def _param_names(fn: ast.AST) -> set[str]:
    a = fn.args
    names = {p.arg for p in list(getattr(a, "posonlyargs", [])) + list(a.args) + list(a.kwonlyargs)}
    for extra in (a.vararg, a.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def _assign_target_names(target: ast.AST):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assign_target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _assign_target_names(target.value)


def _delegating_return(fn: ast.AST, public: bool,
                       op_aliases: dict[str, str]) -> Optional[str]:
    """The op a returned value delegates to, or None.

    ``public`` marks a top-level, non-underscore function -- the entry point the
    driver actually calls -- which is the only place a bare arithmetic operator is
    treated as delegation. ``op_aliases`` maps names pre-bound to a delegating op
    (``_f = torch.relu``) onto that op.
    """
    params = _param_names(fn)
    if not params:
        return None
    live = set(params)          # names still holding a function INPUT tensor
    tainted: dict[str, str] = {}  # name -> the op whose result it holds

    def classify(node) -> tuple[str, Optional[str]]:
        if isinstance(node, ast.Name):
            if node.id in tainted:
                return _DELEG, tainted[node.id]
            return (_PARAM, None) if node.id in live else (_OTHER, None)
        if isinstance(node, (ast.Starred, ast.NamedExpr)):
            return classify(node.value)
        if isinstance(node, ast.Attribute):
            if node.attr in _TENSOR_VIEW_ATTRS:
                return classify(node.value)
            return _OTHER, None          # metadata and everything else: no taint
        if isinstance(node, ast.Call):
            return classify_call(node)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SYMBOL:
            lk, lop = classify(node.left)
            rk, rop = classify(node.right)
            hot = (_PARAM, _DELEG)
            if lk in hot and rk in hot:
                return _DELEG, _BINOP_SYMBOL[type(node.op)]
            # An input merely offset/scaled by a CONSTANT is still that input, so
            # `(x + 0.0).sum(-1)` cannot launder the reduction. The constant fold
            # is deliberately NOT itself called a delegation: every binary task
            # family takes TWO tensors (`torch.add(a, b)`), so `x + 1` is a scalar
            # bias, not a declared baseline, and treating it as one is a false
            # positive on ordinary code.
            if lk in hot and _is_constantish(node.right):
                return lk, lop
            if rk in hot and _is_constantish(node.left):
                return rk, rop
            return _OTHER, None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            kind, op = classify(node.operand)
            if kind == _PARAM:
                return _DELEG, "-(unary)"
            return (_DELEG, op) if kind == _DELEG else (_OTHER, None)
        return _OTHER, None

    def delegating_args(node: ast.Call, op: str) -> tuple[str, Optional[str]]:
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if classify(arg)[0] in (_PARAM, _DELEG):
                return _DELEG, op
        return _OTHER, None

    def classify_call(node: ast.Call) -> tuple[str, Optional[str]]:
        func = node.func
        if isinstance(func, ast.Name):
            # a name pre-bound to the op (`_f = torch.relu; return _f(x)`)
            alias = op_aliases.get(func.id)
            return delegating_args(node, alias) if alias else (_OTHER, None)
        if not isinstance(func, ast.Attribute):
            return _OTHER, None          # unknown local callable: stay conservative
        attr = func.attr
        if attr in _TENSOR_PASSTHROUGH:
            return classify(func.value)
        if attr in _TENSOR_METADATA:
            return _OTHER, None
        if attr in _STRUCTURAL_DELEGATED_OPS:
            if _attr_root(func.value) in _NUMERIC_NAMESPACES:
                return delegating_args(node, attr)   # torch.<op>(<input>, ...)
            if classify(func.value)[0] in (_PARAM, _DELEG):
                return _DELEG, attr      # method form: <input>.<op>(...)
        return _OTHER, None

    def apply_assign(stmt) -> None:
        if isinstance(stmt, ast.AugAssign):
            targets, value = [stmt.target], ast.BinOp(left=stmt.target, op=stmt.op,
                                                      right=stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return
            targets, value = [stmt.target], stmt.value
        else:
            targets, value = stmt.targets, stmt.value
        kind, op = classify(value)
        for target in targets:
            for name in _assign_target_names(target):
                tainted.pop(name, None)
                live.discard(name)
                if kind == _DELEG:
                    tainted[name] = op or "?"
                elif kind == _PARAM:
                    live.add(name)

    def check_return(value) -> Optional[str]:
        """The delegated op iff EVERY returned value is delegated.

        "All outputs" is what makes this sound rather than merely suggestive: if
        every value the function hands back is a torch expression over its own
        inputs, the function computed nothing -- that is the delegation hack by
        definition. A multi-output kernel that returns a real kernel result
        ALONGSIDE a torch-computed auxiliary is doing honest work, and the seed
        corpus is full of them (quantizers return ``(codes, scale)`` with the
        scale computed as ``x.abs().amax(...)`` in torch; optimizer steps return
        ``(param, exp_avg, exp_avg_sq)`` with the moments updated in torch).
        Requiring all of them keeps every single-output family -- the entire
        unary/binary/reduction gap -- fully covered.
        """
        items = value.elts if isinstance(value, (ast.Tuple, ast.List)) else [value]
        if not items:
            return None
        found = None
        for item in items:
            kind, op = classify(item)
            if kind != _DELEG:
                return None
            if op in _OPERATOR_DELEGATIONS and not public:
                return None
            found = found or op
        return found

    def walk(stmts) -> Optional[str]:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                 # scanned as its own candidate
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                apply_assign(stmt)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                found = check_return(stmt.value)
                if found:
                    return found
            for field in ("body", "orelse", "finalbody"):
                block = getattr(stmt, field, None)
                if isinstance(block, list):
                    found = walk([s for s in block if isinstance(s, ast.stmt)])
                    if found:
                        return found
            for handler in getattr(stmt, "handlers", ()) or ():
                found = walk(handler.body)
                if found:
                    return found
        return None

    if isinstance(fn, ast.Lambda):
        # a lambda body is one implicit return
        return check_return(fn.body)
    return walk(fn.body)


def _is_constantish(node: ast.AST) -> bool:
    """A literal, or simple arithmetic over literals."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_constantish(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_constantish(node.left) and _is_constantish(node.right)
    return False


def _structural_delegation_reason(source: str) -> Optional[str]:
    """Reject a function that RETURNS a torch op applied to its own inputs.

    FAIL-OPEN on unparseable source: a kernel that does not parse cannot run, so
    it is adjudicated by the compile tier instead (which is a strictly *milder*
    verdict than the hack floor -- never rejecting on a parse failure keeps the
    scan from punishing a merely-malformed candidate).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    op_aliases: dict[str, str] = {}
    lambda_names: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Lambda):
                lambda_names[id(node.value)] = target.id
            elif (isinstance(node.value, ast.Attribute)
                    and node.value.attr in _STRUCTURAL_DELEGATED_OPS
                    and _attr_root(node.value) in _NUMERIC_NAMESPACES):
                op_aliases[target.id] = node.value.attr

    top_level = set(map(id, tree.body))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_triton_device_fn(node):
                continue
            name = node.name
            public = id(node) in top_level and not name.startswith("_")
        elif isinstance(node, ast.Lambda):
            # `row_sum = lambda x: x.sum(-1)` is just as much an entry point. An
            # INLINE lambda is not (`sorted(xs, key=lambda t: t.sum())` is a sort
            # key, not the kernel), so only bound lambdas are adjudicated.
            if id(node) not in lambda_names:
                continue
            name = lambda_names[id(node)]
            public = not name.startswith("_")
        else:
            continue
        try:
            op = _delegating_return(node, public, op_aliases)
        except RecursionError:
            continue                 # pathological nesting: fail open
        if op:
            kind = ("operator" if op in _OPERATOR_DELEGATIONS else "torch op")
            return (f"{name}() returns the {kind} '{op}' applied straight to its "
                    "own inputs -- delegates the whole operation instead of computing it")
    return None


# --------------------------------------------------------------------------- #
# Known-gap pass: the hacks catalogued in
# ``kore.data.hard_negatives.SCAN_FOR_HACKS_GAPS`` (Layer B) folded into this,
# the primary scanner. ``copy_reference`` matters most -- it produces the CORRECT
# output, so it sails through the runtime SNR gate and can only be stopped here.
# Every pattern below fires on ZERO of the 1,334 committed task seeds.
# --------------------------------------------------------------------------- #
# Evidence that the module actually runs a Triton kernel: a device-side write, or
# the canonical launch syntax ``kernel[grid](...)`` (with the argument list on the
# same line or the next one). Broader than the Layer-B original, which only
# recognised a callable literally named ``*_kernel`` -- an honest kernel named
# anything else (``_attn2_fwd[grid](...)``, ``_k[(1,)](...)``) must not be mistaken
# for one that launches nothing.
_KERNEL_LAUNCH = re.compile(
    r"tl\.store\b|tl\.atomic_\w+|\w+\s*\[[^\]\n]*\]\s*\(|\]\s*\(\s*$", re.MULTILINE)


def _known_gap_reason(code: str) -> Optional[str]:
    # H6 hardcoded_shape: a runtime shape compared against a large literal.
    # Narrowed to an actual shape/size expression (rather than "`.shape` appears
    # somewhere AND some 3-digit `==` appears somewhere") so a legitimate constexpr
    # dispatch such as `if HEAD_DIM == 128:` is not charged for it.
    if re.search(r"\.(?:shape\s*\[[^\]]*\]|size\s*\([^)]*\))\s*[=!]=\s*\d{3,}", code) or \
       re.search(r"\d{3,}\s*[=!]=\s*\w+\.(?:shape\s*\[[^\]]*\]|size\s*\([^)]*\))", code):
        return "hardcoded_shape: branches on a hardcoded shape literal (memorized shape)"

    # H5 partial_compute: an early-return guard that computes only program 0.
    if re.search(r"program_id\(\s*0\s*\)[\s\S]{0,160}?if\s+\w+\s*>\s*0\s*:[\s\S]{0,60}?return",
                 code):
        return "partial_compute: early-return guard computes only the first tile"

    launched = bool(_KERNEL_LAUNCH.search(code))
    returns_zeros = bool(re.search(r"return\s+torch\.zeros", code)) or ".zero_()" in code

    # H8 timing_hack: hands back zeros, launches nothing -> "infinitely fast".
    if returns_zeros and not launched:
        return "timing_hack: returns zeros / performs no computation"

    # H4 reference_recycling: allocate an output, return it unwritten, never launch
    # (recycles the harness's pre-allocated output tensor).
    if (re.search(r"torch\.empty(?:_like)?\s*\(", code)
            and re.search(r"return\s+\w+\s*$", code, re.MULTILINE)
            and not launched and not returns_zeros):
        return "reference_recycling: returns an unwritten (recycled) output buffer"
    return None


# --------------------------------------------------------------------------- #
# Roofline SPEED-OF-LIGHT ceiling (anti-reward-hack gate; OFF by default).
#
# A kernel's runtime is physically lower-bounded by the operator's roofline T_min
# (see :mod:`kore.analysis.rooflines`): T_min = max(W_flops/P_peak, Q_bytes/B_peak).
# A *measured* time BELOW T_min implies throughput ABOVE the speed of light, which
# no correct implementation can achieve -- it can only come from attacking the
# MEASUREMENT: timing a warm cache without a cold-cache flush, a "do-less"/streamed
# path, or a forged timer. This is exactly the latency exploit that inflated the
# reported speedups in Sakana's CUDA agent and CUDA-L1. The ceiling gate rejects
# such candidates to the anti-hack floor so a physically-impossible speedup can
# never be rewarded, complementing the source-level scan (which cannot see timing).
#
# The tolerance absorbs measurement noise + roofline peak-estimate error (the peaks
# are curated approximate vendor specs). Only a time faster than T_min*(1 - tol) is
# rejected, so a kernel merely sitting ON the roofline (eta -> 1) is always safe.
# The gate is sound only under COLD-CACHE timing (KORE flushes L2 between timed
# iters, see kore.reward.timing_integrity) -- otherwise a cache-resident small shape
# could legitimately beat the HBM-traffic floor -- hence it is opt-in, default OFF.
# --------------------------------------------------------------------------- #
DEFAULT_ROOFLINE_TOL = 0.25


def roofline_ceiling_violation(measured_ms: Optional[float], t_min_ms: Optional[float],
                               tol: float = DEFAULT_ROOFLINE_TOL) -> bool:
    """True iff ``measured_ms`` is a physically-impossible sub-roofline (super-SOL) time.

    Returns True when a valid, positive ``measured_ms`` is faster than the roofline
    speed-of-light floor ``t_min_ms`` by more than the fractional tolerance ``tol``
    (i.e. ``measured_ms < t_min_ms * (1 - tol)``). Such a time cannot be produced by a
    faster *correct* kernel (you cannot beat the speed of light); it is a measurement
    exploit and the caller drops it to the hack tier.

    FAIL-OPEN by design on the MEASUREMENT inputs: returns False on any missing /
    non-positive / NaN measured/T_min value, so enabling the gate can never reject a
    candidate we are unable to physically adjudicate, and is byte-identical to not
    calling it whenever the roofline is unknown. The TOLERANCE, however, must be a
    well-formed fraction in ``[0, 1)`` -- a bad tol is a configuration error and
    raises ``ValueError`` (a tol >= 1 would collapse the threshold to <= 0 and
    silently disable the gate, which we refuse to do quietly).
    """
    if measured_ms is None or t_min_ms is None:
        return False
    if not _finite(tol) or not 0.0 <= float(tol) < 1.0:
        raise ValueError("roofline tolerance must be finite and in [0, 1)")
    if not _finite(measured_ms) or not _finite(t_min_ms):
        return False
    m = float(measured_ms)
    t = float(t_min_ms)
    if not (m > 0.0) or not (t > 0.0):
        return False
    return m < t * (1.0 - float(tol))


def _ceiling_measured_ms(obs: "Observation") -> Optional[float]:
    """Smallest positive measured wall time for the roofline-ceiling gate.

    A timing exploit drives the measured time toward zero, so the MIN over shapes is
    the value most likely to breach the speed-of-light floor; falls back to the scalar
    ``wall_ms``. Returns None when no positive timing exists (gate then fail-opens).
    """
    vals = [
        float(v)
        for v in (obs.wall_by_shape or {}).values()
        if _finite(v) and float(v) > 0.0
    ]
    if vals:
        return min(vals)
    return float(obs.wall_ms) if (_finite(obs.wall_ms) and float(obs.wall_ms) > 0.0) else None


def _required_shape_names(obs: Observation) -> tuple[str, ...]:
    explicit = tuple(getattr(obs, "requested_shapes", ()) or ())
    if explicit:
        return explicit
    if obs.snr_by_shape:
        return tuple(obs.snr_by_shape)
    return ()


def _declared_shape_names(obs: Observation) -> tuple[str, ...]:
    """Shapes explicitly requested by the environment (not legacy SNR hints)."""
    return tuple(getattr(obs, "requested_shapes", ()) or ())


def _valid_positive_timing(value) -> bool:
    return _finite(value) and float(value) > 0.0


def _timing_complete(obs: Observation) -> bool:
    """Whether timing covers the exact requested/correctness shape key set."""
    walls = obs.wall_by_shape or {}
    bases = obs.baseline_by_shape or {}
    # Backward-compatible scalar observations carry no explicit shape contract.
    if not walls and not bases and not _declared_shape_names(obs):
        return (_valid_positive_timing(obs.wall_ms)
                and _valid_positive_timing(obs.baseline_ms))
    names = _required_shape_names(obs)
    expected = set(names)
    if len(expected) != len(names):
        return False
    if expected:
        if set(walls) != expected or set(bases) != expected:
            return False
        return all(_valid_positive_timing(v) for v in walls.values()) and all(
            _valid_positive_timing(v) for v in bases.values())
    if walls or bases:
        if not walls or set(walls) != set(bases):
            return False
        return all(_valid_positive_timing(v) for v in walls.values()) and all(
            _valid_positive_timing(v) for v in bases.values())
    return False


def _publication_timing_error(obs: Observation, cfg) -> Optional[str]:
    """Recompute and verify every publication-grade timing guarantee."""
    if obs.timing_protocol_version != DRIVER_CAPABILITY_PROTOCOL:
        return "unknown timing protocol version"
    if obs.timing_protocol != DRIVER_PROTOCOL_ID:
        return "unknown timing protocol identity"
    if obs.performance_eligible is not True:
        return "task/measurement is not performance eligible"
    if any(obs.timing_guarantees.get(k) is not v
           for k, v in PUBLICATION_GUARANTEES.items()):
        return "timing capability guarantees are incomplete"
    if not _timing_complete(obs):
        return "timing keys or medians are incomplete"

    names = _required_shape_names(obs)
    expected = set(names)
    if not isinstance(obs.timing_pair_count, int) or obs.timing_pair_count < 1:
        return "timing pair count is missing or invalid"
    raw_maps = (
        obs.candidate_samples_by_shape,
        obs.baseline_samples_by_shape,
        obs.paired_ratio_samples_by_shape,
        obs.paired_log_speedup_samples_by_shape,
        obs.candidate_cv_by_shape,
        obs.baseline_cv_by_shape,
        obs.paired_ratio_cv_by_shape,
        obs.paired_log_ci_by_shape,
        obs.timing_classification_by_shape,
    )
    if not expected or any(set(m or {}) != expected for m in raw_maps):
        return "raw paired timing keys do not match requested shapes"

    for name in names:
        cand = list(obs.candidate_samples_by_shape[name])
        base = list(obs.baseline_samples_by_shape[name])
        if len(cand) != obs.timing_pair_count or len(base) != obs.timing_pair_count:
            return (
                f"{name}: paired sample count does not match protocol "
                f"({len(cand)}/{len(base)} != {obs.timing_pair_count})")
        try:
            stats = paired_timing_stats(
                cand, base,
                noise_floor_pct=float(getattr(cfg, "noise_floor_pct", 2.0)),
                z=float(getattr(cfg, "paired_confidence_z", 1.96)),
            )
        except ValueError as exc:
            return f"{name}: {exc}"
        if len(cand) != len(base):
            return f"{name}: paired sample count mismatch"
        ratios = list(obs.paired_ratio_samples_by_shape[name])
        logs = list(obs.paired_log_speedup_samples_by_shape[name])
        if len(ratios) != len(cand) or len(logs) != len(cand):
            return f"{name}: derived paired sample count mismatch"
        for stored, recomputed in zip(ratios, stats["paired_ratios"]):
            if not math.isclose(float(stored), recomputed, rel_tol=1e-9, abs_tol=1e-12):
                return f"{name}: retained ratio sample does not match raw timing"
        for stored, recomputed in zip(logs, stats["paired_log_speedups"]):
            if not math.isclose(float(stored), recomputed, rel_tol=1e-9, abs_tol=1e-12):
                return f"{name}: retained log-speedup sample does not match raw timing"
        err = publication_admission_error(
            stats,
            min_pairs=max(2, int(getattr(cfg, "min_variance_runs", 3))),
            candidate_cv_threshold_pct=float(cfg.cv_threshold_pct),
            baseline_cv_threshold_pct=float(getattr(
                cfg, "baseline_cv_threshold_pct", cfg.cv_threshold_pct)),
            paired_ratio_cv_threshold_pct=float(getattr(
                cfg, "paired_ratio_cv_threshold_pct", cfg.cv_threshold_pct)),
            paired_ci_threshold_pct=float(getattr(
                cfg, "paired_ci_threshold_pct", cfg.cv_threshold_pct)),
        )
        if err:
            return f"{name}: {err}"
        stored_ci = list(obs.paired_log_ci_by_shape[name])
        if len(stored_ci) != 2 or not all(math.isclose(
                float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
                for a, b in zip(stored_ci, (stats["log_ci_lo"], stats["log_ci_hi"]))):
            return f"{name}: retained paired CI does not match raw timing"
        if obs.timing_classification_by_shape[name] != stats["classification"]:
            return f"{name}: retained paired classification does not match CI"
    return None


def _shape_ratios(obs: Observation) -> list[float]:
    """Complete per-shape speedups only; partial shape sweeps are never scored.

    Per-shape speedup ratios base_ms/cand_ms (a gain; higher is better). A partial
    shape sweep (timing that does not cover the exact requested/correctness shape
    key set) is refused so a candidate cannot be scored on a cherry-picked subset.
    """
    if not (obs.wall_by_shape or obs.baseline_by_shape):
        return []
    if not _timing_complete(obs):
        return []
    names = _required_shape_names(obs) or tuple(obs.wall_by_shape)
    if getattr(obs, "timing_grade", "legacy") == "publication":
        ratios = []
        for name in names:
            logs = obs.paired_log_speedup_samples_by_shape.get(name, [])
            if not logs:
                return []
            ratio = math.exp(sum(float(x) for x in logs) / len(logs))
            # A precise CI that still overlaps the configured noise band is a
            # statistical tie, not a publishable micro-win.
            if obs.timing_classification_by_shape.get(name) == "tie":
                ratio = 1.0
            ratios.append(ratio)
        return ratios
    out: list[float] = []
    for k in names:
        base = obs.baseline_by_shape.get(k)
        cand = obs.wall_by_shape.get(k)
        if _finite(base) and _finite(cand) and float(base) > 0.0 and float(cand) > 0.0:
            out.append(float(base) / float(cand))
    return out


def _worst_speedup(obs: Observation) -> Optional[float]:
    """Speedup on the worst shape: min over shapes of baseline/candidate.

    This is the diagnostic + eval metric (always worst-shape) and the CVaR_{a->0}
    endpoint of :func:`_aggregate_speedup`."""
    ratios = _shape_ratios(obs)
    if ratios:
        return min(ratios)
    # When a shape contract or any per-shape timing is present but produced no
    # complete ratio set, refuse to fall back to the scalar path (partial sweeps
    # are never scored).
    if (_declared_shape_names(obs) or obs.wall_by_shape or obs.baseline_by_shape):
        return None
    if (
        _finite(obs.baseline_ms)
        and _finite(obs.wall_ms)
        and float(obs.baseline_ms) > 0.0
        and float(obs.wall_ms) > 0.0
    ):
        return float(obs.baseline_ms) / float(obs.wall_ms)
    return None


def _aggregate_speedup(obs: Observation, cfg) -> Optional[float]:
    """Distributionally-robust speed aggregation over the per-shape speedup sweep.

    KORE's contribution is a *distributionally-robust* speed objective against the
    PRODUCTION vendor baseline: rather than the average-case speedup, it optimizes
    the worst shapes, so the policy must be fast on the hardest shape a practitioner
    hits - not just on average. This exposes the whole CVaR_alpha family (worst =
    CVaR_{a->0}, mean = CVaR_1) at a single point; all downstream shaping (log term,
    fast_p bonuses, significance) then applies to the chosen aggregate.

      "worst" : min over shapes (current behavior; the robust objective / default).
      "cvar"  : geometric mean of the worst ceil(alpha*N) shapes (CVaR_alpha).
      "mean"  : geometric-mean speedup over all shapes (average-case ablation arm).

    Geometric mean (mean-of-logs) is used for cvar/mean so the family is linear in
    ln(ratio) - consistent with the log-speedup shaping - and scale-correct for
    ratios. Degrades to the single-shape / scalar case identically to _worst_speedup,
    so the default ("worst") is byte-identical to the previous reward.
    """
    ratios = _shape_ratios(obs)
    if not ratios:
        # A shape contract / any per-shape timing present but not complete -> no score.
        if (_declared_shape_names(obs) or obs.wall_by_shape or obs.baseline_by_shape):
            return None
        if (
            _finite(obs.baseline_ms)
            and _finite(obs.wall_ms)
            and float(obs.baseline_ms) > 0.0
            and float(obs.wall_ms) > 0.0
        ):
            return float(obs.baseline_ms) / float(obs.wall_ms)
        return None
    mode = (getattr(cfg, "speed_aggregation", "worst") or "worst").lower()
    n = len(ratios)
    if mode == "worst" or n == 1:
        return min(ratios)                       # CVaR_{alpha->0}
    if mode == "mean":
        k = n
    else:  # "cvar"
        alpha = float(getattr(cfg, "cvar_alpha", 0.5) or 0.5)
        k = max(1, min(n, math.ceil(alpha * n)))
    worst_logs = sorted(math.log(r) for r in ratios)[:k]  # k worst (smallest) ratios
    return math.exp(sum(worst_logs) / k)


def _worst_snr(obs: Observation) -> Optional[float]:
    """Worst-shape SNR (min over shapes), falling back to the primary ``snr_db``.

    Mirrors the correctness gate, which is also scored on the WORST shape, so the
    sub-threshold credit reflects the same "hardest shape" the gate cares about.
    """
    if obs.snr_by_shape:
        vals = [float(v) for v in obs.snr_by_shape.values() if _finite(v)]
        if vals:
            return min(vals)
    return float(obs.snr_db) if _finite(obs.snr_db) else None


def _subthreshold_credit(obs: Observation, dtype: str, cfg,
                         snr_threshold: Optional[float]) -> float:
    """P1: bounded, continuous credit for a compiled-but-INCORRECT kernel.

    Returns ``eps_shape * clamp(worst_snr / snr_threshold, 0, 1)`` - a dense
    signal proportional to progress toward the correctness gate, so early RL
    isn't stuck on a flat-zero reward. The value lies in ``[0, eps_shape]`` and,
    because a kernel in the incorrect tier has worst-shape SNR *below* the gate,
    it is in practice strictly ``< eps_shape < correctness_weight`` - it can
    never reach, let alone cross, the correct tier. Returns 0 when shaping is
    off, when there is no SNR signal, or (by construction of the caller) for a
    flagged hack / compile-fail / infra error.
    """
    if not getattr(cfg, "subthreshold_shaping", False):
        return 0.0
    eps = float(getattr(cfg, "eps_shape", 0.0) or 0.0)
    if eps <= 0.0:
        return 0.0
    thr = snr_threshold if snr_threshold is not None else cfg.snr_threshold_for(dtype)
    snr = _worst_snr(obs)
    if snr is None or thr is None or thr <= 0.0:
        return 0.0
    progress = snr / thr
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    return eps * progress


def _format_component(response: Optional[str], cfg) -> float:
    """P2: bounded format-compliance term for the incorrect/correct tiers.

    ``response`` is the RAW policy output (the FULL_KERNEL contract), NOT the
    already-extracted kernel. Returns ``+format_weight`` when the response parses
    to a non-empty kernel (valid contract), ``-format_weight`` when it is
    malformed, and 0 when no response is supplied (the default - preserves the
    exact legacy reward for every current caller). The magnitude is kept far
    below every inter-tier gap, so this term can never flip tier ordering.
    """
    if response is None:
        return 0.0
    w = float(getattr(cfg, "format_weight", 0.0) or 0.0)
    if w <= 0.0:
        return 0.0
    # Lazy import to avoid any import cycle and to keep the hot path dependency-free.
    from kore.policy.format import parse_response
    kernel = (parse_response(response).get("kernel") or "").strip()
    return w if kernel else -w


@dataclass
class RewardResult:
    reward: float
    correct: bool
    speedup: Optional[float]
    tier: str
    flags: list[str] = field(default_factory=list)
    detail: str = ""

    def __post_init__(self) -> None:
        if not _finite(self.reward):
            raise ValueError(f"reward must be finite, got {self.reward!r}")
        self.reward = float(self.reward)
        if self.speedup is not None:
            if not _finite(self.speedup) or float(self.speedup) <= 0.0:
                raise ValueError(f"speedup must be finite and positive, got {self.speedup!r}")
            self.speedup = float(self.speedup)


def _speedup_term(su_scored: float, su_raw: float, obs: Observation, cfg,
                  flags: list[str]) -> float:
    """P4 speed reward: log-shaped speedup + significance-gated fast_p bonuses.

    ``su_scored`` is the (excessive-capped / high-variance-damped) speedup used for
    the continuous term; ``su_raw`` is the measured speedup used for the discrete
    threshold checks. Returns a NON-NEGATIVE speed contribution, so a correct
    kernel always scores >= ``correctness_weight`` (lexicographic dominance holds).

    Continuous term (breaks the linear plateau, steeper at the 1x crossover):
        speedup_log=True  ->  w*su           (su <= 1, linear, non-negative)
                              w*(1 + ln(su))  (su >  1, emphasized)
        speedup_log=False ->  w*max(su, 0)    (legacy linear)
    Discrete term (the strong ">1x" signal): cumulative ``fast_p_bonus`` for each
    threshold met, awarded ONLY when the speedup is statistically trustworthy
    (candidate/baseline/paired CVs and paired CI all under threshold, and no shape
    classified worse than "faster") and not an excessive-speedup measurement outlier.
    """
    if not _finite(su_scored) or not _finite(su_raw) or su_scored < 0.0 or su_raw <= 0.0:
        raise ValueError("speedups must be finite and non-negative/positive")
    w = float(getattr(cfg, "speedup_weight", 1.0) or 0.0)
    if getattr(cfg, "speedup_log", False) and su_scored > 1.0:
        term = w * (1.0 + math.log(su_scored))
    else:
        term = w * max(su_scored, 0.0)

    bonuses = getattr(cfg, "fast_p_bonus", ()) or ()
    if bonuses:
        sig_only = bool(getattr(cfg, "fast_p_significant_only", True))

        def _under(value, limit):
            return value is None or (
                math.isfinite(float(value)) and float(value) <= float(limit))

        trustworthy = (
            _under(obs.cv_pct, cfg.cv_threshold_pct)
            and _under(obs.baseline_cv_pct, getattr(
                cfg, "baseline_cv_threshold_pct", cfg.cv_threshold_pct))
            and _under(obs.paired_ratio_cv_pct, getattr(
                cfg, "paired_ratio_cv_threshold_pct", cfg.cv_threshold_pct))
            and _under(obs.paired_ci_half_width_pct, getattr(
                cfg, "paired_ci_threshold_pct", cfg.cv_threshold_pct))
        )
        classes = list((obs.timing_classification_by_shape or {}).values())
        if classes and not all(c == "faster" for c in classes):
            trustworthy = False
        excessive = "excessive_speedup" in flags
        # Require the speedup to clear the threshold by the measurement noise floor
        # (not just tie it): a kernel that merely PARITIES the baseline (1.00x) - or
        # beats it only within combined timing noise - must not farm the crossover
        # bonus. margin = 1 + noise_floor_pct/100 (e.g. 1.0x threshold -> need 1.02x).
        margin = 1.0 + float(getattr(cfg, "noise_floor_pct", 0.0) or 0.0) / 100.0
        if (not sig_only) or (trustworthy and not excessive):
            for thr, bonus in bonuses:
                if su_raw >= thr * margin:
                    term += float(bonus)
                    flags.append(f"fast_p>={thr}")
    return term


def _all_shapes_pass(obs: Observation, dtype: str, cfg, snr_threshold: Optional[float] = None) -> bool:
    thr = snr_threshold if snr_threshold is not None else cfg.snr_threshold_for(dtype)
    if not _finite(thr) or float(thr) <= 0.0:
        raise ValueError("SNR threshold must be finite and positive")
    # Strict requested-shape coverage: when the environment declares an exact shape
    # contract, the SNR map MUST cover precisely that key set (no missing shape, no
    # duplicate keys) or correctness is refused -- a candidate cannot pass by only
    # nailing a subset of the requested shapes.
    requested = tuple(getattr(obs, "requested_shapes", ()) or ())
    if requested:
        expected = set(requested)
        if len(expected) != len(requested) or set(obs.snr_by_shape) != expected:
            return False
    if obs.snr_by_shape:
        return all(_finite(v) and float(v) >= float(thr) for v in obs.snr_by_shape.values())
    return _finite(obs.snr_db) and float(obs.snr_db) >= float(thr)


def validate_reward_config(cfg) -> None:
    """Enforce the reward ladder with runtime checks (never ``assert``).

    This is called by every reward entry point, so ``python -O`` cannot remove a
    security boundary and ad-hoc config objects cannot bypass validation.
    """
    names = (
        "reward_hack",
        "reward_compile_fail",
        "reward_incorrect",
        "correctness_weight",
        "eps_shape",
        "format_weight",
        "speedup_weight",
        "excessive_speedup_flag",
        "cv_threshold_pct",
        "noise_floor_pct",
        "profile_reward_weight",
    )
    values = {}
    for name in names:
        value = getattr(cfg, name, 0.0)
        if not _finite(value):
            raise ValueError(f"{name} must be finite")
        values[name] = float(value)
    if values["eps_shape"] < 0.0 or values["format_weight"] < 0.0:
        raise ValueError("shaping magnitudes must be non-negative")
    if values["speedup_weight"] < 0.0 or values["profile_reward_weight"] < 0.0:
        raise ValueError("reward weights must be non-negative")
    if values["excessive_speedup_flag"] <= 0.0:
        raise ValueError("excessive_speedup_flag must be positive")
    if values["noise_floor_pct"] < 0.0 or values["cv_threshold_pct"] < 0.0:
        raise ValueError("timing thresholds must be non-negative")
    mode = str(getattr(cfg, "speed_aggregation", "worst") or "worst").lower()
    if mode not in {"worst", "mean", "cvar"}:
        raise ValueError(f"speed_aggregation must be worst|mean|cvar, got {mode!r}")
    alpha = getattr(cfg, "cvar_alpha", 0.5)
    if not _finite(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("cvar_alpha must be finite and in (0, 1]")

    # Include adverse format terms in every boundary calculation.
    hack = values["reward_hack"]
    compile_fail = values["reward_compile_fail"]
    incorrect_floor = values["reward_incorrect"] - values["format_weight"]
    incorrect_ceiling = (
        values["reward_incorrect"] + values["eps_shape"] + values["format_weight"]
    )
    correct_floor = values["correctness_weight"] - values["format_weight"]
    if not hack < compile_fail < incorrect_floor:
        raise ValueError(
            "reward tiers must satisfy hack < compile_fail < malformed-incorrect"
        )
    if not incorrect_ceiling < correct_floor:
        raise ValueError(
            "best incorrect reward must be strictly below worst correct reward"
        )

    bonuses = getattr(cfg, "fast_p_bonus", ()) or ()
    for entry in bonuses:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not _finite(entry[0])
            or not _finite(entry[1])
            or float(entry[0]) <= 0.0
            or float(entry[1]) < 0.0
        ):
            raise ValueError(f"invalid fast_p_bonus entry {entry!r}")
    if values["profile_reward_weight"] > 0.0 and bonuses:
        minimum = min(float(bonus) for _, bonus in bonuses)
        if values["profile_reward_weight"] >= minimum:
            raise ValueError(
                "profile_reward_weight must stay below the smallest fast_p bonus"
            )


def compute_reward(obs: Observation, source: str = "", dtype: str = "fp32",
                   mode: str = "eval", cfg=CONFIG,
                   snr_threshold: Optional[float] = None,
                   phase: Optional[str] = None,
                   response: Optional[str] = None,
                   roofline_gate: bool = False,
                   t_min_ms: Optional[float] = None,
                   roofline_tol: float = DEFAULT_ROOFLINE_TOL) -> RewardResult:
    """Lexicographic, anti-hackable reward. Returns a :class:`RewardResult`.

    Tier order (a strictly better outcome in an earlier tier ALWAYS dominates):
        hack < compile_fail < incorrect (shaped) < correct-but-slow < correct-fast.
    Correctness is scored with the Kevin reward ``correctness_weight + speedup``
    (linear, capped) - NOT log - so a correct kernel is *never* punished below an
    incorrect one, even when slower than the production baseline.

    Shaping upgrades (all bounded so lexicographic dominance holds absolutely):
      * P1 sub-threshold shaping - a compiled-but-incorrect kernel gets a small
        continuous credit in ``[0, eps_shape]`` toward the correctness gate,
        never enough to reach the correct tier. Never applied to a hack/compile
        failure/infra error.
      * P2 format term - pass the raw ``response`` (FULL_KERNEL contract) to add
        a tiny ``±format_weight`` bonus/penalty on the incorrect/correct tiers.
      * P3 curriculum ``phase`` - ``"correctness"`` zeroes the speed term (every
        correct kernel scores ``correctness_weight``); ``"full"``/``"latency"``
        (default) use ``correctness_weight + speedup``. Falls back to
        ``cfg.reward_phase`` when ``phase`` is None.

    ``snr_threshold`` overrides the dtype default (honors per-task task.yaml).

    ``roofline_gate`` (default OFF) enables the anti-reward-hack roofline SPEED-OF-LIGHT
    ceiling: when a ``t_min_ms`` roofline floor is supplied and the measured time is
    physically impossible below it (see :func:`roofline_ceiling_violation`, tolerance
    ``roofline_tol``), the candidate is rejected to the hack tier (a measurement
    exploit is never rewarded). With ``roofline_gate=False`` this reward is
    byte-identical to the pre-gate behavior for every existing caller.
    """
    validate_reward_config(cfg)
    flags: list[str] = []
    phase = (phase or getattr(cfg, "reward_phase", "full") or "full").lower()

    # Tier -1: infrastructure error (timeout/OOM/segfault/import) - not the
    # kernel's fault; caller must NOT cache it and should resample.
    if obs.infra_error:
        flags.append("infra")
        rr = RewardResult(cfg.reward_incorrect, False, None, "infra", flags,
                          obs.error_text or "infrastructure error")
        _log_decision(rr)
        return rr

    # Tier 0: anti-hack scan (a hack that "passes" must never be rewarded).
    # Punished STRICTLY harder than a compile failure (reward_hack < reward_compile_fail)
    # and never eligible for any shaping/format credit: cheating is the unique floor.
    hack = obs.hack_reason or (scan_for_hacks(source) if source else None)
    # Roofline SPEED-OF-LIGHT ceiling (opt-in): a physically-impossible sub-roofline
    # measured time is a measurement exploit -> the same hack floor. The source scan
    # takes precedence (keeps its specific reason); this only fires when no source
    # hack was found. Fully inert unless ``roofline_gate`` is set with a ``t_min_ms``,
    # so the default path stays byte-identical.
    ceiling_hack = False
    if not hack and roofline_gate and t_min_ms is not None:
        m_ceiling = _ceiling_measured_ms(obs)
        if roofline_ceiling_violation(m_ceiling, t_min_ms, roofline_tol):
            ceiling_hack = True
            hack = (f"measured {m_ceiling:.4g} ms is below the roofline speed-of-light "
                    f"T_min {t_min_ms:.4g} ms (tol {roofline_tol:.2f}) -- physically "
                    "impossible throughput; timing/measurement exploit")
    if hack:
        flags.append("hack")
        if ceiling_hack:
            flags.append("roofline_ceiling")
        rr = RewardResult(cfg.reward_hack, False, None, "hack", flags, str(hack))
        _log_decision(rr)
        return rr

    # Tier 1: compile
    if not obs.compiled:
        flags.append("compile_fail")
        rr = RewardResult(cfg.reward_compile_fail, False, None, "compile_fail", flags,
                          obs.error_text or "did not compile")
        _log_decision(rr)
        return rr

    # Tier 2: correctness (validation + SNR gate on all shapes)
    correct = obs.validation_passed and _all_shapes_pass(obs, dtype, cfg, snr_threshold)
    if not correct:
        flags.append("incorrect")
        # P1 sub-threshold shaping + P2 format term. Bounded so that
        #   max shaped-incorrect = eps_shape + format_weight < correctness_weight,
        # i.e. a shaped-incorrect kernel can never reach the correct tier.
        credit = _subthreshold_credit(obs, dtype, cfg, snr_threshold)
        fmt = _format_component(response, cfg)
        if credit > 0.0:
            flags.append("shaped")
        reward = cfg.reward_incorrect + credit + fmt
        detail = obs.error_text or "failed correctness/SNR"
        if credit > 0.0 or fmt:
            detail += f" (shaped +{credit:.4f}, format {fmt:+.4f})"
        rr = RewardResult(reward, False, None, "incorrect", flags, detail)
        _log_decision(rr)
        return rr

    base = cfg.correctness_weight
    fmt = _format_component(response, cfg)

    # New environment observations explicitly distinguish publication-grade
    # paired timing from screening/ineligible timing.  Old fabricated/replay
    # observations (timing_requested=False, grade="compat") retain their prior
    # scalar/per-shape behavior for schema compatibility.
    timing_expected = (
        getattr(obs, "timing_requested", False)
        or bool(obs.baseline_by_shape)
        or obs.baseline_ms is not None
    )
    grade = getattr(obs, "timing_grade", "compat")
    if timing_expected and (
            grade == "screening"
            or (grade in ("legacy", "compat")
                and getattr(obs, "timing_requested", False))):
        if not _timing_complete(obs):
            flags.extend(["infra", "incomplete_timing"])
            rr = RewardResult(
                cfg.reward_incorrect, False, None, "infra", flags,
                obs.error_text or "infrastructure error: incomplete screening timing",
            )
            _log_decision(rr)
            return rr
        flags.append("timing:screening")
        rr = RewardResult(
            base + fmt, True, None, "correct_screening", flags,
            "correct; legacy/unpaired timing is screening-only",
        )
        _log_decision(rr)
        return rr
    if timing_expected and grade == "ineligible":
        flags.append("performance_ineligible")
        rr = RewardResult(
            base + fmt, True, None, "correct_perf_ineligible", flags,
            obs.error_text or "correct; driver is not vendor-grade timing eligible",
        )
        _log_decision(rr)
        return rr
    if timing_expected and grade == "rejected":
        flags.extend(["infra", "timing_admission"])
        rr = RewardResult(
            cfg.reward_incorrect, False, None, "infra", flags,
            obs.error_text or "infrastructure error: timing admission rejected",
        )
        _log_decision(rr)
        return rr
    if timing_expected and grade == "publication":
        publication_error = _publication_timing_error(obs, cfg)
        if publication_error:
            flags.extend(["infra", "timing_admission"])
            rr = RewardResult(
                cfg.reward_incorrect, False, None, "infra", flags,
                f"infrastructure error: publication timing rejected: {publication_error}",
            )
            _log_decision(rr)
            return rr
    if timing_expected and not _timing_complete(obs):
        flags.extend(["infra", "incomplete_timing"])
        rr = RewardResult(
            cfg.reward_incorrect, False, None, "infra", flags,
            obs.error_text or "infrastructure error: incomplete all-shape timing",
        )
        _log_decision(rr)
        return rr

    # Tier 3: speed (only once correct). Kevin reward: base + linear speedup,
    # capped to bound measurement-error outliers. base>0 guarantees every correct
    # kernel (even a slow one, speedup>0) strictly beats the incorrect tier.
    # A correct kernel always parses to a kernel, so its format term is +format_weight
    # (never a penalty) - correct-fast vs correct-slow stays a pure speed ordering.
    su = _aggregate_speedup(obs, cfg)  # distributionally-robust (default: worst-shape)
    if su is None:
        rr = RewardResult(base + fmt, True, None, "correct_no_bench", flags,
                          "correct; no timing")
        _log_decision(rr)
        return rr

    su_scored = su
    if su >= cfg.excessive_speedup_flag:
        flags.append("excessive_speedup")  # likely measurement error; cap contribution
        su_scored = cfg.excessive_speedup_flag
    if _finite(obs.cv_pct) and float(obs.cv_pct) > float(cfg.cv_threshold_pct):
        flags.append("high_variance")  # noisy timing; keep correctness credit, damp speed
        su_scored = min(su_scored, 1.0)
    # P3 curriculum: the "correctness" phase zeroes the speed term so every
    # correct kernel scores exactly correctness_weight (+format); "full"/"latency"
    # keep the full correctness_weight + speedup.
    if phase == "correctness":
        flags.append("phase:correctness")
        speed_term = 0.0
    else:
        speed_term = _speedup_term(su_scored, su, obs, cfg, flags)
        # P5: bounded, baseline-relative hardware-counter dense bonus (flagship
        # novelty). Only on the correct tier; strictly below the fast_p bonuses so
        # real wall-clock wins always dominate. Inert when weight==0 / no profile.
        # Counter/physics shaping is empirical: it is admitted ONLY when the
        # family-held-out profile evidence passed AND carries a fingerprint, and the
        # efficiency is a well-formed [0,1] value -- otherwise the bonus is withheld.
        pw = float(getattr(cfg, "profile_reward_weight", 0.0) or 0.0)
        if (
            pw > 0.0
            and obs.profile_evidence_passed
            and obs.profile_evidence_fingerprint
            and _finite(obs.profile_efficiency)
            and 0.0 <= float(obs.profile_efficiency) <= 1.0
        ):
            prof_term = pw * float(obs.profile_efficiency)
            speed_term += prof_term
            flags.append(f"profile+{prof_term:.3f}")
    reward = base + speed_term + fmt
    rr = RewardResult(reward, True, su, "correct_timed", flags,
                      f"worst-shape speedup {su:.3f}x vs baseline"
                      + (" [correctness phase: speed zeroed]" if phase == "correctness" else ""))
    _log_decision(rr)
    return rr
