#!/usr/bin/env python
"""Run our model over AgentKernelArena tasks and score it against the Opus bar.

Three subcommands, deliberately separable so a failure in one does not cost the
others' GPU time:

  discover   list gfx950-runnable tasks by type (no GPU, no model)
  baseline   time the UNMODIFIED kernel of each task, establishing the reference
  run        generate an optimized kernel per task, then compile/check/time it

Each task runs in its own copied workspace. That is not tidiness: the tasks are
checked into a git tree we do not own, and the harnesses write artefacts next to
the source, so editing in place would corrupt the benchmark for every subsequent
run and silently change what later numbers mean.

Scoring lives in kore/eval/agent_kernel_arena.py and is AKA's own formula
unchanged, so the output is directly comparable to the published results:

  PyTorch-to-HIP 6.89x | HIP-to-HIP 6.69x | Triton-to-Triton 2.13x  (Opus)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kore.eval.agent_kernel_arena import (  # noqa: E402
    ArenaResult, discover_tasks, evaluate_task, summarize)

DEFAULT_ARENA = Path.home() / "third_party" / "AgentKernelArena"

PROMPT = """You are writing a GPU kernel for an AMD MI355X (gfx950).

{instructions}

{task} Keep the function name(s) {targets} and the public interface exactly as
they are -- the test harness imports them by name, and the result must be
numerically identical to the reference.

Return ONLY the complete contents of `{filename}` in a single ```{lang}
code block, with no commentary before or after.
{context}
Current contents of `{filename}`:
```{source_lang}
{source}
```"""

#: Fence language per target extension. The fence tells the model which language
#: to emit, and it is not always Python: a torch2hip task reads a .py module and
#: must produce a .hip translation unit. Asking for ```python there invites a
#: Python answer that cannot possibly build.
_FENCE = {".py": "python", ".hip": "cpp", ".cu": "cpp", ".cuh": "cpp",
          ".cpp": "cpp", ".cc": "cpp", ".h": "cpp", ".hpp": "cpp"}


def _fence_lang(rel: str) -> str:
    for ext, lang in _FENCE.items():
        if rel.endswith(ext):
            return lang
    return "python"


#: Cap on a single context file. A reference module is normally a few hundred
#: lines; anything far bigger is a vendored blob that would crowd out the actual
#: question.
_CONTEXT_CHARS = 24_000


def _render_context(task, ws) -> str:
    """Show the problem statement that is not in the file being written.

    torch2flydsl ships model.py (the PyTorch to translate) next to kernel.py (the
    blank target) and declares only kernel.py, so without this the model is shown
    an empty file and asked to match semantics it was never given -- which is
    exactly the shape of that category's result: 41 of 45 compile, none correct.
    """
    parts = []
    for rel in task.context_files():
        p = ws / rel
        if not p.exists():
            continue
        body = p.read_text()[:_CONTEXT_CHARS]
        parts.append(f"Reference `{rel}` (the behaviour to reproduce):\n"
                     f"```{_fence_lang(rel)}\n{body}\n```")
    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


def _preserve_note(ws, dst_rel: str, task) -> str:
    """Warn when the file being rewritten also holds the tests that grade it.

    The rocmbench tasks behind instruction2triton keep the kernel and its pytest
    suite in one module, and the correctness command runs pytest against that same
    file. A model told to "return the complete contents" returns the kernel and
    drops the tests, so pytest collects nothing and exits 5 -- scored as incorrect
    though the kernel was never actually judged. Every one of the 24 scored so far
    failed this way, not on numerics.
    """
    p = ws / dst_rel
    if not p.exists():
        return ""
    body = p.read_text(errors="ignore")
    if "def test_" not in body and "pytest" not in body:
        return ""
    return ("\nThis file also contains the test suite that grades your work, and "
            "it is run against this same file. Reproduce every existing test and "
            "helper VERBATIM alongside your implementation -- if the tests are "
            "missing, nothing can be collected and the task scores zero however "
            "good the kernel is.\n")


def _loader_entry_point(task) -> str:
    """The attribute the task's own loader reads off the built extension.

    Read rather than assumed. Every one of the 79 templates present today wants
    ``forward``, but hardcoding it would silently mislead the model on the first
    task that does not, and a wrong entry point fails identically to no entry
    point at all.
    """
    tpl = task.root / "eval_tools" / "kernel_loader_template.py"
    if not tpl.exists():
        return ""
    m = re.search(r"_ext\.([A-Za-z_]\w*)", tpl.read_text(errors="ignore"))
    return m.group(1) if m else ""


def _extension_contract(ws, dst_rel: str, task) -> str:
    """State the torch-extension contract when the target starts out empty.

    torch2hip ships all 57 of its targets as zero-byte .hip files, so there is no
    existing interface to preserve, and the loader the task generates does:

        ext = torch.utils.cpp_extension.load(name=..., sources=[that .hip])
        fn  = ext.forward

    A model that writes only a __global__ kernel produces a translation unit with
    no pybind11 module, and the build fails before numerics are ever considered.
    All 62 torch2hip candidates across both arms failed that way. hip2hip, whose
    56 targets all ship non-empty, scored normally on the same toolchain -- the
    boilerplate is simply visible there in the file being rewritten.

    kernel_loader_template.py stays out of the context window as harness
    scaffolding, which is correct: it is not the problem statement. The entry
    point it requires is, so name it here.
    """
    if _fence_lang(dst_rel) != "cpp":
        return ""
    p = ws / dst_rel
    if p.exists() and p.read_text(errors="ignore").strip():
        return ""   # an existing implementation already demonstrates the contract
    entry = _loader_entry_point(task)
    if not entry:
        return ""
    return (
        f"\nThis file is empty and is built as a PyTorch C++ extension with "
        f"`torch.utils.cpp_extension.load`, then called as `ext.{entry}`. So it "
        f"must be a COMPLETE translation unit, not just a kernel: include "
        f"<torch/extension.h> and <hip/hip_runtime.h>, implement the host-side "
        f"function that allocates the output and launches the kernel, and export "
        f"it as\n"
        f"    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) "
        f"{{ m.def(\"{entry}\", &your_host_function); }}\n"
        f"Without that module block the extension has no entry point and the task "
        f"scores zero no matter how good the kernel is.\n"
    )


def _task_verb(task, src_rel: str, dst_rel: str) -> str:
    """Say whether this is an in-place optimization or a translation.

    Same-language tasks rewrite one file; translation tasks read one file and
    write a different one, and telling the model to "rewrite" the file it is
    supposed to be translating INTO is actively misleading.
    """
    if dst_rel == src_rel:
        return f"Rewrite the file `{dst_rel}` so the kernel is FASTER."
    return (f"Implement `{dst_rel}` as the {_fence_lang(dst_rel)} equivalent of "
            f"`{src_rel}` shown below, and make it as FAST as possible.")


def _extract_code(text: str) -> str:
    """Pull the first fenced code block, whatever language it is tagged with.

    Any tag, not just python: the prompt asks for ```cpp on .hip targets, and a
    pattern that only accepted ```python silently fell through to "return the
    whole reply", handing the compiler the model's prose along with its kernel.
    That regressed hip2hip from 23 compiled to 12 -- a change to the prompt
    breaking an assumption in the parser, with nothing in between to catch it.

    The fallback is still deliberate: a model that ignores the fence usually emits
    valid source anyway, and refusing that would score a formatting slip as a
    compile failure.
    """
    import re

    m = re.search(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


#: Repositories a task may expect to find beside itself, checked out once and
#: linked into each workspace. The `repository` and `image_kernel` tasks are
#: edits to a real library: their task dir holds only a runner, and the file to
#: change lives at e.g. aiter/ops/triton/... inside a checkout the runner adds to
#: sys.path. Without it those 16 tasks cannot compile no matter what the model
#: writes -- the file it is asked to edit does not exist.
_REPO_CACHE = Path.home() / "third_party"


#: A checkout can be needed under a name that appears nowhere in the task's declared
#: source paths. `repository/rocprim/*` asks for a directory called `rocPRIM` -- the
#: repository's own capitalisation -- while its sources are listed as `rocprim/...`,
#: and the run failed with "Source directory not found: <ws>/rocPRIM". Matching case-
#: insensitively against what is on disk is what connects the two.
def _resolve_checkout(name: str) -> Optional[Path]:
    direct = _REPO_CACHE / name
    if direct.is_dir():
        return direct
    if not _REPO_CACHE.is_dir():
        return None
    lowered = name.lower()
    for cand in _REPO_CACHE.iterdir():
        if cand.is_dir() and cand.name.lower() == lowered:
            return cand
    return None


def _stage_repo(src: Path, dst: Path) -> None:
    """Hard-link a checkout into a workspace, copying when links are impossible.

    Hard links: a fresh 165MB copy per task, times eight parallel workers, is
    minutes of I/O for files nobody edits. Only the answer file is written, and the
    writer unlinks first so it never truncates a shared inode and corrupts another
    workspace's copy.

    A link cannot cross filesystems (EXDEV), so linking is an optimisation and not a
    requirement: with the workspace on node-local disk and the checkout on /home,
    every file raises "Invalid cross-device link" and the whole staging fails. Fall
    back per file so the task still runs, just slower.
    """
    if dst.exists():
        return

    def _link_or_copy(a, b):
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)

    shutil.copytree(src, dst, copy_function=_link_or_copy,
                    ignore=shutil.ignore_patterns(".git"))


def _provide_task_repo(ws: Path) -> Optional[str]:
    """Give a task the repository checkout it expects, locally.

    Two families, two mechanisms, and both scored zero *compiled* in the v4 run with
    nothing judged on its code.

    ``image_kernel`` declares ``image_repo_path: /sgl-workspace/aiter`` -- an absolute
    path inside the arena's Docker image. On bare metal it does not exist and cannot
    be created, since /sgl-workspace is not writable without root, so the task's own
    runner resolved an empty repo root and died in ``os.chdir('')`` with
    FileNotFoundError. The key is ``image_repo_path``, not ``repo_path``: matching the
    shorter name as a whole line silently matched nothing, and grepping for it matched
    the longer key as a substring, which made a broken rewrite look like a working one.

    ``repository`` declares ``repo_url: https://github.com/ROCm/rocPRIM.git`` and
    expects the clone already present under the repository's own name, failing with
    "Source directory not found: <ws>/rocPRIM". Nothing clones it during a run, so it
    is staged from the local checkout -- found case-insensitively, because the
    directory is ``rocPRIM`` while the task's own source paths say ``rocprim``.
    """
    cfg = ws / "config.yaml"
    if not cfg.is_file():
        return None
    text = cfg.read_text(errors="ignore")
    missing = []

    m = re.search(r"^(\s*[A-Za-z_]*repo_path:[ \t]*)(\S+)[ \t]*$", text, re.M)
    if m:
        declared = Path(m.group(2))
        if not declared.is_dir():
            local = _resolve_checkout(declared.name)
            if local is None:
                missing.append(f"{declared.name} (for {m.group(1).strip()})")
            else:
                # Stage inside the workspace and point the task at that copy, so it
                # edits its own hard-linked tree and never the shared checkout.
                _stage_repo(local, ws / declared.name)
                text = (text[:m.start()] + m.group(1) + str(ws / declared.name)
                        + text[m.end():])
                cfg.write_text(text)

    for um in re.finditer(r"^\s*repo_url:[ \t]*(\S+)[ \t]*$", text, re.M):
        name = um.group(1).rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if not name or (ws / name).exists():
            continue
        local = _resolve_checkout(name)
        if local is None:
            missing.append(f"{name} (for repo_url)")
            continue
        _stage_repo(local, ws / name)

    return "missing checkout(s): " + "; ".join(missing) if missing else None


def _link_required_repo(task, ws: Path) -> Optional[str]:
    """Make every checkout a task needs present in its workspace."""
    for rel in task.source_files:
        top = rel.split("/")[0]
        if not top or "/" not in rel or (ws / top).exists():
            continue
        src = _resolve_checkout(top)
        if src is None:
            # Not every leading path segment is a repository: aiter's own sources
            # are declared as csrc/... and sglang's as python/sglang/..., so a
            # missing directory here is only fatal if repo_path cannot supply it.
            continue
        _stage_repo(src, ws / top)
    return _provide_task_repo(ws)


def _attempt_task(task, ws, dst_rel, prompt, policy, args, ref_latency):
    """Generate, score, and retry with the harness's own feedback. Best wins.

    AKA's reference agents run with ``max_iterations: 3`` and full tool access in
    the workspace: they compile, read the compiler error, and try again. Scoring
    one shot against those published numbers compares two different procedures,
    and the gap falls hardest on exactly the categories gated on compiling a
    translation unit correctly first time -- torch2hip, which carries the highest
    bar of all.

    The workspace is deliberately NOT reset between attempts. AKA's agent
    accumulates state in it across iterations, so rebuilding a clean tree each
    time would be a different (easier, and less faithful) task.

    The best attempt by AKA score is kept rather than the last, because a later
    attempt can regress -- a model chasing speed can break correctness it already
    had, and reporting the final state would score the regression.
    """
    best = None
    feedback = None
    for attempt in range(1, max(1, args.attempts) + 1):
        reply = policy(prompt) if feedback is None else policy(prompt, feedback)
        code = _extract_code(reply)
        # Distinguish "the model produced nothing usable" from "the model wrote a
        # bad kernel". Both score zero, and without this the ledger cannot tell
        # them apart -- which is how a generation-side bug hides for a whole
        # sweep looking like a capability result.
        note = _generation_health(reply, code)
        if note:
            print(f"    attempt {attempt}: generation: {note}", flush=True)
        _write_answer(ws / dst_rel, code)
        r = evaluate_task(task, ws, timeout=args.timeout,
                          reference_latency=ref_latency)
        r.detail = {**(r.detail or {}), "attempt": attempt,
                    "attempts_allowed": args.attempts}
        if best is None or r.score > best.score:
            best = r
        if args.attempts == 1:
            break
        print(f"    attempt {attempt}/{args.attempts}: compiled={r.compiled} "
              f"correct={r.correct} score={r.score:.0f}", flush=True)
        # Feed the harness's own verdict back, the way the reference agent reads
        # its compiler output. _render_feedback turns a correct result into
        # "propose one further optimization", so a passing kernel keeps being
        # improved rather than ending the budget early.
        feedback = {"compiled": r.compiled, "correct": r.correct,
                    "error_text": r.error, "speedup": r.speedup}
    if best is not None and args.attempts > 1:
        best.detail = {**(best.detail or {}), "best_of": args.attempts}
    return best


#: Models reached over the gateway rather than served from a checkpoint. Matched
#: by name because that is what the operator types; a local path never looks
#: like one of these.
_API_MODEL_PREFIXES = ("claude-", "anthropic/", "gpt-", "opus", "sonnet")


def _is_api_model(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _API_MODEL_PREFIXES)


def _api_generate(args):
    """A ``generate`` for model_policy backed by the AMD LLM gateway.

    model_policy calls ``gen(messages, max_tokens=..., temperature=...)`` and
    ClaudeTeacher.generate takes the same message list, so the adapter is only
    about the keyword arguments. Nothing else in the run changes: an API arm and
    a checkpoint arm see the same prompts, the same retry budget and the same
    scorer, which is the only way the comparison means anything.

    The teacher is built once per worker. Building it per call would re-read
    .env.local and re-create the HTTP client for every attempt of every task.
    """
    import os

    from kore.data.teacher import ClaudeTeacher, load_env_local

    load_env_local()
    # ClaudeTeacher resolves its model as os.environ.get("KORE_TEACHER_MODEL",
    # model), so .env.local silently wins over the constructor argument. That is
    # right for datagen, where one variable retargets every sweep at once, and
    # wrong here: an arm labelled claude-opus-4.8 in the ledger must be that
    # model and not whatever the datagen default happens to be that week. A
    # benchmark row naming the wrong model is worse than no row.
    os.environ["KORE_TEACHER_MODEL"] = args.model
    teacher = ClaudeTeacher(model=args.model, temperature=args.temperature,
                            max_tokens=args.max_tokens)
    if teacher.model != args.model:
        raise SystemExit(f"refusing to run: asked for {args.model}, "
                         f"teacher resolved {teacher.model}")

    def gen(messages, max_tokens=None, temperature=None, **_):
        return teacher.generate(messages)

    return gen


def _generation_health(reply: str, code: str) -> str:
    """Describe a reply that is suspect on its face, or "" when it looks fine.

    A truncated or empty generation compiles to zero exactly like a wrong kernel,
    so without a note here the two are indistinguishable in the ledger and a
    generation-side regression reads as a capability result for the whole sweep.
    """
    if not (reply or "").strip():
        return "EMPTY reply from the model"
    if not (code or "").strip():
        return f"reply had {len(reply)} chars but no code survived extraction"
    # An opening fence with no closing fence is what a hit token-limit looks like.
    if reply.count("```") % 2 == 1:
        return f"UNCLOSED code fence ({len(reply)} chars) -- likely truncated"
    return ""


def _materialize_perf_helpers(ws: Path, arena_root: str) -> None:
    """Replace the sabotaged helper stubs the arena ships in task sources.

    AKA deliberately commits `performance_utils_pytest.py` and the vLLM
    benchmark helper as stubs that `raise RuntimeError`, and replaces them per
    workspace in setup_workspace(). Copying a task directory without that step
    leaves the traps armed: every rocmbench kernel imports the helper at module
    scope, so `pytest <kernel>.py` dies during collection and the task is scored
    incorrect -- 61 tasks whose correctness result says nothing about the model.
    The vLLM runners call the other stub in their performance phase, so a further
    124 tasks can be correct and never report a speedup.

    We call AKA's own function rather than reimplement it, so the helper we run
    against is byte-identical to the one the published numbers used.
    """
    root = Path(arena_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.perf_helper_materialization import (  # noqa: PLC0415
            materialize_perf_helpers_in_workspace,
        )
    except Exception as exc:  # noqa: BLE001 - report once, do not kill the sweep
        global _MATERIALIZE_WARNED
        if not _MATERIALIZE_WARNED:
            _MATERIALIZE_WARNED = True
            print(f"WARNING: cannot import AKA perf-helper materialization "
                  f"({type(exc).__name__}: {exc}); rocmbench and vLLM tasks will "
                  f"fail on the shipped stubs", flush=True)
        return
    try:
        materialize_perf_helpers_in_workspace(ws, root=root)
    except Exception as exc:  # noqa: BLE001 - one task must not end the run
        print(f"WARNING: perf-helper materialization failed for {ws.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)


_MATERIALIZE_WARNED = False


#: A pytest config of our own, placed in the workspace so ours cannot reach it.
#:
#: A workspace lives inside this repository, so pytest walks up from it, finds our
#: pyproject.toml, and adopts it as the rootdir config. Ours carries
#: `addopts = [..., "-m", "not gpu and not release"]`, and that filter applies to the
#: task's tests: measured on a two-test file, "1 passed, 1 deselected" against
#: "2 passed" once isolated. On a GPU benchmark most tests are marked gpu, so a task
#: whose tests are all GPU tests collected nothing and scored incorrect having never
#: been run.
#:
#: An ini file beside the tests wins over an ancestor pyproject.toml, so this makes
#: the workspace its own rootdir with no inherited opinions. (`python_files` is not
#: the problem -- pytest collects an explicitly named file whatever the pattern -- but
#: it is set permissively here so a task naming a directory behaves too.)
_WORKSPACE_PYTEST_INI = """[pytest]
python_files = *.py
python_classes = Test*
python_functions = test_*
addopts =
"""


def _workspace(task, out_root: Path, arena_root: str) -> Path:
    ws = out_root / "workspaces" / task.task_id.replace("/", "__")
    if ws.exists():
        shutil.rmtree(ws)
    ws.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task.root, ws)
    _link_required_repo(task, ws)
    _materialize_perf_helpers(ws, arena_root)
    # Only if the task ships none of its own, so a task that has an opinion keeps it.
    if not any((ws / n).exists() for n in
               ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")):
        (ws / "pytest.ini").write_text(_WORKSPACE_PYTEST_INI)
    return ws


def _splice_answer(original: str, code: str) -> str:
    """Put the model's functions back into the file they came from.

    The rocmbench tasks behind instruction2triton hand over a whole module --
    imports, helpers, and the pytest suite that grades it -- and ask for one
    function to be filled in. A model reliably returns just that function, and
    writing the reply as the file therefore deletes the imports it needs. The
    result reads as a model failure and is not one: the compile check is
    ``ast.parse``, which a bare decorated function passes, so the task scores
    "compiled" and then dies at import with ``NameError: name 'triton' is not
    defined``. All 18 scored instruction2triton tasks failed exactly that way, and
    none was judged on numerics.

    Asking the model to reproduce the scaffolding verbatim was tried first and does
    not hold. Splicing does not depend on the model cooperating: each function the
    reply defines replaces the one of that name in the original, and everything
    else -- imports, helpers, tests -- is preserved because it is never rewritten.

    Returns the merged text, or ``code`` unchanged when the reply is already a
    complete module or nothing can be matched up.
    """
    import ast

    try:
        new_tree = ast.parse(code)
        old_tree = ast.parse(original)
    except SyntaxError:
        return code   # cannot reason about it; caller's behaviour is unchanged

    def _defs(tree):
        return {n.name: n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    new_defs, old_defs = _defs(new_tree), _defs(old_tree)
    shared = set(new_defs) & set(old_defs)
    if not shared:
        return code

    # A reply that already carries the original's imports is a complete file and
    # must be left alone -- splicing it would duplicate the module.
    old_imports = {a.name.split(".")[0]
                   for n in ast.walk(old_tree) if isinstance(n, ast.Import)
                   for a in n.names}
    new_imports = {a.name.split(".")[0]
                   for n in ast.walk(new_tree) if isinstance(n, ast.Import)
                   for a in n.names}
    if old_imports and old_imports <= new_imports:
        return code

    def _span(node):
        """Line span of a definition, decorators included."""
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return start - 1, node.end_lineno

    old_lines = original.splitlines()
    new_lines = code.splitlines()
    # Replace from the bottom so earlier spans keep their line numbers.
    for name in sorted(shared, key=lambda n: old_defs[n].lineno, reverse=True):
        o_start, o_end = _span(old_defs[name])
        n_start, n_end = _span(new_defs[name])
        old_lines[o_start:o_end] = new_lines[n_start:n_end]
    return "\n".join(old_lines) + "\n"


def _write_answer(path: Path, code: str) -> None:
    """Replace a file, never truncate it in place.

    Workspaces share unedited files with a cached checkout by hard link, so
    opening for write would rewrite the bytes every linked copy sees -- including
    other tasks running concurrently. Unlinking first breaks this file out of the
    link set and leaves the rest shared.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".py" and path.exists():
        try:
            code = _splice_answer(path.read_text(errors="ignore"), code)
        except Exception:  # noqa: BLE001 - a splice failure must not lose the answer
            pass
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(code)


def cmd_discover(args) -> int:
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)
    by = {}
    for t in tasks:
        by.setdefault(t.task_type, []).append(t)
    print(f"{len(tasks)} tasks runnable on {args.gpu_arch}")
    for k, v in sorted(by.items()):
        print(f"  {len(v):>4}  {k}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"n": len(tasks), "by_type": {k: len(v) for k, v in by.items()},
             "task_ids": [t.task_id for t in tasks]}, indent=2) + "\n")
    return 0


def cmd_baseline(args) -> int:
    """Time each task's shipped kernel, unmodified.

    This is the control. Without it a low score is ambiguous between "our model
    wrote a slow kernel" and "this task does not run on our stack at all", and
    those call for completely different responses.
    """
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)[: args.limit or None]
    # Shards and a durable ledger, for the same reasons `run` needs them: the
    # control has to cover all 402 tasks to be worth anything, and single-process
    # it does not fit in one allocation.
    ledger = out_root / (f"baseline.shard{args.shard}of{args.num_shards}.jsonl"
                         if args.num_shards > 1 else "baseline.partial.jsonl")
    done = set()
    for f in sorted(out_root.glob("baseline*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - torn last line after a kill
                continue
            if row.get("task_id"):
                done.add(row["task_id"])
    if done:
        print(f"resume: {len(done)} task(s) already timed", flush=True)
    if args.num_shards > 1:
        tasks = tasks[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(tasks)} task(s)", flush=True)
    results = []
    for i, task in enumerate(tasks, 1):
        if task.task_id in done:
            continue
        ws = _workspace(task, out_root, args.arena_root)
        r = evaluate_task(task, ws, timeout=args.timeout)
        results.append(r)
        with ledger.open("a") as fh:
            fh.write(json.dumps(r.to_dict()) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[{i}/{len(tasks)}] {task.task_id}: compiled={r.compiled} "
              f"correct={r.correct} speedup={r.speedup} score={r.score:.0f}",
              flush=True)
        if not args.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)
    if args.num_shards > 1:
        print(f"shard {args.shard} done; run `baseline-merge` when all exit", flush=True)
        return 0
    _write(out_root / "baseline_results.json", results, args)
    return 0


def cmd_baseline_merge(args) -> int:
    """Combine baseline shard ledgers into baseline_results.json."""
    out_root = Path(args.out)
    seen, results = set(), []
    for f in sorted(out_root.glob("baseline*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            tid = row.get("task_id")
            if tid and tid not in seen:
                seen.add(tid)
                results.append(_result_from_dict(row))
    if not results:
        print("baseline-merge: no ledgers", flush=True)
        return 1
    print(f"baseline-merge: {len(results)} task(s)", flush=True)
    _write(out_root / "baseline_results.json", results, args)
    return 0


def cmd_run(args) -> int:
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)[: args.limit or None]
    print(f"{len(tasks)} tasks; model={args.model}", flush=True)

    from kore.eval.policies import model_policy  # local: heavy import

    # model_policy takes the checkpoint positionally and has no `revision` or
    # `model_id` parameter; the previous call passed both by keyword and raised
    # TypeError before a single task ran. `revision` belongs to the hub-pinning
    # path and is meaningless for a local checkpoint directory, which is what an
    # evaluated run always points at.
    #
    # An API model is injected as `generate` rather than served, which is the
    # seam model_policy already exposes. Everything downstream is untouched: the
    # same transcript, the same three attempts, the same verifier feedback and
    # the same scoring. That is the whole point of running one -- a comparison
    # against a frontier model is only worth anything if the harness either side
    # of it is identical.
    policy = model_policy(args.model,
                          generate=_api_generate(args) if _is_api_model(args.model)
                          else None,
                          max_tokens=args.max_tokens,
                          temperature=args.temperature)

    # Durable, append-as-you-go ledger. The previous version accumulated results
    # in memory and wrote once after the loop, so a preempted run lost every task
    # it had scored -- and on this cluster preemption is the normal way a long job
    # ends, not an exception. A full sweep is 254 tasks at 6-11 minutes each, so
    # that was throwing away many hours at a time.
    #
    # Under sharding each worker owns its own ledger. Eight processes appending to
    # one file would interleave partial lines under concurrent writes, and the
    # torn-line tolerance below is meant for a process that was killed, not for
    # routine corruption.
    ledger = out_root / _ledger_name(args)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Skip anything any ledger has already scored, not just this shard's. That
    # covers resuming after a re-shard and after the earlier unsharded runs, whose
    # results stay valid: a task's score does not depend on which GPU produced it.
    done = _scored_task_ids(out_root, args.arm)
    if done:
        print(f"resume: {len(done)} task(s) already scored across "
              f"{args.arm} ledgers in {out_root}", flush=True)
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn last line is expected after a kill
                continue
            if row.get("task_id"):
                results.append(_result_from_dict(row))

    # Per-task reference latency from the baseline run, which is the denominator
    # for every suite that reports an absolute time instead of a ratio.
    ref_latency = _reference_latencies(out_root)
    print(f"reference latencies available for {len(ref_latency)} task(s)"
          + ("" if ref_latency else " -- run `baseline` first or speedups will be"
             " unavailable for GEAK-style suites"), flush=True)

    if args.num_shards > 1:
        # Stride rather than contiguous blocks: task cost varies a lot by category
        # and the categories are discovered in order, so contiguous slices would
        # hand one worker all the expensive kernels and leave others idle.
        tasks = tasks[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(tasks)} task(s) of this "
              f"worker's slice", flush=True)

    t0 = time.time()
    pending = [t for t in tasks if t.task_id not in done]

    def _score_one(task):
        """Everything one task needs, start to finish, with no shared state."""
        ws = _workspace(task, out_root, args.arena_root)
        try:
            src_rel = task.source_files[0] if task.source_files else task.answer_path()
            dst_rel = task.answer_path()
            source = (ws / src_rel).read_text() if (ws / src_rel).exists() else ""
            prompt = PROMPT.format(
                instructions=task.instructions or "Optimize this kernel.",
                filename=dst_rel, targets=", ".join(task.target_functions) or "all",
                source=source, lang=_fence_lang(dst_rel),
                source_lang=_fence_lang(src_rel),
                task=_task_verb(task, src_rel, dst_rel)
                     + _preserve_note(ws, dst_rel, task)
                     + _extension_contract(ws, dst_rel, task),
                context=_render_context(task, ws))
            r = _attempt_task(task, ws, dst_rel, prompt, policy, args,
                              ref_latency.get(task.task_id))
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            r = ArenaResult(task_id=task.task_id, task_type=task.task_type,
                            error=f"{type(exc).__name__}: {exc}")
        if not args.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)
        return r

    # Several tasks in flight per worker. Each task's three attempts stay strictly
    # sequential -- attempt two needs attempt one's compiler output -- but tasks are
    # independent, so overlapping them is what lets vLLM batch at all. Scored one
    # task at a time, every generate call was a batch of one, which is single-stream
    # latency from an engine whose whole advantage is continuous batching: 8 GPUs
    # were busy and none was saturated.
    #
    # Each task owns its workspace and writes only its own answer file, so the
    # concurrency needs no coordination beyond serialising the ledger append.
    lock = threading.Lock()
    n_done = 0

    def _record(i: int, r: ArenaResult) -> None:
        nonlocal n_done
        n_done += 1
        results.append(r)
        # Append + flush + fsync before moving on. A score that is only in memory
        # is a score we will pay for twice.
        with ledger.open("a") as fh:
            fh.write(json.dumps(r.to_dict()) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[{n_done}/{len(pending)}] {r.task_id}: compiled={r.compiled} "
              f"correct={r.correct} speedup={r.speedup} score={r.score:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    if args.task_concurrency > 1 and len(pending) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.task_concurrency) as pool:
            futs = {pool.submit(_score_one, t): t for t in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                with lock:
                    _record(i, fut.result())
    else:
        for i, task in enumerate(pending, 1):
            _record(i, _score_one(task))
    if args.num_shards > 1:
        # Only the merge step may write results_<arm>.json: it is the signal that
        # the whole arena is finished, and a shard that completes its own slice has
        # no idea whether its siblings have.
        print(f"shard {args.shard}/{args.num_shards} finished its slice; "
              "run `merge` once every shard exits", flush=True)
        return 0
    _write(out_root / f"results_{args.arm}.json", results, args)
    return 0


def _reference_latencies(out_root: Path) -> dict:
    """Baseline latency per task id, from whatever the baseline run left behind.

    Reads the shard ledgers as well as the merged file so a baseline that is still
    in flight is already usable -- there is no reason to wait for all 402 before
    scoring the tasks it has already timed.
    """
    out = {}
    files = sorted(out_root.glob("baseline*.jsonl"))
    merged = out_root / "baseline_results.json"
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001 - torn last line after a kill
                continue
    if merged.exists():
        try:
            rows.extend(json.loads(merged.read_text()).get("results", []))
        except Exception:  # noqa: BLE001 - a half-written merge must not stop a run
            pass
    for r in rows:
        lat = r.get("optimized_seconds")
        tid = r.get("task_id")
        # Only a correct baseline is a meaningful denominator: timing a reference
        # that failed its own correctness check would inflate every speedup
        # measured against it.
        if tid and lat and r.get("correct"):
            out[tid] = lat
    return out


def _ledger_name(args) -> str:
    """Ledger filename, per-shard when sharded.

    The unsharded name is left exactly as it was so an in-flight single-process
    run keeps resuming from the file it has been writing.
    """
    if args.num_shards > 1:
        return f"results_{args.arm}.shard{args.shard}of{args.num_shards}.partial.jsonl"
    return f"results_{args.arm}.partial.jsonl"


def _scored_task_ids(out_root: Path, arm: str) -> set:
    """Every task id recorded by any ledger for this arm."""
    done = set()
    for f in sorted(out_root.glob(f"results_{arm}*.partial.jsonl")):
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - torn last line after a kill
                continue
            if row.get("task_id"):
                done.add(row["task_id"])
    return done


def cmd_merge(args) -> int:
    """Combine every shard ledger into the final results_<arm>.json.

    Runs after the shards exit. Deduplicates by task id, keeping the first
    occurrence, because a re-shard can legitimately score a task twice.

    results_<arm>.json is written only when every discovered task is present. It
    is what the supervisor treats as "the arena is finished", so writing it after a
    worker crashed would end the sweep early and quietly report a partial sweep as
    a complete one. An incomplete merge instead reports what is missing and exits
    non-zero, which leaves the job absent from the queue and gets it resubmitted.
    """
    out_root = Path(args.out)
    ledgers = sorted(out_root.glob(f"results_{args.arm}*.partial.jsonl"))
    seen, results = set(), []
    for f in ledgers:
        for line in f.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - torn last line after a kill
                continue
            tid = row.get("task_id")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            results.append(_result_from_dict(row))
    if not results:
        print(f"merge: no ledgers found under {out_root}", flush=True)
        return 1

    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)[: args.limit or None]
    expected = {t.task_id for t in tasks}
    missing = expected - seen
    print(f"merge: {len(results)} task(s) from {len(ledgers)} ledger(s); "
          f"{len(expected)} expected, {len(missing)} missing", flush=True)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        print(f"merge: INCOMPLETE -- not writing results_{args.arm}.json. "
              f"missing e.g. {sample}", flush=True)
        return 1
    _write(out_root / f"results_{args.arm}.json", results, args)
    return 0



def _result_from_dict(row: dict) -> "ArenaResult":
    """Rebuild an ArenaResult from a ledger line.

    Only the fields ArenaResult actually declares are passed through: the ledger
    is written by to_dict() and may gain keys later, and a resumed run must not
    die because it was written by a newer build than the one reading it.
    """
    import dataclasses
    allowed = {f.name for f in dataclasses.fields(ArenaResult)}
    return ArenaResult(**{k: v for k, v in row.items() if k in allowed})


def _write(path: Path, results, args) -> None:
    summary = summarize(results)
    path.write_text(json.dumps(
        {"summary": summary, "results": [r.to_dict() for r in results]},
        indent=2) + "\n")
    print("\n==== SUMMARY ====")
    for ttype, row in summary["by_type"].items():
        bar = row.get("opus_published_mean_speedup")
        verdict = ""
        if bar is not None:
            verdict = ("  BEATS Opus" if row.get("beats_opus")
                       else f"  (Opus {bar}x)")
        ms = row["mean_speedup"]
        print(f"  {ttype:<18} n={row['n']:<4} correct={row['correct']:<4} "
              f"mean_speedup={ms if ms is None else round(ms, 3)}{verdict}")
    print(f"\nwrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("discover", "baseline", "run", "merge",
                                     "baseline-merge"))
    ap.add_argument("--arena-root", default=str(DEFAULT_ARENA))
    ap.add_argument("--gpu-arch", default="gfx950")
    ap.add_argument("--types", nargs="*", default=[],
                    help="task types, e.g. triton2triton hip2hip torch2hip")
    ap.add_argument("--out", default="runs/aka")
    ap.add_argument("--arm", default="kore")
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=0)
    # AKA allows 3600s per gate (src/evaluator.py, src/performance.py) and no task
    # overrides it. At 900s we were failing exactly the expensive ones -- cpp
    # extension JIT builds, rocPRIM CMake, multi-shape attention benchmarks -- and
    # a compile timeout scores 0 rather than 20, so the penalty lands twice.
    ap.add_argument("--timeout", type=int, default=3600)
    # 8192 truncates real answers. A smoke run caught a reply cut off at 27,481
    # characters -- almost exactly 8192 tokens -- mid-kernel, which then fails to
    # compile and is scored as though the model wrote a broken kernel. A .hip
    # translation unit carrying its own launcher and pybind bindings routinely
    # runs past that, and instruction2triton files that must be reproduced whole
    # are larger still.
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.0)
    # Matches AKA's reference agents (agents/*/agent_config.yaml: max_iterations:
    # 3). Comparing a single shot against published numbers from a 3-iteration
    # agentic loop with compiler feedback measures two different procedures, not
    # two models. Set 1 for a deliberate single-shot measurement.
    ap.add_argument("--attempts", type=int, default=3,
                    help="generation attempts per task, with harness feedback "
                         "between them (AKA reference agents use 3)")
    ap.add_argument("--task-concurrency", type=int, default=4,
                    help="tasks in flight per worker. Each task's attempts stay "
                         "sequential; overlapping tasks is what lets vLLM batch, "
                         "and at 1 every generate call is a batch of one")
    ap.add_argument("--keep-workspaces", action="store_true")
    ap.add_argument("--json-out", default="")
    # Task-level sharding. transformers loads this checkpoint with
    # device_map="auto", and a 30B in bf16 is ~60GB against 192GB of HBM per
    # MI350X, so the whole model lands on one GPU and the other seven idle --
    # 12.5% of the node for a sweep measured in tens of hours. Tasks are
    # independent and each already has a durable ledger entry, so the way to use
    # the node is one worker per GPU on a disjoint slice.
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of parallel workers (one per GPU)")
    ap.add_argument("--shard", type=int, default=0,
                    help="which slice this worker owns, 0-based")
    args = ap.parse_args()
    if args.num_shards < 1:
        ap.error("--num-shards must be >= 1")
    if not 0 <= args.shard < args.num_shards:
        ap.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
    return {"discover": cmd_discover, "baseline": cmd_baseline,
            "run": cmd_run, "merge": cmd_merge,
            "baseline-merge": cmd_baseline_merge}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
