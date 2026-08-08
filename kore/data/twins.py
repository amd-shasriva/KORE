"""Which source tasks already have a twin, across every output root.

A twin is a task dir re-expressed in another language: the same oracle and
driver, a task.yaml naming the new backend, and a seed the teacher writes. The
materializers keep a ``seed_attempts.jsonl`` ledger so a killed sweep resumes
where it stopped, but that ledger lives inside the run's own ``--out``.

That scoping is the bug this module exists to close. Two roots twinning the
same source into the same language cannot see each other, so aiming a fresh
``--out`` at a source a previous run already swept restarts it from the first
task and every teacher call rewrites a file that is already on disk. Measured
on the frontier HIP root: 514 of 514 tasks it seeded were already materialized
under data/pool_hip.

Twins are counted from the directories rather than from the ledgers, for two
reasons. The directory is the artifact -- a ledger line whose task dir was
rolled back is not a twin, and a twin whose ledger line was lost to a torn
write still is. And the suffix names the language, so a FlyDSL twin is never
mistaken for a HIP one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def read_task_cfg(task_dir: Path) -> dict:
    """A task's task.yaml, whichever dialect of it the task speaks.

    Generated pool tasks write JSON; hand-authored registry tasks write real
    YAML with nested shape maps and comments. ``json.loads`` on the latter
    raises, which is what stopped the twin path at the registry boundary.
    """
    text = (task_dir / "task.yaml").read_text(errors="ignore")
    if text.lstrip().startswith("{"):
        return json.loads(text)
    import yaml  # noqa: PLC0415 - only registry tasks need it

    return yaml.safe_load(text) or {}

#: Suffix a twin's directory carries, per backend. Order matters within a
#: backend: the longest suffix is tried first so that ``x__hipf`` is read as
#: task ``x`` twinned functionally, not as task ``x_`` twinned as ``__hip``.
TWIN_SUFFIXES: dict[str, tuple[str, ...]] = {
    "hip": ("__hipf", "__hip"),
    "flydsl": ("__flydsl",),
}


def extract_code(reply: str, must_contain: str = "") -> str:
    """The candidate file out of a teacher reply, not merely the first block.

    Taking the first fenced block assumes the reply is one. It stopped being
    one: after the FlyDSL prompt grew an API listing, the model began answering
    with a short illustrative block -- an import line, a layout sketch -- before
    the actual kernel, and the first-block rule handed back the sketch. The
    reply was fine, 6k to 22k characters and never truncated, and 199 of 298
    ports were still discarded for "no @flyc.jit launch wrapper".

    So: among the fenced blocks, prefer the ones that contain the marker the
    caller is going to check for anyway, and take the longest of those. A file
    that defines the entry point is what was asked for, and it is essentially
    always the longest block in the reply.
    """
    blocks = re.findall(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```", reply, re.S)
    if not blocks:
        return reply.strip()
    if must_contain:
        marked = [b for b in blocks if must_contain in b]
        if marked:
            return max(marked, key=len).strip()
    return max(blocks, key=len).strip()


def read_task_list(path: Path) -> set[str]:
    """Task ids from a selection file, one per line, ``#`` for comments.

    A source root is not a work list. kore/tasks holds 1,549 task dirs and only
    482 of them are frontier; the rest are generated elementwise and reduction
    ops -- gen_abs, gelu_tanh, gen_add_add_relu -- which is precisely the
    launch-bound work the frontier selection exists to skip. Walking the root
    directly takes them in name order, so both registry streams spent their
    first thousand teacher calls there: 66% of the twins they had produced were
    off-target.
    """
    wanted = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            wanted.add(line)
    return wanted


def registry_spec(task_dir: Path) -> dict:
    """Synthesize the pool's spec shape for a hand-authored registry task.

    The two task kinds describe themselves differently. A pool task embeds a
    JSON ``_SPEC`` whose ``module_source`` is a PyTorch nn.Module and whose
    ``entry_class`` names it. A registry task's reference.py *is* the oracle --
    plain Python defining ``ref_fn`` and ``get_inputs``, often importing AITER
    -- and its shapes live in task.yaml.

    That difference is why the frontier set was unreachable: flash attention,
    fused MoE and fp8 GEMM are all registry tasks, so a twin path that only
    understood the embedded ``_SPEC`` could see nothing but the external pool,
    whose median baseline is 17us.

    ``entry_class`` is deliberately omitted. There is no nn.Module to
    instantiate, so functionalization must not run on one -- and it does not
    need to: functionalization exists to turn a module's hidden parameters into
    explicit arguments, and a registry task is already a pure function of its
    declared inputs.
    """
    cfg = read_task_cfg(task_dir)
    source = (task_dir / "reference.py").read_text(errors="ignore")
    shapes = (cfg.get("shapes") or {}).get("primary") or {}
    dims = [v for v in shapes.values() if isinstance(v, int) and v > 0]
    scale = 1
    for d in dims:
        scale *= d
    targets = cfg.get("targets") or {}
    return {
        "module_source": source,
        "entry_name": cfg.get("operation") or task_dir.name,
        "dtype": cfg.get("dtype") or "fp32",
        "snr_threshold": cfg.get("snr_threshold") or targets.get("snr_db") or 30,
        "family": cfg.get("op_family") or cfg.get("taxonomy_family") or "registry",
        "primary_scale": scale or "a larger size",
        # One entry per declared dimension is wrong as an arity and right as a
        # hint: the prompt only uses it for an example shape, and the true
        # signature is visible in module_source, which is the whole reference.
        "input_specs": [{"shape": [d for d in dims] or [1]}],
        "registry_task": True,
        "task_id": cfg.get("task_id") or task_dir.name,
    }


def spec_of(task_dir: Path) -> dict:
    """The spec for a task, whichever kind it is.

    Pool tasks carry an embedded JSON ``_SPEC``; registry tasks are adapted.
    Falling back rather than branching on the source root keeps a mixed
    ``--source-root`` working and keeps the caller from having to know.

    Shared by both dialects on purpose. When only the HIP path knew how to read
    a registry task, HIP twinned the frontier and FlyDSL silently could not --
    it raised on every registry reference.py and was left porting the pool,
    which is the launch-bound half of the corpus.
    """
    text = (task_dir / "reference.py").read_text(errors="ignore")
    start = text.find('_SPEC = json.loads("')
    if start < 0:
        return registry_spec(task_dir)
    literal_start = text.index('"', start + len("_SPEC = json.loads"))
    literal_end = text.index('")', literal_start)
    return json.loads(json.loads(text[literal_start:literal_end + 1]))


def mark_exhausted(out_root: Path, selected: int, examined: int) -> None:
    """Record whether a root still has anything to seed.

    A sweep that selects nothing is not free. Deciding a pool task is
    HIP-eligible means running its module to check that the weights can be
    supplied from outside, so an empty sweep still pays that per task -- ~90s
    of CPU to conclude there is no work -- and the pipeline restarts a finished
    materializer every pass. The marker lets the caller skip a settled root;
    it is removed as soon as there is work again, so it can never wedge one
    shut.
    """
    marker = out_root / ".exhausted"
    if selected:
        marker.unlink(missing_ok=True)
        return
    out_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"examined": examined, "selected": 0,
                    "at": _now()}) + "\n")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def existing_twins(suffixes, data_dir: Path) -> set[str]:
    """Source task ids that already have a twin with one of ``suffixes``.

    Every ``<root>/tasks`` directory under ``data_dir`` is scanned, so a task
    counts as twinned no matter which run produced it.
    """
    suffixes = tuple(sorted(suffixes, key=len, reverse=True))
    seen: set[str] = set()
    if not suffixes or not data_dir.is_dir():
        return seen
    for tasks_dir in sorted(data_dir.glob("*/tasks")):
        if not tasks_dir.is_dir():
            continue
        for entry in tasks_dir.iterdir():
            name = entry.name
            for suffix in suffixes:
                if name.endswith(suffix) and entry.is_dir():
                    seen.add(name[: -len(suffix)])
                    break
    return seen
