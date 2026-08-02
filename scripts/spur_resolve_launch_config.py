#!/usr/bin/env python3
"""Resolve a shipped stage config into the launch config a SPUR job actually runs.

The shipped ``configs/<stage>_14b_full.json`` files name the RAW base model
(``Qwen/Qwen3-14B``) because they double as the reference recipe. A real
pipeline run trains the PREVIOUS stage's output instead
(midtrain -> sft -> dpo -> grpo), so ``model_id`` (and usually ``output_dir``)
has to be rewritten before ``kore.policy.<stage>`` sees the file.

``scripts/spur_{sft,dpo,grpo}_1node.sbatch`` all call this resolver, so the
rewrite happens once, in one place, and is identical on a first launch and on a
requeued child. Doing it here rather than inline in each sbatch also makes it
CPU-testable (``tests/test_spur_stage_launchers.py``).

What it does, in order:

1. Reads ``--config`` (JSON).
2. ``--from-stage DIR`` -> ``model_id = DIR``, after checking DIR really is a
   consolidated HF checkpoint (``config.json`` + safetensors). A stale
   ``model_revision`` is deliberately LEFT in place: ``kore.policy.model_spec``
   already ignores it for a local directory and says so in its notes, and
   dropping it would lose the record of which base the lineage started from.
3. ``--output-dir DIR`` -> ``output_dir = DIR``. Keeping this stable across
   requeues is what lets each stage's own ``latest_checkpoint`` /
   ``_discover_grpo_resume`` resume instead of restarting from step 0.
4. ``--data-root DIR`` (grpo only) re-roots the co-evolution artifact paths.
   ``kore.policy.grpo._build_opus_scores`` derives the archive root from
   ``dirname(coevolve_distill_path)``, so leaving the shipped
   ``data/full14b/...`` in place silently points the regret curriculum at a
   different (older) data root than the one the run trains against.
5. For sft/dpo, checks ``dataset_path`` exists. Both stages read the dataset
   after (sft) or near (dpo) a 14B ``from_pretrained``, so a typo otherwise
   costs an 8-rank weight load per rank to report.
6. Parses the result with the stage's own strict loader, so an unknown or
   misspelt key fails here (milliseconds) instead of on eight ranks.
7. Writes ``--out`` atomically and prints every field it changed.

Pass ``-`` for any of ``--from-stage`` / ``--output-dir`` / ``--data-root`` to
leave that part of the config alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

STAGES = ("sft", "dpo", "grpo")

#: Co-evolution artifacts whose directory doubles as the archive root that
#: ``_build_opus_scores`` mines, so they must follow the run's real data root.
_GRPO_DATA_ROOT_KEYS = ("coevolve_distill_path", "coevolve_opus_scores_path")


def _is_unset(value: str | None) -> bool:
    """``-`` (and empty) mean "leave this alone", so a launcher can pass through."""
    return value is None or value == "" or value == "-"


def checkpoint_defects(directory: Path) -> list[str]:
    """Why ``directory`` is not a loadable consolidated HF checkpoint (empty = it is).

    Only the cheap structural facts ``from_pretrained`` needs; the architecture
    itself is verified from the safetensors headers by
    ``kore.policy.model_spec.model_identity_for_config`` at load time.
    """
    if not directory.exists():
        return [f"{directory} does not exist"]
    if not directory.is_dir():
        return [f"{directory} is not a directory"]
    defects = []
    if not (directory / "config.json").is_file():
        defects.append(f"{directory}/config.json is missing")
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        defects.append(f"{directory} holds no *.safetensors weight shard")
    return defects


def _stage_loader(stage: str):
    if stage == "sft":
        from kore.policy.sft import sft_config_from_dict

        return lambda payload: sft_config_from_dict(payload)[0]
    if stage == "dpo":
        from kore.policy.dpo import dpo_config_from_dict

        return dpo_config_from_dict
    from kore.policy.grpo import grpo_config_from_dict

    return grpo_config_from_dict


def resolve(
    stage: str,
    config: dict,
    *,
    from_stage: str | None = None,
    output_dir: str | None = None,
    data_root: str | None = None,
    repo_root: Path | None = None,
) -> tuple[dict, list[str]]:
    """Return ``(resolved_config, changes)``; raise ``ValueError`` on a bad handoff.

    ``repo_root`` anchors the relative paths inside the config (the sbatch jobs
    all ``cd`` to the repo first, so it defaults to the cwd).
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (want one of {'|'.join(STAGES)})")
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    resolved = dict(config)
    changes: list[str] = []

    if not _is_unset(from_stage):
        previous = Path(from_stage)
        defects = checkpoint_defects(previous if previous.is_absolute() else root / previous)
        if defects:
            raise ValueError(
                f"{stage}: --from-stage {from_stage!r} is not a loadable checkpoint: "
                + "; ".join(defects)
                + ". The previous stage must have finished and consolidated its "
                "weights before this one can train them."
            )
        if resolved.get("model_id") != from_stage:
            changes.append(f"model_id: {resolved.get('model_id')!r} -> {from_stage!r}")
            resolved["model_id"] = from_stage

    if not _is_unset(output_dir):
        if resolved.get("output_dir") != output_dir:
            changes.append(f"output_dir: {resolved.get('output_dir')!r} -> {output_dir!r}")
            resolved["output_dir"] = output_dir

    if stage == "grpo" and not _is_unset(data_root):
        for key in _GRPO_DATA_ROOT_KEYS:
            current = resolved.get(key)
            if not current:
                continue
            wanted = str(Path(data_root) / Path(current).name)
            if wanted != current:
                changes.append(f"{key}: {current!r} -> {wanted!r}")
                resolved[key] = wanted

    if stage in ("sft", "dpo"):
        dataset = resolved.get("dataset_path")
        if not dataset:
            raise ValueError(f"{stage}: config has no dataset_path")
        absolute = Path(dataset)
        if not absolute.is_absolute():
            absolute = root / absolute
        if not absolute.is_file():
            raise ValueError(
                f"{stage}: dataset_path {dataset!r} does not exist (looked at "
                f"{absolute}). Run `cd data/release && ./reassemble.sh` to "
                "materialize the packaged corpus."
            )

    # Strict parse LAST, so it validates exactly what will be written out.
    _stage_loader(stage)(dict(resolved))
    return resolved, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", required=True, help="shipped stage config JSON")
    parser.add_argument("--out", required=True, help="where to write the resolved JSON")
    parser.add_argument("--from-stage", default="-",
                        help="previous stage's output dir to train ('-' keeps model_id)")
    parser.add_argument("--output-dir", default="-",
                        help="this stage's output dir ('-' keeps output_dir)")
    parser.add_argument("--data-root", default="-",
                        help="grpo only: re-root the co-evolution artifact paths")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    try:
        resolved, changes = resolve(
            args.stage, config,
            from_stage=args.from_stage,
            output_dir=args.output_dir,
            data_root=args.data_root,
            repo_root=args.repo_root,
        )
    except ValueError as error:
        print(f"FATAL launch-config resolution: {error}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + f".partial.{os.getpid()}")
    staging.write_text(json.dumps(resolved, indent=2) + "\n")
    staging.replace(out)

    print(f"[resolve] stage={args.stage} config={args.config} -> {out}")
    for change in changes or ["(no overrides; config used verbatim)"]:
        print(f"[resolve]   {change}")
    print(f"[resolve] model_id={resolved.get('model_id')!r} "
          f"output_dir={resolved.get('output_dir')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
