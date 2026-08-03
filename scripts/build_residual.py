#!/usr/bin/env python
"""Build or verify an instruction-residual transfer over safetensors shards.

    theta_out = theta_target + scale * (theta_instruct - theta_base)

Tensors are streamed one at a time and grouped by the TARGET's shard layout, so
peak memory is a few copies of the largest tensor (the 151936x5120 embedding)
rather than three 28GB models.

Two modes:

  verify  Apply the residual to the base and check it reproduces the instruct
          checkpoint. This is the one input whose answer is known in advance,
          so it catches a sign error, a dtype bug or a mispaired snapshot
          before any of it is baked into a model we then train on. Writes
          nothing.

  build   Materialise the transfer into --out.

Run this under the scheduler, not on the login node: it reads tens of GB, and
the login node routinely sits at load 150+.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def _snapshot(spec: str) -> pathlib.Path:
    """Resolve a local dir or an HF cache repo id to a snapshot directory."""
    p = pathlib.Path(spec)
    if (p / "model.safetensors.index.json").exists():
        return p
    hub = pathlib.Path.home() / ".cache/huggingface/hub"
    hits = sorted(hub.glob(f"models--{spec.replace('/', '--')}/snapshots/*/model.safetensors.index.json"))
    if hits:
        return hits[-1].parent
    raise SystemExit(f"no safetensors index found for {spec!r}")


def _weight_map(d: pathlib.Path) -> dict:
    return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]


def _open_shards(d: pathlib.Path, names):
    from safetensors import safe_open

    return {n: safe_open(str(d / n), framework="pt", device="cpu") for n in sorted(set(names))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("verify", "build"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--target", help="model receiving the residual (build mode)")
    ap.add_argument("--out")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sample", type=int, default=0,
                    help="verify only N tensors (0 = all)")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.policy.residual import ResidualError

    base_d, inst_d = _snapshot(args.base), _snapshot(args.instruct)
    tgt_d = _snapshot(args.target) if args.mode == "build" else base_d
    print(f"base     {base_d}", flush=True)
    print(f"instruct {inst_d}", flush=True)
    print(f"target   {tgt_d}", flush=True)

    wm_b, wm_i, wm_t = _weight_map(base_d), _weight_map(inst_d), _weight_map(tgt_d)
    if set(wm_b) != set(wm_i) or set(wm_b) != set(wm_t):
        raise ResidualError(
            f"tensor-name mismatch: base={len(wm_b)} instruct={len(wm_i)} target={len(wm_t)}; "
            f"only_in_base={sorted(set(wm_b) - set(wm_i))[:5]}"
        )
    names = sorted(wm_t)
    print(f"{len(names)} tensors", flush=True)

    fb = _open_shards(base_d, wm_b.values())
    fi = _open_shards(inst_d, wm_i.values())
    ft = _open_shards(tgt_d, wm_t.values())

    if args.mode == "verify":
        check = names[:: max(1, len(names) // args.sample)] if args.sample else names
        n_exact = n_float = 0
        worst = 0.0
        zero_delta = 0
        rel = []
        t0 = time.time()
        for k in check:
            b = fb[wm_b[k]].get_tensor(k)
            i = fi[wm_i[k]].get_tensor(k)
            if not b.is_floating_point():
                continue
            n_float += 1
            d32 = i.to(torch.float32) - b.to(torch.float32)
            got = (b.to(torch.float32) + d32).to(b.dtype)
            if torch.equal(got, i):
                n_exact += 1
            else:
                worst = max(worst, float((got.to(torch.float32) - i.to(torch.float32)).abs().max()))
            dn, bn = float(d32.norm()), float(b.to(torch.float32).norm())
            if dn == 0.0:
                zero_delta += 1
            if bn > 0:
                rel.append(dn / bn)
        rel.sort()
        print(f"\nchecked {n_float} float tensors in {time.time()-t0:.0f}s")
        print(f"  exact reconstruction : {n_exact}/{n_float}")
        print(f"  max abs diff         : {worst}")
        print(f"  zero-delta tensors   : {zero_delta} (nonzero expected for a real pair)")
        if rel:
            print(f"  ||delta||/||base||   : min={rel[0]:.5f} median={rel[len(rel)//2]:.5f} max={rel[-1]:.5f}")
        ok = (n_exact == n_float) and zero_delta == 0
        print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not args.target or not args.out:
        raise SystemExit("build mode needs --target and --out")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_shard: dict = {}
    for k in names:
        by_shard.setdefault(wm_t[k], []).append(k)
    total = 0
    for shard, keys in sorted(by_shard.items()):
        buf = {}
        for k in keys:
            tv = ft[wm_t[k]].get_tensor(k)
            if not tv.is_floating_point():
                buf[k] = tv.clone()
                continue
            acc = tv.to(torch.float32)
            acc.add_(
                fi[wm_i[k]].get_tensor(k).to(torch.float32)
                - fb[wm_b[k]].get_tensor(k).to(torch.float32),
                alpha=args.scale,
            )
            buf[k] = acc.to(tv.dtype)
            del acc
        save_file(buf, str(out / shard), metadata={"format": "pt"})
        total += len(buf)
        print(f"  wrote {shard} ({len(buf)} tensors)", flush=True)
        del buf
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": wm_t}, indent=2)
    )
    # Config comes from the target, but the TOKENIZER must come from the
    # instruct model: it carries the chat template plus <think>, </think>,
    # <tool_response> and </tool_response>, four special tokens the base
    # tokenizer does not define. The residual already supplies trained
    # embeddings for those rows, so the tokenizer has to be able to emit them.
    import shutil

    for fn in ("config.json", "generation_config.json"):
        if (tgt_d / fn).exists():
            shutil.copy2(tgt_d / fn, out / fn)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "special_tokens_map.json", "added_tokens.json"):
        if (inst_d / fn).exists():
            shutil.copy2(inst_d / fn, out / fn)
    print(f"\nwrote {total} tensors to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
