#!/usr/bin/env python3
"""Hold the nodes that free up soonest, not whichever ones came to hand.

A reservation on a busy node is a claim on its future: the flag is IGNORE_JOBS,
so the job running there is untouched and the node comes to us when it ends
rather than going back to the pool. That makes *which* node we claim entirely a
question of how long its current job has left -- and picking without looking is
how this reservation ended up holding two nodes whose jobs had 26 days to run
while nodes finishing in two hours went to someone else.

The scheduler does not expose time-left (``%L`` returns "?") or an end time, so
remaining is computed as ``TimeLimit - Elapsed``. Jobs with no limit are skipped
rather than guessed at: an UNLIMITED job may end in a minute or never, and
claiming its node is exactly the bet that already failed.

Nodes running our own jobs are kept regardless of their clock -- releasing one
would hand away capacity we are actively using.

    python scripts/claim_soonest_nodes.py --want 6
    python scripts/claim_soonest_nodes.py --want 6 --dry-run
"""

from __future__ import annotations

import argparse
import getpass
import re
import subprocess

NODE_RE = re.compile(r"^crsuse2-m2m-\d+$")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120).stdout
    except Exception:
        return ""


def parse_duration(text: str) -> int | None:
    """Slurm duration to seconds; None when there is no finite limit."""
    text = (text or "").strip()
    if not text or text.upper() in {"UNLIMITED", "INFINITE", "?", "N/A"}:
        return None
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def reserved_nodes() -> set[str]:
    out = _run(["scontrol", "show", "reservation"])
    nodes: set[str] = set()
    for m in re.finditer(r"Nodes=(\S+)", out):
        nodes.update(m.group(1).split(","))
    return {n for n in nodes if n}


def idle_nodes() -> set[str]:
    """Nodes the scheduler calls idle, which on a full cluster means broken."""
    out = _run(["sinfo", "-h", "-N", "-o", "%N %T"])
    return {ln.split()[0] for ln in out.splitlines()
            if len(ln.split()) > 1 and ln.split()[1].startswith("idle")}


def reservation_nodes(name: str) -> list[str]:
    out = _run(["scontrol", "show", "reservation"]).replace("\n", " ")
    for block in out.split("ReservationName=")[1:]:
        if block.split()[0] != name:
            continue
        m = re.search(r"Nodes=(\S+)", block)
        return [n for n in (m.group(1).split(",") if m else []) if n]
    return []


def running() -> list[tuple[int | None, str, str, str]]:
    """(remaining_seconds, node, user, jobid) for every running job."""
    out = _run(["squeue", "-t", "R", "-h", "-o", "%i|%u|%N|%M|%l"])
    rows = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 5:
            continue
        jid, user, node, used, lim = f[:5]
        if not NODE_RE.match(node or ""):
            continue
        used_s, lim_s = parse_duration(used), parse_duration(lim)
        rem = None if (used_s is None or lim_s is None) else max(lim_s - used_s, 0)
        rows.append((rem, node, user, jid))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservation", default="kore_mine")
    ap.add_argument("--want", type=int, default=6,
                    help="how many nodes the reservation should hold")
    ap.add_argument("--min-gain-hours", type=float, default=1.0,
                    help="a swap must shorten the wait by at least this much")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    me = getpass.getuser()
    rows = running()
    mine_nodes = {n for _, n, u, _ in rows if u == me}
    held = reservation_nodes(args.reservation)
    if not held:
        print(f"reservation {args.reservation} not found")
        return 2

    rem_by_node = {n: r for r, n, _, _ in rows}
    # Keep what we are running on; rank the rest by how long they have left,
    # treating "no finite limit" as the worst possible case.
    keep = [n for n in held if n in mine_nodes]
    swappable = sorted(
        (n for n in held if n not in mine_nodes),
        key=lambda n: (rem_by_node.get(n) is None, rem_by_node.get(n) or 0),
    )

    taken = reserved_nodes()
    idle = idle_nodes()
    # A node sitting idle on a full cluster is idle because it cannot launch --
    # three of them took a five-minute hello-world straight to
    # JobLaunchFailure. Claiming one buys a slot that never starts a job.
    fresh = [(r, n) for r, n, u, _ in rows
             if u != me and n not in taken and r is not None
             and r > 0 and n not in idle]

    # Rank what we already hold against what is available in one list. Ranking
    # only the fresh candidates and swapping blind is worse than not swapping:
    # the first version of this proposed trading four nodes that free in 2-4h
    # for four that free in 7-9h.
    pool = fresh + [(rem_by_node[n], n) for n in swappable
                    if rem_by_node.get(n) is not None and n not in idle]
    pool.sort(key=lambda t: t[0])

    slots = max(args.want - len(keep), 0)
    want_nodes = [n for _, n in pool[:slots]]

    def hours(n: str) -> str:
        r = rem_by_node.get(n)
        return "no limit" if r is None else f"{r/3600:.1f}h"

    print(f"holding {len(held)} node(s) in {args.reservation}; "
          f"{len(keep)} running our jobs")
    for n in keep:
        print(f"  keep    {n:<20} (ours)")

    add = [n for n in want_nodes if n not in held]
    drop = [n for n in swappable if n not in want_nodes]

    # Hysteresis. "Soonest" moves every minute as clocks run down, so swapping
    # on any improvement would trade a node freeing in 2.0h for one freeing in
    # 1.9h and then back again, churning the reservation without ever
    # shortening the wait. Pair each candidate against the node it displaces,
    # cheapest first, and only make the trade when it buys a real margin.
    if args.min_gain_hours > 0:
        margin = args.min_gain_hours * 3600
        add_ranked = sorted(add, key=lambda n: rem_by_node.get(n) or 0)
        drop_ranked = sorted(
            drop, key=lambda n: (rem_by_node.get(n) is None,
                                 rem_by_node.get(n) or 0), reverse=True)
        paired_add, paired_drop = [], []
        for new, old in zip(add_ranked, drop_ranked):
            old_rem = rem_by_node.get(old)
            new_rem = rem_by_node.get(new) or 0
            if old_rem is None or old_rem - new_rem >= margin:
                paired_add.append(new)
                paired_drop.append(old)
        # Filling an empty slot is always worth it; only swaps need the margin.
        free_slots = slots - len(swappable)
        if free_slots > 0:
            for n in add_ranked:
                if n not in paired_add and len(paired_add) - len(paired_drop) < free_slots:
                    paired_add.append(n)
        add, drop = paired_add, paired_drop

    for n in add:
        print(f"  claim   {n:<20} frees in {hours(n)}")
    for n in drop:
        print(f"  release {n:<20} frees in {hours(n)}")
    if args.dry_run:
        return 0

    for n in add:
        ok = subprocess.run(
            ["scontrol", "update-reservation", "--name", args.reservation,
             "--add-nodes", n], capture_output=True, text=True).returncode == 0
        print(f"  {'claimed' if ok else 'could not claim'} {n}")
    for n in drop:
        subprocess.run(
            ["scontrol", "update-reservation", "--name", args.reservation,
             "--remove-nodes", n], capture_output=True, text=True)
        print(f"  released {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
