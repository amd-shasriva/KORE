#!/usr/bin/env python3
"""Watch for capacity the moment it appears and take it, rather than betting on
when a particular node will free.

Holding named nodes and waiting for their jobs to end is a forecast, and the
forecast is wrong often enough to hurt: a 30-day limit can end in an hour, a
4-hour job gets extended, and a node this reservation was counting on turns out
to be running something with 627 hours left. Watching for the transition
instead needs no forecast at all.

Two kinds of capacity, and the second is the valuable one:

  idle    -- free and contested. Every other user's queued job wants it too,
             and on this cluster it is gone in under a minute: resuming 13
             nodes put 7 idle on the board and 69 queued burst jobs had taken
             all of them before the next poll.

  drained -- free and *uncontested*. The node's prolog or epilog hook failed
             and the scheduler pulled it from service, so no one else's job can
             land on it however long it sits there. Reserve it first and then
             resume it and the capacity is ours alone. Resuming before
             reserving is what handed those 13 nodes to everybody else.

So: reserve, then resume, never the other way round. The hold is capped at what
the job limit lets us actually run, because a reserved node we cannot use is a
node nobody can use.
"""

from __future__ import annotations

import argparse
import getpass
import re
import subprocess
import sys
import time

NODE_RE = re.compile(r"^crsuse2-m2m-\d+$")
DEAD_STATES = ("drain", "down", "fail")

#: The only QoS this account may submit under. amd-primus-qos refuses us
#: outright -- "QOS 'amd-primus-qos' is not permitted for user 'shasriva'
#: under account 'amd-general'" -- and the same is true of collectives,
#: aifw-dev, silo-tiger, infiniai and hyperloom.
#:
#: This governs what we may *submit* under. It deliberately does NOT govern
#: which nodes we may claim, and an earlier version of this file had that
#: wrong: it refused to hold any node whose current job belonged to another
#: pool, on the theory that pools own nodes.
#:
#: They do not. Measured on the live cluster, single nodes host jobs from two
#: pools at once -- m2m-012 runs burst alongside aifw-aim, m2m-030 runs spur
#: alongside burst, m2m-238 runs burst alongside aac. A burst job and a
#: foreign-pool job coexist on the same hardware, so the pool of whatever
#: happens to be running there says nothing about whether we can use it next.
#: Filtering on it threw away a node 25 minutes from freeing in favour of
#: nodes 28 hours out.
USABLE_QOS = ("amd-burst-qos", "amd-general-qos")


def node_pools() -> dict[str, str]:
    """Node -> QoS of the job on it now, which is the pool it belongs to."""
    out = sh(["squeue", "-t", "R", "-h", "-o", "%N %q"])
    pools: dict[str, str] = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and NODE_RE.match(p[0]):
            pools[p[0]] = p[1]
    return pools


def sh(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""


def node_states() -> dict[str, tuple[str, int]]:
    """(state, allocated_cpus) per node.

    State alone is not enough to know a node is takeable. m2m-012 reported
    State=IDLE with CPUAlloc=1 -- a one-CPU job someone else was running -- and
    every miner here asks for --exclusive, which needs the whole node. The
    scheduler answered Reason=Resources against a node that looked free in
    every summary view. A node is only worth claiming when nothing at all is
    allocated on it.
    """
    out = sh(["sinfo", "-h", "-N", "-o", "%N %T"])
    states: dict[str, tuple[str, int]] = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) < 2 or not NODE_RE.match(p[0]):
            continue
        states[p[0]] = (p[1].lower().rstrip("*~#$@+"), -1)
    return states


def cpus_allocated(node: str) -> int:
    """Allocated CPUs on one node, or -1 when it cannot be read.

    Has to come from ``scontrol show node``: this scheduler answers "?" for
    sinfo's %C, so the obvious cheap version of this check silently passed
    every node and claimed one running somebody else's single-CPU job.

    Unknown counts as busy. Spending a hold slot on a node we cannot use costs
    more than skipping a node we could have.
    """
    out = sh(["scontrol", "show", "node", node], timeout=30)
    m = re.search(r"CPUAlloc=(\d+)", out)
    return int(m.group(1)) if m else -1


def reservation_block(text: str, name: str) -> list[str]:
    """Nodes of one reservation, parsed from an already-fetched dump."""
    out = text.replace("\n", " ")
    for block in out.split("ReservationName=")[1:]:
        if block.split()[0] != name:
            continue
        m = re.search(r"Nodes=(\S+)", block)
        return [n for n in (m.group(1).split(",") if m else []) if n]
    return []


def running_nodes(user: str) -> set[str]:
    """Nodes currently running one of our jobs -- never evict these."""
    out = sh(["squeue", "-u", user, "-h", "-t", "R", "-o", "%N"])
    return {n.strip() for n in out.split() if n.strip()}


def pending_miner_ids(user: str) -> list[str]:
    """Queued mining jobs, which are the ones safe to drop and resubmit."""
    out = sh(["squeue", "-u", user, "-h", "-t", "PD", "-o", "%i %j"])
    ids = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1].startswith("kore-mine-"):
            ids.append(p[0])
    return ids


def my_jobs(user: str) -> tuple[int, int]:
    out = sh(["squeue", "-u", user, "-h", "-o", "%T"])
    states = [s.strip() for s in out.splitlines() if s.strip()]
    return (sum(1 for s in states if s == "RUNNING"),
            sum(1 for s in states if s == "PENDING"))


def add_to_reservation(name: str, node: str) -> bool:
    r = subprocess.run(
        ["scontrol", "update-reservation", "--name", name, "--add-nodes", node],
        capture_output=True, text=True)
    return r.returncode == 0


def resume(node: str) -> bool:
    r = subprocess.run(["scontrol", "update", f"NodeName={node}", "State=RESUME"],
                       capture_output=True, text=True)
    return r.returncode == 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reservation", default="kore_mine")
    ap.add_argument("--max-hold", type=int, default=8,
                    help="never hold more nodes than we can run jobs on")
    ap.add_argument("--poll", type=float, default=1.0,
                    help="seconds between polls; one sinfo + one scontrol each")
    ap.add_argument("--staff-cmd", default="bash scripts/staff_datagen.sh")
    ap.add_argument("--staff-cooldown", type=float, default=120.0,
                    help="minimum seconds between staffing passes")
    ap.add_argument("--rebind-cooldown", type=float, default=300.0,
                    help="seconds between rebinding queued miners")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    user = getpass.getuser()
    log(f"scavenger up: reservation={args.reservation} max_hold={args.max_hold} "
        f"poll={args.poll}s")
    last_rebind = 0.0
    last_staff = 0.0
    # node -> the pool we last saw running on it; an idle node keeps its
    # last known pool, which is the only way to tell whose idle capacity it is.
    pool_seen: dict[str, str] = {}

    while True:
        # One sinfo and one scontrol per poll in the common case. The first
        # version read every reservation twice and then asked scontrol for the
        # CPU count of every held node on every pass -- about ten calls a
        # second at a one-second interval, which is a denial of service aimed
        # at the thing we depend on. The expensive per-node checks now happen
        # only when the cheap pass says there is something worth checking.
        states = node_states()
        pool_seen.update(node_pools())
        resv_text = sh(["scontrol", "show", "reservation"])
        # Tell "the scheduler did not answer" apart from "the reservation is
        # not there". Both look like an empty string, and conflating them cost
        # 45 consecutive polls reporting the hold as gone while it sat intact:
        # the daemon had been started without /etc/profile.d/spur.sh, so it had
        # no SPUR_CONTROLLER_ADDR and every scontrol call failed to connect.
        if not resv_text.strip():
            log("scontrol returned nothing -- is SPUR_CONTROLLER_ADDR set? "
                "(source /etc/profile.d/spur.sh before starting this)")
            time.sleep(max(args.poll, 5.0))
            continue
        held = reservation_block(resv_text, args.reservation)
        if not held and not args.once:
            log(f"reservation {args.reservation} is gone; waiting")
            time.sleep(max(args.poll, 5.0))
            continue

        taken = {n for m in re.finditer(r"Nodes=(\S+)", resv_text)
                 for n in m.group(1).split(",") if n}
        room = args.max_hold - len(held)
        grabbed: list[str] = []

        # Trade up when the hold is full. Holding the cap is not the goal --
        # holding nodes we can run on is. Right now three of our eight are
        # unusable: two carry a neighbour's one-CPU job that pins all eight
        # GPUs, and one is draining. With room at zero the scavenger would skip
        # a genuinely free node cluster-wide rather than swap out any of them.
        #
        # A node that is free this second strictly dominates one occupied by
        # somebody else's job, whenever the second is not running work of ours.
        if room <= 0:
            usable_now = [n for n, (s, _) in states.items()
                          if n not in taken
                          and s.startswith(("idle",) + DEAD_STATES)
                          and pool_seen.get(n) in USABLE_QOS
                          and cpus_allocated(n) == 0]
            if usable_now:
                mine_running = {n for n, (s, _) in states.items()
                                if n in held and cpus_allocated(n) > 0
                                and n in running_nodes(user)}
                evictable = [n for n in held if n not in mine_running]
                if evictable:
                    drop = evictable[0]
                    if subprocess.run(
                            ["scontrol", "update-reservation", "--name",
                             args.reservation, "--remove-nodes", drop],
                            capture_output=True, text=True).returncode == 0:
                        log(f"hold full; released {drop} (busy) to make room "
                            f"for a node free right now")
                        room += 1

        if room > 0:
            # Dead nodes first: nobody else can take them, so they are ours for
            # the cost of a reserve plus a resume. Idle nodes second, and only
            # because we may win the race; usually we will not.
            dead = [n for n, (s, _) in states.items()
                    if s.startswith(DEAD_STATES) and n not in taken]
            idle = [n for n, (s, _) in states.items()
                    if s.startswith("idle") and n not in taken]
            for node in dead + idle:
                if room <= 0:
                    break
                # --exclusive needs the whole node, so anything allocated at
                # all disqualifies it. Checked per candidate rather than for
                # the cluster, because it costs a scontrol call each.
                if cpus_allocated(node) != 0:
                    continue
                if not add_to_reservation(args.reservation, node):
                    continue
                room -= 1
                grabbed.append(node)
                # Reserve first, resume second. The other order publishes the
                # node to every queued job on the cluster.
                if states[node][0].startswith(DEAD_STATES):
                    ok = resume(node)
                    log(f"claimed dead node {node} ({states[node][0]}) "
                        f"-> resume {'ok' if ok else 'FAILED'}")
                else:
                    log(f"claimed idle node {node}")

        # Staff only when something changed or a held node is sitting free --
        # the staffing pass is not cheap and calling it every poll would spend
        # more time partitioning shards than mining them.
        # Held nodes worth a closer look: anything not plainly running
        # something. Only these cost a scontrol call.
        maybe_free = [n for n in reservation_block(resv_text, args.reservation)
                      if states.get(n, ("", -1))[0].startswith(("idle", "resv"))]
        free_held = [n for n in maybe_free if cpus_allocated(n) == 0]

        # Revive nodes we already hold. Resuming only happened at claim time,
        # which misses the case that matters most: a held node whose job ends
        # badly goes DRAINED and then sits inside our own reservation doing
        # nothing, uncontested and unusable, until somebody notices. m2m-212
        # was draining while we were queueing miners for it. Nobody else can
        # take a node of ours, so there is no race here -- just resume it.
        for node in held:
            state = states.get(node, ("", -1))[0]
            if state.startswith(DEAD_STATES) and cpus_allocated(node) == 0:
                ok = resume(node)
                log(f"held node {node} was {state}; resume "
                    f"{'ok' if ok else 'FAILED'}")

        # Staffing has to be rate limited independently of the poll. Polling at
        # one second is right for spotting a node; running a staffing pass at
        # one second is not, and the two were tied together: with a held node
        # free it called staff_datagen 58 times in a minute, and because each
        # pass can cancel and resubmit the queued miners, the jobs were being
        # thrown back into the queue faster than the scheduler could ever place
        # them. Two genuinely free nodes sat idle the whole time. Fast to
        # notice, slow to act.
        now = time.time()
        due = (now - last_staff) >= args.staff_cooldown
        if grabbed or (free_held and due):
            running, pending = my_jobs(user)

            # Rebind before staffing. A queued miner is bound to the hold as it
            # stood when it was submitted, and staff_datagen counts that miner
            # as already covering its stream -- so a node grabbed one second
            # ago would sit idle behind four jobs that cannot use it and a
            # staffing pass that thinks there is nothing to do. Dropping them
            # lets the same pass resubmit against the hold we have now.
            #
            # Cooled down because the check cannot tell "cannot use this node"
            # from "has not started yet": without it, a node that stays free
            # for any other reason would have us cancelling four jobs a second.
            if (free_held and pending
                    and now - last_rebind >= args.rebind_cooldown):
                stale = pending_miner_ids(user)
                if stale:
                    log(f"{len(free_held)} held node(s) free with {len(stale)} "
                        f"miner(s) queued on a stale node set -> rebinding")
                    subprocess.run(["scancel", *stale],
                                   capture_output=True, text=True)
                    last_rebind = now
                    pending = 0

            log(f"grabbed={len(grabbed)} free_held={len(free_held)} "
                f"running={running} pending={pending} -> staffing")
            subprocess.run(args.staff_cmd, shell=True,
                           capture_output=True, text=True, timeout=900)
            last_staff = time.time()

        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
