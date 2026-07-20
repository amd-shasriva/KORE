#!/usr/bin/env python3
"""Pick idle GPUs and report them as HIP/torch indices (what HIP_VISIBLE_DEVICES wants).

On this node (MI350X) rocm-smi's physical GPU indices and torch/HIP's device indices
are enumerated in a DIFFERENT order, so masking with rocm-smi's "physical" ids lands
the model on the wrong GPUs (e.g. HIP_VISIBLE_DEVICES=0,2,5 actually selected the
factory's physical GPUs 3,2,4). We therefore join rocm-smi (physical util / VRAM /
PCI-bus) to torch (HIP index / PCI-bus) on the PCI bus id, so the HIP indices we emit
truly correspond to the idle *physical* GPUs.

Env:
  SFT_UTIL_MAX     max GPU-use %% to count as idle           (default 20)
  SFT_VRAM_MAX_GB  max VRAM used (GB) to count as idle        (default 8)
  GATE_GPUS        preferred PHYSICAL (rocm-smi) ids, csv;    (default "")
                   intersected with the idle set, order kept
  GATE_NGPU        cap on how many GPUs to return             (default 3)

Prints ONE line:  "<hip_csv>\t<phys_csv>"  (both empty if none idle).
"""
from __future__ import annotations

import os
import re
import subprocess
import time


def _smi(args: list[str]) -> str:
    try:
        return subprocess.run(["rocm-smi", *args], capture_output=True,
                              text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return ""


def _bus_byte(pci: str) -> int:
    """'0000:08:00.0' -> 0x08 (the bus byte, matching torch's pci_bus_id)."""
    try:
        return int(pci.split(":")[1], 16)
    except Exception:  # noqa: BLE001
        return -1


def _phys_stats() -> tuple[dict, dict, dict]:
    """Per physical (rocm-smi) GPU: util%%, VRAM-used bytes, bus-byte. Sampled twice
    (max) so a momentarily-idle busy GPU is not mistaken for free."""
    util: dict[int, float] = {}
    vram: dict[int, float] = {}
    bus: dict[int, int] = {}
    for s in range(2):
        if s:
            time.sleep(3)
        u = _smi(["--showuse"])
        m = _smi(["--showmeminfo", "vram"])
        b = _smi(["--showbus"])
        for ln in u.splitlines():
            mm = re.search(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", ln)
            if mm:
                g = int(mm.group(1))
                util[g] = max(util.get(g, 0.0), float(mm.group(2)))
        for ln in m.splitlines():
            mm = re.search(r"GPU\[(\d+)\].*?Used Memory \(B\):\s*(\d+)", ln)
            if mm:
                g = int(mm.group(1))
                vram[g] = max(vram.get(g, 0.0), float(mm.group(2)))
        for ln in b.splitlines():
            mm = re.search(r"GPU\[(\d+)\].*?PCI Bus:\s*([0-9A-Fa-f:\.]+)", ln)
            if mm:
                bus[int(mm.group(1))] = _bus_byte(mm.group(2))
    return util, vram, bus


def _hip_bus_map() -> dict[int, int]:
    """{pci-bus-byte: hip/torch index} from torch (all GPUs visible)."""
    import torch  # heavy import, done unmasked so we see every GPU
    out: dict[int, int] = {}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out[int(getattr(p, "pci_bus_id", -1))] = i
    return out


def main() -> int:
    util_max = float(os.environ.get("SFT_UTIL_MAX", "20"))
    vram_max = float(os.environ.get("SFT_VRAM_MAX_GB", "8")) * 1e9
    ngpu = int(os.environ.get("GATE_NGPU", "3"))
    pref = [int(x) for x in os.environ.get("GATE_GPUS", "").split(",") if x.strip() != ""]

    util, vram, bus = _phys_stats()
    hipmap = _hip_bus_map()

    idle: list[tuple[int, int]] = []  # (physical_id, hip_index)
    for g in sorted(util):
        if util.get(g, 100.0) <= util_max and vram.get(g, 9e12) <= vram_max:
            hip = hipmap.get(bus.get(g, -1))
            if hip is not None:
                idle.append((g, hip))

    if pref:  # keep only preferred physical GPUs that are actually idle, in pref order
        rank = {g: i for i, g in enumerate(pref)}
        idle = sorted((x for x in idle if x[0] in rank), key=lambda x: rank[x[0]])

    idle = idle[:ngpu]
    hip_csv = ",".join(str(h) for _, h in idle)
    phys_csv = ",".join(str(g) for g, _ in idle)
    print(f"{hip_csv}\t{phys_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
