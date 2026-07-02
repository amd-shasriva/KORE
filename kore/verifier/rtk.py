"""RTK shim: identity passthrough (KORE reads tool output programmatically)."""
from __future__ import annotations
from typing import Sequence

def smart_wrap(command: Sequence[str]) -> list[str]:
    return list(command)
