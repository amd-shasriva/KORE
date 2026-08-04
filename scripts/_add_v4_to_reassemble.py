#!/usr/bin/env python3
"""Teach data/release/reassemble.sh about the v4 mixture.

reassemble.sh is what makes the release layout worth having: a fresh checkout
runs it and gets exactly the files the training configs expect, with no network.
Adding v4's parts without adding this line would ship the data and leave the one
step that turns it back into a usable file undocumented.
"""
from __future__ import annotations

import pathlib
import sys

BLOCK = """
# v4: the mixture the 30B SFT config points at -- 244,732 rows, measured
# 65.9/21.5/12.3 kernel/chat/coding by tokens. Rebuilt by concatenation like the
# others so a fresh checkout reproduces it with no network and no hub account.
cat sft/multicap_v4.jsonl.gz.part* | gunzip > ../b05factory/sft/multicap_v4.jsonl
"""


def main() -> int:
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                     else "data/release/reassemble.sh")
    if not p.is_file():
        print(f"missing: {p}", file=sys.stderr)
        return 1
    text = p.read_text()
    if "multicap_v4" in text:
        print("already present")
        return 0
    if "echo reassembled" not in text:
        print("anchor 'echo reassembled' not found", file=sys.stderr)
        return 1
    p.write_text(text.replace("echo reassembled",
                              BLOCK.strip() + "\necho reassembled", 1))
    print("added v4 to reassemble.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
