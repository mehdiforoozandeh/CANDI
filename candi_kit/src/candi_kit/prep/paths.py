"""Side files the bake needs, as explicit validated arguments instead of repo-root-relative paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SideFiles:
    chrom_sizes: Path
    fasta: Path
    blacklist: Optional[Path] = None
    ccres: Optional[Path] = None

    def validate(self, *, need_ccres: bool) -> None:
        for name in ("chrom_sizes", "fasta"):
            p = getattr(self, name)
            if not Path(p).is_file():
                raise FileNotFoundError(f"--{name.replace('_', '-')} is not a file: {p}")
        fai = Path(str(self.fasta) + ".fai")
        if not fai.is_file() and not os.access(Path(self.fasta).parent, os.W_OK):
            raise RuntimeError(f"missing {fai} and its directory is not writable (pysam cannot index)")
        if need_ccres and self.ccres is None:
            raise ValueError("--ccres is required when type2_ccre > 0 or type2_non > 0")
        if self.ccres is not None and not Path(self.ccres).is_file():
            raise FileNotFoundError(f"--ccres is not a file: {self.ccres}")
        n = _interval_count(self.blacklist) if self.blacklist is not None else 0
        if n == 0:
            print(f"WARNING: blacklist empty ({self.blacklist}) -> 0 regions excluded", flush=True)


def _interval_count(path: Path) -> int:
    if not Path(path).is_file():
        raise FileNotFoundError(f"--blacklist is not a file: {path}")
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n
