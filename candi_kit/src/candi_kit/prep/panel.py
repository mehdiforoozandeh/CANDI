"""The bake panel: the single place scale is declared, and the only config file the kit reads."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Tuple

_PREFIXES = ("T_", "V_", "B_")


@dataclass(frozen=True)
class Panel:
    assays: Tuple[str, ...]
    biosamples: Tuple[str, ...]
    dsf_list: Tuple[int, ...] = (1, 2, 4, 8)
    resolution: int = 25
    context_bins: int = 768
    train_chroms: Tuple[str, ...] = ("chr19",)
    eval_chroms: Tuple[str, ...] = ("chr21",)

    def __post_init__(self):
        if not self.assays:
            raise ValueError("panel.assays is empty")
        if len(set(self.assays)) != len(self.assays):
            raise ValueError(f"panel.assays has duplicates: {sorted(self.assays)}")
        bad = [b for b in self.biosamples if not b.startswith(_PREFIXES)]
        if bad:
            raise ValueError(f"panel.biosamples must start with T_/V_/B_: {bad}")
        n_train = sum(1 for b in self.biosamples if b.startswith("T_"))
        if n_train < 2:
            raise ValueError(
                f"need >=2 T_ biosamples for whole-assay cloze masking, got {n_train}")
        if self.context_bins % 8 != 0:
            raise ValueError(
                f"context_bins must be divisible by pool_size**n_cnn_layers = 8, got {self.context_bins}")
        root = int(math.isqrt(self.resolution))
        if root * root != self.resolution:
            raise ValueError(f"resolution must be a perfect square (dna_pool_size**2), got {self.resolution}")
        overlap = sorted(set(self.train_chroms) & set(self.eval_chroms))
        if overlap:
            raise ValueError(f"train_chroms and eval_chroms overlap: {overlap}")
        if len(self.dsf_list) < 2:
            print(
                f"WARNING: dsf_list={list(self.dsf_list)} has a single level -> x_dsf != y_dsf is inert "
                "and the depth-steering signal the recipe rests on is absent",
                flush=True)


def load_panel(path: Path) -> Panel:
    """Load a panel JSON. Unknown keys are REJECTED — a typo'd knob must not be silently ignored.

    Exception: keys beginning with `_` are ignored as user comments. JSON has no comment syntax, and a
    panel is exactly the kind of file people want to annotate ("why these biosamples?"). Reserving the
    `_` prefix keeps that possible without weakening the typo check, since no real knob starts with one.
    """
    raw = json.loads(Path(path).read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    known = {f.name for f in fields(Panel)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"unknown panel keys {unknown}; allowed: {sorted(known)} "
            "(prefix a key with '_' to keep it as a comment)")
    return Panel(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
