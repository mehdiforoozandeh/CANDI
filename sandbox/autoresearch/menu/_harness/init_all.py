"""Initialise all 5 loops from the candi_v2 anchor (the G1 run). Each loop's iter-0 = the same
row-0 adapter (the common origin); program.md comes from build_programs.py. Idempotent: skips a loop
that is already initialised.

  python init_all.py            # uses _adapter/_g1.out for the anchor score/aspects
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

MENU = HERE.parent
ROW0 = MENU / "_adapter" / "train_v2_row0.py"
G1_OUT = MENU / "_adapter" / "_g1.out"
LOOPS = ["single_lambda", "crps_calibration", "factorized", "axial_longrange", "repr_first"]


def main() -> int:
    score, aspects, status = driver.parse_footer(G1_OUT.read_text())
    if status != "ok":
        print(f"[init_all] anchor run not OK ({status}) — refusing to init"); return 1
    print(f"[init_all] v2 anchor ERA_SCORE={score:.4f}  Q_imp={aspects.get('Q_imp')}  dcr={aspects.get('dcr')}")
    for tag in LOOPS:
        loop = MENU / tag
        if (loop / "results.tsv").exists():
            print(f"  {tag}: already initialised, skip"); continue
        prog = (loop / "program.md").read_text() if (loop / "program.md").exists() else f"# {tag}\n"
        driver.init_loop(loop, ROW0, prog, score, aspects)
        print(f"  {tag}: initialised (base committed, best tag set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
