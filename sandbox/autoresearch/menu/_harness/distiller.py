"""Distiller — compress older history into backlog.md so the step-N context stays bounded while
nothing is lost. Deterministic (no LLM call): the kept lineage + rejected approaches + the union
of parked ideas. Run by the driver AFTER each iteration, stamping updated_at_iter = that iter.

The context validator asserts backlog.updated_at_iter == N-1 before assembling iter N, so a
missed distiller run is caught rather than silently serving stale memory.
"""
from __future__ import annotations

from pathlib import Path

import journal


def update_backlog(loop: Path, completed_iter: int, keep_recent: int = 8) -> None:
    loop = Path(loop)
    rows = journal.read_results(loop)
    refl = journal.read_reflections(loop)

    kept, rejected = [], []
    cutoff_iter = completed_iter - keep_recent          # older-than-recent gets compressed into lessons
    for r in rows:
        try:
            it = int(r["iter"])
        except (ValueError, KeyError):
            continue
        st = r.get("status", "")
        chg = r.get("change_summary", "").strip()
        sc = r.get("era_score", "")
        d = r.get("d_best", "")
        if st in ("keep", "base"):
            kept.append(f"- iter {it}: {chg}  (era {sc}, Δ {d})")
        elif it <= cutoff_iter and st in ("reset", "crash", "gate"):
            rejected.append(f"- iter {it} [{st}]: {chg}  (era {sc})")

    lessons = "### kept (validated lineage)\n" + ("\n".join(kept) or "- (none yet)") + \
              "\n\n### rejected (do not repeat)\n" + ("\n".join(rejected) or "- (none yet)")

    seen, parked = set(), []
    for b in reversed(refl):                            # most-recent-first, deduped
        for line in (b.get("parked", "") or "").splitlines():
            s = line.strip().lstrip("-• ").strip()
            if s and s.lower() not in seen and s.lower() not in ("none", "n/a", ""):
                seen.add(s.lower())
                parked.append(f"- {s}")
    parked_txt = "\n".join(parked) or "- (none parked)"

    journal.write_backlog(loop, completed_iter, lessons, parked_txt)
