"""Durable per-loop memory: results.tsv (outcomes) + reflections.md (reasoning) + backlog.md.

These files ARE the stateless agent's memory — they are reconstructed into the step-N context
bundle (context.py) and strictly validated before any agent call. Schemas are fixed here so the
context validator can assert them.
"""
from __future__ import annotations

import re
from pathlib import Path

# results.tsv columns (DESIGN_DOC §8); everything after `d_best` comes from the ERA_ASPECTS footer
RESULTS_COLS = [
    "iter", "ts", "commit", "parent", "status", "era_score", "d_best",
    "S_A", "Q_imp", "Q_den", "ece", "dcr", "c_index", "peak_auroc",
    "imp_pval_sp", "imp_pval_pe", "imp_count_sp", "imp_count_pe",
    "steps", "epochs", "sec", "vram_mb", "smoke", "change_summary",
]
REFLECTION_FIELDS = ["hypothesis", "rationale", "expected", "result", "interpretation", "parked"]
STATUSES = {"keep", "reset", "crash", "gate", "base"}


# --------------------------------------------------------------------------- results.tsv
def results_path(loop: Path) -> Path:
    return Path(loop) / "results.tsv"


def append_result(loop: Path, row: dict) -> None:
    p = results_path(loop)
    if not p.exists():
        p.write_text("\t".join(RESULTS_COLS) + "\n")
    vals = [str(row.get(c, "")) for c in RESULTS_COLS]
    with p.open("a") as f:
        f.write("\t".join(vals) + "\n")


def read_results(loop: Path) -> list[dict]:
    p = results_path(loop)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def best_row(loop: Path) -> dict | None:
    rows = [r for r in read_results(loop) if r.get("status") in ("keep", "base")]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["era_score"]))


# --------------------------------------------------------------------------- reflections.md
def reflections_path(loop: Path) -> Path:
    return Path(loop) / "reflections.md"


def append_reflection(loop: Path, it: int, status: str, era_score, d_best, fields: dict) -> None:
    p = reflections_path(loop)
    blk = [f"## iter {it} · {status} · era_score {era_score} (Δbest {d_best})"]
    for k in REFLECTION_FIELDS:
        blk.append(f"- {k}: {fields.get(k, '').strip()}")
    with p.open("a") as f:
        f.write("\n".join(blk) + "\n\n")


def read_reflections(loop: Path) -> list[dict]:
    p = reflections_path(loop)
    if not p.exists():
        return []
    out = []
    for m in re.finditer(r"## iter (\d+) · (\S+) · era_score (\S+) \(Δbest (\S+)\)\n(.*?)(?=\n## iter |\Z)",
                         p.read_text(), re.S):
        body = m.group(5)
        fields = {k: "" for k in REFLECTION_FIELDS}
        for fm in re.finditer(r"- (\w+): ?(.*?)(?=\n- \w+: |\Z)", body, re.S):
            if fm.group(1) in fields:
                fields[fm.group(1)] = fm.group(2).strip()
        out.append({"iter": int(m.group(1)), "status": m.group(2),
                    "era_score": m.group(3), "d_best": m.group(4), **fields})
    return out


def last_reflections(loop: Path, k: int) -> list[dict]:
    return read_reflections(loop)[-k:]


# --------------------------------------------------------------------------- backlog.md
def backlog_path(loop: Path) -> Path:
    return Path(loop) / "backlog.md"


def read_backlog(loop: Path) -> tuple[int, str]:
    """Returns (updated_at_iter, text). -1 if absent/unparseable."""
    p = backlog_path(loop)
    if not p.exists():
        return -1, ""
    txt = p.read_text()
    m = re.search(r"updated_at_iter:\s*(\d+)", txt)
    return (int(m.group(1)) if m else -1), txt


def write_backlog(loop: Path, updated_at_iter: int, lessons: str, parked: str) -> None:
    backlog_path(loop).write_text(
        f"# backlog (distilled)\nupdated_at_iter: {updated_at_iter}\n\n"
        f"## lessons so far\n{lessons.rstrip()}\n\n## parked ideas\n{parked.rstrip()}\n")
