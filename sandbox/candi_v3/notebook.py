"""CANDI v3 — live lab notebook (PLAN §5). Written from the ERA `on_node` hook; never edits
the frozen `futs.py`.

Three artifacts, all regenerated per node (small, always consistent, resumable):
  - RESULTS.tsv : one row/candidate — metrics + which iterations PUCT picked it as a parent.
  - tree.mmd    : Mermaid `graph TD` of the search tree (best lineage bold, dead nodes greyed).
  - NOTE.md     : raw per-node reflection log + a distilled "observations" section (the latter
                  is injected back into generation by generate.py; see PLAN §5 feedback loop).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ASPECTS_RE = re.compile(r"ERA_ASPECTS:\s*(\{.*\})")
_REFLECT_RE = re.compile(r"ERA_REFLECTION:\s*(.+)")
FAIL = -1e8

# metric keys parsed from each candidate's ERA_ASPECTS footer
ASPECT_KEYS = ["S_A", "Q_imp", "Q_den", "imp_pval_spearman", "imp_pval_pearson",
               "imp_count_spearman", "imp_count_pearson", "ece", "c_index", "peak_auroc",
               "dcr", "den_pen", "cal_pen", "cidx_pen", "peak_pen", "dcr_pen", "n_imp"]
TSV_COLS = (["iter", "node_index", "parent_index", "status", "ERA_SCORE"]
            + ASPECT_KEYS + ["selected_as_parent_at"])


class Notebook:
    def __init__(self, workdir: Path):
        self.dir = Path(workdir)
        self.runs = self.dir / "runs"
        self.rows: list[dict] = []
        self.selected: dict[int, list[int]] = {}
        self.reflections: list[str] = []

    # -- find a node's run output and parse the aspects/reflection footer ----------------
    def _run_output(self, program: str) -> str:
        for p in sorted(self.runs.glob("cand*/program.py")):
            try:
                if p.read_text() != program:
                    continue
            except OSError:
                continue
            d = p.parent
            outs = sorted(d.glob("slurm-*.out")) + [d / "stdout.log"]
            for o in outs:
                if o.exists():
                    return o.read_text()
        return ""

    def _parse(self, text: str) -> tuple[dict, str]:
        asp = {}
        m = _ASPECTS_RE.findall(text)
        if m:
            try:
                asp = json.loads(m[-1])
            except json.JSONDecodeError:
                pass
        r = _REFLECT_RE.findall(text)
        return asp, (r[-1].strip() if r else "")

    # -- the ERA on_node callback --------------------------------------------------------
    def on_node(self, nd) -> None:
        it = len(self.rows)
        if nd.parent_index is not None:
            self.selected.setdefault(nd.parent_index, []).append(nd.index)
        asp, reflection = self._parse(self._run_output(nd.solution.program))
        self.rows.append({
            "iter": it, "node_index": nd.index, "parent_index": nd.parent_index,
            "status": "ok" if nd.score > FAIL else "fail",
            "ERA_SCORE": round(nd.score, 6) if nd.score > FAIL else nd.score,
            **{k: asp.get(k) for k in ASPECT_KEYS},
        })
        if reflection:
            self.reflections.append(f"node {nd.index} (score {nd.score:.4f}): {reflection}")
        self._write_tsv()
        self._write_mmd(nd)
        self._write_note()

    def _write_tsv(self) -> None:
        lines = ["\t".join(TSV_COLS)]
        for r in self.rows:
            sel = ",".join(map(str, self.selected.get(r["node_index"], []))) or "-"
            vals = ([r["iter"], r["node_index"], r["parent_index"], r["status"], r["ERA_SCORE"]]
                    + [r.get(k) for k in ASPECT_KEYS] + [sel])
            lines.append("\t".join("" if v is None else str(v) for v in vals))
        (self.dir / "RESULTS.tsv").write_text("\n".join(lines) + "\n")

    def _write_mmd(self, nd) -> None:
        best = max((r for r in self.rows if r["status"] == "ok"),
                   key=lambda r: r["ERA_SCORE"], default=None)
        best_lineage = set()
        if best is not None:
            by_idx = {r["node_index"]: r for r in self.rows}
            cur = best["node_index"]
            while cur is not None:
                best_lineage.add(cur)
                cur = by_idx.get(cur, {}).get("parent_index")
        out = ["graph TD"]
        for r in self.rows:
            i = r["node_index"]
            sc = r["ERA_SCORE"]
            lbl = f'n{i}["{i} · {sc:.3f}"]' if r["status"] == "ok" else f'n{i}["{i} · FAIL"]'
            out.append(f"  {lbl}")
            if r["parent_index"] is not None:
                out.append(f"  n{r['parent_index']} --> n{i}")
        out.append("  classDef dead fill:#eee,stroke:#bbb,color:#999;")
        out.append("  classDef best fill:#dfd,stroke:#2a2,stroke-width:2px;")
        dead = [f"n{r['node_index']}" for r in self.rows if r["status"] != "ok"]
        bestn = [f"n{i}" for i in best_lineage]
        if dead:
            out.append(f"  class {','.join(dead)} dead;")
        if bestn:
            out.append(f"  class {','.join(bestn)} best;")
        (self.dir / "tree.mmd").write_text("\n".join(out) + "\n")

    def _write_note(self) -> None:
        ok = [r for r in self.rows if r["status"] == "ok"]
        best = max(ok, key=lambda r: r["ERA_SCORE"], default=None)
        head = ["# CANDI v3 — ERA lab notebook (NOTE.md)", "",
                f"nodes: {len(self.rows)} | valid: {len(ok)} | "
                f"best: node {best['node_index']} @ {best['ERA_SCORE']:.4f}" if best else
                f"nodes: {len(self.rows)} | valid: 0", ""]
        # distilled "observations" section is filled by the distiller (generate.py); keep marker.
        obs = ["## Observations (distilled — injected into generation)",
               "<!-- AUTO: distiller writes neutral observations here -->", ""]
        log = ["## Trial log (raw per-node reflections)", *self.reflections[-200:]]
        (self.dir / "NOTE.md").write_text("\n".join(head + obs + log) + "\n")
