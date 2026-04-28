"""Inspect ``sandbox/hpo_graph.json``: leaderboard, lineage, axes diffs, GraphViz.

Examples::

    # Table of all nodes, sorted by quality_score (lower=better) with eligibility.
    python -m sandbox.hpo.view_graph --leaderboard

    # Quick summary — node count, edge count, schema version.
    python -m sandbox.hpo.view_graph --summary

    # Print the ancestor chain of one run.
    python -m sandbox.hpo.view_graph --lineage baseline_anchor

    # Emit a GraphViz `.dot` file (use `dot -Tpng` to render).
    python -m sandbox.hpo.view_graph --graphviz > hpo.dot

The script is read-only — it never mutates the graph file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sandbox.hpo.axes import diff as axes_diff
from sandbox.hpo.graph import load_graph

DEFAULT_GRAPH = Path("sandbox/hpo_graph.json")


def _fmt(v: Any, width: int = 8, prec: int = 4) -> str:
    if v is None:
        return "—".rjust(width)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "nan".rjust(width)
        return f"{v:>{width}.{prec}f}"
    return str(v).rjust(width)


def _eligible(node: Dict[str, Any]) -> bool:
    if node.get("diverged"):
        return False
    if int(node.get("nan_inf_count", 0)) > 0:
        return False
    return True


def cmd_summary(graph: Dict[str, Any]) -> int:
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    print(f"schema_version : {graph.get('schema_version')}")
    print(f"nodes          : {len(nodes)}")
    print(f"edges          : {len(edges)}")
    if nodes:
        labels: Dict[str, int] = {}
        for n in nodes.values():
            labels[n.get("experiment_label") or "(unlabeled)"] = (
                labels.get(n.get("experiment_label") or "(unlabeled)", 0) + 1
            )
        print("by experiment_label:")
        for k, v in sorted(labels.items()):
            print(f"  {k:25s} {v}")
    return 0


def cmd_leaderboard(graph: Dict[str, Any]) -> int:
    nodes = list(graph.get("nodes", {}).values())
    if not nodes:
        print("(empty graph)")
        return 0

    rows = []
    for n in nodes:
        res = n.get("results_at_best_epoch", {}) or {}
        rows.append({
            "run_id": n.get("run_id"),
            "label": n.get("experiment_label") or "",
            "elig": _eligible(n),
            "diverged": bool(n.get("diverged")) if n.get("diverged") is not None else None,
            "best_ep": res.get("best_epoch"),
            "q": res.get("quality_score"),
            "pval_i": res.get("eval_losses/pval_imp_loss"),
            "cnt_i": res.get("eval_losses/count_imp_loss"),
            "pk_i": res.get("eval_losses/peak_imp_loss"),
            "imP_p": res.get("eval_metrics/imp_pval_pearson_gw"),
            "imP_s": res.get("eval_metrics/imp_pval_spearman_gw"),
            "imPk": res.get("eval_metrics/imp_peak_auroc_gw"),
        })

    def _key(r):  # eligible runs first; then by quality_score (lower better).
        q = r["q"] if isinstance(r["q"], (int, float)) else float("inf")
        return (0 if r["elig"] else 1, q)

    rows.sort(key=_key)

    hdr = (
        f"{'run_id':32s} {'label':14s} {'elig':>5s} {'div':>4s} "
        f"{'best_ep':>7s} {'q':>7s} {'pval_i':>7s} {'cnt_i':>7s} {'pk_i':>6s} | "
        f"{'imP_p':>6s} {'imP_s':>6s} {'imPk':>6s}"
    )
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        elig = "yes" if r["elig"] else "no"
        div = "—" if r["diverged"] is None else ("yes" if r["diverged"] else "no")
        print(
            f"{r['run_id'][:32]:32s} {(r['label'] or '')[:14]:14s} {elig:>5s} {div:>4s} "
            f"{_fmt(r['best_ep'], 7, 0)} {_fmt(r['q'], 7, 3)} "
            f"{_fmt(r['pval_i'], 7)} {_fmt(r['cnt_i'], 7)} {_fmt(r['pk_i'], 6)} | "
            f"{_fmt(r['imP_p'], 6, 3)} {_fmt(r['imP_s'], 6, 3)} {_fmt(r['imPk'], 6, 3)}"
        )
    return 0


def cmd_lineage(graph: Dict[str, Any], run_id: str) -> int:
    nodes = graph.get("nodes", {})
    if run_id not in nodes:
        print(f"error: run_id {run_id!r} not in graph", file=sys.stderr)
        return 2

    chain: List[str] = [run_id]
    seen = {run_id}
    cur = run_id
    while True:
        parents = nodes.get(cur, {}).get("parent_run_ids") or []
        if not parents:
            break
        # Follow first parent only; multiple parents are unusual but we record them.
        nxt = parents[0]
        if nxt in seen:  # cycle guard.
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt

    # Print root → leaf.
    chain.reverse()
    print(f"lineage of {run_id} (root → leaf):")
    prev_axes: Optional[Dict[str, Any]] = None
    for rid in chain:
        n = nodes[rid]
        res = n.get("results_at_best_epoch", {}) or {}
        q = res.get("quality_score")
        elig = "ELIG" if _eligible(n) else "ineligible"
        print(f"  {rid:30s}  q={_fmt(q, 7, 3)}  {elig}  label={n.get('experiment_label') or ''}")
        cur_axes = n.get("config_axes", {}) or {}
        if prev_axes is not None:
            d = axes_diff(prev_axes, cur_axes)
            for axis, (a, b) in d.items():
                print(f"      Δ {axis}: {a!r} → {b!r}")
        prev_axes = cur_axes
    return 0


def cmd_graphviz(graph: Dict[str, Any]) -> int:
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    print("digraph hpo {")
    print('  rankdir=LR;')
    print('  node [shape=box, fontname="monospace", fontsize=10];')
    for rid, n in nodes.items():
        res = n.get("results_at_best_epoch", {}) or {}
        q = res.get("quality_score")
        q_s = "—" if q is None else f"{q:.3f}"
        elig = "ok" if _eligible(n) else "DIV"
        label = n.get("experiment_label") or ""
        text = f"{rid}\\n{label}\\nq={q_s} {elig}"
        color = "lightgreen" if _eligible(n) else "lightcoral"
        print(f'  "{rid}" [label="{text}", style=filled, fillcolor="{color}"];')
    for e in edges:
        d = e.get("diff") or {}
        keys = list(d.keys())[:3]
        diff_label = "\\n".join(f"{k}: {d[k][0]}→{d[k][1]}" for k in keys)
        if len(d) > 3:
            diff_label += f"\\n(+{len(d) - 3} more)"
        print(f'  "{e["from"]}" -> "{e["to"]}" [label="{diff_label}", fontsize=8];')
    print("}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Read-only viewer for sandbox/hpo_graph.json")
    p.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help=f"Graph file (default: {DEFAULT_GRAPH})")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", action="store_true")
    g.add_argument("--leaderboard", action="store_true")
    g.add_argument("--lineage", type=str, metavar="RUN_ID")
    g.add_argument("--graphviz", action="store_true")
    g.add_argument("--dump", action="store_true", help="Print the raw JSON.")
    args = p.parse_args(argv)

    graph = load_graph(args.graph)
    if args.summary:
        return cmd_summary(graph)
    if args.leaderboard:
        return cmd_leaderboard(graph)
    if args.lineage:
        return cmd_lineage(graph, args.lineage)
    if args.graphviz:
        return cmd_graphviz(graph)
    if args.dump:
        print(json.dumps(graph, indent=2, default=float))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
