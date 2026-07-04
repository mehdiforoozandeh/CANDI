"""CANDI v3 ERA — live monitoring dashboard (futuristic).

Standalone loop (run detached in tmux, independent of the search): every REFRESH s it reads the
live artifacts (tree.json, RESULTS.tsv, NOTE.md, constants_frozen.yaml), `squeue`, and the
process table, then regenerates a self-contained, auto-refreshing `dashboard.html` (meta-refresh,
no server, no CORS). Shows: round/phase, baseline reference, a live ERA_SCORE progress chart,
the search tree, winners + core ideas, failures, and live GPU jobs.

    python dashboard.py            # loop, writing dashboard.html every few s
    python dashboard.py --once
"""
from __future__ import annotations

import html
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFRESH = 30
USER = "mforooz"


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def _cfg():
    import yaml
    for name in ("config_real.yaml", "config.yaml"):
        p = HERE / name
        if p.exists():
            return yaml.safe_load(p.read_text()), name
    return {}, "?"


def _constants():
    import yaml
    p = HERE / "constants_frozen.yaml"
    d = yaml.safe_load(p.read_text()) if p.exists() else {}
    # baseline ERA_SCORE of the marginal predictor: S_A=0, cal_pen=0, dcr_pen for DCR=1.0
    dcr_lo = float(d.get("dcr_lo", 3.0)); w_dcr = float(d.get("w_dcr", 0.02))
    d["baseline_era_score"] = round(w_dcr * (1.0 - dcr_lo), 4)   # = -0.04
    return d


def _nodes():
    p = HERE / "tree.json"
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except Exception:
        return []


def _results():
    p = HERE / "RESULTS.tsv"
    if not p.exists():
        return []
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if not lines:
        return []
    cols = lines[0].split("\t")
    return [dict(zip(cols, l.split("\t"))) for l in lines[1:]]


def _reflections():
    p = HERE / "NOTE.md"
    out = {}
    if p.exists():
        for m in re.finditer(r"node (\d+) \(score [^)]+\): (.+)", p.read_text()):
            out[int(m.group(1))] = m.group(2).strip()
    return out


def _gate_reason(node_idx, nodes):
    for nd in nodes:
        if nd["index"] == node_idx and nd.get("program_path"):
            d = (HERE / nd["program_path"]).parent
            for o in sorted(d.glob("slurm-*.out")) + [d / "stdout.log"]:
                if o.exists():
                    txt = o.read_text()
                    g = re.findall(r"\[harness\].*?gate.*", txt)
                    if g:
                        return g[-1][:160]
                    err = [l for l in txt.splitlines() if "Error" in l or "Traceback" in l]
                    return (err[-1][:160] if err else "no ERA_SCORE / crash")
    return "?"


def collect():
    cfg, cfg_name = _cfg()
    const = _constants()
    N = int(cfg.get("num_iterations", 0)); B = int(cfg.get("batch_size", 4))
    nodes = _nodes(); rows = _results(); refl = _reflections()
    jobs = [l.split(None, 3) for l in _sh(["squeue", "-h", "-u", USER, "-o", "%i %T %M %j"]).splitlines()
            if "submit.sh" in l]
    era_alive = bool(_sh(["pgrep", "-u", USER, "-f", "run.py --config"]).strip())
    gen_active = len([l for l in _sh(["pgrep", "-u", USER, "-af", "claude -p|cursor-agent -p"]).splitlines()
                      if "pgrep" not in l])
    n_cands = max(0, len(nodes) - 1)
    cur_round = min((n_cands // B) + 1, (N + B - 1) // B) if era_alive else (n_cands + B - 1) // B
    total_rounds = (N + B - 1) // B if N else 0
    if gen_active:
        phase = f"GENERATING · {gen_active} candidate program(s) being written"
    elif jobs:
        phase = f"EXECUTING · {len(jobs)} GPU job(s) training / scoring"
    elif era_alive:
        phase = "SELECTING next parent (PUCT)"
    else:
        phase = "STOPPED"
    return dict(cfg_name=cfg_name, N=N, B=B, nodes=nodes, rows=rows, refl=refl, jobs=jobs,
                era_alive=era_alive, phase=phase, cur_round=cur_round, total_rounds=total_rounds,
                n_cands=n_cands, const=const)


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _col_series(rows, col):
    """[(node_index, value)] for valid rows with a finite `col`, sorted by node order."""
    out = []
    for r in sorted(rows, key=lambda r: int(r["node_index"])):
        if r.get("status") != "ok":
            continue
        try:
            v = float(r[col])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > -1e8:
            out.append((int(r["node_index"]), v))
    return out


def _svg_lines(rows, series, refs, dots_col=None):
    """Inline SVG: running-best line per series (col,color,label) vs variant order, + horizontal
    reference lines (value,color,label). Optional magenta per-variant dots for `dots_col`."""
    allpts = {c: _col_series(rows, c) for c, _, _ in series}
    flat = [v for pts in allpts.values() for _, v in pts] + [rv for rv, _, _ in refs]
    W, H, pad = 660, 250, 40
    if not any(allpts.values()):
        return (f'<svg width="100%" viewBox="0 0 {W} {H}"><text x="{W/2}" y="{H/2}" fill="#456" '
                f'text-anchor="middle" font-size="13">awaiting first scored variant…</text></svg>')
    ymin, ymax = min(flat) - 0.03, max(flat) + 0.03
    if ymax - ymin < 1e-6:
        ymax += 0.05
    xs_all = [x for pts in allpts.values() for x, _ in pts]
    xmin, xmax = min(xs_all), max(max(xs_all), min(xs_all) + 1)

    def X(v): return pad + (v - xmin) / (xmax - xmin) * (W - 2 * pad)
    def Y(v): return H - pad - (v - ymin) / (ymax - ymin) * (H - 2 * pad)

    grid = "".join(
        f'<line x1="{pad}" y1="{Y(gy):.0f}" x2="{W-pad}" y2="{Y(gy):.0f}" stroke="#1b2740"/>'
        f'<text x="{pad-6}" y="{Y(gy)+3:.0f}" fill="#3a4a66" font-size="9" text-anchor="end">{gy:+.2f}</text>'
        for gy in [ymin + (ymax - ymin) * i / 4 for i in range(5)])
    rl = "".join(
        f'<line x1="{pad}" y1="{Y(rv):.0f}" x2="{W-pad}" y2="{Y(rv):.0f}" stroke="{rc}" '
        f'stroke-dasharray="3 4" stroke-width="1" opacity="0.65"/>'
        f'<text x="{W-pad}" y="{Y(rv)-3:.0f}" fill="{rc}" font-size="9" text-anchor="end">{rl}</text>'
        for rv, rc, rl in refs)
    body, leg = "", ""
    for i, (col, color, label) in enumerate(series):
        pts = allpts[col]
        if not pts:
            continue
        if dots_col == col:
            body += "".join(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="2.5" fill="#e879f9" opacity="0.7"/>'
                            for x, v in pts)
        best = -1e9; rb = []
        for _, v in pts:
            best = max(best, v); rb.append(best)
        body += (f'<path d="M' + " L".join(f"{X(x):.1f},{Y(b):.1f}" for (x, _), b in zip(pts, rb))
                 + f'" fill="none" stroke="{color}" stroke-width="2.2" '
                 + f'style="filter:drop-shadow(0 0 3px {color})"/>')
        leg += (f'<span style="color:{color}">━ {label} '
                f'<b style="color:#cdd9ee">{rb[-1]:+.3f}</b></span>&nbsp;&nbsp;')
    return (f'<div style="font-size:11px;margin-bottom:4px">{leg}</div>'
            f'<svg width="100%" viewBox="0 0 {W} {H}">{grid}{rl}{body}'
            f'<text x="{pad}" y="{H-8}" fill="#3a4a66" font-size="9">variant order →</text></svg>')


def render(d) -> str:
    c = d["const"]
    base_imp = float(c.get("Q_imp_baseline", 0.4857))      # Q_imp zero-point
    base_ece = float(c.get("tau_cal", 0.0734))
    base_cidx = float(c.get("cidx_baseline", 0.4985))
    base_auroc = float(c.get("auroc_baseline", 0.7161))
    base_score = float(c.get("baseline_era_score", -0.04))
    dcr_lo, dcr_hi = c.get("dcr_lo", 3.0), c.get("dcr_hi", 5.0)
    rows = sorted(d["rows"], key=lambda r: float(r.get("ERA_SCORE") or -1e9), reverse=True)
    valid = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in d["rows"] if r.get("status") != "ok"]
    best = valid[0] if valid else None
    beat = [r for r in valid if float(r.get("Q_imp") or -9) > base_imp]

    by_idx = {int(r["node_index"]): r for r in d["rows"]}
    best_lineage = set()
    if best:
        cur = int(best["node_index"])
        while cur is not None and cur in by_idx:
            best_lineage.add(cur)
            pv = by_idx[cur].get("parent_index")
            cur = int(pv) if pv not in ("", None) else None
    mm = ["graph TD"]
    for nd in d["nodes"]:
        i = nd["index"]; sc = nd["score"]; r = by_idx.get(i, {})
        imp = r.get("Q_imp", "")
        if sc <= -1e8:
            lbl, cls = f'"{i} ✕ FAIL"', "fail"
        else:
            cls = "beat" if (imp and float(imp) > base_imp) else ("best" if i in best_lineage else "ok")
            lbl = f'"{i} · {_f(sc)}<br/>Qimp {_f(imp)}"'
        mm.append(f"  n{i}[{lbl}]:::{cls}")
        if nd.get("parent_index") is not None:
            mm.append(f"  n{nd['parent_index']} --> n{i}")
    mm += ["classDef ok fill:#10203a,stroke:#2b4a6e,color:#9fb3d0;",
           "classDef best fill:#0c3b2e,stroke:#34d399,stroke-width:3px,color:#a7f3d0;",
           "classDef beat fill:#06402a,stroke:#22d3ee,stroke-width:3px,color:#67e8f9;",
           "classDef fail fill:#3b1020,stroke:#f43f5e,color:#fca5a5;"]
    mermaid = "\n".join(mm)

    win = ""
    for r in valid[:5]:
        i = int(r["node_index"])
        idea = html.escape(d["refl"].get(i, "(no reflection captured)"))
        badge = '<span class="badge">▲ BEATS BASELINE</span>' if float(r.get("Q_imp") or -9) > base_imp else ""
        win += (f'<div class="win"><div class="wh"><b>node {i}</b>'
                f'<span class="mono">score {_f(r.get("ERA_SCORE"))} · Q_imp {_f(r.get("Q_imp"))} · '
                f'Q_den {_f(r.get("Q_den"))} · ece {_f(r.get("ece"))} · C {_f(r.get("c_index"))} · '
                f'peak {_f(r.get("peak_auroc"))} · dcr {_f(r.get("dcr"))}</span>{badge}</div>'
                f'<div class="idea">{idea}</div></div>')

    fail_html = '<span class="dim">none</span>' if not failed else "".join(
        f'<div class="failrow">node {int(r["node_index"])}: {html.escape(_gate_reason(int(r["node_index"]), d["nodes"]))}</div>'
        for r in failed)
    jobs_html = '<span class="dim">no ERA GPU jobs in queue</span>' if not d["jobs"] else "".join(
        f'<div class="mono job">▸ {j[0]} · {j[1]} · {j[2]}</div>' for j in d["jobs"])

    show = ["node_index", "parent_index", "status", "ERA_SCORE", "S_A", "Q_imp", "Q_den",
            "imp_pval_spearman", "imp_pval_pearson", "imp_count_spearman", "imp_count_pearson",
            "ece", "c_index", "peak_auroc", "dcr", "selected_as_parent_at"]
    th = "".join(f"<th>{c2.replace('_',' ')}</th>" for c2 in show)
    trs = ""
    for r in rows:
        cls = "trbeat" if (r.get("status") == "ok" and float(r.get("Q_imp") or -9) > base_imp) \
            else ("trfail" if r.get("status") != "ok" else "")
        tds = "".join(f"<td>{html.escape(_f(r.get(c2,'')) if c2 not in ('node_index','parent_index','status','selected_as_parent_at') else str(r.get(c2,'')))}</td>" for c2 in show)
        trs += f'<tr class="{cls}">{tds}</tr>'

    running = d["era_alive"]
    banner_cls = "run" if running else "stop"
    status_txt = "● ERA RUNNING" if running else "■ ERA STOPPED — process not found (finished or died)"
    nb = len(beat)
    prog = _svg_lines(d["rows"], [("ERA_SCORE", "#22d3ee", "running-best ERA_SCORE")],
                      [(0.0, "#2dd4bf", "S_A=0"), (base_score, "#f59e0b", f"baseline {base_score:+.2f}")],
                      dots_col="ERA_SCORE")
    corr = _svg_lines(d["rows"], [
        ("Q_imp", "#22d3ee", "Q_imp"), ("imp_pval_spearman", "#34d399", "pval ρ"),
        ("imp_pval_pearson", "#a7f3d0", "pval r"), ("imp_count_spearman", "#e879f9", "count ρ"),
        ("imp_count_pearson", "#f0abfc", "count r")],
        [(base_imp, "#f59e0b", f"baseline Q_imp {base_imp:.3f}")])
    pct = int(100 * d["n_cands"] / d["N"]) if d["N"] else 0
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content={REFRESH}><title>CANDI v3 · ERA</title>
<script>setTimeout(function(){{location.reload();}}, {REFRESH}*1000);</script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:'dark',maxTextSize:200000,
  themeVariables:{{fontFamily:'ui-monospace,monospace',fontSize:'12px'}}}});</script>
<style>
:root{{--cy:#22d3ee;--mg:#e879f9;--gn:#34d399;--am:#f59e0b;--rd:#f43f5e;--bg:#070b18;--card:rgba(20,28,48,.55)}}
*{{box-sizing:border-box}}
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:20px;color:#cdd9ee;
 background:radial-gradient(1200px 600px at 20% -10%,#10204a 0%,transparent 60%),
 radial-gradient(1000px 500px at 100% 0%,#2a1040 0%,transparent 55%),var(--bg);min-height:100vh}}
h1{{margin:0;font-size:20px;letter-spacing:3px;color:#fff;text-shadow:0 0 14px var(--cy)}}
h1 span{{color:var(--cy)}}
.sub{{color:#5a6c8c;font-size:11px;letter-spacing:1px;margin-top:3px}}
.banner{{margin:14px 0;padding:12px 16px;border-radius:10px;font-weight:700;letter-spacing:1px;
 display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;backdrop-filter:blur(8px)}}
.run{{background:linear-gradient(90deg,rgba(16,185,129,.15),rgba(34,211,238,.08));
 border:1px solid var(--gn);box-shadow:0 0 24px rgba(52,211,153,.25);animation:pulse 2.4s infinite}}
.stop{{background:rgba(244,63,94,.12);border:1px solid var(--rd);box-shadow:0 0 24px rgba(244,63,94,.2)}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 18px rgba(52,211,153,.2)}}50%{{box-shadow:0 0 34px rgba(52,211,153,.45)}}}}
.bar{{height:5px;border-radius:4px;background:#13203c;margin:10px 0;overflow:hidden}}
.bar>i{{display:block;height:100%;width:{pct}%;background:linear-gradient(90deg,var(--cy),var(--mg));
 box-shadow:0 0 12px var(--cy)}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--card);border:1px solid rgba(80,120,200,.18);border-radius:12px;padding:14px 16px;
 margin:12px 0;backdrop-filter:blur(10px);box-shadow:inset 0 0 30px rgba(40,70,140,.06)}}
.card h3{{margin:0 0 10px;font-size:11px;letter-spacing:2px;color:#6f86ad;text-transform:uppercase}}
.kpis{{display:flex;flex-wrap:wrap;gap:26px}}
.kpi b{{font-size:26px;color:#fff;text-shadow:0 0 12px rgba(34,211,238,.4)}}.kpi span{{color:#5a6c8c;font-size:10px;display:block;letter-spacing:1px}}
.ref{{display:flex;flex-wrap:wrap;gap:22px}}.ref div{{font-size:12px}}.ref b{{color:var(--am)}}
.mono{{font-size:11px;color:#7f93b5}} .dim{{color:#456}}
table{{border-collapse:collapse;width:100%;font-size:11px}}
th,td{{border:1px solid rgba(80,120,200,.12);padding:4px 7px;text-align:right}}
th{{background:rgba(40,60,110,.4);color:#8aa0c4;letter-spacing:1px}}
.trbeat{{background:rgba(34,211,238,.09)}}.trfail{{background:rgba(244,63,94,.1);color:#fca5a5}}
.win{{border-left:2px solid var(--gn);padding:7px 11px;margin:9px 0;background:rgba(16,40,30,.3);border-radius:0 8px 8px 0}}
.wh{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.idea{{color:#8ea4c6;font-size:11px;margin-top:4px;line-height:1.4}}
.badge{{color:#04111a;background:var(--cy);padding:1px 7px;border-radius:10px;font-size:10px;font-weight:800;box-shadow:0 0 10px var(--cy)}}
.failrow{{color:#fca5a5;font-size:11px;margin:3px 0}} .job{{color:#9fb3d0;margin:2px 0}}
.mermaid{{background:transparent;overflow:auto}}
</style></head><body>
<h1>CANDI&nbsp;<span>v3</span>&nbsp;· ERA EMPIRICAL-SOFTWARE SEARCH</h1>
<div class=sub>{d['cfg_name']} · N={d['N']} variants · {d['B']} concurrent · refresh {REFRESH}s · {datetime.now():%Y-%m-%d %H:%M:%S}</div>
<div class="banner {banner_cls}"><span>{status_txt}</span>
 <span>ROUND {d['cur_round']}/{d['total_rounds']}</span><span>{d['phase']}</span></div>
<div class=bar><i></i></div>
<div class=card><div class=kpis>
  <div class=kpi><b>{d['n_cands']}</b><span>VARIANTS DONE / {d['N']}</span></div>
  <div class=kpi><b>{_f(best['ERA_SCORE']) if best else '—'}</b><span>BEST ERA_SCORE</span></div>
  <div class=kpi><b style="color:#67e8f9">{_f(best.get('Q_imp')) if best else '—'}</b><span>BEST Q_imp (baseline {base_imp:.3f})</span></div>
  <div class=kpi><b style="color:{'#34d399' if nb else '#5a6c8c'}">{nb}</b><span>BEAT BASELINE</span></div>
  <div class=kpi><b style="color:{'#f43f5e' if failed else '#5a6c8c'}">{len(failed)}</b><span>FAILED</span></div>
</div></div>
<div class=card><h3>◇ Baseline reference (marginal average-reference predictor — the bar to beat)</h3>
 <div class=ref>
  <div>Q_imp (4-corr) <b>{base_imp:.4f}</b> &nbsp;<span class=mono>S_A=0 line</span></div>
  <div>ECE floor τ_cal <b>{base_ece:.4f}</b></div>
  <div>C-index floor <b>{base_cidx:.4f}</b></div>
  <div>peak AUROC floor <b>{base_auroc:.4f}</b></div>
  <div>DCR band <b>[{dcr_lo}, {dcr_hi}]</b></div>
  <div>baseline ERA_SCORE <b>{base_score:+.3f}</b></div>
 </div></div>
<div class=row>
 <div><div class=card><h3>◇ ERA_SCORE progress — running best · per-variant dots</h3>{prog}</div></div>
 <div><div class=card><h3>◇ Imputation correlations — running best</h3>{corr}</div></div>
</div>
<div class=row>
 <div><div class=card><h3>◇ Search tree</h3><div class=mermaid>{mermaid}</div>
  <div class=sub style="margin-top:6px">cyan=beats baseline · green=best lineage · blue=ok · red=failed</div></div></div>
 <div>
  <div class=card><h3>◇ Winners &amp; core ideas</h3>{win or '<span class=dim>no valid candidates yet</span>'}</div>
  <div class=card><h3>◇ Failures</h3>{fail_html}</div>
  <div class=card><h3>◇ Live GPU jobs (squeue)</h3>{jobs_html}</div>
 </div>
</div>
<div class=card><h3>◇ All variants</h3><table><tr>{th}</tr>{trs}</table></div>
</body></html>"""


def main():
    once = "--once" in sys.argv
    while True:
        try:
            (HERE / "dashboard.html").write_text(render(collect()))
        except Exception as e:
            (HERE / "dashboard.html").write_text(
                f"<html><body style='background:#070b18;color:#f43f5e;font-family:monospace'>"
                f"dashboard error: {html.escape(str(e))} (retrying)</body></html>")
        if once:
            break
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
