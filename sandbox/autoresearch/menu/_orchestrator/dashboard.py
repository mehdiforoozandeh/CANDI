"""Menu-AR dashboard — regenerate a self-contained, auto-refreshing dashboard.html from every
loop's results.tsv + journals + squeue. No server (meta-refresh). Theme matches the approved mock
and the candi_v3 ERA dashboard. Shows: header status, baseline + candi_v2 side-by-side (v2 = the
common origin), the race, the loops table, per-loop running-best + per-variant-dots charts, score
components, and the orchestrator panel.

  python dashboard.py            # loop, writing dashboard.html every REFRESH s
  python dashboard.py --once
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

MENU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MENU / "_harness"))
import journal  # noqa: E402

REFRESH = 30
LOOPS = [  # tag, color, thesis
    ("single_lambda", "#22d3ee", "one λ → NB+pval+peak derived"),
    ("crps_calibration", "#f59e0b", "calibrated-by-design output"),
    ("factorized", "#34d399", "low-rank ct×assay×pos"),
    ("axial_longrange", "#e879f9", "axial + long-range ctx"),
    ("repr_first", "#818cf8", "latent + ELBO/SIGReg"),
]


def _const():
    d = yaml.safe_load((MENU / "_judge" / "constants_frozen.yaml").read_text())
    d["baseline_era_score"] = round(d["w_dcr"] * (1.0 - d["dcr_lo"]), 4)   # marginal: only DCR fails
    return d


def _rows(tag):
    return journal.read_results(MENU / tag)


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _inflight():
    return sum(1 for s in _job_states().values() if s)


def _job_states() -> dict:
    """{tag: 'R'|'PD'} for menu-<tag> SLURM jobs currently in the queue."""
    r = subprocess.run(["squeue", "-h", "-u", "mforooz", "-o", "%j|%t"], capture_output=True, text=True)
    out = {}
    for ln in r.stdout.splitlines():
        if "|" in ln and ln.startswith("menu-"):
            name, st = ln.split("|", 1)
            out[name[len("menu-"):]] = st.strip()
    return out


def _tmux_alive(tag: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"menu-{tag}"], capture_output=True).returncode == 0


def _loop_live(tag: str, jobs: dict) -> str:
    """Live phase of a loop, inferred WITHOUT touching the running driver."""
    if (MENU / tag / "DONE").exists():
        return "done"
    if not _tmux_alive(tag):
        return "stopped"
    st = jobs.get(tag)
    if st == "R":
        return "GPU running"
    if st in ("PD", "CF"):
        return "GPU queued"
    return "editing"        # tmux alive, no job → agent editing / smoking (pre-GPU)


def _v2_anchor():
    for tag, *_ in LOOPS:
        for r in _rows(tag):
            if r.get("status") == "base":
                return r
    return None


# ---- per-loop running-best + per-variant-dots SVG (ERA style) ----
def _progress_svg(rows, color, base, v2):
    pts = [_f(r["era_score"]) for r in rows if r.get("status") in ("keep", "reset", "crash")
           and _f(r["era_score"]) > -1.0]
    W, H, pad = 660, 240, 40
    if not pts:
        return (f'<svg width="100%" viewBox="0 0 {W} {H}"><text x="{W/2}" y="{H/2}" fill="#456" '
                f'text-anchor="middle">awaiting first scored iteration…</text></svg>')
    flat = pts + [base, v2]
    ymin, ymax = min(flat) - 0.015, max(flat) + 0.015
    xmax = max(len(pts) - 1, 1)
    X = lambda i: pad + i / xmax * (W - 2 * pad)
    Y = lambda v: H - pad - (v - ymin) / (ymax - ymin) * (H - 2 * pad)
    s = [f'<svg width="100%" viewBox="0 0 {W} {H}">']
    for i in range(5):
        gy = ymin + (ymax - ymin) * i / 4
        s.append(f'<line x1="{pad}" y1="{Y(gy):.0f}" x2="{W-pad}" y2="{Y(gy):.0f}" stroke="#1b2740"/>'
                 f'<text x="{pad-6}" y="{Y(gy)+3:.0f}" fill="#3a4a66" font-size="9" text-anchor="end">{gy:+.2f}</text>')
    for rv, rc, rl in ((base, "#5a6c8c", "baseline"), (v2, "#22d3ee", "v2")):
        s.append(f'<line x1="{pad}" y1="{Y(rv):.0f}" x2="{W-pad}" y2="{Y(rv):.0f}" stroke="{rc}" '
                 f'stroke-dasharray="3 4" stroke-width="1" opacity="0.6"/>'
                 f'<text x="{W-pad}" y="{Y(rv)-3:.0f}" fill="{rc}" font-size="9" text-anchor="end">{rl}</text>')
    for i, v in enumerate(pts):
        s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" fill="#7c8cb0" opacity="0.55"/>')
    best, d = -1e9, []
    for v in pts:
        best = max(best, v); d.append(best)
    s.append('<path d="M' + " L".join(f"{X(i):.1f},{Y(b):.1f}" for i, b in enumerate(d)) +
             f'" fill="none" stroke="{color}" stroke-width="2.2" style="filter:drop-shadow(0 0 3px {color})"/>')
    s.append(f'<text x="{pad}" y="{H-8}" fill="#3a4a66" font-size="9">variant order →</text></svg>')
    return "".join(s)


def render() -> str:
    c = _const()
    base = c["baseline_era_score"]
    v2row = _v2_anchor()
    v2 = _f(v2row["era_score"], base) if v2row else base
    inflight = _inflight()

    # loops table + best
    jobs = _job_states()
    summ = []
    for tag, color, thesis in LOOPS:
        rows = _rows(tag)
        scored = [r for r in rows if r.get("status") in ("keep", "reset", "base")]
        best = max((_f(r["era_score"]) for r in scored), default=base)
        keeps = sum(1 for r in rows if r.get("status") == "keep")
        iters = sum(1 for r in rows if r.get("status") != "base")     # committed iterations (excl. base)
        last = rows[-1]["status"] if rows else "—"
        ece = next((_f(r["ece"]) for r in reversed(rows) if r.get("ece")), None)
        summ.append(dict(tag=tag, color=color, thesis=thesis, rows=rows, best=best,
                         keeps=keeps, iters=iters, last=last, ece=ece, live=_loop_live(tag, jobs)))
    summ_sorted = sorted(summ, key=lambda x: -x["best"])
    live_counts = {k: sum(1 for s in summ if s["live"] == k) for k in
                   ("editing", "GPU queued", "GPU running", "done", "stopped")}
    live_summary = " · ".join(f"{k} {v}" for k, v in live_counts.items() if v)

    lo = min([base] + [s["best"] for s in summ]) - 0.01
    hi = max([v2] + [s["best"] for s in summ]) + 0.01
    pos = lambda v: max(0, min(100, (v - lo) / (hi - lo) * 100))

    race = []
    for s in summ_sorted:
        race.append(
            f'<div class="race-row"><div class="race-name">{s["tag"]}<span class="th">{s["thesis"]}</span></div>'
            f'<div class="track"><div class="refline" style="left:{pos(base):.0f}%;border-left:1px dashed #5a6c8c"></div>'
            f'<div class="refline" style="left:{pos(v2):.0f}%;border-left:1px dashed #22d3ee"></div>'
            f'<div class="fill" style="width:{pos(s["best"]):.0f}%;background:linear-gradient(90deg,{s["color"]}33,{s["color"]})"></div></div>'
            f'<div class="race-val">{s["best"]:+.4f}</div></div>')

    _LIVE_CLR = {"editing": "#22d3ee", "GPU queued": "#f59e0b", "GPU running": "#34d399",
                 "done": "#5a6c8c", "stopped": "#f43f5e"}
    trows = []
    for s in summ_sorted:
        ece_html = "—" if s["ece"] is None else (
            f'<span class="ok">✓ {s["ece"]:.3f}</span>' if s["ece"] <= c["tau_cal"]
            else f'<span class="bad">✗ {s["ece"]:.3f}</span>')
        lv = s["live"]
        pulse = "dot" if lv in ("editing", "GPU running", "GPU queued") else ""
        live_html = (f'<span class="{pulse}" style="background:{_LIVE_CLR[lv]};box-shadow:0 0 8px {_LIVE_CLR[lv]}">'
                     f'</span><span style="color:{_LIVE_CLR[lv]}">{lv}</span>')
        trows.append(
            f'<tr><td style="color:{s["color"]}">{s["tag"]}</td><td style="color:#5a6c8c">{s["thesis"]}</td>'
            f'<td>{live_html}</td>'
            f'<td class="pos">{s["best"]:+.4f}</td><td>{s["best"]-base:+.4f}</td><td>{s["best"]-v2:+.4f}</td>'
            f'<td>{s["iters"]}</td><td>{s["keeps"]}</td><td>{ece_html}</td>'
            f'<td><span class="st st-{s["last"]}">{s["last"]}</span></td></tr>')

    prog = []
    for s in summ:
        prog.append(
            f'<div class="card"><div class="prog-h"><span style="color:{s["color"]}">◇ {s["tag"]}</span>'
            f'<span class="leg">━ running-best <b>{s["best"]:+.4f}</b> · <span style="color:#7c8cb0">● per-variant</span></span></div>'
            f'{_progress_svg(s["rows"], s["color"], base, v2)}</div>')

    def vrow(r, score):
        ece = _f(r.get("ece")) if r and r.get("ece") else None
        ece_h = "—" if ece is None else (f'<span class="ok">{ece:.3f} ✓</span>' if ece <= c["tau_cal"]
                                         else f'<span class="bad">{ece:.3f} ✗</span>')
        dcr = _f(r.get("dcr")) if r and r.get("dcr") else None
        dcr_h = "—" if dcr is None else (f'<span class="ok">{dcr:.2f} ✓</span>' if c["dcr_lo"] <= dcr <= c["dcr_hi"]
                                         else f'<span class="bad">{dcr:.2f} ✗</span>')
        return ece_h, dcr_h
    v2_ece, v2_dcr = vrow(v2row, v2)

    return PAGE.format(
        ts=time.strftime("%H:%M:%S"), refresh=REFRESH, inflight=inflight,
        live_summary=live_summary or "idle",
        total_iters=sum(s["iters"] for s in summ),
        base=base, v2=v2, v2d=v2 - base,
        qb=c["Qimp"] if "Qimp" in c else c["Q_imp_baseline"], tau=c["tau_cal"],
        auroc=c["auroc_baseline"], cidx=c["cidx_baseline"],
        v2_qimp=_f(v2row.get("Q_imp")) if v2row else 0, v2_ece=v2_ece, v2_dcr=v2_dcr,
        race="".join(race), table="".join(trows), prog="".join(prog))


PAGE = """<!doctype html><html><head><meta charset="utf-8"/>
<meta http-equiv="refresh" content="{refresh}"/><title>CANDI v3 · MENU AUTORESEARCH</title>
<style>
:root{{--bg:#070b18;--card:#13203c;--line:#23355e;--cy:#22d3ee;--gn:#34d399;--rd:#f43f5e;--tx:#cdd9ee;--tx2:#9fb3d0;--tx3:#5a6c8c;--mono:ui-monospace,Menlo,monospace}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#0e1733,var(--bg) 55%);color:var(--tx);font-family:var(--mono);font-size:13px}}
.wrap{{max-width:1320px;margin:0 auto;padding:18px 22px 60px}}
h1{{font-size:16px;letter-spacing:.14em;margin:0;background:linear-gradient(90deg,var(--cy),#e879f9);-webkit-background-clip:text;background-clip:text;color:transparent}}
.sub{{color:var(--tx3);font-size:11px;margin-top:3px}}
.head{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:14px}}
.chip{{border:1px solid var(--line);border-radius:20px;padding:4px 11px;font-size:11px;color:var(--tx2);margin-left:8px}}.chip b{{color:var(--tx)}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--gn);box-shadow:0 0 8px var(--gn);margin-right:6px}}
.sectlabel{{font-size:10px;letter-spacing:.22em;color:var(--tx3);text-transform:uppercase;margin:22px 2px 9px}}
.card{{background:linear-gradient(180deg,rgba(30,46,86,.35),rgba(13,18,36,.35));border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.refgrid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.bignum{{font-size:24px;font-weight:600}}.mlist{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px 14px;margin-top:11px}}
.mlist .k{{color:var(--tx3);font-size:10px}}.mlist .v{{font-size:14px}}.ok{{color:var(--gn)}}.bad{{color:var(--rd)}}.pos{{color:var(--gn)}}
.race-row{{display:grid;grid-template-columns:150px 1fr 70px;align-items:center;gap:12px;margin:9px 0}}
.race-name{{font-size:12px}}.race-name .th{{color:var(--tx3);font-size:10px;display:block}}
.track{{position:relative;height:22px;background:rgba(20,32,60,.45);border-radius:5px;border:1px solid var(--line)}}
.fill{{position:absolute;top:0;bottom:0;left:0;border-radius:5px;opacity:.85}}.refline{{position:absolute;top:-4px;bottom:-4px;width:2px}}
.race-val{{text-align:right}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th{{text-align:left;color:var(--tx3);font-size:10px;text-transform:uppercase;padding:7px 9px;border-bottom:1px solid var(--line)}}
td{{padding:8px 9px;border-bottom:1px solid rgba(35,53,94,.45)}}
.st{{font-size:10px;padding:2px 8px;border-radius:10px}}.st-keep{{background:rgba(52,211,153,.14);color:var(--gn)}}.st-reset{{background:rgba(90,108,140,.18);color:var(--tx3)}}.st-crash,.st-gate{{background:rgba(244,63,94,.14);color:var(--rd)}}.st-base{{background:rgba(34,211,238,.14);color:var(--cy)}}
.prog-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:14px}}
.prog-h{{display:flex;justify-content:space-between;align-items:center;font-size:12px;margin-bottom:4px}}.prog-h .leg{{font-size:11px;color:var(--tx2)}}.prog-h .leg b{{color:var(--tx)}}
</style></head><body><div class="wrap">
<div class="head"><div><h1>CANDI v3 · MENU AUTORESEARCH</h1><div class="sub">5-loop design-menu traversal · stateless ratchet · chr19→chr21 · fixed wall-clock</div></div>
<div><span class="chip"><span class="dot"></span><b>LIVE</b></span><span class="chip">{live_summary}</span><span class="chip">committed iters <b>{total_iters}</b></span><span class="chip">GPU in-flight <b>{inflight}/5</b></span><span class="chip">refresh <b>{refresh}s</b> · {ts}</span></div></div>

<div class="sectlabel">Starting point — references every loop races against</div>
<div class="refgrid">
<div class="card"><div style="color:var(--tx2);letter-spacing:.1em">MARGINAL BASELINE <span style="color:var(--tx3)">(frozen)</span></div>
<div class="bignum" style="color:var(--tx2)">{base:+.3f} <span style="font-size:12px;color:var(--tx3)">ERA_SCORE</span></div>
<div class="mlist"><div><div class="k">Q_imp</div><div class="v">{qb:.4f}</div></div><div><div class="k">ECE floor τ</div><div class="v">{tau:.4f}</div></div><div><div class="k">peak AUROC</div><div class="v">{auroc:.4f}</div></div><div><div class="k">C-index</div><div class="v">{cidx:.4f}</div></div><div><div class="k">DCR band</div><div class="v">[3–5]</div></div><div><div class="k">S_A</div><div class="v">0.000</div></div></div></div>
<div class="card" style="border-color:rgba(34,211,238,.4)"><div style="color:var(--cy);letter-spacing:.1em">CANDI v2 — row-0 <span style="color:var(--tx3)">(common origin of all 5 loops)</span></div>
<div class="bignum" style="color:var(--cy)">{v2:+.3f} <span style="font-size:12px;color:var(--tx3)">ERA_SCORE</span> <span style="font-size:12px;color:var(--gn)">(Δbase {v2d:+.3f})</span></div>
<div class="mlist"><div><div class="k">Q_imp</div><div class="v">{v2_qimp:.3f}</div></div><div><div class="k">ECE</div><div class="v">{v2_ece}</div></div><div><div class="k">DCR</div><div class="v">{v2_dcr}</div></div></div></div>
</div>

<div class="sectlabel">The race — best ERA_SCORE per loop (origin = v2, floor = baseline)</div>
<div class="card">{race}</div>

<div class="sectlabel">Loops</div>
<div class="card" style="padding:4px 8px"><table><thead><tr><th>loop</th><th>thesis</th><th>live</th><th>best</th><th>Δbase</th><th>Δv2</th><th>iters</th><th>keeps</th><th>ECE</th><th>last</th></tr></thead><tbody>{table}</tbody></table></div>

<div class="sectlabel">ERA_SCORE progress — running best · per-variant dots (one box per loop)</div>
<div class="prog-grid">{prog}</div>

</div></body></html>"""


def main():
    once = "--once" in sys.argv
    while True:
        try:
            (MENU / "dashboard.html").write_text(render())
        except Exception as e:  # never let a render error kill the dashboard loop
            print("dashboard render error:", e)
        if once:
            break
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
