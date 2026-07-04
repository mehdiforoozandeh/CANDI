"""Render report.md -> self-contained, styled report.html (light theme, figures embedded as base64).
Run: python -m sandbox.diagnostics.covariate_probes.to_html"""
from __future__ import annotations
import os, re, base64, html, datetime

DIR = "sandbox/diagnostics/covariate_probes"
CSS = """
:root{--ink:#1f2933;--muted:#6b7785;--accent:#0f766e;--accent2:#14b8a6;--line:#e3e8ee;--card:#ffffff}
*{box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 line-height:1.6;color:var(--ink);background:#eef1f5;margin:0;padding:40px 16px;
 -webkit-font-smoothing:antialiased}
.card{max-width:1040px;margin:0 auto;background:var(--card);border:1px solid var(--line);
 border-radius:16px;box-shadow:0 8px 30px rgba(20,40,60,.08);padding:48px 56px}
.banner{margin:-48px -56px 28px;padding:36px 56px;border-radius:16px 16px 0 0;
 background:linear-gradient(135deg,#0f766e 0%,#14b8a6 100%);color:#fff}
.banner h1{margin:0;border:0;font-size:30px;font-weight:800;letter-spacing:-.4px;color:#fff}
.banner .sub{margin-top:6px;opacity:.92;font-size:15px}
h1{font-size:26px;font-weight:800;letter-spacing:-.3px;border-bottom:2px solid var(--line);padding-bottom:.3em}
h2{margin-top:2em;font-size:20px;font-weight:700;color:var(--accent);
 border-bottom:1px solid var(--line);padding-bottom:.3em}
p{margin:.7em 0}
strong{color:#10403b}
table{border-collapse:separate;border-spacing:0;margin:1.2em 0;font-size:13.5px;width:100%;
 border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);
 font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-weight:600}
thead th{background:#0f766e;color:#fff;font-weight:600;border-bottom:0;position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:rgba(15,118,110,.05)}
blockquote{border-left:4px solid #f59e0b;background:#fffbeb;margin:1.4em 0;padding:.9em 1.2em;
 border-radius:0 8px 8px 0;color:#7c5b16}
code{background:#eef2f4;padding:2px 6px;border-radius:5px;font-size:90%;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#0f766e}
figure{margin:1.4em 0;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;
 box-shadow:0 2px 10px rgba(20,40,60,.06)}
figcaption{margin-top:.6em;font-size:13px;color:var(--muted);font-style:italic}
.foot{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
"""

def _inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s

def _cellstyle(text):
    """Teal heatmap for numeric cells (leading float, clamped [0,1]); non-numeric -> no shade."""
    m = re.match(r"^(-?\d+\.?\d*)", text.strip())
    if not m:
        return ""
    t = max(0.0, min(1.0, float(m.group(1))))
    r = int(255 - t * (255 - 15)); g = int(255 - t * (255 - 118)); b = int(255 - t * (255 - 110))
    fg = "#fff" if t > 0.62 else "inherit"
    return f' style="background:rgb({r},{g},{b});color:{fg}"'

def _img(alt, path):
    p = os.path.join(DIR, path)
    if not os.path.exists(p):
        return f"<p><em>[missing figure: {html.escape(path)}]</em></p>"
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    cap = f"<figcaption>{_inline(alt)}</figcaption>" if alt else ""
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">{cap}</figure>'

def md_to_html(md):
    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1; continue
        m = re.match(r"(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1)); out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>"); i += 1; continue
        mi = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
        if mi:
            out.append(_img(mi.group(1), mi.group(2))); i += 1; continue
        if ln.lstrip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip()); i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>"); continue
        if ln.lstrip().startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i].strip()); i += 1
            cells = lambda r: [c.strip() for c in r.strip("|").split("|")]
            body = [r for r in tbl if not re.match(r"^\|[\s:|-]+\|$", r)]
            head, rows = body[0], body[1:]
            h = "".join(f"<th>{_inline(c)}</th>" for c in cells(head))
            rs = ""
            for r in rows:
                cs = cells(r)
                tds = "".join(f"<td{(_cellstyle(c) if j else '')}>{_inline(c)}</td>" for j, c in enumerate(cs))
                rs += f"<tr>{tds}</tr>"
            out.append(f"<table><thead><tr>{h}</tr></thead><tbody>{rs}</tbody></table>"); continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"(#|\||>|!\[)", lines[i].lstrip()):
            buf.append(lines[i].strip()); i += 1
        out.append(f"<p>{_inline(' '.join(buf))}</p>")
    return "\n".join(out)

def main():
    md = open(f"{DIR}/report.md").read()
    body = md_to_html(md)
    # promote the first H1 into a gradient banner
    body = re.sub(r"^<h1>(.*?)</h1>",
                  r'<div class="banner"><h1>\1</h1>'
                  r'<div class="sub">CANDI · covariate decodability probes · MERGED dataset</div></div>',
                  body, count=1)
    stamp = datetime.date.today().isoformat()
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Covariate decodability probes</title><style>{CSS}</style></head>"
           f"<body><div class='card'>{body}"
           f"<div class='foot'>Generated {stamp} · cells shaded by value (teal = higher)</div>"
           f"</div></body></html>")
    open(f"{DIR}/report.html", "w").write(doc)
    print(f"wrote {DIR}/report.html ({len(doc)//1024} KB)")

if __name__ == "__main__":
    main()
