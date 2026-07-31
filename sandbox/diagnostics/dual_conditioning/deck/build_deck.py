#!/usr/bin/env python
"""Inline every assets/* reference in template.html as a base64 data URI ->
one self-contained portable file: dual_conditioning.html."""
import base64, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
MIME = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg"}

html = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()

def datauri(path):
    ext = os.path.splitext(path)[1].lower()
    raw = open(path, "rb").read()
    return f"data:{MIME[ext]};base64," + base64.b64encode(raw).decode()

seen = {}
def repl(m):
    rel = m.group(1)
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        raise SystemExit(f"MISSING ASSET: {rel}")
    seen[rel] = seen.get(rel, 0) + 1
    return 'src="' + datauri(p) + '"'

out = re.sub(r'src="(assets/[^"]+)"', repl, html)
dest = os.path.join(HERE, "dual_conditioning.html")
open(dest, "w", encoding="utf-8").write(out)

kb = len(out.encode()) / 1024
print(f"built {dest}  ({kb:.0f} KB, self-contained)")
print("inlined assets:")
for k in sorted(seen):
    print(f"  {seen[k]}x  {k}")
if 'src="assets/' in out:
    print("WARNING: some assets/ refs remain uninlined!")
