"""Digitise the real curves out of the MLCB figure PNGs.

The SVGs turned out to be raster wrappers, so the only way to replot the published
results as clean vector art (rather than embedding a 404x243 blur) is to read the
curves back off the rendered image. Axes transforms are pinned to detected tick
pixel positions, so the recovered values are the published ones, not eyeballed.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Paths resolve from this file, not the working directory, so the script runs
# from anywhere. The source PNGs are tracked in this repo.
HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "manuscript2.0" / "CANDI-MLCB" / "mlcbCANDI_figs_png"
out = {}


def load(path):
    """Composite onto white — several figures carry alpha, which would otherwise
    read back as black and swamp the colour matching."""
    im = Image.open(FIGS / path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(bg, im).convert("RGB")).astype(int)


def digitize(path, box, xmap, ymap, colors, tol=70, row_lo=None, row_hi=None,
             masks=()):
    """box = (left_px, right_px, top_px, bottom_px); xmap/ymap = ((px,val),(px,val)).

    masks blanks out rectangles (col0, col1, row0, row1) — use it for legend
    blocks. A blanket row_lo cannot do that job: the legend sits at the top of
    the plot and so does the right-hand end of a rising curve, so a row floor
    high enough to clear the legend also truncates the curve.
    """
    im = load(path)
    L, R, T, B = box
    (xp0, xv0), (xp1, xv1) = xmap
    (yp0, yv0), (yp1, yv1) = ymap
    L, R = int(np.ceil(L)), int(np.floor(R))
    r0 = int(row_lo if row_lo is not None else T + 1)
    r1 = int(row_hi if row_hi is not None else B - 1)

    res = {}
    for name, rgb in colors.items():
        target = np.array(rgb)
        tsat = int(target.max() - target.min())
        xs, ys = [], []
        for c in range(L + 1, R):
            col = im[r0:r1, c, :]
            d = np.abs(col - target).sum(1)
            # Saturation gate: grey gridlines and anti-aliased text sit within a
            # loose RGB distance of pastel targets, so require the pixel to be at
            # least as colourful as the target before accepting it.
            sat = col.max(1) - col.min(1)
            if tsat > 40:
                # Coloured target: grey gridlines and anti-aliased text fall
                # within a loose RGB distance of pastel colours, so demand the
                # pixel be at least as colourful as the target.
                ok = sat > tsat * 0.35
            else:
                # Achromatic target (the black "Observed" line). A saturation
                # floor would reject it outright, so require neutrality and
                # darkness instead.
                ok = (sat < 30) & (col.mean(1) < 120)
            for mc0, mc1, mr0, mr1 in masks:
                if mc0 <= c < mc1:
                    ok[max(0, mr0 - r0):max(0, mr1 - r0)] = False
            hit = np.where((d < tol) & ok)[0]
            if hit.size == 0:
                continue
            # Several disjoint groups can match in one column (curve + legend +
            # a crossing curve). Averaging them yields nonsense, so keep the
            # longest contiguous run, which is the line itself.
            runs, start = [], hit[0]
            for a, b in zip(hit, hit[1:]):
                if b != a + 1:
                    runs.append((start, a))
                    start = b
            runs.append((start, hit[-1]))
            lo_r, hi_r = max(runs, key=lambda t: t[1] - t[0])
            row = (lo_r + hi_r) / 2 + r0
            xs.append(xv0 + (c - xp0) * (xv1 - xv0) / (xp1 - xp0))
            ys.append(yv0 + (row - yp0) * (yv1 - yv0) / (yp1 - yp0))
        res[name] = {"x": [round(v, 4) for v in xs], "y": [round(v, 4) for v in ys]}
        print(f"  {path} :: {name:10s} n={len(xs):4d} "
              f"x[{min(xs):.2f},{max(xs):.2f}] y[{min(ys):.3f},{max(ys):.3f}]")
    return res


print("rna_vs_ntracks — Pearson r vs number of available tracks")
out["rna"] = digitize(
    "rna_vs_ntracks.png",
    box=(58.5, 393.5, 50, 190),
    xmap=((73, 4), (345, 12)),
    ymap=((52.5, 0.8), (160, 0.5)),
    colors={"Observed": (0, 0, 0), "Den": (0, 0, 255),
            "DenImp": (0, 117, 0), "Latent": (255, 0, 0)},
    tol=90, row_lo=50,
)

print("\nunc_calib_dnase — empirical coverage vs stated confidence")
# Axis calibration comes from the detected tick pixels: x ticks at 30/78/126/
# 173/221 for 0.0-0.8 put 1.0 at 269; y ticks at 60/107/154/201/248 for 0.8-0.0
# put 1.0 at row 13.
# B_DND-41 (green) is the series to take. The three cell lines nearly coincide,
# but green is drawn on top, so it stays visible where they converge past
# c~0.85; B_RWPE2 (pink) is occluded there.
out["calib"] = digitize(
    "unc_calib_dnase.png",
    box=(30, 269, 13, 248),
    xmap=((30, 0.0), (269, 1.0)),
    ymap=((13, 1.0), (248, 0.0)),
    colors={"B_DND-41": (107, 189, 107)},
    tol=100,
    masks=((30, 122, 0, 60),),   # legend block, top-left
)
# A calibration curve passes through (0,0) and (1,1) by construction, and the
# published figure shows all three curves doing so. The extraction stops just
# short at both ends where the curves overlap, so both corners are anchored.
_c = out["calib"]["B_DND-41"]
_c["x"] = [0.0] + _c["x"] + [1.0]
_c["y"] = [0.0] + _c["y"] + [1.0]

dest = HERE / "digitized.json"
dest.write_text(json.dumps(out, indent=1))
print(f"\nwrote {dest}")
