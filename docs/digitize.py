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


def digitize(path, box, xmap, ymap, colors, tol=70, row_lo=None, row_hi=None):
    """box = (left_px, right_px, top_px, bottom_px); xmap/ymap = ((px,val),(px,val))."""
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
out["calib"] = digitize(
    "unc_calib_dnase.png",
    box=(30, 269, 12, 248),
    xmap=((30, 0.0), (269, 1.0)),
    ymap=((12, 1.0), (248, 0.0)),
    # Only B_RWPE2 extracts cleanly across the full 0-1 range; the other two
    # cell lines lie almost on top of it and blend past c~0.7, so one
    # representative curve it is.
    colors={"B_RWPE2": (236, 160, 212)},
    tol=100, row_lo=58,         # skip the legend block at the top-left
)

dest = HERE / "digitized.json"
dest.write_text(json.dumps(out, indent=1))
print(f"\nwrote {dest}")
