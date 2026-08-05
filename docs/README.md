# Graphical abstract

The CANDI graphical abstract, built for CEEHRC 2026 (12th Annual Canadian Conference on
Epigenetics, abstract deadline 7 August 2026). Shown at the top of the repository README
and served on its own at **[mehdiforoozandeh.github.io/CANDI](https://mehdiforoozandeh.github.io/CANDI)**,
which is the URL printed under the submitted abstract.

## Regenerate

```bash
python docs/build_ga.py
```

Writes `candi-graphical-abstract.{svg,pdf,png}` next to the script. Runs from any working
directory. Takes a few seconds. Needs `matplotlib`, `numpy` and `pillow` — no pinned
versions; it uses nothing version-sensitive beyond `GridSpec.subgridspec` (matplotlib ≥ 3.3).

To re-extract the published curves first — only necessary if a source figure changes:

```bash
python docs/digitize.py     # rewrites docs/digitized.json
python docs/build_ga.py
```

## Files

| File | What it is |
|---|---|
| `build_ga.py` | Builds the figure. Self-contained; the only input is `digitized.json`. |
| `digitize.py` | Recovers panel A and B curves from the published figure PNGs. |
| `digitized.json` | Its output. Committed so `build_ga.py` runs without re-digitizing. |
| `index.html` | The GitHub Pages entry point: the figure alone, nothing else. |
| `candi-graphical-abstract.{svg,pdf,png}` | Build products, committed so the README and Pages can serve them. |

GitHub Pages is configured to serve this folder from `main`.

## What is real and what is schematic

This distinction matters if the figure is ever reused in a paper or a talk where someone
may ask.

**The workflow strip (top) is entirely schematic.** The tensor, the availability pattern,
the sliced heatmap, the read-count tracks, the DNA strip and the predicted intervals are
all generated from seeded random draws in `build_ga.py`. They illustrate the shape of the
problem, not measured data. The seeds are fixed, so the figure is byte-reproducible, but
none of those numbers mean anything. The confidence bands do follow a real functional form
— `1.96·sqrt(mu) + 0.30·mu`, a Poisson term plus overdispersion, which is what a negative
binomial gives — so the band widens with signal the way a real one does.

**The three result panels are real published numbers.**

- **A — calibration.** Digitized from `mlcbCANDI_figs_png/unc_calib_dnase.png`, the
  `B_RWPE2` curve. Recovered range 0.03 → 0.80. Only that one cell line extracts cleanly
  across the full range; the other two lie almost on top of it and blend past c ≈ 0.7.
- **B — expression vs. assay count.** Digitized from `mlcbCANDI_figs_png/rna_vs_ntracks.png`,
  all four sources. Recovered: Observed 0.596 → 0.722, Denoised 0.602 → 0.739,
  Denoised+Imputed 0.722 → 0.750, Latent 0.789 → 0.790 (flat).
- **C — SAGAconf robustness ratio.** Not digitized. The exact values are copied from
  `manuscript2.0/CANDI-MLCB/mlcbCANDI_figs_svg/saga_repro_make.py`, which is the script that
  drew the published panel. All six biosamples are shown, including the two where Latent
  loses (cardiac muscle, hepatocyte); the manuscript figure shows a subset.

## Why digitizing was necessary

The obvious route — pull the vectors out of the manuscript SVGs — does not work. The files
in `mlcbCANDI_figs_svg/` are raster wrappers: they contain `<image>` and `<pattern>`
elements and no plottable paths. Embedding the PNGs directly would have put a 404×243 blur
next to crisp vector artwork. So `digitize.py` reads the curves back off the rendered
images, with the axis transforms pinned to detected tick pixel positions, which means the
recovered values are the published ones rather than eyeballed.

Four things in that script exist because the naive version produced garbage:

1. **Composite onto white before reading.** Several figures carry an alpha channel;
   `.convert("RGB")` turns transparent margins black, which breaks axis detection.
2. **Saturation gate.** A pastel target colour at a loose RGB tolerance also matches grey
   gridlines. Coloured targets must be at least as saturated as the target; achromatic
   targets (the black "Observed" line) instead require neutrality and darkness, since a
   saturation floor would reject them outright.
3. **Longest contiguous run, not the column mean.** Several disjoint groups can match in
   one column — the curve, the legend, a crossing curve. Averaging them yields a
   non-monotonic mess.
4. **Rolling median in `build_ga.py`.** Where a curve crosses the dashed diagonal the colour
   match drops a pixel or two, leaving single-column notches. A width-5 median removes them
   without moving the curve.

## Design decisions worth keeping

Recorded because each one was arrived at by iterating and would otherwise be re-litigated.

- **SVG text is vector paths** (`svg.fonttype: "path"`), not `<text>`. With `"none"` the SVG
  merely names DejaVu Sans, and any browser lacking it substitutes a serif — the Pages page
  then stops matching the PNG and PDF. Costs ~110 KB and makes the text unselectable. The
  PDF keeps live embedded text (`pdf.fonttype: 42`) for editing in Illustrator.
- **The tensor encodes availability only** — one flat teal, one flat grey, no intensity
  variation. Signal intensity first appears on the sliced heatmap, where it belongs. Face
  shading (front 100%, right 60%, top 40%) is 3D lighting, not data.
- **Missing experiments are threads, not squares.** The top and right faces are ruled into
  lanes running the full depth, so a missing experiment reads as absent along the entire
  genome. The front-face squares are those same threads seen end-on.
- **Six assays, not eight.** At eight the window tracks were 0.20 in tall and the confidence
  band was ~12 px on screen — too small to read as an interval. Six gets the band to roughly
  double that. The cube, slice and window all use the same six for consistency.
- **The predicted mean is deliberately wrong.** It is the underlying signal times a slowly
  varying multiplicative error, and the observed track carries sharp features the smooth
  prediction cannot follow. Without that the band has nothing to do and the panel does not
  make its point. Compare `mlcbCANDI_figs_png/unc_dist_h3k4me3.png`.
- **Draw order in the output tracks:** band, then the measurement on top of it, then the
  mean. Painting the band last hides the measurement entirely.
- **"State-of-the-art" is scoped to signal structure.** CANDI leads on Spearman across all
  assays but underperforms on Pearson for most, because it compresses dynamic range. Panel B
  of the manuscript imputation figure is the one to have ready.

## Layout mechanics

The workflow strip is drawn in a **single axes** with `set_aspect("equal")` and data
coordinates `0..100 × 0..46`. Heatmaps, track panels and the model column are `inset_axes`
positioned in those same data coordinates. This was the third attempt; a gridspec of
separate panels made the cabinet projection skew with the panel aspect ratio and made every
arrow a guess.

Consequence: the strip is **width-limited**. Its height is `width / (100/46)`, so making the
top gridspec row taller only opens whitespace. To give the strip more of the page, widen the
figure or change the `0..46` y-extent — not the height ratios.

`figsize` and `height_ratios` are tuned together so the result row is exactly small enough
that no vertical slack appears between the two. Changing one means re-checking the other.
