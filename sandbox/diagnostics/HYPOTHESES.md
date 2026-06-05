# Synthetic Overfit — Hypotheses & Ideas

## H1 — Architecture can overfit tiny deterministic data (P1)
**Status:** **confirmed** — flat P1, rel_mae=2.6%, pearson=0.99

## H2 — Metadata collapse is decoder-side (Q5)
**Status:** **confirmed** — P3 default dcr=1.30; P3 offset dcr=4.0
**Conclusion:** Raw-μ NB head cannot enforce depth on masked imputation; FiLM alone insufficient (E01, E18)

## H3 — Per-assay mask tokens enable imputation (P3)
**Status:** **confirmed** with depth-offset — imp rel_mae≈2–8%, pearson_imp high

## H4 — DSF denoising requires depth in target metadata (P4)
**Status:** **confirmed** with depth-offset — P4 pass; stochastic DSF OK (E14)

## H5 — E29 library-size offset fixes depth collapse
**Status:** **confirmed** — dcr≈4.0 on P3/P4/P5, stochastic NB (E05), SGD (E12), all clip norms
**Ablation:** Works with enc/dec FiLM off (E06–E08) — offset is sufficient
**Next:** production integration (user: not yet promoted)

## H6 — DNA tower unnecessary for count scale
**Status:** **partial** — P5 flat+motif pass; spatial+motif fails (E15)

## H7 — Optimizer/LR fixes Q5 without offset
**Status:** **rejected** — E09/E10 fail to learn; E11/E18 good fit but dcr<2; E13 explodes

## H9 — Raw E29 offset transfers to real chr19
**Status:** **rejected** — R02/R08 dcr≈1.0 (log2 depth ~24, counts ~0–6)

## H11 — Centered offset survives multi-epoch iterable training
**Status:** **confirmed** — R20 offset dcr≈4.0 epochs 0–2

## H12 — count_peak multi-head needs centered offset on real data
**Status:** **confirmed** — R19 dcr≈1, R19b dcr≈4
- Longer spatial budget / bigger model for L=768
- Per-step FiLM scale logging in training loop (snapshot at end only today)
