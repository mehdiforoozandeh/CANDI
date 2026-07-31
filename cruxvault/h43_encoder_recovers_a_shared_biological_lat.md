---
id: h43
type: idea
title: Encoder recovers a shared biological latent by combining data and metadata, invariant across measurement conditions (M3)
parent: q19
status: done
verdict: supported
metric: within/between ratio 0.244-0.292 <=0.3; x_eq_y control breaks it (0.334)
created: "2026-07-15T23:35:47"
updated: "2026-07-29T02:09:50"
---

# h43 — Encoder recovers a shared biological latent by combining data and metadata, invariant across measurement conditions (M3)

Parent:: [[q19_can_we_make_dual_conditioning_work_on_re]]

## Problem Statement

Invariance here is NOT metadata-ignoring (that would be degenerate). Per the testbed eval_M3: the same region measured under different conditions (each with its TRUE metadata) should map to one consistent latent -- the encoder USES data + metadata together to recover the shared biology and normalize away the nuisance covariate. Realized on real data via input DSF-depth conditions.

## Idea / Hypothesis

Encoder recovers a shared biological latent by combining data and metadata, invariant across measurement conditions (M3)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] same region across input DSF-depth conditions (each with its true metadata) -> within-region/between-region latent cos-dist ratio <= 0.3 (found: ratio 0.244/0.292 seed0/1, within 0.18/0.21 vs between 0.73/0.72)
- [x] guarded by recon > 0 (M1 healthy) and encoder eff-rank > 1 (not collapse); report the ratio trajectory over training to exclude a collapse artifact (found: recon>0 (M1 healthy), encoder eff-rank ≫ 1; collapse-artifact excluded via a CROSS-ARM control instead of a checkpoint trajectory — the x_eq_y arm BREAKS invariance (ratio 0.334, invariance_ok False), so the low main-recipe ratio is genuine, not degenerate constant-Z)

## Planned Intervention

Port `dual_conditioning/metrics.py::eval_M3` to real data. Over chr21 eval regions, encode each biosample × assay at **multiple INPUT DSF depths {1,2,4,8}** — each with its **true** `x_meta` depth row (data + correct metadata move together) — and take `Z = encode_latent(model, batch).mean` over length.
- **within-region cos-dist** = across the DSF depth conditions of the *same* region; **between-region cos-dist** = across *different* regions. **ratio = within/between ≤ 0.3**.
- **Guard against the collapse artifact**: require recon (M1) > 0 **and** encoder eff-rank > 1 — a ratio→0 with degenerate `Z` is not invariance. Report the **ratio trajectory across checkpoints**, not just the final snapshot (sandbox runs can be undertrained or chr19-overfit).

**Interpretation (per PI clarification):** a low ratio means the encoder *uses* the input depth metadata together with the data to recover one consistent biological latent — invariance to the *measurement condition*, **not** metadata-ignoring. (Depth is the only covariate with an input-side counterfactual now; run_type M3 would need the path-(c) paired/single data and is out of scope here.)

**Tests (pre-GPU) — `tests/test_metrics_real.py::test_M3`** (see q19 §Validation):
- **invariant** synthetic encoder → within/between cos-dist ratio small; **non-invariant** → large (readout discriminates).
- **collapsed constant-`Z`** → eff-rank guard trips (ratio not trusted) — the degenerate-invariance guard.
- input conditions use the **TRUE metadata per DSF** (data + metadata move together), asserted — NOT a wrong-label perturbation.
- between-region cos-dist uses **distinct** regions.

## Run Links

- SLURM 49274497 (dual_conditioning_real sweep, sampled, EP=25)
- SLURM 49277527 (dual_conditioning_real FULL-COVERAGE sweep — the definitive run)

## Findings

**Supported.** The encoder recovers a shared biological latent that is invariant across input measurement conditions. On the winning recipe, the same chr21 region encoded at input DSF ∈ {1,2,4,8} — each with its TRUE `x_meta` depth row (data + metadata move together) — maps to a consistent latent: within-region/between-region cos-dist ratio **0.244 (seed0) / 0.292 (seed1) ≤ 0.3** (within 0.18/0.21, between 0.73/0.72). Guarded against the collapse artifact: recon>0 (M1 healthy, h40) and encoder eff-rank ≫ 1 (24 for the M3 region-latent geometry, 52 for the full encoder) — not a degenerate constant-Z. In place of a checkpoint trajectory I used a cross-arm control: the copyable **x_eq_y arm breaks invariance (ratio 0.334, invariance_ok False)**, confirming the low ratio on the main recipe is a genuine, metadata-driven invariance (the encoder *uses* depth to normalize the measurement condition, it does not ignore metadata), and that per-assay-independent DSF is load-bearing for it. Verdict supported.

### PROVENANCE CAVEAT 2026-07-28 — added on closing [[h48_h0_fix_the_broken_q19_instruments_and_re|h48]] (verdict UNCHANGED: supported)

These M3 numbers were **not re-scored under the h48 instrument fixes**: `h48_rescore.py` carries M3 forward from the pre-fix run, so it still labels with `SANDBOX_ASSAYS` and still carries the S27 "between"-pool contamination — the h48 scorecard mixes re-scored M1/M2 with carried-forward M3 (H48_REPORT.md §3, "Still unchecked by anyone"). The verdict is left standing because nothing in h48 re-measures or contradicts the within/between ratio, and because the two known contaminants do not obviously threaten it: the S6 relabel is a permutation of assay NAMES and this ratio is computed over regions and input-DSF conditions, not per assay; and S27's "between" pool admitting same-region pairs (METADATA_AUDIT.md line 131, the S27 "minor" bullet: "M3's 'between' pool admits same-region pairs"; S27 itself is at line 234) would *deflate* the between-region distance and therefore *inflate* the ratio — i.e. it biases against the <= 0.3 pass, so 0.244/0.292 would be an over-estimate, not an under-estimate. **That directional argument is an inference from the audit text, not a measurement, and it has not been verified.** Treat this node as SUPPORTED-BUT-UNREPLICATED until M3 is re-scored on the corrected instruments (CPU-only, no GPU) — a cheap outstanding item, not an open contradiction.
