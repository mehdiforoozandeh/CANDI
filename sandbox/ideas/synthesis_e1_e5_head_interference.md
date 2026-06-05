# E1-E5 Synthesis - Head Interference + LR Floor

Status: synthesis (read-only)
Parents: [`idea_b7_log1p_type1.md`](idea_b7_log1p_type1.md), [`idea_e1_lrfloor_low.md`](idea_e1_lrfloor_low.md), [`idea_e2_head_count_only.md`](idea_e2_head_count_only.md), [`idea_e3_head_pval_only.md`](idea_e3_head_pval_only.md), [`idea_e4_head_peak_only.md`](idea_e4_head_peak_only.md), [`idea_e5_head_count_peak.md`](idea_e5_head_count_peak.md)
Linked from: each parent idea file's Findings section.
Date: 2026-04-27

This document aggregates findings from the E1-E5 sweep into one place. Every claim cites a metric key and value extracted from `metrics.jsonl` or printed by the bundled log-observability scripts.

## Headline conclusions

1. **Pval head interferes with count + peak training.** Removing pval (E5) and removing pval+peak (E2) both lift count's imputation Pearson dramatically; pval is doing more harm than good for the encoder representation that count uses.
2. **Pval head is the most fragile head and cannot be trained alone.** E3's pre-clip grad-norm max is 1408 (vs B7's 501), `inspect_training_steps.py` flags late-stage divergence, and the run ends with a degenerate `total_loss=0`.
3. **Peak head depends on auxiliary heads.** E4's `imp_peak_auroc_gw` is essentially chance (0.4830); peak overfits trivial denoising without count and pval as cross-assay anchors.
4. **Lowering `min_lr_ratio` from 0.1 to 0.01 produces no measurable change under log1p.** E1 ties B7 on `quality_score`, every per-branch loss, and the entire grad-norm trajectory.

## Cross-run quantitative table

Best-epoch values from `metrics.jsonl`. `—` means the head's loss is not emitted because that branch was muted (loss_weight=0).

| run | count_imp_loss | count_obs_loss | pval_imp_loss | pval_obs_loss | peak_imp_loss | peak_obs_loss | imp_pval_pearson_gw (peak) | imp_count_pearson_gw (peak) | imp_peak_auroc_gw (peak) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B7** (log1p anchor)         | 1.7972 @ 14  | 1.1262 @ 119 | 0.6175 @ 74  | 0.4872 @ 79  | 0.4530 @ 194 | 0.4256 @ 199 | **0.1730** | 0.0592 | **0.5530** |
| E1 (lrfloor_low)              | 1.7972 @ 14  | 1.1267 @ 119 | 0.6192 @ 74  | 0.4867 @ 79  | 0.4895 @ 164 | 0.4616 @ 164 | 0.1729     | 0.0571 | 0.5420     |
| E2 (count only)               | **1.7367 @ 59**  | **1.0491 @ 84**  | —            | —            | —            | —            | 0.0340     | **0.1154** | 0.4629     |
| E3 (pval only)                | —            | —            | **0.5860 @ 179** | 0.5467 @ 194 | —            | —            | 0.1168     | -0.0146 | 0.4949     |
| E4 (peak only)                | —            | —            | —            | —            | 0.5539 @ 194 | **0.3218 @ 194** | 0.0041     | -0.0168 | 0.4830     |
| E5 (count+peak, pval muted)   | **1.7177 @ 99**  | 1.0791 @ 84  | —            | —            | 0.4591 @ 184 | **0.2692 @ 179** | 0.0182     | 0.0508 | 0.4861     |

**Reading the table.**
- Bold = best across runs for that metric.
- E5 owns the lowest `count_imp_loss` (1.7177) — better than E2's count-only setting and 4.4% better than B7.
- E5 also owns the lowest `peak_obs_loss` and is essentially tied with B7 on `peak_imp_loss`.
- B7 retains the imputation lead on `imp_pval_pearson_gw` and `imp_peak_auroc_gw` — pval gradients carry information specifically useful for these two quantities.
- E2 owns the count Pearson lead by a wide margin (0.1154 vs B7's 0.0592).

## Pre-clip gradient norm trajectories (from `inspect_training_steps.py`)

| run | total_loss range | grad_pre_clip median | p95 | max | clip_fraction (running) |
|---|---|---:|---:|---:|---:|
| B7 | first=5.75 min=1.27 last=2.96 | 33.4 | 220.9 | 500.97 | 0.968 |
| E1 | first=5.75 min=1.27 last=1.66 | 32.8 | 220.8 | 488.04 | 0.968 |
| E2 | first=1.57 min=1.22 last=2.86 | **4.89** | **12.7** | **20.0** | 0.990 |
| E3 | first=2.66 min=-0.50 last=0.00 | 35.4 | **327.0** | **1408.3** | 0.713 |
| E4 | first=1.47 min=0.00 last=0.00 | 4.85 | 26.2 | 84.8 | 0.677 |
| E5 | first=3.08 min=1.25 last=3.45 | 7.89 | 25.7 | 82.0 | 0.986 |

**Reading the table.**
- E2's grad-norm collapses ~7× vs B7 — count head alone is operating in a much smaller magnitude regime.
- E3 has the largest p95 and max grad spikes — pval-only is intrinsically unstable.
- E5 sits between E2 and B7, consistent with peak head adding a moderate gradient signal back.
- E1 and B7 are statistically indistinguishable on every grad-norm summary statistic.

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| E1 | Lowering `min_lr_ratio` from 0.1 to 0.01 reduces late training pressure / improves stability | **Rejected.** Tied with B7 on every metric and on grad trajectory. Log1p alone covers the F3 stability concern. | High |
| E2 | If multi-head competition hurts counts, count-only training should improve count metrics | **Confirmed.** count_imp_loss 3.4% better; `imp_count_pearson_gw` ~2× better; grad-norm collapses ~7×. | High |
| E3 | If pval gradients are diluted by other heads, pval-only training should improve pval metrics | **Rejected for Pearson; partial for loss.** Slight loss improvement but Pearson regresses 32%; head goes unstable. | High |
| E4 | Peak-only training clarifies whether peak AUROC is multi-head-limited or objective-limited | **Objective-limited rejected; isolation-limited confirmed.** Peak AUROC drops to chance without auxiliary heads. | High |
| E5 | If pval is noisy or dominant, count+peak training should improve count and peak branch behavior | **Confirmed for count, mixed for peak.** Best count_imp; peak_imp tied with B7; AUROC slightly worse. | Medium-High (walltime-killed at 189/200 epochs) |

## Implications for next batch

In priority order:

1. **Rerun E5 with 4 h walltime** to confirm 200-epoch trends and remove the walltime caveat from the strongest finding. Cost: 1 run.
2. **Sweep `pval_weight ∈ {0.0, 0.1, 0.3, 1.0}` with count+peak active.** Goal: find the smallest pval contribution that recovers B7's imputation Pearson and peak AUROC without sacrificing E5's count_imp gain. Cost: 4 runs.
3. **Distribution sweep on the pval head: `model.signal_dist ∈ {laplace, student_t}`** under multi-head log1p. Goal: test whether pval fragility is distributional (heavy tails) rather than head-architectural. Cost: 2 runs.

Total: ~7 runs, ~21 GPU-h.

## Standing findings (carried forward)

- **F1 (depth metadata ignored)** still open across all 7 runs. `depth_count_ratio` ranges 0.998-1.005 (target 4.0).
- **F3 (late-stage divergence under low LR floor)** appears resolved-by-log1p for chr19-scale runs. Re-flag if a future non-log1p or longer-horizon run diverges.

## Caveats and limits

- E1 was walltime-killed at epoch 165, but its best epoch (74) is identical to B7's. The tie verdict is robust at the cornerstone level.
- E5 was walltime-killed at epoch 189. The peak head's `imp_peak_auroc_gw` was still trending upward at the cut; absolute conclusions about peak AUROC vs B7 should be treated as **medium** confidence until rerun.
- Head-isolation runs (E2/E3/E4/E5) cannot be ranked by the cornerstone `quality_score` because muted-head losses are not emitted. All comparisons here are per-branch and explicit about that.
- F1 is not addressable from these runs alone — no E-run varied any depth-metadata-related axis. Carry F1 forward unchanged.
