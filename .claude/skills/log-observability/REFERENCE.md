# Log Observability — Reference

Detailed rules behind [SKILL.md](SKILL.md). Read on demand; the workflow checklist in SKILL.md
points here for every section a deeper question lands in.

## Parsing order (strict)

1. `resolved_config.yaml` first — establishes what the run was supposed to do.
2. All rows in `metrics.jsonl` — both `kind="epoch"` and `kind="training_step"` (post-2026-04).
3. SLURM stderr — check for `DUE TO TIME LIMIT`, OOM, NaN signals; especially when `epoch` count
   is well below `training.epochs`.
4. W&B — only as higher-frequency detail. If W&B disagrees with `metrics.jsonl`, treat the jsonl
   as authoritative and report the conflict.

## Metric language contract

### `metrics.jsonl` record kinds

- `kind: "epoch"` — one per evaluated epoch. Fields: `epoch`, `global_step`, `epoch_seconds`,
  optionally `eval_metrics/*`, `eval_metrics_median/*`, `eval_losses/*`, `training_metadata_probes/*`,
  optionally `early_stop_triggered` / `early_stop_best_total_loss` / `early_stop_strikes`.
- `kind: "training_step"` — emitted every `training.training_stats_jsonl_every_n_steps` optimizer
  steps (default 200, 0 disables). Fields: `epoch`, `global_step`, `training_stats/*`,
  `training_losses/*`, `training_grad_norms/*`. This is the *offline* source of truth for grad-norm
  and clip-fraction history. Older runs lack this kind — fall back to W&B.

### Metric families

- `training_stats/*` — step / lr / total_loss / grad_pre_clip_norm / grad_clipped (0/1) / clip_cap /
  running and windowed clip fraction.
- `training_losses/*` — unweighted per-branch losses (count_obs / count_imp / pval_obs / pval_imp /
  peak_obs / peak_imp).
- `training_grad_norms/*` — per-branch gradient norms (W&B + jsonl since 2026-04).
- `training_metadata_probes/*` — first keys: `depth_count_ratio`, `runtype_mse`, `readlen_mse`.
  Plus the `meta_sensitivity_*_mse` family.
- `eval_metrics/*` — batch-level pooled diagnostics (cross-assay pool per batch, then mean across batches). Keys like `*_gw` are fast legacy summaries; **do not treat as interchangeable with medians**.
- `eval_metrics_median/*` — **preferred decision metrics** for run comparison: compute per assay on gw scope, skip invalid assays, median across assays (production-matched). Logged by `train_candi_v2.py` and JEPA decoder trainer. `summarize_runs.py` may **fallback** legacy runs without median keys to `eval_metrics/*_gw` — flag those comparisons as lower confidence.
- `eval_losses/*` — eval-time per-branch losses + total weighted loss.

### Forbidden / excluded

- Any reconstruction or eval MSE/RMSE family — removed in the 2026-04 refactor. If present, the
  run predates the refactor; flag and downgrade confidence.
- `eval_metrics/*_r2_1obs` — sparse target → unstable R2. Excluded from success criteria. Use
  Spearman/Pearson on 1obs slices instead. Genome-wide `*_r2_gw` remains valid.

## Quantitative interpretation contracts

Thresholds derived from probe definitions in `sandbox/eval.py`. Do not invent your own.

| metric | healthy | warning | failure |
|---|---|---|---|
| `training_metadata_probes/depth_count_ratio` (default lo=22, hi=24, +2 in log2 = 4× depth) | 3.0 ≤ x ≤ 5.0 | 1.5 ≤ x < 3.0 or 5.0 < x ≤ 8.0 | x < 1.5 (depth metadata ignored) or x > 8.0 |
| `eval_losses/total_loss` divergence | last ≤ best × 1.1 | last ≤ best × 1.5 | last > best × 1.5 (diverged) |
| NaN/Inf scan over eval families | 0 | n/a | any |
| Walltime utilization (Σ `epoch_seconds` / SLURM walltime) | ≥ 70% | 40–70% | < 40% (eval/probe overhead too high) |

## No-hallucination rules

1. No claim without explicit observed key/value evidence.
2. No causal claim without at least one competing hypothesis listed.
3. If a key is missing, write `unknown` instead of guessing. In particular:
   - If `kind="training_step"` rows are absent, all grad-norm / clip claims are `unknown` from
     disk; redirect to W&B or note the limit.
   - If `*_r2_gw` is missing, do not fall back to `*_r2_1obs`.
4. With ≤ 3 eval points, downgrade certainty to Low.
5. Report anomalies (NaN/Inf, missing family, walltime kill, OOM, R2 explosion on sparse slices)
   *before* any optimization advice.

## Confidence rubric

- **High** — direct metric evidence across multiple epochs, stable trend, multiple corroborating
  sources.
- **Medium** — direct evidence but short run, noisy trend, or single source.
- **Low** — sparse points, conflicting signals, or missing families.

Every recommendation must include a confidence level.

## Insight extraction template

Use this exact structure per insight:

1. **Observed** — quote metric keys and values.
2. **Interpretation** — what this likely means.
3. **Competing explanations** — at least one alternative.
4. **Action** — one bounded experiment.
5. **Expected delta** — what should move if the hypothesis is correct.
6. **Stop condition** — when to abort the experiment.

## Cornerstone Decision Rule (is run A better than run B?)

The deterministic ranking contract for the autoresearch loop. Stop at the first tier that decides;
never skip ahead.

### Tier 1 — Primary quality: the 6 eval losses

CANDI's main goal is imputation, so imputation losses outweigh denoising 2:1. All six are
unweighted per-branch losses (lower is better) and directly comparable across runs.

- Imputation triple (priority): `eval_losses/pval_imp_loss`, `count_imp_loss`, `peak_imp_loss`.
- Denoising triple (secondary): `eval_losses/pval_obs_loss`, `count_obs_loss`, `peak_obs_loss`.

```
quality_score(run) =
    2.0 * (pval_imp + count_imp + peak_imp)
  + 1.0 * (pval_obs + count_obs + peak_obs)
```

Evaluate at the run's *best* epoch (= epoch with minimum `eval_losses/total_loss`), not the last
epoch — this de-couples ranking from divergence (handled in Tier 2).

**Decision rule**: A > B iff
- A's `quality_score` is lower than B's by > 1% relative, AND
- A is no worse than B by > 5% relative on **any individual** imp loss
  (per-branch worst-case guard; prevents gaming the composite by sacrificing one imp branch).

### Tier 1b — Eval metric corroboration (veto only)

After Tier 1 picks a winner, sanity-check that the win is real (not "fit the mean"):

- Imp triple: `eval_metrics_median/imp_pval_pearson`, `imp_pval_spearman`, `imp_peak_auroc`.
- Den triple: `eval_metrics_median/den_pval_pearson`, `den_pval_spearman`, `den_peak_auroc`.
- Count corroboration (noisy but kept): `eval_metrics/imp_count_pearson_gw`,
  `eval_metrics/den_count_pearson_gw` for legacy runs; prefer
  `eval_metrics_median/imp_count_pearson` and `eval_metrics_median/den_count_pearson` when present.

**Veto rule**: if A wins Tier 1 but any *imp* corroboration metric drops by > 10% relative vs B,
downgrade to "tied — investigate". Den-metric drops do not veto, only flag. Tier 1b never
*promotes* a run; it only blocks an unsupported Tier 1 win.

### Tier 2 — Stability gate (eligibility before any ranking)

A run is **ineligible** — regardless of Tier 1 — if any of:

- Divergence flag: `last(eval_losses/total_loss) > 1.5 × best(eval_losses/total_loss)`.
- NaN/Inf count > 0 across `eval_metrics/*`, `eval_metrics_median/*`, and `eval_losses/*`.
- Walltime status = killed by SLURM walltime AND not all runs in the comparison were killed at the
  same `global_step` (walltime-killed runs are comparable only at a matched step).

Apply this gate *before* computing `quality_score`. If both runs fail the gate for the same reason,
do not declare a winner — escalate to the user.

### Tier 3 — Health probes (attribution, not ranking)

Used to explain *why* Tier 1 moved; produces the "why" half of the Insight Extraction Template:

- `training_metadata_probes/depth_count_ratio` (target ≈ 4.0; standing finding F1).
- `training_stats/grad_pre_clip_norm` p95 (from `kind="training_step"` rows).
- `training_stats/grad_clipped_frac_running` mean (> 0.5 → optimizer fighting clip cap).
- Per-branch `training_grad_norms/<branch>` p95 — if `quality_score` improves but one imp branch's
  grad norm explodes, the win is fragile; recommend a follow-up that probes that branch.

### Tier 4 — Efficiency tiebreaker

Only when Tier 1 reports a tie (within 1% on `quality_score`):

- `global_step` at the run's best `eval_losses/total_loss` (fewer = faster convergence; use steps
  not epochs because steps/epoch can vary across regimes).
- Mean `epoch_seconds` (wall-time efficiency sanity check).

### Cornerstone field list (ranker reads only these)

Mirror this list in `scripts/rank_runs.py`. Anything outside is *evidence*, not a *decision input*.
To promote a metric into the cornerstones, edit this section first; never silently extend.

```
TIER1_LOSSES = [
    "eval_losses/pval_imp_loss",   # imp_weight=2
    "eval_losses/count_imp_loss",  # imp_weight=2
    "eval_losses/peak_imp_loss",   # imp_weight=2
    "eval_losses/pval_obs_loss",   # obs_weight=1
    "eval_losses/count_obs_loss",  # obs_weight=1
    "eval_losses/peak_obs_loss",   # obs_weight=1
]
TIER1B_VETO_METRICS = [
    "eval_metrics_median/imp_pval_pearson",
    "eval_metrics_median/imp_pval_spearman",
    "eval_metrics_median/imp_peak_auroc",
]
TIER1B_FLAG_METRICS = [
    "eval_metrics_median/den_pval_pearson",
    "eval_metrics_median/den_pval_spearman",
    "eval_metrics_median/den_peak_auroc",
    "eval_metrics_median/imp_count_pearson",
    "eval_metrics_median/den_count_pearson",
]
TIER2_STABILITY = [
    "divergence_flag (last/best ratio of eval_losses/total_loss)",
    "nan_inf_count over eval_*",
    "walltime_killed and not at matched global_step",
]
TIER3_ATTRIBUTION = [
    "training_metadata_probes/depth_count_ratio",
    "training_stats/grad_pre_clip_norm (p95)",
    "training_stats/grad_clipped_frac_running (mean)",
    "training_grad_norms/<branch> (p95)",
]
TIER4_EFFICIENCY = [
    "global_step at min(eval_losses/total_loss)",
    "mean(epoch_seconds)",
]
```
