# E32 autoresearch — imp count R² vs Pearson disparity (May31 loop)

Status: synthesis (read-only)  
Parents: [`autoresearch_may31_r2vscorr_disparity.md`](autoresearch_may31_r2vscorr_disparity.md)  
Linked from: [`EXPERIMENTS.md`](EXPERIMENTS.md) (E32)  
Date: 2026-06-02

Artifacts: `sandbox/autoresearch/may31/results.tsv`, `results_legacy.tsv`, `run.log`, branch `autoresearch/may31`, best keep commit **`be0d38e2`**.

---

## Headline conclusions

1. **V/B-natural eval metadata (A1) is necessary but not sufficient:** After the harness fix, best `imp_count_r2_gw` reached **+0.122** (commit `be0d38e2`) while `imp_count_r2_gw_canonical` stayed **−0.161** on the same run — a ~0.28 R² gap attributable to prompt scale. Competing explanation: canonical eval may still be mis-specified on a subset of assays; vb_natural is nonetheless required for meaningful imp R². **Confidence: High.**

2. **`imp_weight` is the dominant tunable in this pinned 5000-step budget:** Moving `imp_weight` from 0.5 → **0.59** (holding `obs=3.5`, `count=2`, `dc=22.5`, `dsf=off`, `mse_obs=0.2`) raised imp R² from **0.063 → 0.122** (+94%). Neighbors 0.57–0.62 and 0.585–0.595 were worse or unstable. **Confidence: High.**

3. **`depth_center=22.5` beats higher centers on this chr19 pin:** Session 1 exp21 (`dc=22.5`) first crossed positive imp R² (**0.103**); `dc=23–26` runs stayed negative or lower imp. Fine steps `22.4` / `22.52` regressed den or imp. **Confidence: Medium** (single pin, no multi-seed).

4. **Auxiliary MSE on imp (`lambda_mse_imp ≥ 0.02`) and aggressive May28-style reweighting (`imp=8`, `obs=0.5`) hurt or collapse imp R²** in session 2; `lambda_mse_imp=0.1+` was catastrophic. `lambda_mse_obs` at 0.25 also hurt imp. **Confidence: High.**

5. **Train/eval task gap is moderate at the best config, not the whole story:** Best run: `imp_r2=0.122`, `imp_r2_cloze_T=0.098`, gap **−0.025**. Some runs show large negative gaps (e.g. count=1.95: imp 0.115 vs cloze NaN/negative canonical), but the winning recipe does not require cloze-T to exceed vb eval. **Confidence: Medium.**

6. **Loop outcome vs spec validate gate: Partial, not Validate.** Spec target `imp_count_r2_gw > 0.15` with `den ≥ 0.35` was not met. Best: **imp=0.122**, **den=0.279**, DCR **3.93** (in band). Session 3 used a Pareto keep rule with **den floor 0.25** (not 0.35). **Confidence: High.**

---

## Cross-run quantitative table

Milestone rows from `results_legacy.tsv` + session-3 `results.tsv`. **Bold** = best in column among listed rows.

| commit | phase | **imp_r2** | **den_r2** | **dcr** | imp_pearson | cloze_T | canonical | description |
|--------|-------|------------|------------|---------|-------------|---------|-----------|-------------|
| f91dab7e | den | −0.552 | −0.280 | 3.974 | 0.372 | — | — | row0-style: dsf=off, neutral weights |
| 5f4d0210 | den | −2.322 | −1.500 | 3.959 | 0.259 | — | — | May28 weights imp=8 obs=0.5 |
| aca82557 | den | **0.103** | 0.242 | 3.999 | 0.398 | — | — | exp21 dc=22.5 obs=3.5 |
| 694ac68c | den | −0.589 | **0.335** | 3.986 | 0.388 | — | — | exp23 count=2 obs=3.5 dc=23 |
| 4c895221 | den | 0.063 | 0.259 | 4.002 | 0.331 | — | — | s3 replicate exp21 dc=22.5 |
| 67077967 | imp | 0.098 | **0.305** | 4.003 | 0.365 | 0.061 | −0.108 | s3 imp=0.6 |
| 7e362866 | imp | 0.115 | 0.310 | 3.970 | 0.402 | — | −0.344 | s3 imp=0.59 count=1.95 (near-miss) |
| **be0d38e2** | imp | **0.122** | 0.279 | 3.934 | 0.395 | 0.098 | −0.161 | **best keep** imp=0.59 dc=22.5 count=2 |

Winning `TrainConfig` (agent-editable fields only):

```
count_weight=2.0, obs_weight=3.5, imp_weight=0.59, depth_center=22.5
lambda_mse_imp=0.0, lambda_mse_obs=0.2, calib_loss=raw, dsf_sampling=off
signal_transform=log1p, clip_norm=0.5, adamax lr=1e-3
```

---

## Per-experiment outcome vs hypothesis

| Axis | hypothesis | outcome | confidence |
|------|------------|---------|------------|
| A1 vb_natural eval | Canonical meta depresses imp R² | **Confirmed** — canonical ≪ vb at best keep | High |
| A2 cloze vs vb task gap | Cloze-T ≫ vb implies task mismatch dominates | **Partial** — gap small (−0.025) at best; large gaps on some configs | Medium |
| B4 imp/obs weights | Higher imp_weight raises imp R² | **Confirmed** — optimum ≈0.59 on this pin | High |
| B1/B3 lambda_mse_imp | Small MSE aux improves calibration | **Rejected** — λ≥0.02 hurt; λ≥0.1 destroyed imp | High |
| C1 dsf_sampling=off | Identity eval alignment helps | **Partial** — kept from early exp1; not re-beating vs later tuned recipes alone | Low |
| depth_center | EIC-aligned center (22–24) helps | **Confirmed** — 22.5 best; 23+ worse on imp | Medium |
| count_weight | Trade imp vs den along count axis | **Partial** — sharp peak at 2.0; 1.95 trades +0.007 imp for +0.03 den | Medium |
| D1 signal_transform=none | Encoder/head mismatch drives R² gap | **Rejected** — none hurt badly in s3 | High |
| May28 imp=8 | Transfer May28 winner | **Rejected** — session 1 exp2 | High |
| Validate gate (imp>0.15, den≥0.35) | Full success | **Inconclusive / Partial** — imp 0.122, den 0.28 | High |

---

## Implications for next batch

1. **Promote to real v2 run (E33, separate from AR):** Translate `be0d38e2` weights to `sandbox/configs/` + `train_candi_v2` — 200 ep, full type1_chr19, same depth_offset head. Predict: imp R² > 0 on chr21 holdout; may not reach 0.15 without more data/steps. **Cost:** 1 GPU job.

2. **Two-objective follow-up (optional):** `count=1.95`, `imp=0.59` gave imp=0.115 / den=0.310 — if product needs den≥0.30, run a short confirm + multi-seed. **Cost:** 2–3 AR runs or 1 v2 job.

3. **Do not spend more AR steps on `lambda_mse_imp` or May28 extreme weights** unless architecture changes. **Cost:** 0 (avoid).

4. **Architecture tier (D2/D4 groupwise / per-assay affine):** Not explored in AR (VRAM/t scope). If imp R² plateaus on full v2 run, prioritize D4 per-assay scale head. **Cost:** harness extension + 5–10 AR runs.

5. **Commit harness fixes outside `train.py`:** `prepare.py` cloze footer key, `keep_rule.py` legacy baseline, session-3 Pareto TSV — one human commit on `autoresearch/may31` before promotion. **Cost:** negligible.

---

## Standing findings (carried forward)

No existing `F*` in `log-observability/FINDINGS.md` tagged Q11. New standing finding for downstream runs:

- **F-E32-1 (open → mitigated on pin):** Imp count Pearson ~0.4 with imp R² ≤ 0 on E31 v2 runs. **E32 AR mitigated on pinned chr19/21 subset:** vb_natural eval + `imp_weight≈0.59` → imp R² **0.122**; rank–magnitude gap narrowed but not closed to validate threshold. Full-data v2 confirmation pending.

Touches META **Q11** (partially resolved). Related to E31 depth_center but E32 did not wait on E31 sweep completion.

---

## Caveats and limits

- **~60+ agent commits** across three sessions (`results_legacy.tsv` + `results.tsv`); all **single-seed**, **5000 steps**, **32 pinned train batches** on chr19 only.
- **No `rank_runs.py` composite** — AR used custom Pareto keep (session 3: imp↑ with den≥0.25, DCR∈[3.25,4.75]). Original spec used den gate **0.35**; best den **0.279** never crossed production gate on this pin.
- **Metric variance:** Neighboring hyperparams (e.g. imp=0.592) occasionally collapsed imp R² — treat 0.122 as best point estimate, not a stable plateau.
- **Harness edits during loop:** Session 3 allowed one-time fixes to `prepare.py`, `eval_pass.py`, `keep_rule.py`, `agent_step.py` (not agent loop scope); document in promotion PR.
- **Not promoted to `sandbox/candi_v2/` or configs** — per spec, translation is a separate experiment.

---

## Loop chronology (short)

| Session | Focus | Best imp_r2 | Notes |
|---------|-------|-------------|-------|
| 1 | DSF off, depth_center, obs/count weights, mse_obs | 0.103 (exp21) | 31 exps in legacy |
| 2 | lambda_mse_imp sweep, calib modes | −0.25 (s2 baseline) | imp aux harmful |
| 3 | Pareto keep, imp_weight fine sweep | **0.122** (be0d38e2) | imp=0.59 wins |

Total logged rows: **59 legacy + 28 session-3 TSV** (some s3 rows duplicate legacy tail).
