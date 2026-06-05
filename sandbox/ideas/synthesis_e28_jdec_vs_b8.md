# E28 JEPA Decoder vs B8 Reconstruction Baseline

Status: synthesis (read-only)
Parents: [`idea_e28_jepa_decoder_training.md`](idea_e28_jepa_decoder_training.md), [`idea_b8_baseline_e7_e13.md`](idea_b8_baseline_e7_e13.md)
Linked from: EXPERIMENTS.md
Date: 2026-05-22

## Headline conclusions

1. B8 remains the reconstruction-quality winner by the cornerstone ranker: `baseline_E7_E13` has `quality_score=5.6812`, while the best E28 `jdec` run is `jdec_e27_lam005_40908829_sens_a` at `quality_score=6.2880`. B8 beats every current E28 run in pairwise `rank_runs.py` output. Confidence: High.
2. The gap is broad, not one bad head: B8 best epoch has `eval_losses/pval_imp_loss=0.4583`, `count_imp_loss=1.7157`, `peak_imp_loss=0.0773`, while the best `jdec` run has `0.5161`, `1.7341`, `0.0773`; denoising losses are also worse (`pval_obs_loss=0.5285` vs B8 `0.2450`, `count_obs_loss=1.0258` vs `0.8765`, `peak_obs_loss=0.0787` vs `0.0570`). Confidence: High.
3. The best `jdec` signal is not metadata depth sensitivity. `training_metadata_probes/depth_count_ratio` remains failure-level near 1.0 for all runs: B8 best/last `1.0048/0.9992`; best `jdec` best/last `1.0046/1.0010`; `jdec_e27_lam01_40908830_sens_a` reaches only `1.0111`. Confidence: High.
4. The metadata-related improvement is mainly read-length perturbation sensitivity, not run-type or depth. B8 last `training_metadata_probes/readlen_mse=1.65e-08`; `jdec_e27_lam01_40908830_sens_a` last `4.09e-05` and type1 lam01 partial last `1.41e-05`. Run-type perturbation stays tiny in all `jdec` runs (`~1e-11`) and below B8 last `2.66e-06`. Confidence: Medium.
5. Among E27 checkpoints, `e27_lam005_40908829` is the current Stage 2 reference candidate. It is the best type2 `jdec` by ranker (`quality_score=6.2880`) and best type1 partial run (`quality_score=6.7077`) among the type1 matrix. Confidence: Medium.

## Cross-run quantitative table

| run | regime | status | quality_score ↓ | best total_loss ↓ | imp_count_pearson ↑ | imp_pval_pearson ↑ | imp_peak_auroc ↑ | depth_count_ratio last | readlen_mse last |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_E7_E13` | `type1_chr19` | walltime-killed at 294/400 | **5.6812** | **3.0448** | **0.4126** | **0.3294** | **0.8030** | 0.9992 | 1.65e-08 |
| `jdec_e27_lam005_40908829_sens_a` | `type2_loci` | complete 200/200 | 6.2880 | 3.5777 | 0.1964 | **0.1861** | **0.7309** | 1.0010 | 3.93e-07 |
| `jdec_e27_lam01_40908830_sens_a` | `type2_loci` | complete 200/200 | 6.6433 | 3.7533 | 0.1209 | 0.1371 | 0.7079 | **1.0060** | **4.09e-05** |
| `jdec_e27_lam02_40908831_sens_a` | `type2_loci` | complete 200/200 | 6.3381 | 3.6356 | 0.1892 | 0.1829 | 0.7262 | 1.0001 | 5.59e-06 |
| `jdec_e27_lam005_40908829_type1_sens_a` | `type1_chr19` | partial 163/200 | 6.7077 | 3.8181 | 0.1277 | 0.1440 | 0.6601 | 1.0001 | 7.12e-06 |
| `jdec_e27_lam01_40908830_type1_sens_a` | `type1_chr19` | partial 130/200 | 6.9419 | 3.9626 | 0.0896 | 0.1455 | 0.6790 | 0.9991 | 1.41e-05 |
| `jdec_e27_lam02_40908831_type1_sens_a` | `type1_chr19` | partial 129/200 | 6.9213 | 3.9738 | 0.0925 | 0.1459 | 0.7005 | 0.9987 | 1.06e-06 |

## Per-run grad / stability table

| run | eval points | NaN/Inf | diverged | depth metadata status | mean sec/epoch |
|---|---:|---:|---|---|---:|
| `baseline_E7_E13` | 58 | 0 | no | ignored | 49.0705 |
| `jdec_e27_lam005_40908829_sens_a` | 40 | 0 | no | ignored | 19.4093 |
| `jdec_e27_lam01_40908830_sens_a` | 40 | 0 | no | ignored | 19.5078 |
| `jdec_e27_lam02_40908831_sens_a` | 40 | 0 | no | ignored | 19.4952 |
| `jdec_e27_lam005_40908829_type1_sens_a` | 32 | 0 | no | ignored | 30.0180 |
| `jdec_e27_lam01_40908830_type1_sens_a` | 26 | 0 | no | ignored | 29.2272 |
| `jdec_e27_lam02_40908831_type1_sens_a` | 25 | 0 | no | ignored | 29.6492 |

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| `jdec_e27_lam005_40908829_sens_a` | Frozen JEPA `z_pred` can support competitive signal-space decoding. | Partial: best E28 run but worse than B8 on `quality_score=6.2880` vs `5.6812`. | High |
| `jdec_e27_lam01_40908830_sens_a` | Higher SIGReg strength might produce a better Stage 2 decoder substrate. | Rejected: `quality_score=6.6433`, worse than lam005 and lam02. | High |
| `jdec_e27_lam02_40908831_sens_a` | Higher SIGReg strength might recover geometry enough for better decoding. | Partial: tied-ish with lam005 by ranker within E28, but still behind B8. | Medium |
| type1 `jdec` matrix | Type1 training/eval might close the baseline comparison because B8 is type1. | Rejected so far: partial type1 quality scores `6.7077`, `6.9419`, `6.9213` trail B8 `5.6812`. | Medium |

## Implications for next batch

1. Reference checkpoint: use `e27_lam005_40908829` unless final type1 epochs overturn the current ordering. Predicted move: best chance to improve `eval_losses/count_imp_loss` and `imp_peak_auroc`; cost: 3-4 runs for `jdec_b/c/d/e`.
2. Run head-isolation correctness on `e27_lam005_40908829`: `count_only`, `pval_only`, `peak_only`. Predicted move: if decoder towers are truly independent, per-head metrics should match or exceed `jdec_a` branch metrics; cost: 3 runs.
3. Test predictor fine-tuning (`decoder.freeze_mode=predictor_decoder`) before encoder fine-tuning. Predicted move: improve reconstruction losses if the frozen predictor's `z_pred` is not aligned with NLL decoding; cost: 1-2 runs.
4. Test grouped deconv only after head-isolation. Predicted move: if cross-assay deconv is leaking or smoothing assay identity, grouped deconv may improve per-assay `imp_*` medians; cost: 1 run.
5. Do not use current depth metadata probes as evidence that JEPA fixed metadata collapse. Stop condition: abort any metadata-fix claim until `training_metadata_probes/depth_count_ratio >= 1.5`, preferably 3-5.

## Standing findings (carried forward)

- F1 — Depth metadata ignored: still open. This synthesis adds E28 evidence that `jdec` does not fix depth response; all `depth_count_ratio` values remain near 1.0.
- F5 — Pval head interferes with count + peak training: still open. E28 does not test head isolation yet; `jdec_a` still jointly trains all three decoder towers.
- F7 — Pval Gaussian NLL variance collapse: mitigated by `gaussian_var_min=0.1`, still relevant. E28 is stable so far (`diverged=no`, `NaN/Inf=0`) but pval quality trails B8.
- F8 — E7 single-shot decoder FiLM is the best multi-head architecture to date: still open. B8's E7+E13 reconstruction baseline remains ahead of frozen JEPA decoder training.
- FJ7 — meta_tgt conditioning improves run-type sensitivity during JEPA pretraining: not reproduced in Stage 2 output probes. E28 `training_metadata_probes/runtype_mse` remains tiny.
- FJ10 — pred_hidden=16 as best JEPA encoder knob: relevant background only. E27 checkpoints tested here are lambda variants, not a pred_hidden ablation.

## Caveats and limits

- The three type1 `jdec` runs were still running when this synthesis was written; they had 25-32 eval points, enough for direction but not final status.
- B8 and E28 are not controlled single-axis comparisons. Config differences include objective family, model architecture, dataset regime for type2 runs, masking mixture (`p_full_assay=0.8`, `p_full_loci=0.5` in B8 vs assay-only in E28), batch size, AMP, and number of epochs.
- B8 was walltime-killed at 294/400, but it already has a lower best total loss and better quality score than all current E28 runs.
- Metadata `readlen_mse` being larger is sensitivity, not necessarily correctness. It may indicate useful metadata use, oversensitivity, or changed output scale.
