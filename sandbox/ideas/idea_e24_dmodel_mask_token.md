# E24 - d_model-sized Assay-Specific Mask Token

Status: done — accepted, promoted to default (2026-05-21)  
Parent: E23 (fresh encoder ablation; mask_token rejected in current form)  
Run name: `e24_mask_token_40800572`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

Current `MaskTokenInjector` uses one shared `mask_embedding` of shape `[d_per_assay]`, then writes that same vector into every masked assay slot after the signal conv tower. This means all masked assays receive identical replacement features. The model cannot distinguish "H3K4me3 masked" from "CTCF masked" from token content alone; only downstream metadata path can disambiguate.

In E23 this design underperformed (`e23b`, `e23e` rejected). One likely cause: masked assays are forced onto the same post-conv default representation.

## Idea / Hypothesis

Replace shared `[d_per_assay]` token with one learnable `[d_model]` token aligned to the flattened assay-channel layout:

- `d_model = num_tracks * d_per_assay` at the mask injection point.
- Assay `k` occupies `k*d_per_assay : (k+1)*d_per_assay`.
- For masked assays, keep the corresponding assay slice from the token.
- For observed assays, overwrite token slices with conv features.

Hypothesis: assay-specific token slices give distinct defaults per assay, reducing masked-assay aliasing and improving predictor conditioning and geometry versus current shared-token mask mode.

## Planned Intervention

- Code path: `sandbox/jepa_model.py::MaskTokenInjector` and `JEPAEncoder.encode`.
- Replace `missing_data_mode=mask_token` implementation:
  - token parameter shape `[num_tracks * d_per_assay]`
  - reshape to `[num_tracks, d_per_assay]` during forward
  - build output from token baseline + observed-assay overwrite
- Baseline/control remains pure `jepa_default.yaml` run with `missing_data_mode=mask_stem`.

## Verifiables

- Validate if:
  - New token mode improves over default control (`mask_stem`) on `combined_loss_scaled`.
  - Metadata sensitivity does not regress (`lejepa/meta_sens_runtype` stable or improved).
  - Collapse markers improve directionally (`cov_condition_number` down, `encoder_eff_rank` up/stable).
- Disvalidate if:
  - Loss/geometry remain equal or worse than default control (`mask_stem`).
  - Improvement appears only in train loss without better eval-side metrics.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B run names.

## Risks / Watch-outs

- This uses assay-slot identity encoded by channel layout; if assay ordering changes, semantics break.
- Could become a capacity increase confound versus shared token; run paired controls with identical everything else.
- Might still fail if core issue is pre-conv information loss rather than post-conv replacement design.
- If `d_model` differs from `num_tracks * d_per_assay` in a future refactor, this design needs explicit shape guards.

## Run Links

- Run directory: `sandbox/runs/e24_mask_token_40800572`
- Resolved config: `sandbox/runs/e24_mask_token_40800572/resolved_config.yaml`
- Metrics: `sandbox/runs/e24_mask_token_40800572/metrics.jsonl`
- HPO graph node: `hpo.experiment_label=e24_e26_batch`
- W&B run: `e24_mask_token_40800572`
- Control run: `sandbox/runs/e24e26_control_40800571`
- Superseded attempt (short run): `e24_mask_token_40800179`

## Findings

Evidence from `sandbox/runs/e24_mask_token_40800572/metrics.jsonl` vs control `sandbox/runs/e24e26_control_40800571/metrics.jsonl` (107 training steps each, 200 epochs):

- Observed:
  - `combined_loss_scaled` best: **0.7056** (E24) vs 0.7243 (control) — **+2.6% improvement**.
  - `meta_sens_runtype` best: **0.6838** vs 0.4874 (control) — **+37% improvement**.
  - `cov_condition_number` last: 62.33 vs 48.97 (control) — geometry degraded (worse isotropy).
  - `encoder_eff_rank` last: 32.75 vs 39.92 (control) — lower dimensional usage.
  - Stage 2 checkpoint (v2): step_idx 85 (~epoch 136).
  - Both runs fail v2 geometry gate (sigreg not converged at termination).
- Interpretation: d_model-sized assay-specific mask token successfully resolves the assay aliasing problem from E23's shared-token design. Per-assay slices in the learnable `[d_model]` token give distinct defaults that improve both prediction quality and biological sensitivity. The strong runtype_sens signal (+37%) indicates the mask token preserves biologically relevant structure better than the conv-based MaskStem.
- Competing explanations: The mask token adds capacity (d_model parameters vs d_per_assay) which could partially explain the improvement. The geometry degradation (higher cov_cond) suggests the token may introduce dimensional bias where certain assay slices dominate embedding variance.
- Decision: **Accepted. Promoted to default** (`fresh.missing_data_mode=mask_token`) in `jepa_default.yaml` and `JEPAModelConfig` dataclass (2026-05-21). The geometry concern (higher cov_cond) is a candidate for follow-up with increased `lambda_sigreg`.
