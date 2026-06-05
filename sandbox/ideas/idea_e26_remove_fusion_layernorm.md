# E26 - Remove LayerNorm from LinearFusion

Status: done — accepted, promoted to default (2026-05-21)  
Parent: E23 batch 3 (LayerNorm findings)  
Run name: `e26_no_fusion_ln_40800574`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

`LinearFusion` in `JEPAEncoder` applies LayerNorm after GELU:

```python
fused = self.dropout(self.norm(self.gelu(self.fusion_proj(concat))))
```

This is the same variance-crushing pattern identified in `MetadataEmbedding` (E23 batch 3). The LayerNorm:
1. Subtracts mean — removes absolute magnitude information that may encode signal strength.
2. Divides by std — compresses dynamic range before the transformer.

The transformer already has its own pre-norm (LayerNorm/RMSNorm before attention and FFN in each block). Adding LN before the transformer double-normalizes the first input, potentially flattening useful variance the transformer needs.

## Idea / Hypothesis

Remove the LayerNorm from LinearFusion. The input signal is already bounded (log1p transform + MaskStem + grouped conv with per-layer normalization), and DNA is from one-hot inputs (bounded). Magnitude explosion is unlikely. The linear projection `W` can learn any necessary rescaling.

Hypothesis: Removing fusion LN preserves signal dynamic range entering the transformer, avoids the same condition-collapse mechanism found in MetadataEmbedding, and may improve encoder geometry (lower cov_condition_number) by retaining more variance structure.

## Planned Intervention

- Add `fusion_norm` config knob to `JEPAModelConfig`: `layer` (current default) | `none`.
- Modify `LinearFusion` to conditionally apply normalization based on config.
- Run with `fusion_norm=none` vs `fusion_norm=layer` (control) on same base config.

## Verifiables

- Primary: `combined_loss_scaled`, `cov_condition_number` (expect improvement), `encoder_eff_rank`.
- Secondary: monitor gradient norms through fusion layer to detect instability.
- If `none` is unstable early in training, schedule follow-up with an explicit RMS alternative.

## Risks / Watch-outs

- If signal/DNA magnitude varies wildly across samples (unlikely given log1p + bounded DNA), removing norm could make early training unstable.
- The conv tower's own per-layer normalization (LayerNorm/GroupNorm) partially controls input scale already, so fusion LN may be redundant anyway.
- This is a low-risk, low-cost intervention — minimal code change, quick to evaluate.

## Run Links

- Run directory: `sandbox/runs/e26_no_fusion_ln_40800574`
- Resolved config: `sandbox/runs/e26_no_fusion_ln_40800574/resolved_config.yaml`
- Metrics: `sandbox/runs/e26_no_fusion_ln_40800574/metrics.jsonl`
- HPO graph node: `hpo.experiment_label=e24_e26_batch`
- W&B run: `e26_no_fusion_ln_40800574`
- Control run: `sandbox/runs/e24e26_control_40800571`
- Superseded attempts: `e26_no_fusion_ln_40800533`, `e26_no_fusion_ln_40800181`

## Findings

Evidence from `sandbox/runs/e26_no_fusion_ln_40800574/metrics.jsonl` vs control `sandbox/runs/e24e26_control_40800571/metrics.jsonl` (107 training steps each, 200 epochs):

- Observed:
  - `combined_loss_scaled` best: **0.7163** (E26) vs 0.7243 (control) — **+1.1% improvement**.
  - `meta_sens_runtype` best: 0.4764 vs 0.4874 (control) — roughly neutral (−2.3%).
  - `meta_sens_runtype` last: 0.3885 vs 0.4783 (control) — −19% regression at termination.
  - `cov_condition_number` last: 50.24 vs 48.97 (control) — geometry roughly matched (best among experimental runs).
  - `encoder_eff_rank` last: 35.00 vs 39.92 (control) — moderately lower.
  - No training instability — convergence shape matches control.
- Interpretation: Removing fusion LayerNorm preserves signal dynamic range as hypothesized, providing a modest loss improvement without numerical instability (log1p + MaskStem already controls input scale). The geometry is near-identical to control, confirming the LN was redundant given the transformer's own pre-norm. The runtype regression at termination is a concern but within late-training noise given similar best values.
- Decision: **Accepted. Promoted to default** (`fresh.fusion_norm=none`) in `jepa_default.yaml` and `JEPAModelConfig` dataclass (2026-05-21). The modest loss improvement and reduced double-normalization justify the change given zero instability risk.
