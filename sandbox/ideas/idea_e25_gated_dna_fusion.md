# E25 - Gated DNA Fusion

Status: done — rejected (2026-05-21)  
Parent: E23 (fresh encoder ablation)  
Run name: `e25_gated_fusion_40800573`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

Current `LinearFusion` in `JEPAEncoder` concatenates signal and DNA features then projects through a single linear layer:

```python
fused = self.fusion_proj(torch.cat([signal, dna], dim=-1))  # → GELU → LN → dropout
```

This treats DNA information identically at every genomic position. In reality, DNA sequence context matters far more at regulatory elements (promoters, enhancers, CTCF sites) than at featureless intergenic regions. The linear projection has no mechanism to learn position-specific signal/DNA weighting.

## Idea / Hypothesis

Replace concatenation+projection with a gated fusion mechanism where DNA features control how much they contribute at each position:

```python
gate = sigmoid(W_gate(dna))              # [B, L2, d_signal] → values 0-1
signal_modulated = signal * gate         # DNA decides which signal dims pass
dna_contribution = W_dna(dna)            # DNA's own feature projection
output = signal_modulated + dna_contribution
```

Hypothesis: Gated fusion should improve representation quality at regulatory regions (where DNA is informative) while avoiding pollution of signal features at non-regulatory regions (where DNA adds noise). This should be visible as improved metadata sensitivity and potentially better geometry.

## Planned Intervention

- Add `fusion_mode` config knob to `JEPAModelConfig`: `linear` (current) | `gated`.
- Implement `GatedDNAFusion` module in `sandbox/jepa_model.py`.
- `GatedDNAFusion`:
  - `W_gate`: Linear(dna_dim, signal_dim) + Sigmoid
  - `W_dna`: Linear(dna_dim, signal_dim)
  - Output: `signal * gate + dna_proj`
  - No output projection (requires `d_model == signal_dim`)
  - GELU + LayerNorm + dropout (match linear path; isolate fusion mechanism from E26)
- Compare gated vs linear fusion on same base config (current `jepa_default.yaml` control).

## Verifiables

- Primary: `combined_loss_scaled`, `cov_condition_number`, `encoder_eff_rank`.
- Secondary: `meta_sens_runtype` (expect improvement if DNA-gating helps at regulatory loci).
- Control: identical config with `fusion_mode=linear`.

## Risks / Watch-outs

- If gate collapses to all-ones or all-zeros, fusion degenerates to signal-only or additive-only.
- Gated fusion changes parameter count slightly — ensure fair comparison.
- If DNA tower features are weak (poorly trained, low expressiveness), the gate won't have useful signal to work with. May need to evaluate DNA tower quality independently.

## Run Links

- Run directory: `sandbox/runs/e25_gated_fusion_40800573`
- Resolved config: `sandbox/runs/e25_gated_fusion_40800573/resolved_config.yaml`
- Metrics: `sandbox/runs/e25_gated_fusion_40800573/metrics.jsonl`
- HPO graph node: `hpo.experiment_label=e24_e26_batch`
- W&B run: `e25_gated_fusion_40800573`
- Control run: `sandbox/runs/e24e26_control_40800571`
- Superseded attempts: `e25_gated_fusion_40800532`, `e25_gated_fusion_40800180`

## Findings

Evidence from `sandbox/runs/e25_gated_fusion_40800573/metrics.jsonl` vs control `sandbox/runs/e24e26_control_40800571/metrics.jsonl` (107 training steps each, 200 epochs):

- Observed:
  - `combined_loss_scaled` best: 0.7230 (E25) vs 0.7243 (control) — **+0.2% improvement** (within noise).
  - `meta_sens_runtype` best: 0.4046 vs 0.4874 (control) — **−17% regression**.
  - `cov_condition_number` last: 61.46 vs 48.97 (control) — geometry degraded.
  - `encoder_eff_rank` last: 32.21 vs 39.92 (control) — lower dimensional usage.
- Interpretation: Gated fusion provided no meaningful improvement over linear fusion. The sigmoid gate did not learn useful position-specific DNA weighting — either the DNA tower features were insufficiently informative, or the added gating complexity hurt geometry without compensating in prediction quality.
- Decision: **Rejected.** Default remains `fusion_mode=linear`.
