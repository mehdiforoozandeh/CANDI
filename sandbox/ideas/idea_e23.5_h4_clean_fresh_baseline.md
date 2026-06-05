# E23.5-H4 — Clean fresh encoder baseline with all fixes and new defaults

Status: staged
Parent: E23 (e23i: best fresh config), MaskStem CLOZE fix (2026-05-16)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

The best fresh encoder configuration (e23i: pre_conv + xtransformers + raw_meta_tgt, combined_loss=0.7269) was run with two known confounds:

1. **MaskStem CLOZE sentinel leakage** — `MaskStem.forward` only detected MISSING=-1, not CLOZE=-2. CLOZE values (-2) leaked through the conv tower as "present" data, giving the model an unfair information advantage (the conv could trivially detect which assays were masked). Fixed 2026-05-16.

2. **Stale defaults** — e23i used raw_meta_tgt (ordinal assay_id), pre_conv FiLM (not per_conv_and_transformer), and did not preserve_assay_id.

All 22 E23 runs used the leaky MaskStem code. We need a clean baseline with the fixed code and all promoted defaults to establish the true fresh encoder capability.

## Hypothesis

The clean baseline with fixed MaskStem and new defaults (separate embed no-LN, per_conv_and_transformer FiLM, preserve_assay_id) will establish the true fresh encoder performance floor. If combined_loss improves or stays comparable to e23i (0.7269), the CLOZE leak was not a major confound. If it degrades significantly, the leak was providing useful gradient signal that the model relied on.

## Planned Intervention

Config: pure `jepa_default.yaml` defaults with `model_type: fresh`.

```yaml
model_type: fresh

# All new defaults automatically apply:
# fresh.film_mode: per_conv_and_transformer
# fresh.cond_source: meta_tgt_embed
# fresh.cond_embed_shared: separate
# fresh.meta_embed_layernorm: true
# fresh.pred_meta_embed_layernorm: false
# fresh.transformer_type: xtransformers
# training.masking.preserve_assay_id: true
# data.regime: type2_loci
# training.masking: p_full_assay=1.0, p_full_loci=0.0
```

No code changes needed — MaskStem CLOZE fix was already applied 2026-05-16. New defaults were promoted above.

## Verifiables

- **Primary comparison:** e23i (combined_loss=0.7269, cov_cond=158, enc_er=18.4, runtype=0.024).
- Validate if: combined_loss ≤ 0.75 (CLOZE leak was not a major confound). Geometry metrics comparable or improved vs e23i.
- Disvalidate if: combined_loss > 0.80 (CLOZE leak was artificially helping the fresh encoder). This would mean all E23 fresh encoder results were confounded.
- v2 geometry gate: likely FAIL (all prior fresh runs fail), but track whether cov_condition_number improves vs e23i's 158.
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B UMAP.

## Risks / Watch-outs

- **Multiple simultaneous changes.** This run changes 5 things vs e23i: (1) MaskStem fix, (2) per_conv_and_transformer instead of pre_conv FiLM, (3) embedded instead of raw conditioning, (4) separate embed with no-LN, (5) preserve_assay_id. If performance changes, attribution is ambiguous. However, the purpose is to establish a clean baseline for the new default stack, not single-axis attribution.
- **Embedded conditioning may cost 2-2.5% combined_loss** (E23 batch 2 evidence). The other changes may or may not offset this.
- If this run is substantially worse, a follow-up single-axis run with ONLY the MaskStem fix (all other settings matching e23i) would disambiguate.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- W&B run: TBD

## Findings

**Superseded** by clean_ab_fresh_enc (SLURM 40548216). DualAttention test deferred until clean A/B encoder result reviewed. See [`synthesis_clean_ab_encoder.md`](synthesis_clean_ab_encoder.md).
