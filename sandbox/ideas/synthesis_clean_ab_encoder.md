# Clean A/B Encoder Comparison: CANDI vs Fresh under E23.5 Defaults

Status: synthesis (read-only)
Parents: resolves film_mode confound from [`synthesis_ab_encoder_dropout_comparison.md`](synthesis_ab_encoder_dropout_comparison.md)
Linked from: [`EXPERIMENTS.md`](EXPERIMENTS.md)
Date: 2026-05-19

Both runs used pure `jepa_default.yaml` with only `model_type` overridden: `per_conv_and_transformer`, `xtransformers`, `meta_tgt_embed`, separate no-LN predictor embed, fixed MaskStem.

---

## Headline conclusions

1. **Fresh encoder wins on combined_loss and geometry** under matched E23.5 defaults (0.7141 vs 0.7277; cov_cond 63.9 vs 81.2; enc_er 31.9 vs 26.4).
2. **CANDI encoder retains runtype sensitivity advantage** (runtype_best 0.751 vs 0.469) — consistent with FJ15, but residual dropout confound (fresh 0.02 vs CANDI 0.1) may inflate fresh's combined_loss win.
3. **`model_type: fresh` promoted as JEPA default** (2026-05-19); CANDI available via explicit override.

---

## Cross-run table

| Run | SLURM | combined_loss | cov_cond | enc_er | runtype_best |
|---|---|---:|---:|---:|---:|
| clean_ab_candi_enc | 40548215 | 0.7277 | 81.2 | 26.4 | **0.751** |
| clean_ab_fresh_enc | 40548216 | **0.7141** | **63.9** | **31.9** | 0.469 |

---

## Implications

- E23.5-H1/H2/H4 superseded: H1/H2/H4 checklist entries point here instead of separate runs.
- Biological sensitivity vs loss/geometry tradeoff remains open; see [`synthesis_ab_encoder_dropout_comparison.md`](synthesis_ab_encoder_dropout_comparison.md) for dropout sensitivity.
