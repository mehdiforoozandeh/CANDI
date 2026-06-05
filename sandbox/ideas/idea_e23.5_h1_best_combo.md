# E23.5-H1 — Best-of-all combo: CANDI encoder + fresh xfm predictor + pred_hidden=16

Status: staged
Parent: E21 (e21o), E19 (e19u), E23 (promoted defaults)
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

Three independently validated winners have never been tested together:
1. CANDI encoder — definitive winner over fresh (FJ15, 4-8× on runtype sensitivity)
2. Fresh transformer predictor — e21o: unique late enc_er peak at ep=155, best UMAP
3. pred_hidden=16 — e19u: enc_er=23.5 (highest 200-ep run), collapse-resistant (FJ10)

Additionally, all prior candi+fresh_transformer runs (e21o) used raw_meta_tgt conditioning (treats assay_id as ordinal). The new default uses a separate MetadataEmbedding (no LN) for proper categorical handling.

## Hypothesis

Combining all three winners with the new promoted defaults should produce the best JEPA encoder checkpoint to date: the pred_hidden=16 bottleneck (FJ10) prevents the gamma overcompensation seen in e21o (gamma=1691), while the transformer predictor's cross-position attention improves over the MLP in e19u. Separate embedded conditioning (no LN) provides proper categorical handling without killing predictor AdaLN.

Expected: `runtype_last > 0.70`, `enc_er_last > 23`, structured biological UMAP, no collapse through 200 epochs. This run is the primary Stage 2 checkpoint candidate.

## Planned Intervention

Config deltas vs `jepa_default.yaml`:
```yaml
model_type: candi

jepa:
  predictor_type: fresh_transformer
  pred_hidden_dim: 16               # FJ10 bottleneck
  pred_cond_source: meta_tgt_embed  # new default: separate embed
  pred_meta_embed_dim: 32
  pred_meta_embed_layernorm: false   # predictor embed: no LN
  pred_mask_cond_type: meta_tgt     # FJ7: target metadata conditioning

# All other defaults from jepa_default.yaml:
# data.regime: type2_loci
# training.masking: p_full_assay=1.0, preserve_assay_id=true
# jepa.lambda_sigreg: 0.5
# training.epochs: 200
```

**Required code changes** (in `train_jepa.py`):
- When `jepa.predictor_type=fresh_transformer` and `jepa.pred_cond_source=meta_tgt_embed`:
  create a separate `MetadataEmbedding(use_layernorm=False)` for the predictor,
  compute `cond_dim = (signal_dim + 1) * pred_meta_embed_dim` instead of `4 * (signal_dim + 1)`.
- Pass the predictor MetadataEmbedding to `CANDIJepa` (new constructor parameter).
- In `CANDIJepa.forward`, call `pred_metadata_embedding(meta_tgt)` to get embedded
  conditioning, flatten to `[B, cond_dim]`, and pass to the predictor.

## Verifiables

- Validate if: `runtype_last ≥ 0.70`, `enc_er_last ≥ 23`, structured UMAP, no collapse (last/best ratio < 1.5 on combined_loss). AdaLN gamma active but not explosive (50 < gamma_last < 500).
- Disvalidate if: `runtype_last < 0.50` (embedded conditioning loses too much signal), or enc_er collapses below 18 before ep=150 (pred_hidden=16 insufficient with transformer predictor).
- Comparison targets: e21o (candi+fresh xfm, raw_meta, no bottleneck) and e19u (candi+MLP, raw_meta, pred_hidden=16).
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B UMAP.

## Risks / Watch-outs

- **Embedded conditioning may reduce gamma vs raw.** E23 batch 3 showed separate-no-LN embed gives gamma ≈ 54-58 vs raw's 114-144 on the fresh encoder. With the CANDI encoder (which provides richer gradients, FJ14), gamma should be higher, but the gap may persist.
- **pred_hidden=16 + transformer predictor is untested.** e19u used the legacy MLP with pred_hidden=16. The transformer predictor has higher base capacity (attention + FFN). The 16-dim bottleneck may be too aggressive for the transformer, or it may interact differently with AdaLN-zero.
- **cond_dim changes.** Switching from raw (cond_dim=36) to embedded (cond_dim=288) is an 8× increase in AdaLN linear layer parameters. Any improvement could partially reflect capacity, not representation quality.

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- W&B run: TBD

## Findings

**Superseded** by clean A/B pair (2026-05-19). See [`synthesis_clean_ab_encoder.md`](synthesis_clean_ab_encoder.md). pred_hidden=16 component deferred.
