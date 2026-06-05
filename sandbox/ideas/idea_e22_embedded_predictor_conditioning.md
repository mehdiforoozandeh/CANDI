# E22 - Embedded Predictor Conditioning for JEPA

Status: idea  
Parent: E21 (e21o — candi encoder + fresh transformer predictor, raw meta_tgt conditioning)  
Run name: TBD  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

e21o conditions the JEPA fresh transformer predictor on **raw** `meta_tgt` — the 4 covariates (`log2(seq_depth)`, `assay_id`, `read_length`, `run_type`) flattened per assay into a `[B, 4*(F+1)]` vector fed directly to the AdaLN linear layer. This is problematic because:

1. **`assay_id` is multi-class categorical but treated as ordinal.** The AdaLN linear layer can only learn a single slope for `assay_id`, implying e.g. H3K27ac (id=2) is "between" H3K4me3 (id=1) and CTCF (id=3). With 8 sandbox assays the linear layer may hack around this; with 35 production assays it is untenable.
2. **Scale mismatch across covariate types.** `read_length` (35–150) dominates `run_type` (0–1) and `assay_id` (0–7) in magnitude, biasing the AdaLN linear projection and wasting capacity on compensating weights.
3. **No cross-field interaction capacity.** The predictor needs to distinguish "H3K27ac at low depth" from "CTCF at low depth" — with raw scalars this requires the AdaLN linear layer to learn the interaction; an embedding with nonlinearity handles it upstream.

Despite these issues, e21o is the best run in the E21 batch (`runtype_last=0.708`, active AdaLN gamma peaking at 1647, best visual UMAP). The question is whether proper metadata embedding can improve on this already-strong result.

## Idea / Hypothesis

Replace raw `meta_tgt` conditioning with a **separate** (not shared) `MetadataEmbedding` that provides proper categorical lookup for `assay_id`, scale normalization across all 4 fields, and nonlinear interaction capacity — without gradient coupling to the encoder's metadata embeddings.

**Prediction:** Embedded conditioning should match or exceed e21o's quality (`runtype_last ≥ 0.70`, active AdaLN) and produce a cleaner conditioning signal that scales to 35 production assays. The separate embedding avoids the gradient conflict risk present in a shared pathway (the encoder optimizes embeddings for encoding biology; the predictor needs them for task specification — these objectives can conflict).

## Planned Intervention

- Submit/config path: `sandbox/slurm/submit_e22a_candi_freshxfm_embed_cond.sbatch`
- Run name: `e22a_candi_freshxfm_embed_cond_<SLURM_JOB_ID>`
- Parent run: e21o (`sandbox/runs/e21o_candi_enc_fresh_pred_39819437`)
- Config/code deltas vs e21o:
  - Plumb `cond_source` and `cond_embed_shared` into the `CANDIJepa` / `train_jepa.py` harness (currently only wired for `FreshJepa`).
  - Use `cond_source=meta_tgt_embed`, `cond_embed_shared=separate` so a fresh `MetadataEmbedding` module is instantiated for the predictor.
  - `cond_dim` changes from `4 * (F+1) = 36` (raw) to `(F+1) * metadata_embed_dim` (e.g. `9 * 32 = 288` with default embed_dim=32).
  - All other settings identical to e21o: `model_type=candi`, `jepa.predictor_type=fresh_transformer`, `jepa.lambda_sigreg=0.5`, `data.regime=type2_loci`, `training.epochs=200`.

## Verifiables

- Validate if: `runtype_last ≥ 0.70` (matches e21o), AdaLN gamma active (gamma_last > 100), enc_er trajectory does not collapse below 20 before ep=150. UMAP shows structured biological geometry (visual check on W&B).
- Disvalidate if: `runtype_last < 0.50` or AdaLN goes dead (gamma_last < 10) — would indicate the embedding adds harmful complexity or gradient coupling despite being separate.
- Stretch goal: `runtype_last > 0.80` or late enc_er peak beyond ep=155 (surpasses e21o).
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B run with UMAP.

## Risks / Watch-outs

- **Confound with cond_dim change.** Embedding increases cond_dim from 36 to 288 (8× more parameters in AdaLN linear layer). Any improvement could be from higher capacity, not better representation. If e22a wins, a follow-up with `metadata_embed_dim=4` (keeping cond_dim≈36) would disambiguate.
- **Gradient coupling even with separate embedding.** The separate `MetadataEmbedding` still receives gradients from the predictor loss. If the predictor loss dominates, the embedding may learn task-specification features that are good for prediction but unrelated to biological metadata structure. This is acceptable (the predictor embedding has a different job than the encoder embedding) but worth monitoring.
- **Single seed.** Like all E21 runs, conclusions from one seed should be confirmed before committing to architectural changes.
- **Comparability.** Only directly comparable to e21o (same encoder, same predictor architecture, same everything except conditioning representation).

## Run Links

- Run directory: TBD
- Resolved config: TBD
- Metrics: TBD
- SLURM logs: TBD
- HPO graph node: TBD
- W&B run: TBD

## Findings

Do not fill this from memory. Use concrete artifact evidence and cite metric keys/values.

- Observed: TBD
- Interpretation: TBD
- Competing explanations: TBD
- Decision: TBD
