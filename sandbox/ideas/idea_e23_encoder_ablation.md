# E23 - Ablation-ready JEPA encoder redesign

Status: done (e23a-h batch 1 + e23i-p batch 2 complete; analysis 2026-05-16)  
Parent: E21 (e21m, e21n, e21o, e21p)  
Run name: e23[a-h, i-p]_...  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

E21's 2x2 matrix isolated the root issue to the fresh encoder path: substituting the original CANDI encoder fixes representation quality, while keeping the fresh encoder degrades it. The current fresh encoder differs from CANDI in multiple coupled places (signal transform semantics, missing-data path, FiLM placement, fusion norm, transformer block type), making root-cause attribution ambiguous.

## Idea / Hypothesis

Build a single ablation-ready fresh encoder in `sandbox/jepa_model.py` with full CLI-switchable knobs that can exactly emulate the CANDI encoder path or selectively toggle individual differences. With this harness we can run controlled deltas and identify which components are necessary to match CANDI-level quality.

## Planned Intervention

- Replace encoder-related pieces in `sandbox/jepa_model.py` with an E23-configurable implementation.
- Keep JEPA predictor classes and JEPA training API intact.
- New `fresh.*` knobs:
  - `missing_data_mode`: `mask_stem` | `mask_token`
  - `film_mode`: `per_conv` | `per_conv_and_transformer` | `post_conv` | `pre_conv`
  - `conv_norm`: `layer` | `group` | `batch`
  - `dna_pool_order`: `late` | `early`
  - `transformer_type`: `dual` | `xtransformers`
- Signal transform fixed to CANDI semantics: apply `log1p` / `arcsinh` only where `(src != -1) & (src != -2)`.
- Wire new knobs through `sandbox/train_jepa.py` and `sandbox/jepa_config.py`.
- Submitted runs `e23a` to `e23h` with one- or two-knob deltas (batch 1).
- Added CLI control for predictor conditioning (`cond_source`, `cond_embed_shared`) and `predictor_type=legacy_mlp` support.
- Fixed MaskStem CLOZE sentinel leakage bug (only detected MISSING=-1, not CLOZE=-2).
- Submitted runs `e23i` to `e23p` combining winners and testing conditioning/predictor variants (batch 2).

## Submitted Run Matrix (2026-05-15)

- `e23a` - job `40025385` - encoder control: `mask_stem`, `per_conv`, `layer`, `late`, `dual`.
- `e23b` - job `40025386` - missing-data ablation only: `mask_token` (all other knobs as control).
- `e23c` - job `40025387` - FiLM placement ablation only: `post_conv` (all other knobs as control).
- `e23d` - job `40025388` - transformer ablation only: `xtransformers` (all other knobs as control).
- `e23e` - job `40025389` - fresh-combo: `mask_token` + `post_conv` + `xtransformers`.
- `e23f` - job `40025390` - extra FiLM capacity: `per_conv_and_transformer` (all other knobs as control).
- `e23g` - job `40025391` - DNA pooling order ablation only: `early` (all other knobs as control).
- `e23h` - job `40025392` - FiLM placement ablation only: `pre_conv` (all other knobs as control).

## Verifiables

- Primary: `lejepa/meta_sens_runtype`, `lejepa/encoder_eff_rank`, `lejepa/pred_loss`, `lejepa/cos_sim_ctx_tgt`, UMAP geometry.
- Gate: each ablation compared to `e23a` control under same train regime and masking.
- Success criterion: one or more fresh-style settings match or improve the CANDI-like control on geometry and metadata-sensitivity without collapse signatures.

## Risks / Watch-outs

- `mask_token` mode requires metadata/signal availability agreement; mismatch is now a hard error.
- `conv_norm=group` can change optimization scale vs `layer`; compare only within controlled deltas.
- `per_conv_and_transformer` adds extra conditioning capacity; gains may reflect capacity, not mechanism.

## Run Links

- `e23a` (`40025385`)
  - run dir: `sandbox/runs/e23a_encoder_control_40025385`
  - resolved config: `sandbox/runs/e23a_encoder_control_40025385/resolved_config.yaml`
  - metrics: `sandbox/runs/e23a_encoder_control_40025385/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23a_enc_ctrl_40025385.out`, `sandbox/slurm_logs/e23a_enc_ctrl_40025385.err`
  - W&B run name: `e23a_encoder_control_40025385`
- `e23b` (`40025386`)
  - run dir: `sandbox/runs/e23b_mask_token_only_40025386`
  - resolved config: `sandbox/runs/e23b_mask_token_only_40025386/resolved_config.yaml`
  - metrics: `sandbox/runs/e23b_mask_token_only_40025386/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23b_masktok_40025386.out`, `sandbox/slurm_logs/e23b_masktok_40025386.err`
  - W&B run name: `e23b_mask_token_only_40025386`
- `e23c` (`40025387`)
  - run dir: `sandbox/runs/e23c_postconv_film_only_40025387`
  - resolved config: `sandbox/runs/e23c_postconv_film_only_40025387/resolved_config.yaml`
  - metrics: `sandbox/runs/e23c_postconv_film_only_40025387/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23c_postfilm_40025387.out`, `sandbox/slurm_logs/e23c_postfilm_40025387.err`
  - W&B run name: `e23c_postconv_film_only_40025387`
- `e23d` (`40025388`)
  - run dir: `sandbox/runs/e23d_xtransformer_only_40025388`
  - resolved config: `sandbox/runs/e23d_xtransformer_only_40025388/resolved_config.yaml`
  - metrics: `sandbox/runs/e23d_xtransformer_only_40025388/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23d_xtfm_40025388.out`, `sandbox/slurm_logs/e23d_xtfm_40025388.err`
  - W&B run name: `e23d_xtransformer_only_40025388`
- `e23e` (`40025389`)
  - run dir: `sandbox/runs/e23e_fresh_combo_40025389`
  - resolved config: `sandbox/runs/e23e_fresh_combo_40025389/resolved_config.yaml`
  - metrics: `sandbox/runs/e23e_fresh_combo_40025389/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23e_freshcombo_40025389.out`, `sandbox/slurm_logs/e23e_freshcombo_40025389.err`
  - W&B run name: `e23e_fresh_combo_40025389`
- `e23f` (`40025390`)
  - run dir: `sandbox/runs/e23f_perconv_plus_transformer_film_40025390`
  - resolved config: `sandbox/runs/e23f_perconv_plus_transformer_film_40025390/resolved_config.yaml`
  - metrics: `sandbox/runs/e23f_perconv_plus_transformer_film_40025390/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23f_filmplus_40025390.out`, `sandbox/slurm_logs/e23f_filmplus_40025390.err`
  - W&B run name: `e23f_perconv_plus_transformer_film_40025390`
- `e23g` (`40025391`)
  - run dir: `sandbox/runs/e23g_dna_early_pool_40025391`
  - resolved config: `sandbox/runs/e23g_dna_early_pool_40025391/resolved_config.yaml`
  - metrics: `sandbox/runs/e23g_dna_early_pool_40025391/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23g_dnaearly_40025391.out`, `sandbox/slurm_logs/e23g_dnaearly_40025391.err`
  - W&B run name: `e23g_dna_early_pool_40025391`
- `e23h` (`40025392`)
  - run dir: `sandbox/runs/e23h_preconv_film_40025392`
  - resolved config: `sandbox/runs/e23h_preconv_film_40025392/resolved_config.yaml`
  - metrics: `sandbox/runs/e23h_preconv_film_40025392/metrics.jsonl`
  - slurm logs: `sandbox/slurm_logs/e23h_prefilm_40025392.out`, `sandbox/slurm_logs/e23h_prefilm_40025392.err`
  - W&B run name: `e23h_preconv_film_40025392`

### Batch 2 (2026-05-16) — conditioning, predictor, lambda ablations

- `e23i` (`40101731`) — pre_conv + xtfm + raw_meta_tgt conditioning (transformer predictor)
  - run dir: `sandbox/runs/e23i_preconv_xtfm_40101731`
- `e23j` (`40101732`) — pre_conv + xtfm + meta_tgt_embed (shared) conditioning (transformer predictor)
  - run dir: `sandbox/runs/e23j_preconv_xtfm_embedcond_40101732`
- `e23k` (`40101733`) — pre_conv + xtfm + lambda=0.75 + meta_tgt_embed (shared)
  - run dir: `sandbox/runs/e23k_preconv_xtfm_lam075_40101733`
- `e23m` (`40101735`) — per_conv_and_transformer + raw_meta_tgt conditioning
  - run dir: `sandbox/runs/e23m_filmplus_rawcond_40101735`
- `e23n` (`40101736`) — pre_conv + xtfm + separate MetadataEmbedding conditioning
  - run dir: `sandbox/runs/e23n_preconv_xtfm_sepembed_40101736`
- `e23p` (`40101737`) — pre_conv + xtfm + legacy_mlp predictor + raw_meta_tgt
  - run dir: `sandbox/runs/e23p_preconv_xtfm_legacymlp_40101737`

## Findings

### Batch 1 (2026-05-15)

All evidence from `extract_jepa_metrics.py` and `compare_configs.py`.

**Execution:** e23a-e23h all completed (ExitCode=0:0). e21o replica (40031166) also completed.

**Ranking (combined_loss_scaled, lower=better):** e23h (0.7135) > e23f (0.7169) > e23d (0.7336) > e23a-ctrl (0.7353) > e23g (0.7481) > e23e (0.7801) > e21o (0.7890) > e23b (0.7912) > e23c (0.7925).

**Per-ablation verdicts:**
- **FiLM placement** is the highest-impact knob. pre_conv (e23h: −3.0% combined, best SIGReg 0.996, best cov_cond 139) > per_conv_and_transformer (e23f: −2.5%, best runtype retention 0.168) > per_conv (control) > post_conv (e23c: +7.8%, worst). Pre-conv optimizes best; per_conv+xfm retains metadata best; post_conv is strictly harmful.
- **mask_token (e23b): REJECTED.** +7.6% combined_loss, +64% pred_loss, −74% runtype_best, −85% gamma. Strictly inferior to mask_stem. Root-cause analysis (2026-05-16): MaskTokenInjector has a design flaw — post-conv injection misses the conv+FiLM stack, and the single shared mask_embedding is indistinguishable across assays. Combined with CLOZE metadata `[-2,-2,-2,-2]`, the model cannot differentiate which assays are masked.
- **xtransformers (e23d): marginally positive.** −0.2% combined (tied), −17% pred_loss (best), +28% runtype_best. Slight enc_er acceleration. Net small positive.
- **dna_pool_order early (e23g): neutral.** +1.7% combined; +4% enc_er_last; <2% on all other metrics.
- **combo (e23e): REJECTED.** mask_token + post_conv negatives dominate xtransformers gains.

**Bug found: MaskStem CLOZE sentinel leakage (fixed 2026-05-16).** `MaskStem.forward` only checked for `MISSING=-1`, not `CLOZE=-2`. CLOZE positions leaked raw `-2` values through the conv tower (marked as "present" with value=-2). This gave the model an unfair signal advantage — the conv could trivially detect masked assays via the distinctive negative value. Fix: `MaskStem` now accepts `sentinels=(-1, -2)` and masks both. All e23a-p runs used the old (leaky) code; future runs will use the corrected version.

### Batch 2 (2026-05-16)

**Execution:** e23i-p all completed (ExitCode=0:0).

**Ranking (combined_loss_scaled, all E23 + e21o reference):**

| Rank | Run | combined_loss | cov_cond | enc_er_last | runtype_last | gamma_last | Delta |
|------|-----|--------------|----------|-------------|-------------|------------|-------|
| 1 | e23h | 0.7135 | 139 | 20.1 | 0.036 | 5.9 | batch 1 winner |
| 2 | e23i | 0.7269 | 158 | 18.4 | 0.024 | 129.5 | pre_conv + xtfm + raw_meta_tgt |
| 3 | e23p | 0.7303 | 149 | 17.1 | 0.013 | 616.3 | pre_conv + xtfm + legacy_mlp |
| 4 | e23d | 0.7336 | 179 | 17.4 | 0.094 | 0.5 | xtransformers only |
| 5 | e23n | 0.7351 | 161 | 18.3 | 0.026 | 0.5 | separate embed |
| 6 | e23a | 0.7353 | 191 | 19.6 | 0.074 | 2.2 | control |
| 7 | e23j | 0.7454 | 168 | 18.2 | 0.023 | 0.4 | meta_tgt_embed (shared) |
| 8 | e23m | 0.7566 | 140 | 17.4 | 0.112 | 140.0 | per_conv_and_tfm + raw |
| 9 | e21o | 0.7890 | 52.9 | 26.2 | 0.708 | 123.2 | CANDI encoder ref |
| 10 | e23k | 0.9062 | 124 | 19.6 | 0.032 | 0.4 | lambda=0.75 |

**Batch 2 per-ablation verdicts:**
- **raw_meta_tgt conditioning is clearly better than meta_tgt_embed (shared).** e23i (raw, 0.7269) vs e23j (embed, 0.7454) = −2.5% combined. raw_meta_tgt drives adaLN_gamma to 129.5 (vs 0.4 for embedded) — the predictor heavily uses raw conditioning. Adopt raw_meta_tgt as default for fresh encoder.
- **legacy_mlp predictor is competitive on loss but worst on biology.** e23p (0.7303) is only 0.5% behind e23i (0.7269). Drives enormous gamma (616.3) but runtype_peak=0.158 and runtype_last=0.013 — weakest metadata sensitivity of any run. MLP over-relies on conditioning, under-develops encoder representations.
- **Separate MetadataEmbedding (e23n) offers no advantage.** e23n (0.7351) ≈ e23j (0.7454), both worse than raw conditioning. Separate module maps the same `[-2,-2,-2,-2]` CLOZE metadata — no information gain.
- **lambda=0.75 is over-regularized (e23k: REJECTED).** 0.9062 combined (worst by far). pred_loss stalls at 0.043 (vs 0.024 for e23i). Anti-collapse pressure crushes prediction without meaningfully improving geometry (cov_cond=124, still fails gate). Stay at lambda=0.5.
- **per_conv_and_transformer + raw_meta_tgt (e23m) retains metadata best.** runtype_last=0.112 (best in batch 2), runtype_best=0.535, but mediocre combined_loss (0.7566). FiLM capacity preserves biology at the cost of prediction quality.

**Universal collapse persists.** All 14 fresh encoder runs (e23a-p) fail the v2 geometry gate (cov_cond 124–191, threshold=50). The collapse is not addressable by conditioning type, predictor architecture, or moderate lambda changes. It is structural to the fresh encoder.

**Competing explanations (updated):** (1) ~~Post-transformer LayerNorm.~~ RETRACTED 2026-05-15. (2) ~~Higher lambda_sigreg.~~ REJECTED by e23k (lambda=0.75 hurts without fixing geometry). (3) The CANDI encoder's DualAttention may provide structural collapse resistance (still untested). (4) The MaskStem CLOZE leakage may be inadvertently providing useful information to the encoder (needs rerun with fix to assess).

**Decision:** raw_meta_tgt conditioning is the clear winner — adopt as default. Combine winners: pre_conv + xtfm + raw_meta_tgt (e23i config) is the best fresh encoder. The collapse problem requires architectural intervention (DualAttention, or post-encoder normalization). Rerun e23i with fixed MaskStem as next priority. See [`synthesis_e23_encoder_ablation.md`](synthesis_e23_encoder_ablation.md).

### Batch 3 (2026-05-16) — MetadataEmbedding LayerNorm + assay_id preservation ablation

Two code interventions tested in a 2×2×2 matrix (8 runs):
- `meta_embed_layernorm`: remove final LayerNorm from MetadataEmbedding fusion (hypothesis: LN compresses dynamic range, killing predictor AdaLN).
- `preserve_assay_id`: always keep assay_id in metadata for masked/missing assays instead of sentinel-overwriting (hypothesis: gives embedding a discriminative signal per assay).
- `cond_source`: raw_meta_tgt (e23q-t) vs meta_tgt_embed/shared (e23u-x).

**Submitted runs:**
- `e23q` (`40203239`) — raw, no_layernorm=true, preserve_assay_id=false
- `e23r` (`40203240`) — raw, no_layernorm=false, preserve_assay_id=true
- `e23s` (`40203241`) — raw, no_layernorm=true, preserve_assay_id=true (combined)
- `e23t` (`40203242`) — raw, no_layernorm=false, preserve_assay_id=false (control)
- `e23u` (`40205237`) — embed, no_layernorm=true, preserve_assay_id=false
- `e23v` (`40205238`) — embed, no_layernorm=false, preserve_assay_id=true
- `e23w` (`40205239`) — embed, no_layernorm=true, preserve_assay_id=true (combined)
- `e23x` (`40205240`) — embed, no_layernorm=false, preserve_assay_id=false (control)

**Ranking (combined_loss_scaled):**

| Rank | Run | combined_loss | cov_cond | enc_er | gamma | runtype |
|------|-----|--------------|----------|--------|-------|---------|
| 1 | e23r (raw, preserve) | 0.7664 | 142.1 | 19.4 | 114.0 | 0.038 |
| 2 | e23t (raw, control) | 0.7683 | 114.4 | 22.2 | 132.8 | 0.039 |
| 3 | e23w (embed, combined) | 0.7821 | 122.2 | 18.4 | 58.3 | 0.039 |
| 4 | e23v (embed, preserve) | 0.7860 | 149.8 | 18.9 | 0.09 | 0.074 |
| 5 | e23q (raw, no_ln) | 0.8001 | 114.9 | 22.5 | 136.0 | 0.094 |
| 6 | e23x (embed, control) | 0.8042 | 108.2 | 20.8 | 0.19 | 0.049 |
| 7 | e23s (raw, combined) | 0.8124 | 110.5 | 22.5 | 143.7 | 0.088 |
| 8 | e23u (embed, no_ln) | 0.8237 | 117.7 | 18.9 | 54.0 | 0.046 |

**Batch 3 per-ablation verdicts:**

- **LayerNorm IS the primary cause of predictor condition collapse for embedded conditioning.** With LN: gamma ≈ 0.1–0.2 (dead). Without LN: gamma ≈ 54–58. But raw conditioning achieves gamma ≈ 114–144 regardless. LN removal is necessary but not sufficient.
- **LayerNorm is BENEFICIAL for the encoder's own FiLM path.** Raw predictor runs without encoder LN (e23q: 0.8001, e23s: 0.8124) are worse than control with LN (e23t: 0.7683). The encoder benefits from LayerNorm normalization even though it harms predictor conditioning.
- **Contradictory dual role of LayerNorm:** helpful for encoder FiLM, harmful for predictor conditioning when predictor uses the same embedding. Implication: global LN toggle is suboptimal. Use separate embedding modules with different LN if embedded predictor conditioning is needed.
- **preserve_assay_id is a consistent modest improvement via encoder path.** Raw: 0.7664 vs 0.7683 (0.3%). Embedded+noLN: 0.7821 vs 0.8237 (5.3%). Assay_id is the only surviving discriminative signal in masked metadata (depth/readlen/runtype are sentineled).
- **raw_meta_tgt still wins by 2.0%.** Best raw (e23r: 0.7664) > best embed (e23w: 0.7821). Even with both fixes the embedding module's 4:1 compression + learned representations lose ~60–70% of conditioning signal.
- **All 8 runs fail v2 geometry gate** (cov_cond 108–150, threshold 50). Collapse is structural to the fresh encoder, not addressable by metadata/conditioning fixes.

**Decisions from batch 3:**
- Keep `raw_meta_tgt` as default predictor conditioning (naturally avoids LN issue).
- Adopt `preserve_assay_id=true` as default (modest but free improvement).
- Do NOT globally remove encoder LayerNorm — it helps the encoder.
- If embedded predictor conditioning is revisited, use separate MetadataEmbedding without LN for the predictor (keep encoder's embedding with LN).
