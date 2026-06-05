# E28 - JEPA Decoder Training (Stage 2)

Status: running (top-3 E27 `jdec_a` baselines submitted for `type2_loci` and `type1_chr19`)  
Parent: E19/E21/E23/E24-E27 (JEPA Stage 1 pretrained encoder+predictor)  
Run naming: jdec_* (jepa decoder runs)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e28)

## Problem Statement

We have established good JEPA-pretrained encoder+predictor checkpoints (E19–E27) that produce biologically structured latent representations. The encoder maps sparse/noisy input to `z_ctx`, and the predictor maps `z_ctx` + target metadata → `z_pred` (the imputed latent). We now need to train decoders that map `z_pred` back to signal space (raw counts, processed signal, peak calls) — completing the full JEPA-based imputation/denoising pipeline.

Unlike end-to-end CANDI where all components receive gradients from reconstruction loss, here the encoder+predictor are pretrained and the decoder training is decoupled. This separation:
1. Eliminates the metadata collapse issue (Q5) by delegating metadata conditioning to the predictor
2. Removes head interference at the encoder level (Q2 finding: pval interferes with count+peak)
3. Allows independent decoder optimization without risking pretrained representation quality

## Idea / Hypothesis

Train three independent decoder towers (count/NB, signal/Gaussian, peak/BCE) on top of frozen encoder+predictor, consuming the predictor's output `z_pred` as input. The decoders are purely latent→signal mappers with no metadata conditioning (FiLM removed). The predictor already handles imputation semantics via metadata-conditioned latent prediction.

**Core hypothesis:** A frozen JEPA encoder+predictor produces `z_pred` of sufficient quality that simple decoder towers can match or exceed end-to-end CANDI reconstruction metrics (E7 best: `imp_peak_auroc=0.765`, `imp_count_pearson=0.339`, `imp_pval_pearson=0.277`).

**Secondary hypothesis:** With frozen encoder+predictor and separate decoder towers per head, there are zero shared trainable parameters between heads → training heads jointly vs separately should produce identical results (serves as a correctness check).

## Architecture

### Pipeline (inference)

```
encoder(x_ctx, x_dna, meta_ctx)     → z_ctx_raw  [B, L2, F2]
projector(z_ctx_raw)                 → proj_ctx   [B, L2, proj_dim]
predictor(proj_ctx, meta_tgt_embed)  → z_pred     [B, L2, proj_dim]
input_proj_count(z_pred)             → [B, L2, decoder_hidden]
count_decoder(...)                   → [B, L, F]
NegativeBinomialLayer(...)           → (p, n)     [B, L, F]
```

Same pattern for pval (GaussianLayer → mu, var) and peak (PeakLayer → sigmoid).

### Decoder tower (per head)

- `Linear(proj_dim → decoder_input_dim)` — learnable input projection (decouples from proj_dim)
- `n_cnn_layers` × transpose-conv upsampling stages (`L2 → L`)
- **Non-grouped** convolutions (cross-assay mixing allowed) as default
- Optional grouped mode for ablation (per-assay independent decoding)
- No FiLM conditioning
- Output head: same production layers (NB/Gaussian/Peak) with `var_min=0.1`

### Freeze modes (configurable)

| Mode | Encoder | Predictor | Decoder | Use case |
|---|---|---|---|---|
| `decoder_only` (default) | frozen | frozen | trained | Clean isolation: is z_pred good enough? |
| `predictor_decoder` | frozen | trained | trained | Can predictor adapt to decoder needs? |
| `encoder_decoder` | trained | frozen | trained | Can encoder adapt without predictor? |
| `all` | trained | trained | trained | Full fine-tuning (risks catastrophic forgetting) |

### Head training modes

| Mode | Description |
|---|---|
| `joint` (default) | All 3 heads train simultaneously (separate towers, no shared params) |
| `count_only` | Only count decoder trained |
| `pval_only` | Only signal decoder trained |
| `peak_only` | Only peak decoder trained |

With frozen encoder+predictor + separate towers: joint == sum of individual head trainings (mathematical equivalence, serves as correctness check).

## Planned Intervention

- Model wrapper: `sandbox/jepa_decoder.py`
- Training script: `sandbox/train_jepa_decoder.py`
- Config: `sandbox/configs/decoder_training.yaml` (layered with `jepa_default.yaml`)
- Smoke tests: `sandbox/test_jepa_decoder.py`
- Submit script: `sandbox/jobs/submit_jdec_a.sh`
- Checkpoint: E27 best (default), configurable to any JEPA checkpoint path
- Optimizer: Adamax, `lr=1e-3`, `clip_norm=2.0`, cosine schedule (matches end-to-end CANDI defaults)
- Masking: Same as JEPA pretraining (`p_full_assay=1.0`, `min_available_frac=0.3`)
- Loss: 6-branch `CANDI_LOSS` (obs/imp split, default). Configurable unified mode for ablation.
- Epochs: 200
- W&B: `candi_sandbox` project, `jdec_` run prefix

Implementation note: decoder training reuses `prepare_masked_batch` for the reconstruction loss masks and exposes a CANDI-compatible model forward signature, allowing `sandbox/train.py::run_eval_pass` to be reused directly for apples-to-apples evaluation. The model reconstructs the JEPA target metadata by appending the control metadata column from `x_meta` to `y_meta`.

## Ablation matrix (planned)

| Run | Freeze | Heads | Grouped | Loss mode | Notes |
|---|---|---|---|---|---|
| jdec_a | decoder_only | joint | non-grouped | obs_imp | Baseline |
| jdec_b | decoder_only | count_only | non-grouped | obs_imp | Correctness check vs jdec_a count metrics |
| jdec_c | decoder_only | pval_only | non-grouped | obs_imp | Correctness check vs jdec_a pval metrics |
| jdec_d | decoder_only | peak_only | non-grouped | obs_imp | Correctness check vs jdec_a peak metrics |
| jdec_e | decoder_only | joint | grouped | obs_imp | Cross-assay mixing ablation |
| jdec_f | decoder_only | joint | non-grouped | unified | Obs/imp unified loss ablation |
| jdec_g | predictor_decoder | joint | non-grouped | obs_imp | Predictor unfreezing |
| jdec_h | all | joint | non-grouped | obs_imp | Full fine-tuning |

## Verifiables

- **Validate if:**
  - jdec_a `imp_peak_auroc ≥ 0.70` (within ~10% of E7 best 0.765)
  - jdec_a `imp_count_pearson ≥ 0.30` (within ~10% of E7 best 0.339)
  - jdec_a `imp_pval_pearson ≥ 0.25` (within ~10% of E7 best 0.277)
  - jdec_b/c/d per-head metrics match jdec_a within noise (correctness check)
  - No pval variance collapse (F7 mitigated by var_min=0.1)
  - Loss converges monotonically (no late divergence — encoder is frozen)

- **Disvalidate if:**
  - All imp_* metrics are substantially below end-to-end CANDI (>30% regression) → z_pred quality is insufficient
  - jdec_b/c/d metrics differ substantially from jdec_a → implementation bug (shared params leak)
  - Loss diverges despite frozen encoder → decoder architecture problem

- **Specific checks:**
  - `eval_losses/{count,pval,peak}_{obs,imp}_loss` trajectories
  - `imp_count_pearson_gw`, `imp_pval_pearson_gw`, `imp_peak_auroc_gw`
  - `depth_count_ratio` (should be > 1.0 now that metadata collapse is addressed via predictor)
  - Gradient norms (should be well-behaved with frozen upstream)
  - Per-head loss curves (verify no interference)

- **Required artifacts:** `resolved_config.yaml`, `metrics.jsonl`, SLURM logs, W&B run.

## Risks / Watch-outs

- `z_pred` may not encode sufficient spatial detail for count/pval reconstruction at bin-level resolution (proj_dim=72 bottleneck for L2=96 tokens → 72-dim vectors representing 768 output positions per assay)
- Non-grouped deconv allows cross-assay information flow which was never present in production CANDI decoders — could help or introduce new artifacts
- Masking randomness means different batches see different mask patterns → `z_pred` quality varies per-batch, adding noise to decoder gradients
- The predictor was trained with MSE in projection space (not reconstruction NLL) — its `z_pred` optimizes for representation-matching, not signal-reconstruction. Mismatch between pretraining objective and decoder objective is the fundamental risk.
- Single checkpoint dependency: if E27 best is suboptimal, all decoder results are capped. Multiple checkpoints should be tested.

## Run Links

- Cancelled no-probe jobs: `40933759`, `40933761`, `40933762`, `40934480`, `40934481`, `40934482` (`eval.meta_sensitivity_probe_every_n_steps=0`).
- `jdec_e27_lam005_40908829_sens_a` — SLURM `40935878`, checkpoint `sandbox/runs/e27_lam005_40908829/jepa_checkpoint_best.pt`, override `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam005_40908829_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935878.{out,err}`.
- `jdec_e27_lam01_40908830_sens_a` — SLURM `40935879`, checkpoint `sandbox/runs/e27_lam01_40908830/jepa_checkpoint_best.pt`, override `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam01_40908830_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935879.{out,err}`.
- `jdec_e27_lam02_40908831_sens_a` — SLURM `40935880`, checkpoint `sandbox/runs/e27_lam02_40908831/jepa_checkpoint_best.pt`, override `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam02_40908831_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935880.{out,err}`.
- `jdec_e27_lam005_40908829_type1_sens_a` — SLURM `40935881`, checkpoint `sandbox/runs/e27_lam005_40908829/jepa_checkpoint_best.pt`, overrides `data.regime=type1_chr19`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam005_40908829_type1_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935881.{out,err}`.
- `jdec_e27_lam01_40908830_type1_sens_a` — SLURM `40935882`, checkpoint `sandbox/runs/e27_lam01_40908830/jepa_checkpoint_best.pt`, overrides `data.regime=type1_chr19`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam01_40908830_type1_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935882.{out,err}`.
- `jdec_e27_lam02_40908831_type1_sens_a` — SLURM `40935883`, checkpoint `sandbox/runs/e27_lam02_40908831/jepa_checkpoint_best.pt`, overrides `data.regime=type1_chr19`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam02_40908831_type1_sens_a`, logs `sandbox/slurm_logs/jdec_a_40935883.{out,err}`.
- `jdec_e27_lam005_40908829_b2_count` — SLURM `40975758`, checkpoint `sandbox/runs/e27_lam005_40908829/jepa_checkpoint_best.pt`, overrides `decoder.freeze_mode=predictor_decoder`, `decoder.heads=count_only`, `data.regime=type2_loci`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam005_40908829_b2_count`, logs `sandbox/slurm_logs/jdec_a_40975758.{out,err}`.
- `jdec_e27_lam005_40908829_b2_pval` — SLURM `40975759`, checkpoint `sandbox/runs/e27_lam005_40908829/jepa_checkpoint_best.pt`, overrides `decoder.freeze_mode=predictor_decoder`, `decoder.heads=pval_only`, `data.regime=type2_loci`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam005_40908829_b2_pval`, logs `sandbox/slurm_logs/jdec_a_40975759.{out,err}`.
- `jdec_e27_lam005_40908829_b2_peak` — SLURM `40975760`, checkpoint `sandbox/runs/e27_lam005_40908829/jepa_checkpoint_best.pt`, overrides `decoder.freeze_mode=predictor_decoder`, `decoder.heads=peak_only`, `data.regime=type2_loci`, `eval.meta_sensitivity_probe_every_n_steps=200`, run dir `sandbox/runs/jdec_e27_lam005_40908829_b2_peak`, logs `sandbox/slurm_logs/jdec_a_40975760.{out,err}`.

## Implementation / Validation Log

- 2026-05-21 — Implemented `sandbox/jepa_decoder.py` with `JEPADecoderTower` and `JEPADecoderModel`. Supports `decoder_only`, `predictor_decoder`, `encoder_decoder`, and `all` freeze modes; `joint` / single-head training modes; grouped vs non-grouped deconv; `obs_imp` vs `unified` training loss.
- 2026-05-21 — Implemented `sandbox/train_jepa_decoder.py`. It loads `jepa_default.yaml` then `decoder_training.yaml`, accepts arbitrary checkpoint path via `--checkpoint` or `decoder.checkpoint_path`, trains only parameters enabled by the freeze mode, writes `metrics.jsonl`, saves `jepa_decoder_checkpoint_last.pt`, and reuses `run_eval_pass` for reconstruction metrics.
- 2026-05-21 — Implemented `sandbox/test_jepa_decoder.py`. GPU smoke tests on `fc11020` passed: unit shapes, grouped/non-grouped towers, dummy checkpoint loading, forward output validity, CANDI loss/backward, frozen encoder/predictor gradient isolation, all four freeze modes, two-step training on real H5, and eval key-format compatibility.
- 2026-05-21 — `py_compile` passed for all new/modified Python files. `train_jepa_decoder --dry-run` passed strict config validation. A one-batch main-entry smoke run passed with a matching dummy checkpoint, producing `metrics.jsonl` and `jepa_decoder_checkpoint_last.pt`.
- 2026-05-21 — Initial submissions `40932978`–`40932980` failed before Python because the batch shell could not find `conda`. Updated `sandbox/jobs/submit_jdec_a.sh` to match prior successful sandbox jobs: canonical repo path, `sandbox/slurm_logs`, `gpubase_bygpu_b2`, direct `candi_venv` activation, tolerant `module load samtools`, and host/GPU diagnostics. `bash -n` and `sbatch --test-only` passed before resubmission.
- 2026-05-21 — Submitted top-3 E27 `jdec_a` baselines: `40933759` (`e27_lam005_40908829`), `40933761` (`e27_lam01_40908830`), and `40933762` (`e27_lam02_40908831`).
- 2026-05-21 — Submitted matching `type1_chr19` baselines with `data.regime=type1_chr19`: `40934480` (`e27_lam005_40908829`), `40934481` (`e27_lam01_40908830`), and `40934482` (`e27_lam02_40908831`).
- 2026-05-21 — Re-enabled decoder metadata sensitivity logging for future runs: per-step null-target-metadata head MSEs plus epoch-level depth/read-length/run-type prompt perturbation probes. GPU smoke validation on `fc11020` passed with a real E27 checkpoint and batch.
- 2026-05-21 — Cancelled the six no-probe `jdec_a` jobs and resubmitted the same top-3 E27 × (`type2_loci`, `type1_chr19`) matrix with explicit `eval.meta_sensitivity_probe_every_n_steps=200`: `40935878`–`40935883`.
- 2026-05-22 — Submitted E28 batch 2 on the provisional reference checkpoint `e27_lam005_40908829`: predictor+decoder fine-tuning with one active head per job (`count_only`, `pval_only`, `peak_only`) on `type2_loci`. Updated `sandbox/jobs/submit_jdec_a.sh` walltime from 6h to 4h and added `sandbox/jobs/submit_jdec_batch2_predictor_heads.sh`.

## Findings

- Observed: Preliminary cross-run synthesis versus B8 shows the reconstruction-only baseline remains stronger: B8 `quality_score=5.6812` vs best E28 `jdec` `6.2880` (`jdec_e27_lam005_40908829_sens_a`). See [`synthesis_e28_jdec_vs_b8.md`](synthesis_e28_jdec_vs_b8.md).
- Interpretation: Frozen JEPA `z_pred` is decodable, but the current Stage 2 setup has not matched end-to-end reconstruction quality. Metadata depth sensitivity remains collapsed (`depth_count_ratio≈1.0`), while read-length perturbation sensitivity is larger in some `jdec` runs.
- Competing explanations: `z_pred` may lack bin-level reconstruction detail; decoder-only training may under-adapt the predictor; or this is not a clean comparison because B8 differs in masking mix, epochs, batch size, and architecture.
- Decision: Use `e27_lam005_40908829` as the provisional reference checkpoint. E28 batch 2 tests whether predictor adaptation helps when each head fine-tunes the predictor in isolation, avoiding shared-head predictor interference.
