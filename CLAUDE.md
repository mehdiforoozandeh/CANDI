# CANDI / EpiDenoise — project guide

## What this repo is
CANDI (Confidence-Aware Neural Denoising Imputer) is a self-supervised deep-learning model for **epigenome imputation and denoising** from raw sequencing read counts + experimental covariates. Unlike prior methods (ChromImpute, Avocado, eDICE) it consumes **raw counts**, not normalized signal, and outputs **probability distributions** — so it handles batch effects directly, gives calibrated uncertainty, and does **zero-shot** imputation/denoising on new cell types.

Outputs per assay/position: **Negative Binomial** `(n̂,p̂)` for raw counts, **Gaussian** `(μ̂,σ̂²)` for arcsinh log-pval signal, **Bernoulli** (sigmoid) for peaks. ~42M params: parallel Conv1D encoder towers for reads `M` and DNA sequence `S`, **MetadataCrossAttention (FiLM)** on 4 covariates after each conv layer, transformer encoder (`n_sab=4`, `n_head=9`, **RoPE** via x-transformers), three deconv decoders (`D_count`, `D_signal`, `D_peak`).

Runs on Compute Canada (RHEL 9, SLURM), conda env `candi` + venv `candi_venv`, `module load samtools`.

## Hard constraints
- **SLURM GPU:** every sandbox job (baseline *and* experiment) MUST use `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`. Never `--gres=gpu:h100:1` or any other spec on any `#SBATCH --gres` line or `sbatch --gres` flag, unless I explicitly request otherwise.
- **`obs` vs `imp`** in losses/metrics = **unmasked vs masked** positions, NOT biological observed/imputed.
- **Control channel** = index `A` in `M`; always available, **never masked** — handle masking accordingly.
- **Covariates** are 4-dim (`log2(seq_depth)`, `assay_id`, `read_length`, `run_type`) and flow through FiLM at **every** conv layer, not just the input.
- **Archived code:** skip by default. Do not read or modify `legacy/__archive__.py` unless I explicitly ask.

## Data
- Per 30 kb locus @ 25 bp → 1,200 bins. `M`: counts for `A=35` ENCODE assays + 1 ChIP control, `arcsinh`-transformed. `S`: one-hot DNA over 30,000 bp.
- **EIC** = ENCODE Imputation Challenge (35 assays × 50 biosamples) for benchmarking. **Extended ENCODE (MERGED)** = main train set (3,064 biosamples → 361 merged cell types). ENCODE blacklist excluded.

## Evaluation
- Peaks: precision / recall / AUROC vs. ENCODE narrowPeak.
- **Uncertainty**: calibration curves (empirical coverage vs. nominal CI) and **C-index** for distributional ranking.
- **Biological validation**: predict RNA-seq log TPM from TSS / gene-body / TES features computed on (1) observed, (2) denoised, (3) denoised+imputed (35 assays), (4) latent `Z`. Latent `Z` is the strongest predictor and the most robust to input sparsity. CANDI never sees RNA-seq during training.

## Sandbox harness
`sandbox/` is the **fast single-GPU prototyping harness** (diagnostics, not production). Data is an HDF5 slice of the EIC panel (8 assays) at `sandbox/data/sandbox.h5` (bake once via `python -m sandbox.prepare_h5 bake`). Train: `python -m sandbox.train --config sandbox/configs/default.yaml --config sandbox/configs/{type1_chr19,type2_loci}.yaml --h5 …`. Configs deep-merge (defaults → YAML → `--config` → CLI shortcuts → dotted overrides) with **strict rejection of unknown keys**.

## Skill routing (Claude Code)
Installed skills auto-route by their `description`; the triggers below reinforce that. When a trigger matches, **read the full `SKILL.md` as the first action** (the body is the playbook, not just the description). Manual override wins if I name a skill. Don't auto-route on typos/one-line tweaks/readonly questions.

**Project skills** (`.claude/skills/`):
- Genomic intervals / BED / BAM / coverage / ML tokenization → `gtars`, `geniml`, `polars-bio`, `pysam`; track QC & bigWig/ChIP/ATAC viz → `deeptools`
- Inspect/compare/rank/diagnose sandbox runs (`metrics.jsonl`, `resolved_config.yaml`, SLURM/W&B, grad norms) → `log-observability`
- JEPA runs (`lejepa/*`, `cos_sim_ctx_tgt`, `encoder_eff_rank`, SIGReg collapse, encoder geometry) → `jepa-observability`
- Sandbox ideas/experiments (`EXPERIMENTS.md`, `META.md`, `idea_*.md`) → `sandbox-idea-hub`
- Cross-run rollups / sweep summaries / synthesis docs → `sandbox-synthesis`
- Karpathy-style autoresearch harness design (`sandbox/autoresearch/*`) → `candi-autoresearch`

**User skills** (`~/.claude/skills/`): SLURM jobs / interactive sessions / sbatch / `--gres` → `slurm-hpc`; training/eval → `pytorch-lightning`, `transformers`, `optimize-for-gpu`; manuscripts → `scientific-writing`, `paper-lookup`, `literature-review`, `scientific-visualization`, `citation-management`; stats → `statistical-analysis`, `statsmodels`, `pymc`; big arrays/dataframes → `zarr-python`, `polars`, `dask`; ERA-style whole-program search (Flat UCB / FUTS, evolutionary program design, `generate_fn`/`execute_fn`) → `era`.

## Memory
Durable, non-obvious project facts live in auto-memory (`MEMORY.md` index + files). Check it for context; record new learnings there, not in this file.

