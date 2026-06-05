# E29 — NB count head with library-size offset

Status: partially validated (via E30 on CANDI v2; E31 depth_center sweep staged; production B8 still open)  
Parent: B8 (E7 single-shot FiLM + `gaussian_var_min=0.1`)  
Run name: validated indirectly via E30 (`e30_v2_nboffset`)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e29)

Theory and GLM precedent: [libsize_offset_NB.md](libsize_offset_NB.md)

## Problem Statement

`NegativeBinomialLayer` predicts raw $\mu$ via `softplus(linear_mean(x))` while sequencing depth enters only through FiLM/metadata embeddings. Universal `depth_count_ratio ≈ 1.0` (target ≈ 4.0) shows the count head does not use depth the way a count GLM should: depth is a known technical scale factor, not a free learned shortcut.

Passing `log2(seq_depth)` as a network input and hoping $h_\theta$ learns $\mu \approx s \cdot e^{\eta}$ forces the model to rediscover a constraint DESeq2, edgeR, and scVI encode by construction.

## Idea / Hypothesis

Treat per-sample library size $s_s$ (from `depth_log2` / size-factor normalization) as a **fixed offset** in the NB mean:

$$\log \mu_{sb} = \log s_s + \eta_{sb}, \qquad \mu_{sb} = s_s \cdot e^{\eta_{sb}}$$

The network predicts log enrichment $\eta_{sb}$ and dispersion $N_b$; the loss constructs $\mu$ inside `negative_binomial_loss`. This should (a) raise `training_metadata_probes/depth_count_ratio` toward the probe target, (b) improve cross-depth generalization of `imp_count_pearson` / `count_imp_loss`, and (c) reduce depth–count shortcutting on background bins.

## Planned Intervention

- **Code:** `model.py::NegativeBinomialLayer` — add `predict_log_rate` mode (linear head → $\eta$, no softplus on rate); `candi_loss.py` / sandbox loss — multiply by $s_s = 2^{\text{depth\_log2}}$ (or median-normalized size factor) before NLL; keep $(p, n)$ interface or switch loss to $(\mu, n)$ directly.
- **Config:** B8 stack (`single_shot_decoder_film`, `gaussian_var_min=0.1`, type1 chr19, multi-head); single-axis change on count parameterization only.
- **Baseline:** B8 or latest completed B8-equivalent run for apples-to-apples `depth_count_ratio` and count metrics.
- Submit/config path: TBD  
- Run name: TBD  

## Verifiables

- Validate if: `training_metadata_probes/depth_count_ratio` moves materially toward probe expectation (≈ 4.0 under default depth_lo/hi); `imp_count_pearson` and/or `count_imp_loss` improve vs B8 at matched epoch; masked and unmasked count NLL stable (no blow-up).
- Disvalidate if: `depth_count_ratio` unchanged; count metrics regress vs B8; training diverges or `count_imp_loss` worse while obs branch only improves (offset mis-specified).
- Required artifacts: `resolved_config.yaml`, `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- Size-factor definition matters (total reads vs median-of-ratios vs per-locus); mismatch with training covariates can hurt more than raw-$\mu$ baseline.
- Masked assays use cloze metadata — ensure $s_s$ for imputed targets uses **output** depth (`C_out`), not masked input depth.
- Grouped deconv / per-assay heads: offset must apply per bin consistently across assays.
- Confound with Q2 (pval interference): run full multi-head first; optional E2-style count-only follow-up if signal is buried.

## Run Links

- Run directory: see E30 B run `sandbox/runs/e30_v2_nboffset_42441523`
- Resolved config: `sandbox/runs/e30_v2_nboffset_42441523/resolved_config.yaml`
- Metrics: `sandbox/runs/e30_v2_nboffset_42441523/metrics.jsonl`
- SLURM logs: job 42441523 (paired with baseline 42441522)
- HPO graph node: TBD
- W&B run: `candi_sandbox` / `e30_v2_nboffset`

## Findings

- Observed: E30 B run (pow2-centered offset on CANDI v2) at ep199: `depth_count_ratio`=4.03, `imp_count_pearson_gw`=0.371, `count_imp_loss`=1.60 vs plain baseline 1.10 / 0.289 / 2.07.
- Interpretation: Library-size offset in the NB mean restores depth sensitivity and improves count reconstruction on real sandbox data — hypothesis validated on v2 backbone.
- Competing explanations: v2 uses fresh encoder + count-only + assay-only masking; production multi-head B8 may behave differently.
- Decision: Mechanism accepted for v2. Remaining work: land offset in production `model.py` and run B8-equivalent multi-head A/B.
