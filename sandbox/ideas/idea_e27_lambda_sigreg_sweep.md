# E27 - Lambda SIGReg Sweep (E24+E26 Defaults)

Status: running  
Parent: E24/E26 (promoted defaults: mask_token + no fusion LN)  
Run names: `e27_lam005_40908829`, `e27_lam01_40908830`, `e27_lam02_40908831`, `e27_lam05_40908832`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md)

## Problem Statement

E24 (mask_token) improved `combined_loss_scaled` by 2.6% and `runtype_sens` by 37% over control, but degraded geometry: `cov_condition_number` rose from 49.0 to 62.3. The current `lambda_sigreg=0.5` was tuned under the old defaults (mask_stem + fusion LN). With two promoted architectural changes (mask_token + no fusion LN), the balance between prediction pressure and SIGReg regularization may have shifted.

## Idea / Hypothesis

Sweep `lambda_sigreg` across {0.05, 0.1, 0.2, 0.5} to find the optimal regularization strength under E24+E26 defaults. Prior evidence:
- lambda=0.1 was insufficient under old defaults (FJ1: trivial position encoding, cos_sim→0.97).
- lambda=0.5 was the sweet spot under old defaults (near-Gaussian sigreg baseline).
- lambda=1.0 was over-regularized (E23 batch 2: e23k rejected).
- lambda=0.05 is the LeJEPA paper default — included for completeness.

Hypothesis: lambda=0.2 or lambda=0.5 will remain optimal. If mask_token increases effective prediction pressure, a lower lambda may recover geometry (lower cov_cond) without sacrificing loss. If no lambda recovers geometry, the cov_cond increase is inherent to the mask_token design.

## Planned Intervention

- 7 runs, pure `jepa_default.yaml` with only `jepa.lambda_sigreg` and `fresh.lambda_sigreg` overridden.
- All runs use E24+E26 promoted defaults: `missing_data_mode=mask_token`, `fusion_norm=none`.
- `lambda_sigreg=0.5` serves as the control (current default).

## Verifiables

- Primary: `combined_loss_scaled`, `cov_condition_number`, `encoder_eff_rank`.
- Secondary: `meta_sens_runtype` (must not regress vs E24 control).
- If lambda=0.05 or 0.1 shows trivial representations (cos_sim→1, enc_er collapse), confirm FJ1 holds under new defaults.
- If lambda=0.2 matches 0.5 on loss with better geometry, promote as new default.

## Risks / Watch-outs

- Very low lambda (0.05) may reproduce FJ1 trivial position encoding — this is an expected negative result confirming prior findings.
- lambda=2.0 previously halted enc_er collapse but prevented pred convergence (E19); expect similar under new defaults.

## Run Links

| Run | lambda | SLURM Job | Run Directory |
|---|---|---|---|
| e27_lam005 | 0.05 | 40908829 | `sandbox/runs/e27_lam005_40908829` |
| e27_lam01 | 0.1 | 40908830 | `sandbox/runs/e27_lam01_40908830` |
| e27_lam02 | 0.2 | 40908831 | `sandbox/runs/e27_lam02_40908831` |
| e27_lam05 | 0.5 | 40908832 | `sandbox/runs/e27_lam05_40908832` |
| e27_lam075 | 0.75 | 40913895 | `sandbox/runs/e27_lam075_40913895` |
| e27_lam10 | 1.0 | 40913896 | `sandbox/runs/e27_lam10_40913896` |
| e27_lam20 | 2.0 | 40913897 | `sandbox/runs/e27_lam20_40913897` |

- HPO graph node: `hpo.experiment_label=e27_lambda_sweep`
- Submit scripts: `sandbox/slurm/submit_e27_lam005.sbatch`, `submit_e27_lam01.sbatch`, `submit_e27_lam02.sbatch`, `submit_e27_lam05.sbatch`
- Superseded attempts: SLURMs 40906539–40906542 (pycache), 40908012–40908015 (wrong config class)

## Findings

Do not fill this from memory. Use concrete artifact evidence and cite metric keys/values.

- Observed: TBD
- Interpretation: TBD
- Competing explanations: TBD
- Decision: TBD
