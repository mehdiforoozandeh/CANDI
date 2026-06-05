# E31 — depth_center sweep on CANDI v2 depth-offset head

Status: running  
Parent: E30 ([idea_e30_v2_depth_offset_head.md](idea_e30_v2_depth_offset_head.md)); closes E29 centering question ([idea_e29_libsize_offset_nb.md](idea_e29_libsize_offset_nb.md))  
Run name prefix: `e31_c<C>` for `C ∈ {0, 22, 23, 25, 27, 28, 30}`  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e31)

## Problem Statement

E30 validated the depth-offset count head at fixed `depth_center=24`. Autoresearch on a pinned chr19 batch favored `c=27`; sandbox EIC depths cluster ~22–28 ([`sandbox/data/sandbox_log2_depths.csv`](../data/sandbox_log2_depths.csv)). Before marking E29 done, we need evidence on which `c` maximizes count quality and depth sensitivity on real v2 training — not a hand-picked constant.

## Idea / Hypothesis

Hold the E30-winning stack fixed (`count_only`, `depth_offset`, type1 chr19, 200 epochs, assay-only masking, adamax/cosine) and sweep only `decoder.depth_center`. A data-aligned center (near batch median log₂ depth, ~24–26) should match or beat off-median values on `depth_count_ratio`, `count_imp_loss`, and `imp_count_pearson_gw`. Extreme `c=0` should reproduce the uncentered failure mode (η scale mismatch); `c` far from typical depth should degrade count metrics.

## Planned Intervention

- **Entrypoint:** `python -m sandbox.train_candi_v2`
- **Config stack:** `candi_v2_default.yaml` → `e30_v2_common.yaml` → `e30_v2_nboffset.yaml` → `e31_depth_center_sweep.yaml` → `--set decoder.depth_center=<c>`
- **Sweep:** `c ∈ {0, 22, 23, 25, 27, 28, 30}` (24 held out as E30 reference)
- **Submit:** `sandbox/slurm_tmp/e31_depth_center_sweep_submit.sh` (7 jobs)
- **Validate:** `sandbox/slurm_tmp/e31_validate.sh`

## Verifiables

- Validate if: at least one sweep `c` matches or beats E30 B (`e30_v2_nboffset`) on count imputation + stable `depth_count_ratio ≈ 4`; best `c` is interpretable vs sandbox depth distribution.
- Disvalidate if: all non-24 centers regress vs E30 B; or `depth_count_ratio` collapses for all runs (offset broken).
- Required artifacts per run: `resolved_config.yaml` (only `depth_center` differs), `metrics.jsonl`, SLURM logs, W&B `e31_c*`.
- Rank with log-observability after all jobs complete; write `synthesis_e31_depth_center_sweep.md` before updating E29 status.

## Risks / Watch-outs

- `c=0` may be numerically stiff (η must absorb full log₂ depth scale) — expect worse early loss, not necessarily OOM.
- Single-knob sweep assumes log₂ parameterization change (post-E30) does not interact with `c`; all runs use same code revision.
- 24 omitted from sweep — compare E30 B directly as external reference, not in-rank.

## Run Links

- Run directory: `sandbox/runs/e31_c<C>_<jobid>/` (in flight)
- SLURM jobs (2026-06-01): c0=42520313, c22=42520314, c23=42520315, c25=42520316, c27=42520317, c28=42520318, c30=42520319
- W&B run: `candi_sandbox` / `e31_c<C>`

## Findings

- Observed: TBD
- Interpretation: TBD
- Competing explanations: TBD
- Decision: TBD
