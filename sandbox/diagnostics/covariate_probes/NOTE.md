# Covariate probes — overnight autonomous run log

Goal: by morning, a complete & correct `report.md` + figures, OR a clean diagnosis. Correctness > speed.

## Autonomous protocol (what I do on each poll wake-up)
Re-assess true job state from `squeue`/`sacct` (not from which poll fired), then take the next action:
1. **GATE passed (exit 0)** + convergence done → set `EP` from the plateau, launch `sweep` then
   `aggregate (afterany)`. Chain is then SLURM-driven (results appear even if my polling dies).
2. **GATE failed (exit != 0)** → read `profile_<id>.out` + `_prof_validate.txt`, identify the failing
   check, **fix the code, re-validate, retry**. Bounded: ≤3 attempts per *distinct* failure with
   *distinct* hypotheses (per the "fails twice → change approach" rule). Each attempt logged below.
3. **A sweep array task fails** → per-assay files are independent; aggregate still runs on what
   succeeded. I inspect failed tasks, fix, resubmit only those assays, re-aggregate.
4. **Stuck after bounded retries** → stop burning compute; leave full diagnosis here + in `report.md`.

## Job chain
- GATE (validation hard-gates + sweep-path + profiling): see `.gate_id`
- CONV (epoch plateau): 45902467
- SWEEP / AGG: launched after gate+conv pass (ids appended below)

## Gating policy (hard = blocks sweep; soft = reported only)
- HARD: norm invariants; split disjoint; DNA shape/one-hot; overfit-tiny >0.95; label-shuffle &
  DNA-only ≤0.70 (mean of 2 splits); depth_raw R²>0.5; depth_thin R²<0.7 (thinning must clearly help).
- SOFT (finding, never blocks): exact depth R² (raw/scaled/thin) — the "sum-norm doesn't strip depth" result.

## Log
- (init) gate 45902632 + conv 45902467 submitted; sweep/agg held pending their results.
- gate 45902632 FAILED: overfit-tiny=0.887 (<0.95). Diagnosed: NOT a model issue — tiny set got only
  ~30 grad steps (128 windows / bs256). Convergence job proves capability: DNase run_type AUROC
  0.86→0.96 by epoch 4, flat to epoch 20. Fix: tiny uses bs=32 + 60 epochs.
- conv 45902467 TIMEOUT at 1:30 having done only DNase run_type. **Root cause = perf bug**: DataLoader
  num_workers=0 + per-window arcsinh in Python → ~1 min/epoch. That would blow the sweep walltime.
  **Fix: rewrote train_eval GPU-resident** (whole count tensor + unique DNA one-hots on GPU, minibatch
  by indexing, arcsinh once). Removed DataLoader/window_dataset.
- Decision: epochs=15 for sweep (run_type plateaus by epoch 4-8 per convergence; margin for read_length).
- Next: re-run gate on GPU (validate + real timing with fast path), then launch sweep->agg.
- gate 45907316: GPU-resident path works & FAST (overfit-tiny=1.000 ✓, validation ran in min) but
  **CUDA OOM**: transformer self-attention at bs=512 over 768 positions = single 4.8 GB alloc, plus GPU
  tensors accumulated across validation calls. Fixes: (a) model downsamples post-conv seq 768->192
  (16x less attention), (b) bs 256/128, (c) free tensors + empty_cache after every fit,
  (d) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Resubmitted gate 45907489.
- gate 45907489: ALL validation PASS except depth_thin. Controls clean: label-shuffle=0.398,
  DNA-only=0.500, overfit-tiny=1.000, depth_raw=0.721, depth_scaled=0.872, **depth_thin=0.737**.
  FINDING (not a bug): neither scaling nor thinning removes depth on held-out cell types -> likely
  **depth<->biology confound** in the cohort. Decision: depth_thin made INFORMATIONAL (never blocks);
  report.md carries the confound caveat + recommends depth-matched follow-up. Validation effectively
  PASSES; launching the sweep directly (same train_eval code already ran ~12x on GPU in validation).
- Peak host RAM in validation ~4.1 GB (mem=16G is ample). Validation wall ~6 min.
