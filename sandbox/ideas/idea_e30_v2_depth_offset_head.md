# E30 — NB depth-offset count head on CANDI v2: matched A/B vs v2 baseline

Status: done  
Parent: CANDI-v2 ([candiv2.md](candiv2.md)); tests E29 ([idea_e29_libsize_offset_nb.md](idea_e29_libsize_offset_nb.md))  
Run name: `e30_v2_baseline` (A), `e30_v2_nboffset` (B)  
Checklist entry: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e30)

Locked design (2026-05-31): dedicated entrypoint (new, not autoresearch); `heads=count_only`; `regime=type1_chr19`; `depth_center=24` (matches the dcr probe's 22/24 window); long budget (200 epochs); **assay-only masking** (`p_full_assay=1.0`, `p_full_loci=0`, `p_chunks=0`) — now the CANDI v2 default for all runs.

Motivating signal (diagnostic only, not evidence to be reused): [autoresearch_may28_count_head.md](autoresearch_may28_count_head.md).

## Problem Statement

E29 (NB library-size offset) is still open: no **real sandbox run** has tested whether replacing the direct-$\mu$ count head with a depth-offset parameterization restores depth sensitivity (`depth_count_ratio ≈ 1.0` is universal across every B/E run). The only positive signal so far comes from the throwaway autoresearch harness (`sandbox/diagnostics/autoresearch/`), which (a) runs a single pinned batch overfit composite and (b) implements the head as a runtime monkeypatch. That harness is a scratch diagnostic — its **code must not be imported or used** by real sandbox experiments, and its single-batch composite is not a sandbox result.

We need to bring the *idea* (not the code) into `sandbox/candi_v2/` as a clean, config-selectable count head, and run a properly controlled A/B on real sandbox data to see if the depth-offset formulation actually works in the v2 backbone.

## Idea / Hypothesis

Implement the centered depth-offset NB head natively in the v2 decoder:

$$\mu_{sbp} = 2^{(d_s - c)} \cdot \exp(\eta_{sbp}), \qquad p = \frac{n}{n+\mu}$$

where $d_s = \log_2(\text{seq\_depth})$ is read from **target** metadata `y_meta[:, 0, :]`, $\eta$ and $n$ come from the shared decoder trunk, and $c$ = `depth_center`. Run two CANDI v2 jobs on real sandbox data:

- **Run A — `v2_baseline`**: pure `candi_v2_default.yaml`, plain `NegativeBinomialLayer` count head.
- **Run B — `v2_nboffset`**: identical config in every respect **except** `decoder.count_head=depth_offset`.

Hypothesis: the single-knob switch to the depth-offset head moves `depth_count_ratio` materially above ~1.0 (toward the probe target ≈ 4.0) and does not regress count imputation (`imp_count_pearson` / `count_imp_loss`) versus the baseline — reproducing, on a real v2 run, the depth-sensitivity recovery seen in the autoresearch diagnostic.

The A/B is the test that actually closes (or rejects) E29.

## Planned Intervention

- Submit/config path: TBD — two configs differing only by `decoder.count_head`.
- Run name: `v2_baseline` (A) and `v2_nboffset` (B).
- Parent run or idea: CANDI-v2 reference.
- Config/code/data deltas: native depth-offset head + config knob; identical data/masking/optimizer/schedule/eval for both runs.

### Hard constraint

Do **not** import from or call into `sandbox/diagnostics/autoresearch/`. That harness is scratch-only. Reuse of the *parameterization* is fine; reuse of its code is not. The existing `sandbox/diagnostics/depth_offset_nb.py` is an earlier non-autoresearch prototype and may be used as a reference, but the landed implementation should live in `sandbox/candi_v2/` as a first-class config option, not a decoder subclass/patch.

### Implementation (landed 2026-05-31)

1. **`sandbox/candi_v2/config.py::DecoderConfig`** — added count-head knobs (strict-key infra requires declaration):
   - `count_head: Literal["plain", "depth_offset"] = "plain"`
   - `depth_center: float = 24.0`
   - `mu_eps: float = 1e-6`
   - pow2 + softplus only; `depth_scale_mode`/`n_mode` from the autoresearch search deliberately out of scope.

2. **`sandbox/candi_v2/decoder.py`** — added a native `DepthOffsetNegativeBinomialLayer` (math reimplemented, no autoresearch/diagnostics import):
   - `forward(x, depth_log2)`: `eta = linear_eta(x)`; `d = depth_log2.unsqueeze(1) - depth_center`; `mu = (2**d) * exp(eta)`, clamp `mu ≥ mu_eps`; `n = softplus(linear_n(x)) + eps`; `p = n/(n+mu)` clamped.
   - `V2Decoder.__init__`: builds the offset layer when `cfg.count_head=="depth_offset"`; sets `self._count_depth_offset`.
   - `V2Decoder.forward`: computes `count_depth = y_meta[:, 0, :]` (row 0 = `log2(seq_depth)`; `y_meta` is `[B,4,signal_dim]`, control excluded) and branches the count-head call in **both** `shared` and `separate` paths. One decoder class serves both runs.

3. **`sandbox/candi_v2/loss.py`** — unchanged. `(p, n)` flow through `CANDI_LOSS` NB NLL; offset absorbed into `(p, n)`.

4. **`sandbox/train_candi_v2.py`** — new real entrypoint (none existed; `sandbox/train.py` only builds the old `sandbox.model` shell). Loads `candi_v2_default.yaml` → `--config` overlays → `--set` overrides; builds `CANDIv2` + `build_v2_loss` + adamax/cosine; trains on `SandboxH5Dataset` (type1_chr19) with the production masker; per-epoch eval emits `training_metadata_probes/depth_count_ratio` (canonical `sandbox.eval.prompt_sensitivity_depth_count_ratio`) + count obs/imp NLL + `imp_count_pearson`; writes `resolved_config.yaml` + `metrics.jsonl`. No autoresearch imports.

5. **Configs:** `sandbox/configs/e30_v2_common.yaml` (shared: count_only, loss weights, adamax/cosine, masking, epochs) + one-knob variants `e30_v2_baseline.yaml` (`count_head: plain`) and `e30_v2_nboffset.yaml` (`count_head: depth_offset`).

6. **Submit:** `sandbox/slurm_tmp/e30_ab_submit.sh` — two jobs, mandated gres `gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`, 200 epochs.

### Latent bug fixed during implementation

`candi_v2_default.yaml` carried a stale `training.masking.preserve_assay_id: true` key. The current `sandbox.config_types.MaskingConfig` has no such field (it exists only in the JEPA config schema), so strict `config_from_dict` rejected every v2 config that loaded the default — and `SandboxH5Dataset`/`make_masker` don't accept `preserve_assay_id` either (the diagnostics scripts only work via an autoresearch monkeypatch that pops it). Removed the stale key from `candi_v2_default.yaml`; the v2 trainer does not reference it. This unblocked v2 config loading generally.

## Verifiables

- Validate if: Run B `depth_count_ratio` is materially > Run A's (which should sit ≈ 1.0) and trends toward ≈ 4.0, with Run B `imp_count_pearson` / `count_imp_loss` no worse than Run A at matched epochs; no NB blow-up (masked + unmasked count NLL stable).
- Disvalidate if: Run B `depth_count_ratio` stays ≈ 1.0 (offset not actually wired to target depth), or count metrics regress vs baseline, or `mu`/`p` clamp saturation (degenerate offset).
- Required artifacts: `resolved_config.yaml` for both runs (diff confirming the single-knob delta), `metrics.jsonl`, SLURM logs.

## Risks / Watch-outs

- **Single-knob discipline.** The A/B is only interpretable if the configs differ solely by `decoder.count_head`. Diff the two `resolved_config.yaml` before trusting the comparison.
- **Depth row indexing.** Must read **target** depth `y_meta[:, 0, :]`, not input `x_meta` (masked assays carry cloze input metadata). Wrong row silently breaks the offset.
- **`depth_center=27` is borrowed from a different harness.** It was tuned on the autoresearch chr19 batch; on real v2 data the right center may differ (batch-median is the principled choice). For this first A/B, keep a fixed center for both interpretability and parity, and treat center-tuning as a follow-up only if the offset works.
- **No autoresearch imports** (restated): the run is invalid as a sandbox result if it touches `sandbox/diagnostics/autoresearch/`.
- **Generalization is not the question here.** This A/B asks "does the offset work at all on a real v2 run." Multi-batch/long-training/calibration follow-ups are deferred until B beats A.

## Run Links

- Run directory: `sandbox/runs/e30_v2_baseline_42441522`, `sandbox/runs/e30_v2_nboffset_42441523`
- Resolved config: `sandbox/runs/e30_v2_baseline_42441522/resolved_config.yaml`, `sandbox/runs/e30_v2_nboffset_42441523/resolved_config.yaml` (single-knob diff: `decoder.count_head`)
- Metrics: `sandbox/runs/e30_v2_*/metrics.jsonl` (200 epochs, 40 eval points)
- SLURM logs: jobs 42441522 (baseline), 42441523 (nboffset)
- HPO graph node: TBD
- W&B project: `candi_sandbox` — run names `e30_v2_baseline`, `e30_v2_nboffset`. Logged families: `eval_metrics/*`, `eval_losses/*`, `training_metadata_probes/*`. **No `eval_metrics_median/*` in W&B** (see Findings).

## Pre-submission validation (2026-05-31, interactive H100 1g.10gb)

Implementation validated before submitting the long runs:

- **Single-knob, artifact-level:** `diff` of the two resolved `resolved_config.yaml` shows exactly one differing line (`count_head: plain` vs `depth_offset`); `depth_center=24.0` and `mu_eps` identical in both.
- **Head wiring:** Run A builds `NegativeBinomialLayer`, Run B builds `DepthOffsetNegativeBinomialLayer`; both ~300,696 params (only +144 for the offset head, ~equal).
- **Depth-offset mechanism (decisive):** on a real chr19 batch, shifting `y_meta` row 0 by +2 in log2 (22→24, i.e. 4× depth) gives total-NB-mean ratio **A(plain)=1.001** (depth-invariant — the Q5 failure) vs **B(offset)=4.007** (≈4× by construction), confirming the offset reads the correct target-depth row and scales the mean correctly.
- **End-to-end (2-epoch smoke, both runs):** entrypoint trains, steps the cosine schedule, evaluates, and writes `metrics.jsonl` + `resolved_config.yaml`. Per-epoch `training_metadata_probes/depth_count_ratio` ≈ 1.00 (baseline) vs ≈ 4.00 (offset); count NLL decreases.

These confirm the harness tests what we intend; the open question (does the offset *help* count imputation, not just restore depth sensitivity) is answered only by the full 200-epoch A/B.

## Findings

Artifact evidence from log-observability on jobs 42441522 (A) / 42441523 (B), 200 epochs, `count_only`, assay-only masking, type1_chr19.

### Primary metrics (epoch 199)

| Metric key | A (plain) | B (depth_offset) |
|---|---|---|
| `training_metadata_probes/depth_count_ratio` | 1.10 | **4.03** |
| `eval_metrics/imp_count_pearson_gw` | 0.289 | **0.371** |
| `eval_metrics/den_count_pearson_gw` | 0.481 | **0.660** |
| `eval_losses/count_imp_loss` | 2.065 | **1.598** |
| `eval_losses/count_obs_loss` | 0.912 | **0.709** |

Best `eval_losses/count_imp_loss`: A 1.572 @ ep9; B **1.386** @ ep119.

Both runs completed without divergence (`nan_inf_count=240` from count_only pval/peak NaN in total_loss only). `rank_runs.py`: INELIGIBLE (no finite `eval_losses/total_loss`).

### W&B / eval_metrics_median note

`eval_metrics_median/*` does **not** appear in W&B or `metrics.jsonl` for these runs. `train_candi_v2.py` uses `run_eval_pass` from `sandbox/train.py`, which logs only genome-wide pooled `eval_metrics/*_gw` keys. The log-observability `summarize_runs.py` script **aliases** `eval_metrics_median/den_count_pearson` → `eval_metrics/den_count_pearson_gw` for legacy compatibility — those "median" columns in summarize output are gw pool metrics, not per-assay medians. True per-assay medians exist only in older JEPA decoder runs (historical eval path); they are not wired into the v2 trainer.

- Observed: Offset head moves depth probe to healthy range and improves all count metrics vs matched plain baseline on real v2 data.
- Interpretation: Depth enters the count likelihood correctly via target `y_meta[:,0,:]`; the offset is not just a probe artifact. Count imputation benefits, not only sensitivity.
- Competing explanations: (1) assay-only masking may amplify depth signal vs multi-mask regimes — not tested here; (2) gw-pooled metrics may hide per-assay regressions — per-assay median eval not yet available in v2 trainer.
- Decision: **Promote `decoder.count_head=depth_offset` as v2 default candidate.** Next: 3-head v2 run with offset; optionally port per-assay median eval + W&B logging to `train_candi_v2.py`.

## Spawned design questions

Opened from E30 code review (2026-05-31). Tracked in [`META.md`](META.md) as Q9 and Q10.

### Q9 — Depth-offset head: missing/cloze depth sentinels

The depth-offset parameterization reads `log2(seq_depth)` directly from `y_meta`. Sentinel values (`-1` MISSING, `-2` CLOZE) produce near-zero μ instead of a learned missing-depth behavior — breaking null-meta probes and any eval path where target depth is not a real log2 value.

**Decision (2026-05-31):** **Partial fix landed in code.** `DepthOffsetNegativeBinomialLayer` gates on metadata sentinels only (`MISSING`, `CLOZE`). Valid depth keeps the offset path `μ = 2^d · exp(η)` unchanged (E30 checkpoint-compatible on supervised bins). Invalid depth uses fallback `μ = exp(η)` — same `linear_eta` head, no fabricated depth, always on (no config knob). See `sandbox/candi_v2/decoder.py` class docstring + unit tests in `test_candi_v2_core.py`.

**Still open:**

- What prompt metadata to use at eval for truly missing assays (median, per-sample depth estimate, separate `prompt_meta` tensor) — see brainstorm in Q9 thread; ideas 2–5 from planning notes.
- Fallback path is untrained under normal assay-only masking (supervised cloze bins always have valid `y_meta` depth); train-time prompt randomization may be needed before relying on fallback for zero-shot imputation.

### Q10 — Encoder vs decoder metadata geometry: control-channel pooling

Encoder transformer FiLM pools metadata over signal assays **plus control**; decoder pre-FiLM pools signal assays only. Control metadata is always present and never cloze-masked, biasing encoder conditioning in a way the decoder never sees.

Per-assay conv FiLM is already correct (grouped tower; each track uses its own metadata). The asymmetry applies only to post-fusion global FiLM: encoder transformer mean-pools `A+1` columns; decoder pre-FiLM mean-pools signal assays `A` only.

**Decision (2026-05-31):** **Accepted as-is for now.** Conv-stack per-assay conditioning is intentional; post-fusion pooling asymmetry (encoder includes always-observed control anchor, decoder is signal-only target meta) is acceptable without an ablation. Reopen only if metadata probes or imputation regressions implicate global FiLM geometry.
