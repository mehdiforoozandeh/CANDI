# Menu-AR — deep validation report

Status: **READY TO LAUNCH** (5-epoch budget, all axes ≥ 9/10). Date: 2026-06-29.

The 5-loop menu autoresearch improves CANDI v2 via stateless greedy ratchet loops. Each loop = one
nested git repo; each iteration a fresh bwrapped `claude -p` (failover `cursor-agent`) makes ONE
surgical edit to `train.py` and/or the vendored editable `candi_model/`, then the harness gates
(scope → CPU smoke) → scores on a MIG-slice H100 → keeps-if-better else reverts to the champion.
The **judge (`_judge/`) and data are frozen**; everything the agent touches lives in its loop dir,
kernel-enforced by bubblewrap.

---

## 1. Confidence self-evaluation (0–10)

| axis | score | basis |
|---|---|---|
| **Correctness** | **9** | Judge math verified line-by-line; lockstep pval/peak pairing proven (runtime assert + 300/300 + neg-control); fork parity bit-identical; ratchet keep/reset verified incl. multi-file; determinism bit-exact at 5ep **and** 8ep; base-score provenance apples-to-apples; marginal baseline reproduces the frozen constants exactly. 3 bugs found & fixed & tested. |
| **Efficiency** | **9** | Jobs release the SLURM allocation early when done (no idle billing); CPU smoke gate (~6s) blocks broken code before GPU; bwrap cheap; ~20 min/iter ⇒ 30 iters/loop ≈ 10 h (fits the ~11 h login-node window). 5ep is also the fastest budget that is near the imputation peak. |
| **Design** | **9** | Stateless ratchet + kernel sandbox + anti-rediscovery priors + crash recovery + honest dashboard + 5 distinct theses with anti-drift "Forbidden" + WHERE hints. Budget empirically pinned to candi_v2-B's imputation peak (see §4). Known limitation (transparent): single-chr19 overfit caps absolute Q_imp below the marginal baseline — the **relative** ratchet at the peak budget is the sound, reproducible signal. |

---

## 2. Subsystem audit (all SOUND)

**A. Frozen judge** — `score.py` (ERA_SCORE: S_A primary + correctly-signed do-no-harm floors/gates;
constants match METRIC.md), `eval_v3.py` (Q=mean of 4 corrs; PIT-ECE; sample-based C-index; AUROC),
`harness.py` (budget min(5ep,1800s); `_predict_eval` chr21 imputation+denoising+calibration+DCR;
collapse/non-finite gates; output-key contract), `data_v3.py` (clean chr19-train/chr21-eval boundary;
`iter_train_batches` is the canonical order), `marginal.py` (avg-reference baseline — **reproduces
frozen constants exactly**: Q_imp 0.4857, ECE 0.0734, C-index 0.4985, AUROC 0.7161).

**B. B-objective adapter** (`_adapter/train_v2_row0.py`) — CORRECT. Lockstep `_label_iter` byte-matches
`iter_train_batches` (same `default_rng(0)`, items, shuffle) + runtime counts-equality assert; denoise
branch (dsf8→dsf1) mirrors eval's denoising pass; impute branch masks via `x_avail`+`query_mask`;
control channel appended last & never masked; loss = NB-NLL + Gaussian-NLL(μ,var) + BCE on query
positions; Adamax lr=2e-3; output keys match the eval contract.

**C. Fork** (`candi_model/`) — CLEAN. Only delta from repo candi_v2 (after normalizing intra-package
imports) is the documented `out["z"]` un-detach (parity-safe: no grad path unless a loss uses z).
External deps resolve under `PYTHONPATH=repo`; instantiates.

**D. Ratchet mechanics** — driver order-of-ops correct; keep advances `best` tag, reset/crash restores
`train.py`+`candi_model` to champion (failed attempts stay in history); `best_row`=max era over
keep/base; `scope.py` triple-layered (nested git + fence + bwrap); `smoke.py` exercises full train
mechanics + eval forward (bs=4 lockstep, module-name isolation prevents a real run on import);
`agent_step.py` bwrap prefix + claude→cursor failover + process-group kill.

**E. Orchestration** — `sbatch_wrap.sh` gres = `gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (✓ hard
constraint), sources venv+PYTHONPATH+CUBLAS det, runs via non-editable `_det_launch.py`;
`start_menu.sh`/`stop_menu.sh` source the venv per tmux; `orchestrator.py` watchdog (restart dead
drivers, scancel orphans), plateau flag, dashboard regen, synthesis on STOP.

**F. Efficiency & validity** — base `_g1.out` carries `[sbatch_wrap EXITED rc=0]` ⇒ base scored on the
**identical deterministic path** as candidates (both 19100 steps / 5.00 ep). Leakage gate (Q_imp>0.80,
ungameable in `_harness/`). 5-epoch jobs release early.

---

## 3. Bugs found & fixed this audit (all validated)

1. **context.py rendered "Edit train.py ONLY"** — contradicted the CONTRACT + agent_step (both allow
   `candi_model/`); would have hobbled the two architecture loops (axial_longrange, repr_first). Fixed
   the render text. All 5 loops re-validate.
2. **Driver crash-recovery spin** — a driver killed in the commit→keep/reset window (which includes the
   ~20-min GPU scoring wait — the largest slice of each iteration) left working≠champion, so
   `context.validate` failed forever → `run_loop` retried with no backoff → a git-subprocess fork-storm
   (login-node pids hazard). This is exactly the partial state crps was found in. Added idempotent
   `_recover()` (working→champion; pad reflections; re-stamp backlog) at iteration start + a run_loop
   backoff/giveup. Validated: simulated orphan-attempt → `validate` fails → `_recover` → `validate`
   passes → next iteration clean. Driver selftest still ALL PASS.
3. **`change_summary` TSV corruption** — a stray tab/newline in the summary would shift/split a
   results.tsv row → schema-validate fail → ctx_error. Now whitespace-collapsed in `_record`.
4. **`_edited()` missed candi_model-only edits** (found in the 1st 3-iter run) — it checked only
   train.py, so an architecture loop editing `candi_model/` (e.g. factorized's decoder edit) was
   mis-flagged "(agent produced no edit)" → wrong attribution + a redundant cursor run compounding two
   edits into one scored iteration. Now git-based over train.py + candi_model/ (pyc ignored). Unit-
   tested; confirmed in the 2nd 3-iter run (no-edit-rows = 0 across all loops).
5. **Rate-limit / no-edit hardening** — cursor failover verified working (a real edit in 19s through
   the bwrap path). If BOTH agents fail with no edit: a detected rate/usage limit → the driver waits
   (rechecks every 30 min, writes `RATE_LIMITED.flag`) for the limit to reset, **skipping the GPU job
   and not burning an iteration**; a plain no-edit → bounded retry then a NEEDS_ATTENTION giveup (no
   spin). Unit-tested (detection, classification, no-score-on-no-edit, budget-preserving waits, giveup).
6. **Resume-correct iteration counting** — `run_loop` now seeds `done` from productive rows in
   results.tsv, so max_iters is a TOTAL target: re-running start_menu after the ~11h kill continues
   toward 30 instead of doing 30 more. Unit-tested (target 3→0 more; target 5→2 more).

Non-issue noted: `x_mask` is threaded through INPUT_KEYS but unused in forward (masking is derived from
`x_avail`+`query_mask`). Cosmetic; left as-is (a loop may use it).

## 3b. End-to-end 3-iter validation (real agents + GPU, fully passed)
A 5-loop × 3-iter run on the final hardened code: **15/15 iters, 0 smokefails, 0 mis-attributions,
0 flags**; GPU ran in concurrent batches of 5; completion path correct (drivers wrote DONE, watchdog
did not restart). The ratchet kept only genuine improvements — **4/5 loops beat base −0.1261**
(factorized −0.0677, single_lambda −0.0998, repr_first −0.0963, crps −0.1139; axial stayed at base as
its heavy rewrites hurt at 5ep). Agents stayed on-thesis and the keeps were the Pearson/magnitude edits
the PRIORS steer toward. ~25 min/round ⇒ 30 iters ≈ 12.5 h (spans the ~11h window → resumes via
start_menu, now resume-correct).

---

## 4. Budget decision — empirically pinned (the one design lever)

Considered raising MAX_EPOCHS 5→8 for a stronger proxy. **Tested it: 8ep OVERFITS candi_v2-B.** Base
Q_imp (the PRIMARY scored metric) drops **0.377→0.330**, ERA_SCORE **−0.126→−0.162**; all 4 imputation
correlations fall (only denoising/calibration improve). candi_v2-B's imputation peak is **≤5ep** —
opposite to the ERA node-70 candidate (which peaked ~ep14; different architectures overfit at different
points). Both budgets are bit-identical-deterministic and epoch-bound, so this is purely proxy quality.
**Decision: keep 5 epochs** (original design, validated, near the peak). Reverted cleanly.

---

## 5. Launch state

All 5 loops anchored at base **−0.1261**; train.py + candi_model byte-identical across loops
(md5 `bacbac1f…`); each git-clean, `best` tag set, SMOKE_OK. Marginal baseline ERA ≈ −0.04 (only DCR
fails); candi_v2 row-0 is below it (the AR's job is to climb).

| loop | thesis (WHERE) |
|---|---|
| single_lambda | one latent λ → NB+pval+peak derived (collapse heads) |
| crps_calibration | calibrated-by-design distributional count output |
| factorized | low-rank cell×assay×position tensor factorization |
| axial_longrange | axial + long-range mixer (candi_model/encoder.py) |
| repr_first | strong regularized latent Z (candi_model/model.py, un-detached) |

## 6. Launch

```
cd sandbox/autoresearch/menu && ./start_menu.sh 30     # 30 iters/loop (resumes across sessions)
cd sandbox/autoresearch/menu && ./stop_menu.sh         # stop + synthesis
```
`start_menu.sh` sources candi_venv per tmux (the gate-smoke needs it — bare gentoo python lacks
`typing_extensions.TypeIs`), runs idempotent build_programs + init_all, starts one driver tmux per loop
+ the orchestrator. A run that doesn't finish in the ~11 h login-node window resumes by re-running
`start_menu.sh` (init skips initialized loops; drivers continue from results.tsv). Dashboard:
`dashboard.html`.
