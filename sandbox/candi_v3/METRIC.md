# CANDI v3 — ERA objective (dialectic-derived, FROZEN)

*Resolved by a Hegelian dialectic (2026-06-17). Thesis proposed a multiplicative hybrid
(skill × validity factors); Antithesis showed the multiplicative form annihilates the best
imputer lineages over cheaply-fixable defects, becomes the gamed metric itself, and amplifies
single-eval noise near zero factors; Synthesis converged on an **additive-hinge ε-Pareto**
objective.*

---

## The formula

```
ERA_SCORE = S_A
          + w_cal · min(0, τ_cal − ECE)                            # calibration floor (one-sided)
          + w_dcr · ( min(0, DCR − DCR_lo) + min(0, DCR_hi − DCR) ) # DCR BAND (two-sided)
          + (−∞ if structurally degenerate)   # hard gate (structure, not performance)
```

One clear main term (`S_A`), discounted only when a candidate drops **below baseline** on a
validity axis. ERA-compliant ("clear main term"), frozen for the whole search.

### Main term — `S_A` (imputation skill)
- **Real zero-shot imputation** on the reserved **V_/B_ assays** (never seen in training for
  that biosample), scored on `imp_eval_map` per the v2 protocol (`eval.py`, PLAN §2.5). NOT a
  synthetic leave-k-out.
- Raw, UNCLAMPED units vs the marginal/mean baseline:
  `S_A = (imp_metric_candidate − imp_metric_baseline) / scale_A`.
- Headline metric = **`imp_pval_spearman_gw`** (rank-based, gaming-resistant), optionally
  blended with `imp_count_spearman_gw` / `imp_peak_auroc_gw` — **never NB-NLL alone** (NB-NLL
  is minimized by mean-matching while getting noise structure wrong). Final blend fixed in
  Stage 0 against the gaming probes.
- Unclamped so a genuine outlier improvement (e.g. 1.3·scale_A) is ranked as such under
  single-eval noise, not censored to 1.0. Background-everywhere scores ≈0 (loses to the
  marginal baseline on every nonzero region).

### Feasibility hinges — calibration & DCR
- **Calibration (one-sided floor):** `min(0, τ_cal − ECE)`: zero when `ECE ≤ τ_cal`, linearly
  negative above. ECE = mean |nominal − empirical coverage| across CI levels. Forces
  calibration to be a *live, climbable* dimension without rewarding overshoot.
- **DCR (two-sided band):** `min(0, DCR − DCR_lo) + min(0, DCR_hi − DCR)`: zero inside
  `[DCR_lo, DCR_hi]` (~[3,5], centred on the physics target 4.0 = +2 log2 ⇒ 4× depth), linear
  outside on **either** side. Both DCR→1.0 (depth-blind collapse) and DCR≫4 (unstable depth
  head) are failures, so feasibility is a **band**, not a one-sided floor. The band is
  physics-absolute — NOT anchored to the marginal baseline (which is depth-blind, DCR≈1.0).
  This is the Pareto/denoising guard: you cannot buy imputation skill by collapsing depth
  calibration, but anywhere inside the band is free.
- **Additive, not multiplicative** — a high-skill lineage with a cheaply-fixable defect pays a
  *bounded* penalty and keeps its credit, so the bandit keeps selecting and improving it.

### Hard degeneracy gate — `−∞` (sentinel below any feasible score)
Reserved for **structural collapse only**, never soft shortfalls: constant/near-constant
output, NaN/Inf, mask leakage, posterior collapse (predicted variance below a floor / σ pinned
at a boundary), peak-rate outside biological range, or a hard-constraint violation
(forbidden import, memory ceiling, control required). Structure-based → a single noisy eval
cannot turn a degenerate program into a winner.

---

## Constants (FROZEN before search → `constants_frozen.yaml`)

- `scale_A` = MAD of the held-out **imputation** metric (`imp_pval_spearman_gw`) across a
  marginal-baseline bootstrap → one unit of `S_A` ≈ one noise-σ of real improvement
  (self-calibrating to eval noise).
- `τ_cal` = marginal-baseline ECE ("do no harm vs the trivial predictor").
- `[DCR_lo, DCR_hi]` = **physics-absolute band** (~[3,5], centred on 4.0), NOT baseline-derived
  — the marginal predictor is depth-blind (DCR≈1.0), so it cannot anchor this floor.
- `w_cal = scale_A/scale_cal`, `w_dcr = scale_A/scale_dcr` (MADs of ECE, DCR under the
  baseline bootstrap) → one noise-σ of violation costs ≈ one noise-σ of skill. No hand-tuning.

> Note: baseline = marginal/mean predictor (architecture-agnostic). This is a *weak* floor
> ("beat trivial + do no harm"); the production-promotion bar in `PLAN.md` Stage 3 is the
> stronger "beat the best known model" test, run separately and not inside the per-candidate
> score.

---

## Why not the alternatives
- **Pure weighted-sum:** no clear main term (ERA anti-pattern); lets a dead imputer buy back
  score with cheap calibration points.
- **Pure multiplicative hybrid:** annihilates best-imputer lineages over fixable defects;
  the product becomes the gamed target; amplifies single-eval noise near zero.
- **Pure hard-gating on A/B/C:** brittle under single noisy eval; discards the gradient the
  bandit needs.
- **Additive hinge (chosen):** one maximand, credit preserved, calibration forced live,
  denoising guarded, noise stays additive, degeneracy handled structurally.
