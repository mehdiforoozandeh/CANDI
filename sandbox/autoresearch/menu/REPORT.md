# Menu-AR — 5-loop autoresearch report

CANDI v2 improvement via a 5-loop stateless greedy-ratchet autoresearch, 30 iterations/loop, on the
frozen v3 judge. Run completed 2026-06-29. Companion: `VALIDATION.md` (harness audit + fixes).

---

## 1. The idea

Prior whole-program search (ERA / Flat-UCB) on this task **collapsed to a monoculture**: from an early
node onward, ~every good candidate re-derived the same "counterfactual-prior" (CF) trick, so the search
stopped exploring genuinely different designs. The menu-AR is the antidote: **5 independent loops, each
committed to ONE distinct design thesis**, that greedily improve candi_v2 along only that axis.

Mechanics per loop (one nested git repo):
- Each iteration a fresh, sandboxed `claude -p` agent (failover `cursor-agent`) makes **one surgical
  edit** to `train.py` and/or the vendored editable `candi_model/`, guided by its thesis + shared
  anti-rediscovery PRIORS (ERA's already-found tricks, given as knowledge so no GPU re-derives them).
- Gate (scope fence → CPU smoke) → score on a MIG-slice H100 → **keep if ERA_SCORE > champion, else
  revert**. Every attempt is committed; `best` tag tracks the champion.
- Rival theses are deliberately **withheld** from each agent so a greedy agent can't drift back into
  the exploited CF basin — isolation is the whole point.

**What's frozen:** the judge (`_judge/`: data layout, training budget, eval protocol, ERA_SCORE) and
the data. Everything the agent can touch is inside its loop dir (kernel-enforced by bubblewrap).

### Task, references, score
- 8-assay sandbox slice; **train chr19 → eval chr21**. Budget = min(5 epochs, 1800s) — empirically near
  candi_v2's imputation peak (8 epochs overfits: Q_imp 0.377→0.330). Bit-exact deterministic.
- **Objective B**: three heads trained on REAL labels read in lockstep from the h5 (counts NB-NLL,
  pval Gaussian-NLL(μ,var), peak BCE); `signal_pred` = pval mean.
- **ERA_SCORE** = (Q_imp − 0.4857) − 0.5·max(0,Q_imp−Q_den) + 0.4·min(0,0.0734−ECE)
  + 0.4·min(0,c_index−0.4985) + 0.4·min(0,peak_auroc−0.7161) + 0.02·DCR-band[3,5].
  Primary term = **Q_imp** (mean of imp {pval,count}×{spearman,pearson} on held-out V/B assays); the
  rest are do-no-harm floors. Higher is better.
- **Reference points:** candi_v2 row-0 base = **−0.1261** (Q_imp 0.377); marginal average-reference
  baseline Q_imp = **0.4857** (ERA floor ≈ −0.04, only DCR fails). The base is *below* the baseline —
  candi_v2 wins on rank (Spearman) but loses on magnitude (Pearson: count 0.165, pval 0.300 vs the
  baseline's 0.52 / 0.49). Closing that Pearson gap is the headroom.

---

## 2. Results

| rank | loop | best ERA | Δ base | champ iter | keeps | Q_imp | count-Pe | ECE | winning change |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **crps_calibration** | **−0.0408** | **+0.085** | 21 | 13 | 0.450 | 0.292 | 0.086 | CRPS proper score + LOO-ref deviation-correlation on pval & count |
| 2 | single_lambda | −0.0812 | +0.045 | 23 | 7 | 0.414 | 0.152 | 0.098 | λ-collapse + LOO-ref count-mean delta + decoder GroupNorm |
| 3 | axial_longrange | −0.0822 | +0.044 | 1 | 1 | 0.427 | 0.278 | 0.131 | light masked cross-assay axial attention (only) |
| 4 | factorized | −0.0828 | +0.043 | 20 | 12 | 0.419 | 0.218 | 0.113 | CP low-rank deviation + deviation-corr + log-dispersion delta |
| 5 | repr_first | −0.1024 | +0.024 | 24 | 6 | 0.408 | 0.245 | 0.136 | light Gaussian-prior latent reg + arcsinh + GroupNorm |

Base −0.1261 · marginal-baseline floor ≈ −0.04. **All 5 loops beat base; none reached positive S_A**
(best Q_imp 0.450 < baseline 0.4857).

---

## 3. Per-loop thesis, findings, champion

### crps_calibration — WINNER (−0.0408)
**Thesis:** replace NB + second-moment band-aids with a **calibrated-by-design** distributional output
(quantile/CRPS), attacking the ECE floor from the likelihood side.
**Findings:** the thesis worked and compounded. Wholesale likelihood swaps regressed (quantile /
Gaussian-approx replacements, it1/it2 ≈ −0.18). The wins were **additive**: a zero-init LOO-ref count
delta (it9), a dispersion delta (it12), then **replacing pval Gaussian-NLL with a closed-form Gaussian
CRPS proper score** (it15, −0.085→−0.053), then **light (w=0.02) LOO-ref deviation-correlation aux
losses on pval and count** (it20/it21) to −0.0408. Lifted count-Pearson 0.165→0.292 (the single lagging
metric) and ECE 0.117→0.086. **Plateaued at it21** (it23–30 flat).
**Verdict:** best loop; CRPS + leak-free deviation modeling is the effective recipe.

### single_lambda (−0.0812)
**Thesis:** predict ONE depth-free latent enrichment field λ; derive counts/pval/peaks from it (collapse
the heads so calibrating λ once calibrates every view).
**Findings:** the collapse itself was neutral (it1, held score). Gains came from the same deviation
motif (LOO-ref count-mean delta, it10) and, decisively, **decoder LayerNorm→GroupNorm** (it23, the big
jump to −0.0812). Its count-Pearson stayed low (0.152) — the λ-collapse didn't fix magnitude.
**Verdict:** modest; the win was a generic decoder-norm improvement, not the collapse thesis.

### axial_longrange (−0.0822) — thesis FAILED
**Thesis:** replace the per-position cross-assay spine with **axial (assay×position) attention + a
long-range mixer** (enhancer–promoter / TAD scale).
**Findings:** a **light** masked cross-assay axial attention *added before fusion* helped once (it1,
−0.0822). **Every heavier rewrite regressed** — dilated-sparse position attention (it10, −0.248),
replacing RoPE layers, multi-res conv pyramids on RoPE (it26). 29 of 30 iters reset.
**Verdict:** the ambitious thesis does not pay at the 5-epoch budget — the existing spine is load-bearing
and long-range restructuring needs far more training to justify itself.

### factorized (−0.0828)
**Thesis:** model the data as an explicit **low-rank tensor factorization** (cell×assay×position
factors) combined by a small DNA-conditioned net — NOT attention.
**Findings:** steady, disciplined climb — zero-init CP deviation on the enrichment reference (it4→it7),
a light deviation-correlation aux (it7), mid-scale DNA position factor (it10), additive count-reference
CP (it15), log-dispersion deviation (it20). Heavy PIT/dispersion calibration aux was catastrophic
(it5, −0.4576). **Plateaued at it20.**
**Verdict:** the factorization natively expressed the average-reference + deviation decomposition and
reached mid-pack; same deviation motif as the winners, just a different parameterization.

### repr_first (−0.1024) — weakest
**Thesis:** build a strong regularized latent Z with light heads + a light latent regularizer
(Gaussian-prior/ELBO/SIGReg), betting a better latent generalizes on held-out imputation.
**Findings:** a light Gaussian-prior moment-match on the un-detached latent helped marginally (it1).
Latent regularization gave little at 5 epochs; the only real gains were, again, generic
(log1p→arcsinh signal transform it23, decoder GroupNorm it24). Heavy latent/decoder additions regressed
badly (NB variance-to-residual calibration it6, −0.4553; per-deconv FiLM it25, −0.212).
**Verdict:** the latent-first bet doesn't cash out at this budget — the readout/decoder, not the latent,
was the binding constraint.

---

## 4. Conclusions

1. **The harness worked as designed.** Reproducible ratchet, on-thesis edits, real monotone
   improvement, no monoculture collapse (5 distinct trajectories). All 5 loops beat base candi_v2
   (+0.024 to +0.085 ERA).
2. **The winning mechanism is leak-free deviation modeling.** Across *every* successful loop the gains
   came from **zero-init deltas that model the cell-type-specific deviation from a leave-one-out
   average-reference** — exactly the PRIORS' headline hint. crps stacked it best (on pval, count, and
   dispersion) and added a **CRPS proper score** for calibration.
3. **The Pearson/magnitude gap — the identified headroom — was largely closed.** Base count-Pearson
   0.165 → 0.292 (crps); Q_imp 0.377 → 0.450.
4. **But no loop beat the marginal baseline** (best Q_imp 0.450 < 0.4857; S_A still negative). The
   binding ceiling is the **single-chromosome (chr19-only) training regime**, not architecture — a
   frozen data property, unbreakable within the current judge.
5. **Convergent evidence for two robust wins:** decoder **GroupNorm** (found independently by
   single_lambda and repr_first) and **light deviation-correlation losses** (found by crps, factorized,
   single_lambda).

---

## 5. Lessons learned

**Methodological**
- **Thesis isolation prevents monoculture** and yields a diverse portfolio — but it also means every
  loop independently re-discovers the same universal wins (deviation modeling, GroupNorm). Cross-loop
  *consolidation* at the end captures more than any single loop.
- **Light, zero-init, additive edits win; wholesale replacements lose.** Every big regression was a
  wholesale swap (replace the NB likelihood, replace RoPE, heavy calibration aux). Starting from the
  strong base's identity and adding a small correction is what the ratchet rewards.
- **A short deterministic budget is a good *relative* ranker** for light edits, but **penalizes
  architecture theses** (axial): heavy restructuring can't amortize in 5 epochs, so it always resets.
  Architecture search needs a longer budget — which here would overfit imputation, so it's a genuine
  tension, not a tunable.
- **Plateau is real and detectable:** crps/factorized flatlined by it20–21; the last ~10 iterations of
  every loop added ≤1 keep. ~20 iterations captured essentially all the gain.

**Scientific**
- On this task the model's job is to **predict the deviation from the average-reference**, calibrated;
  raw likelihood/architecture cleverness is secondary.
- **Calibration (ECE) and magnitude (Pearson) are coupled** — the CRPS proper score improved both.
- The **spine (conv towers + per-position cross-assay attention + RoPE) is load-bearing**; replacing it
  is a net loss at this scale/budget.

---

## 6. Recommendations

1. **Consolidate, don't grind.** The loops have plateaued; more iterations yield marginal gains. Build
   ONE candidate stacking the cross-loop wins — CRPS proper score + LOO-ref deviation-correlation on
   pval & count & dispersion (crps) + decoder GroupNorm (single_lambda/repr_first) — and score it once.
   Likely a new best. Do this as a deliberate merge, NOT by opening continuous cross-loop sharing (that
   reintroduces the monoculture the design avoids).
2. **Retire axial_longrange and repr_first at this budget** (dead / weak theses). Keep crps, factorized,
   single_lambda as the productive axes if any loop continues.
3. **The one lever with real upside is the data regime.** Q_imp is capped ~0.45 < baseline 0.4857 by
   chr19-only overfitting. Multi-chromosome training (the h5 has ~2000 unused windows beyond chr19/21)
   is what would push S_A positive — but it is a judge/data change (currently frozen). Flag for a
   decision before any further architecture search.
4. **Wrap the exploration.** It achieved its purpose (escaped the monoculture, found the winning
   recipe). Lock the consolidated champion and this report as the outcome.

---

## 7. Follow-up: consolidation + data-regime ceiling test (2026-07-01)

Recs #1 and #3 were tested empirically (6-cell grid; full report in `CEILING.md`). Outcome updates the
recommendations above:

- **Rec #1 (consolidate) — FALSIFIED.** The naive stack (crps + single_lambda's GroupNorm) scored
  −0.0663, **worse than crps alone (−0.0408)**: GroupNorm anti-synergizes with crps's already-calibrated
  CRPS/deviation heads. Cross-loop wins are **not additive**; **crps alone remains the best candidate.**
- **Rec #3 (data regime is the lever) — confirmed in direction, refuted in magnitude.** Adding the 2000
  type2 loci to training (leak-free wrt chr21, eval/baseline frozen) lifts base Q_imp 0.377→0.394
  (+0.017) — diversity does regularize the chr19 overfit — but **no cell reached S_A ≥ 0** (best Q_imp
  0.436 < baseline 0.4857). The ceiling is data-limited and overfitting-driven; breaking the baseline
  needs a **much larger panel (multi-chromosome / MERGED)**, not the 2000 sandbox windows.
- **New mechanism:** the overfit point moves with regularization — the weakly-regularized base peaks at
  augmented@3ep then re-overfits at 5ep, while the GroupNorm+CRPS consol keeps improving to 5ep
  (best full-model cell −0.0540). Same conclusion: architecture is not the binding constraint; data is.
