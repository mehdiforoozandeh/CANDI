# h48 — instrument fixes, re-score, and adversarial verification: report and PROPOSED verdict

**Status: gate GREEN · 4 checkpoints re-scored · every load-bearing claim adversarially verified ·
verdict PROPOSED, NOT recorded.** `crux close h48` has **not** been run.

Date: 2026-07-24. CPU only — no retraining, no GPU, no SLURM. The four existing checkpoints were the
only model input.

> **Read §3 before §2.** An adversarial verification pass (6 independent skeptics + a completeness
> critic, each re-deriving the numbers with its own code) upheld every scientific claim but corrected
> several of the statements made about them, and found two bugs in the h48 tooling itself. This document
> is the post-verification version; earlier numbers circulated for F2 were prefix-truncated.

---

## 1. What was done

Six instruments in `metrics_real.py` were fixed, each gated by a three-way verification:

| leg | where | result |
|---|---|---|
| (L) known-answer / logical | `tests/test_metrics_real.py` | green |
| (C) code-correctness vs independent brute force | `tests/test_metrics_real.py` | green |
| (R) real data (`sandbox.h5` + the 4 checkpoints) | `h48_real_checks.py --fix all` | green |

`pytest tests/` → **76 passed** (was 45). `h48_real_checks.py --fix all` → **S1, S3, S4, S5, S6, S14 all
GREEN**, zero FAILs. `h48_rescore.py` then re-scored all four checkpoints at full chr21 coverage
(608 units, 1215 target-records, 12 held-out targets).

*Gate caveat (verifier C3):* this is a **consistency gate against `METADATA_AUDIT.md`, not an orthogonal
validation** — a shared conceptual error would pass it untouched. The S3 leg in particular reproduces the
*old* selector's pathology, not the *new* selector's correctness. Test-coverage gaps: **S14 has no (C)
leg, S6 has no (L) leg**, and the suite has never been mutation-tested.

### Regression check

Raw macro CRPS reproduces every recorded anchor **to 4 decimals** (1.4950 / 1.9023 / 1.3413 / 2.0561 vs
1.495 / 1.902 / 1.341 / 2.056). Per-assay values move by 2e-6…3.3e-4 — GPU-vs-CPU float, and itself the
proof that the re-score genuinely recomputes from the checkpoint rather than reading a cache. This is a
regression check on a code path the fixes did not touch: **the h47 M1 numbers were never wrong.** What
was wrong was what they were compared against and what was concluded from them.

---

## 2. Findings (post-verification)

### F1 — The ON/OFF "magnitude Pareto" is ~84% a per-assay scale artifact ✅ *upheld*; the reordering is ⚠️ *not established*

Under the oracle per-assay scale `c* = argmin_c CRPS(NB(n, mu·2^c), y)`, the four-arm macro-CRPS spread
compresses **0.7148 → 0.1133 (84%)** — independently reproduced at 0.7099 → 0.1138 (84.0%) and at 83.5%
on a separate genomic spread. **This settles audit §6.5 and is the strongest result in the node.**

Point estimates: `wd0_on` 1.3077 < `offoff` 1.3871 < `wd0_off` 1.4026 < `main` 1.4210.

**Only "`wd0_on` is best on capability" survives inference.** A paired bootstrap clustered on the 12
held-out targets:

| comparison | Δ | 95% CI | verdict |
|---|---|---|---|
| offoff − wd0_on | +0.093 | [+0.004, +0.217] | excludes 0 ✅ |
| main − offoff | +0.023 | [−0.117, +0.153] | covers 0 |
| main − wd0_off | +0.013 | [−0.091, +0.120] | covers 0 |
| wd0_off − offoff | +0.010 | [−0.047, +0.055] | covers 0 |

Target-level sign test main vs offoff: 7+/5−, **p = 0.77**. P(main worst of four) = **0.54** — a coin
flip. The claimed ordering is the modal bootstrap ordering at only 45% of replicates; main is 3rd rather
than last in 4/12 leave-one-target-out replicates, under a no-DNase n-weighted macro, and on an
independent 1/7 genomic spread (where main − offoff flips sign to −0.036).

**Defensible statement:** `main`'s raw 2nd place *is* shown to be scale rather than capability — that
part stands. But `offoff`, `wd0_off` and `main` are **statistically indistinguishable** on capability
(spread 0.034 against ±0.13 per-comparison uncertainty), all three behind `wd0_on`.

*Instrument notes:* `scale_error` is not guaranteed non-negative — it is −0.0008 on `wd0_on`/H3K4me3
because `_oracle_scale` fits `c*` on a 20k subsample and evaluates on the full pool; and the coarse-then-
refine search lands 0.039 above the true optimum on `main`/H3K4me3, which slightly *inflates* main's
capability number, i.e. biases toward the claim. `crps_oracle_scaled` is an **in-sample** oracle (`c*`
fitted on the same 12 targets it scores) — an upper bound on capability, not an achievable score.

### F2 — h47's "functional assay steering" is a MISSING-sentinel artifact ✅ *upheld, four ways*

h47 V2 was upgraded to *supported* on `wd0_on_s0` assay-permute max|Δη| = 0.833 (bar: ≥0.10). That
protocol permutes the whole `assay_id` row across all 8 slots, sliding the MISSING(−1) sentinel onto and
off **unavailable** slots — whose prompt columns are fully `(−1,−1,−1,−1)`.

**Corrected, at true full coverage** (608 units × all 7 shifts × all positions):

| ckpt | h47 anchor protocol | REAL→REAL ids | sentinel part | 8-id sweep | cross-target ablation |
|---|---|---|---|---|---|
| main_s0_perassay | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| offoff_s0_perassay | 4.1772 | **4.1772** | 2.7312 | 4.3761 | 3.0205 |
| **wd0_on_s0** | **4.6686** | **0.0023** | 4.6686 | **0.0024** | **0.0008** |
| wd0_off_s0 | 9.7144 | **9.7144** | 8.3417 | 11.7964 | 9.0021 |

For **both** offset-OFF arms the permutation value equals the real→real value bit-for-bit — no
contamination. The artifact is specific to the arm h47 crowned: `wd0_on` is **1816× / 4224×** less
assay-discriminative than `offoff` / `wd0_off`, and misses h47's own ≥0.10 bar by **43×**.

Verified against every attack: per-slot independence is **exact** (max|Δ| at non-perturbed slots =
0.000e+00 for eta, n, mu, log2_mu on all four arms), so the decomposition is valid; the roll genuinely
changes every real→real value (0 no-ops); the signal is not hiding in μ or n (real→real max relative
|Δn|/n = 7.4e-04, |Δμ|/μ = 5.2e-04); and 0.0023 holds on den (0.0023) and imp (0.0015) positions alike.

**Mechanism (corrected — this is the actionable part for q20).** Row separation is **anti-diagnostic**:
`offoff`, the *strongest* steerer, has mean pairwise ‖·‖ 0.67 and effective rank 1.79 — 10× *smaller*
than `wd0_on`'s 7.0. And `wd0_on`'s table is at **random-init statistics** (element std 0.94 vs 0.97 for
a fresh N(0,1) table; cosine 0.988 against `wd0_off`'s independently-trained table). So "revived" is the
wrong word — the table was never destroyed and never trained. Traced stage by stage on sentinel-free
prompts, the assay signal survives the fusion MLP intact (2.24 at the second Linear vs `wd0_off`'s 2.06)
and is destroyed at the **fusion LayerNorm**: `wd0_on`'s pre-LN activation norm is **396 vs 8.9**, so LN
divides the assay perturbation by ~70 instead of ~1.6 (post-LN 0.0091 vs 1.41). `film_proj` then
attenuates a further ~18× (gain along the assay subspace 0.33× random, vs 3.3× / 3.8× for
`wd0_off` / `offoff`). End-to-end deficit **~4 orders**, not 3. A causal confirmation: forcing
depth+read_length to MISSING at the queried slot restores `wd0_on`'s assay response **260×** while
*reducing* both other arms'. Concatenation-level swamping by depth/read_length is identical in the arm
that steers fine and is therefore **not** the cause.

**Known confound (state it when recording):** in this dataset `assay_id ≡ slot index` (verified over all
608×4 prompts) and the decoder trunk emits a dedicated channel block per slot, so assay identity is
carried **structurally** and the prompt row is informationally redundant. Any verdict here bounds the
**prompt pathway**, not assay-awareness. This does not rescue the 0.833: the identical real→real swap
moves both offset-OFF arms by 4.18 and 9.71, so the probe demonstrably has power and the null is
arm-specific, not protocol-induced.

### F3 — the depth counterfactual failure is a LEVEL failure, not a steering failure ⚠️ *reframed*

The flagged assumption **holds exactly**: decoder outputs at slot *a* are **bit-identical** under
arbitrary changes to every other slot's metadata, on all four checkpoints. The all-MISSING prompt is
safe and the instrument passes a positive control (an exactly-depth-scaled oracle scores 0.7292 /
0.9167 → `dsf_counterfactual_ok=True`).

At raw scale no arm passes: `frac_min_at_true` 0.229–0.271, `CRPS(told=k,GT=k) < CRPS(told=1,GT=k)` on
3–17% of (target, level) pairs — because all four **under-predict μ** (mean μ/target at dsf1 = 0.17–0.76
over all positions; 0.07–0.23 on the scored foreground), so CRPS rises monotonically as told-depth falls
on 83–92% of cells.

**But correcting one per-target constant fitted at (GT=1, told=1) — leaving the told-depth response
exactly as produced — flips all four arms to passing:** `frac_beats_told1` → main 1.000, wd0_on 0.972,
wd0_off 1.000, offoff 0.778. So this is **the same per-assay scale error F1 quantifies**, not evidence
that the models ignore told depth.

Two instrument calibrations: **0.25 is not a chance level** — it is the deterministic value of "argmin
always at told=1"; and because the foreground is the top 2% of the level-k realization *being scored*, an
exactly-correct model reads as 2.2× under-predicting at dsf8, capping `frac_min_at_true` at ~0.73.

### F4 — under clustering, most M2 steering claims lose significance ✅ *upheld*

Target-clustered (n=12, n_fg-weighted, sign-aware) rather than position-level (n≈893k); median CI
widening **18×**. `offoff` run_type [+0.0655,+0.0739] → [−0.019,+0.183] (endpoints seed-dependent at
±0.002); `wd0_off` run_type [+0.2119,+0.2578] → [−0.086,+0.761]. Neither supports a direction. Only
`offoff`'s assay-ablation survives as directionally supported.

*Fixed during verification:* the sign test counted exact ties as negatives, so `main_s0` — whose 12
run_type deltas are all bit-exactly 0 — printed **p = 0.0005**, the most significant-looking p-value in
the table. Ties are now dropped (standard practice) and `n_tied` is emitted.

### F5 — the two headline per-assay biology claims relabel ✅ *upheld, strongly*

Exactly **one** of the 40,320 possible assay orderings scores 38/38 on the metadata join, and it is
`H5_ASSAY_ORDER`; `SANDBOX_ASSAYS` scores 5/38 and the permutation-score mean is 4.75. Every one of the
38 records matches exactly one assay (fully discriminative). Independent counts-side corroboration:
aligning counts columns to meta columns gives pearson(log2 mean counts, told depth) = **0.961, rank 1 of
40,320**, and per-column peakiness is biologically coherent only under `H5_ASSAY_ORDER`.

So: "ATAC-seq contributes 56% of the CRPS gap (3.727→5.555)" is **DNase-seq**; the collapse outlier is
**H3K9me3**, not H3K27ac.

### F6 — S23 (condition-recoverability) is NOT a validated instrument ❌ *withdrawn*

The probe's ordering is **inverted against every other measurement**, and de-prefixing it did not fix
that:

| ckpt | assay acc | feature energy | run_type acc | rt energy |
|---|---|---|---|---|
| main_s0_perassay | 0.1250 (= chance) | 2.3e-17 | 0.5000 (= chance) | 0.0 |
| offoff_s0_perassay | **0.0907 (BELOW chance)** | 1.62e-01 | 0.8333 | 5.97e-01 |
| wd0_on_s0 | **0.3142** | 2.73e-05 | 0.9688 | 1.97e-04 |
| wd0_off_s0 | 0.4197 | 4.13e-01 | 0.9167 | 1.13e+00 |

`offoff` — which carries **5,900×** more feature energy than `wd0_on` and is one of the two arms with
real assay steering — scores *below* the 0.125 chance level, while `wd0_on` scores 2.5× higher on
essentially no signal. Diagnosis (verifier C5): `_leave_group_out_nearest_centroid` on within-target-
centred features **penalises a target-adaptive response** whose direction flips sign between targets
(offoff's mean cross-record cosine is 0.072, bimodal) and can **reward a deterministic near-zero one**.
It correctly identifies a bit-exactly dead pathway (`main` = exact chance, energy ~0) and nothing more.

**Do not cite S23 in either direction.** Report it as a dead/alive detector only, or hold it pending a
redesign.

---

## 3. Verification pass — what it changed

6 adversarial skeptics + 1 completeness critic, ~45 min, each re-deriving numbers with its own code.
**All six returned UPHELD_WITH_CAVEAT at high confidence.** No claim was refuted; several statements were
corrected.

### Bugs found in the h48 tooling itself (all now fixed, 76 tests still pass)

1. **`_anchor_probe(n_units=20)` and `_assay_id_sweep(n_units=15)` sliced a contiguous PREFIX**
   (`h48_real_checks.py`), i.e. exactly the audit-S24 region that is ~8× sparser than the panel — while
   the report claimed "full coverage". These are max-statistics, so truncation *understated* them.
   Corrected values raise every number and change no ordering (`wd0_on` 0.0016 → **0.0023**; bar miss
   60× → **43×**). Both now take an even `stride`, never a prefix.
2. **`_recoverability_probe(max_records=120)` truncated by unit order** — same prefix bug. Now strides.
3. **`_sign_test_p` counted exact ties as negatives** → the perfectly-null arm got the most significant
   p-value in the scorecard. Ties now dropped.
4. **`median_frac_log2mu_at_clamp` hid a heavy tail.** Now emits `frac_targets_any_clamp` / `p90` / `max`:
   `wd0_on` **16.9%** of targets clamp (p90 0.475), `wd0_off` 15.1% (p90 0.475), `main` 6.3%, `offoff`
   0.2%. So `total_slope = 1.0000` for `wd0_on` is read partly through the clamp on a minority of targets.
5. Stale source comment (the old "31% of foreground" figure, measured against the pre-S3 foreground) —
   reconciled.

### Still unchecked by anyone (carry forward)

* **`_foreground_mask` purity fallback fires on ~18% of real held-out records and is invisible.** When
  <5 positions have `target ≥ 1` the filter is dropped and the mask becomes ~61 arbitrary all-zero bins —
  S3's original pathology at reduced scale — and such a record reports `n_fg=61`,
  `fg_frac_realized=0.0199`, i.e. it looks healthy.
* **Rank-selector tie-breaking is genomically biased.** 87.8% of records break a tie at the threshold,
  mean 43.5% of the mask filled by `argsort(kind="stable")` taking the *last* index among ties → mean
  normalized genomic position 0.618 (unbiased 0.50). Interaction with deconv edge behaviour unknown.
* **M3 was never re-scored** — `h48_rescore.py` carries it forward from the pre-fix run, so it still
  labels with `SANDBOX_ASSAYS` and carries the S27 "between"-pool contamination. The scorecard mixes
  re-scored M1/M2 with carried-forward M3.
* **The test suite has never been mutation-tested**, despite the audit having found the *old* null test
  passed under mutation. "76 passed" is a liveness signal, not a sensitivity one.

---

## 4. PROPOSED verdict (for PI decision — NOT recorded)

**h48: SUPPORTED WITH CAVEATS — four of six verifiables MET, two PARTIAL.**

| # | verifiable | verdict |
|---|---|---|
| 1 | validation gate | **MET** (76/76; reword "independently-measured" → consistency gate vs the audit) |
| 2 | total-slope + real-z metadata-ablation | **MET, numbers corrected** — `wd0_on` assay Δη = 0.0023 at true full coverage |
| 3 | oracle decomposition + honest marginal | **MET** — strongest result; 84% compression reproduced twice |
| 4 | M2 CIs clustered / sign-aware / real-foreground | **MET** — with the tie bug fixed; no 5-pair conservative bar, foreground fallback unflagged |
| 5 | per-assay labels corrected | **PARTIAL** — fixed in `metrics_real.py` only; `data.py:84-100`, `run_real.py:174`, `chr21_umap.py:49`, `report*.py` and the bake-time assertion remain |
| 6 | DSF counterfactual + z-controlled recoverability | **PARTIAL** — S14 valid but its verdict is confounded by absolute level; **S23 withdrawn** |

### On h47 V2 — recommend splitting the claim

h48's verifiable 2 reads: *"if wd0_on ≤ main_s0 then h47 V2 is REFUTED not inconclusive."* Literally the
inequality does **not** hold (0.0023 > 0.0000 bit-exactly). But h47's own pre-registered bar was a
**functional threshold** (max|Δη| ≥ 0.10), and the evidence that cleared it is 99.95% sentinel.

h47 conflated two claims. Recommend recording them separately:

* **V2a (weight-level): `wd=0` prevents annihilation of the decoder assay embedder — SUPPORTED.**
  `main`'s table is annihilated (~1e-40) and bit-exactly blind from the fusion's first `Linear` onward;
  `wd0_on`'s is full-rank and injective (effective rank 6.8/8). Caveat the word "revived": the table is
  at random-init statistics, so it was not destroyed *and* not trained.
* **V2b (function-level): `wd=0` produces functional assay steering — UNMET.** 0.0023 against a 0.10
  bar, 43× short, with a mechanism (fusion-LayerNorm scale) and a positive control firing at 1816–4224×
  on other arms. Record the `assay_id ≡ slot index` confound: this bounds the prompt pathway only.

Not "refuted" (the literal inequality holds and the weight-level claim is real); not "inconclusive" (we
have full coverage, three concordant sentinel-free probes, a positive control and a mechanism).
**"Unmet-for-assay" is the reading that keeps both facts.**

h47's V1 / V3 / V4 / V5 are untouched, with **V3 qualified**: the magnitude margin is 0.079 on capability
rather than 0.561 raw, and within the target-clustered noise floor for the three trailing arms.

### Consequences for q20

* **Arbiter:** beat `wd0_on`'s capability term **1.3077** *by more than the target-clustered noise floor
  (~0.09)* — not a bare 4-dp threshold. Constraints: macro-Sp ≥ 0.5653, ECE ≤ 0.0533.
* **h50 (explicit per-assay output factor) is the highest-value arm** — rest this on the **compression**
  result (0.52–0.65 macro CRPS of fixable per-assay scale on the offset-OFF arms), which is rock solid,
  **not** on the reordering, which is not established.
* **h55 / h52 verifiables must be rewritten.** Both currently read "real-z assay-permute Δη ≥ wd0_on
  0.833" — the contaminated number. Sentinel-free the bar is **0.0023**, which those arms would clear
  trivially; the real question is why offset-ON cannot reach the 4.18–9.71 the offset-OFF arms already
  show. F2's mechanism suggests the target is **fusion-LayerNorm scale**, not embedder revival.
* **F3 means no arm currently has a working depth counterfactual**, and the failure is level, not
  steering — another point for h50.

---

## 5. Out of scope / flagged, not fixed

* `sandbox/data.py:84-100` (`build_canonical_meta`) carries the same permuted-assay-order bug and is
  **live in production** via `sandbox/train.py:560` (`use_canonical_missing_meta=True` default). Inert in
  q19 (`use_canonical=False`), not inert in production.
* `run_real.py:174` still passes `SANDBOX_ASSAYS` to `build_canonical_meta`; `chr21_umap.py:49` and
  `report*.py` unaudited; no bake-time assertion in `prepare_h5.py`.
* Verified sound and NOT an open item: the h5-column → dataset → model-slot → metric-index chain
  (`sandbox/data.py:290-302` indexes `counts_dsf{k}` and `meta_dsf{k}` by the same `fi` with no
  reordering), so `H5_ASSAY_ORDER[a]` is correct end-to-end.
* The whole `sandbox/diagnostics/dual_conditioning_real/` directory is **untracked in git**.
