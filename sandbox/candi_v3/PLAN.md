# CANDI v3 — ERA-driven first-principles redesign: master plan

*Status: LOCKED via grilling session 2026-06-17. Companion docs: [`DESIGN_MENU.md`](DESIGN_MENU.md)
(→ becomes ERA `problem.description`), [`METRIC.md`](METRIC.md) (dialectic-derived objective).*

CANDI v3 is designed **from first principles** by an ERA (Flat-UCB empirical-software
search) loop. ERA proposes whole candidate programs (model + objective), a frozen harness
scores each, and a flat PUCT bandit keeps a population and decides what to improve. We do
**not** hand-design v3; we hand-design the *frame* ERA searches in (seed, constraints,
metric, harness) and let it iterate.

---

## 0. Locked decisions

| # | Decision | Value |
|---|----------|-------|
| 1 | **ERA rewrite surface** | ERA rewrites exactly two files: `model.py` (architecture, forward, heads) + `objective.py` (loss, likelihoods, corruption/masking process). Everything else is FROZEN. |
| 2 | **Frozen (ERA may not touch)** | `data.py` (reuses v2 sandbox loader), `eval.py`, `score.py`, `constants_frozen.yaml`, `harness.py`, SLURM wrapper, `futs.py` (the judge). |
| 3 | **Metric** | Additive-hinge ε-Pareto objective (see [`METRIC.md`](METRIC.md)). One main term (imputation skill) + one-sided feasibility hinges (calibration, DCR) + hard `−∞` degeneracy gate. |
| 4 | **Score aspects** | (A) held-out imputation skill **[primary]** = `imp_*` metrics on the **reserved V_/B_ assays** (real zero-shot imputation, see §2.5), (B) calibration (coverage/ECE), (C) denoising (`den_*`) + DCR. **Z→RNA-seq biological validation EXCLUDED** from the per-candidate score (too slow); used only in final promotion validation. |
| 5 | **Baseline anchor** | **Marginal / mean predictor** scored through the v3 harness fixes `scale_A`, `τ_cal`, `w_cal`, `w_dcr`. **DCR is anchored to a physics-absolute band around 4.0, NOT to the baseline** (a marginal predictor is depth-blind → DCR≈1.0, the failure value). See §2.5. Frozen pre-search. |
| 6 | **v3 inherits** | **Data only** — `sandbox/data/sandbox.h5` (8-assay EIC slice) + the frozen v2 data loader/batching. **Cold-start stub seed** (no v2 backbone carried over). |
| 7 | **Design priors** | `problem.description` = a MENU of critique-derived directions to try (ERA chooses freely) + non-negotiable HARD CONSTRAINTS enforced by the harness. See [`DESIGN_MENU.md`](DESIGN_MENU.md). |
| 8 | **Scope fence** | Everything lives under `sandbox/candi_v3/` (harness + ERA scaffold + runs). Candidates are self-contained per ERA contract; `execute.py` guard rejects any candidate that writes outside its own temp workdir. **Nothing outside `sandbox/candi_v3/` is ever modified.** |
| 9 | **Per-candidate budget** | Decided in Stage 1 from measured seed timing; sandbox 8-assay regime on `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1` (MIG 10GB H100). |
| 10 | **Generator** | `claude -p --model opus` headless (subscription-billed); Cursor/Composer failover. |
| 11 | **Iteration model** | **One-shot search per round**; metric FROZEN within a round. Between rounds we may hand-pick a winner to re-seed a fresh round with possibly-refined hard constraints (never a changed metric mid-run). |

---

## 1. Hard constraints (frozen; harness-enforced where checkable)

From the user's standing requirements — these bound every candidate ERA produces:

- **Native DNA tower only.** No Borzoi/Enformer/genomic-LM or any external pretrained
  model. ERA may redesign/optimize a native DNA tower, but may not import or fine-tune
  external weights.
- **Memory-efficient context.** The bottleneck to longer context is the quadratic
  attention memory. Any long-context proposal MUST be memory-efficient (dilated /
  hierarchical / linear-/flash-style / two-scale) and respect this constraint.
- **Control-optional.** Some biosamples have no ChIP control. The design must work *without*
  a control track and use it *if available* — never require control as an input.
- **Fixed decoder first.** Start with a fixed decoder. Query-based decoding is implemented
  as an *option* to explore later, not the initial requirement.
- **No full JEPA.** Latent regularization is welcome via the *lighter* routes — Gaussian
  prior + ELBO (known to curb overfitting) or SIGReg on the latent. Full JEPA training is
  deprioritized (heavier, opens a pandora box) -- we already have some experiments where we explored JEPA (sandbox/ideas/EXPERIMENTS.md).

A candidate violating a checkable constraint (e.g. imports a forbidden model, exceeds a
memory ceiling, hard-requires control) → degeneracy gate → `−1e9`.

---

## 2. Objective (summary — full derivation in METRIC.md)

```
ERA_SCORE = S_A
          + w_cal · min(0, τ_cal − ECE)     # calibration floor (one-sided hinge)
          + w_dcr · min(0, DCR − τ_dcr)      # denoising floor (one-sided hinge)
          + (−∞ if structurally degenerate)  # hard gate, structure not performance
```
- `S_A` = held-out-assay imputation skill in **raw unclamped units** vs the marginal
  baseline, using a gaming-resistant metric (Spearman / peak-AUPRC / −MSE — **never**
  NB-NLL alone).
- Hinges are **zero when feasible**, linear when a candidate drops below baseline → a
  high-skill lineage with a cheaply-fixable defect keeps its credit (the bandit does not
  abandon it), yet cannot win imputation by sacrificing calibration or denoising.
- Constants fixed before search; weights `w_k = scale_A/scale_k` (MAD-based) so one noise-σ of
  floor violation costs one noise-σ of skill. `scale_A, τ_cal` from the marginal baseline; the
  **DCR feasibility band is physics-absolute** (§2.5), not baseline-derived.

---

## 2.5 Evaluation protocol (inherited verbatim from v2 — the frozen `eval.py`)

The v3 harness reuses v2's train/eval data semantics exactly (`sandbox/data.py`,
`sandbox/eval.py`):

- **Biosample/assay split.** 8 fixed assays (`SANDBOX_ASSAYS`). Each biosample's assays are
  partitioned into **T** (train/observed), **V** and **B** (reserved). `T_<name>`, `V_<name>`,
  `B_<name>` are the *same biosample*, disjoint assay subsets. Training masks/denoises only
  T_ assays; **V_/B_ assays are never seen in training**.
- **(A) Imputation skill — the real metric.** At eval (`train=False`,
  `eval_include_vb_ground_truth=True`, `imp_prefixes=("V_","B_")`), predict the reserved
  V_/B_ assays from the observed T_ assays. Scored on `imp_eval_map` (T-unavailable &
  V/B-available) as `imp_*` metrics — **`imp_pval_spearman_gw` is the headline** (rank-based,
  gaming-resistant), optionally blended with `imp_count_spearman_gw` / `imp_peak_auroc_gw`.
  `S_A` = (headline imp metric − marginal-baseline imp metric) / `scale_A`. This is genuine
  zero-shot imputation — not synthetic leave-k-out.
- **(C) Denoising.** Reconstruct T_ assays at higher depth via DSF downsampling → `den_*`
  metrics. **DCR** = `prompt_sensitivity_depth_count_ratio` (NB-mean ratio at +2 log2 depth):
  healthy ≈ **4.0** (= the 4× depth ratio), collapse → 1.0, unstable ≫4. Feasibility is a
  **band** `DCR ∈ [DCR_lo, DCR_hi]` (~[3,5]) centred on 4.0 — a two-sided hinge, NOT a
  one-sided floor (see `METRIC.md`).
- **(B) Calibration.** Empirical coverage vs nominal CI from the distributional outputs → ECE;
  `τ_cal` anchored to the marginal baseline ("do no harm").
- **Eval region.** Held-out chromosome (chr21) windows, as in v2 (`_eval_indices`).

---

## 2.6 Harness ↔ candidate contract (FROZEN)

The frozen harness trains the candidate (fixed budget) and scores it. The candidate's
`model.py` must expose a model whose forward returns a dict the harness reads — minimal and
**likelihood-agnostic** (this is the critique §9 design encoded as a contract):

```python
forward(x_data, x_dna, x_meta, y_meta, query_mask, query_mask_signal) -> {
    "count_dist":  torch.distributions.Distribution,  # over raw counts, broadcastable [B,L,F]
    "signal_pred": Tensor[B,L,F],                      # enrichment prediction (imputation Spearman)
    "peak_prob":   Optional[Tensor[B,L,F]],           # optional; enables peak AUROC
}
```
- `count_dist` is **the one learned likelihood** (NB/Poisson/…); ERA picks it. Harness derives:
  count mean (DCR + count Pearson/Spearman/R²), and coverage of held-out counts → **ECE
  (calibration computed on the count distribution)**.
- `signal_pred` feeds the **headline** imputation metric (`imp_pval_spearman_gw` = Spearman vs
  `y_pval_imp` on `imp_eval_map`). ERA freely chooses whether `signal_pred` is a deterministic
  readout of `count_dist` (critique §9.1) or a separate head (v2-style) — the search decides.
- `objective.py` owns the loss + corruption; the harness only requires the forward dict above.
- **DCR** = harness re-runs forward with `y_meta` depth shifted +2 log2 and takes the
  `count_dist.mean` ratio (the v2 `prompt_sensitivity_depth_count_ratio` probe).

**Naive imputation baseline (S_A zero-point):** per-position **average-reference track** —
predict a held-out V_/B_ assay as the per-position mean of that assay over the training (T_)
biosamples that have it (classic ChromImpute/Avocado baseline; honest position-aware floor).
Frozen into `constants_frozen.yaml`.

---

## 3. Stages, gates, deliverables

### Stage 0 — Frozen harness + baseline + selftest  *(GATE before any search)* — ✅ DONE (2026-06-18)
**Built & validated:** `data_v3.py`/`eval_v3.py`/`score.py`/`harness.py`/`contract.py` (frozen harness),
`baselines/marginal.py` (imp Spearman 0.4652), `constants_frozen.yaml`, `notebook.py`
(RESULTS.tsv/tree.mmd/NOTE.md), `problem.py`/`config.yaml`/`run.py`/`generate.py`/`execute.py`/`futs.py`
(ERA driver, flat in `candi_v3/`). **selftest.py ALL PASS**: baseline S_A=0; score monotone in
skill + DCR band; collapse & non-finite gates fire; neutral seed trains end-to-end → finite score
(DCR 3.97 in-band). FUTS mock + generate/execute selftests green. **GATE 0→1 cleared.**

<details><summary>original Stage 0 spec</summary>
- Build `data.py` (reuse v2 loader), `eval.py` (computes A/B/C aspects), `score.py`
  (the objective above), `constants_frozen.yaml`, `harness.py`, SLURM wrapper; vendor the
  ERA `scaffold/` (`futs.py` frozen).
- Score the **marginal/mean predictor** through the harness → write `constants_frozen.yaml`
  (`τ_cal, τ_dcr, scale_A, w_cal, w_dcr`).
- `selftest.py` assertions: degenerate stub → `−1e9`; marginal baseline → finite, `S_A≈0`;
  a hand-written decent model → `S_A>0`; gaming probes (background-everywhere,
  mean-matching collapse) → caught by the degeneracy gate.
- **GATE 0→1:** all selftest assertions green; baseline scores reproducibly; constants
  frozen and committed. **Deliverable:** working frozen harness + `constants_frozen.yaml`.
</details>

### Stage 1 — Seed timing + budget lock
- Run the cold-start stub + one minimal hand-seed end-to-end once on the MIG slice; measure
  wall-clock per candidate.
- Set per-candidate train budget, `num_iterations` N, `batch_size`, `c_puct` to fit a target
  wall-clock for the whole search.
- **GATE 1→2:** one full candidate runs end-to-end under budget and emits a valid
  `ERA_SCORE:` line. **Deliverable:** locked `config.yaml`.

### Stage 2 — ERA search (one-shot)
- Launch ERA with `problem.description` (menu + constraints), frozen metric, cold-start
  seed. Monitor `tree.json`.
- **GATE 2→3:** budget N reached. **Deliverable:** `tree.json` + printed top-k diverse
  portfolio.

### Stage 3 — Confirm + promote
- **Re-run** each top-k winner (ERA scores are single-eval hypotheses) to kill lucky
  degenerates; keep those that reproduce.
- Pick confirmed best 1–3 architectures; write up findings; optionally re-seed a Round 2 ERA
  with refined hard constraints.
- Promote confirmed best to **production-scale (35-assay, MERGED) validation** incl.
  Z→RNA-seq biological validation — outside the sandbox.
- **Done criterion:** ≥1 architecture reproduces, meets the calibration+DCR floors, and
  (goal) beats the marginal baseline on `S_A` by a margin exceeding `scale_A`.

---

## 4. Directory layout (everything under one fence)

```
sandbox/candi_v3/
  PLAN.md  DESIGN_MENU.md  METRIC.md          # this plan
  harness.py  data.py  eval.py  score.py      # FROZEN harness
  constants_frozen.yaml                        # FROZEN metric constants (from baseline)
  baselines/marginal.py                        # marginal/mean predictor (anchors metric)
  scaffold/  futs.py futs_batched.py ...        # vendored ERA engine (futs.py FROZEN)
  problem.py  config.yaml                       # ERA problem + run config
  seed/model.py  seed/objective.py              # cold-start stub seed (ERA edits copies)
  selftest.py
  runs/<node>/                                  # per-candidate temp workdirs (self-contained)
  tree.json                                     # live ERA tree (machine source of truth)
  notebook.py                                   # on_node hook: writes the 3 lab-notebook files
  NOTE.md  RESULTS.tsv  tree.mmd                # live lab notebook (see §5)
```

---

## 5. Lab notebook — live ERA artifacts

Three files are kept live throughout each search, all written from the `on_node` callback in
`futs_batched.py` + the executor wrapper (`notebook.py`). **`futs.py` (the judge) is never
touched.** Archived per round (one-shot search) before any re-seed.

### `NOTE.md` — carried-forward learnings (feeds back into generation)
- **Raw trial log:** after each candidate, the generator appends a 2–3 line reflection
  (what design it tried, why it likely scored as it did). Mechanical metadata (id, score,
  gate reason) is filled by `notebook.py`; the reflection text comes from the generator.
- **Distilled lessons:** every K nodes a distiller condenses the raw log into a stable
  "observations so far" section.
- **Feedback loop:** the distilled section is injected into every subsequent generate
  prompt — this is the cross-trial memory. **Diversity safeguard:** lessons are injected as
  neutral *observations* ("X correlated with higher S_A on nodes …"), never directives, and
  each generation additionally sees only *its own lineage's* reflections, not the global
  consensus — preserving the population diversity ERA depends on.

### `RESULTS.tsv` — chronological metrics + ERA's choices
One row per candidate, written live on completion. Columns:
`timestamp  node_index  parent_index  status  runtime_s  ERA_SCORE  S_A  ECE  DCR
feasible_cal  feasible_dcr  selected_as_parent_at_iter`
where `status` ∈ {ok, gate:<reason>} and `selected_as_parent_at_iter` lists the iterations at
which PUCT picked this node to expand (so the file shows both the metrics achieved and what
ERA chose to build on, chronologically).

### `tree.mmd` — live design-tree graph
A Mermaid `graph TD` regenerated on each node from `tree.json`: nodes = candidates
(label = id · score · key design tag), edges = parent links, best lineage highlighted,
dead/`−1e9` nodes greyed. Renders directly in the IDE; `tree.json` stays the machine source.

---

## 6. Open / deferred
- Per-candidate budget numbers (Stage 1).
- ERA knobs `N`, `batch_size`, `c_puct` (Stage 1; expect modestly-raised `c_puct` —
  whole-program candidates are high-variance).
- Exact `S_A` metric form (Spearman vs AUPRC vs −MSE blend) — finalized in Stage 0 against
  the gaming probes.
- Query-based decoder, longer context, control-as-local-background — available as MENU items
  ERA may reach, gated by the hard constraints.
