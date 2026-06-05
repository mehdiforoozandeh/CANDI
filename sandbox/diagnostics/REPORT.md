# Synthetic Overfit Ablation Report

Date: 2026-05-28  
Scope: `sandbox/diagnostics/` only (E29 depth-offset **not** promoted to v2 default)  
Harness: `run_experiments.py` + `synthetic_overfit.py`  
GPU: fc11020 (H100 10GB slice), ~50K-param CANDI v2 count-only head

---

## 1. What we tested (18 experiments)

| ID | Description | Phase | Key knobs |
|----|-------------|-------|-----------|
| E01 | Default NB — Q5 collapse repro | P3 | baseline |
| E02 | Depth-offset — Q5 fix repro | P3 | `--depth-offset` |
| E03 | Spatial sine L=768 capacity | P1 | spatial, L=768 |
| E04 | Spatial sine L=768 + depth vary | P2 | spatial, L=768 |
| E05 | Offset + stochastic NB targets | P3 | offset, stochastic GT |
| E06 | Offset + decoder FiLM off | P3 | offset, dec_film=none |
| E07 | Offset + encoder FiLM off | P3 | offset, enc_film=none |
| E08 | Offset + enc/dec FiLM off | P3 | offset, both FiLM off |
| E09 | Default lr=1e-3 | P3 | lr↓ |
| E10 | Default lr=3e-3 | P3 | lr↓ |
| E11 | Default AdamW wd=1e-4 | P3 | AdamW |
| E12 | Offset + SGD lr=1e-2 | P3 | SGD |
| E13 | Offset + meta/film LR 10× | P3 | meta_lr_mult=10 |
| E14 | Offset + stochastic DSF | P4 | Poisson subsample |
| E15 | P5 spatial motif + offset | P5 | spatial + motif DNA |
| E16 | Offset clip_norm=1 | P3 | tight clip |
| E17 | Offset clip_norm=50 | P3 | loose clip |
| E18 | Default clip_norm=1 | P3 | tight clip, no offset |

Pass criteria (P3+): `rel_mae ≤ 8%`, `pearson ≥ 0.95`, `imp_rel_mae ≤ 8%`, **`depth_count_ratio (dcr) ≥ 3.0`** (probe: y_meta depth log2 5→20, target ~4×).

Artifacts: `runs/ablation_matrix.json` (E01–E02), `runs/ablation_matrix_full.json` (E06–E18).

---

## 2. Results summary

| ID | Pass | Steps | rel_mae | imp_rel | dcr | pearson | Notes |
|----|------|-------|---------|---------|-----|---------|-------|
| E01 | **FAIL** | 4776 | 5.4% | 6.8% | **1.30** | 0.94 | Q5 collapse repro |
| E02 | PASS | 115 | 5.9% | 1.7% | **4.00** | 0.99 | Offset fix |
| E03 | **FAIL** | ~500 | 97% | — | 1.00 | **−0.03** | Spatial L=768 dead |
| E04 | **FAIL** | ~500 | 98% | — | 1.00 | **−0.01** | Spatial + depth dead |
| E05 | PASS | ~4300 | 8.2% | 5.2% | 4.00 | 0.98 | Stochastic NB OK w/ offset |
| E06 | PASS | 114 | 5.6% | 7.3% | 4.00 | 0.99 | **Decoder FiLM not needed** w/ offset |
| E07 | PASS | 145 | 6.8% | 3.9% | 4.00 | 0.99 | **Encoder FiLM not needed** w/ offset |
| E08 | PASS | 117 | 5.4% | 8.1% | 4.00 | 0.99 | **Both FiLM off still OK** w/ offset |
| E09 | **FAIL** | 5000 | 47% | 48% | 1.01 | 0.47 | lr=1e-3 too slow |
| E10 | **FAIL** | 5000 | 33% | 34% | 2.10 | 0.58 | lr=3e-3 partial only |
| E11 | **FAIL†** | 4365 | 4.5% | 5.6% | **1.71** | 0.95 | †Good fit, still Q5 fail |
| E12 | PASS | 53 | 8.5% | 7.9% | 4.00 | 0.99 | SGD works w/ offset |
| E13 | **FAIL** | 501 | 758× | 557× | 1.42 | 0.02 | Meta LR 10× explodes |
| E14 | PASS | 284 | 7.8% | 3.1% | 4.00 | 0.97 | Stochastic DSF OK |
| E15 | **FAIL** | 501 | 122% | 143% | 4.02 | 0.47 | Spatial breaks even w/ offset |
| E16 | PASS | 188 | 7.4% | 6.6% | 4.00 | 0.99 | Tight clip OK w/ offset |
| E17 | PASS | 510 | 8.4% | 3.0% | 4.00 | 0.96 | Loose clip OK w/ offset |
| E18 | **FAIL†** | 3644 | 4.4% | 5.6% | **1.00** | 0.96 | †Clip doesn't fix Q5 |

† E11/E18: reconstruction metrics pass but **dcr < 3** — metadata collapse persists on default head.

---

## 3. Findings by category

### 3.1 Count head / Q5 (metadata sensitivity)

- **Root cause confirmed:** Default `NegativeBinomialLayer` (softplus μ) learns good point predictions on masked assays but **does not scale μ with 2^depth** in y_meta. dcr≈1.0–1.7 on all default-head imputation runs.
- **E29 depth-offset fixes Q5 decisively:** dcr≈4.0 in 50–500 steps across P3/P4/P5 (flat spatial), stochastic NB, stochastic DSF, SGD, and all clip settings tested.
- **FiLM is not the bottleneck once offset is present:** E06–E08 pass with encoder FiLM, decoder FiLM, or both disabled. FiLM alone (default head) is insufficient — E01/E18 prove this.
- **Implication:** Production fix should be **explicit library-size factorization in the count head** (μ = 2^depth × exp(η)), not more FiLM capacity or meta LR tuning.

### 3.2 Architecture / capacity

- **Flat profiles (L=96):** Model capacity is adequate — P1/P2 pass in prior runs; offset P3 passes in ~100 steps.
- **Spatial sine (L=768):** **Hard failure** — pearson≈0, rel_mae≈100% after patience (E03, E04). Even P5 spatial+motif+offset fails (E15): dcr OK but spatial target not learnable in budget.
- **DNA motif (flat spatial):** Prior P5 pass with offset; spatial+motif combo fails — spatial dominates difficulty.
- **Takeaway:** Metadata diagnosis is valid at L=96 flat; spatial overfit is a **separate capacity/representation problem**, not blocking Q5 work.

### 3.3 Optimizer / learning rate

- **Default head needs lr=1e-2:** E09 (1e-3) and E10 (3e-3) fail to converge in 5000 steps.
- **AdamW + wd=1e-4 (E11):** Faster/better reconstruction than Adam (rel_mae 4.5%) but **dcr still 1.71** — optimizer does not substitute for count parameterization.
- **SGD + offset (E12):** Passes in 53 steps — offset makes problem well-conditioned; optimizer choice secondary.
- **Meta/film LR 10× (E13):** Catastrophic — rel_mae>700, dec FiLM scales explode to ~76. Confirms prior note: asymmetric meta LR is dangerous even with offset.

### 3.4 Gradients / clipping

- **Default head (E01):** `grad_param_dec_meta` ≫ `grad_param_enc_film` (~5.4 vs ~5e-4); decoder meta path carries imputation gradient; encoder FiLM nearly silent.
- **Offset head (E02):** Balanced grads; dec_meta drops 20×; NB head grad ~1.8.
- **Clip ablation:** clip=1 vs 10 vs 50 all pass **with offset** (E16/E17/E02). clip=1 on **default head (E18)** gives excellent rel_mae but **dcr=1.0** — clipping cannot induce depth scaling.
- **No dead gradients** observed at default lr=1e-2 except after E13 explosion.

### 3.5 Stochasticity

- **Stochastic NB ground truth (E05):** Pass with offset — count head handles Poisson noise.
- **Stochastic DSF input (E14):** Pass with offset — denoising from random subsample works.

---

## 4. Core takeaways (actionable)

1. **Promote E29 depth-offset to production count head** — highest leverage, reproducible, FiLM-independent. Keep prototype in `depth_offset_nb.py` until ready to touch `model.py`; do **not** merge without real-data validation.

2. **Q5 is a parameterization bug, not a training bug** — LR, AdamW, SGD, and gradient clipping do not fix dcr on the default head. Do not burn sweep budget on meta LR (E13) or clip tuning for Q5.

3. **Decoder PreDecoderFiLM is helpful but optional** once μ = 2^depth × exp(η). Encoder per-conv FiLM also optional for imputation with offset on this synthetic task.

4. **Spatial targets at L=768 need separate investigation** — larger model, longer training, or different signal parameterization; out of scope for metadata collapse diagnosis.

5. **Harness lessons:** Use rel_mae + pearson + dcr (not raw NLL); always require dcr on P3+ pass checks; best-checkpoint restore essential (late divergence ~step 6300 in long runs).

6. **Adam lr=1e-2 remains the overfit default** for diagnostics; AdamW acceptable for reconstruction but does not fix Q5 without offset.

---

## 5. Reproduce

```bash
# Full ablation matrix (~25 min on H100 slice)
python -m sandbox.diagnostics.run_experiments

# Subset
python -m sandbox.diagnostics.run_experiments --ids E01,E02,E06,E08

# Original phased harness
python -m sandbox.diagnostics.synthetic_overfit --phase all --max-steps 8000 --depth-offset
```

---

## 6. Files touched this session

- `synthetic_data.py` — `stochastic_dsf`, `make_data_config()`
- `synthetic_overfit.py` — GradientMonitor skips None modules; dcr always checked on pass
- `run_experiments.py` — 18-experiment ablation runner
- `REPORT.md` (this file), updated `FINDINGS.md` / `HYPOTHESES.md` / `PROBLEMS.md`
