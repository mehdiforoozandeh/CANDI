# Real-Data chr19 Diagnostic Plan

Date: 2026-05-28  
Data: `sandbox/data/sandbox.h5`, regime `type1_chr19`, L=768, 8 assays  
Harness: `run_real_experiments.py`, `real_overfit.py`  
Probe: `depth_count_ratio` with production depths log2 22→24 (target ≈4.0)

---

## Remaining high-value work (synthetic → real)

| Priority | Question | Batch |
|----------|----------|-------|
| P0 | Does Q5 (dcr≈1) reproduce on real chr19 with default v2? | R01 |
| P0 | Does E29 depth-offset fix dcr on real chr19? | R02 |
| P1 | Can v2 overfit real counts at all (no masking)? | R03/R04 |
| P1 | Does fix hold across 8 pinned batches (not one lucky window)? | R05/R06 |
| P2 | Assay-only masking stress (pure imputation) | R07/R08 |
| P2 | FiLM ablation on real data (if offset works) | R09/R10 |
| P2 | Multi-head interference (count_peak) | R11/R12 |
| P3 | LR rescue if real overfit is slow | R13/R14 |

**Deprioritized** (synthetic already answered): LR/clip/meta-LR sweeps, spatial sine L=768.

**Open from synthetic:** E22 production depth scale mismatch — real chr19 uses native log2 depth (~22–25) automatically.

---

## Batch 1 (run first)

| ID | Description | Pass criteria |
|----|-------------|---------------|
| R01 | 1-batch, default, sandbox masking | loss ↓15%+, dcr≥3 if masked |
| R02 | 1-batch, depth-offset, masking | loss ↓15%+, dcr≥3 |
| R03 | 1-batch, no masking, default | loss ↓15%+ |
| R04 | 1-batch, no masking, offset | loss ↓15%+ |
| R05 | 8-batch cycle, default | loss ↓15%+, dcr≥3 |
| R06 | 8-batch cycle, offset | loss ↓15%+, dcr≥3 |

---

## Batch 2 (adaptive, after batch 1)

Triggered by `run_real_experiments.py --batch 2` reading `runs/real_batch1.json`:

| ID | Condition to include | Purpose |
|----|---------------------|---------|
| R07/R08 | always | Assay-only mask stress, default vs offset |
| R09/R10 | R02 dcr≥3 | FiLM ablation on real data |
| R11/R12 | R01 dcr<2 and R02 dcr≥3 | Multi-head vs count-only |
| R13 | R01 and R03 both fail loss drop | LR=3e-3 rescue |
| R14 | R02 dcr≥3 | Faster offset convergence check |
| R15–R17 | R02 dcr<2 | **depth_center=24** scale fix |

---

## Results summary

**Batch 1:** R01 confirms Q5 on real data (dcr=1.001). R02 raw offset fails (dcr=1.0). R03/R04 reconstruct OK.

**Batch 2:** R15–R17 **PASS** with `depth_center=24` (dcr≈4.0–4.2, imp_p≈0.99). Raw offset still fails (R08).

**Batch 3 (if time):** R11/R12 count_peak; R05/R06 redo with depth_center; short sandbox.train smoke.

---

## Batch 3 results (done)

| ID | dcr | Notes |
|----|-----|-------|
| R18 | 4.01 | multi-batch overfit unstable; dcr still OK |
| R19 / R19b | 0.996 / 4.02 | count_peak needs centered offset |
| R20 | 1.00 / 4.00 | multi-epoch iterable train smoke |

**Ready for sandbox E29 staging** with `depth_center≈24`.

## Commands

```bash
python -m sandbox.diagnostics.run_real_experiments --batch 1
python -m sandbox.diagnostics.run_real_experiments --batch 2
```
