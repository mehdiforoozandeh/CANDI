# E32 / Autoresearch May 31 — Imp count R² vs correlation disparity

Status: **done** (autoresearch loop complete, 2026-06-02) — **Partial** vs validate gate  
Experiment ID: **E32**  
META: **Q11** — Imp count calibration: Pearson ~0.5 but R² ≈ 0  
Implementation directory: **`sandbox/autoresearch/may31/`** (create from this doc)  
Prior art: [`autoresearch_may28_count_head.md`](autoresearch_may28_count_head.md) (diagnostics harness; different objective), [`idea_e31_depth_center_sweep.md`](idea_e31_depth_center_sweep.md) (observed disparity), [Karpathy autoresearch](https://github.com/karpathy/autoresearch)

---

## Executive summary

CANDI v2 depth-offset runs (E31) show **moderate imp count correlation** (`imp_count_pearson_gw` / `imp_count_spearman_gw` ~0.45–0.51) but **near-zero or negative imp count R²** (`imp_count_r2_gw` ≤ 0; per-assay median R² often ≈ −0.02 to −0.09). The model learns **which loci are high vs low** (rank) but not **absolute count magnitudes** (calibration). Denoising on the same forward pass achieves **`den_count_r2_gw` ~0.2–0.4**, so the gap is not purely “metrics are broken” — it signals a real **rank–magnitude decoupling**, worst on imputation eval.

**E32** is a Karpathy-style autoresearch loop whose **primary objective is a gated score**: first learn identity denoising (`den_count_r2_gw ≥ 0.35` at dsf=1) and metadata sensitivity (`depth_count_ratio ≈ 4`), then maximize **`imp_count_r2_gw`** on chr21 zero-shot imp eval (vb_natural metadata). See **Primary metric** section.

**Hard constraint:** All harness code lives under `sandbox/autoresearch/may31/`. **Zero edits outside that directory** during the loop. Import `sandbox.candi_v2.*`, `sandbox.eval`, `sandbox.batch`, etc. read-only; fork/wrap/subclass inside the tag dir when behavior must change.

---

## Problem statement

### Observed (E31, epoch ~150–175, count_only + depth_offset)

| Metric | Typical range | Interpretation |
|--------|---------------|----------------|
| `imp_count_pearson_gw` | 0.40 – 0.48 | Linear rank association on imp positions |
| `imp_count_spearman_gw` | 0.45 – 0.51 | Monotonic rank (often ≥ Pearson) |
| `imp_count_r2_gw` | −3 … −0.02 (batch mean often negative) | Pointwise NB-mean vs raw counts |
| `eval_metrics_median/imp_count_r2` | ≈ −0.10 … +0.01 | Per-assay median; closer to “~0” |
| `den_count_r2_gw` | 0.13 – 0.44 | Same model, same eval pass, observed positions |
| `training_metadata_probes/depth_count_ratio` | ~4.0 (c≠0 runs) | Depth offset works when probed with **real T y_meta** |

Example (`e31_c27` ep149): imp_p=0.476, imp_r2=−0.739, den_r2=0.411, dcr=4.006.

### Metric definitions (authoritative: `sandbox/eval.py`)

- **Pearson / Spearman:** correlation after centering (Pearson) or ranks (Spearman) on masked `(nb_mean, y_target)` pairs.
- **R²:** `1 − SS_res/SS_tot` where `SS_res = Σ(pred − target)²`, `SS_tot = Σ(target − mean_target)²`. **Not** equal to `r²` unless predictions lie on the OLS line.
- **NB mean:** `nb_mean = n(1−p)/p` from decoder `(p, n)`.
- **Imp eval mask:** `imp_eval_map = (y_avail_T == 0) & (y_pval_imp valid)` — assays **missing in T** but present in paired V/B ground truth.
- **Imp count mask:** `imp_gw_mask & (y_data_imp != −1)`.

---

## Biosample pairing (T / V / B) — intended design vs bugs to audit

### Intended (user requirement)

For cell line **DND-41** (example):

- **`T_DND-41`** — training biosample; sparse assay panel used as input/target during training.
- **`V_DND-41` / `B_DND-41`** — **same underlying biosample**, different assay subsets held out for **imputation eval** (validation/test tracks).

Pairing is **not cross-cell-line**. It is **same biosample, different assay availability** — zero-shot **assay** imputation, not cross-sample transfer.

Code intent (`sandbox/data.py`):

```python
# T_DND-41 → imps = [V_DND-41, B_DND-41] via _all_imp_biosamples
base = t_bios[2:]  # strip "T_"
return [pref + base for pref in ("V_", "B_") if pref + base in bios_order]
```

Eval batches load `(T_bios, V_or_B_bios)` pairs; `y_data_imp` comes from the **imp biosample** at the **same loci/windows**.

### Mandatory audit in `prepare.py` (first implementation task)

On harness init, **assert and log**:

1. Every eval batch has `biosample_name.startswith("T_")` and `imp_biosample_name` in `{V_, B_}<same_base>`.
2. Parse base names: `assert imp_name[2:] == t_name[2:]`.
3. Log assay-level table: for each imp eval position, which assays have `y_avail_T==0` and valid `y_data_imp`.

If any pair violates same-base rule → **raise loudly** (silent bug).

---

## Eval metadata prompt — likely bug (must fix in harness baseline)

### Current production eval (`sandbox/train.py::run_eval_pass`)

For **`y_avail=0`** (truly missing in T) slots:

```python
y_meta_fwd[miss_exp] = canonical_meta  # EIC median depth/readlen/run_type per assay TYPE
```

For **cloze-masked** slots: `_build_mixed_meta(T, y_meta_imp, masked_map)` — but eval uses **`apply_mask=False`**, so `masked_map` is all False → **V/B metadata is never injected** during standard eval.

### Why this is wrong for imp R²

Depth-offset head (`sandbox/candi_v2/decoder.py`):

```
log2_mu = (d - depth_center) + eta
mu = 2^log2_mu
```

If **`d` = canonical median depth** but **ground-truth counts** come from **V/B at V/B’s actual sequencing depth**, μ is systematically mis-scaled → **R² collapses** while **η** can still encode rank → Pearson ~0.5.

Example from `sandbox/data/sandbox_log2_depths.csv`: B_RWPE2 ATAC log2 depth ≈ 27.5; canonical EIC median for ATAC may differ by several log2 units (4×–16× count scale).

### Required fix in `prepare.py` eval (baseline, not optional)

For imp eval positions (`y_avail_T == 0`), set **`y_meta_fwd` from `y_meta_imp`** (V/B natural metadata loaded at `meta_dsf1` in `sandbox/data.py`), **not** `build_canonical_meta`.

Pseudocode:

```python
missing_in_T = (y_avail == 0)  # [B, F]
# For each (b, f) where missing_in_T and y_meta_imp valid:
y_meta_fwd[b, :, f] = y_meta_imp[b, :, f]
# Only fall back to canonical_meta when V/B metadata absent (-1)
```

Keep canonical fallback for assays with no V/B meta row.

**Eval metadata policy is fixed in `prepare.py` (not agent-tunable).** Always `vb_natural` for `primary_score`. Canonical and cloze-T variants are **diagnostics only** (logged every run in footer; row **0b** optional human-only script before agent loop to quantify A1 delta).

Document three reported metrics every run:

| Key | Meaning |
|-----|---------|
| `imp_count_r2_gw` | Primary — V/B GT, corrected metadata |
| `imp_count_r2_gw_canonical` | Same preds, eval with old canonical prompt (diagnostic) |
| `imp_count_r2_gw_cloze_T` | Cloze-on-T eval (train-aligned task) |

---

## Train vs eval task alignment

| | **Training `count_imp`** | **Eval `imp_count_*` (cornerstone)** |
|--|--------------------------|--------------------------------------|
| Assays | Cloze-masked among **T-available** | **T-unavailable**, V/B-available |
| Target | **T `y_data`** (same biosample) | **`y_data_imp`** (V/B biosample, same cell line) |
| Input | CLOZE on masked tracks | MISSING (−1) on absent tracks |
| `y_meta` at loss | **True T metadata** (unmasked in `prep["y_meta"]`) | Should be **V/B metadata** (after fix) |

Rank skills may transfer; **absolute calibration** requires correct **depth/readlen/runtype prompt** at inference.

**Autoresearch should optimize `imp_count_r2_gw` (V/B eval)** while logging **`imp_count_r2_gw_cloze_T`** so we can see how much gap is task vs prompt vs objective.

---

## Hypothesis axes (FAFO search space)

Sorted by priority for the autoresearch agent. Each axis should be toggled **one at a time** per Karpathy loop iteration unless `program.md` authorizes a structured sweep.

### Tier A — Eval / prompt (human implements in `prepare.py` before agent loop)

| ID | Hypothesis | Where | Expected direction if true |
|----|------------|-------|----------------------------|
| A1 | Canonical vs V/B metadata causes R² gap | `prepare.py` vb_natural fix (fixed) | row 0b: `imp_r2_vb` ≫ `imp_r2_canonical` |
| A2 | Train/eval task gap (cloze vs missing) limits R² | diagnostic metrics in footer | cloze_T r2 ≫ vb r2 → task gap |
| A3 | T/V/B pairing bug | audit asserts in `prepare.py` | fix → discontinuous r2 jump |

### Tier B — Loss / calibration objective

| ID | Hypothesis | Tunable in `train.py` | Notes |
|----|------------|----------------------|-------|
| B1 | NB NLL alone does not penalize mean calibration | add `lambda_mse * MSE(nb_mean, y)` on masked | start λ ∈ {0.1, 0.5, 1.0} |
| B2 | NB NLL on log1p(counts) vs raw | transform in loss only | may align with encoder log1p |
| B3 | Direct MAE on log2 counts | `MAE(log2(nb_mean+1), log2(y+1))` | stable for heavy tails |
| B4 | Higher `imp_weight` / lower `obs_weight` | `imp_weight`, `obs_weight` | autoresearch May28: imp=8, obs=0.5 helped Pearson |
| B5 | Heteroscedastic precision on imp branch | optional per-branch weight | |

### Tier C — Data / DSF

| ID | Hypothesis | Tunable in `train.py` | Notes |
|----|------------|----------------------|-------|
| C1 | DSF train uniform {1,2,4,8} vs eval dsf1 hurts calibration | `dsf_sampling` off vs uniform | den r2 may move too |
| C2 | Train with `dsf_sampling=off` matches eval | fixed dsf1 | simpler identity-like denoising |

### Tier D — Architecture (wrap in tag dir only)

| ID | Hypothesis | Implementation sketch |
|----|------------|----------------------|
| D1 | log1p encoder + raw count head mismatch | `encoder.signal_transform` override via config clone |
| D2 | Non-groupwise decoder hurts assay-specific scale | optional **groupwise deconv** wrapper (copy `DeconvTower` per assay group) — larger VRAM |
| D3 | Depth-offset fallback `μ=exp(η)` untrained for bad depth | train-time random depth dropout on y_meta row 0 |
| D4 | Pre-decoder FiLM insufficient for per-assay scale | per-assay affine calibration head on nb_mean (tag-dir only): `mean' = a_f * mean + b_f` |

### Tier E — Broader ceiling (den R² ~0.4)

| ID | Hypothesis | Action |
|----|------------|--------|
| E1 | Even denoising is miscalibrated — not imp-specific | track `den_count_r2_gw`; if B1–B3 lift imp but not den, different root cause |
| E2 | 5000 steps / pinned subset may plateau | **fixed** in prepare for comparability; note in synthesis if score plateaus |
| E3 | chr19 train vs chr21 eval windows | log eval regime; optional `eval_chr=chr19` ablation in train.py via eval helper flag |

---

## Locked decisions (grill rounds 1–3, 2026-05-31)

Decisions from MCQ grill before implementation. **Do not change without explicit human re-spec.**

| Topic | Decision |
|-------|----------|
| **Model shell** | Tiny sandbox CANDI v2 matching E31/May28: L=768, 8 assays, 2 transformer layers, `count_only` + `depth_offset` (~0.3M params). Use `build_real_v2_config()` pattern from `sandbox/diagnostics/real_data.py`. |
| **Training budget** | **5000 optimizer steps** (fixed in `prepare.py`). Timeout = **2× baseline wall time** (Karpathy rule). Expect ~12–17 min/run after baseline. |
| **Train data pin** | Cache **32 fixed batches** (batch_size=4, chr19 `type1_chr19`). Each batch = one `(T_biosample, chr19_window_idx)` pair. Selection: cover all **5 T biosamples** (≥6 batches each) + spread window indices along chr19 (quartile stratification). Cycle batches round-robin over 5000 steps. Write manifest JSON at harness init. **No explicit fg/bg labels** — diversity via biosample + genomic spread only. |
| **Eval data pin** | **8 fixed chr21 eval batches** (batch_size=4). Fixed window indices (spread along chr21). **Rotate T/V/B pairs** across batches: DND-41, RWPE2, heart_left_ventricle, H1-hESC (skip H9 — no V/B). **Eval protocol:** T biosample = input (missing assays = −1, **no cloze mask**); metrics scored vs **V/B `y_data_imp`** at same loci; **`y_meta_fwd` = V/B natural metadata** (`vb_natural`, fixed). |
| **Eval primary** | **Gated primary** (see metric section below): identity-first via den gate, then imp R². |
| **Guard-rails (keep)** | Ratchet on `primary_score` + hard guards: `dcr ∈ [3.25, 4.75]` always; when `metric_phase=imp`: also `den ≥ 0.35`, `imp_pearson ≥ max(baseline−0.05, 0.38)`; when `metric_phase=den`: pearson guard **skipped**; always `peak_vram_ok`, `status=ok`. |
| **`prepare.py` edits** | **Strict Karpathy: 100% frozen after human baseline.** Agent edits **`train.py` only.** |
| **Baseline init** | Random init. **obs_weight=1.0, imp_weight=1.0**. Optimizer: adamax 1e-3, clip_norm=0.5. **dsf_sampling=uniform** {1,2,4,8}. |
| **Agent scope** | **Tier B + C + D** in `train.py`: loss weights, calibration aux losses, DSF, optimizer, `depth_center`, signal_transform, depth dropout, tag-dir architecture wrappers (D2 groupwise deconv, D4 per-assay affine). **VRAM hard cap 9500 MB** — OOM = discard. |
| **Loss implementation** | Agent adds B1–B3 via **`train_step()` hook** in `train.py`; `prepare.py` calls hook inside fixed loop. |
| **`depth_center`** | **Agent-tunable** via `TrainConfig`; `prepare.py` reads value when building model. Default 27.0. |
| **Success (synthesis)** | **Validate:** `metric_phase=imp`, `imp_count_r2_gw > 0.15`, `den ≥ 0.35`, DCR in band, `imp_pearson ≥ 0.40`. **Partial:** vb fix → imp R² ∈ (0, 0.15]. |
| **A1 validation** | Row 0 baseline + optional row **0b** human script comparing `imp_r2_canonical` vs `imp_r2_vb` before agent loop. Every run footer logs both as diagnostics. |

---

## Primary metric and composite score

Design goal: **imp_count_r2_gw** is the scientific target, but the model must first learn a usable **identity map** (high den R² at dsf=1) and **depth metadata sensitivity** (DCR≈4) before imp calibration wins count.

### Raw eval metrics (all logged every run)

| Key | Eval protocol | Role |
|-----|---------------|------|
| `imp_count_r2_gw` | 8-batch chr21 zero-shot imp, vb_natural meta | Core scientific target |
| `den_count_r2_gw` | chr21 (or pinned den batches), **dsf=1**, uncorrupted T input, observed positions | Identity / denoising calibration |
| `depth_count_ratio` | `prompt_sensitivity_depth_count_ratio` on den batch, real T `y_meta`, depth 22→24 log2 | Metadata sensitivity (target ≈ 4.0) |
| `imp_count_pearson_gw` | same imp eval | Rank guard-rail |
| `imp_count_r2_gw_cloze_T` | cloze-on-T diagnostic | Task-gap diagnostic |
| `imp_count_r2_gw_canonical` | imp eval with canonical meta | A1 diagnostic |
| `dcr_masked_bins` | optional cloze-T masked-bin probe (May28 style) | diagnostic only |

### Gated primary (maximize `primary_score`)

Constants (fixed in `prepare.py`):

```
DEN_GATE         = 0.35
DCR_LO, DCR_HI   = 3.25, 4.75
IMP_PHASE_BIAS   = 1.0    # ratchet continuity (see below)
```

Phase detection:

```
in_imp_phase = (den_count_r2_gw >= DEN_GATE) and (DCR_LO <= depth_count_ratio <= DCR_HI)
```

Primary score:

```
if in_imp_phase:
    primary_score = imp_count_r2_gw + IMP_PHASE_BIAS
else:
    primary_score = den_count_r2_gw - DEN_GATE
```

**Behavior:**

- **Den phase** (`metric_phase=den`): agent improves `primary_score` by raising den R² toward ≥0.35 while keeping DCR in band. Score range ≈ [−0.35, +0.65].
- **Imp phase** (`metric_phase=imp`): once den and DCR pass, primary tracks **imp_count_r2_gw** (shifted by +1 so imp-phase commits ratchet above any den-only win — avoids discontinuity discard when crossing the gate).

`IMP_PHASE_BIAS` is not a “bonus”; it is a fixed offset so `git keep` comparisons work across the phase boundary. Report unpooled `imp_count_r2_gw` in synthesis tables.

### Keep rule (must pass ALL)

```
primary_score > best_primary_score          # ratchet (maximize)
depth_count_ratio in [3.25, 4.75]           # hard DCR band — always
peak_vram_ok == True
status == ok
```

When `metric_phase=imp` (all required):

```
den_count_r2_gw >= DEN_GATE
imp_count_pearson_gw >= max(baseline_pearson - 0.05, 0.38)
```

When `metric_phase=den`: **do not** apply pearson or den≥gate checks (den is what you are optimizing). DCR band still applies — depth_offset head must stay metadata-sensitive even during den phase.

### Tie-break (results.tsv / human review)

When `primary_score` ties within 1e-4: prefer higher `imp_count_r2_gw`, then higher `den_count_r2_gw`, then `|depth_count_ratio - 4|` closer to 0.

### Success criteria (synthesis, unchanged)

| Outcome | Verifiable |
|---------|------------|
| **Validate** | `metric_phase=imp` sustained with `imp_count_r2_gw > 0.15`, `den_count_r2_gw ≥ 0.35`, `depth_count_ratio ∈ [3.25, 4.75]`, `imp_pearson ≥ 0.40` |
| **Partial** | vb_natural fix alone → imp R² ∈ (0, 0.15] with gates passing |
| **Disvalidate** | Cannot reach imp phase, or imp R² ≤ 0.05 after gate with 20+ agent commits |

### Parseable run footer (required)

```
---
primary_score:              1.042000
metric_phase:               imp
imp_count_r2_gw:            0.042000
den_count_r2_gw:            0.385000
depth_count_ratio:          4.010000
imp_count_pearson_gw:       0.468000
imp_count_spearman_gw:      0.512000
imp_count_r2_gw_cloze_T:    0.310000
imp_count_r2_gw_canonical: -0.650000
dcr_masked_bins:            4.068000
count_imp_loss:             1.520000
count_obs_loss:             0.680000
peak_vram_mb:               2100.000000
peak_vram_ok:               True
status:                     ok
---
```

Note: when `metric_phase=den`, `primary_score = den_count_r2_gw - 0.35` (no +1 bias).

---

## Harness specification (`sandbox/autoresearch/may31/`)

### Files to create

| File | Role |
|------|------|
| `prepare.py` | **Fixed (human baseline only).** 32-batch train pin, 8-batch chr21 eval pin, model build, vb_natural eval, footer, VRAM/timeout guards |
| `train.py` | **Agent-editable.** `TrainConfig`, `train_step()` hook, optional tag-dir architecture wrappers |
| `pin_manifest.json` | **Commit once.** 32 train + 8 eval batch specs (see Implementation notes) |
| `eval_pass.py` | **Fixed.** Forked eval logic (vb_natural, canonical, cloze-T); imported by prepare |
| `validate_a1.py` | **Optional human pre-loop.** Row 0b canonical vs vb_natural comparison |
| `program.md` | Agent playbook — loop, axes, priors, keep rule |
| `scope.py` | Fail if git diff outside `sandbox/autoresearch/may31/` |
| `agent_step.py` | Run once + append `results.tsv` |
| `loop.sh` | Optional SLURM 4h driver |
| `__init__.py` | Package marker for `python -m sandbox.autoresearch.may31.train` |
| `.gitignore` | `results.tsv`, `run.log`, `*.pt` |
| `README.md` | Human quick start |

### Fixed in `prepare.py` (do not edit during loop)

| Constant | Value | Rationale |
|----------|-------|-----------|
| Model shell | Tiny E31 stack via `build_real_v2_config(heads="count_only")` | L=768, 8 assays, 2 layers; same stack as E31 disparity |
| `decoder.count_head` | `depth_offset` | E30/E31 validated; do not disable without strong reason |
| `decoder.depth_center` | Read from `TrainConfig` at build time | default 27.0; agent-tunable in `train.py` |
| Data | `sandbox/data/sandbox.h5` | |
| Train regime | chr19 `type1_chr19` | match E31 |
| Train pin | **32 cached batches**, bs=4, cycle round-robin | biosample + chr19 window diversity |
| Eval pin | **8 cached chr21 batches**, bs=4, fixed indices | fast held-out chr; rotate T/V/B pairs |
| Eval metadata | **`vb_natural` only** for primary | fixed; not in `TrainConfig` |
| `max_steps` | **5000** | fixed comparison budget |
| Timeout | **2× baseline wall time** | kill run → `status=crash` |
| Masking (train) | `p_full_assay=1`, `p_full_loci=0`, `p_chunks=0` | assay-only cloze on T |
| Masking (eval imp) | **No cloze** — missing = −1 on T input | zero-shot assay imputation task |
| VRAM cap | 9500 MB | 10 GB H100 slice; Tier D must fit |

### Imports (read-only from repo)

```python
from sandbox.candi_v2.model import CANDIv2
from sandbox.candi_v2.config import CANDIv2Config
from sandbox.eval import eval_batch_metrics, pearson_r, r2_score  # copy helpers if eval must fork
from sandbox.batch import prepare_masked_batch, make_masker
from sandbox.data import SandboxH5Dataset
from sandbox.losses import SandboxCompositeLoss
from sandbox.candi_v2.loss import build_v2_loss
```

If eval metadata policy cannot be injected without editing `sandbox/train.py`, **copy** `run_eval_pass` logic into `prepare.py` inside the tag dir with the V/B metadata fix.

### Tunable in `train.py` (agent edits)

Baseline row 0 defaults (neutral loss weights; agent tunes from here):

```python
@dataclass
class TrainConfig:
    # Optimizer
    optimizer: str = "adamax"
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_norm: float = 0.5

    # Loss weights (CANDI_LOSS) — baseline both 1.0; May28 winner was imp=8, obs=0.5
    count_weight: float = 1.0
    obs_weight: float = 1.0
    imp_weight: float = 1.0

    # Count head (agent-tunable)
    depth_center: float = 27.0

    # Calibration extras (agent may add; applied in train_step())
    lambda_mse_imp: float = 0.0      # B1
    lambda_mse_obs: float = 0.0
    mse_on_log1p: bool = False       # B2

    # Data (C)
    dsf_sampling: str = "uniform"    # uniform | off

    # Encoder transform ablation (D1)
    signal_transform: str = "log1p"  # log1p | none | arcsinh

    # D3 depth dropout
    y_meta_depth_dropout_p: float = 0.0


def train_step(
    model, batch, prep, base_loss_fn, cfg: TrainConfig
) -> tuple[torch.Tensor, dict]:
    """Agent-editable forward + loss. prepare.py calls this each step."""
    ...
```

Agent may add `TrainConfig` fields and edit `train_step()` / architecture helpers. **`eval_meta_policy` is not here** — fixed in `prepare.py`.

### Eval suite in `prepare.py`

After 5000 training steps, run on **pinned 8-batch chr21 eval set**:

1. **Zero-shot imp (primary):** T biosample input (missing assays = −1, **no cloze**); targets = V/B `y_data_imp`; **`y_meta_fwd` = V/B natural** (`vb_natural`, fixed). Pool → `imp_count_r2_gw`, `imp_count_pearson_gw`.
2. **Canonical imp (diagnostic):** same preds, re-score with canonical median metadata → `imp_count_r2_gw_canonical`.
3. **Cloze-T imp (diagnostic):** assay masker on T, target T `y_data`, T metadata → `imp_count_r2_gw_cloze_T`.
4. **Den (identity):** dsf=1, uncorrupted input, observed positions on T → `den_count_r2_gw`. Same batch used for `depth_count_ratio` probe (real T `y_meta`, depth 22→24).
5. **Depth probe:** `prompt_sensitivity_depth_count_ratio` — hard band [3.25, 4.75] for keep rule.
6. **Optional:** `dcr_masked_bins` on cloze-T batch (diagnostic, May28-style).

**T/V/B audit** runs on every eval batch (raise if same-base pairing violated). Compute metrics via same formulas as `sandbox/eval.py` (fork `run_eval_pass` into tag dir if needed).

### `results.tsv` schema

```
commit	primary_score	metric_phase	imp_r2	den_r2	dcr	imp_pearson	vram_mb	vram_ok	status	description
```

Tab-separated. Gitignored.

### `scope.py`

```python
ALLOWED_PREFIX = "sandbox/autoresearch/may31/"
# Fail if any staged/changed file path does not start with ALLOWED_PREFIX
```

### Branch naming

```
autoresearch/may31
```

One commit per experiment; reset on discard.

---

## Implementation notes (next agent — follow exactly)

Reference harness to **copy structure from** (do not import at runtime): `sandbox/diagnostics/autoresearch/`. May28 differs in metric and eval; use it for file layout, `run_experiment()` flow, `agent_step.py`, `loop.sh`.

### Run commands and environment

From repo root on a compute node with GPU:

```bash
cd ~/projects/def-maxwl/mforooz/EpiDenoise
conda activate candi && source candi_venv/bin/activate && module load samtools
export PYTHONPATH=$PWD

# One experiment (baseline row 0):
python -m sandbox.autoresearch.may31.train 2>&1 | tee sandbox/autoresearch/may31/run.log

# Scope check before every loop commit:
python -m sandbox.autoresearch.may31.scope
git add sandbox/autoresearch/may31/train.py
git commit -m "autoresearch/may31: <description>"

# Agent loop step (parse footer → append results.tsv):
python -m sandbox.autoresearch.may31.agent_step --description "..."

# Optional A1 ablation (before agent loop):
python -m sandbox.autoresearch.may31.validate_a1
```

**SLURM interactive** (4 h max):

```bash
srun --account=def-maxwl --cpus-per-task=2 \
  --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --mem=14G --time=4:00:00 --pty bash
```

**Branch:** `git checkout -b autoresearch/may31` before row 0.

**Entrypoint pattern:** `train.py` imports `prepare.run_from_train()` in `main()` (same as May28). Agent never edits `prepare.py`.

### Model build (required overrides)

`build_real_v2_config(heads="count_only")` alone is **not sufficient** — dataclass default is `count_head="plain"`. After merge, **must set**:

```python
cfg = build_real_v2_config(heads="count_only", lr=tc.lr, clip_norm=tc.clip_norm)
cfg.decoder.count_head = "depth_offset"
cfg.decoder.depth_center = float(tc.depth_center)
cfg.encoder.signal_transform = tc.signal_transform  # D1
# loss weights read from TrainConfig when building build_v2_loss(cfg) — override cfg.training.loss_weights from tc
```

Then `model = CANDIv2(cfg).to(device)` and `loss_fn = build_v2_loss(cfg)`.

### `pin_manifest.json` (commit once)

- **Path:** `sandbox/autoresearch/may31/pin_manifest.json`
- **Git:** **commit after first successful generation** (reproducibility). Not gitignored.
- **Regenerate only** if `sandbox/data/sandbox.h5` or selection changes (human decision).

Schema:

```json
{
  "seed": 42,
  "h5_path": "sandbox/data/sandbox.h5",
  "train": [
    {"t_bios": "T_DND-41", "window_indices": [120, 121, 122, 123]},
    ...
  ],
  "eval_imp": [
    {"t_bios": "T_DND-41", "imp_bios": "V_DND-41", "window_indices": [0, 1, 2, 3]},
    ...
  ],
  "eval_cloze_t_index": 0
}
```

32 entries in `train`; 8 in `eval_imp`. Each `window_indices` has length 4 (= `batch_size`). Indices are into the global `windows` table in H5 (same indexing as `SandboxH5Dataset._windows`).

#### Algorithm: `build_pin_manifest()` (run once at prepare init if missing)

```python
SEED = 42
N_TRAIN = 32
N_EVAL = 8
BS = 4

# 1) Collect window index lists from H5 (mirror SandboxH5Dataset._filter_window_indices)
chr19_train_wi = [i for i, w in enumerate(windows) if w[0] == "chr19"]  # type1 train pool
chr21_eval_wi  = [i for i, w in enumerate(windows) if w[0] == "chr21"]

# 2) Quartile pick helper: split sorted indices into 4 bins, round-robin pick
def spread_pick(idxs: list[int], n: int, *, offset: int = 0) -> list[int]:
    """Return n window indices spread across quartiles of idxs (deterministic)."""
    if not idxs:
        raise ValueError("empty index pool")
    s = sorted(idxs)
    q = max(1, len(s) // 4)
    quartiles = [s[i * q : (i + 1) * q if i < 3 else len(s)] for i in range(4)]
    out: list[int] = []
    qi, pi = offset % 4, 0
    while len(out) < n:
        pool = quartiles[qi]
        if pool:
            out.append(pool[pi % len(pool)])
            pi += 1
        qi = (qi + 1) % 4
        if pi > len(s) * 2:
            break
    return out[:n]

# 3) Train entries: 6 batches × each T biosample + 2 extra
T_BIOS = ["T_DND-41", "T_RWPE2", "T_heart_left_ventricle", "T_H1-hESC", "T_H9"]
for t in T_BIOS:
    for k in range(6):
        wi = spread_pick(chr19_train_wi, 1)[0]  # vary k to get different quartile offsets
        train.append({"t_bios": t, "window_indices": spread_pick(chr19_train_wi, BS)})
# +2 extras on biosamples with V/B imp pairs:
train.append({"t_bios": "T_DND-41", "window_indices": spread_pick(chr19_train_wi, BS)})
train.append({"t_bios": "T_RWPE2", "window_indices": spread_pick(chr19_train_wi, BS)})

# 4) Eval imp entries (fixed T/V/B rotation — skip H9, no V/B)
EVAL_PAIRS = [
    ("T_DND-41", "V_DND-41"),
    ("T_DND-41", "B_DND-41"),
    ("T_RWPE2", "B_RWPE2"),
    ("T_heart_left_ventricle", "V_heart_left_ventricle"),
    ("T_H1-hESC", "V_H1-hESC"),
    ("T_DND-41", "B_DND-41"),              # 2nd window quartile
    ("T_RWPE2", "B_RWPE2"),
    ("T_heart_left_ventricle", "V_heart_left_ventricle"),
]
for i, (t, imp) in enumerate(EVAL_PAIRS):
    eval_imp.append({
        "t_bios": t, "imp_bios": imp,
        "window_indices": spread_pick(chr21_eval_wi, BS, offset=i),
    })

# 5) eval_cloze_t_index = 0  (first eval batch used for cloze-T diagnostic)
```

Implement `spread_pick` deterministically from `SEED` so manifest is stable.

#### `load_pinned_batch(entry, *, dsf_sampling, train)` 

Do **not** rely on `SandboxH5Dataset` iterator for pinned loads. Copy tensor assembly from `sandbox/data.py` `__iter__` (lines ~259–352):

- Read `t_bios` from H5 `biosamples/{gname}` for T track.
- If `imp_bios` in entry: attach `y_data_imp`, `y_pval_imp`, `y_peaks_imp`, `y_meta_imp`, `imp_biosample_name` exactly as dataset does when `eval_include_vb_ground_truth=True`.
- **Train:** `dsf_sampling` from `TrainConfig` (`uniform` or `off`).
- **Eval (den + imp):** force `dsf_list=(1,)`, `dsf_sampling="off"` so all assays at dsf=1 (identity eval).

Cache all 32+8 loaded batches in RAM at `prepare.py` init (one-time cost).

### Eval suite implementation (`evaluate_suite()` in prepare.py)

Implement as **`eval_pass.py`** inside tag dir (imported by prepare) — fork logic from `sandbox/train.py::run_eval_pass` (lines ~520–620) and `sandbox/eval.py::update_assay_gw_pools`.

**Per pinned eval batch** (`eval_imp` manifest entry):

1. **Audit:** assert `t_bios.startswith("T_")`, `imp_bios.startswith(("V_","B_"))`, `t_bios[2:]==imp_bios[2:]`. Log imp assay counts (`y_avail==0` & valid `y_data_imp`).
2. **Zero-shot prep:** `prepare_masked_batch(batch, eval_masker, device, apply_mask=False)` — missing assays stay −1 on input.
3. **Build `y_meta_fwd` (vb_natural — primary):**
   ```python
   y_meta_fwd = prep["y_meta"].clone()
   y_meta_imp = batch["y_meta_imp"].to(device)  # [B,4,F]
   missing = (batch["y_avail"].to(device) == 0).unsqueeze(1).expand_as(y_meta_fwd)
   y_meta_fwd[missing] = y_meta_imp[missing]  # only where V/B meta valid (row0 != -1)
   # fallback: where y_meta_imp invalid, use build_canonical_meta("data/eic_metadata.csv", SANDBOX_ASSAYS)
   query_mask_fwd = prep["query_mask"] | missing
   ```
4. **Forward** → `p, n, mu, ...`
5. **imp_eval_map:** `(y_avail==0) & (y_pval_imp != -1)`; count mask adds `y_data_imp != -1`.
6. **Per-batch metrics:** `eval_batch_metrics(...)` → append to `metric_agg` (same as `sandbox/train.py::run_eval_pass`).
7. **Also** `update_assay_gw_pools(...)` for median diagnostics (`finalize_eval_metrics_median_gw` → log `eval_metrics_median/*` in footer if desired).

**After all 8 batches:**

- **Primary gw metrics:** mean across batches for each key in `metric_agg` (production pattern):
  `imp_count_r2_gw = mean(batch["imp_count_r2_gw"])`, same for `den_count_r2_gw`, `imp_count_pearson_gw`, etc.
- **Median diagnostics (optional footer keys):** `finalize_eval_metrics_median_gw(assay_gw_pools)`.

**Canonical diagnostic (second forward per batch):** same batch, same input; rebuild `y_meta_fwd` with `build_canonical_meta` on `y_avail==0` slots (mirror production bug). Forward again → pool into separate `pools_canonical` → `imp_count_r2_gw_canonical`.

**Cloze-T diagnostic:** use **`eval_imp[eval_cloze_t_index]`** batch only. `prepare_masked_batch(..., imp_masker, apply_mask=True)` with `p_full_assay=1`; targets = T `y_data`; metadata = T `y_meta`. Forward → imp metrics on `masked_map` (not imp_eval_map) → `imp_count_r2_gw_cloze_T`.

**Den + DCR:** pool den metrics across **all 8 eval batches** (same forwards as step 4 — reuse `y_meta_fwd` vb_natural, observed_map). `den_count_r2_gw` from observed positions. **DCR:** run `prompt_sensitivity_depth_count_ratio(model, prep, prep["y_meta"], device)` on **eval batch 0** (real T metadata, depth 22→24). Single scalar for keep rule.

**Optional `dcr_masked_bins`:** run `run_probe_battery` from `sandbox/diagnostics/meta_probes.py` on cloze-T prep (May28 pattern); diagnostic only.

### `baseline.json` (written after row 0)

Path: `sandbox/autoresearch/may31/baseline.json` (gitignored). Written when row 0 completes successfully.

```json
{
  "peak_vram_mb": 2100.0,
  "training_seconds": 890.5,
  "timeout_seconds": 1781.0,
  "primary_score": -0.08,
  "metric_phase": "den",
  "imp_count_r2_gw": -0.12,
  "den_count_r2_gw": 0.27,
  "depth_count_ratio": 3.95,
  "imp_count_pearson_gw": 0.22
}
```

`timeout_seconds = training_seconds * 2` from row 0. Kill runs exceeding this → `status=crash`.

### `agent_step.py`

Copy May28 pattern; changes:

- Module: `sandbox.autoresearch.may31.train`
- TSV header: `commit\tprimary_score\tmetric_phase\timp_r2\tden_r2\tdcr\timp_pearson\tvram_mb\tvram_ok\tstatus\tdescription`
- Parse footer keys: `primary_score`, `metric_phase`, `imp_count_r2_gw`, `den_count_r2_gw`, `depth_count_ratio`, `imp_count_pearson_gw`, `peak_vram_mb`, `peak_vram_ok`, `status`
- Keep/discard logic lives in **agent** (program.md); `agent_step.py` only runs + logs

### `validate_a1.py` (optional row 0b)

Runs one untrained or row-0 checkpoint eval; prints:

```
imp_count_r2_gw:           <vb>
imp_count_r2_gw_canonical: <canonical>
delta:                     <vb - canonical>
```

**Pass A1 hypothesis:** `delta > 0.10` (vb materially better). If `delta ≤ 0.05`, stop and fix `prepare.py` eval metadata before agent loop.

### `scope.py`

```python
ALLOWED_PREFIX = "sandbox/autoresearch/may31/"
```

Fail on any staged/changed path not under prefix. Allow gitignored artifacts without staging.

### `.gitignore` (tag dir)

```
results.tsv
run.log
loop.log
baseline.json
*.pt
__pycache__/
```

**Not** gitignored: `pin_manifest.json`, source `.py`, `program.md`, `README.md`.

### Training loop (5000 steps on 32-batch cycle)

```python
train_batches = load_all_train_pins(manifest)  # len 32, cached RAM
train_masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, p_chunks=0.0, preserve_assay_id=True)
for step in range(MAX_STEPS):
    batch = train_batches[step % len(train_batches)]
    prep = prepare_masked_batch(batch, train_masker, device)
    loss, stats = train_step(model, batch, prep, loss_fn, tc)  # agent hook
    loss.backward()
    clip_grad_norm_(model.parameters(), tc.clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### Gated `primary_score` (implement in prepare.py)

```python
DEN_GATE, DCR_LO, DCR_HI, IMP_PHASE_BIAS = 0.35, 3.25, 4.75, 1.0

def compute_primary(imp_r2, den_r2, dcr) -> tuple[float, str]:
    in_imp = (den_r2 >= DEN_GATE) and (DCR_LO <= dcr <= DCR_HI)
    if in_imp:
        return imp_r2 + IMP_PHASE_BIAS, "imp"
    return den_r2 - DEN_GATE, "den"
```

---

## Agent playbook outline (`program.md` contents)

Copy/adapt into tag dir when implementing.

1. **Goal:** maximize `primary_score` (gated: den-first, then imp R²) with guard-rails.
2. **Read first:** this doc metric section, `prepare.py`, `../ideas/META.md` Q11.
3. **Never edit** `prepare.py` during loop.
4. **One change per iteration** to `train.py`.
5. **Early loop:** expect `metric_phase=den` — tune obs_weight, DSF, depth_center, calibration losses until `den_count_r2_gw ≥ 0.35` and DCR in band.
6. **Late loop:** `metric_phase=imp` — optimize imp R² without collapsing den or DCR.
7. **Priority order:** den/identity (B4 obs/imp weights, C1 DSF off, depth_center) → B1/B3 MSE aux → D1/D3 → D2/D4 (VRAM).
8. **Do not** disable depth_offset (DCR collapses toward ~1).
9. **On crash/OOM/timeout:** log status=crash|oom, reset, simplify change.
10. **Run until** SLURM wall or human stop; ~3–4 runs/hour at ~15 min/run (5000 steps).
11. **Never edit `prepare.py`.** If eval harness seems wrong, stop and ask human.

Reference: [Karpathy autoresearch program.md pattern](https://github.com/karpathy/autoresearch/blob/master/program.md).

---

## Success criteria for E32 (post-loop synthesis)

See **Primary metric → Success criteria** above. Required artifacts:

- `sandbox/autoresearch/may31/results.tsv` (full loop log)
- `sandbox/ideas/synthesis_e32_imp_r2_autoresearch.md` (post-loop; not part of initial implementation)
- Update Q11 in META.md and E32 row in EXPERIMENTS.md

---

## Promotion path (out of scope for loop)

Winning `train.py` config → translate to `sandbox/configs/` + controlled **`train_candi_v2`** run (200 ep, full data) — **separate experiment**, not autoresearch commits.

Do **not** auto-merge autoresearch code into `sandbox/candi_v2/`.

---

## Relationship to May 28 autoresearch

| | May28 (`sandbox/diagnostics/autoresearch/`) | May31 (E32) |
|--|---------------------------------------------|-------------|
| Location | diagnostics/ | `sandbox/autoresearch/may31/` |
| Primary metric | composite (Pearson-heavy) | **Gated:** den≥0.35 + DCR≈4 → maximize `imp_count_r2_gw` |
| Scope rule | diagnostics only | may31 only |
| Metadata eval | masked-bin DCR focus | **V/B natural meta + R²** |
| Code reuse | Reference only | Copy patterns, do not import diagnostics at runtime |

---

## Implementation checklist for next agent

- [ ] Create `sandbox/autoresearch/may31/` package layout (`__init__.py`, `.gitignore`)
- [ ] Implement `scope.py`
- [ ] Implement `build_pin_manifest()` + commit `pin_manifest.json`
- [ ] Implement `load_pinned_batch()` + RAM cache for 32+8 batches
- [ ] Implement `eval_pass.py` (vb_natural + canonical + cloze-T + den + DCR)
- [ ] Implement `prepare.py` (train loop, gated metric, footer, baseline.json, timeout)
- [ ] Implement `train.py` (`TrainConfig`, `train_step()`, `main()` → prepare)
- [ ] Implement `agent_step.py`, `validate_a1.py`, `loop.sh`, `program.md`, `README.md`
- [ ] `git checkout -b autoresearch/may31`; run row 0 on GPU; verify footer parses
- [ ] Optional row 0b: `validate_a1.py` — expect `imp_r2_vb` ≫ `imp_r2_canonical`
- [ ] Start agent loop (human or Cursor agent per `program.md`)

## Handoff prompt (paste into new session)

```
Implement E32 autoresearch harness per sandbox/ideas/autoresearch_may31_r2vscorr_disparity.md.
Copy structure from sandbox/diagnostics/autoresearch/ (reference only).
All code under sandbox/autoresearch/may31/. Do not edit files outside that dir.
Follow "Implementation notes" section exactly. Run GPU baseline (row 0) before declaring done.
Skill: .cursor/skills/candi-autoresearch/SKILL.md
```

---

## Key code references (read-only)

- Eval metrics: `sandbox/eval.py` (`r2_score`, `pearson_r`, `eval_batch_metrics`)
- Eval metadata bug: `sandbox/train.py` lines 542–561 (`canonical_meta` for `y_avail==0`)
- V/B metadata load: `sandbox/data.py` lines 325–351 (`y_meta_imp` from `meta_dsf1`)
- Depth-offset head: `sandbox/candi_v2/decoder.py` `DepthOffsetNegativeBinomialLayer`
- Training forward (no V/B meta mix): `sandbox/train.py` `train_one_epoch` uses `prep["y_meta"]` only
- Pairing: `sandbox/data.py` `_all_imp_biosamples`, eval_pairs loop

---

## Findings (post-loop)

**Outcome: Partial** — imp R² crossed zero and reached **0.122** (`be0d38e2`) but not the **0.15 validate** threshold; den R² **0.279** stayed below the **0.35** spec gate on this pin.

Full rollup: [`synthesis_e32_imp_r2_autoresearch.md`](synthesis_e32_imp_r2_autoresearch.md)

| Result | Detail |
|--------|--------|
| Best keep | `be0d38e2` — `imp=0.59`, `count=2`, `obs=3.5`, `dc=22.5`, `dsf=off`, `mse_obs=0.2` |
| imp / den / dcr | 0.122 / 0.279 / 3.93 |
| A1 (vb vs canonical) | vb +0.122 vs canonical −0.161 at best keep |
| Dominant lever | `imp_weight` 0.5 → 0.59 |
| Rejected | `lambda_mse_imp≥0.1`, May28 `imp=8`, `signal_transform=none`, high `depth_center` |

**Promotion:** translate winning config to a controlled `train_candi_v2` run (separate experiment; not auto-merge).

---

## Open questions (remaining)

1. Does **0.122 imp R²** reproduce on full v2 training (200 ep, unpinned data)?
2. Does **D4 per-assay affine** or groupwise decode lift R² past 0.15?
3. Is **count=1.95 / imp=0.59** a better Pareto point for production (imp 0.115, den 0.31)?

---

## Changelog

- **2026-06-02:** Loop marked done; synthesis written; best keep `be0d38e2` (imp_r2=0.122).

- **2026-05-31 (d):** Implementation notes: pin manifest algorithm, eval_pass fork, model overrides, ops/SLURM, baseline.json, pearson guard den-phase skip, handoff prompt.
- **2026-05-31 (c):** Metric redesign: gated primary (den gate 0.35 → imp R²), DCR hard band [3.25, 4.75], dsf=1 den eval, IMP_PHASE_BIAS for ratchet continuity.
- **2026-05-31 (b):** Grill rounds 1–3 locked: 5000 steps, 32-batch train pin, 8-batch chr21 eval, strict frozen prepare, neutral baseline weights, train_step hook, Tier B/C/D scope.
- **2026-05-31:** Initial spec from Q11 investigation conversation; E32 registered; skill `candi-autoresearch` added.
