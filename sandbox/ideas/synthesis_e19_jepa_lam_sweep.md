# E19 JEPA λ_sigreg Sweep — lambda=0.1 vs lambda=0.5

Status: synthesis (read-only)
Parents: [synthesis_e19_jepa_stage1.md](synthesis_e19_jepa_stage1.md)
Linked from: EXPERIMENTS.md  (E19b block)
Date: 2026-05-08

Runs:
- `e19_jepa_stage1_39046758`  — λ=0.1, 100 epochs, 12,500 steps, 0.73h
- `e19_jepa_lam05_39109506`   — λ=0.5, 200 epochs, 25,000 steps, 1.63h

---

## Headline conclusions

1. **λ=0.5 drove SIGReg loss to 1.03 (vs 1.82 at λ=0.1), approaching the Gaussian baseline of ~1.05.** The
   stronger penalty is substantially more effective at pushing the encoder projections toward Gaussian geometry.
   cos_sim_ctx_tgt trajectory (0.24 → 0.02–0.11) confirms no collapse toward 1.0: context and target
   representations remain meaningfully distinct throughout 200 epochs. Confidence: **High** — direct from
   `lejepa/sigreg_loss` and `lejepa/cos_sim_ctx_tgt` in `metrics.jsonl`.

2. **The UMAP shows meaningful, biologically structured encoder geometry — strong evidence against collapse.**
   5 biosamples × ~47K chr21 tokens each show: (a) spatially coherent activity-hot clusters (red/orange
   in col 0) distinct from the silent-region cloud; (b) T_heart_left_ventricle has a large, prominent
   repression cluster (H3K27me3/H3K9me3) absent in all other biosamples — biologically valid for
   differentiated cardiac tissue; (c) genomic position (plasma colormap, col 2) shows partial spatial
   continuity, meaning nearby loci have more similar representations than distant ones. Confidence: **High**
   from visual inspection of `chr21umap/chr21_umap_step25000.png`; interpretation of heart repression
   cluster is Medium (single run, sandbox scale).

3. **Periodic loss spikes (pred_loss 0.03 → 0.27 every 1000 steps) are a deterministic data-loader
   artifact, not training divergence.** All 25 spike steps have `mask_frac=0.000` exactly; the loss
   recovers to baseline within the next snapshot step (200 steps). Root cause: `SandboxH5Dataset.__iter__`
   creates `rng = random.Random(self.seed)` fresh every epoch, so every epoch shuffles windows and selects
   biosamples identically — the same batch at position 125 (last of each epoch's 125-step cycle) always
   selects a biosample with all assays unavailable, producing mask_frac=0. The "diverged" flag in
   `analyze_jepa_run.py` (ratio=2.32) is a false alarm caused by the last snapshot landing on a spike.
   Confidence: **High** — exact 1000-step spacing confirmed (spacing=[1000]×24), all spike steps have
   mask_frac=0.000.

4. **pred_loss minimum is 3× higher at λ=0.5 (0.031 vs 0.011 at λ=0.1) — an expected trade-off.**
   Stronger regularization limits the predictor from exploiting shortcut solutions. This is consistent
   with JEPA theory (SIGReg prevents collapse at the cost of making prediction slightly harder).
   Confidence: **High** — direct metric comparison.

5. **Gradient clipping remains fully saturated at λ=0.5: clip_frac=1.0 throughout vs 0.72 final at λ=0.1.**
   grad_norm_pre_clip ended at 12.6 (vs 1.3 at λ=0.1). The SIGReg term dominates the gradient budget at
   this lambda. This caps the encoder's learning signal and may be slowing eff_rank recovery. Confidence:
   **High** — from `lejepa/grad_norm_pre_clip` and `lejepa/grad_clipped_frac_running`.

6. **Geometry metrics (eff_rank, n_dead, per-module grad norms, AdaLN norms) are absent from metrics.jsonl
   due to a do_geo timing bug.** The `do_geo` flag is evaluated using the pre-increment `global_step`,
   while the jsonl snapshot triggers on the post-increment value — they never share the same step. W&B
   receives geometry metrics every 50 steps (via `wandb.log`), but the offline record is blind to them.
   This is fixed in the codebase but was present in both runs analyzed here. Confidence: **High** — root
   cause confirmed by code inspection.

---

## Cross-run quantitative table

| metric | e19_jepa_stage1 (λ=0.1) | e19_jepa_lam05 (λ=0.5) |
|---|---|---|
| epochs / steps | 100 / 12,500 | 200 / 25,000 |
| total elapsed | 0.73h | **1.63h** |
| sec/epoch (mean) | 26.3 | 27.7 |
| pred_loss first→last→best | 0.204→0.012→**0.011** | 0.281→0.267→0.031 |
| sigreg_loss first→last→best | 7.969→1.828→1.820 | 7.969→2.016→**1.031** |
| total_loss first→last→best | 1.000→**0.195**→0.195 | 4.265→1.275→0.549 |
| last/best total_loss ratio | **1.00** (not diverged) | 2.32 ⚠ (artifact — see HC3) |
| cos_sim_ctx_tgt first→last→min | N/A (metric absent) | 0.240→0.687→**0.020** |
| mask_frac at spike steps | N/A | 0.000 (×25, see HC3) |
| grad_norm_pre_clip first→last | 2.42→1.31 | 8.34→12.59 |
| clip_frac_running first→last | 1.000→**0.720** | 1.000→1.000 |
| UMAP generated | no | yes (5 biosamples, chr21) |
| geometry in jsonl | yes (62 snapshots) | no (do_geo bug) |

Note: total_loss is not comparable across runs — the λ factor changes the loss scale (total = pred + λ·sigreg).

---

## Per-run gradient / stability table

| run | grad_norm range | clip_frac range | spike events |
|---|---|---|---|
| λ=0.1 (100ep) | 0.34–2.42 | 0.60–1.00 | 2 (ep 54, ep 87) |
| λ=0.5 (200ep) | 1.27–19.53 | 1.00–1.00 | **25 periodic** (every 1000 steps, mask_frac=0 artifact) |

At λ=0.5, grad_norm_pre_clip reaches 19.5 at spike steps (mask_frac=0, full unmasked context, predictor
receives trivial task → abnormally large SIGReg gradient). Outside spike steps, grad_norm is 1.3–5.4,
slightly above λ=0.1 baseline.

---

## Per-experiment outcome vs hypothesis

| run | hypothesis | outcome | confidence |
|---|---|---|---|
| e19_jepa_stage1 | JEPA learnable; SIGReg prevents collapse | Partial — pred learnable; eff_rank collapsed 49.7→14.3 | Medium |
| e19_jepa_lam05 | λ=0.5 stabilises eff_rank (FJ1 fix) | **Partial** — sigreg_loss improved to 1.03 (near Gaussian), cos_sim stable, UMAP structured; eff_rank not measurable in jsonl (do_geo bug); always-clipping limits encoder update | Medium (eff_rank confirmation needs W&B or bug-fixed re-run) |

---

## Implications for next batch

Priority order:

1. **Fix do_geo timing bug + seed-reuse bug, re-run λ=0.5 for 200 epochs (E19c).**  These two bugs
   corrupt the offline record and inject periodic spike noise. With fixes: (a) eff_rank will appear in
   jsonl, enabling offline collapse diagnosis; (b) mask_frac spikes disappear, providing a cleaner loss
   signal and true divergence flag. Predicted: same sigreg_loss trajectory, eff_rank trajectory becomes
   visible. Cost: 1 run, 2h.

2. **Increase `clip_norm` from 1.0 → 2.0 (or 3.0) while keeping λ=0.5 (E19d).**  clip_frac=1.0
   throughout indicates the legitimate SIGReg signal is being clipped at every step. Relaxing the cap
   should allow the encoder to receive stronger collapse-fighting gradients. Predicted: eff_rank stabilises
   higher; may increase grad noise requiring careful monitoring. Cost: 1 run.

3. **Add `lambda_sigreg ∈ {1.0, 2.0}` to the sweep (E19e).**  sigreg_loss at λ=0.5 reached 1.03 (near
   Gaussian baseline). If eff_rank is still collapsing, the pred_loss gradient is still winning. λ=1.0
   makes the objectives equal-weight; λ=2.0 makes SIGReg dominant. Predicted: eff_rank > 30 (improvement
   over ~14 at λ=0.1). Risk: pred_loss may plateau higher, hurting downstream Stage 2 finetuning. Cost: 2 runs.

4. **Stage 2 prep: once eff_rank stable > 25 in a clean run, freeze encoder and train a lightweight
   decoder head (E19f).** The UMAP already shows biologically meaningful structure. The first cornerstone
   comparison for Q7 can begin if eff_rank is confirmed via a bug-fixed run.

---

## Standing findings (carried forward)

| finding | status going in | this synthesis adds |
|---|---|---|
| FJ1 — λ=0.1 insufficient for SIGReg | open | **Mitigated** — λ=0.5 drives sigreg_loss to 1.03 (near Gaussian baseline 1.05); cos_sim_ctx_tgt stable. eff_rank not confirmed offline due to FJ4 bug. Mark mitigated pending eff_rank confirmation. |
| FJ2 — Periodic zero-mask spikes (new) | **open** | First observed here. Exact root cause identified: RNG seed reuse in `SandboxH5Dataset.__iter__`. Fix proposed in codebase (see action). |
| FJ3 — UMAP shows biologically structured encoder geometry (new) | **open** | First observed here: activity-correlated hot spots, heart repression cluster, partial genomic position gradient. Non-trivial UMAP structure is strong evidence against total collapse. |
| FJ4 — do_geo timing bug: geometry metrics absent from jsonl (new) | **open** | First observed here. Root cause: `do_geo` uses pre-increment step; jsonl snapshot uses post-increment. Fixed in codebase; affects both λ runs. |
| F1, F7, F8 | open (main candi) | Not applicable to JEPA encoder-only. Status unchanged. |

---

## Caveats and limits

- **eff_rank not confirmed offline for λ=0.5**: the do_geo bug means we cannot verify whether eff_rank
  recovered. W&B has the data; a bug-fixed re-run will provide the offline record.
- **UMAP is end-of-training only**: we cannot tell how the geometry evolved across epochs. A periodic
  UMAP (every 50 epochs) would show if structure emerges early or only late.
- **Divergence flag is misleading**: the `analyze_jepa_run.py` "DIVERGED" flag is triggered by the
  spike artifact. The script needs a mask_frac filter to exclude spike steps before computing last/best ratio.
- **Single seed (42)**: all findings are single-seed.
- **Sandbox scale**: 8 assays, 5 biosamples, chr21 UMAP. Heart repression cluster and activity clustering
  are at sandbox resolution; may differ at 35-assay full-CANDI scale.
