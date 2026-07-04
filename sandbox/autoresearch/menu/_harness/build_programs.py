"""Generate the 5 per-loop program.md (the agent-facing steering doc).

Each gets: the shared CONTRACT + anti-rediscovery PRIORS (ERA's exploited findings, as knowledge,
so no GPU re-derives them) + the keep-rule — and ONLY its own thesis. The rival menu directions are
deliberately WITHHELD so a greedy agent cannot drift back into the exploited CF basin.
"""
from __future__ import annotations

from pathlib import Path

MENU = Path(__file__).resolve().parents[1]

CONTRACT = """\
## Editable surface (inside THIS loop dir only; one surgical change per iteration)
- ./train.py      — adapter `Model` (v3 9-input forward → {count_dist, signal_pred, peak_prob}) +
                    `Objective` (corrupt / loss / configure_optimizer) + the run_and_score call.
- ./candi_model/  — the VENDORED, FULLY EDITABLE model (fork of candi_v2): config.py, encoder.py
                    (conv towers + metadata FiLM + transformer/attention), decoder.py (NB count +
                    peak heads), model.py (forward; latent Z is UN-detached at out["z"]).
The forward CONTRACT (9 inputs → that 3-key dict) and the FROZEN judge `run_and_score` cannot change.
Budget fixed = min(5 epoch, 1800 s). Data: chr19 TRAIN / chr21 EVAL (8-assay slice). _judge/ is
READ-ONLY. Crash / OOM / non-finite / constant-output / forbidden import → −1e9.

## Row-0 objective (B): three heads trained on REAL labels
The h5 T_ groups (chr19) hold real pval (arcsinh −log10 p) + peaks {0,1}; the Objective reads them via
a LOCKSTEP iterator + a counts-equality assertion that PROVES correct (bios,window) pairing. Heads:
NB count NLL + Gaussian-NLL pval(mu,var) + BCE peak, on query/available positions. signal_pred = pval mu.

## HARD GUARDS — never remove (removing them invalidates the run)
- Keep the cuDNN-determinism flags AND the pairing counts-equality assertion in train.py.
- Keep TRAIN_BS=4 / TRAIN_SEED=0 (= run_and_score defaults) or the pairing assertion will crash.
- NEVER read chr21 / eval windows in training: use `_win_indices(train=True)` (chr19) only. Leakage = invalid.

ERA_SCORE = (Q_imp − 0.4857)  − 0.5·max(0,Q_imp−Q_den)  + 0.4·min(0,0.0734−ECE)
          + 0.4·min(0,c_index−0.4985) + 0.4·min(0,peak_auroc−0.7161) + 0.02·DCR-band[3,5].
PRIMARY = raise Q_imp (mean of imp {pval,count}×{spearman,pearson} on held-out V/B assays); the
floors are do-no-harm. Higher is better.
"""

PRIORS = """\
## Row-0 status (B anchor, deterministic) — where the headroom actually is
- ERA_SCORE −0.126. pval-Spearman 0.52, count-Spearman 0.52, peak-AUROC 0.78 ALREADY BEAT the baseline
  (0.47 / 0.47 / 0.72). The deficit to a POSITIVE score is the two PEARSON (magnitude) terms:
  count-Pearson ≈ 0.17 is the single lagging metric (rank is fine, the MEAN's magnitude/scale is off),
  pval-Pearson ≈ 0.30. ECE rose to ≈0.12 (> 0.073 floor → small penalty). Attack magnitude calibration
  + ECE; do not chase rank (already won).

## Proven priors (ERA's exploited findings — use as KNOWLEDGE, do not waste iterations re-deriving)
- The marginal average-reference baseline is brutally strong; the ONLY thing that beats it is the
  cell-type-specific DEVIATION on a leak-free average-reference (zero-init deviation head over
  reference+CF, and/or a (pred−ref)·(target−ref) deviation-correlation loss). Model the residual.
- The headroom is MAGNITUDE (Pearson/R²), not rank (Spearman). count-mean Pearson is the lagging
  correlation. Pure rank/soft-Spearman losses REGRESSED skill — avoid as a dominant term.
- ECE (count calibration) is the binding, still-unsolved floor; ~78% of candidates miss it. A LIGHT
  dispersion/variance/coverage term helps; a HEAVY aux term is neutral-to-harmful (capacity-limited,
  not gradient-limited).
- Depth enters ONLY as a multiplicative size-factor on the count mean (μ=2^(d−c)·exp(η)); this holds
  DCR in band by construction. The NB/size-factor head is FRAGILE — innovate around it, not inside it.
- arcsinh/log1p the counts; per-assay mask/query tokens beat one shared token; mask cross-assay
  attention to genuinely-present assays; do NOT heavily up-weight masked positions (memorization).
- Multi-resolution regional pyramids (~200/800/3200 bp) help magnitude + peak shape. GroupNorm in the
  decoder and a single-shot target-metadata FiLM were the biggest v2 decoder wins.
- Dead-ends (don't repeat): soft-Spearman-dominant loss; standalone extra heads; MACS local-ratio as
  the regression TARGET; heavy auxiliary MSE/coverage.

## Keep-rule & discipline
- Keep iff ERA_SCORE > current best AND no floor regressed into a degenerate gate. Else it auto-resets.
- ONE change per iteration, minimal and attributable. Prefer deleting code that didn't help.
- You may `git log`/`git show <sha>`/`git diff <a> <b>` on THIS loop to see the exact code + score of
  every prior attempt (kept and failed). Build on the kept lineage; don't repeat the rejected ones.
- Stay strictly inside YOUR thesis below. Do NOT drift toward other architectures.
"""

THESES = {
    "single_lambda": """\
# THESIS — single_lambda
Predict ONE depth-free latent enrichment field λ(assay,pos) as the ONLY learned quantity. Derive all
three views from it instead of independent heads: counts ~ NB(λ·size_factor, φ) as the only learned
likelihood; the pval/signal as a deterministic transform of λ; peaks as P(λ>τ) under the count
posterior. Collapsing the heads should make calibrating λ ONCE calibrate every view — directly
attacking the unsolved ECE wall by construction. Forbidden: keeping separate independent count /
signal / peak heads (that is the exploited CF design; the whole point is to collapse them).""",
    "crps_calibration": """\
# THESIS — crps_calibration
Replace the NB + second-moment band-aids with a CALIBRATED-BY-DESIGN distributional output: quantile
regression or a CRPS-trained predictive distribution over counts whose coverage is correct by
construction. Attack the unsolved ECE floor from the LIKELIHOOD side (ERA only ever patched a fixed
NB's variance/PIT). Keep depth as the multiplicative size-factor. Forbidden: reverting to a plain NB
with bolted-on coverage penalties (the exploited band-aid path).""",
    "factorized": """\
# THESIS — factorized
Model the data as an EXPLICIT low-rank tensor factorization — separate learned embeddings for
cell-type × assay × multi-resolution genomic-position factors (fine + coarse for cheap long range),
combined by a small DNA-conditioned network — NOT attention. This is net-new for CANDI (ERA used only
attention-based borrowing). Factorization should natively express the average-reference + deviation
decomposition. Forbidden: making cross-assay self-attention the core mechanism (that is the exploited
CF basin).""",
    "axial_longrange": """\
# THESIS — axial_longrange
Replace the per-position cross-assay set-transformer with AXIAL attention over (assay × position) PLUS
a memory-efficient dilated/hierarchical long-range position mixer reaching enhancer–promoter / TAD
scale (no dense O(L²)). The bet: cell-type specificity and within-peak shape need long-range context
the per-position spine never had. Forbidden: collapsing back to per-position-only cross-assay attention
+ a small conv (the exploited spine).
WHERE: edit ./candi_model/encoder.py — that is where the transformer block is wired in (currently
`DualAttentionEncoderBlock`); replace it with your axial + long-range mixer (you own this file now).""",
    "repr_first": """\
# THESIS — repr_first
Build a STRONG latent Z with LIGHT readout heads + a LIGHT latent regularizer (Gaussian prior + ELBO,
or SIGReg on the latent — NOT full JEPA). Bet: a regularized latent generalizes better on held-out
imputation (the scored S_A). Latent quality (a linear probe / eff-rank) is a DIAGNOSTIC only — the
ratchet still keys on ERA_SCORE. Forbidden: heavy auxiliary objectives, or full JEPA training.
WHERE: the latent is exposed UN-detached at `out["z"]` in ./candi_model/model.py; add your latent
regularizer in the ./train.py Objective.loss (and lighten heads there). Do not collapse Z.""",
}


def main():
    for tag, thesis in THESES.items():
        (MENU / tag).mkdir(exist_ok=True)
        (MENU / tag / "program.md").write_text(thesis + "\n\n" + CONTRACT + "\n" + PRIORS)
        print("wrote", tag, "/program.md")


if __name__ == "__main__":
    main()
