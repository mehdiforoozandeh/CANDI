# CANDI crux seed — reconstructed from the repo (2026-07-02). Edit freely; `#` lines are ignored.

- Project: CANDI — self-supervised, confidence-aware epigenome imputation & denoising from raw ENCODE read counts

  # ═══ Pillar A — the published v1 model (bioRxiv 2025.01.23.634626) ═══

  - Q: Does a raw-count, distribution-output SSL model do state-of-the-art zero-shot epigenome imputation?
    - H: [tested] CANDI beats EIC-challenge competitors on rank correlation at ~42M params with no cell-type embeddings
      - v: [x] Spearman >= top EIC competitor on most assays (found: SOTA Spearman across most assays, zero-shot via covariates)
      - v: [ ] Pearson >= competitors (found: Pearson lags — compressed dynamic range, high-signal magnitude underestimated)
      - finding: Structural/rank imputation is SOTA and cell-type-agnostic, but absolute magnitude of high-signal regions is underestimated. This rank-vs-magnitude gap is the recurring theme of all later work.
    - H: [tested] Peak calling is preserved even where correlation is only moderate
      - v: [x] peak AUROC near 1.0 for sharp marks such as H3K4me3 (found: ~1.0 AUROC even at Pearson ~0.35)
      - finding: Biological signal-to-noise is preserved despite magnitude compression, supporting imputation-as-denoising.

  - Q: Are CANDI's aleatoric uncertainty estimates calibrated and useful?
    - H: [tested] Distributional outputs give near-nominal calibration at practically relevant confidence intervals
      - v: [x] empirical coverage ~ nominal at CI >= 0.9 (found: reliable/conservative at 0.9-0.95)
      - v: [ ] calibrated across the full CI range (found: overconfident at CI < 0.5; assay-dependent, DNase better than histones)
      - finding: Calibrated where it matters most (high CI) but overconfident at low CI and assay-dependent. Calibration later proves to be the binding constraint in v3.
    - H: [tested] The predicted distributions rank signal correctly (C-index)
      - v: [x] genome-wide C-index high and rises in relevant regions (found: H3K4me3 ~0.6 gw rising to ~0.83 in promoters)
      - finding: Distributional predictions capture relative ordering; predicted CV flags likely errors (high variance where it predicts low but truth is high).
    - H: [tested] Alternative signal likelihoods change the calibration trade-off
      - v: [x] Gaussian/Laplace/Student-t/Gamma/const-var heads implemented, ablated at ~52M params on EIC chr19 with calibration curves (found: dist_report.md, calibration_imputed_log_normal/laplace.svg)
      - v: [-] a default likelihood chosen for the paper (found: winner not recorded — needs confirm)
      - finding: The signal head is pluggable and the distributional assumption was actually tested; which to present as the default is still unconfirmed.

  - Q: Do CANDI's imputed signals and latent Z encode biology (predict RNA-seq it never saw)?
    - H: [tested] Latent Z predicts gene expression better than observed/denoised signal and is robust to input sparsity
      - v: [x] Z beats observed and denoised+imputed features on RNA-seq log-TPM via nested CV (found: Z is the strongest predictor)
      - v: [x] Z and imputed+denoised performance are robust to the number of available input assays (found: robust to sparsity unlike observed)
      - finding: The latent encodes higher-order regulatory information beyond the decoded tracks; strongest and most sparsity-robust RNA-seq predictor. Core biological-validation claim.
    - H: [tested] Denoised+imputed 35-assay features beat the observed subset
      - v: [x] denoised+imputed 35 assays > observed available assays on RNA-seq prediction (found: imputation adds regulatory context)
      - v: [x] denoised >= observed (found: marginal gain — noise removed, regulatory info preserved)
      - finding: Imputation and denoising add real regulatory signal, supporting imputation-as-denoising.

  # ═══ Pillar B — CANDI 2.0 method additions (post-v1, ~18 months, scattered across repo) ═══

  - Q: Can CANDI's counts be made depth-controllable to denoise toward a canonical "supertrack"?
    - H: [tested] The v1 NB count head is depth-controllable via the output prompt
      - v: [ ] output count ratio responds to a +2 log2 depth prompt (found: DCR ~ 1.0 on the default head — invariant to y_meta, depth collapse)
      - problem: Asking for a higher-depth supertrack did nothing because the count mean carried no depth dependence.
      - finding: Root cause of the supertrack failure, diagnosed honestly as a negative result — the NB mean was depth-blind.
    - H: [tested] A depth-centered size-factor reparam restores depth sensitivity to DCR ~ 4
      - v: [x] the depth-centered size factor mu = 2^[d-24]·exp[eta] gives DCR ~ 4.0 across overfit, assay-only mask, count+peak and 3-epoch training (found: R15-R20, DCR 3.99-4.02 from epoch 0)
      - v: [ ] the raw 2^d offset works (found: raw offset FAILS at DCR ~ 1.0)
      - v: [-] it reproduces at 35-assay MERGED production scale (found: validated only on the 8-assay chr19 diagnostic — needs confirm)
      - finding: depth_center ~ batch-median log2 depth (~24 on EIC) is the fix; raw 2^d fails. Enables controllable denoising to a canonical depth — the strongest candidate new results subsection; production-scale confirmation still open.
    - H: Per-assay independent DSF sampling makes depth a necessary signal at production scale
      - v: per-assay DSF in the default training recipe improves DCR and robustness versus shared DSF

  - Q: Does query-based decoding fix the fixed-channel shortcut and improve controllability?
    - H: [tested] Query decoders (MoE / DynConv / CondConv) decode only queried assays from an assay-id-keyed decoder
      - v: [x] the query-decoder family is implemented and smoke-trained, backward-compatible with the fixed decoder (found: models/*QueryDecoder_CondConv5_Hybrid*, Mar 2026)
      - v: [ ] query decoding beats the fixed decoder on imputation or controllability (found: full parity benchmark not yet run, spec flags it as needed)
      - problem: With fixed output channels a decoder can learn "channel 5 = H3K4me3" and ignore the output prompt — the supertrack root cause.
      - finding: The mechanism to break the fixed-channel shortcut is built and runs; whether it actually won over the fixed baseline is unconfirmed.
    - H: [tested] Explicit assay_id metadata (platform dropped) improves prompt-conditioning
      - v: [x] covariates are now depth+assay_id+read_length+run_type with an assay_id embedding, cloze/missing sentinel split and a MaskStem (found: model.py MetadataEncoder/QueryMetadataEncoder)
      - finding: Small but real method delta versus the submitted Methods, motivated by the prompt-invariance failure.

  - Q: Does a unified SuperLoss beat the static weighted-sum loss?
    - H: [tested] SuperLoss variants train, but a keeper is not yet identified
      - v: [x] assay-EMA reweighting, robust-stable count, uncertainty weighting and fg/bg are implemented and trained (found: models/*SuperLoss_*, Mar 2026)
      - v: [-] a winning variant with ablation deltas is recorded (found: not captured in any synthesis doc — needs confirm)
      - finding: The consolidated loss mechanisms exist and ran; which variant is canonical is unrecorded. Likely a Methods refinement plus an ablation table, not a headline.

  # ═══ Pillar C — sandbox training diagnostics (fast 8-assay harness) ═══

  - Q: Why do sandbox training runs diverge late, and what stabilizes them?
    - H: [tested] log1p input scaling and SGD remove late divergence, but not depth collapse
      - v: [x] log1p B7 and SGD-lr1e-4 B4 show no divergence and improve imputation over the raw B1 baseline (found: both accepted for stability)
      - v: [ ] depth collapse is also fixed by these (found: depth collapse persists, needs the size-factor fix)
      - finding: Input scaling and optimizer choice control divergence but not depth collapse — the two failure modes are separate.
    - H: [tested] Gradient clipping is load-bearing at full LR
      - v: [x] removing clip at default LR trails the clipped baseline (found: E0b stable but below B8; clipping load-bearing)
      - v: [x] a durable clip-active-fraction pressure metric is logged (found: E10 implemented in metrics.jsonl)
      - finding: Clipping is required at full LR, not just a safety net; removing it underperforms even when training stays stable.

  - Q: Do the count / pval / peak heads cooperate or interfere during multi-head training?
    - H: [tested] Isolating the pval head (pval-only training) improves pval learning
      - v: [ ] pval-only training improves pval (found: E3 rejected — variance collapse on obs, pval_imp explodes, root cause F7)
      - finding: The pval head is the source of instability and motivated the Gaussian variance floor.
    - H: [tested] The peak head is healthiest in isolation while counts are capacity-limited
      - v: [x] peak-only is the healthiest head with a strong AUROC ceiling and no divergence (found: E4 accepted)
      - v: [ ] count-only reaches good imputed counts (found: E2 count_imp plateaus ~1.92; count+peak best for counts but peak still needs pval gradients)
      - finding: The heads have asymmetric health — peak robust, counts capacity-limited, pval fragile; full multi-head is a compromise.

  - Q: How should conditioning (FiLM) and metadata be routed through the model?
    - H: [tested] A single-shot decoder FiLM beats per-layer decoder FiLM
      - v: [x] one latent FiLM E7 is the best multi-head run in the sweep and was promoted default (found: F8; beats linear FiLM E6)
      - finding: A single FiLM that makes the decoder a pure spatial upsampler wins; per-layer decoder FiLM over-conditions.
    - H: [tested] The metadata pathway collapses depth-of-coverage (depth ignored)
      - v: [x] depth-of-coverage is ignored by the model (found: F1 metadata collapse, dcr ~ 1)
      - v: [ ] this is fixed on the production stack (found: only a partial fix on v2 via the depth-offset head; production B8 open)
      - finding: Depth collapse (Q5) is the sandbox mirror of the supertrack failure; the E29/E30 depth-offset head is the fix, validated on v2 but not yet at production scale.

  - Q: Does the GaussianLayer variance floor prevent pval collapse?
    - H: [tested] gaussian_var_min = 0.1 stops pval obs/imp divergence
      - v: [x] the variance floor mitigates F7 in pval-only isolation (found: E13 accepted)
      - v: [-] it is confirmed in full multi-head training (found: validation still pending)
      - finding: The floor mitigates the collapse; multi-head confirmation is still needed.

  - Q: Does JEPA / SIGReg latent pretraining beat reconstruction-only training?
    - H: [tested] Encoder-only JEPA yields usable Stage-1 latents only when warm-started from the CANDI encoder
      - v: [x] lambda=0.5 with pred_hidden=16 prevents collapse and is the best config (found: E19 accepted for Stage 1)
      - v: [ ] a purpose-built fresh encoder recovers CANDI-encoder geometry (found: fresh encoder is the root cause of blob UMAPs; all 22 E23 fresh runs fail the v2 geometry gate)
      - finding: Encoder-only JEPA works from a warm start; a fresh-from-scratch encoder collapses geometry.
    - H: [tested] JEPA Stage-2 decoding on frozen latents matches end-to-end reconstruction
      - v: [ ] frozen-latent decoders >= the B8 baseline (found: E28 rejected so far, trails B8)
      - finding: Two-stage JEPA decoding underperforms end-to-end CANDI at sandbox scale; JEPA is deprioritized in v3 ("no full JEPA").

  # ═══ Pillar D — v2 modular backbone + autoresearch program ═══

  - Q: In the v2 backbone, why is imputed-count R2 ~ 0 despite Pearson ~ 0.5 (rank vs magnitude)?
    - H: [tested] Autoresearch (E32) lifts imputed-count R2 above zero via an eval fix plus loss reweighting
      - v: [x] vb_natural eval with imp_weight ~ 0.59 raises imp R2 to 0.122 (found: E32 partial, denoising peaked ~0.31)
      - v: [ ] imp R2 clears the 0.15 validate gate (found: 0.122 below the gate; E33 full-data confirm peaked 0.162 at ep44 then collapsed)
      - finding: The rank-magnitude decoupling is real; AR moved imp R2 positive but below the gate, with late-epoch collapse.

  - Q: What architecture changes move held-out imputation skill in the v2 backbone (E34 autoresearch)?
    - H: [tested] A stacked encoder/decoder config (KEEP12) is the single-knob optimum
      - v: [x] KEEP12 reaches primary ~ -0.4438 (found: +0.026 from decoder GroupNorm and +0.0038 from output_rms_norm over KEEP9)
      - v: [ ] single-knob search yields further real gains (found: exhausted; noise floor ~0.002; every untested knob is locked or toxic)
      - finding: KEEP12 is locked as best. imp-vs-den is a hard Pareto frontier rooted in the shared transformer backbone; the NB/depth head and transformer internals are structurally immutable.
    - H: [tested] Scaling capacity breaks the imp-vs-den Pareto frontier
      - v: [ ] capacity scaling improves the combined score (found: decoder capacity gives den_r2 +0.256 but wrecks imputation; transformer capacity gives best imp but wrecks denoising)
      - finding: Under-convergence is real but capacity cannot beat the score — the frontier is fundamental to the shared backbone. Breaking it needs a regime change (loss weights toward the score, or larger compute/data for a dual backbone).
    - H: [tested] Isolated single-thesis loops (menu-AR) beat base candi_v2 without monoculture collapse
      - v: [x] all 5 loops beat base -0.1261 by +0.024 to +0.085, crps_calibration best at -0.0408 (found: CRPS proper score plus LOO-ref deviation-correlation lifts count-Pearson 0.165 to 0.292)
      - v: [ ] any loop beats the marginal average-reference baseline (found: best Q_imp 0.450 < 0.4857, S_A still negative)
      - v: [ ] cross-loop consolidation stacks additively (found: FALSIFIED — GroupNorm anti-synergizes with CRPS, crps alone stays best)
      - finding: The universal winning mechanism is leak-free deviation-from-average-reference modeling plus calibration; the ceiling is the chr19-only training data, not architecture.

  # ═══ Pillar E — CANDI v3: first-principles ERA redesign (current frontier) ═══

  - Q: Can a from-scratch ERA-searched CANDI v3 cross the average-reference imputation baseline?
    - H: [tested] Spending capacity on cell-type-specific deviation, not relearning the shared average, is the only thing that crosses the marginal baseline
      - v: [x] a zero-init residual head on a leak-free average-reference plus a deviation-correlation loss crosses the baseline (found: ERA best node 267, S_A = +0.0183 > 0 over ~281 candidates)
      - finding: Confirms the ENCODE-challenge lesson — the per-position average reference is brutally strong, and skill is the cell-type deviation on top of it. This is v3's central mechanism.
    - H: [tested] Calibration (ECE) is the binding, still-unsolved constraint
      - v: [ ] NB count calibration meets the ECE floor from NLL alone (found: ~78% of candidates fail ECE, NB counts systematically over-confident)
      - v: [x] explicit second-moment / CRPS / dispersion-cap terms recover coverage (found: moment-matching and CRPS lift calibration where NLL cannot)
      - finding: NLL does not yield coverage; explicit calibration terms are required. This is the honest "calibration is the hard part" framing for the paper.
    - H: Relaxing the frozen single-chromosome regime to multi-chromosome / MERGED pushes S_A clearly positive
      - v: multi-chromosome training lifts held-out imputation Spearman above the marginal baseline by more than scale_A
