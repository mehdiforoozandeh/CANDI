---
id: q20
type: question
title: How should CANDI architecture and training condition on experimental metadata to improve imputation magnitude AND genuine metadata use
parent: q18
status: open
stale: true
created: "2026-07-24T11:10:04"
updated: "2026-08-01T18:18:11"
---

# q20 — How should CANDI architecture and training condition on experimental metadata to improve imputation magnitude AND genuine metadata use

Parent:: [[q18_do_the_dual_conditioning_testbed_finding]]
Literature:: [[wiki/film-conditioning]], [[wiki/covariate-conditioning-and-counterfactuals]], [[wiki/query-decoders-and-conditional-computation]], [[wiki/digest-normalization-assumptions-of-prior-imputation-methods]]

## Question

**How should CANDI's architecture and training condition on experimental metadata so that imputation magnitude improves AND metadata is genuinely used, not ignored?**

This supersedes [[h45_removing_the_depth_offset_head_recovers_|h45]] (the offset-off/hybrid framing). The pivot is [[h47_the_offset_on_steering_null_is_a_weight_|h47]]: the recorded q19 "offset-ON/OFF Pareto" was largely a **weight-decay artifact**. With `weight_decay=0` the offset-ON model beats the old anchor on every M1 axis (macro CRPS 1.495→**1.341**, imp-Spearman 0.533→0.637, ECE 0.062→0.053), and the offset-OFF wd=0 control does NOT improve, so the gain is pathway-revival-specific. That real-z re-probe is now **retracted**: [[h48_h0_fix_the_broken_q19_instruments_and_re|h48]]/F2 shows the 0.833 was a **MISSING-sentinel artifact** (the whole-row permute slid the MISSING(−1) sentinel across UNAVAILABLE slots, whose prompt columns are fully (−1,−1,−1,−1)). Sentinel-free at true full coverage `wd0_on` reads **0.0023** — 43× below h47's own ≥0.10 bar, and 1816×/4224× below `offoff`/`wd0_off` (**4.1772** / **9.7144**) on the identical probe. So the offset-ON decoder assay-FiLM is **not** functional-but-redundant; it is **near-silent on the prompt pathway**, and the block is scale, not the embedder: `wd0_on`'s pre-fusion-LayerNorm activation norm is **396 vs 8.9**, so LN divides the assay perturbation by ~70 (post-LN 0.0091 vs 1.41) and `film_proj` attenuates a further ~18× — end-to-end deficit ~4 orders. Confound: `assay_id ≡ slot index` on this data and the trunk emits a per-slot channel block, so every steering number here bounds the **prompt pathway**, not assay-awareness. Row separation is **anti-diagnostic** (`offoff`, the strongest steerer, has 10× *smaller* effective rank than `wd0_on`) — do not use embedding-table geometry as a steering proxy.

So the residual problem is no longer "the pathway is dead" (solved) but **"even alive, conditioning is under-expressed, redundant with `z`, or data-unidentifiable (run_type, bound B1)."** q20 asks how to fix that.

**Arbiter (PI):** MAGNITUDE arbitrates — beat `wd0_on`'s **oracle-scale-decomposed capability** term **1.3077** by **more than the target-clustered noise floor (~0.09)**, not by a bare 4-dp threshold — with SHAPE (macro Sp ≥ **0.5653**), CALIBRATION (ECE ≤ **0.0533**) and STEERING as constraints. STEERING must be measured **sentinel-free** (real→real / cross-target ablation); the whole-row assay permute is retired. `weight_decay=0` remains a baked default for every child. run_type-steering-as-success is unwinnable on the current panel (B1) and lives only under [[h57_a_re_selected_biosample_panel_is_the_onl|h57]]. The bare `macro CRPS <= 1.341` bar is RETIRED for every child of q20 (H48:L269-270); ~~the nine child nodes that still cite it (h49, h50, h51, h52, h54, h55, h56, h57, h58) are corrected in a single PI-approved batch.~~ **[CORRECTED 2026-07-28 — the earlier list was stale: h52 and h55 have already been rewritten off 1.341. Re-verified by grep over the vault, the nodes that STILL cite the retired bar are the seven q20 children h49, h50, h51, h54, h56, h57, h58 (all in their `## Verifiables`), plus h44 under q19 (its `metric:` frontmatter string, V1 and Findings, "to be tested vs wd0_on 1.341"). These are corrected in a single PI-approved batch.]**

**Structure.** [[h48_h0_fix_the_broken_q19_instruments_and_re|h48]] (H0) is the **blocking, 0-GPU** instrument fix; it has now RUN (verdict *partial*, 2026-07-28) and its re-scored capability baselines are what every arm below is judged against. Its two standing caveats bind every child: the validation gate is a **consistency gate against `METADATA_AUDIT.md`, not an orthogonal validation**, and the four-arm capability **reordering is NOT established** — only `wd0_on`'s lead survives target-clustered inference. Round-1 architecture/training arms (wd=0 default, 1 seed screen): [[h49_read_length_as_a_fixed_coefficient_physi|h49]], [[h50_an_explicit_per_assay_output_factor_loca|h50]], [[h51_a_no_decay_parameter_group_decoupled_ada|h51]], [[h52_live_decoder_film_init_xavier_n_0_0_1_re|h52]], [[h54_conditioning_dropout_manufactures_the_mi|h54]], [[h55_a_grouped_decoder_trunk_groups_a_optiona|h55]] → ≥3 seeds on the finalists that ~~beat 1.341~~ **[UPDATED 2026-07-28 (h48/F2): beat `wd0_on`'s oracle-scale-decomposed capability `1.3077` by more than the ~0.09 target-clustered noise floor (H48:L269-270) — the bare 1.341 gate is retired]** (~12 arms). Deferred data-side track (needs re-bake/grids): [[h53_metadata_steered_dispersion_read_length_|h53]], [[h56_adding_sequencing_platform_lab_with_oov_|h56]] (= the old h44), [[h57_a_re_selected_biosample_panel_is_the_onl|h57]], [[h58_downsample_input_dsf_augmentation_plus_a|h58]]. Full working proposal + drop-list (2 killed: told-depth dispersion, aux-decodability loss) at `scratchpad/PROPOSAL.md`.

See-also [[q19_can_we_make_dual_conditioning_work_on_re|q19]] · [[h46_the_offset_off_imputation_gap_is_scale_m|h46]] (scale-not-biology, now formalized as h48's oracle-scale decomposition) · [[h44_adding_sequencing_platform_lab_with_an_u|h44]] (subsumed by h56).

## Answer so far

_(interpretation — written by the PI/agent; auto-flagged stale when new evidence lands)_

<!-- crux:ledger:start -->
**11 children** · ideas 1/11 done (supported 0, partial 1, refuted 0, inconclusive 0)

- `h48` [[h48_h0_fix_the_broken_q19_instruments_and_re|H0 — Fix the broken q19 instruments and re-score the four existing checkpoints (0 GPU) before any new arm]] — *done* — verdict **partial**, metric `gate GREEN (76 tests; S1/S3/S4/S5/S6/S14) + 4 ckpts re-scored (M1/M2 full chr21 coverage): oracle-scale compression 0.7148->0.1133 (84%), capability wd0_on 1.3077 best (only surviving pairwise: offoff-wd0_on +0.093 [+0.004,+0.217]; reordering NOT established); h47's assay steering 0.833 = MISSING-sentinel artifact, sentinel-free 0.0023 (43x below its own 0.10 bar; offoff/wd0_off 4.18/9.71); depth failure is LEVEL not steering; S23 withdrawn; labels fixed in metrics_real.py only`
- `h49` [[h49_read_length_as_a_fixed_coefficient_physi|read_length as a fixed-coefficient physical exposure term completes the size-factor offset]] — *idea*
- `h50` [[h50_an_explicit_per_assay_output_factor_loca|An explicit per-assay output factor (location+scale on eta, dispersion offset on n) absorbs the oracle per-assay scale error]] — *idea*
- `h51` [[h51_a_no_decay_parameter_group_decoupled_ada|A no-decay parameter group (decoupled AdamW, trunk-only weight decay) is the production-safe equivalent of h47's global wd=0]] — *idea*
- `h52` [[h52_live_decoder_film_init_xavier_n_0_0_1_re|Live decoder-FiLM init (xavier + N(0,0.1)) removes the second half of the annihilation mechanism h47 left in place]] — *idea*
- `h53` [[h53_metadata_steered_dispersion_read_length_|Metadata-steered dispersion (read_length + imputation-context, NOT told-depth) is the clean PI-thesis test on a channel with no arithmetic shortcut]] — *idea*
- `h54` [[h54_conditioning_dropout_manufactures_the_mi|Conditioning dropout manufactures the missing 'use the metadata' gradient and yields a free inference-time guidance dial]] — *idea*
- `h55` [[h55_a_grouped_decoder_trunk_groups_a_optiona|A grouped decoder trunk (groups=A), optionally with per-deconv per-assay FiLM, gives the revived conditioning genuine per-assay channels]] — *idea*
- `h56` [[h56_adding_sequencing_platform_lab_with_oov_|Adding sequencing_platform + lab (with OOV/MISSING tokens) supplies the only NEW covariates that are identifiable on this slice (= h44, unblocked by h47)]] — *idea*
- `h57` [[h57_a_re_selected_biosample_panel_is_the_onl|A re-selected biosample panel is the only lever that makes run_type (and depth-vs-read_length attribution) learnable at all]] — *idea*
- `h58` [[h58_downsample_input_dsf_augmentation_plus_a|Downsample-input DSF augmentation plus a thinning-consistency term trains the untrained upward-depth regime]] — *idea*
<!-- crux:ledger:end -->
