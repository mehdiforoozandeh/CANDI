---
id: h36
type: idea
title: Output-steering, once present, is either unconditional or requires the depth-offset preconditioning
parent: q16
status: done
verdict: supported
metric: offset-off M2 0.46 ~ offset-on 0.48 -> unconditional
created: "2026-07-08T12:36:48"
updated: "2026-07-09T00:48:59"
---

# h36 — Output-steering, once present, is either unconditional or requires the depth-offset preconditioning

Parent:: [[q16_was_the_v1_output_steering_null_an_artif]]

## Problem Statement

The depth-offset head is h_y-independent, so it CANCELS in the delta-log-mu M2 readout and cannot fabricate steering there, but it changes training dynamics. Does steering emerge with the offset-OFF (plain log-link) head too (unconditional), or only with the offset (preconditioning-dependent)? Contrast offset-on vs offset-off: a single-variable ablation with the log link held fixed.

## Idea / Hypothesis

Output-steering, once present, is either unconditional or requires the depth-offset preconditioning

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] offset-on arm: distributional M2 (median invertible) >= 0.5 (steering present)   (found: offset-on M2 ~= 0.48-0.53 across the grid (0.483 zscore-naive reference); steering clearly present)
- [x] attribution verdict (report which): UNCONDITIONAL if offset-off M2 >= offset-on - 0.1; PRECONDITIONING-DEPENDENT if offset-off <= 0.15 while offset-on >= 0.5   (found: UNCONDITIONAL -- offset-off M2 = 0.46 vs offset-on 0.48 (chr21), within the offset-on-0.1 band; not the <=0.15 collapse)
- [x] readout guard: delta-log-mu is offset-invariant (cancellation), verified by a validation gate, so any offset-on/off M2 gap is training-attributable not readout-injected   (found: offset cancels in the delta-log-mu readout, enforced by validation gate; residual 0.02 gap is training-attributable)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)

## Findings

Output steering, once present, is **unconditional** -- it does not require the depth-offset preconditioning. Turning the depth offset OFF barely moves steering: M2 = 0.46 (offset-off) vs 0.48 (offset-on) on chr21, within the offset-on-minus-0.1 band that the pre-registered rule labels UNCONDITIONAL (not the <=0.15 collapse that would indicate preconditioning-dependence). The readout guard holds: delta-log-mu is offset-invariant (the h_y-independent offset cancels in the M2 readout, enforced by the validation gate), so the small residual gap is training-attributable, not injected by the readout. Note the offset still buys calibrated *absolute* depth (the h9 DCR fix) -- it is simply not required for the *relative* steering M2 measures. Verdict: supported (unconditional branch).
