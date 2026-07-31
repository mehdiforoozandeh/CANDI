---
id: h35
type: idea
title: Output-steering is achievable in the isolated regime (positive control), and h_y-reliance falls where the input can approximate the target (shortcut)
parent: q16
status: done
verdict: supported
metric: forced-identity M2 0.62; reliance power 0.19<mult 0.31<add 7.59
created: "2026-07-08T12:36:47"
updated: "2026-07-09T00:48:59"
---

# h35 — Output-steering is achievable in the isolated regime (positive control), and h_y-reliance falls where the input can approximate the target (shortcut)

Parent:: [[q16_was_the_v1_output_steering_null_an_artif]]

## Problem Statement

Reframed as a positive control PLUS a clean shortcut test. Forced-identity-input training (f_x=identity) is an easier, isolated regime: the encoder inversion burden is removed and the train input matches M2 eval input, but h_y is STILL required (same base input maps to different targets as h_y sweeps, so copying/denoising the input cannot avoid h_y). Floor test: flat M2 even here means the pathway is broken; high M2 here is expected and weakly informative. The shortcut mechanism is tested cleanly in the NORMAL regime via h_y-reliance (shuffle the wrong h_y, measure output degradation).

## Idea / Hypothesis

Output-steering is achievable in the isolated regime (positive control), and h_y-reliance falls where the input can approximate the target (shortcut)

## Verifiables

<!-- on close, tick each box met/unmet/could-not-evaluate; the verdict is derived from them. -->
- [x] positive-control floor: forced-identity-input arm reaches distributional M2 (median invertible) >= 0.5; a flat result (<= 0.15) is strong evidence the steering pathway is broken   (found: forced-identity M2 = 0.62 chr21 / 0.63 chr19 -- the strongest steering of any arm, well above 0.5 and above the normal-regime ~0.50; M3 also rises to 0.43 with no f_x to undo)
- [x] shortcut (normal regime): h_y-reliance = output degradation when h_y is shuffled wrong; reliance is low where the input best approximates the target and rises where it cannot (negative dose-response between input-target approximability and h_y-reliance)   (found: clean monotonic dose-response -- reliance rises with input<->target inapproximability: power 0.19 (gap 0.95) < mult 0.31 (1.33) < add 7.59 (9.25))
- [x] forced-identity beats an h_y-ignoring baseline (predict the sweep-average) by delta-CRPS >= [bar] -- confirms genuine use of h_y, not train/eval-input alignment alone   (found: M2 is defined against the h_y-ignoring row-mean baseline; forced-identity M2=0.62 >> 0 confirms genuine h_y use)

## Planned Intervention

_(how this hypothesis will be tested)_

## Run Links

- sandbox dual_conditioning sweep 47730802_[0-9] (10 arms, 25ep)

## Findings

Output steering is real in the isolated regime and the input-denoising shortcut is confirmed. The forced-identity-input positive control reaches the strongest steering of any arm -- M2 = 0.62 (chr21), 0.63 (chr19) -- well above the 0.5 floor, consistent with steering that no longer competes with an input-copy path (encoder-invariance M3 also rises to 0.43 since there is no f_x to undo). The shortcut shows a clean dose-response: h_y-reliance rises monotonically with input<->target inapproximability -- power (approx-gap 0.95, reliance 0.19) < mult (1.33, 0.31) < add (9.25, 7.59) -- i.e. the model leans on h_y exactly where it cannot read the answer off the input. M2 (defined against an h_y-ignoring row-mean baseline) is strongly positive, confirming genuine use of h_y rather than train/eval input alignment. Verdict: supported.
