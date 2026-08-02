---
type: wiki
title: Imbalance-aware and cost-sensitive objectives
summary: Focal loss, effective-number reweighting and hard-example mining — the three families of answer to a loss dominated by an easy majority class, and how to tell whether reweighting helped.
category: concept
sources: raw/lin-2017-focal-loss.pdf, raw/cui-2019-class-balanced-loss.pdf, raw/shrivastava-2016-ohem.pdf, raw/saito-2015-precision-recall-plot.xml
created: 2026-08-01T18:14:33
updated: 2026-08-01T18:14:33
---

# Imbalance-aware and cost-sensitive objectives

Three distinct mechanisms get called "handling class imbalance" and they intervene at different
places — on the per-example loss, on the per-class weight, and on which examples are used at all.
Choosing between them is a question about *why* the majority dominates, not just *that* it does.

## Reweighting the example: focal loss

`raw/lin-2017-focal-loss.pdf` addresses the case where the majority class is not merely numerous
but **easy**, so that a huge number of well-classified examples drown the gradient from the few
that are wrong. Its remedy modulates cross-entropy by how confident the model already is:

    FL(p_t) = −(1 − p_t)^γ · log(p_t)

γ = 0 recovers cross-entropy; the paper settles on **γ = 2**, usually with an α-balanced variant
that also carries a per-class weight. The asymmetry is the point, and the numbers are worth
quoting because they show how aggressive it is: at γ = 2 an example at p_t = 0.9 receives
**100× lower loss** than under CE, and at p_t ≈ 0.968, **1000× lower** — while a misclassified
example (p_t ≤ 0.5) is scaled down **by at most 4×**. Nearly all the down-weighting lands on
examples the model has already solved.

Two of the paper's own caveats matter as much as the formula. It reports that focal loss
outperforms **both** sampling heuristics **and** hard-example mining, i.e. the three families
below are alternatives rather than complements. And it states plainly that "the exact form of the
focal loss is not crucial" — other instantiations achieve similar results, so the mechanism
(down-weight the confident) is the transferable idea, not the particular exponent.

## Reweighting the class: effective number of samples

`raw/cui-2019-class-balanced-loss.pdf` attacks a different failure: inverse-frequency weighting
over-corrects, because the *n*-th example of a class adds much less information than the first.
It models each sample as covering a small neighbourhood rather than a point, so the information a
class actually contributes saturates:

    E_n = (1 − β^n) / (1 − β),    β ∈ [0, 1)

and weights each class by 1/E_n. β → 0 gives no reweighting, β → 1 approaches plain
inverse-frequency, and intermediate β interpolates. This is the principled version of "divide by
the class count", and the reason to prefer it is that plain inverse-frequency reweighting assigns
enormous weight to classes that are rare *and* redundant, which is a common way to make a
minority class train worse rather than better.

## Changing which examples are used: hard-example mining

`raw/shrivastava-2016-ohem.pdf` takes the third route — not reweighting at all, but **selecting**.
Its motivating complaint is procedural and applies broadly: detector training "still includes many
heuristics and hyperparameters that are costly to tune", such as fixed foreground/background
sampling ratios. OHEM removes them by selecting high-loss examples online, so the sampling
distribution is determined by the model's current errors rather than by a hand-set ratio.

The trade-off is the one every hard-example scheme carries: high loss and *informative* are not
the same thing. Label noise, mislabelled positives, and irreducibly ambiguous positions all
present as high loss, so mining concentrates capacity on them. Where the minority class is
partly a **noisy proxy** — as peak labels are, see [[peak-calling-and-signal-tracks]] — this
argues for the smooth down-weighting of focal loss over hard selection.

## Measuring whether it worked

Reweighting is easy to fool yourself about, because the standard metric is insensitive in exactly
this regime. `raw/saito-2015-precision-recall-plot.xml` shows the precision–recall plot is more
informative than the ROC plot for binary classifiers on **imbalanced** datasets. The mechanism:
ROC's false-positive rate divides by the (large) negative count, so a change that matters
enormously to a user of the predictions — most predicted positives being wrong — moves FPR, and
therefore AUROC, only slightly. Precision conditions on the predicted positives instead. Report
AUPRC / average precision when positives are rare, and treat an AUROC improvement under
reweighting as unverified until the PR curve agrees. See [[imputation-evaluation-measures]].

## Reading these together

The three mechanisms answer different diagnoses, and the diagnosis is the choice:

| If the majority is… | the fix is… | via |
|---|---|---|
| numerous **and easy** | down-weight by model confidence | focal loss |
| numerous **and redundant** | down-weight by saturating class information | effective number |
| numerous **and uninformative** | stop sampling it | hard-example mining |

A foreground/background reweighting scheme with a fixed weight and a minimum-fraction floor sits
outside all three: it is a *static* class weight, so it corrects frequency but not easiness or
redundancy. That is the gap focal loss was invented to close, and the reason a static weight
often moves foreground and background metrics in opposite directions without improving either
diagnosis — the weight cannot tell an easy background bin from a hard one.

## See also

Related:: [[imputation-evaluation-measures]], [[peak-calling-and-signal-tracks]], [[multi-task-optimization]], [[training-mechanics]], [[uncertainty-calibration]]
