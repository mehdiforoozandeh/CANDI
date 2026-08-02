---
type: wiki
title: Multi-task optimisation and gradient conflict
summary: Why several heads on one trunk can be worse than separate models — conflicting gradients, magnitude imbalance, and the three families of fix: gradient surgery, gradient-magnitude balancing, and not sharing at all.
category: concept
sources: raw/yu-2020-pcgrad.pdf, raw/chen-2018-gradnorm.pdf, raw/standley-2020-task-grouping.pdf, raw/kendall-2017-kendall-uw.pdf
created: 2026-08-01T18:14:33
updated: 2026-08-01T18:14:33
---

# Multi-task optimisation and gradient conflict

Multi-task learning is usually justified by transfer between related tasks, and this literature is
mostly about when that justification fails. The failures have distinguishable causes, and the fix
depends on which one you have.

## The diagnosis: conflict, curvature, magnitude

`raw/yu-2020-pcgrad.pdf` gives the sharpest account. Two task gradients **conflict** when they
point away from one another — negative cosine similarity. Conflict alone is survivable; the paper
argues it becomes damaging when it coincides with two other conditions, which together it calls
the **tragic triad**:

1. conflicting gradients (negative cosine similarity),
2. **high positive curvature**, and
3. a **large difference in gradient magnitudes**.

With all three, the multi-task update overestimates the improvement on the high-curvature
direction and underestimates the damage, and the large-magnitude task's gradient dominates the
small one. The proposed fix, **gradient surgery** (PCGrad), is local and cheap: project a task's
gradient onto the **normal plane** of any other task's gradient it conflicts with, removing the
interfering component while keeping the rest.

This gives the concrete diagnostic to run before changing anything: log **pairwise cosine
similarity between per-head gradients** and their relative magnitudes. Both quantities are
computable during a normal training run, and conflict without magnitude imbalance is a much
weaker problem than the triad.

## Balancing magnitudes: GradNorm

`raw/chen-2018-gradnorm.pdf` targets condition (3) directly, on the grounds that hand-tuned loss
weights are both unprincipled and exponentially expensive to search as tasks are added. GradNorm
dynamically tunes the loss weights so that the **gradient magnitudes** each task contributes are
balanced, with a single **asymmetry hyperparameter α** controlling how strongly tasks that are
learning faster get pulled back. The reported result is the useful one: it matches or surpasses
exhaustive grid search over static weights while requiring a few runs rather than a grid.

Contrast this with `raw/kendall-2017-kendall-uw.pdf`, which is the other route to the same
outcome and is already used in this vault's [[uncertainty-calibration]] page. Kendall weights each
task by a **learned per-task log-variance**, `L = Σ_t exp(−s_t)·L_t + ½·s_t`. The difference is
the target: Kendall balances by **loss scale** (how noisy each task is), GradNorm balances by
**gradient magnitude** (how much each task moves the shared trunk). These come apart whenever a
task has small loss values but large gradients, or vice versa — which is exactly what happens when
several heads use different likelihood families, since an NLL's scale is set by its distribution's
normalising constant and its gradient scale is not.

## The option of not sharing

`raw/standley-2020-task-grouping.pdf` asks the prior question. Multi-task learning saves
computation at inference because only one network is evaluated, but "this often leads to inferior
overall performance as task objectives can compete" — so the useful framing is **assignment**:
given a compute budget, which tasks should share a network and which should be split out?

Its most transferable finding is a negative one, and it is worth knowing before importing intuition
from pretraining: multi-task affinity and **transfer-learning affinity are uncorrelated**
(Pearson r = −0.12, p = 0.74 in its setting), and its two 3-D tasks — depth estimation and surface
normal prediction — did **not** group well together despite being the pair with the highest transfer
affinity in prior work. Tasks that transfer well are not thereby tasks that train well together.
So "these heads predict related quantities" is not evidence they belong on one trunk; that has to
be measured.

## What this implies for several distributional heads on one target

The case where heads predict different *views* of the same underlying quantity — a count, a
transformed signal, a binary label at the same position — is not directly covered by any of these
papers, which study genuinely different tasks. Two things do transfer:

- The heads' gradients are **not** guaranteed to agree merely because their targets are
  deterministically related. A negative-binomial NLL, a Gaussian NLL on a transformed scale, and a
  Bernoulli cross-entropy have different gradient magnitudes at the same position by construction,
  which supplies condition (3) of the triad for free.
- Balancing by loss scale (Kendall) and balancing by gradient magnitude (GradNorm) will disagree
  here, and the second is the one the triad implicates.

The complaint in [[sequence-conditioned-epigenome-models]] — that multitask sequence models
underperform single-task models on cell-type-specific accessibility because cell-type-specific
features are under-represented — is the same phenomenon seen from the capacity side: when a shared
trunk is optimised by a summed objective, capacity flows to whatever the tasks agree on, which is
the [[average-activity-baseline]] component.

## See also

Related:: [[uncertainty-calibration]], [[imbalance-aware-objectives]], [[sequence-conditioned-epigenome-models]], [[training-mechanics]], [[average-activity-baseline]], [[regression-likelihoods]]
