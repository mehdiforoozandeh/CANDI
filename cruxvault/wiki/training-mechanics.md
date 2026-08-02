---
type: wiki
title: Training mechanics
summary: The optimiser, schedule and stability choices that have derivations behind them — decoupled weight decay, cosine annealing, AdaMax's ∞-norm denominator, Muon's orthogonalised updates, and why clipping accelerates rather than merely guards.
category: concept
sources: raw/loshchilov-2017-adamw.pdf, raw/loshchilov-2016-sgdr.pdf, raw/aygun-2025-era.pdf, raw/kingma-2015-adam-adamax.pdf, raw/liu-2025-muon-scalable.pdf, raw/zhang-2020-gradient-clipping.pdf, raw/zhang-2020-heavy-tailed-noise.pdf
created: 2026-07-31T23:27:28
updated: 2026-08-01T18:14:33
---

# Training mechanics

Two of the most-copied lines in any training script come from the same two authors, and the first of them is a correction to a bug that most implementations had been shipping for years.

## Decoupled weight decay

`raw/loshchilov-2017-adamw.pdf` establishes that **L2 regularisation and weight decay are equivalent for plain SGD** (up to a learning-rate rescaling) but **not for adaptive methods like Adam**. In Adam, an L2 penalty added to the loss is scaled by the same per-parameter adaptive denominator as the gradient, so parameters with large historical gradients are decayed *less* — the opposite of the intent. AdamW decouples the decay: it is applied directly to the weights, outside the adaptive rescaling.

Consequences that matter in practice:
- The effective regularisation strength under Adam+L2 depends on gradient history, which makes the hyperparameter non-transferable across models and schedules.
- Parameters whose gradients are small relative to their decay — those receiving weak learning signal — are the ones most affected. Under coupled L2, a parameter with a persistently tiny gradient can be driven toward zero by the decay term rather than by the objective.
- Note what the paper does *not* argue. Excluding norms, biases and embedding-like parameters from the decay group is near-universal practice, but it is **not** in `raw/loshchilov-2017-adamw.pdf` — that paper argues the opposite direction, that decoupling is what makes a single global rate meaningful ("decoupled weight decay regularizes all weights with the same rate λ"). The no-decay-group convention rests on a separate argument from a different literature, not yet sourced here. Treat it as convention.

## Warm restarts and cosine annealing

`raw/loshchilov-2016-sgdr.pdf` (SGDR) proposes periodically restarting SGD with a **warm restart** — resetting the learning rate to a high value and annealing it down on a cosine curve — rather than decaying monotonically. Restart techniques are standard in gradient-free optimisation for multimodal objectives; SGDR imports the idea to gradient-based training to improve **anytime** performance.

The lasting practical residue is the **cosine annealing curve itself**, which is now routinely used without restarts and typically preceded by a short linear warmup. Note this is a different thing from what the paper argued for: single-cycle cosine is SGDR's schedule shape without its restart mechanism.

## The optimisers actually in use

The decoupling argument above is made about **Adam**, and it does not transfer unchanged to every
adaptive method — which matters, because Adam is often not the one running.

`raw/kingma-2015-adam-adamax.pdf` introduces Adam and, in the same paper, **AdaMax** — "a variant
of Adam based on the infinity norm." The difference is the denominator. Adam accumulates an
exponential moving average of **squared** gradients (an L2 norm over history); AdaMax replaces it
with a running **max** of gradient magnitudes, an L∞ norm:

    u_t = max(β₂ · u_{t−1}, |g_t|)

Two consequences follow for the weight-decay argument. First, a running maximum is **monotone
non-increasing only by decay**, so a parameter that once saw a large gradient keeps a large
denominator long after the gradient shrinks — the history-dependence AdamW complains about is
present in AdaMax too, with a different memory. Second, the max is not divided by a sum of
squares, so the denominator is not driven toward zero by a long run of small gradients the way
Adam's can be; the "persistently tiny gradient" case above therefore behaves differently under
AdaMax than the AdamW analysis assumes. *This is a mechanism, not a measured result — no source in
`raw/` compares coupled-vs-decoupled decay under AdaMax specifically.* Treat any transfer of the
AdamW argument to AdaMax as a hypothesis.

**Muon** occupies a different design point altogether. `raw/liu-2025-muon-scalable.pdf` reports on
scaling the Muon optimiser, which is built on **matrix orthogonalisation** of the update rather
than per-coordinate rescaling: the update for a weight *matrix* is orthogonalised before being
applied, treating the layer as a matrix instead of a bag of scalars. The paper identifies two
techniques as crucial for making it work at scale — **adding weight decay** and **carefully
adjusting the per-parameter update scale** — and reports ≈**2× computational efficiency versus
AdamW** under compute-optimal training, with a 3B/16B MoE model trained on 5.7T tokens as the
demonstration. Note that weight decay is reported here as *necessary for scaling*, not as
optional regularisation, which is the opposite of the usual framing.

## Stability

`raw/zhang-2020-gradient-clipping.pdf` explains why clipping is more than a safety valve. Standard
convergence analysis assumes globally Lipschitz gradients; the paper replaces this with a
**relaxed smoothness** condition in which the local smoothness may grow with the gradient norm,
and proves that under it, **gradient descent with a fixed step size can be arbitrarily slower than
clipped gradient descent**. Clipping is therefore an accelerator in this regime, not merely a
guard — which is consistent with removing it degrading training rather than just permitting
occasional spikes.

`raw/zhang-2020-heavy-tailed-noise.pdf` supplies the companion result and the diagnostic. It gives
empirical and theoretical evidence that a **heavy-tailed distribution of stochastic-gradient
noise** is one cause of SGD's poor performance relative to adaptive methods on attention models,
and shows clipping is key to addressing it (developing a coordinate-wise adaptive clipping scheme,
ACClip). The practical reading: if pre-clip gradient-norm logs show a heavy right tail rather than
occasional isolated spikes, that is the regime these two papers describe, and both the choice of
an adaptive optimiser and the load-bearing role of clipping follow from the same underlying fact
rather than being two independent tuning decisions.

## Automating the search over configurations

`raw/aygun-2025-era.pdf` (ERA — Empirical Research Assistance) reframes writing scientific software as a **scorable task**: search for a program whose output maximises a quality metric. It drives a tree search with an LLM that rewrites whole candidate programs, allowing domain knowledge and external research ideas to be injected as part of the rewrite. The framing draws on genetic programming, AutoML, and LLM-plus-search work; the distinguishing element is LLM-driven **whole-program rewriting** rather than parameter mutation.

Its relevance to this page is that the mechanics above — optimiser, decay grouping, schedule, warmup length — are exactly the axes such a search operates over, and a search harness makes the choice of them empirical rather than inherited.

## See also

Related:: [[jepa-and-collapse-prevention]], [[count-distributions-for-sequencing-data]], [[uncertainty-calibration]], [[query-decoders-and-conditional-computation]], [[multi-task-optimization]], [[imbalance-aware-objectives]], [[regression-likelihoods]]
