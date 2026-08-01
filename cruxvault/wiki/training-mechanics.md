---
type: wiki
title: Training mechanics
summary: The optimiser and schedule choices that have derivations behind them — decoupled weight decay and cosine annealing with warm restarts — and automated search over the configurations they define.
category: concept
sources: raw/loshchilov-2017-adamw.pdf, raw/loshchilov-2016-sgdr.pdf, raw/aygun-2025-era.pdf
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Training mechanics

Two of the most-copied lines in any training script come from the same two authors, and the first of them is a correction to a bug that most implementations had been shipping for years.

## Decoupled weight decay

`raw/loshchilov-2017-adamw.pdf` establishes that **L2 regularisation and weight decay are equivalent for plain SGD** (up to a learning-rate rescaling) but **not for adaptive methods like Adam**. In Adam, an L2 penalty added to the loss is scaled by the same per-parameter adaptive denominator as the gradient, so parameters with large historical gradients are decayed *less* — the opposite of the intent. AdamW decouples the decay: it is applied directly to the weights, outside the adaptive rescaling.

Consequences that matter in practice:
- The effective regularisation strength under Adam+L2 depends on gradient history, which makes the hyperparameter non-transferable across models and schedules.
- Parameters whose gradients are small relative to their decay — those receiving weak learning signal — are the ones most affected. Under coupled L2, a parameter with a persistently tiny gradient can be driven toward zero by the decay term rather than by the objective.
- This is the argument for **excluding** norms, biases and embedding-like parameters from the decay group, rather than applying one global weight decay.

## Warm restarts and cosine annealing

`raw/loshchilov-2016-sgdr.pdf` (SGDR) proposes periodically restarting SGD with a **warm restart** — resetting the learning rate to a high value and annealing it down on a cosine curve — rather than decaying monotonically. Restart techniques are standard in gradient-free optimisation for multimodal objectives; SGDR imports the idea to gradient-based training to improve **anytime** performance.

The lasting practical residue is the **cosine annealing curve itself**, which is now routinely used without restarts and typically preceded by a short linear warmup. Note this is a different thing from what the paper argued for: single-cycle cosine is SGDR's schedule shape without its restart mechanism.

## Automating the search over configurations

`raw/aygun-2025-era.pdf` (ERA — Empirical Research Assistance) reframes writing scientific software as a **scorable task**: search for a program whose output maximises a quality metric. It drives a tree search with an LLM that rewrites whole candidate programs, allowing domain knowledge and external research ideas to be injected as part of the rewrite. The framing draws on genetic programming, AutoML, and LLM-plus-search work; the distinguishing element is LLM-driven **whole-program rewriting** rather than parameter mutation.

Its relevance to this page is that the mechanics above — optimiser, decay grouping, schedule, warmup length — are exactly the axes such a search operates over, and a search harness makes the choice of them empirical rather than inherited.

## See also

Related:: [[jepa-and-collapse-prevention]], [[count-distributions-for-sequencing-data]], [[uncertainty-calibration]], [[query-decoders-and-conditional-computation]]
