---
type: wiki
title: Set-conditioned modelling and structured missingness
summary: Attention over a variable-size observed set rather than a fixed tensor with holes — Set Transformer, Conditional Neural Processes, and why block-structured missingness breaks MCAR/MAR assumptions.
category: concept
sources: raw/lee-2018-set-transformer.pdf, raw/garnelo-2018-cnp.pdf, raw/mitra-2023-structured-missing.pdf
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Set-conditioned modelling and structured missingness

The right abstraction for "which experiments exist for this sample" is a **set of observations**, not a fixed-width vector with missing entries — and the two framings lead to genuinely different architectures.

## Permutation-invariant set encoding

`raw/lee-2018-set-transformer.pdf` addresses tasks defined on **sets of instances**, where the answer must not depend on element order. Its contribution is an attention-based module — an encoder and a decoder both built from attention — designed to model **interactions among set elements**, unlike pooling-based set encoders (deep sets) which aggregate elements independently and can only combine them after the fact. To control the quadratic cost of self-attention over a set, it introduces an **inducing-point** scheme borrowed from sparse Gaussian processes, giving an attention block whose cost is linear in set size.

The building blocks are the **Set Attention Block (SAB)** — self-attention among set elements — and its induced variant (ISAB), plus a pooling-by-multihead-attention decoder. Any architecture whose layer count is expressed in "SAB layers" is using this vocabulary.

## Conditional Neural Processes

`raw/garnelo-2018-cnp.pdf` frames the same structure as a learned stochastic process. A CNP consumes a **context set** of observed (input, output) pairs, aggregates it into a fixed-size representation, and then predicts an output distribution at arbitrary **target** inputs conditioned on that representation. Learning happens in two phases: the model learns the statistics of a generic domain from a large training set without committing to a specific task, then specialises at test time by conditioning on whatever context it is given.

Two properties matter for a model with missing observations:
- The **context set is variable-size and unordered** — a sample with two observed assays and one with thirty are the same kind of input, not special cases.
- The output is a **distribution**, not a point estimate, and its width naturally reflects how much context was supplied.

This is the formal template for "decode a queried target from an arbitrary set of observed inputs," and any decoder that takes a query alongside a context representation is an instance of it.

## Structured missingness

`raw/mitra-2023-structured-missing.pdf` names the regime that MCAR/MAR tooling does not cover. Standard missing-data theory assumes values are missing at random, conditionally on observed data. But in large heterogeneous datasets, missing values exhibit **association or structure** — they arrive in blocks, and whether a value is missing depends on the variables themselves.

The paper's argument, framed as a set of grand challenges, is that structured missingness is a **fundamental hindrance to machine learning at scale** and has not been systematically addressed. Its relevance here is direct: which experiments were performed in a given biosample is decided by lab priorities, assay cost and prior interest — not by a random mask. That means:
- a model trained with **random masking** is trained on a missingness distribution that does not match deployment;
- the observed set and the target are **not independent** — well-characterised biosamples have systematically different assay subsets than poorly characterised ones (see [[cross-cell-type-generalization-pitfall]]);
- imputation accuracy measured under random held-out masks may not transfer to the blocks that are actually missing.

## See also

Related:: [[masked-self-supervised-learning]], [[epigenome-imputation]], [[query-decoders-and-conditional-computation]], [[jepa-and-collapse-prevention]], [[cross-cell-type-generalization-pitfall]]
