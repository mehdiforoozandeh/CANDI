---
type: wiki
title: FiLM conditioning
summary: Feature-wise Linear Modulation: conditioning a network by predicting per-channel scale and shift from side information, applied throughout the network rather than at the input.
category: method
sources: raw/perez-2018-film.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# FiLM conditioning

FiLM's contribution is showing that a conditioning signal is far more effective when it modulates *every* layer's features than when it is concatenated to the input once.

## The mechanism

A FiLM layer applies a **feature-wise affine transformation** to a network's intermediate activations, with the scale γ and shift β produced by a separate network from the conditioning information (`raw/perez-2018-film.pdf`):

    FiLM(F_c | γ_c, β_c) = γ_c · F_c + β_c

per feature map / channel *c*. The conditioning generator is arbitrary — in the original work an RNN reading a natural-language question modulates a CNN processing an image.

## Results and properties

On CLEVR visual reasoning, FiLM **halves** the state-of-the-art error. The paper's four claims are that FiLM layers (1) halve SOTA error on CLEVR, (2) modulate features coherently, (3) are robust to ablations and architectural modifications, and (4) **generalise well to challenging new data from few examples or even zero-shot**.

The properties that make it attractive as a general conditioning primitive:

- **Cheap.** Two parameters per channel per conditioned layer, regardless of feature-map size.
- **Placeable anywhere.** Because it is a per-channel affine map, it can be inserted after any layer producing channelled features — convolutional or otherwise — so conditioning can act at every depth rather than only at the input.
- **Multiplicative.** The γ term lets the conditioning *gate* features (suppressing or amplifying channels), which additive-only conditioning (concatenation, bias) cannot do.
- **Zero-shot capable.** Claim (4) is the one that matters for conditioning on continuous experimental covariates: unseen combinations of conditioning values produce sensible modulations because γ and β are predicted by a network rather than looked up.

## Relation to normalisation layers

FiLM is the general form of the conditional-affine trick that also appears as conditional batch norm and, later, as adaptive layer norm (adaLN) in conditional transformers — see [[transformers-and-positional-encoding]]. The distinguishing feature is that FiLM separates the modulation from any normalisation statistics, so it can be applied independently of where normalisation happens.

## See also

Related:: [[transformers-and-positional-encoding]], [[masked-self-supervised-learning]], [[sequence-conditioned-epigenome-models]]
