---
type: wiki
title: FiLM conditioning
summary: Feature-wise Linear Modulation: conditioning a network by predicting per-channel scale and shift from side information, applied throughout the network rather than at the input.
category: method
sources: raw/perez-2018-film.pdf, raw/peebles-2022-dit-adaln.pdf, raw/ho-2022-cfg.pdf, raw/karras-2019-stylegan2.pdf, raw/dieng-2018-dieng-skip.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
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

## adaLN-Zero — the transformer-native form

`raw/peebles-2022-dit-adaln.pdf` (DiT) runs the ablation that FiLM's original paper could not: how best to inject a small conditioning vector into a **transformer**. It compares in-context conditioning (append the conditioning as an extra token), cross-attention, adaptive layer norm (adaLN), and **adaLN-Zero**, and finds adaLN-Zero clearly best at equal compute.

adaLN regresses the layer-norm scale and shift (γ, β) from the conditioning vector — FiLM applied at the normalisation layer. **adaLN-Zero** adds a third regressed parameter, a per-block output gate α, and **initialises the gate-producing layer to zero**, so every conditioned block starts as the identity and the network learns how much conditioning to admit rather than being perturbed by it at initialisation. This is why zero-initialised conditioning layers are standard practice, and why the norm of the learned gate is a meaningful diagnostic: it measures how much conditioning the block has actually chosen to use.

## Conditioning dropout and guidance

`raw/ho-2022-cfg.pdf` (classifier-free guidance) provides two things at once from a single training change. During training, the conditioning is **randomly dropped** with some probability, so one network learns both the conditional and unconditional models. At inference, the two are combined by extrapolation:

    ε̃ = (1 + w) · ε(z, c) − w · ε(z)          [Eq. 6, the paper's own form]
       = uncond + (1 + w) · (cond − uncond)    [rearranged]

Mind the convention: under `raw/ho-2022-cfg.pdf`'s own variable, **`w = 0` is the unguided model** ("w = 0.0 refers to non-guided models", Table 1) and guidance begins at `w > 0`. The rearranged form widely used in diffusion tooling writes a guidance *scale* `s = 1 + w`, where `s = 1` is unguided and `s > 1` amplifies. The two differ by exactly one, and implementing the paper's `w` against the tooling's threshold double-counts the guidance. The training-time dropout is the more fundamental half: it forces the network to function *without* the conditioning, which means the conditional pathway must carry information the unconditional one lacks — a direct structural remedy for a conditioning input that is otherwise redundant with the data. The trade-off the guidance weight controls is fidelity against diversity.

## Normalisation can destroy what conditioning writes

`raw/karras-2019-stylegan2.pdf` reports a mechanism relevant to any architecture that normalises after conditioning. StyleGAN's instance normalisation destroys "information found in the magnitudes of features relative to each other," producing characteristic artefacts. The fix was to move the modulation off the normalised activation path and onto the **weights** — demodulating the convolution weights instead of the activations.

The general lesson: if a conditioning signal writes a per-channel scale and a subsequent normalisation layer re-standardises across those channels, the relative magnitudes the conditioning just set can be partially undone. Whether conditioning is applied before or after normalisation is therefore not a cosmetic choice.

## Depth of injection

`raw/dieng-2018-dieng-skip.pdf` addresses the complementary question of *where* conditioning should enter. In the VAE setting it shows that adding **skip connections from the latent to every layer of the decoder** increases the mutual information between the latent and the output, and it proves this raises the dependence rather than merely appearing to. The failure it prevents — the decoder becoming powerful enough to model the data while ignoring the conditioning variable entirely — is the same failure a conditioned decoder faces.

This is direct evidence for conditioning at every layer rather than once, and it cuts against single-injection designs. What it does not settle is per-output-channel conditioning granularity, where the trade-off against parameter sharing is empirical.

## Testing whether it worked

None of the above establishes that a trained model *uses* its conditioning — see [[covariate-conditioning-and-counterfactuals]] for the ablation and steerability instruments that do, and [[query-decoders-and-conditional-computation]] for the weight-generating alternatives to feature-wise modulation.

## See also

Related:: [[transformers-and-positional-encoding]], [[masked-self-supervised-learning]], [[sequence-conditioned-epigenome-models]], [[covariate-conditioning-and-counterfactuals]], [[query-decoders-and-conditional-computation]], [[jepa-and-collapse-prevention]]
