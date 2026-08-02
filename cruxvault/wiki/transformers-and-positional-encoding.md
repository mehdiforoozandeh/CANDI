---
type: wiki
title: Transformers and positional encoding
summary: Self-attention as the backbone architecture, and RoPE as the rotation-matrix scheme that injects relative position directly into the attention product.
category: method
sources: raw/vaswani-2017-attention-is-all-you-need.pdf, raw/su-2021-roformer-rope.pdf, raw/devlin-2019-bert.pdf, raw/ji-2021-dnabert.html, raw/avsec-2021-enformer.xml, raw/shaw-2018-shaw-relpos.pdf, raw/ho-2019-axial.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
---

# Transformers and positional encoding

Attention is permutation-invariant, so every transformer needs a positional scheme; RoPE is the one that makes relative position fall out of the dot product itself rather than being added to the inputs.

## The transformer

`raw/vaswani-2017-attention-is-all-you-need.pdf` introduces an architecture "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely." Each attention layer transforms a position by a weighted sum over the representations of all other positions, with weights determined by query–key similarity; multi-head attention runs several such maps in parallel. The practical consequence relevant here is **parallelism**: unlike recurrence, all positions are computed simultaneously, so the architecture scales with sequence length in a way RNNs do not.

## Rotary position embedding (RoPE)

`raw/su-2021-roformer-rope.pdf` proposes encoding **absolute position with a rotation matrix** applied to the query and key vectors, such that their inner product depends only on the **relative** offset between positions. The paper's theoretical framing is that relative position can be formulated naturally via vector products in self-attention, with absolute position carried by the rotation.

Properties that follow: relative-position dependency is explicit in the attention score rather than injected as an additive input embedding; the scheme requires no learned position table; and it decays sensibly with distance. The paper's own adoption claim is narrow — RoPE "is already integrated into Huggingface". (Its subsequent spread to most transformer implementations, `x-transformers` among them, is an implementation-landscape observation, not a claim from `raw/su-2021-roformer-rope.pdf`; no source in `raw/` establishes it.)

## Masked bidirectional pre-training

`raw/devlin-2019-bert.pdf` (BERT) pre-trains deep **bidirectional** representations by jointly conditioning on left and right context in all layers, then fine-tunes with a single added output layer. Its significance for this wiki is architectural rather than linguistic: it establishes the encoder-only transformer trained by masked reconstruction as a general-purpose representation learner. See [[masked-self-supervised-learning]].

## In genomics

- **DNABERT** (`raw/ji-2021-dnabert.html`) transplants BERT to DNA by k-merising sequence, giving pre-trained representations that transfer across genomic tasks.
- **Enformer** (`raw/avsec-2021-enformer.xml`) uses **11 transformer blocks** on top of 7 convolutional blocks, and attributes its long-range **reach** — distal elements "up to 100 kb away", against Basenji2's and ExPecto's 20 kb — directly to attention's ability to move information between distal elements — the mechanism by which enhancers can influence a distant promoter in the model. See [[sequence-conditioned-epigenome-models]].

The convolution-then-attention pattern in Enformer is the standard compromise for genomic sequence: convolutions reduce base-pair resolution to a manageable token count, attention then operates over the reduced sequence.

## Learned relative position bias

`raw/shaw-2018-shaw-relpos.pdf` predates RoPE and remains a widely used default. Instead of adding absolute position embeddings to the inputs, it extends self-attention to consider representations of the **relative position** — the signed distance between query and key — typically as a learned bias added to the attention logits, with distances clipped beyond a maximum so the table stays finite. Reported gains over absolute position embeddings: +1.3 BLEU on WMT14 EN-DE, +0.3 on EN-FR.

Contrast with RoPE. A learned relative bias is **additive on the logits** and each distance bucket has free parameters, so it can represent arbitrary distance-dependent preferences but must learn them from data and cannot extrapolate past the clip. RoPE is **multiplicative on the query/key vectors** with a fixed functional form, so it has no learned parameters and generalises smoothly with distance. Neither dominates; the learned bias is often stronger at fixed, modest sequence lengths, and RoPE is preferred when length extrapolation matters.

## Axial attention

`raw/ho-2019-axial.pdf` addresses data organised as a **high-dimensional tensor** rather than a flat sequence. Full self-attention over an *n*-dimensional tensor with N total elements costs O(N²), which is prohibitive once N is a product of several axes. Axial attention instead applies attention **along one axis at a time**, keeping the other axes independent — reducing cost to O(N·√N) for a 2-D tensor, or more generally to attention over each axis's length rather than their product.

The paper's claim is that this is achieved **without losing full expressiveness** over the joint distribution — stacking attention over each axis in turn allows information to flow between any two positions — and without the implementation compromises other efficient-attention schemes require.

For data indexed by (genomic position, assay), this is the natural factorisation: attention along the position axis models spatial structure within a track, attention along the assay axis models cross-assay correlation at a locus, and stacking them gives interaction between the two at a fraction of the cost of attending over all (position, assay) pairs jointly.

## See also

Related:: [[masked-self-supervised-learning]], [[sequence-conditioned-epigenome-models]], [[film-conditioning]], [[edice]]
