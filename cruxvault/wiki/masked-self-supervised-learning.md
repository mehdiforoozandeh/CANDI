---
type: wiki
title: Masked self-supervised learning
summary: Corrupt-and-reconstruct pre-training across text, images and tabular data — BERT, MAE, I-JEPA, VIME — and the design choices (mask ratio, prediction space) that decide what is learned.
category: concept
sources: raw/devlin-2019-bert.pdf, raw/he-2022-masked-autoencoders.pdf, raw/assran-2023-ijepa.pdf, raw/yoon-2020-vime.pdf, raw/chen-2025-epiagent.pdf, raw/shrikumar-2017-revcomp-parameter-sharing.pdf
created: 2026-07-31T21:26:00
updated: 2026-08-01T18:14:33
---

# Masked self-supervised learning

The four sources agree on the recipe and disagree on one question that turns out to be decisive: whether to reconstruct the input itself or its representation.

## The recipe

Hide part of the input, ask the model to recover it, and use the resulting encoder as a representation. No labels are required, so the method scales with unlabelled data.

- **BERT** (`raw/devlin-2019-bert.pdf`) — masked language modelling over text; deep bidirectional conditioning on left and right context in all layers; fine-tune with one extra output layer. Established the paradigm.
- **MAE** (`raw/he-2022-masked-autoencoders.pdf`) — mask random image patches and reconstruct the **missing pixels**. Two design choices carry the result: an **asymmetric encoder–decoder** where the encoder sees only the *visible* patches (no mask tokens) and a lightweight decoder reconstructs from latents plus mask tokens; and a **high mask ratio (≈75%)**, which is what makes the task non-trivial. Together these give ≥3× faster training and better accuracy, and let a vanilla ViT-Huge reach 87.8% on ImageNet-1K-only.
- **VIME** (`raw/yoon-2020-vime.pdf`) — the tabular case, where no spatial or semantic structure can be exploited. Adds a second pretext task alongside reconstruction: **mask-vector estimation**, i.e. predicting *which* entries were corrupted, plus a tabular data-augmentation scheme. Relevant because it shows the recipe survives the loss of domain structure, and because predicting the corruption mask is itself informative.
- **I-JEPA** (`raw/assran-2023-ijepa.pdf`) — predicts the **representations** of target blocks from a context block, rather than pixels, and uses no hand-crafted augmentations. Two masking requirements are called out as essential: target blocks must be **sufficiently large (semantic) in scale**, and the context block must be **sufficiently informative and spatially distributed**. Converges faster than pixel-reconstruction methods and yields higher-level semantics; a ViT-H/14 trains on ImageNet with 16 A100s in under 72 hours.

## The design axes

1. **Prediction space.** Pixels/tokens (MAE, BERT) versus representations (I-JEPA). Reconstructing the input forces capacity onto high-frequency detail that may be irrelevant; predicting representations concentrates it on semantics, at the cost of a collapse risk that must be managed — the failure mode, and the machinery for preventing it, are the subject of [[jepa-and-collapse-prevention]]. Reconstruction targets in input space cannot collapse, which is the underrated argument for them.
2. **Mask ratio and mask structure.** MAE's 75% and I-JEPA's large, spatially distributed blocks both say the same thing: masking must be aggressive and structured enough that the task cannot be solved by local interpolation.
3. **Asymmetry.** MAE's encoder never processes mask tokens, which is both a large efficiency win and a way to prevent the encoder from allocating capacity to the masking artefact.
4. **Auxiliary pretext tasks.** VIME's mask estimation shows a second objective over the corruption pattern can add signal where reconstruction alone is weak.

## Relevance to imputation

Epigenome imputation is structurally a masked-reconstruction problem — the missing (cell type, assay) entries *are* the mask — with one difference: the mask is imposed by which experiments were performed, not chosen by the modeller, and is therefore neither random nor uniform. See [[epigenome-imputation]].

## The recipe in single-cell epigenomics

`raw/chen-2025-epiagent.pdf` (EpiAgent) is the largest instance of this recipe in the epigenomic
domain: **~5 million cells and 35 billion tokens** of scATAC data, encoded as ordered "cell
sentences" of accessible cCREs and pretrained with bidirectional attention. It lands on the
BERT side of design axis 1 — reconstruction in the discrete input space, not in representation
space — and gets the corresponding property: no collapse risk, but a target that carries no
magnitude information.

Its downstream results include **zero-shot** cell-type annotation and imputation on unseen
scATAC data, which is the same generalisation claim the [[epigenome-imputation]] lineage makes,
reached by masked pretraining rather than by factorising a compendium.

## Augmentation as the other route to invariance

One design axis is missing from the four above because none of these sources isolates it:
whether an invariance should be **trained in by augmentation** or **built in architecturally**.
For DNA the canonical case is reverse-complement symmetry — the two strands carry the same
regulatory content, so a model can either see randomly reverse-complemented inputs during
training or share parameters between the two orientations by construction. Augmentation is
cheaper and imposes the invariance only approximately, which is sometimes desirable (strand
asymmetry is real for some assays); architectural sharing imposes it exactly. I-JEPA's deliberate
avoidance of hand-crafted augmentations is the opposite pole of this same axis.

`raw/shrikumar-2017-revcomp-parameter-sharing.pdf` takes the architectural side of that choice for
genomics: rather than augmenting with randomly reverse-complemented inputs, it **shares parameters
between the two orientations by construction**, so a model's prediction on a sequence and on its
reverse complement agree exactly rather than approximately. The distinction is testable and worth
testing — an augmented model's two predictions are only as consistent as training made them, and
the size of that disagreement is a free diagnostic for whether the augmentation actually took.

## See also

Related:: [[transformers-and-positional-encoding]], [[epigenome-imputation]], [[film-conditioning]], [[sequence-conditioned-epigenome-models]], [[jepa-and-collapse-prevention]], [[set-conditioned-modelling-and-missingness]]
