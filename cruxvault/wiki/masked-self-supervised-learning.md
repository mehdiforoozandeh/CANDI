---
type: wiki
title: Masked self-supervised learning
summary: Corrupt-and-reconstruct pre-training across text, images and tabular data — BERT, MAE, I-JEPA, VIME — and the design choices (mask ratio, prediction space) that decide what is learned.
category: concept
sources: raw/devlin-2019-bert.pdf, raw/he-2022-masked-autoencoders.pdf, raw/assran-2023-ijepa.pdf, raw/yoon-2020-vime.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
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

1. **Prediction space.** Pixels/tokens (MAE, BERT) versus representations (I-JEPA). Reconstructing the input forces capacity onto high-frequency detail that may be irrelevant; predicting representations concentrates it on semantics, at the cost of a collapse risk that must be managed.
2. **Mask ratio and mask structure.** MAE's 75% and I-JEPA's large, spatially distributed blocks both say the same thing: masking must be aggressive and structured enough that the task cannot be solved by local interpolation.
3. **Asymmetry.** MAE's encoder never processes mask tokens, which is both a large efficiency win and a way to prevent the encoder from allocating capacity to the masking artefact.
4. **Auxiliary pretext tasks.** VIME's mask estimation shows a second objective over the corruption pattern can add signal where reconstruction alone is weak.

## Relevance to imputation

Epigenome imputation is structurally a masked-reconstruction problem — the missing (cell type, assay) entries *are* the mask — with one difference: the mask is imposed by which experiments were performed, not chosen by the modeller, and is therefore neither random nor uniform. See [[epigenome-imputation]].

## See also

Related:: [[transformers-and-positional-encoding]], [[epigenome-imputation]], [[film-conditioning]], [[sequence-conditioned-epigenome-models]]
