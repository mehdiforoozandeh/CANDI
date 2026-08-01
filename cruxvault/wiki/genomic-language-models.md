---
type: wiki
title: Genomic language models
summary: Large self-supervised models over raw DNA at megabase context — Evo 2 and Nucleotide Transformer v3 — and what they add beyond supervised sequence-to-function models.
category: comparison
sources: raw/brixi-2026-evo2.xml, raw/boshar-2025-ntv3.xml
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Genomic language models

The interesting development in this line is not scale but convergence: the newest models fold supervised functional-track prediction into the self-supervised backbone, which is exactly what the gLM critiques said was missing.

## Evo 2

`raw/brixi-2026-evo2.xml` is a biological foundation model trained on **9 trillion DNA base pairs** spanning all domains of life, with a **1 million token context window at single-nucleotide resolution**, in 7B and 40B parameter versions. Its headline capability is **zero-shot variant effect prediction** — predicting the functional impact of genetic variation, from noncoding pathogenic mutations to BRCA1 variants, without task-specific fine-tuning — by using sequence likelihood as a proxy for evolutionary constraint.

Two aspects are relevant beyond variant effects. First, the paper addresses the black-box critique directly by using **sparse autoencoders** to extract interpretable latent features from the model's representations. Second, it includes **generative design** experiments, among them chromatin-accessibility design — the closest a pure gLM gets to producing epigenomic output.

The critical caveat: gLMs had historically **lagged badly** at eukaryotic variant effect prediction relative to alignment-based methods, and Evo 2's claim is that this gap is now closed. That claim is contested — see [[sequence-model-critiques]].

## Nucleotide Transformer v3

`raw/boshar-2025-ntv3.xml` is the more architecturally relevant of the two. Its explicit motivation is that three strengths have historically been realised in *isolation*: supervised sequence-to-function accuracy, self-supervised transferable representations, and conditional generative design. NTv3 unifies them in one backbone:

- Pre-trained on OpenGenome2 (>128,000 species genomes, >8 trillion nucleotides), then **post-trained on multispecies functional data** — 15,889 functional tracks — turning the representation model into a genome-to-function model.
- **U-Net architecture** giving base-pair resolution output over 1 Mb input, rather than the binned output of the convolution-plus-attention lineage.
- **adaLN species conditioning** — the same adaptive-layer-norm conditioning primitive used to inject class information into transformers (see [[film-conditioning]]), here carrying organism identity.
- **Controllable generation** alongside prediction.
- Reports state-of-the-art functional track and annotation prediction against Borzoi under Borzoi's own training regime, plus a 106-task benchmark at 32 kb input / single-bp output.

The two-stage recipe — self-supervised sequence pretraining, then supervised functional post-training — is the direct response to the finding that sequence-only pretraining does not transfer to regulatory tasks while functional pretraining does (`raw/brixi-2026-evo2.xml` takes the opposite bet by staying self-supervised).

## Reading them against the supervised lineage

Both models are conditioned on **sequence and organism**, not on measured assays from the target sample. That is the axis on which they differ from [[sequence-conditioned-epigenome-models]] variants like EPCOT and EpiBERT, which take a cell-type-specific accessibility track as input, and from the [[epigenome-imputation]] lineage, which conditions on other assays in the same biosample. Scale on the sequence axis does not substitute for observing the sample.

## See also

Related:: [[sequence-conditioned-epigenome-models]], [[sequence-model-critiques]], [[transformers-and-positional-encoding]], [[film-conditioning]], [[masked-self-supervised-learning]]
