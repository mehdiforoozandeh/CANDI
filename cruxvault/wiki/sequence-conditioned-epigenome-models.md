---
type: wiki
title: Sequence-conditioned epigenome models
summary: Models that predict epigenomic and transcriptomic signal from DNA sequence (± chromatin accessibility) — Enformer, BPNet, EPCOT, dHICA, TFImpute, DNABERT — and how they differ from tensor-based imputation.
category: comparison
sources: raw/avsec-2021-enformer.xml, raw/avsec-2021-bpnet.html, raw/zhang-2023-epcot.xml, raw/wen-2024-discriminative-histone-imputation.pdf, raw/qin-2017-tf-binding-deep-learning-imputation.xml, raw/ji-2021-dnabert.html, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Sequence-conditioned epigenome models

These models share an input DNA sequence and differ mainly in what breaks the sequence's cell-type invariance — nothing (Enformer's per-cell-type output heads), chromatin accessibility (EPCOT, dHICA), or a learned cell-line embedding (TFImpute).

## The cell-type problem

DNA sequence is identical across cell types in one individual, so a purely sequence-based model cannot by itself explain cell-type-specific signal. Each method resolves this differently:

- **Enformer** (`raw/avsec-2021-enformer.xml`) — one shared trunk, 5,313 human output tracks (one per experiment), each its own head. Architecture: 7 conv blocks with pooling, then **11 transformer blocks**, then cropping and pointwise convolutions into organism-specific heads. Input 196,608 bp → 896 bins of 128 bp. Receptive field up to ~100 kb, trained with a **Poisson negative log-likelihood** loss inherited from Basenji2. Cell-type-specific correlation rose from 0.81 to 0.85 against an experimental ceiling of 0.94.
- **BPNet** (`raw/avsec-2021-bpnet.html`) — base-resolution prediction of ChIP-nexus profiles for pluripotency TFs, plus interpretation tooling that extracts motifs and "soft syntax" rules (e.g. Nanog's helical-periodicity preference). The lineage that establishes base-resolution profile prediction and model interpretation as a goal in itself.
- **EPCOT** (`raw/zhang-2023-epcot.xml`) — pre-training/fine-tuning: a cell-type-specific pre-training model supervised by epigenomic features takes **sequence + chromatin accessibility**, and downstream heads predict gene expression, Hi-C/Micro-C contact maps, ChIA-PET, and enhancer activity for **new cell types** from accessibility alone. Its explicit motivation is that prior models' representations do not generalise across tasks or cell types.
- **dHICA** (`raw/wen-2024-discriminative-histone-imputation.pdf`) — Transformer plus **dilated convolutions** over sequence + chromatin accessibility to predict multiple histone-mark tracks at once; reports better performance at cell-specific loci and gene elements, with downstream use in chromatin-state segmentation and SNP interpretation.
- **TFImpute** (`raw/qin-2017-tf-binding-deep-learning-imputation.xml`) — an early deep multi-task model predicting cell-specific TF binding for TF × cell-line combinations, trained on only ~4% of combinations; beats DeepBind and gkm-SVM specifically on combinations with **no** ChIP-seq data. The clearest early statement of imputation-as-generalisation across a factor × cell-type matrix.
- **DNABERT** (`raw/ji-2021-dnabert.html`) — BERT-style masked pre-training on k-merised DNA, giving transferable sequence representations; see [[masked-self-supervised-learning]] and [[transformers-and-positional-encoding]].

## Contrast with tensor-based imputation

The [[epigenome-imputation]] lineage ([[chromimpute]], [[predictd]], [[avocado]], [[edice]]) conditions on **other assays in the same sample** and is essentially blind to sequence; these models condition on **sequence** and are blind (or only weakly conditioned) on other assays. `raw/schreiber-2023-encode-imputation-challenge.pdf` notes that of the five major imputation methods it surveys, only one used nucleotide sequence at all, and that only 5 of 23 challenge submissions did — sequence was, at the time, an under-explored axis of the design space.

The two conditioning sources are complementary rather than competing: sequence supplies position-specific priors that transfer to unseen cell types, while observed assays supply the cell-type identity that sequence cannot.

## See also

Related:: [[epigenome-imputation]], [[transformers-and-positional-encoding]], [[count-distributions-for-sequencing-data]], [[peak-calling-and-signal-tracks]], [[masked-self-supervised-learning]]
