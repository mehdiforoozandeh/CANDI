---
type: wiki
title: Sequence-conditioned epigenome models
summary: Models that predict epigenomic and transcriptomic signal from DNA sequence (± chromatin accessibility) — Enformer, BPNet, EPCOT, dHICA, TFImpute, DNABERT — and how they differ from tensor-based imputation.
category: comparison
sources: raw/avsec-2021-enformer.xml, raw/avsec-2021-bpnet.html, raw/zhang-2023-epcot.xml, raw/wen-2024-discriminative-histone-imputation.pdf, raw/qin-2017-tf-binding-deep-learning-imputation.xml, raw/ji-2021-dnabert.html, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/avsec-2026-alphagenome.xml, raw/aksu-2026-corgi.pdf, raw/linder-2025-borzoi.xml, raw/kelley-2018-basenji.xml, raw/fu-2025-get.xml, raw/javed-2025-epibert.xml, raw/sun-2026-succeed.pdf, raw/hingerl-2025-scooby.xml, raw/gao-2024-epigept.xml, raw/murphy-2024-enformer-celltyping.xml, raw/chen-2022-sei.xml, raw/lal-2025-grelu.xml, raw/barbadilla-martinez-2025.pdf, raw/vafa-2025-steerability.pdf
created: 2026-07-31T21:26:00
updated: 2026-08-01T18:05:24
---

# Sequence-conditioned epigenome models

These models share an input DNA sequence and differ mainly in what breaks the sequence's cell-type invariance — nothing (Enformer's per-cell-type output heads), chromatin accessibility (EPCOT, dHICA), or a learned cell-line embedding (TFImpute).

## The cell-type problem

DNA sequence is identical across cell types in one individual, so a purely sequence-based model cannot by itself explain cell-type-specific signal. Each method resolves this differently:

- **Enformer** (`raw/avsec-2021-enformer.xml`) — one shared trunk, 5,313 human output tracks (one per experiment), each its own head. Architecture: 7 conv blocks with pooling, then **11 transformer blocks**, then cropping and pointwise convolutions into organism-specific heads. Input 196,608 bp → 896 bins of 128 bp (114,688 bp of output after cropping), trained with a **Poisson negative log-likelihood** loss inherited from Basenji2. Cell-type-specific correlation rose from 0.81 to 0.85 against an experimental ceiling of 0.94.

  > **Two receptive-field numbers, both correct.** **196,608 bp** is the *input span* the network consumes. **~100 kb** is the paper's own claim about *reach* — "reaching distal regulatory elements up to 100 kb away", a one-sided distance from the TSS, roughly the input half-width, against Basenji2's and ExPecto's 20 kb. Quote the span when comparing architectures; quote the reach when discussing how far regulatory influence is integrated. [[transformers-and-positional-encoding]] uses the reach figure and [[sequence-model-critiques]] the span, both deliberately.
- **BPNet** (`raw/avsec-2021-bpnet.html`) — base-resolution prediction of ChIP-nexus profiles for pluripotency TFs, plus interpretation tooling that extracts motifs and "soft syntax" rules (e.g. Nanog's helical-periodicity preference). The lineage that establishes base-resolution profile prediction and model interpretation as a goal in itself.
- **EPCOT** (`raw/zhang-2023-epcot.xml`) — pre-training/fine-tuning: a cell-type-specific pre-training model supervised by epigenomic features takes **sequence + chromatin accessibility**, and downstream heads predict gene expression, Hi-C/Micro-C contact maps, ChIA-PET, and enhancer activity for **new cell types** from accessibility alone. Its explicit motivation is that prior models' representations do not generalise across tasks or cell types.
- **dHICA** (`raw/wen-2024-discriminative-histone-imputation.pdf`) — Transformer plus **dilated convolutions** over sequence + chromatin accessibility to predict multiple histone-mark tracks at once; reports better performance at cell-specific loci and gene elements, with downstream use in chromatin-state segmentation and SNP interpretation.
- **TFImpute** (`raw/qin-2017-tf-binding-deep-learning-imputation.xml`) — an early deep multi-task model predicting cell-specific TF binding for TF × cell-line combinations, trained on only ~4% of combinations; beats DeepBind and gkm-SVM specifically on combinations with **no** ChIP-seq data. The clearest early statement of imputation-as-generalisation across a factor × cell-type matrix.
- **DNABERT** (`raw/ji-2021-dnabert.html`) — BERT-style masked pre-training on k-merised DNA, giving transferable sequence representations; see [[masked-self-supervised-learning]] and [[transformers-and-positional-encoding]].

## Contrast with tensor-based imputation

The [[epigenome-imputation]] lineage ([[chromimpute]], [[predictd]], [[avocado]], [[edice]]) conditions on **other assays in the same sample** and is essentially blind to sequence; these models condition on **sequence** and are blind (or only weakly conditioned) on other assays. `raw/schreiber-2023-encode-imputation-challenge.pdf` notes that of the five major imputation methods it surveys, only one used nucleotide sequence at all, and that only 5 of 23 challenge submissions did — sequence was, at the time, an under-explored axis of the design space.

The two conditioning sources are complementary rather than competing: sequence supplies position-specific priors that transfer to unseen cell types, while observed assays supply the cell-type identity that sequence cannot.

## The lineage in order

**Basenji** (`raw/kelley-2018-basenji.xml`) is the origin: dilated convolutions predicting cell-type-specific epigenetic and transcriptional profiles in 128 bp bins from DNA alone, extending the peak-classification approach of Basset to quantitative coverage. Enformer replaced its dilated stack with transformer blocks; **Borzoi** (`raw/linder-2025-borzoi.xml`) extended the targets to **RNA-seq coverage** at base-ish resolution, so that transcription, splicing and polyadenylation effects can all be read off one predicted track. **Sei** (`raw/chen-2022-sei.xml`) took the breadth route instead — 21,907 chromatin profiles across >1,300 cell lines and tissues, distilled into a vocabulary of "sequence classes."

**AlphaGenome** (`raw/avsec-2026-alphagenome.xml`) is the current ceiling: **1 Mb of input** predicting **5,930 human tracks across 11 modalities** — gene expression, transcription initiation, chromatin accessibility, histone modifications, TF binding, contact maps, splice sites/usage/junctions — up to **single-base-pair resolution**, dissolving the length-versus-resolution trade-off that constrained everything before it. It matches or exceeds the strongest external model in 25 of 26 variant-effect evaluations.

**NTv3** and **Evo 2** push further on context and self-supervision — see [[genomic-language-models]].

## Breaking sequence's cell-type invariance

The models that generalise to **unseen cell types** all add a non-sequence input:

- **EPCOT** and **dHICA** (above) take chromatin accessibility.
- **GET** (`raw/fu-2025-get.xml`) uses accessibility plus motif content over ~2 Mb across 213 fetal and adult cell types, reaching experimental-level expression accuracy in unseen cell types, and R² 0.53 on adult cell types when trained only on fetal data.
- **EpiBERT** (`raw/javed-2025-epibert.xml`) is the closest analogue to masked-assay self-supervision: **masked-accessibility pretraining** over sequence + cell-type ATAC, then fine-tuning for expression. It imputes masked ATAC signal for **17 held-out cell types** never seen in training — i.e. it is doing cross-cell-type imputation with a masked objective.
- **Enformer Celltyping** (`raw/murphy-2024-enformer-celltyping.xml`) transfers Enformer to unseen cell types using DNA + accessibility, beating Epitome, and proposes an rQTL-based evaluation. It also reports the sharp caveat that genome-wide performance can look strong simply because the average signal is a good genome-wide predictor — the [[average-activity-baseline]] problem in this literature.
- **EpiGePT** (`raw/gao-2024-epigept.xml`) conditions on **transcription-factor activity** plus 3D interactions to steer prediction into a new cellular context.
- **Corgi** (`raw/aksu-2026-corgi.pdf`) conditions on **trans-regulator expression** (TFs, histone modifiers, coactivators, RBPs), and claims a new state of the art for joint cross-sequence and cross-cell-type epigenetic track prediction, approaching experimental accuracy for expression in unseen cell types.
- **SUCCEED** (`raw/sun-2026-succeed.pdf`) takes the supervised-pretraining route: conv + transformer pretrained on **6,389 ENCODE functional tracks**, then transferred to predict cell-type-specific epigenomic profiles, **denoise sparse accessibility signal**, and predict 3-D contacts — evidence that supervised functional pretraining transfers where sequence-only pretraining does not (see [[sequence-model-critiques]]).

## Conditioned decoders

**scooby** (`raw/hingerl-2025-scooby.xml`) is the most architecturally relevant: it takes pretrained Borzoi and equips it with a **cell-specific decoder** whose weights are produced from a single-cell embedding — a hypernetwork, the weight-generating cousin of FiLM (see [[query-decoders-and-conditional-computation]]) — and trains on **raw coverage with Poisson and multinomial losses**. Because cells are represented by an embedding rather than as separate output tasks, it extends to cells not seen in training.

**gReLU** (`raw/lal-2025-grelu.xml`) is the practical entry point to this whole family: a framework with a pretrained model zoo covering preprocessing, training, evaluation, interpretation and variant-effect prediction, and the shortest route to running Enformer- and Borzoi-class baselines without reimplementing them.

## What the field calls itself, and what it argues about

`raw/barbadilla-martinez-2025.pdf` is the current review of this area and supplies both the
field's own terminology and its live disagreements. These models are **S2E** —
sequence-to-expression — and the review is explicit about their defining limitation:
there is currently no reason to think they make reliable predictions **beyond the cell
types and conditions represented in their training data**. Its example is concrete: they
cannot predict expression in a cell type treated with a hormone unless training data were
generated from that cell type under that treatment. Generalisation to a new *sequence* and
generalisation to a new *context* are different problems, and only the first is solved.

**Multitask learning is contested, not settled.** The usual justification for many output
heads is knowledge transfer between related tasks, and the review reports the opposite
finding: multitask models such as Enformer and Sei **underperformed single-task models on
cell-type-specific accessibility data**, and Enformer additionally captured distal
regulatory elements poorly and erred on sequence-variant effects. The diagnosis offered is
that **cell-type-specific features are under-represented** when many tasks share a trunk —
the model's capacity flows to what is common across tasks, which is precisely the
[[average-activity-baseline]] component. Two remedies are proposed in preprint studies:
**balance the importance of each task**, and account for expression variation across cell
types at each locus. The first is per-*task* loss balancing, not per-locus weighting — the
same family of intervention as the learned per-task log-variance in [[uncertainty-calibration]],
and the reason the multitask complaint is an *optimisation* complaint before it is a capacity
one. See [[multi-task-optimization]].
The review's overall verdict is that smaller single-task models "may result in equal or
better performance, faster training, quicker predictions and facilitate interpretability,
provided that high-quality training data are available."

**Bigger is not automatically better.** Transformers carry quadratic compute and memory
cost, and the review notes that where training uses large numbers of short sequences (MPRA
settings), **CNN models can outperform transformer-based ones** — and that recent published
and preprint studies working from genome-wide measurements "still prefer simpler models such
as CNNs." Receptive field growth has also bought less than advertised: for Enformer, most
predictive signal derives from **proximal regions and promoters rather than distal
enhancers**, with the proposed explanation that long-range regulatory interactions are
comparatively rare and so supply few training examples. See [[sequence-model-critiques]].

For orientation on scale (all figures here are **input spans**, per the note at Enformer above):
Basset operated on ≤1 kb, Basenji extended to 131 kb with dilated
convolutions, Enformer to ~196 kb with transformers, and Borzoi to 524 kb while modelling
RNA-seq coverage — with the review attributing Borzoi's gains as much to modelling multiple
regulatory layers as to receptive field alone.

## Producible is not reachable

`raw/vafa-2025-steerability.pdf` supplies the evaluation concept these models are usually missing.
Its distinction is between what a generative model **can produce** and what a user can actually
**steer it to produce** — a model whose output distribution covers a target is not thereby a model
you can drive to that target. Measuring steerability is a separate exercise from measuring
predictive accuracy or coverage.

Applied to a conditioned epigenome model, the two questions come apart cleanly: "does the model
represent cell-type-specific signal?" is answered by held-out correlation, while "can I move it
to a specified cell type by setting the conditioning?" is not answered by any accuracy measure.
The multitask critique above is a symptom of the same gap — a shared trunk can score well on
aggregate accuracy while the conditioning pathway does little work. See
[[covariate-conditioning-and-counterfactuals]] for the instruments that test the second question.

## The honest caveat

Every model in this section is evaluated primarily on held-out **genomic regions**. That is a different task from prediction across individuals or across poorly characterised cell types, and the models perform very differently on them — see [[sequence-model-critiques]].

## See also

Related:: [[epigenome-imputation]], [[transformers-and-positional-encoding]], [[count-distributions-for-sequencing-data]], [[peak-calling-and-signal-tracks]], [[masked-self-supervised-learning]], [[sequence-model-critiques]], [[genomic-language-models]], [[query-decoders-and-conditional-computation]]
