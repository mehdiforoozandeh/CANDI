---
type: wiki
title: Critiques of sequence-to-function models
summary: The evidence that DNA-sequence models fail on individual variation, ignore distal enhancers, and that DNA language-model embeddings underperform one-hot — while supervised functional pretraining does transfer.
category: comparison
sources: raw/sasse-2023.xml, raw/karollus-2023.xml, raw/tang-2025-tang-glm.xml, raw/mostafavi-2026-modality-gap.pdf, raw/spiro-2026-sagenet.pdf, raw/patel-2024-dart-eval.pdf
created: 2026-07-31T23:27:28
updated: 2026-07-31T23:27:28
---

# Critiques of sequence-to-function models

This literature is the strongest available argument for conditioning on measured assays rather than on sequence alone, and it is unusually blunt about what current models cannot do.

## Failure across individuals

`raw/sasse-2023.xml` is the founding result. Using paired whole-genome sequencing and RNA-seq from **839 individuals** in the ROSMAP cohort, it evaluates Enformer, Basenji2, ExPecto and Xpresso as *personal* DNA interpreters — predicting each individual's expression from their own phased genome. Prior evaluations had only tested prediction **across genomic regions**, which conflates "can distinguish a highly expressed gene from a silent one" with "can predict a person's expression level." Across individuals the models fail, frequently getting even the **direction** of cis-regulatory effects wrong.

`raw/spiro-2026-sagenet.pdf` follows up with a scalable framework (SAGE-net) for training sequence-to-expression models **on personal genomes**, and reports the sobering diagnosis: personal-genome training does improve accuracy, but largely by **memorising predictive variants** rather than learning transferable cis-regulatory grammar. Improvement on the metric is not evidence of the mechanism you wanted.

## The modality gap

`raw/mostafavi-2026-modality-gap.pdf` is the most directly relevant result for epigenome work. Evaluating AlphaGenome and its peers on personal-genome prediction, it finds a **gap between modalities**: for **chromatin accessibility** the models approach the heritability ceiling, while for **gene expression** they remain far below baseline. The epigenome is the tractable modality; expression is not, yet. Any claim about sequence-to-function limitations should be stated per modality rather than in general.

## Where the signal comes from

`raw/karollus-2023.xml` confronts state-of-the-art models with two large observational studies and five deep perturbation assays. Enformer largely captures the causal determinants of human **promoters**, but the models **fail to capture the causal effects of distal enhancers**. Despite Enformer's 196 kb receptive field, most of that field has very minor impact on its predictions — the signal is concentrated within roughly **30 kb of the TSS**. This is an empirical bound on what receptive-field expansion has actually bought, and a caution against equating context length with modelled context.

The paper's framing is worth carrying: training on genome-wide assays is **fundamentally correlative**, exposing the model only to sequence variation that arose through evolution, which is why perturbation and personal-genome evaluations disagree with held-out-region evaluations.

## DNA language models

`raw/tang-2025-tang-glm.xml` is the single most important source here for a model that consumes both sequence and assays. Prior gLM evaluations fine-tuned the whole model per task, which cannot distinguish a good pretrained representation from a good initialisation. Evaluating the **representations themselves** on regulatory-genomics tasks, it finds gLM embeddings offer **no advantage over one-hot encoded DNA** — while models pretrained on **functional genomics data do transfer**. That is a direct empirical endorsement of supervised/functional pretraining over sequence-only self-supervision for regulatory tasks.

`raw/patel-2024-dart-eval.pdf` (DART-Eval) reaches a compatible conclusion from a benchmark-design angle: across zero-shot, probed and fine-tuned settings on regulatory DNA, DNA language models do **not offer compelling gains over ab initio baselines** on most tasks, at substantially greater computational cost. The paper is also a critique of benchmarking practice — earlier evaluations used flawed baselines and inappropriate protocols.

## How to read this collectively

Three separable claims, often conflated:
1. Sequence models are good at **across-region** prediction and poor at **across-individual** prediction (`raw/sasse-2023.xml`, `raw/spiro-2026-sagenet.pdf`).
2. Their effective context is far smaller than their nominal receptive field, and distal enhancers are largely missed (`raw/karollus-2023.xml`).
3. Self-supervised **sequence** pretraining underperforms supervised **functional** pretraining for regulatory tasks (`raw/tang-2025-tang-glm.xml`, `raw/patel-2024-dart-eval.pdf`).

None of them says sequence is uninformative — see [[sequence-conditioned-epigenome-models]] for what these models do achieve.

## See also

Related:: [[sequence-conditioned-epigenome-models]], [[genomic-language-models]], [[imputation-evaluation-measures]], [[cross-cell-type-generalization-pitfall]]
