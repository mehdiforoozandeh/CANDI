---
type: wiki
title: Critiques of sequence-to-function models
summary: The evidence that DNA-sequence models fail on individual variation, ignore distal enhancers, and that DNA language-model embeddings underperform one-hot — while supervised functional pretraining does transfer.
category: comparison
sources: raw/sasse-2023.xml, raw/karollus-2023.xml, raw/tang-2025-tang-glm.xml, raw/mostafavi-2026-modality-gap.pdf, raw/spiro-2026-sagenet.pdf, raw/patel-2024-dart-eval.pdf, raw/barbadilla-martinez-2025.pdf
created: 2026-07-31T23:27:28
updated: 2026-08-01T00:51:15
---

# Critiques of sequence-to-function models

This literature is the strongest available argument for conditioning on measured assays rather than on sequence alone, and it is unusually blunt about what current models cannot do.

## Failure across individuals

`raw/sasse-2023.xml` is the founding result. Using paired whole-genome sequencing and RNA-seq from **839 individuals** in the ROSMAP cohort, it evaluates Enformer, Basenji2, ExPecto and Xpresso as *personal* DNA interpreters — predicting each individual's expression from their own phased genome. Prior evaluations had only tested prediction **across genomic regions**, which conflates "can distinguish a highly expressed gene from a silent one" with "can predict a person's expression level." Across individuals the models fail, frequently getting even the **direction** of cis-regulatory effects wrong.

`raw/spiro-2026-sagenet.pdf` follows up with a scalable framework (SAGE-net) for training sequence-to-expression models **on personal genomes**, and reports the sobering diagnosis: personal-genome training does improve accuracy, but largely by **memorising predictive variants** rather than learning transferable cis-regulatory grammar. Improvement on the metric is not evidence of the mechanism you wanted.

## The modality gap

`raw/mostafavi-2026-modality-gap.pdf` is the most directly relevant result for epigenome work. Evaluating AlphaGenome and its peers on personal-genome prediction, it finds a **gap between modalities**: for **chromatin accessibility** the models approach the heritability ceiling, while for **gene expression** they remain far below baseline. The epigenome is the tractable modality; expression is not, yet. Any claim about sequence-to-function limitations should be stated per modality rather than in general.

## Where the signal comes from

`raw/karollus-2023.xml` confronts state-of-the-art models with two large observational studies and five deep perturbation assays. Enformer largely captures the causal determinants of human **promoters**, but the models **fail to capture the causal effects of distal enhancers**. Despite Enformer's 196 kb input span (and its claimed ~100 kb reach — see the note in [[sequence-conditioned-epigenome-models]]), most of that context has very minor impact on its predictions — the signal is concentrated within roughly **30 kb of the TSS**. This is an empirical bound on what receptive-field expansion has actually bought, and a caution against equating context length with modelled context.

The paper's framing is worth carrying: training on genome-wide assays is **fundamentally correlative**, exposing the model only to sequence variation that arose through evolution, which is why perturbation and personal-genome evaluations disagree with held-out-region evaluations.

## DNA language models

`raw/tang-2025-tang-glm.xml` is the single most important source here for a model that consumes both sequence and assays. Prior gLM evaluations fine-tuned the whole model per task, which cannot distinguish a good pretrained representation from a good initialisation. Evaluating the **representations themselves** on regulatory-genomics tasks, it finds gLM embeddings offer **no advantage over one-hot encoded DNA** — while models pretrained on **functional genomics data do transfer**. That is a direct empirical endorsement of supervised/functional pretraining over sequence-only self-supervision for regulatory tasks.

`raw/patel-2024-dart-eval.pdf` (DART-Eval) reaches a compatible conclusion from a benchmark-design angle: across zero-shot, probed and fine-tuned settings on regulatory DNA, DNA language models do **not offer compelling gains over ab initio baselines** on most tasks, at substantially greater computational cost. The paper is also a critique of benchmarking practice — earlier evaluations used flawed baselines and inappropriate protocols.

## The review's own summary of the limits

`raw/barbadilla-martinez-2025.pdf` gathers these critiques and adds several of its own,
which is useful because it is a Nature Reviews synthesis rather than a paper defending a
method.

- **eQTL benchmarking is weak evidence.** eQTL map resolution is limited by sample size and
  linkage disequilibrium, making it hard to pinpoint the causal variant. Even with
  fine-mapping, most studies report only **modest correlations** between S2E predictions and
  eQTL data. The review's position is that eQTL benchmarking is useful for side-by-side
  comparison and for exposing shortcomings — not as a validation of correctness.
- **Failures across individuals are not gene-intrinsic.** Restating the personal-genome
  results, the review adds an observation that sharpens them: incorrect predictions were
  **not consistently observed for the same genes across different models**, implying the
  cause is not an intrinsic property of those genes' regulation but something about how each
  model was fit. Its suggested remedy matches the primary sources — train on personalised
  genomes so the model sees within-population sequence variation.
- **Genomic language models remain limited for regulatory tasks.** Despite success on coding
  genes, the review reports gLM utility for the regulatory genome is limited, attributing it
  to differences in **conservation level and grammar complexity** between coding and
  regulatory sequence. This is an independent restatement of the finding above. See
  [[genomic-language-models]].
- **Models hallucinate.** Stated plainly: DL models "can 'hallucinate', that is, make certain
  predictions that do not match the real world", so independent validation by orthogonal
  methodology is essential — especially before clinical or biotechnological use.

## The training-data critique

The review also questions the *inputs*, not just the models. Epigenome maps of open
chromatin and histone modifications are **intrinsically correlative** and do not directly
measure causal regulatory activity; the genome is partitioned into large domains of
**autocorrelated** histone modifications, which makes pinpointing the causally relevant
sequence hard; and **~15–50% of regions marked by open chromatin are not detectably active
as enhancers**. Lists of regions marked by such features are widely used as training targets,
yet "the vast majority of elements in these lists have so far not been experimentally
validated." Models trained on them, the review concludes, "should thus be interpreted with
caution."

A related observation cuts against relying on the genome alone as a training corpus: models
trained on **billions of random DNA sequences** in an MPRA were *more* predictive of known
transcription start site positions than models trained on actual genomic sequences —
attributed to the limited sequence diversity present within the human genome.

## How to read this collectively

Three separable claims, often conflated:
1. Sequence models are good at **across-region** prediction and poor at **across-individual** prediction (`raw/sasse-2023.xml`, `raw/spiro-2026-sagenet.pdf`).
2. Their effective context is far smaller than their nominal receptive field, and distal enhancers are largely missed (`raw/karollus-2023.xml`).
3. Self-supervised **sequence** pretraining underperforms supervised **functional** pretraining for regulatory tasks (`raw/tang-2025-tang-glm.xml`, `raw/patel-2024-dart-eval.pdf`).

None of them says sequence is uninformative — see [[sequence-conditioned-epigenome-models]] for what these models do achieve.

## See also

Related:: [[sequence-conditioned-epigenome-models]], [[genomic-language-models]], [[imputation-evaluation-measures]], [[cross-cell-type-generalization-pitfall]]
