---
type: wiki
title: Avocado
summary: Schreiber et al. 2020: deep multi-scale tensor factorisation that both imputes epigenomic tracks and yields a reusable latent representation of the genome; extended to the full ENCODE3 compendium.
category: method
sources: raw/schreiber-2020-avocado.xml, raw/schreiber-2020-encode3-compendium.xml, raw/durham-2018-predictd.xml, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/hawkins-hooker-2023-edice.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Avocado

Avocado's real claim is not better imputation but a better *representation* — the latent factors outperform the raw epigenomic data as features for downstream tasks the model was never trained on.

## Method

Avocado starts from [[predictd]]'s three-axis factorisation and makes two changes (`raw/schreiber-2020-avocado.xml`):

1. The multilinear product is replaced by a **deep neural network** that takes the concatenated cell-type, assay, and genomic factors and predicts the signal value.
2. Genomic factors are **multi-scale** — separate factor sets at 25 bp, 250 bp, and 5 kb resolution — so the model represents both punctate and broad structure without needing a single resolution to carry both.

## Results

**Imputation.** Better than [[predictd]] and [[chromimpute]] on most but not all measures, and the exception is the one CANDI should care about. Table 1 of `raw/schreiber-2020-avocado.xml` reports six MSE variants (the first three originally defined in `raw/durham-2018-predictd.xml`):

| | Global | 1obs | 1imp | Prom | Gene | Enh |
|---|---|---|---|---|---|---|
| ChromImpute | 0.113 | **0.941** | 1.09 | 0.325 | 0.149 | 0.316 |
| PREDICTD | 0.100 | 1.76 | 0.897 | 0.258 | **0.129** | 0.267 |
| Avocado | 0.100 | 1.66 | **0.845** | **0.249** | 0.130 | **0.260** |

Avocado is best on three (MSE1imp, MSEProm, MSEEnh), **ties** PREDICTD on MSEglobal (p = 0.451) and MSEGene (p = 0.875), and **loses MSE1obs to ChromImpute** by ~1.8× at p = 2.37e−22 — "Conversely, ChromImpute performs the best on MSE1obs". MSE1obs is the top 1% of positions by *observed* signal; MSE1imp is the top 1% by *imputed* signal. The paper reads the split as PREDICTD systematically underpredicting (good on low signal, poor on peaks) and ChromImpute over-calling peaks. `raw/hawkins-hooker-2023-edice.xml` states the same thing from outside: these approaches have outstripped ChromImpute "only on a subset of metrics".

This matters for a peak head: the region where the factorisation lineage is *weakest* relative to ChromImpute is exactly the high-observed-signal region a peak classifier is scored on.

**Representation.** The learned latent factors, used as features, beat models trained directly on the epigenomic signal for gene expression prediction, promoter–enhancer interactions, replication timing, and an element of 3-D chromatin architecture — none of which Avocado was trained on. This is the earliest strong evidence in this literature that an imputation model's *internal state* is more valuable than its output, and it is the direct antecedent of latent-representation evaluation in imputation work generally.

## ENCODE3 extension

`raw/schreiber-2020-encode3-compendium.xml` applies Avocado to 3,814 ENCODE tracks covering chromatin accessibility, histone modification, transcription, and protein binding. Two findings carry forward:

- Avocado's TF-binding imputations improve significantly on the top **ENCODE-DREAM challenge** models — evidence that a general imputation model can beat task-specific ones.
- New assays and new biosamples can be added to a **pre-trained** model by freezing almost all parameters and fitting only the new axis factors. This is the closest the tensor-factorisation lineage gets to zero-shot extension: cheap for a new *biosample*, but still a fitting step, not a forward pass.

## Role in the challenge

Avocado occupies an unusual position in `raw/schreiber-2023-encode-imputation-challenge.pdf`: three of the 23 submissions consumed **Avocado's imputations as input features**, and those submissions had the highest genome-wide correlation with Avocado's predictions. Avocado also served as one of the two reference baselines against which entrant error patterns were compared (the other being the [[average-activity-baseline]]).

## See also

Related:: [[epigenome-imputation]], [[predictd]], [[chromimpute]], [[edice]], [[cross-cell-type-generalization-pitfall]], [[encode-imputation-challenge]], [[imputation-evaluation-measures]]
