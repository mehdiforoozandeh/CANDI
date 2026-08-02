---
type: wiki
title: eDICE
summary: Hawkins-Hooker et al. 2023: attention over observed tracks for epigenomic imputation, extended by transfer learning to individual-specific (donor-level) prediction.
category: method
sources: raw/hawkins-hooker-2023-edice.xml, raw/ernst-2015-chromimpute.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# eDICE

eDICE moves the field's granularity from *cell type* to *individual*, which reframes imputation as a precision-medicine tool rather than a compendium-completion tool.

## Method

eDICE is an **attention-based** deep model trained to impute missing epigenomic tracks by conditioning on the observed tracks for the same sample (`raw/hawkins-hooker-2023-edice.xml`). Instead of factorising the whole tensor into fixed per-axis factors ([[predictd]], [[avocado]]), attention lets the set of available tracks for a given sample determine, at inference time, how information is pooled — a better match for the fact that different biosamples have wildly different assay subsets.

The paper positions itself explicitly against [[chromimpute]] (`raw/ernst-2015-chromimpute.xml`) as the field's origin point and evaluates on the Roadmap-derived dataset used by prior imputation work, for direct comparability. Its own reading of the intervening decade is deflationary: the performance of the deep approaches "has only outstripped ChromImpute on a subset of metrics".

## The individual-specific claim

The distinguishing experiment uses epigenomes from **four individual donors**. With transfer learning across individuals, eDICE predicts individual-specific epigenetic variation **in tissues that are unmapped in a given donor** — i.e. it transfers a donor's epigenomic idiosyncrasies from the tissues where that donor was profiled to tissues where they were not.

The authors are explicit about the limitation: with four individuals, a robust analysis of individual-specific enriched-region overlap is not possible. The result is a demonstration of feasibility, not an established accuracy claim.

## Why the framing matters

The motivating argument in `raw/hawkins-hooker-2023-edice.xml` is that epigenetic modifications vary between individuals as well as between cell types, are shaped by environment, somatic mutation and ageing, and — unlike sequence — are **reversible**, hence therapeutically actionable. Mapping every tissue in every individual is infeasible, so per-individual imputation is the only route to individual-resolution epigenomes.

eDICE's authors were also participants in [[encode-imputation-challenge]], and the attention mechanism it uses is the same primitive discussed in [[transformers-and-positional-encoding]].

## See also

Related:: [[epigenome-imputation]], [[avocado]], [[chromimpute]], [[transformers-and-positional-encoding]]
