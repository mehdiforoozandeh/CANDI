---
type: wiki
title: Chromatin state annotation (SAGA)
summary: ChromHMM and Segway summarise multi-assay epigenomic data into labelled genome segmentations; their annotations are a primary downstream consumer of imputed tracks, and are irreproducible more often than their posteriors suggest.
category: method
sources: raw/ernst-2012-chromhmm.xml, raw/hoffman-2012-segway.xml, raw/shahraki-2024-robust-chromatin-state-annotation.xml, raw/boix-2021-regulatory-genomic-circuitry.xml, raw/ernst-2015-chromimpute.xml, raw/lin-2025-epiverse.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
---

# Chromatin state annotation (SAGA)

SAGA is the reason imputation quality matters downstream: a segmentation consumes many assay tracks at once, so an imputed track's errors propagate into region-level biological calls.

## The two methods

- **ChromHMM** (`raw/ernst-2012-chromhmm.xml`) — a multivariate HMM over **binarised** chromatin-modification tracks, learning a state model whose emission parameters describe which marks co-occur in each state. Its framing is that chromatin state annotation from combinations of modification patterns had already proven powerful for discovering regulatory regions, characterising their cell-type-specific activity, and interpreting disease-association studies; the contribution is automating model learning across many datasets and cell types.
- **Segway** (`raw/hoffman-2012-segway.xml`) — a **dynamic Bayesian network** over continuous signal, applied to 31 ChIP-seq, DNase-seq and FAIRE-seq assays in K562. Unsupervised, it recovered patterns corresponding to transcription start sites, gene ends, enhancers, CTCF elements, and repressed regions. Notable design details: signal is `asinh`-transformed to compress the dynamic range (see [[count-distributions-for-sequencing-data]]); observation tracks are **weighted** so that tracks with different numbers of data points contribute comparably to the likelihood; a countdown variable enforces minimum/maximum segment lengths; inference uses EM training and Viterbi decoding via GMTK.

The design contrast is instructive: ChromHMM binarises and gains speed and interpretability; Segway keeps continuous signal and gains resolution and the ability to model segment duration.

## Reproducibility is the weak point

`raw/shahraki-2024-robust-chromatin-state-annotation.xml` evaluates both methods on replicate pairs across five cell types and finds predictions "frequently irreproducible" — roughly 80% bin-level agreement between annotations of two replicates of the same cell type, well below the 90–95% threshold most applications assume. Two causes are separated: **multiple similar states** that the model cannot confidently distinguish, and **spatial misalignment** of segment boundaries. Posterior probabilities do not flag either. SAGAconf assigns calibrated confidence scores instead — see [[uncertainty-calibration]].

## Imputation feeds SAGA

- `raw/ernst-2015-chromimpute.xml` used imputed data to delineate chromatin states across all 127 reference epigenomes — the point of imputing in the first place was to make a uniform segmentation possible where observed data were missing. See [[chromimpute]].
- `raw/boix-2021-regulatory-genomic-circuitry.xml` (EpiMap) built chromatin states from observed **and** imputed tracks across 833 biosamples, then derived 2.1 million high-resolution enhancer annotations, 300 enhancer modules, and trait-relevant tissue predictions for 20,000 GWAS loci. It reports that imputed datasets clustered more cleanly and were less affected by technical covariates than observed ones — the strongest published argument that imputation improves downstream annotation rather than merely substituting for missing data.

## Imputed tracks as an input layer

`raw/lin-2025-epiverse.xml` (EpiVerse) is a concrete demonstration of imputation as infrastructure. Its pipeline **begins with Avocado**, uses the imputed epigenomic signals as input, and predicts cross-cell-type **Hi-C contact maps** — with chromatin-state prediction folded in as a second task in a multitask framework. It reports that building on imputed rather than observed inputs improves cross-cell-type Hi-C accuracy, because the imputed tracks are available for all cell types and marks rather than only where experiments exist.

This makes the quality argument concrete in a way a correlation metric does not: errors in an imputed track propagate into 3-D genome predictions and chromatin-state calls downstream, so imputation accuracy has consequences beyond the track itself.

## See also

Related:: [[reference-epigenome-compendia]], [[uncertainty-calibration]], [[epigenome-imputation]], [[chromimpute]], [[count-distributions-for-sequencing-data]]
