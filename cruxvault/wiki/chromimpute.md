---
type: wiki
title: ChromImpute
summary: Ernst & Kellis 2015: per-target ensembles of regression trees over correlated marks and samples; the first large-scale epigenome imputation, and the only imputation-based denoiser that avoids per-target retraining.
category: method
sources: raw/ernst-2015-chromimpute.xml, raw/roadmap-2015-111-reference-epigenomes.pdf, raw/durham-2018-predictd.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# ChromImpute

ChromImpute's architectural quirk — one independently trained model per target experiment — is simultaneously its scaling weakness and the single property that makes it usable as a denoiser.

> Scope note: "denoises without retraining" is a claim *within the imputation lineage*, where the alternative ([[predictd]], [[avocado]]) is to hold the target out and refit the whole factorisation. Purpose-built denoisers reach the same property by other routes — see [[epigenome-denoising]] for AtacWorks generalising to unseen cell types and Coda transferring across individuals, cell types and species.

## Method

For each target (sample, mark) pair, ChromImpute trains an **ensemble of regression trees** whose features are drawn from (a) the same mark in other samples, and (b) other marks in the same sample, at the target position and in a window around it (`raw/ernst-2015-chromimpute.xml`). Prediction is at 25 bp resolution. Because each target has its own model and feature set, adding or denoising one experiment does not require touching any other.

## Scale and results

Applied to 127 reference epigenomes (111 Roadmap, `raw/roadmap-2015-111-reference-epigenomes.pdf`, plus 16 ENCODE), ChromImpute produced **4,315 imputed signal tracks**, of which 26% were also experimentally observed and could therefore be checked directly.

The headline claim is that imputed tracks **exceed** the observed data they mirror on three axes: consistency (between replicates and related samples), recovery of gene annotations, and enrichment for disease-associated variants. The paper leans on this to argue imputation is a data-quality tool, not just a gap-filler — the origin of the "impute to denoise" practice described in [[epigenome-imputation]].

Secondary uses demonstrated in the paper: detecting low-quality experimental datasets (large observed–imputed discrepancy flags a suspect experiment), finding loci with unexpected epigenomic signal, prioritising which new experiments to run, and supplying inputs for chromatin-state annotation across the 127 epigenomes — see [[chromatin-state-annotation]].

## Position relative to successors

`raw/ernst-2015-chromimpute.xml` frames imputation by analogy to missing-value imputation in microarrays and genotype imputation in GWAS. [[predictd]] and [[avocado]] both benchmark against ChromImpute and report lower MSE; `raw/durham-2018-predictd.xml` additionally reports that **combining** PREDICTD and ChromImpute predictions beats either alone, indicating the two make partly uncorrelated errors — a per-target discriminative model and a global tensor factorisation are not redundant.

The cost of the per-target design is training expense: a complicated model-and-training procedure tuned per experiment, which is exactly the property [[predictd]] set out to remove.

## See also

Related:: [[epigenome-imputation]], [[predictd]], [[avocado]], [[average-activity-baseline]], [[chromatin-state-annotation]], [[epigenome-denoising]]
