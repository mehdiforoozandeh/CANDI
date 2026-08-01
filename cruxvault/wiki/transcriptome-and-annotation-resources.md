---
type: wiki
title: Transcriptome and annotation resources
summary: GENCODE, GTEx, FANTOM5 and FANTOM6 — the gene models, expression references and promoter atlases used to define features and to validate epigenomic predictions biologically.
category: dataset
sources: raw/harrow-2012-gencode.xml, raw/gtex-2017-genetic-effects-gene-expression.xml, raw/fantom5-2014-promoter-level-expression-atlas.html, raw/ramilowski-2020-lncrna-functional-annotation.xml, raw/lindeboom-2021-human-cell-atlas.pdf, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/xiang-2020-s3norm.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Transcriptome and annotation resources

These resources supply the coordinates and the ground truth for any claim that an epigenomic prediction is biologically meaningful rather than merely numerically close.

## Gene models

**GENCODE** (`raw/harrow-2012-gencode.xml`) is the reference human annotation for ENCODE: a merge of HAVANA manual annotation with Ensembl automatic annotation. GENCODE 7 contained 20,687 protein-coding and 9,640 lncRNA loci, and 33,977 coding transcripts absent from UCSC genes and RefSeq. Completeness caveats stated in the paper: only **35% of transcription start sites are supported by CAGE clusters** and **62% of protein-coding genes have annotated polyA sites** — so "the TSS" is itself an estimate. This matters for any TSS-anchored feature: `raw/schreiber-2023-encode-imputation-challenge.pdf`'s promoter-correlation measure defines promoters as ±2 kb around GENCODE v38 gene starts (see [[imputation-evaluation-measures]]).

## Expression references

- **GTEx** (`raw/gtex-2017-genetic-effects-gene-expression.xml`) — gene expression across **44 human tissues** from non-diseased postmortem donors, with cis-eQTLs (variants within 1 Mb of the TSS), trans-eQTLs (673 genome-wide, 93 genes and 112 loci with inter-chromosomal effects), and allele-specific expression. Directly relevant to epigenomics: eVariants were **enriched in predicted promoter and enhancer chromatin states across all Roadmap cell types**, tying regulatory genetics to chromatin state (see [[chromatin-state-annotation]]).
- **FANTOM5** (`raw/fantom5-2014-promoter-level-expression-atlas.html`) — single-molecule CAGE mapping of transcription start sites and their usage across human and mouse primary cells, cell lines and tissues. Two findings that constrain how promoters should be modelled: few genes are truly housekeeping, and **many mammalian promoters are composite entities made of several closely separated TSSs with independent cell-type-specific expression profiles**. TSSs specific to different cell types evolve at different rates, while promoters of broadly expressed genes are the most conserved.
- **FANTOM6** (`raw/ramilowski-2020-lncrna-functional-annotation.xml`) — knockdown of 285 lncRNAs in human dermal fibroblasts with CAGE molecular phenotyping over 1,000+ libraries; establishes molecular phenotyping as a route to functional annotation of the noncoding transcriptome.
- **Human Cell Atlas** (`raw/lindeboom-2021-human-cell-atlas.pdf`) — the single-cell reference map effort, positioned as the successor to the Human Genome Project.

## Why this belongs in an imputation wiki

Predicting expression from epigenomic features is the standard external validation: it tests whether a predicted track carries the biological information the assay is supposed to measure, using data the model never saw. `raw/xiang-2020-s3norm.xml` and [[avocado]] both use gene-expression prediction as the downstream check, and `raw/gtex-2017-genetic-effects-gene-expression.xml` supplies the cross-tissue expression reference such a check needs.

## See also

Related:: [[reference-epigenome-compendia]], [[sequence-conditioned-epigenome-models]], [[imputation-evaluation-measures]], [[chromatin-state-annotation]]
