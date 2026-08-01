---
type: wiki
title: Read processing and artefact regions
summary: Alignment, filtering, deduplication and blacklisting — the pipeline steps between FASTQ and signal, and the ones that silently create distributional shifts.
category: concept
sources: raw/langmead-2012-bowtie2.xml, raw/li-2009-bwa.xml, raw/mckenna-2010-gatk.pdf, raw/amemiya-2019-encode-blacklist.xml, raw/schreiber-2023-encode-imputation-challenge.pdf
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# Read processing and artefact regions

Nothing in this pipeline looks like a modelling decision, and yet the ENCODE Imputation Challenge was invalidated by one line of it.

## Alignment

- **Bowtie 2** (`raw/langmead-2012-bowtie2.xml`) combines the FM (full-text minute) index — fast and memory-efficient for ungapped matching — with hardware-accelerated dynamic programming for gapped extension, giving gapped alignment at FM-index speed. Used for ATAC-seq in the ENCODE pipeline.
- **BWA** (`raw/li-2009-bwa.xml`) performs backward search over the Burrows–Wheeler transform, with inexact matching by bounded backtracking over suffix-array intervals. Used for histone ChIP-seq in the ENCODE pipeline.

## Filtering and deduplication

The ENCODE convention (`raw/schreiber-2023-encode-imputation-challenge.pdf`): remove unmapped reads and mates, non-primary alignments, reads failing platform/vendor quality checks, and PCR/optical duplicates (`-F 1804`); remove multi-mapping reads (MAPQ < 30); mark duplicates with **Picard MarkDuplicates** (`raw/mckenna-2010-gatk.pdf` describes the GATK MapReduce framework the surrounding toolchain is built on) and remove them. For ATAC-seq specifically, 5′ ends of filtered reads are shifted +4 bp on the plus strand and −5 bp on the minus strand to account for the Tn5 insertion offset.

**The deduplication asymmetry.** For single-end datasets, a single read is chosen from each duplicate set. For paired-end datasets, a read-pair is kept if *either* read in the pair is unique. This is standard practice for both, and the challenge organisers describe it as having "unintended consequences": it produced a distributional shift between the older single-end training data and the newer paired-end test data that was large enough to make the [[average-activity-baseline]] beat all but two of 23 submissions. See [[distributional-shift-and-batch-effects]].

## Artefact regions

`raw/amemiya-2019-encode-blacklist.xml` defines the **ENCODE blacklist** (now "exclusion list"): regions in human, mouse, worm and fly with anomalous, unstructured, or high signal in NGS experiments **independent of cell line or experiment**. It is built from ENCODE input (control) BAMs, merged per donor for human; for each 1 kb bin (100 bp overlap) it computes reads per mappable base and multi-mapping reads per million, quantile-normalises across bins, and flags bins whose 50th-percentile standard value is extreme. The 50% quantile is chosen to avoid both high-signal outliers from individual cell types (e.g. copy-number variants) and low signal from failed or mislabelled inputs. Mappability comes from **Umap**.

The paper's position is that blacklist removal is "an essential quality measure." Both the challenge's ATAC/DNase and ChIP-seq pipelines filter peaks overlapping it.

## Why this page matters for modelling

Every one of these steps is a covariate: aligner, MAPQ threshold, deduplication mode (single- vs paired-end), Tn5 shift, blacklist version, pipeline version. When they vary across a compendium and are not recorded alongside the signal, they become confounded with biology. The challenge's recommendation is to ensure processing steps have been **uniformly applied to raw data** and that the data were collected using similar procedures — advice that is only actionable if one starts from reads rather than from processed tracks.

## See also

Related:: [[chip-seq-assay-and-controls]], [[peak-calling-and-signal-tracks]], [[encode-imputation-challenge]], [[distributional-shift-and-batch-effects]]
