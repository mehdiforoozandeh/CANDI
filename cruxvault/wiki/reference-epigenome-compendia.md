---
type: wiki
title: Reference epigenome compendia
summary: ENCODE, Roadmap, IHEC and EpiMap — the consortium datasets that define the cell-type × assay matrix, its sparsity, and its metadata conventions.
category: dataset
sources: raw/encode-2020-expanded-encyclopedias.xml, raw/roadmap-2015-111-reference-epigenomes.pdf, raw/stunnenberg-2016-ihec-blueprint.pdf, raw/bujold-2016-ihec-data-portal.pdf, raw/boix-2021-regulatory-genomic-circuitry.xml, raw/lindeboom-2021-human-cell-atlas.pdf, raw/schreiber-2023-encode-imputation-challenge.pdf, raw/moore-2026-ccre-v4.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T23:27:28
---

# Reference epigenome compendia

These are simultaneously the training data, the evaluation data, and the source of the confounds that make evaluation hard — the same portal that supplies the matrix also supplies its processing heterogeneity.

## The compendia

- **ENCODE phase III** (`raw/encode-2020-expanded-encyclopedias.xml`) — 5,992 new experimental datasets on top of phase II, spanning RNA transcription, chromatin structure and modification, DNA methylation, chromatin looping, and TF/RBP occupancy; a registry of 926,535 human and 339,815 mouse candidate cis-regulatory elements. All data flow through the ENCODE portal, which also re-serves Roadmap data.
- **Roadmap Epigenomics** (`raw/roadmap-2015-111-reference-epigenomes.pdf`) — 111 reference human epigenomes integrated with 16 ENCODE epigenomes to give the 127-epigenome set that [[chromimpute]] and [[predictd]] were built on.
- **IHEC** (`raw/stunnenberg-2016-ihec-blueprint.pdf`, `raw/bujold-2016-ihec-data-portal.pdf`) — the umbrella consortium coordinating reference epigenome production across ENCODE, NIH Roadmap, CEEHRC, Blueprint, DEEP, AMED-CREST and KNIH; its Data Portal exposes >7,000 reference datasets from >600 tissues under harmonised conventions. IHEC's contribution is as much *metadata standardisation* as data volume.
- **EpiMap** (`raw/boix-2021-regulatory-genomic-circuitry.xml`) — 3,030 observed tracks across 859 biosamples spanning 18 assays, plus 14,952 *imputed* tracks, yielding 833 high-quality reference epigenomes in 33 tissue categories. The definitive demonstration that imputation can be used as a compendium-completion strategy at production scale.
- **Human Cell Atlas** (`raw/lindeboom-2021-human-cell-atlas.pdf`) — the single-cell analogue, framed explicitly against the Human Genome Project; relevant as the direction the cell-type axis is heading.

## Structural facts that matter for modelling

**Sparsity is the point.** `raw/encode-2020-expanded-encyclopedias.xml` and `raw/roadmap-2015-111-reference-epigenomes.pdf` both describe matrices in which only a minority of (cell type, assay) pairs are filled, and the fill rate is extremely uneven — a handful of well-characterised cell lines carry dozens of assays while most biosamples carry one or two. `raw/schreiber-2023-encode-imputation-challenge.pdf` makes this the crux of its evaluation critique: performance on well-characterised cell types does not predict performance on poorly characterised ones, which is where imputation is actually useful. See [[cross-cell-type-generalization-pitfall]].

**Data accumulates over years, not at once.** Because a compendium is assembled over more than a decade, experiments differ in sequencing platform, read length, run type (single- vs paired-end), depth, and pipeline version. `raw/schreiber-2023-encode-imputation-challenge.pdf` traces a challenge-invalidating distributional shift to exactly this — see [[distributional-shift-and-batch-effects]].

**Assay definitions are consortium conventions.** The histone-mark panel, the requirement for a matched control, and the QC thresholds all come from consortium standards rather than from first principles — see [[chip-seq-assay-and-controls]].

## The registry has tripled

`raw/moore-2026-ccre-v4.xml` updates the candidate cis-regulatory element registry from the 0.9 million human / 300 thousand mouse cCREs described above to **2,373,014 human and 926,843 mouse cCREs** — covering 21% of the human and 9% of the mouse genome, a roughly threefold expansion. cCREs are classified into eight categories by distance to annotated TSS and by combinations of biochemical signals, and ENCODE4 additionally tested the activity of millions of regions with four types of functional assay (including STARR-seq), so a large fraction now carries experimental functional characterisation rather than chromatin-signature annotation alone.

For evaluation this matters in two ways: it is the current annotation against which a predicted peak set should be scored, and the functional-assay layer means agreement can be checked against measured activity rather than only against other chromatin signatures.

## See also

Related:: [[epigenome-imputation]], [[encode-imputation-challenge]], [[chromatin-state-annotation]], [[transcriptome-and-annotation-resources]], [[chip-seq-assay-and-controls]]
