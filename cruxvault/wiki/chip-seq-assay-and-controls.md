---
type: wiki
title: ChIP-seq assay design and controls
summary: ENCODE/modENCODE working standards: antibody validation, replication, the mandatory input control, and why the control channel is the reference every enrichment measure is defined against.
category: concept
sources: raw/landt-2012-chip-seq-guidelines.xml, raw/zhang-2008-macs.xml, raw/bonhoure-2014-chip-seq-spiking.xml, raw/jung-2014-sequencing-depth-chip-seq.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# ChIP-seq assay design and controls

The control (input) experiment is not a nicety — every enrichment statistic in this literature is a comparison against it, so a ChIP-seq track without its control is not interpretable on an absolute scale.

## The consortium standards

`raw/landt-2012-chip-seq-guidelines.xml` is the ENCODE/modENCODE working-standards document. Its scope: antibody validation, experimental replication, sequencing depth, data and metadata reporting, and data-quality assessment. Key positions:

- **Antibody specificity governs everything.** The quality of a ChIP experiment is set by antibody specificity and the enrichment achieved in the affinity precipitation step. Epitope tagging is offered as an alternative when good antibodies are unavailable, sidestepping antibody variation and cross-reactivity.
- **Two biological replicates.** Initial Pol II experiments showed more than two replicates did not significantly improve site discovery, which is why the two-replicate convention persists.
- **A control is critical.** DNA breakage during sonication is not uniform; open-chromatin regions are preferentially represented in the sonicated sample, and there are platform-specific sequencing-efficiency biases. Without a control these produce apparent enrichment that is purely technical.

## What the control is used for

- **Peak calling.** MACS scales the total control tag count to match the ChIP tag count and uses a **dynamic local λ** estimated from the control to capture regional background biases (`raw/zhang-2008-macs.xml`). See [[peak-calling-and-signal-tracks]].
- **Normalisation.** The ChIP-to-input ratio in background windows is the scale factor *r* that several normalisation methods estimate — see [[signal-normalization-in-epigenomics]].
- **Fold enrichment and p-value tracks.** Both are defined relative to the control's local Poisson expectation.

Because the control is a property of the *sample*, not of the target assay, it is available for every ChIP-seq experiment in a biosample regardless of which histone marks were profiled — which makes it the one channel that is never missing.

## Where the control is insufficient

`raw/bonhoure-2014-chip-seq-spiking.xml` notes the input control cannot detect a **uniform genome-wide** change in occupancy, because such a change alters ChIP and background proportionally; an exogenous spike-in is needed. And `raw/landt-2012-chip-seq-guidelines.xml` observes that for factors with a small total genomic target, tags mapping to the true target are a small percentage of all tags — so signal-to-noise, not depth alone, limits detection. See [[sequencing-depth-and-coverage]] and `raw/jung-2014-sequencing-depth-chip-seq.xml`.

## See also

Related:: [[peak-calling-and-signal-tracks]], [[sequencing-depth-and-coverage]], [[signal-normalization-in-epigenomics]], [[read-processing-and-artifact-regions]], [[reference-epigenome-compendia]]
