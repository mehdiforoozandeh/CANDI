---
type: wiki
title: ENCODE Imputation Challenge (EIC)
summary: The 2019 prospective benchmark of 23 imputation methods (retrospective analysis published 2023): its two-stage design, the 35-assay × 51-biosample panel, and its finding that a naive baseline beat almost every entrant.
category: dataset
sources: raw/schreiber-2023-encode-imputation-challenge.pdf, raw/encode-2020-expanded-encyclopedias.xml, raw/amemiya-2019-encode-blacklist.xml, raw/zhang-2008-macs.xml
created: 2026-07-31T21:26:00
updated: 2026-07-31T21:26:00
---

# ENCODE Imputation Challenge (EIC)

The challenge's lasting contribution is negative: it is the field's clearest demonstration that a genomics benchmark can be invalidated by processing heterogeneity that nobody in the loop suspected.

## Design

The challenge ran from **20 February to 14 August 2019**; the retrospective assessment was published in 2023. Two stages (`raw/schreiber-2023-encode-imputation-challenge.pdf`). Stage 1 ranked participants on a validation set of experiments drawn at random from within the data matrix. Stage 2 — the primary phase — used a **prospectively collected** blind test set: the test experiments were performed *during* the challenge, so no participant could have seen them. Test data came almost exclusively from **poorly characterised cell types** (only 3 of the 12 test cell types had more than two training experiments), deliberately targeting the regime where imputation matters.

23 models were submitted (up to three per team). Signal was provided as genome-wide **−log10 p-value tracks at basepair resolution**, computed with MACSv2 (`raw/zhang-2008-macs.xml`) against a local Poisson background. Data came from the ENCODE portal (`raw/encode-2020-expanded-encyclopedias.xml`).

Two details of that sentence are routinely misremembered, and both matter to anyone reconstructing the panel:

- **25 bp is an evaluation convention, not a data format.** "Although we provided the signal at basepair resolution, these measures were calculated at 25bp resolution." The nine pre-registered measures bin to 25 bp; the distributed tracks do not. CANDI's own 25 bp binning inherits the *measurement* convention, not the source data's resolution.
- **The Exclusion list was applied to the peak calls, not the signal.** "We filtered out all peaks that overlapped with the ENCODE Exclusion list" (`raw/amemiya-2019-encode-blacklist.xml`), while the signal tracks stayed genome-wide "at each basepair in the genome". Blacklist handling on the signal side is a separate operation the challenge did not perform.

## The method space

`raw/schreiber-2023-encode-imputation-challenge.pdf` (Table 1) organises submissions along three axes:

1. **Signal preprocessing** — almost every method further transformed the supplied p-values: `arcsinh`, `log1p`, quantile, or Cauchy.
2. **Input sources** — most used only assay measurements ("functional"); five used **nucleotide sequence**; eight used the average-activity baseline as an explicit input; three consumed [[avocado]]'s imputations.
3. **Tensor structure** — some modelled the tensor explicitly (deep tensor factorisation, e.g. `imp`, Lavawizard); others handled it implicitly via k-NN or rule-based similarity (e.g. the HLYG and NittanyLions entries).

Model families present: k-NN, deep tensor factorisation, autoencoders, CNNs, HMMs, and gradient-boosted trees.

## What went wrong, and the fix

Under the performance measures fixed *before* the challenge began, the [[average-activity-baseline]] outperformed all but two submissions, and those two only marginally. The cause was a **distributional shift** between the older single-end training data and the newer paired-end test data — traced not to paired-end data quality but to a minor difference in the **deduplication step** (for single-end data a single read is kept per duplicate set; for paired-end, a read-pair is kept if either mate is unique). After correcting for it, more than half the participants beat the baseline. See [[distributional-shift-and-batch-effects]] and [[read-processing-and-artifact-regions]].

## The three stated lessons

1. Processing differences across a compendium create distributional shifts that must be corrected before a fair comparison, and the correction must be **more than a simple rescaling**. The paper's recommendation is a quantile-normalisation approach that normalises **signal in peaks and signal in background separately** — see [[quantile-normalization]].
2. k-fold or leave-one-out cross-validation over a whole compendium over-weights well-characterised cell types. There is **no simple fix**; evaluation must explicitly include both well- and poorly characterised cell types. See [[cross-cell-type-generalization-pitfall]].
3. Performance measures designed without accounting for (1) and (2) become **redundant** with each other — scale-based measures converge as scale differences grow. See [[imputation-evaluation-measures]].

## As a benchmark panel

The EIC panel (35 assays across ~51 biosamples, with its published train/validation/test split) remains a standard benchmark for epigenome imputation, and the paper's recommendation — always compare against naive baselines — is its most quoted line.

## See also

Related:: [[epigenome-imputation]], [[average-activity-baseline]], [[imputation-evaluation-measures]], [[distributional-shift-and-batch-effects]], [[cross-cell-type-generalization-pitfall]], [[quantile-normalization]], [[peak-calling-and-signal-tracks]], [[reference-epigenome-compendia]]
