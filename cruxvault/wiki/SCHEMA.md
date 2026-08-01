---
type: wiki_schema
title: Wiki schema
---

# Wiki schema — conventions for THIS vault's literature wiki

Co-evolved by you (the PI) and the agent. The global rules live in the `crux-wiki`
skill; this file records the choices specific to this project.

## What the wiki is
A literature background layer: prior methods, SOTA, baselines, datasets, definitions —
compiled once from the immutable sources in `raw/`, then kept current. It exists to
sharpen `ask` / `hypothesize` and to interpret findings. It is **not** a record of this
project's own results.

## Flow rule (hard)
Literature → wiki → informs the tree. **Never** the reverse. A wiki page may link other
wiki pages; it must never cite a q/h tree node. Findings never enter the wiki.

## Page conventions
- One concept / entity / comparison per page; concept-slug filenames (`film-conditioning.md`).
- Frontmatter: `title`, `summary` (one line — becomes the index entry), `category`
  (entity | concept | method | comparison | dataset | overview | …), `sources`
  (comma-separated `raw/…` paths every claim traces to).
- Write for the LLM reader: dense and explicit over pretty.

## Categories in use
- `overview` — the framing page for a whole problem area (currently: epigenome imputation).
- `method` — one named method, tool, or technique (ChromImpute, MACS, FiLM, quantile normalisation).
- `comparison` — several related methods held against one axis (sequence-conditioned models).
- `concept` — a phenomenon, failure mode, or class of technique that no single paper owns
  (distributional shift, calibration, masked SSL, evaluation measures).
- `dataset` — a corpus, compendium, benchmark panel, or annotation resource.
- `entity` — reserved; not yet used.

## Scope of this vault's wiki
Compiled from the union of the CANDI manuscript's bibliography and the reference list of
the ENCODE Imputation Challenge paper (Schreiber et al. 2023), plus a small set of
architecture/SSL/calibration papers that CANDI's design rests on but does not yet cite.
The wiki covers **prior art only** — the imputation lineage, the assay and processing
stack that generates the data, normalisation and evaluation methodology, and the ML
primitives. It deliberately does not describe CANDI itself.

## Domain conventions for this project
- **Signal target.** Nearly all prior imputation work predicts −log10 p-value tracks at
  25 bp; when a page says "signal" without qualification it means that. Raw counts, fold
  enrichment, and peak calls are named explicitly. See `peak-calling-and-signal-tracks`.
- **Assay naming.** Histone marks by standard name (H3K4me3); DNase-seq and ATAC-seq are
  accessibility assays and are kept distinct because their pipelines differ (Tn5 shift).
- **`arcsinh`** is the default variance-stabilising transform in this domain; note it when
  a source uses `log1p` or quantile instead, since preprocessing choice is a real axis of
  method variation.
- **Cell type vs biosample.** Sources use both; prefer the source's own term on its page
  and note the mapping where a compendium merges biosamples into cell types.

## Maintenance notes
- A source is compiled when at least one page declares it in `sources:`. `crux validate`
  reports uncompiled sources — treat that as the work queue after any ingest.
- Prefer extending an existing synthesis page over adding a page per paper. A page earns
  its place only if it adds cross-source value beyond paraphrasing one source.
