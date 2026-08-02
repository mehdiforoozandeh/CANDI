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
- `digest` — a **filed-back query**: a page shaped like a question rather than a topic, written
  because answering it required reconstructing across several pages. See below.
- `entity` — reserved; not yet used.

## Digests — the query channel
A digest answers a question a CANDI researcher actually asks, by synthesising sources and pages
this vault already holds. Slug prefix `digest-`. Rules, which are what keep it inside the flow rule:
- It cites **only** `raw/…` paths and `[[wiki-slug]]` links. Never a q/h node, never a project result.
- It says what is **not** established as prominently as what is. A digest whose honest answer is
  "partial" is worth more than one that rounds up.
- It ends with what measurement or source would settle the open part.
- File one when a question took more than two or three page-opens to answer. That is the signal
  that the reconstruction is worth keeping.

The tree points *into* the wiki, never the reverse: question nodes carry a `Literature::` line of
`[[wiki/slug]]` links directly under `Parent::`. `crux validate` checks those resolve, and they
count as inbound links for orphan detection.

## Semantic lint — the agent's job, not the engine's
`crux validate` is mechanical only (broken/flow links, orphans, missing frontmatter, hash drift,
uncompiled sources). It cannot see contradictions, stale claims, or missing cross-references.
After each ingest round, run the semantic pass by hand and **append a `## [date] lint | …` entry to
`log.md`** — a wiki that only validates clean is indistinguishable from one nobody maintains. What
to check:
1. **Contradictions** — the same number or claim stated two ways on different pages. Flag with a
   `> **Note.**` block that reconciles them rather than silently picking one.
2. **Stale claims** — a page superseded by a newer source already sitting in `raw/`.
3. **Missing cross-references** — a new page links out to an old one without the reciprocal link back.
4. **Uncited declarations** — a source in `sources:` that no sentence in the body actually cites.
   The engine cannot catch this; it only checks the file exists.
5. **Unflagged gaps** — a claim resting on transferred rather than measured evidence. Mark these
   with an explicit `> **Gap note.**`; an unflagged gap reads as a settled answer.

## Fan-out
A source should be cross-woven into every page it bears on, not filed once. When a new cluster of
sources arrives, the failure mode is a sealed silo — a new page citing ten sources that appear
nowhere else. After each round, check the sources-to-pages map and backfill the frontier sources
in particular, which almost always bear on several existing pages.

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
- `raw/` is **gitignored** (public repo, non-redistributable sources). Every new source must be
  added to `tools/fetch_raw.py` in the same pass, or a fresh clone silently under-restores.
- Closed-access sources that cannot be fetched are listed at the top of `tools/fetch_raw.py` and
  carry an explicit `> **Gap note.**` on the page that needs them.
