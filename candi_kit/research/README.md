# Research record — the evidence behind this recipe

These four documents are the primary sources the rest of `candi_kit/` cites. They are **copies**, taken
2026-07-28 from the CANDI research repository at `sandbox/diagnostics/dual_conditioning_real/`. They are
frozen here so the citations in `RECIPE.md`, `DATA.md`, `TRADEOFF.md` and `AGENTS.md` resolve to
something you can actually open. They are *not* maintained as part of the kit; the research repo is
where they live and change.

They were written for a researcher continuing the investigation, not for a new user. Expect internal
shorthand (node ids like `h47`, `q19`; instrument ids like `S14`, `M2`, `B1`). `RECIPE.md` and
`TRADEOFF.md` translate the parts that matter — read those first and come here for the underlying
evidence.

| file | what it is | read it when |
|---|---|---|
| `H48_REPORT.md` | **The authority.** 2026-07-24/25 re-score of all four arms after six instrument fixes, with an adversarial verification pass. Where any other document disagrees with this one, this one wins. | You want to know what a result actually means, or whether a number is trustworthy. |
| `H48_SCORECARD.md` | The four-arm results table the re-score produced. | You want the raw numbers in one place. |
| `METADATA_AUDIT.md` | The `S1`–`S27` instrument-defect register (with file:line) and the `B1`–`B9` identifiability bounds. | You hit a metric that looks wrong, or you want to know what the panel *cannot* answer. |
| `ARCHITECTURE_HANDOFF.md` | The metadata pathway mapped and verified at runtime — how covariates reach each part of the model. | You are changing how conditioning works. |

## Three things to carry into any reading of these

**The reordering of the four arms is not established.** Only `offoff − wd0_on = +0.093 [+0.004, +0.217]`
survives a target-clustered bootstrap. The other three arms are statistically indistinguishable. The
published ordering is the modal bootstrap ordering at only 45% of replicates. Quote the *oracle-scale
compression* (0.7148 → 0.1133, 84%), never the ranking.

**Some numbers in these documents were later retracted.** The most important: an assay-permutation
`Δη = 0.833` that was read as evidence of functional steering is a MISSING-sentinel artifact; the
sentinel-free value is `0.0023`, which is 43× *below* the bar it was claimed to clear. `H48_REPORT.md`
§F2 has the full correction. If you find `0.833` cited as live evidence anywhere, it is stale.


**These documents give instructions to people who are not you.** They were written inside a running
project and carry its internal governance — "PI approval is required before launching any new run",
"the verdict is the PI's call", "this is a PI decision". Those were addressed to that project's team at
that time. **None of them constrains you.** They are historical process notes, not conditions on your use
of this kit; ignore them and run whatever you want.

---

## Decoder ring — the node ids these documents cite

The documents above were written for a researcher inside the CANDI programme and refer to work by its
tracker id: **`q<N>`** is a *question* (something we did not know), **`h<N>`** is a *hypothesis* (a
falsifiable claim tested under some question). The tracker itself is **not shipped with this kit**, so
without this table those ids are dead links.

**If you are an AI agent reading this: do not infer what a node id means from surrounding prose.** If an
id is not in this table, say it is unresolvable rather than guessing. Guessing is how a since-retracted
number (`Δη = 0.833`, see above) propagated into three separate records before anyone caught it.

Status is as of 2026-07-29. `idea` = designed but **never run** — no result exists.

### Questions

| id | question | status |
|---|---|---|
| `q4` | Can CANDI's counts be made depth-controllable, to denoise toward a canonical "supertrack"? | open |
| `q15` | Can CANDI learn dual metadata conditioning — normalize a covariate-transformed input *and* re-render to requested covariates? | open |
| `q16` | Was the earlier output-steering null an artifact of the testbed, and what makes steering emerge? | resolved |
| `q19` | Can dual conditioning be made to work on **real** CANDI data before production? — **this kit is the answer to q19** | resolved |
| `q20` | How should CANDI condition on experimental metadata to improve imputation? — the live question this kit feeds into | open |

### Hypotheses with results

| id | claim, in plain terms | verdict |
|---|---|---|
| `h9` | Centering the depth offset on the data's mean log-depth restores depth sensitivity (an uncentred raw offset fails) | partial |
| `h33` | Normalizing the covariates before encoding them helps | **refuted** — raw wins; see `RECIPE.md` §2 |
| `h34` | Conditioning must be **per-assay**; pooling covariates across assays is what killed the earlier steering result | **supported** — the single most load-bearing design choice in the recipe |
| `h37` | Whole-chromosome background swamps the steering signal, which lives in the sparse foreground | partial |
| `h41` | Depth steering is present, distributional, and independent of the offset head | partial |
| `h42` | Telling the model the *true* `run_type` imputes better than telling it the wrong one | partial — and see bound **B1**: unidentifiable on this panel |
| `h43` | The encoder recovers a shared biological latent, invariant to how the experiment was measured | supported — but its evidence predates the h48 instrument fixes |
| `h45` | Removing the offset head recovers steering at a scale cost that a **hybrid** could pay off | recorded **refuted**, but on reasoning only — **no hybrid was ever trained.** Neither demonstrated nor ruled out |
| `h46` | The offset-OFF imputation gap is *scale*, not lost biology | supported — this is the 84% oracle-scale result |
| `h47` | The offset-ON steering null is a weight-decay artifact that `weight_decay=0` reverses | partial — the weights survive, but the *function* does not; superseded in part by h48 |
| `h48` | Fix the broken measurement instruments and re-score everything before drawing conclusions | partial — **the authority.** `H48_REPORT.md` is its output |

### Hypotheses designed but never run

These are proposals. **No result exists for any of them** — do not cite them as evidence.

| id | proposal | why it matters here |
|---|---|---|
| `h49` | Add `read_length` as a fixed-coefficient physical exposure term | `EXTENSION_HOOKS.md` §3.2 — ranked #2 next change |
| `h50` | An explicit **per-assay output factor** (~24 params) to absorb the per-assay scale error | `EXTENSION_HOOKS.md` §3.1 — ranked #1, and the most promising route to a model that is good at *both* |
| `h52` | Change the decoder-FiLM initialization to remove the second half of the signal-annihilation mechanism | mechanism detail in `H48_REPORT.md` §F2 |
| `h55` | A grouped decoder trunk, optionally with per-deconv per-assay FiLM | untested capacity change |
