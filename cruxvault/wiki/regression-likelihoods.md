---
type: wiki
title: Regression likelihoods for continuous heads
summary: Gaussian, Laplace, Student-t and the robust-loss family as one continuous axis — what each assumes about residuals, and how the choice moves the calibration–sharpness trade-off.
category: comparison
sources: raw/barron-2019-robust-loss.pdf, raw/detlefsen-2019-variance-networks.pdf, raw/seitzer-2022-seitzer-betanll.pdf, raw/young-2024-ddpn.pdf, raw/rigby-2005-gamlss.xml
created: 2026-08-01T18:14:33
updated: 2026-08-01T18:14:33
---

# Regression likelihoods for continuous heads

Choosing a likelihood for a continuous head is choosing a residual distribution, and the usual
menu — Gaussian, Laplace, Student-t, Gamma — hides that they are points on a single axis with a
known parameterisation.

## They are one family, indexed by a shape parameter

`raw/barron-2019-robust-loss.pdf` is the unification. It presents a single loss with a shape
parameter α that **reproduces the standard losses as special cases**:

| α | loss | corresponding likelihood |
|---|---|---|
| 2 | L2 / squared error | Gaussian |
| 1 | Charbonnier / pseudo-Huber / L1–L2 | ≈ Laplace in the tails |
| 0 | Cauchy / Lorentzian | Cauchy (Student-t, ν = 1) |
| −2 | Geman–McClure | — |
| −∞ | Welsch / Leclerc | — |

Lower α means heavier tails and therefore a **flatter gradient on large residuals**: an outlier
stops pulling the mean once it is far enough away. That is the entire robustness story, and it is
monotone in α, which is why treating "Gaussian vs Laplace vs Student-t" as a discrete menu
obscures it.

The paper's second contribution matters more for practice: it derives a **probability
distribution** from the loss — with the partition function needed to make it normalisable — so α
becomes a free parameter that can be **learned by maximising likelihood** rather than selected by
sweep. A head that learns its own α is choosing its own robustness per dataset, and the sweep over
{gaussian, laplace, student_t} becomes a single continuous parameter.

The caveat: robustness is not free. Heavier tails buy insensitivity to outliers by *also* being
less certain about the bulk, so a Cauchy head is systematically less sharp than a Gaussian head on
clean data. This is the [[uncertainty-calibration]] sharpness-subject-to-calibration trade-off
appearing as a likelihood choice.

## The variance head is the fragile part, not the mean

Three sources converge on the same warning, and it applies to every entry in the table above once
the scale parameter is predicted rather than fixed.

`raw/seitzer-2022-seitzer-betanll.pdf` gives the mechanism: the NLL gradient with respect to the
mean is scaled by the **inverse predicted variance**, so regions the model currently believes are
noisy get down-weighted mean gradients, stay badly fit, and keep their high predicted variance.
`raw/detlefsen-2019-variance-networks.pdf` attacks the same problem from the estimation side —
proposing a heuristic for robustly fitting mean and variance networks **post hoc** rather than
jointly, and a network architecture with **extrapolation properties similar to a Gaussian
process**, on the observation that ordinary variance networks behave arbitrarily away from the
training data. Its framing is worth carrying: a learned variance is trustworthy only where there
were data, and nothing in the NLL enforces sensible behaviour elsewhere.

`raw/young-2024-ddpn.pdf` supplies the discrete counterpart — that Poisson and negative-binomial
heads **couple mean and dispersion**, so an NB can express over- but never under-dispersion. The
continuous analogue is that a Gaussian head with predicted σ² *can* decouple them, which is
precisely why it needs the two fixes above.

## Routing covariates to scale, not just location

`raw/rigby-2005-gamlss.xml` (GAMLSS) is the framework in which the choice becomes explicit rather
than implicit. A classical GLM lets covariates drive the **mean** and treats the remaining
distributional parameters as nuisance constants; GAMLSS lets **different covariates drive
location, scale and shape separately**, each through its own predictor.

For an experimentally conditioned model this is the statement that makes "sequencing depth belongs
in the mean, read length may belong in the dispersion" well-posed. It also frames Barron's α
correctly: α is a **shape** parameter, so under GAMLSS it too can be covariate-dependent — a
per-assay learned robustness rather than a global one.

## Choosing

- If residual outliers are **real signal you must not down-weight** (a true sharp peak), a heavy
  tail hurts; the Gaussian is right and the problem is the variance head.
- If they are **artefacts** (blacklist-adjacent spikes, amplification bias — see
  [[read-processing-and-artifact-regions]]), a heavy tail is doing the job a preprocessing step
  otherwise has to.
- If you cannot tell, that is the argument for learning α rather than choosing it.
- Whichever you pick, evaluate with a **strictly proper score** so sharpness and calibration are
  not being silently traded — an inconclusive comparison between likelihoods is often an
  evaluation artefact rather than a real tie.

## See also

Related:: [[uncertainty-calibration]], [[count-distributions-for-sequencing-data]], [[count-models-in-single-cell-genomics]], [[training-mechanics]], [[multi-task-optimization]], [[read-processing-and-artifact-regions]]
