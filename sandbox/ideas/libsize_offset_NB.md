# NB Count Modeling with Library-Size Offset

Status: concept note (`E29`, not yet run)  
Date: 2026-05-25  
Context: Negative binomial NLL for binned count data (e.g. ChIP-seq, ATAC-seq, RNA-seq)  
Checklist: [EXPERIMENTS.md](EXPERIMENTS.md#experiment-sweep-e29) · Experiment idea: [idea_e29_libsize_offset_nb.md](idea_e29_libsize_offset_nb.md)

## Summary

When modeling sequencing counts with a negative binomial (NB), do **not** ask a neural network to predict the raw expected count $\mu$ directly. Instead:

1. Treat **library size** $s_s$ as a **known offset** (fixed per sample).
2. Have the network predict **log enrichment** $\eta_{sb}$ and **dispersion** $N_b$.
3. Construct the NB mean as $\mu_{sb} = s_s \cdot e^{\eta_{sb}}$ inside the loss.

This separates sequencing depth from biological signal and matches the standard GLM formulation used in DESeq2, edgeR, and scVI.

---

## Generative model

For sample $s$, bin or peak $b$, and observed count $x_{sb}$:

$$
x_{sb} \sim \mathrm{NB}(\mu_{sb}, N_b)
$$

with variance

$$
\mathrm{Var}(x_{sb}) = \mu_{sb} + \frac{\mu_{sb}^2}{N_b}.
$$

The mean is linked to a **library-size offset** and a **normalized rate**:

$$
\log \mu_{sb} = \log s_s + \eta_{sb}
\qquad \Leftrightarrow \qquad
\mu_{sb} = s_s \cdot e^{\eta_{sb}}.
$$

| Symbol | Role | Who provides it |
|--------|------|-----------------|
| $x_{sb}$ | Observed count | Data |
| $s_s$ | Size factor / library size / sequencing depth | **Known input** (not learned as biology) |
| $\eta_{sb}$ | Log enrichment (normalized signal) | **Network output** |
| $N_b$ | Dispersion (overdispersion beyond Poisson) | **Network output** (or empirical-Bayes estimate) |

Training loss:

$$
\mathcal{L} = \sum_{s,b} \mathrm{NB\_NLL}\!\left(x_{sb} \,\middle|\, \mu_{sb} = s_s \cdot e^{\eta_{sb}},\, N_b\right).
$$

### What $N$ means (and does not mean)

$N_b$ quantifies **overdispersion** around $\mu$, not confidence in $\eta$. Larger $N$ → variance closer to Poisson ($\mathrm{Var} \to \mu$). It is largely a property of the locus/assay, not sequencing depth. Depth enters through $s_s$, not through $N_b$.

For neural training, parameterize $(\mu, N)$ rather than $(p, N)$: apply softplus to $\hat\mu$ and $\hat N$, or equivalently predict $\eta$ and $\log N$.

---

## Precedent: DESeq2 and edgeR

### Formulation

In DESeq2 (Love, Huber & Anders, 2014) and edgeR (Robinson et al., 2010; McCarthy et al., 2012), gene or peak counts are modeled as

$$
Y_{gi} \sim \mathrm{NB}(\mu_{gi}, \phi_g),
\qquad
\log \mu_{gi} = \mathbf{X}_i^\top \beta_g + \log s_i,
$$

where:

- $s_i$ is a **size factor** (normalized sequencing depth for sample $i$),
- $\mathbf{X}_i^\top \beta_g$ is the **log fold-change / design contribution** (biology),
- $\phi_g$ (or $1/N$) is a **dispersion** parameter estimated per feature with shrinkage.

Equivalently:

$$
\frac{\mu_{gi}}{s_i} = e^{\mathbf{X}_i^\top \beta_g},
$$

so the GLM models **counts per unit depth**, not raw counts.

### Motivation in DESeq2

Raw read counts scale **linearly with sequencing depth**. If depth is not accounted for, a deeply sequenced sample appears to have higher expression or enrichment everywhere — a **technical artifact**, not biology.

DESeq2's offset formulation:

1. **Separates technical depth from biological signal** — $\beta_g$ is comparable across samples with different library sizes.
2. **Enables valid differential analysis** — contrasts are on normalized rates, not raw counts.
3. **Improves dispersion estimation** — overdispersion $\phi_g$ is estimated on a scale where technical depth has already been removed.
4. **Uses established size-factor normalization** — median-of-ratios or similar methods estimate $s_i$ from the count matrix itself.

The same logic applies to ChIP-seq peak counts (e.g. via DiffBind, csaw), where window or peak counts are NB with normalization offsets.

---

## Precedent: scVI

**Yes — scVI uses the same factorization**, extended to a deep generative model.

In scVI (Lopez et al., 2018, *Nature Methods*), gene counts follow (zero-inflated) NB:

$$
x_{ng} \sim \mathrm{NB}(\mu_{ng}, \theta_g),
$$

where $\theta_g$ is inverse dispersion (analogous to our $N_b$). The mean is

$$
\log \mu_{ng} = \log \ell_n + h_\theta(z_n, g),
$$

where:

- $\ell_n$ is the **library size** for cell $n$ (by default the observed total UMI count, or a learned latent with prior),
- $h_\theta(z_n, g)$ is the **decoder output** from a latent representation (biology),
- $\theta_g$ is gene-specific dispersion.

So scVI's decoder predicts a **normalized expression scale**; depth is applied via $\log \ell_n$ exactly like a GLM offset. The scvi-tools implementation exposes this via `size_factor_key` and the `(mu, theta)` parameterization of the NB likelihood.

Related chromatin models (e.g. PeakVI for scATAC-seq) use the same **technical vs biological factorization** — cell-wise scaling $\ell_i$ for depth times a biological rate — even when the likelihood is Bernoulli rather than NB.

---

## Alternative considered: depth as network input, predict raw $\mu$

A natural alternative is to pass $s_s$ as an input feature and predict $\mu_{sb}$ directly:

$$
\mu_{sb} = h_\theta(\text{features},\, s_s),
\qquad
\mathcal{L} = \sum_{s,b} \mathrm{NB\_NLL}(x_{sb} \mid \mu_{sb}, N_b).
$$

The network sees depth and is free to learn any function $\mu = h(\cdot, s)$.

### Why the offset formulation is better

| Aspect | Offset: $\mu = s \cdot e^{\eta(\cdot)}$ | Input $s$, predict raw $\mu$ |
|--------|-------------------------------------------|-----------------------------------|
| Depth effect | **Fixed**: coefficient 1 on $\log s$ | **Learned** — must be discovered from data |
| Identifiability | Biology $\eta$ and depth $s$ separated by construction | Can confound depth with condition/batch |
| Generalization across depth | Same $\eta$ at new depth: $\mu = s_{\mathrm{new}} \cdot e^{\eta}$ | May fail at unseen depth if net memorized depth–count mapping |
| Output interpretability | $\eta$ or $e^{\eta}$ = enrichment per unit depth | Raw $\mu$ not comparable across samples without dividing by $s$ |
| Learning difficulty | $\eta$ on log scale, smaller dynamic range | Raw counts span orders of magnitude |
| Shortcut risk | Low — cannot explain high counts by depth alone | High — “high $s$ → high $\mu$” lowers NLL on background bins |
| Literature alignment | DESeq2, edgeR, scVI | Ad hoc; must re-derive normalization properties |

The alternative is **not wrong in principle**: a sufficiently flexible $h_\theta$ *can* learn $\mu \approx s \cdot e^{\eta}$. But it must **rediscover** a constraint the offset **builds in**. Extra freedom mostly buys confounding and poor extrapolation across sequencing depths, not better biology.

### When the alternative is acceptable

- All samples have **nearly identical depth** (offset and direct $\mu$ collapse).
- The task is **absolute count forecasting** for a fixed assay depth, not cross-sample normalized enrichment.
- The architecture **hard-codes** the offset anyway, e.g. $\mu = s \cdot \mathrm{softplus}(f(\text{features}))$ with $s$ passed only as a multiplier — which is the offset formulation in disguise.

---

## Recommended implementation sketch

```text
Per sample s:
  s_s  ← size factor (DESeq2 median-of-ratios, total reads, spike-in, etc.)

Per bin b, sample s:
  η_sb ← network(sequence, condition, batch, …)     # log enrichment
  N_b  ← softplus(network) or empirical-Bayes prior  # dispersion

  μ_sb = s_s * exp(η_sb)
  loss += NB_NLL(x_sb | μ_sb, N_b)
```

Do **not** fold sequencing depth into $N_b$. Depth → $s_s$ in the mean. $N_b$ → locus-specific overdispersion beyond Poisson sampling.

---

## References

- Love, Huber & Anders (2014). *Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2.* Genome Biology.
- Robinson, McCarthy & Smyth (2010). *edgeR: a Bioconductor package for differential expression analysis.* Bioinformatics.
- Lopez et al. (2018). *Deep generative modeling for single-cell transcriptomics.* Nature Methods (scVI).
- Rashid et al. (2011). *ZINBA: Zero-inflated negative binomial algorithm.* Genome Biology (ChIP-seq counts).
- Lun & Smyth (2016). *csaw: a Bioconductor R package for differential binding analysis of ChIP-seq data.* NAR (NB window counts with normalization).
- StatOmics notes: [Working with RNA-seq count data — GLM offsets](https://statomics.github.io/SGA/sequencing_countData.html).
