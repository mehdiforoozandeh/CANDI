# CANDI: Confidence-Aware Neural Denoising Imputer for Epigenomic Data

## Introduction

In recent years, the availability of large-scale functional genomic data such as histone modifications and DNA accessibility has provided unprecedented opportunities to understand the functional roles of diverse genomic loci. However, a major confounding factor is that measurements obtained using sequencing methods often suffer from various sources of noise, including batch effects, technical variability, and biological heterogeneity.

A promising approach for addressing issues of noise is epigenome imputation. Epigenome imputation methods aim to predict the output of a functional genomics experiment. Due to the high cost and complexity of profiling every possible assay in all relevant cell and tissue types, researchers have turned to computational methods to predict missing data, including ChromImpute, Avocado, eDICE and others.

Epigenome imputation methods were originally designed to predict unperformed assays, but researchers have shown that imputed data often has better properties than observed data, even when such observed data sets are available. By integrating patterns across experiments, cell types, and genomic loci, imputation models average out noise distilling consistent and biologically meaningful signals into less noisy predictions. Thus, researchers frequently apply imputation for denoising by re-imputing each assay before inputting the assay into downstream analysis.

However, existing approaches for imputation-based denoising have significant limitations. Most significantly, all existing imputation methods operate on idealized processed signal (for example, fold enrichment over control). They assume that this processing removes all batch effects and results in an idealized "signal strength." This issue jeopardized the recent ENCODE Imputation Challenge, which aimed to evaluate epigenome imputation methods comprehensively. Upon receiving entries from all participants, the organizers found that, due to subtle differences between the train and test sets, a simple baseline outperformed most of the entrants.

Furthermore, to denoise a given experiment using current approaches, one must typically re-train the model without that particular experiment and then impute it *de novo*. This usually would require re-training the underlying machine learning model; thus, the only existing method that can be applied for denoising in practice is ChromImpute, whose learning architecture allows this process to be performed without re-training. This existing strategy also entirely disregards the target assay, which could provide valuable information towards denoising.

To address these issues, we propose CANDI (Confidence-Aware Neural Denoising Imputer), a method for epigenome imputation that:
1. Predicts raw counts, processed signal values, and peak locations while handling experiment-specific covariates such as sequencing depth, read length, etc.
2. Can (optionally) incorporate information from a low-quality existing experiment when predicting a target without retraining
3. Outputs calibrated measures of uncertainty for all prediction types

This approach is enabled using self-supervised learning (SSL), a paradigm that capitalizes on large amounts of unlabeled data by corrupting and then reconstructing subsets of the input to learn without explicit labels. This strategy enables zero-shot imputation and denoising, allowing models to generalize to new cell types without retraining.

---

## Results and Discussion

### 1. Self-supervised, confidence-aware denoising imputation of genomic data

We propose CANDI, a method for epigenome imputation.

Briefly, CANDI works as follows (Methods):
For a given 30kb locus, CANDI takes as input:
1. Observed epigenomic data sets (read counts) for the locus in the target sample along with ChIP-seq control signal
2. Four experimental covariates for each observed assay (sequencing depth, read length, run type, sequencing platform)
3. The DNA sequence of the locus

It also takes as input the experimental covariates of the desired outputs.

CANDI outputs predicted epigenomic data sets for the given locus and sample in three formats:
1. **Raw read counts** as a negative binomial distribution per genomic position
2. **Continuous signal** in log p-value units as a Gaussian distribution per genomic position
3. **Peak calls** as binary classification probabilities per genomic position

All outputs are given in the form of probability distributions, enabling calibrated uncertainty quantification.

CANDI consists of a neural network model that includes convolutional, Transformer and deconvolutional layers with per-layer metadata cross-attention (Methods). We trained CANDI using a self-supervised learning approach by optimizing three complementary objectives:
1. **Full assay masking**: We masked entire assays and asked CANDI to predict these masked assays from remaining available ones (mimicking imputation)
2. **Full loci masking**: We masked the same genomic positions across all assays and asked CANDI to predict these masked positions (mimicking language modeling)
3. **Denoising via downsampling**: We simulated low-quality data by downsampling reads from training tracks and asked CANDI to predict the original high-depth data from low-quality observations (mimicking denoising)

**Figure 1:** CANDI architecture overview. (A) Input data organization showing the three-dimensional structure of epigenomic data across assays, genomic positions, and cell types. (B) Model architecture consisting of encoder and decoder components. The encoder processes DNA sequence and epigenomic count data through parallel Conv1D towers with per-layer metadata cross-attention, integrates experimental covariate information, and generates a latent representation using a transformer encoder with rotary positional embeddings. Three separate decoders predict count data (negative binomial parameters), signal values (Gaussian parameters), and peak locations (binary classification) for each assay and position. (C) Detailed architecture of key model components including Conv1D towers, transformer encoder blocks, metadata cross-attention, and distribution output layers.

### 2. CANDI accurately imputes missing epigenetic signals

We evaluated CANDI's imputation performance against top performers from the ENCODE Imputation Challenge (EIC). Despite being significantly more parameter-efficient (~42 million parameters compared to billions in models like Avocado) and cell-type agnostic, CANDI achieves competitive performance. Unlike previous approaches that rely on learning specific cell-type embeddings—limiting their applicability to unseen cell types—CANDI utilizes experiment-specific metadata (covariates), enabling zero-shot generalization to new biological contexts without retraining.

In terms of global genome-wide correlation, CANDI significantly outperforms all EIC competitors in Spearman correlation across most assays. However, it underperforms in Pearson correlation. This discrepancy highlights a key characteristic of the model: while CANDI is highly effective at recovering the correct rank-ordering and structural patterns of epigenomic signals (high Spearman), it tends to underestimate the absolute magnitude of high-signal regions (lower Pearson). This is evident in the scatter density plots, where predicted values in arcsinh space show strong monotonic correlation with observations but a compressed dynamic range. Comparing against the 'QNorm' and 'Reprocessed' baselines of the EIC, which attempt to correct for train-test covariate shifts, further confirms that CANDI's structural imputation is state-of-the-art, even if signal magnitude scaling remains a challenge.

Beyond correlation, we assessed prediction quality using peak classification metrics. CANDI demonstrates exceptional performance in distinguishing signal from background, achieving near-perfect AUCROC (~1.0) for marks like H3K4me3, even where Pearson correlation was moderate (~0.35). This suggests that while exact signal values may be compressed, the biological signal-to-noise ratio is preserved, allowing for accurate peak calling.

### 3. CANDI provides calibrated aleatoric uncertainty estimates

A core innovation of CANDI is its ability to output probability distributions rather than point estimates, providing a measure of aleatoric uncertainty. Standard evaluation metrics like MSE or Pearson correlation collapse these distributions into their means, discarding valuable information about what the model "knows it doesn't know."

We evaluated the fidelity of these uncertainty estimates using confidence calibration curves. A perfectly calibrated model would show a 1:1 relationship between the predicted confidence interval (e.g., 95%) and the empirical coverage (the fraction of observed data points falling within that interval). We observed that calibration varies by assay type; chromatin accessibility assays (like DNase-seq) generally show better calibration than histone modifications. Typically, the model tends to be overconfident at lower confidence intervals (<0.5) but becomes more reliable or even conservative at practically relevant intervals (0.9–0.95).

To further quantify the quality of these distributions, we utilized the Concordance Index (C-index), which evaluates whether the predicted distributions correctly rank-order the data while accounting for uncertainty. Genome-wide C-index scores were generally high. While H3K4me3 showed a lower genome-wide C-index (~0.6), its performance rose to ~0.83 in promoter regions, where the mark is biologically relevant.

Visualizing the predicted Coefficient of Variation (CV = $\sigma/\mu$) reveals that CANDI assigns higher uncertainty to regions where it predicts low signal but the ground truth might be high. In these "missed" regions, the model effectively flags its own potential error by outputting high variance, a capability absent in deterministic MSE-trained models.

### 4. CANDI latent space and imputed signals predict gene expression

To validate the biological utility of CANDI's predictions, we assessed whether they could predict gene expression levels (RNA-seq log TPM) better than raw data. We extracted features from transcription start site (TSS), gene body, and transcription end site (TES) using four sources: (1) Observed signals (available assays only), (2) Denoised signals (available assays), (3) Denoised + Imputed signals (all 35 assays), and (4) CANDI's Latent Embeddings (Z).

Our results show that denoised signals marginally outperform raw observed signals, indicating that CANDI successfully removes noise while preserving regulatory information. More importantly, the full set of 35 denoised and imputed assays outperforms the subset of available denoised assays, confirming that the imputation process adds meaningful regulatory context missing from the sparse input.

Most strikingly, the best predictive performance was achieved using CANDI's latent embeddings (Z). This suggests that the latent space encodes rich, biologically relevant information—possibly higher-order regulatory logic or chromatin states—that is not fully captured when decoded back into low-dimensional signal tracks. Furthermore, while the predictive power of observed and denoised signals depends heavily on the number of available input assays, the performance of the Imputed+Denoised and Latent settings is remarkably robust to input sparsity, demonstrating CANDI's ability to compensate for missing data.

---

## Methods

### Data Collection and Processing

We trained and evaluated CANDI using two datasets.

#### ENCODE Imputation Challenge Dataset

We used data from the ENCODE Imputation Challenge (EIC), featuring 35 distinct assays measured across 50 biosamples. For each experiment, we obtained aligned sequencing reads (BAM files), signal p-values (BigWig files), peak calls (BED files), and experimental covariates, including sequencing depth, sequencing platform, read length, and run type (single- or paired-end). In the original ENCODE Imputation Challenge, the dataset contained only BigWig files of signal p-values. However, since we use read counts as input to our model, we obtained raw read files (BAM) corresponding to the same experiments and biosamples in the EIC. We followed the original train-validation-test split proposed by the EIC to ensure comparability with previous benchmarks.

#### Extended ENCODE Dataset

To have a more extensive dataset, we systematically collected data for all biosamples in the ENCODE database that contained at least one experiment from the 35 assays of interest. This initial collection yielded 3,064 biosamples, forming a sparse 3,064 × 35 experiment availability matrix. To address this sparsity while maintaining biological relevance, we implemented a cell type-based merging strategy. This merging strategy significantly improved data density, increasing the median number of available assays per sample from 1 (per biosample) to 6 (per merged cell type). The final processed dataset consists of 361 merged cell types.

Our merging strategy groups biosamples by their annotated cell type (biosample term name in ENCODE) and further organizes them based on isogenic replicate relationships. Within each cell type, we identify sets of biosamples that are isogenic replicates (derived from the same donor) and share at least three assays in common—these form replicate groups. Each replicate group is assigned a group identifier (grp), and individual biosamples within a group are labeled as replicates (rep). For example, `cardiac_muscle_cell_grp1_rep1` and `cardiac_muscle_cell_grp1_rep2` represent two isogenic replicates from the same cardiac muscle cell type that share a common set of experiments. Biosamples that lack isogenic replicate annotations or do not share sufficient experiments with other samples are aggregated into a single "non-replicate" sample (e.g., `right_lobe_of_liver_nonrep`), where experiments are selected to maximize consistency of donor, lab, and biosample accession. This strategy preserves biological relationships while maximizing the number of assays available per merged sample for training.

#### Control Integration

For each biosample containing ChIP-seq experiments, we obtained the corresponding ChIP-seq control (input) experiments. These control signals are provided as an additional always-available input channel to the model, forming an (F+1)-dimensional input where feature F represents the control signal. Controls are never masked during training, serving as a stable reference for batch effect normalization.

#### Downsampling

To simulate varying sequencing depths, we implemented read downsampling in BAM files. The Downsampling Factor (DSF) determines the fraction of reads retained: DSF 1 retains all reads, DSF 2 randomly samples 50% of reads, and DSF 4 randomly samples 25% of reads. The default DSF list used during training is [1, 2, 4].

### CANDI Architecture

Let A = 35 represent the number of assay types, and let G = 30,000 bp and R = 25 bp denote the length of the genomic window and the resolution, respectively, such that the window contains L = G / R = 1,200 genomic positions at 25 bp resolution. Each sample (i.e., one data instance in a batch) corresponds to epigenomic data from a single cell type, forming a matrix **M** of shape A × L for that cell type. We process multiple samples in a batch of size N, where each sample has its own **M**, **S**, and covariates.

The model receives:
- **M** ∈ ℝ^(N × L × (A+1)) — Epigenomic read counts including control channel
- **S** ∈ {0,1}^(N × 4 × G) — One-hot encoded DNA sequence
- **C_in** ∈ ℝ^(N × 4 × (A+1)) — Input experimental covariates
- **C_out** ∈ ℝ^(N × 4 × A) — Output experimental covariates

#### Inputs

- **Epigenomic Reads M**: Each M_{l,a} is the read count for assay a at position l. The (A+1)-th channel contains ChIP-seq control signal. Input counts are transformed using arcsinh(x) before being fed to the encoder, which stabilizes variance across the dynamic range of read counts.
- **DNA Sequence S**: One-hot encoded nucleotides (A, C, G, T) across the G = 30,000 bp region.
- **Covariates C_in and C_out**: Four values per assay: log₂(sequencing_depth), run_type (single/paired-end), read_length, and sequencing_platform.

#### Outputs

For each assay a at each position l, the model predicts parameters of probability distributions:
- (n̂_{a,l}, p̂_{a,l}) ∈ ℝ^(N × L × A) — Negative Binomial parameters for raw counts
- (μ̂_{a,l}, σ̂²_{a,l}) ∈ ℝ^(N × L × A) — Gaussian parameters for processed signal p-values
- peak_{a,l} ∈ [0,1]^(N × L × A) — Peak probability scores

The choice of probability distributions is motivated by the statistical properties of each output type. For raw read counts, we use the negative binomial distribution, which is well-established for modeling sequencing count data due to its ability to capture overdispersion—a common characteristic of epigenomic experiments where variance exceeds the mean. The Poisson distribution is a special case of the negative binomial with fixed dispersion, but empirical studies have shown that epigenomic count data typically exhibit overdispersion, making the negative binomial more appropriate. For processed signal values, we predict arcsinh-transformed p-values following previous methods for variance stabilization. We model these transformed values with a Gaussian distribution, predicting both the mean (μ) and variance (σ²). Unlike standard MSE optimization, which implicitly assumes constant variance across all predictions, our approach allows the model to express heteroscedastic (position-varying) uncertainty. By jointly optimizing both parameters via Gaussian negative log-likelihood, the model learns to output higher variance in regions where predictions are inherently more uncertain—capturing aleatoric uncertainty that reflects irreducible noise in the data rather than model limitations.

#### Model Overview

The model consists of an encoder ε and three separate decoders D_count, D_signal, D_peak:

```
ε(M, S, C_in) → Z
D_count(Z, C_out) → (n̂, p̂)
D_signal(Z, C_out) → (μ̂, σ̂²)
D_peak(Z, C_out) → peak
```

#### Encoder ε

The encoder integrates epigenomic data **M**, DNA sequence context **S**, and experimental covariates **C_in** into a latent representation **Z** ∈ ℝ^(N × L' × d_model).

1. **Epigenetic Signal Convolution Tower**: Processes the epigenomic signals with n_cnn = 3 depth-wise separable convolution layers. Each layer applies convolution with kernel size 3, followed by batch normalization, GELU activation, and average pooling with pool_size = 2. This progressively reduces the spatial dimension from L = 1,200 to L' = 150 while expanding the feature dimension by expansion_factor = 3 at each layer. After each convolution layer, a MetadataCrossAttention module conditions the features on experimental covariates using FiLM (Feature-wise Linear Modulation).

2. **DNA Sequence Convolution Tower**: Processes the one-hot encoded DNA sequence with n_cnn + 2 = 5 convolution layers. The first three layers use pool_size = 2, and the final two layers use pool_size = 5 to match the epigenomic resolution (aligning 30,000 bp to L' = 150 positions). This produces DNA features **S**^(final) ∈ ℝ^(N × L' × d_model).

3. **Metadata Cross-Attention**: At each convolutional layer, experimental covariates modulate the learned features through cross-attention. For each assay, we embed the four covariates (depth, run_type, read_length, sequencing platform) into a query vector. This query attends to the spatial features of that assay, producing scale and shift parameters for FiLM conditioning: output = scale ⊙ features + shift.

4. **Fusion**: The epigenomic and DNA features are concatenated along the feature dimension and projected back to d_model dimensions through a linear layer with layer normalization.

5. **Transformer Encoder**: Applies n_sab = 4 transformer encoder layers with n_head = 9 attention heads. We use rotary positional embeddings (RoPE) via the x-transformers library for improved length generalization. Each layer consists of multi-head self-attention followed by a feed-forward network with GELU activation and dropout = 0.1.

#### Decoders D

Given latent representation **Z** and output covariates **C_out**, three separate decoders reconstruct the output distributions:

1. **Output Covariate Embedding**: Output covariates are embedded using the same MetadataCrossAttention mechanism as the encoder, providing the model with information about the desired output experimental conditions.

2. **Deconvolution Towers**: Each decoder employs n_cnn = 3 transposed convolution layers that invert the encoder's pooling and feature compression. After each deconvolution layer, MetadataCrossAttention conditions the features on output covariates, allowing the model to generate outputs appropriate for specific experimental conditions.

3. **Distribution Output Layers**:
   - **NegativeBinomialLayer**: Two linear projections produce n̂ (via softplus for positivity) and p̂ (via sigmoid for [0,1] range)
   - **GaussianLayer**: Two linear projections produce μ̂ (via softplus for positivity) and σ̂² (via softplus for positivity)
   - **PeakLayer**: A linear projection followed by sigmoid produces peak probabilities in [0,1]

### Learning Objectives and Training Strategy

Our approach leverages self-supervised learning, where the model learns to reconstruct deliberately corrupted versions of its own input. We employ three complementary masking strategies that can be combined with configurable probabilities:

#### (1) Full Assay Masking

We randomly mask 1 to (num_available - 1) complete assays for each cell type. Both the data AND metadata for selected assays are masked, simulating the imputation scenario where some experiments are entirely missing. At least one assay always remains available per cell type to provide context.

#### (2) Full Loci Masking
We randomly select contiguous chunks of genomic positions (default chunk_size = 40 positions, ~1kb) and mask these positions across ALL available assays simultaneously. This objective most closely parallels BERT-style masked language modeling, where the model learns to predict masked tokens from surrounding context. Here, genomic positions serve as "tokens," and the model must infer epigenomic signals at masked positions using flanking regions and DNA sequence. 

#### (3) Denoising via Upsampling

For assays that remain unmasked, the model is trained to reconstruct the original high-quality signal from potentially downsampled (noisy) observations. The observed regions provide the upsampling/denoising training signal.


#### Data Augmentation

We apply reverse complement augmentation with probability 0.5. When applied, both the DNA sequence and epigenomic signals are reversed along the genomic axis, and the DNA bases are complemented (A↔T, C↔G). This data augmentation helps the model learn strand-invariant representations.

#### Loss Function

We train CANDI by minimizing a weighted combination of negative log-likelihood losses:

```
L = w_obs × (w_count × L_count,obs + w_pval × L_pval,obs + w_peak × L_peak,obs)
  + w_imp × (w_count × L_count,imp + w_pval × L_pval,imp + w_peak × L_peak,imp)
```

Where:
- L_count = -log P_NB(y_count | n̂, p̂) — Negative binomial NLL for raw counts
- L_pval = -log P_G(y_signal | μ̂, σ̂²) — Gaussian NLL for signal p-values
- L_peak = BCE(peak, y_peak) — Binary cross-entropy for peak classification

Default loss weights: w_count = 0.5, w_pval = 1.0, w_peak = 0.1, w_obs = 0.25, w_imp = 1.0

The subscripts "obs" and "imp" denote losses computed on observed and imputed (masked) entries respectively.

#### Training Loci

We use multiple strategies for selecting training genomic loci, including:

- **cCRE (default)**: Focus on genomic regions containing candidate cis-regulatory elements, which are enriched with informative epigenomic signals
- **random**: Randomly selected non-overlapping regions across the genome
- **full_chr**: Complete coverage of specified chromosomes
- **gw (genome-wide)**: Full coverage across all autosomes and sex chromosomes

For all strategies, we exclude ENCODE blacklist regions (known problematic regions with anomalous signal accumulation) to ensure training data quality.

We trained CANDI on 5,000 randomly selected, non-overlapping regions, each spanning 30kb. We excluded chromosome 21, reserving it exclusively for testing purposes. In total, CANDI was trained on 150 million base pairs of genomic sequence, representing approximately 5% of the human genome.

#### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Context length (L) | 1,200 bins (30,000 bp) |
| Resolution (R) | 25 bp |
| Number of assays (A) | 35 |
| CNN layers (n_cnn) | 3 |
| Transformer layers (n_sab) | 4 |
| Attention heads (n_head) | 9 |
| Expansion factor | 3 |
| Pool size | 2 |
| Kernel size | 3 |
| Dropout | 0.1 |
| Optimizer | Adamax |
| Learning rate | 5 × 10⁻⁴ |
| Batch size | 90 |

### Imputation Performance

To evaluate imputation performance, we compute the following genome-wide metrics: Mean Squared Error (MSE) for average squared differences, Pearson Correlation for linear relationships, Spearman Correlation for monotonic relationships, and R² score between observed and imputed values. The main CANDI model is trained on the Extended ENCODE Dataset, but to ensure comparability with the ENCODE Imputation Challenge (EIC) competitors, we trained a version of CANDI using the EIC dataset and their specified train-test split. All performance evaluations were made on chromosome 21.

### Peak Prediction Performance

For peak prediction, we evaluate using standard binary classification metrics: precision, recall, and area under the ROC curve (AUROC). We compare predicted peak probabilities against observed peak calls from ENCODE narrowPeak files.

### Uncertainty Modeling

#### Confidence Calibration

To evaluate the reliability of the model's uncertainty estimates, we assess its calibration for all three output types. For counts and signal values, we measure the fraction of observed values that fall within the model's predicted C% confidence interval for C ∈ [0, 1]. For negative binomial predictions, confidence intervals are computed from the predicted (n, p) parameters. For Gaussian predictions, intervals are computed as μ ± z_{C/2} × σ. A well-calibrated model should exhibit a linear relationship between expected and observed coverage.

We visualize calibration using calibration curves, which plot the empirical coverage (fraction of observations within interval) against the nominal confidence level. For each confidence level c ∈ [0, 1], we compute the c% confidence interval from the predicted distribution and measure what fraction of true observations fall within this interval. A perfectly calibrated model produces a diagonal line (y = x), indicating that c% of observations fall within the c% confidence interval. 

#### Concordance Index

Beyond interval coverage, we evaluate whether the model's uncertainty estimates correctly rank predictions using the concordance index (C-index). For Gaussian predictions with mean μ and standard deviation σ, the C-index measures the probability that the model correctly ranks pairs of observations by their true values. Formally, for a pair of positions (i, j), we compute the probability that observation i exceeds observation j under the model:

\[
P(Y_i > Y_j) = \Phi\left(\frac{\mu_i - \mu_j}{\sqrt{\sigma_i^2 + \sigma_j^2}}\right)
\]

where Φ is the standard normal CDF. The C-index is then the AUC-ROC score comparing these predicted probabilities against the true binary labels (1 if y_i > y_j, 0 otherwise). A C-index of 0.5 indicates random ranking, while 1.0 indicates perfect discrimination. This metric assesses whether the model's distributional predictions meaningfully capture the relative ordering of true signal values.

### RNA-seq Prediction as Biological Validation

To validate that CANDI captures biologically meaningful information beyond simple signal reconstruction, we evaluate how well epigenomic features derived from the model's predictions can predict gene expression levels measured by RNA-seq. Importantly, CANDI has never seen any RNA-seq data during training—this evaluation tests whether the model's learned representations of epigenomic signals implicitly encode transcriptional regulatory information.

For each gene, we extract summary features from epigenomic signals around three key regions: the transcription start site (TSS), gene body, and transcription end site (TES). We use adaptive margins set to 10% of gene length for TSS and TES regions. From each region, we compute summary statistics including the median, interquartile range, minimum, and maximum signal values. We extract features from four signal sources: (1) observed epigenomic signals from available assays only, (2) denoised signals for assays where observations were provided as input, (3) denoised+imputed signals across all 35 assays—combining denoised predictions for available assays with imputed predictions for missing assays, and (4) latent representations Z extracted from CANDI's transformer encoder. For the latent representations, we compute the same summary statistics across each latent dimension for all three gene regions.

We employ nested cross-validation to rigorously evaluate predictive performance while avoiding overfitting. The outer loop uses 5-fold cross-validation to estimate generalization performance, while the inner loop performs hyperparameter selection via grid search with 4-fold cross-validation. Each fold trains a pipeline consisting of standardization, dimensionality reduction (PCA retaining 50-90% variance), and a regression model. We evaluate ridge and lasso regression, reporting Pearson correlation, Spearman correlation, and R² between predicted and observed log-transformed TPM values. If denoised signals outperform observed signals, this indicates the model successfully removes noise while preserving biologically relevant information. If denoised+imputed signals further improve predictions, this demonstrates that imputation adds meaningful regulatory context by filling in missing assays with biologically accurate predictions. The latent embeddings Z represent a rich, compressed representation of the epigenomic state that may contain biologically relevant information not explicitly decoded into signal predictions but potentially informative for expression prediction.



