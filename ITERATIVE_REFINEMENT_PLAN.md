# Iterative Refinement & Feedback Loops for EpiDenoise
*Drafted for Future Performance Optimization Experiments*

## 1. Problem Statement & Motivation

**Current State:**
EpiDenoise currently operates as a standard "one-shot" Denoising Autoencoder (DAE).
- **Training:** Input is corrupted (masked). Model predicts original values.
- **Inference:** Input is corrupted (missing values). Model predicts original values in a single forward pass.

**Hypothesis:**
A single pass might produce a result that is "better" than the input but still noisy or inconsistent. Since the model is trained to map "corrupted data" to "clean data," its output should theoretically be closer to the true data manifold than its input.
**Therefore:** Feeding the model's *own output* back into itself as a new input (Iterative Refinement) could progressively denoise the sample, leading to higher accuracy and better structural consistency (e.g., peak shapes, correlations).

This concept is conceptually related to:
- **Chain-of-Thought** (in LLMs, where intermediate steps refine the final answer)
- **Diffusion Models** (which are essentially many-step denoising autoencoders)

*However*, our specific approach is mathematically better described by:
- **Plug-and-Play Priors (PnP) / Fixed-Point Imputation** (for Scenario A)
- **Recurrent Inference Machines (RIMs)** (for Scenario B)

---

## 2. Theoretical Scenarios

We can implement this idea in three distinct ways, ranging from "easy to test" to "theoretically optimal."

### Scenario A: Inference-Only (Plug-and-Play Priors / Fixed-Point Imputation)
*The "Free Lunch" Approach*

**Concept:**
Train the model normally. At inference time, instead of stopping after one pass, we run a loop:
$X_0 \to \text{Model} \to X_1 \to \text{Model} \to \dots \to X_T$

**Mathematical Perspective:**
This acts as a **Projection onto the Data Manifold** (PnP). The pre-trained DAE acts as a "Prior" that projects noisy inputs onto valid genomic structures. The loop enforces consistency with the observed data (the "Constraint") while letting the model denoise the missing parts.

**Mechanism (Critical: "Input Replacement"):**
If we just feed the raw output back, the model might "drift" away from the known ground truth (hallucination). We must enforce consistency with the observed data.
1. Predict: $\hat{X} = \text{Model}(X_{t})$
2. **Replacement Step:** Construct $X_{t+1}$:
   - For **Observed** positions: Keep original values from $X_{0}$ (Ground Truth).
   - For **Missing** positions: Use predicted values from $\hat{X}$.
     - *Note:* The model predicts $\mu$ directly, so use $\mu$ as the predicted mean count.
     - **Important:** Do NOT manually apply `arcsinh` transform to the feedback values. The encoder's `forward` method (lines 864-867 in `model.py`) automatically applies `arcsinh` to all non-missing values (anything not marked as `-1`). Just provide raw counts.
3. Repeat.

**Pros:**
- No retraining required.
- Can be tested immediately on existing checkpoints.

**Cons:**
- **Distribution Shift:** The model was trained on "Masked" inputs (values like -1/-2). It has never seen "Complete" (dense, continuous) inputs. Feeding a dense prediction back might confuse it, leading to artifacts.

---

### Scenario B: Post-Training Fine-Tuning (Recurrent Inference Machine)
*The "Learned Refinement" Approach*

**Concept:**
Take the current pretrained model and fine-tune it to learn the refinement process.
We effectively teach the model: "Here is your first guess (which is okay but imperfect). Now, take that guess as input and make it better."

**Mathematical Perspective:**
This formally implements a **Recurrent Inference Machine (RIM)**. Instead of just projecting once, the model learns an iterative update rule (a "trajectory") to minimize the residual error over time. It learns *how* to correct its own specific artifacts.

**Mechanism:**
1. Freeze (or use low LR for) the main encoder/decoder.
2. Training Loop (Unrolled):
   - $X_0$ (Masked) $\to$ Model $\to$ $\hat{X}_1$
   - $\hat{X}_1$ (Input Replacement + Gradient Flow) $\to$ Model $\to$ $\hat{X}_2$
   - ...
   - Loss is computed on $\hat{X}_T$ (or sum of losses at all steps).
3. Backpropagate through time (BPTT).

**Pros:**
- Fixes the Distribution Shift problem from Scenario A.
- Model explicitly learns to correct its own residuals.
- Faster convergence than training from scratch.

**Cons:**
- Requires a second training phase (Fine-tuning).
- Computational cost of training increases linearly with steps $T$.

---

### Scenario C: End-to-End Recurrent Pretraining
*The "Deep Equilibrium" Approach*

**Concept:**
Train the model from scratch as a Recurrent Neural Network (RNN) where the hidden state is the output canvas.

**Pros:**
- Theoretically optimal. The model learns a "trajectory" of denoising from pure noise to clean signal.

**Cons:**
- Very expensive to train.
- High risk of vanishing/exploding gradients.
- Likely overkill compared to Scenario B.

---

## 3. Constraint Strategies
*How to prevent drift and integrate predictions with observations*

The core challenge of iterative refinement is balancing **Denoising** (trusting the model) vs. **Data Consistency** (trusting the observations).

We can range from "No Constraint" to "Strict Locking."

### Strategy 1: No Constraint (Pure Feedback)
**Concept:** Feed the model's output back as the next input, replacing *everything* (even observed values).
- **Pros:** Maximum denoising. The model can rewrite the entire sample to be consistent with the manifold.
- **Cons:** **Drift.** The model can hallucinate features that drift far away from the actual experiment.
- **Verdict:** **Too Risky.** Not recommended.

### Strategy 2: Hard Input Replacement (The Baseline)
**Concept:**
$X_{new} = \text{Mask} \cdot X_{obs} + (1-\text{Mask}) \cdot X_{pred}$
- **Pros:** Zero drift. The observed data is treated as ground truth.
- **Cons:** **No Denoising of Observations.** It assumes the observed data is perfect. If the input is noisy, the noise persists.
- **Verdict:** **Good Starting Point.** Safe and easy to implement.

### Strategy 3: Soft Data Consistency (Proximal Update)
**Concept:**
$X_{new} = (1-\lambda) \cdot X_{pred} + \lambda \cdot X_{obs}$
- **Math:** This is a **Proximal Gradient Step**. We maximize the likelihood of the prediction while minimizing the distance to the observation.
- **Pros:** Allows denoising of observed values while using the observation as a "tether" to prevent drift.
- **Cons:** Requires tuning $\lambda$ (e.g., 0.1 vs 0.5).
- **Verdict:** **Robust.** Good for noisy low-coverage data.

### Strategy 4: Variance-Weighted Fusion (The "Kalman" Approach)
**Concept:**
Merge prediction and observation based on their respective uncertainties (Precisions).
$$ X_{new} = \frac{w_{pred} \cdot X_{pred} + w_{obs} \cdot X_{obs}}{w_{pred} + w_{obs}} $$
Where $w_{pred} = n$ (Model Confidence) and $w_{obs} = \text{ReadDepth}$ (Data Confidence).
- **Pros:** **Statistically Optimal.**
    - If Model is confident ($n$ is high) $\to$ Trust Model (Aggressive Denoising).
    - If Model is unsure ($n$ is low) $\to$ Trust Data (Prevent Drift).
- **Cons:** Slightly more complex to implement.
- **Verdict:** **Best Theoretical Approach.** Solves the "Confidence Dilemma."

### Strategy 5: Stochastic Re-Masking (Gibbs Sampling / MICE)
**Concept:**
1. Fill missing values with prediction.
2. Randomly mask 15% of the *entire* sequence (different spots).
3. Feed back.
- **Precedent:** Used in BERT pretraining and Multiple Imputation by Chained Equations (MICE).
- **Pros:** **Prevents Distribution Shift.** The model always sees "masked" data, so it never faces Out-of-Distribution inputs.
- **Cons:** Slower convergence (stochastic noise).
- **Verdict:** **Essential for Stability.** Combine with Strategy 2 or 4.

### Strategy 6: Curriculum Infilling ("Easy-First")
**Concept:**
Iteratively lock in the top $k\%$ most confident predictions.
- **Pros:** Builds "islands of certainty" first.
- **Cons:** If the model is confidently wrong early on, the error is locked forever.
- **Verdict:** **Risky.** Prone to "Confirmation Bias."

## 4. Metadata Handling Strategy

When we fill in the missing data with predictions, we face a choice: **Do we also reveal the metadata?**

### Recommendation: "Assertive Unmasking"
We should **reveal the true metadata** (Assay, Platform, Runtype) for the filled-in positions.

**Why?**
A blurry prediction might look like "generic signal." By revealing the metadata (e.g., "This is H3K4me3"), we give the model the context needed to sharpen the peak or adjust the distribution to match that specific assay's profile.

**Implementation in the Loop:**
1.  **Step $t$:** Predict $\hat{X}$ using `mX_masked`.
2.  **Update Step:**
    *   $X_{t+1}$: Fill missing slots with $\hat{X}$ (predicted mean).
    *   $mX_{t+1}$: **Replace masked metadata tokens (-2) with TRUE metadata values.**
3.  **Step $t+1$:** Feed ($X_{t+1}$, $mX_{t+1}$) back into the model.

**Fallback:**
If "Assertive Unmasking" causes artifacts (because the model isn't used to seeing "Predicted Data + True Metadata" together), revert to keeping the metadata masked (Conservative Mode).

### Feature Request: Explicit Assay ID
Currently, the model uses `Platform` and `Runtype` but not the explicit `Assay ID` (e.g., "H3K27ac", "CTCF") as an input feature.
**Suggestion:** Add `Assay ID` to the metadata embedding. This would give the model a much stronger signal for refinement.
- *Input:* One-hot or learned embedding of the specific assay type.
- *Benefit:* The model knows exactly *what* it is refining, rather than inferring it from the signal shape.

---

## 5. Distribution Collapse Strategy
*How to convert NB Parameters $(\mu, n)$ to a Scalar Value*

The model predicts a Negative Binomial distribution parameterized by $(\mu, n)$ where $\mu$ is the mean and $n$ is the dispersion parameter. Internally, the model converts $\mu$ to $p$ (via $p = n/(n+\mu)$) to match PyTorch's Negative Binomial NLL loss implementation, but the raw predictions are $\mu$ and $n$. To feed this back as input, we must collapse this distribution into a single representative scalar value.

### Options:
1.  **Mean ($\mu$):** Use $\mu$ directly (already predicted by the model). The expected value.
2.  **Mode:** The most likely integer count.
3.  **Median:** The 50th percentile.
4.  **Sampling:** Randomly drawing $x \sim NB(n, p)$ where $p = n/(n+\mu)$.

### Recommendation: Use the Mean ($\mu$)
**Why?**
-   **Stability:** The mean is a smooth, differentiable function of the parameters. Sampling adds stochastic noise which can destabilize the feedback loop (unless you specifically want "Walk-Jump" sampling).
-   **Signal-to-Noise:** For low-count genomic data (like ChIP-seq), the **Mode** is often 0, even if there is a weak signal. The **Mean** preserves small fractional probabilities (e.g., 0.1) which allows the model to accumulate evidence over iterations.
-   **Consistency:** The input data (after `arcsinh` transform) is effectively treated as continuous intensity. The Mean is the best continuous estimator of this intensity.
-   **Convenience:** The model already predicts $\mu$ directly, so no conversion is needed.

**Implementation Note:**
**Do NOT manually apply `arcsinh` transform.**
-   The encoder class (`CANDI.encode` method in `model.py`) contains logic to automatically apply `arcsinh` to any value that is NOT marked as missing (`-1`).
-   Therefore, when you feed back the predicted mean $\mu$, you should provide it as a **raw count**.
-   The model will see that it is not `-1` (missing) and will apply the transform itself. Double transformation would distort the data.

### 5.1. Advanced Strategy: Uncertainty-Gated Feedback (Min % Error)

While the **Mean ($\mu$)** provides the best point estimate, blindly feeding back predictions from uncertain regions can compound errors (hallucinations). We can use the Negative Binomial dispersion parameter ($n$) to gate this process.

**The Metric: Min % Error**
As established, standard deviation scales with signal, making it poor for thresholding. We use the asymptotic coefficient of variation:
$$ \text{Min \% Error} = \sqrt{\frac{1}{n}} $$
This represents the "noise floor" or intrinsic uncertainty of the prediction.

**Implementation:**
Instead of filling *all* missing values, we apply a **Confidence Mask**:
1. Calculate $\mu$ and $n$ for all missing positions.
2. Calculate $\text{Error} = \sqrt{1/n}$.
3. **Threshold Strategy:**
   - If $\text{Error} > \text{Threshold}$ (e.g., 20% or 0.2), **DO NOT** fill this value (leave as missing/-1) or fill with a damped value.
   - If $\text{Error} \le \text{Threshold}$, fill with $\mu$.
   
**Benefit:** This prevents the model from treating a "low-confidence guess" as "ground truth" in the next iteration, stabilizing the trajectory.

---

## 6. Recommended Experimental Plan (Roadmap)

We should follow a "Fail Fast" approach. Start with the cheapest experiment.

### Phase 1: Proof of Concept (Inference-Only)
**Goal:** Determine if the model has *any* natural capacity for self-correction without retraining.

1.  **Select a Validation Set:** Pick a representative chromosome (e.g., chr19) or a specific difficult locus.
2.  **Run Inference Loop (Scenario A):**
    -   Implement the "Input Replacement" loop (keep observed, update missing).
    -   **Important:** Unmask the metadata (reveal true values) for the filled positions.
    -   **Important:** Use `Mean` for collapsing NB distribution.
    -   **Important:** Feed back raw counts (no manual `arcsinh`).
    -   Run for `T=1, 3, 5, 10` iterations.
3.  **Metrics:**
    -   MSE / Pearson Correlation vs. Ground Truth at each step.
    -   **Confidence Evolution:** Track the average "Min % Error" across iterations. Does the model become more "certain" of its predictions?
    -   Check for "Posterior Collapse" (does the output become blurry or uniform?).
4.  **Advanced Tweak (Gibbs Sampling / Re-masking):**
    -   If the model struggles with dense inputs, try **Re-masking**:
    -   Prediction $\to$ Fill Missing $\to$ **Randomly Mask 10% of filled values** $\to$ Feed back.
    -   This keeps the input looking like the "masked" training data.

### Phase 2: Fine-Tuning (If Phase 1 is promising but unstable)
**Goal:** Stabilize the refinement process by teaching the model to handle its own outputs.

1.  **Dataset:** Use standard training split.
2.  **Config:**
    -   Load best checkpoint.
    -   Set iterations `T=3`.
    -   Loss = `Loss(Step 1) + Loss(Step 2) + Loss(Step 3)`.
3.  **Run for:** Short duration (e.g., 5-10 epochs).
4.  **Evaluate:** Compare `Step 1` performance (Base) vs `Step 3` performance (Refined).

---

## 7. What to Expect & Risks

| Outcome | Interpretation | Action |
| :--- | :--- | :--- |
| **Performance improves, then plateaus** | **Ideal.** The model converges to a better solution. | Deploy as standard inference mode. |
| **Performance degrades immediately** | **Distribution Shift.** Model hates dense inputs. | Try "Re-masking" (Phase 1b) or move to Fine-tuning (Phase 2). |
| **Predictions oscillate/diverge** | **Unstable Dynamics.** | Need to add a "step size" (like a learning rate) to the update: $X_{new} = \alpha \cdot Pred + (1-\alpha) \cdot X_{old}$. |
| **Output becomes blurry** | **Posterior Collapse.** | Model is averaging out uncertainty. Stop early (fewer steps). |
| **Min % Error decreases but MSE increases** | **Delusional Confidence.** | Model is reinforcing its own hallucinations. Increase the "Min % Error" threshold for feedback (be more conservative). |

## 8. Summary of Next Steps

1.  [ ] **Create a script `test_iterative_inference.py`**
    -   Load a trained model.
    -   Load one batch of validation data.
    -   Apply mask.
    -   Loop 5 times: Predict $\to$ Use predicted $\mu$ $\to$ Replace Missing Slots $\to$ **Unmask Metadata** $\to$ Feed Back (Raw).
    -   Plot correlations at step 1, 2, 3, 4, 5.
2.  [ ] **Analyze Results:**
    -   Does correlation go up?
    -   Does the visual quality (peaks) improve?
3.  [ ] **Decide:**
    -   If Good: Integrate into `inference.py`.
    -   If Bad: Plan for **Phase 2 (Fine-tuning)**.
