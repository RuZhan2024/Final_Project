

# The Note for the meeting

Focus: Full Workflow (Pose → Windows → TCN/GCN → Thresholds → Deployment)

## 0. Executive Summary

**Goal:** Prove the pipeline is reproducible, methodologically sound (leakage-free), and operationally realistic.

- **The Flow:** Raw video/image $\rightarrow$ MediaPipe Pose $\rightarrow$ Clean/Gate $\rightarrow$ Split (Subject-wise) $\rightarrow$ Window ($W=48, S=12$) $\rightarrow$ Train TCN/GCN $\rightarrow$ Fit Thresholds (Val) $\rightarrow$ Evaluate (Test).
- **The Key Defense:** We don't just "train a model"; we build a deployment system. We use **Operating Points (OPs)** to tune for safety (Recall) vs. usability (Alerts/24h) and explicitly handle real-world issues like FPS variance and overlapping detections.

## 1. Pipeline Overview & Architecture

### 1.1 The "Map" of the Project

Explain the physical structure of your data to show auditability.

- **`data/raw/`**: Original datasets.
- **`data/interim/`**: JSON/NPZ pose extraction and metadata normalization.
- **`data/processed/<dataset>/windows_W48_S12/`**: The final training artifacts. Each file is one sample.

### 1.2 Key Design Principles

- **Reproducibility:** Config-driven $W$ and $S$. Changes to window size regenerate the entire dataset cleanly.

- Leakage Prevention: Splitting happens at the Subject/Video level before windowing. Windows inherit the split of their parent video.

  $\rightarrow$ What to show:

- A directory tree of `data/processed/...`.

- A "Table of Stages": Input $\rightarrow$ Output $\rightarrow$ Script.

## 2. Pose Extraction (MediaPipe)

### 2.1 The Data Artifact

For every frame $t$, we extract 33 landmarks.

- **Input:** $xy \in [T, 33, 2]$ (Normalized 0-1) and $conf \in [T, 33]$ (Visibility score).
- **Output:** A cleaned, "gated" feature vector.

### 2.2 Confidence Gating (Noise Suppression)

Raw pose detection is noisy. We suppress jitter using confidence scores.

- **Formula:**

$$\tilde{xy}_{t,j} = xy_{t,j} \cdot conf_{t,j}$$

*Where $\tilde{xy}$ represents the weighted coordinate that effectively "zeros out" unreliable joints.*

- **Why:** If a joint is occluded ($conf \approx 0$), it contributes $0$ to the model input rather than a hallucinated position.

- Missing Values: NaNs are zeroed out; the gating handles the "uncertainty".

  $\rightarrow$ Supervisor Q&A

- **Q: Why MediaPipe and not OpenPose?**

  - **A:** MediaPipe is CPU-optimizable and privacy-preserving (skeleton-only), meeting our real-time and privacy-first constraints.

- **Q: How do you handle deep fakes or missing joints?**

  - **A:** Confidence gating ensures the model relies on visible joints. If a person is largely occluded, the input vector shrinks towards zero, which the model learns to associate with "low information" rather than a fall.

## 3. Labeling & Splitting (The "Methodology" Core)

### 3.1 Labeling Strategy

We use **interval-based labeling** where possible.

- **Rule:** A window is labeled "Fall" ($y=1$) if it overlaps significantly with the ground-truth fall interval $[t_{fall\_onset}, t_{fall\_end}]$.
- **Formula:**

$$\text{overlap} = \max(0, \min(t_{end}, t_{fall\_end}) - \max(t_{start}, t_{fall\_onset}) + 1)$$

$$y_{window} = \begin{cases} 1 & \text{if } \text{overlap} \ge k_{frames} \\ 0 & \text{otherwise} \end{cases}$$

- **Unlabeled Data:** Windows from purely unlabeled footage are marked $y=-1$ and excluded from training, used only for false-alarm analysis.

### 3.2 Splitting Strategy (The Leakage Defense)

- **The Golden Rule:** Split **Subjects**, not Windows.

- **Why:** If you split windows randomly, Window A (Frame 0-48) could be in Train, and Window B (Frame 12-60) in Test. This is **leakage**.

- Our Approach: We split the list of video stems first. All windows from subject_01 go to Train; all from subject_02 go to Test.

  $\rightarrow$ Supervisor Q&A

- **Q: Where is the split decision made?**

  - **A:** Before windowing. The script `split_labels.py` generates `train.txt`, which `make_windows.py` reads.

- **Q: Is the test set used during development?**

  - **A:** No. We tune hyperparameters and thresholds on **Validation**. Test is held out for the final report.

## 4. Windows and Stride (Time & Latency)

### 4.1 Definitions

- **Window ($W=48$):** The "observation context". At 25 FPS, duration is:

$$\text{duration} = \frac{W}{fps} = \frac{48}{25} = 1.92s$$

- **Stride ($S=12$):** The "step size". We slide the window forward by 12 frames.
- **Overlap:**

$$\text{Overlap} = W - S = 48 - 12 = 36 \text{ frames}$$

### 4.2 Practical Implications

1. **Dataset Size:**

$$N_{windows} = \lfloor \frac{T_{total} - W}{S} \rfloor + 1$$

1. **Real-time Latency:** You update the decision every $S$ frames.

$$\text{Update Interval} \approx \frac{S}{fps} \approx 0.48s$$

1. Detection Delay: The system waits for enough "fall frames" to enter the window. The worst-case delay is roughly one stride duration.

   $\rightarrow$ Supervisor Q&A

- **Q: Why overlap?**
  - **A:** Overlap ensures we don't miss a fall that happens right on the edge of a window partition. It provides multiple chances to detect the event.
- **Q: How does this meet the <200ms latency requirement?**
  - **A:** "Latency" usually refers to inference time (forward pass), which is milliseconds. "Decision Delay" is driven by stride ($S$). If we need faster reaction, we reduce $S$ (e.g., to 6), accepting higher compute costs.

## 5. Models (TCN vs. GCN)

### 5.1 TCN (Temporal Convolutional Network)

- **Input:** Flat vector $x \in \mathbb{R}^{T \times 66}$.
- **Structure:** 1D convolutions excel at detecting temporal anomalies (sudden accelerations) but ignore spatial connectivity.

### 5.2 GCN (Graph Convolutional Network)

- **Input:** Graph-structured $x \in \mathbb{R}^{T \times 33 \times 4}$.
- **Features:** Relative coordinates plus velocity.

$$x_{feat} = (x_{rel}, y_{rel}, v_x, v_y)$$

*Where $x_{rel}$ is pelvis-centered to remove global translation.*

- Why: Explicitly models body topology and is more robust to camera location.

  $\rightarrow$ Supervisor Q&A

- **Q: How do you handle class imbalance?**

  - **A:** We use class weights in the loss function (BCE/CrossEntropy) or oversample fall windows during training.

- **Q: Checkpoint compatibility issues?**

  - **A:** I'm implementing a "Model Factory" to ensure the evaluation script rebuilds the exact architecture (1-head vs 2-head) used in training.

## 6. Threshold Definitions & Operating Points (fit_ops)

This section defines exactly how raw model outputs become decisions.

### 6.1 From Logits to Probability

The models output a raw logit $z_i \in \mathbb{R}$ for each time window $i$. These are converted to a fall probability using the sigmoid function:

$$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

### 6.2 Threshold $\to$ Predicted Label

Given a specific threshold $\tau \in [0,1]$, the predicted label $\hat{y}_i(\tau)$ is determined by:

$$\hat{y}_i(\tau) = \mathbf{1} \ \ [p_i \ge \tau]$$

Where $y_i \in \{0,1\}$ is the true label (1=fall, 0=ADL).

### 6.3 Confusion Counts & Metrics (at threshold $\tau$)

To evaluate performance at a specific cut-off, we calculate:

$$\begin{aligned} TP(\tau) &= \sum_i \mathbf{1}[y_i=1 \land \hat{y}_i(\tau)=1] \\ FP(\tau) &= \sum_i \mathbf{1}[y_i=0 \land \hat{y}_i(\tau)=1] \\ TN(\tau) &= \sum_i \mathbf{1}[y_i=0 \land \hat{y}_i(\tau)=0] \\ FN(\tau) &= \sum_i \mathbf{1}[y_i=1 \land \hat{y}_i(\tau)=0] \end{aligned}$$

Based on these counts, the core metrics are:

$$\text{Precision}(\tau) = \frac{TP(\tau)}{TP(\tau) + FP(\tau)}$$

$$\text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$$

$$F1(\tau) = \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$

### 6.4 Operating Point Selection (Validation Phase)

The fit_ops.py script performs a sweep over a grid of thresholds $\mathcal{T}$ on the validation set to select three distinct Operating Points (OPs).

OP2: Balanced (Max F1)

Equivalent to the training strategy, but calibrated on validation data:

$$\tau_{\text{OP2}} = \operatorname*{arg\,max}_{\tau \in \mathcal{T}} F1(\tau)$$

OP1: High Recall (Safety Focus)

Given a required recall floor $r_1$ (e.g., 0.95):

$$\tau_{\text{OP1}} = \operatorname*{arg\,max}_{\tau \in \{ \tau \in \mathcal{T} : \text{Recall}(\tau) \ge r_1 \}} F1(\tau)$$

Note: We maximize F1 (or Precision) among thresholds that meet the safety constraint.

OP3: Low Alarm (False Positive Reduction)

First, define the False Positive Rate (FPR):

$$\text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

Given a required recall floor $r_3$ (e.g., 0.90), OP3 selects the threshold that minimizes false alarms while maintaining that recall:

$$\tau_{\text{OP3}} = \operatorname*{arg\,min}_{\tau \in \{ \tau \in \mathcal{T} : \text{Recall}(\tau) \ge r_3 \}} \text{FPR}(\tau)$$

**$\rightarrow$ Supervisor Q&A**

- **Q: Why not just use 0.5?**
  - **A:** 0.5 is arbitrary. In fall detection, a missed fall (FN) is worse than a false alarm (FP). We need data-driven thresholds that respect this asymmetry.
- **Q: Why fit on Validation?**
  - **A:** Fitting on Test is "data snooping" (cheating). We freeze criteria on Validation, then evaluate honestly on Test.

## 7. Evaluation & Alert Rates

### 7.1 Standard Metrics (`metrics.py`)

Produces Precision, Recall, F1, and FPR on the Test set using the frozen OPs.

### 7.2 The "Real" Metric: Alerts/24h (`score_unlabeled_alert_rate.py`)

Classification metrics don't tell you how annoying the system is.

- **The Problem:** Overlapping windows mean one fall event might trigger 20 consecutive "positive" windows.
- **The Solution (Cooldown):** We merge adjacent alerts into a single **Event**.
- **Metric:**

$$\text{Alerts/24h} = \frac{\text{Total Events}}{\text{Hours Covered}} \times 24$$

- **Why it matters:** This proves the system is usable in a real home/hospital.

## 8. Deployment (Real-Time Alignment)

### 8.1 The Contract

- **Frontend:** Captures frames, resamples to target FPS, buffers to length $W$, sends to API.
- **Backend:** Receives window $\rightarrow$ Preprocess (Gate/Normalize) $\rightarrow$ Model Inference $\rightarrow$ Compare $p(fall)$ to OP Threshold.

### 8.2 FPS Mismatch Risk

If training data is 25 FPS and the webcam is 30 FPS, $W=48$ represents different durations.

- **Fix:** We must resample input streams to the training FPS before windowing.

## 9. Meeting Checklist (What to Bring)

1. **Repo Tree:** Screenshot of clean folder structure.
2. **NPZ Sample:** Printout of `xy`, `conf`, `fps`, `label` keys.
3. **Ops YAML:** The file showing the specific $\tau$ values for OP1/2/3.
4. **Results Summary:** Best TCN vs. GCN metrics + Alerts/24h.
5. **Latency Note:** $W=48$, $S=12$, FPS=25 $\rightarrow$ 0.48s update interval.
