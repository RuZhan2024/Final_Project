Nice, let’s turn this into something you can literally paste into your report / thesis and tweak.

I’ll write it as a self-contained section with equations and a small comparison table.

---

## 4. Model Architectures and Incremental Enhancements

This project uses two supervised sequence models for skeleton-based fall detection:

1. A **Temporal Convolutional Network (TCN)** operating on flattened 2D joint coordinates.
2. A **spatio-temporal Graph Convolutional Network (GCN)** that explicitly models the skeleton graph.

Both models are trained on the same windowed pose sequences extracted from LE2I, URFD and CAUCAFall, and both are evaluated using consistent metrics (precision, recall, F1 and FA/24h).

### 4.1 Baseline Temporal Convolutional Network (TCN)

Each input window is stored as an NPZ file containing:

* $xy \in \mathbb{R}^{T \times 33 \times 2}$ – 2D joint coordinates from MediaPipe Pose.
* $\text{conf} \in \mathbb{R}^{T \times 33}$ – visibility / confidence scores.
* A scalar label $y \in {0,1}$ or a string label (“adl” / “fall”).

Pre-processing first cleans and gates the coordinates:

[
xy' = \text{nan_to_num}(xy), \qquad
x(t,j,:) = xy'(t,j,:) \cdot \text{conf}(t,j),
]

then flattens each frame:

[
x \in \mathbb{R}^{T \times C}, \quad C = 33 \times 2 = 66.
]

The **baseline TCN** is a small 1D CNN over time:

* Input: $x \in \mathbb{R}^{B \times T \times C}$, reshaped to $\mathbb{R}^{B \times C \times T}$.
* Two temporal convolutions:

[
h_1 = \text{ReLU}(\text{Conv1d}*{C \to H}(x)), \quad
h_2 = \text{ReLU}(\text{Conv1d}*{H \to H}(h_1)),
]

with $H=128$ and kernel size $5$ (padding preserves length $T$).

* Global average pooling over time:

[
z = \text{GAP}_t(h_2) \in \mathbb{R}^{B \times H}.
]

* Final linear layer and sigmoid:

[
\ell = W z + b \in \mathbb{R}^{B \times 1}, \qquad
\hat{p} = \sigma(\ell) = \frac{1}{1+e^{-\ell}}.
]

The model is trained with binary cross-entropy:

[
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N
\big[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \big].
]

In the **baseline version**, a single decision threshold was chosen by maximising F1 on the validation set and saving that value in the checkpoint.

---

### 4.2 Enhancements to the TCN Pipeline

Rather than radically changing the TCN architecture, the main improvements were to the **training, calibration, and evaluation pipeline**. These enhancements make the TCN more robust and clinically interpretable.

#### 4.2.1 Standardised labels and preprocessing

We introduced a single helper $\texttt{_label_from_npz}(\cdot)$ which:

* accepts numeric fields (`y`, `target`) and string fields (`label`, `y_label`),
* maps variants such as `"fall"`, `"1"`, `"true"` to $1$ and `"adl"`, `"0"`, `"false"` to $0$,
* treats $-1$ or missing labels as **unlabelled**, which are skipped during training.

The same function is used across TCN training, GCN training, and evaluation scripts, so the models all see a **consistent, cleaned label space**.

Pre-processing is also standardised:

* NaNs and infinities are replaced with zero.
* Joint coordinates are always gated by their confidence scores.
* Window length $W$ and stride $S$ are shared across models.

This reduces label noise and ensures fair comparisons between TCN and GCN.

#### 4.2.2 Threshold sweeps and multiple operating points

Instead of committing to a single “best” threshold, we now evaluate the model on the validation set over a grid of thresholds $\tau \in [0.05, 0.95]$ and compute:

[
\text{Precision}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau)+\text{FP}(\tau)}, \quad
\text{Recall}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau)+\text{FN}(\tau)},
]
[
F_1(\tau) = \frac{2, \text{Precision}(\tau), \text{Recall}(\tau)}
{\text{Precision}(\tau) + \text{Recall}(\tau)}.
]

From this curve we derive three **operating points (OPs)**:

* **OP1 – high recall**:
  choose the smallest $\tau$ such that recall is above a high target (e.g. $\ge 0.95$), prioritising sensitivity over false alarms.

* **OP2 – balanced**:
  choose $\tau$ that maximises $F_1(\tau)$ on the validation set.

* **OP3 – low alarm**:
  choose a larger $\tau$ that significantly reduces false alarms (and FA/24h), at the cost of some recall.

On the test set, these OPs are evaluated and saved in a JSON report:

```json
{
  "dataset": "test",
  "n_windows": 361,
  "pos_windows": 78,
  "ops": {
    "OP1_high_recall": { "thr": 0.01,   "precision": ..., "recall": ..., "f1": ... },
    "OP2_balanced":    { "thr": 0.6029, "precision": ..., "recall": ..., "f1": ... },
    "OP3_low_alarm":   { "thr": 0.108,  "precision": ..., "recall": ..., "f1": ... }
  }
}
```

These reports are used by:

* the **FA/24h vs recall plots**, and
* the **Dashboard / Settings** pages in the prototype (e.g. showing OP1/OP2/OP3).

This transforms the TCN from “one accuracy number” into a model with **configurable trade-offs** that match different deployment scenarios.

#### 4.2.3 Regularisation and stability

To support training on noisy real-world sequences, we added:

* **Dropout** between temporal conv layers,
* **Gradient clipping** (`‖∇θ‖₂ ≤ 1.0`) to avoid exploding gradients,
* Fixed random seeds and deterministic data splits to ensure reproducibility.

---

### 4.3 Spatio-Temporal GCN for Skeleton Sequences

The GCN is designed to exploit the **graph structure** of the human skeleton rather than treating each joint as an independent feature.

#### 4.3.1 Skeleton graph and adjacency normalisation

We model the 33 MediaPipe Pose joints as a graph:

* Nodes $V = {0,\dots,32}$ correspond to joints.
* Undirected edges $E$ connect anatomically related joints (torso, limbs, cross-links).

This yields an adjacency matrix $A \in \mathbb{R}^{V \times V}$, where:

[
A_{ij} =
\begin{cases}
1 & \text{if joints } i,j \text{ are connected or } i=j, \
0 & \text{otherwise.}
\end{cases}
]

We use the standard symmetric normalisation:

[
D_{ii} = \sum_j A_{ij}, \qquad
\hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}.
]

$\hat{A}$ is stored as a buffer inside the model and used for graph aggregation.

#### 4.3.2 GCNTemporal architecture

Input windows keep their spatial structure:

* $x \in \mathbb{R}^{B \times T \times V \times C}$ with $V=33$ and $C=2$.

The model comprises:

1. **Spatial graph aggregation**

For each frame $t$, joint features are aggregated from neighbours:

[
x' = \hat{A} x \quad \Rightarrow \quad
x'*{b,t,i,:} = \sum*{j} \hat{A}*{ij}, x*{b,t,j,:}.
]

2. **Two GCN-style feature transforms**

We apply two pointwise MLPs around the aggregation:

[
h_1 = \sigma\big(\text{GC1}(x')\big), \quad
h_2 = \sigma\big(\text{GC2}(\hat{A} h_1)\big),
]

where GC1 and GC2 are linear layers applied to the feature dimension, and $\sigma$ is ReLU.

3. **Pooling over joints**

[
f_{b,t,:} = \frac{1}{V} \sum_{i=1}^V h_2(b,t,i,:) \in \mathbb{R}^{gcn_out}
]

This gives a sequence of frame embeddings $f \in \mathbb{R}^{B \times T \times gcn_out}$.

4. **Temporal Conv1d head**

We then apply the same kind of temporal head as the TCN:

* permute to $f \in \mathbb{R}^{B \times gcn_out \times T}$,
* Conv1d + ReLU + Dropout,
* global average pool over time,
* final linear layer to obtain the logits and $\hat{p}$.

Thus, the GCN model is **TCN-like in time**, but **graph-aware in space**.

---

### 4.4 Enhancements to the GCN Training / Evaluation

We aligned the GCN pipeline with the enhanced TCN pipeline to allow **fair comparisons**.

1. **Shared label and preprocessing logic**

* Uses the same `_label_from_npz` as the TCN.
* Uses the same rules for NaNs, confidence gating, and windowing ($W=48$, $S=12$).
* This ensures performance differences come from the architecture, not data handling.

2. **Validation threshold sweep**

* The training loop runs `evaluate_with_sweep` on the validation set after each epoch.
* The best F1 and its threshold are tracked; when improved, a checkpoint is saved with:

  ```python
  {
      "model": state_dict,
      "num_joints": V,
      "in_feats": C,
      "best_thr": best_threshold
  }
  ```

3. **Test-set JSON reports with OP1 / OP2 / OP3**

* After training, the best checkpoint is reloaded and evaluated on the test set.

* We compute the full precision–recall curve and derive:

  * OP1_high_recall,
  * OP2_balanced,
  * OP3_low_alarm,

* These are stored in a JSON report with exactly the same format as the TCN:

```json
{
  "dataset": "test",
  "n_windows": 361,
  "pos_windows": 78,
  "ops": {
    "OP1_high_recall": { "thr": ..., "precision": ..., "recall": ..., "f1": ... },
    "OP2_balanced":   { "thr": ..., "precision": ..., "recall": ..., "f1": ... },
    "OP3_low_alarm":  { "thr": ..., "precision": ..., "recall": ..., "f1": ... }
  }
}
```

This means TCN and GCN can be compared at the **same three operating regimes**.

---

### 4.5 Summary of Enhancements

You can summarise the design and improvements in a small table, for example:

| Model            | Input representation       | Spatial modelling   | Temporal modelling   | Calibration / OPs              | Extra enhancements                                                         |
| ---------------- | -------------------------- | ------------------- | -------------------- | ------------------------------ | -------------------------------------------------------------------------- |
| **Baseline TCN** | Flattened $[T, 33\times2]$ | None (no graph)     | 2× Conv1d + GAP      | Single threshold (best F1)     | Basic training only                                                        |
| **Enhanced TCN** | Same as baseline           | None (no graph)     | 2× Conv1d + GAP      | OP1 / OP2 / OP3 from val sweep | Label cleaning, dropout, grad clipping, FA/24h, cross-dataset eval         |
| **Baseline GCN** | Full $[T, V, 2]$           | MediaPipe adjacency | Conv1d temporal head | Single threshold (best F1)     | Graph aggregation but simpler evaluation pipeline                          |
| **Enhanced GCN** | Same as baseline GCN       | MediaPipe adjacency | Conv1d temporal head | OP1 / OP2 / OP3, JSON reports  | Shared preprocessing with TCN, aligned evaluation, better interpretability |

In the **discussion** section you can then say:

* The **TCN** is simple, efficient, and already achieves strong performance (e.g. F1 ≈ 0.72 in balanced mode on LE2I).
* The **GCN** introduces explicit skeleton structure, which is conceptually more expressive. In practice, with the current model size and data regime, it achieves similar or slightly lower F1 than the TCN, but it is valuable as:

  * a **model diversity** baseline, and
  * a foundation for future work using stronger ST-GCN / CTR-GCN variants.

If you’d like, I can next help you draft a short **“Ablation / Improvement” subsection** that compares *before vs after* metrics (old TCN vs enhanced TCN, baseline GCN vs enhanced GCN) in a neat table.


Below is a draft “Results & Discussion” section you can drop into your thesis and then trim/reshape as needed.

---

## 6. Results and Discussion

### 6.1 Evaluation setup recap

All models were evaluated on fixed-length pose windows of 48 frames (W = 48) with a stride of 12 frames, extracted from MediaPipe Pose sequences. Each window is labelled as “fall” (1) if it overlaps any annotated fall span, and “ADL” (0) otherwise. Performance is reported at window level using precision, recall and F1-score.

To link to different deployment regimes, three operating points (OPs) were pre-defined:

* OP1 – high-recall setting: prioritises missing as few falls as possible, at the cost of more false positives.
* OP2 – balanced setting: maximises F1 on the validation set (compromise between recall and precision).
* OP3 – low-alarm setting: prefers high precision and low false alarms, accepting that some falls will be missed.

For each dataset and model, thresholds for OP1/OP2/OP3 were fitted on the validation set and then evaluated on the held-out test windows.

---

### 6.2 In-domain performance on individual datasets

#### 6.2.1 LE2I (home-like falls)

LE2I is a relatively small dataset in our windowed representation (361 test windows, 78 positives, ≈21.6% falls). For the TCN model, the automatically fitted operating points for OP1–OP3 collapse to almost the same threshold (around 0.45), giving identical test behaviour: precision ≈0.22, recall 1.0 and F1 ≈0.36.

This indicates that on the LE2I validation set, the best F1 that the TCN could achieve still occurred in a regime where almost every window is classified as “fall”. In other words, the model has learned a useful ranking (falls tend to get higher scores than ADLs), but because the dataset is small and noisy, the F1 optimiser prefers an extreme “always positive” solution that maximises recall and keeps precision at a modest but non-zero level.

The GCN shows a much healthier trade-off. At OP2 (balanced), it reaches F1 ≈0.69 with precision ≈0.68 and recall ≈0.69, and at OP3 (low-alarm) it achieves perfect precision (1.0) with recall ≈0.46 (F1 ≈0.63). This is more in line with what we would expect from a well-calibrated detector: we can dial the threshold to trade some recall for a substantial reduction in false alarms.

Interpretation:

* The LE2I sequences include cluttered home scenes with occlusions and partial views. Modelling the skeleton as a graph (GCN) helps to stabilise predictions when some joints are missing or jittery.
* The TCN, which processes a flattened joint vector over time, does pick up temporal dynamics but struggles to separate ADLs from falls with a non-trivial threshold on such a small dataset. This shows up as the degenerate OP2/OP3 solution.

For deployment in a home-like environment, the LE2I GCN results suggest that OP2 and OP3 give usable trade-offs: roughly 70% recall at balanced false alarm rate, or ~46% recall in a very conservative low-alarm mode.

---

#### 6.2.2 CAUCAFall (scripted falls with camera motion)

On CAUCAFall the test set is even smaller (132 windows, 58 positives), but class balance is less skewed (≈44% falls). The TCN achieves F1 ≈0.61 at OP1 (precision ≈0.44, recall 1.0), but OP2 degenerates: the best balanced point according to the validation set corresponds to a threshold that rejects every fall window on the test set (precision and recall both 0). This again reflects the instability of threshold fitting on a very small validation set.

In contrast, the GCN obtains a consistently better envelope:

* OP1: precision ≈0.44, recall 1.0, F1 ≈0.61 (very similar to TCN OP1).
* OP2: precision ≈0.54, recall ≈0.79, F1 ≈0.64 (a genuinely balanced operating point).
* OP3: precision 1.0, recall ≈0.05 (very conservative, almost no false alarms but most falls missed).

Here the gain of GCN over TCN is clearest at OP2: where the TCN collapses to a trivial solution, the GCN maintains a sensible precision–recall trade-off. CAUCAFall includes camera motion and viewpoint variation; explicitly encoding the human skeleton graph appears to make the model more robust to these geometric changes than a purely temporal convolution on flattened joint coordinates.

---

#### 6.2.3 URFD (short sequences, limited size)

For URFD we report only GCN in-domain performance (the main TCN experiment on URFD uses cross-dataset transfer from LE2I; see Section 6.3). The URFD test set is very small (70 windows, 13 falls), making metrics volatile. Despite this, the GCN reaches:

* OP1: precision ≈0.22, recall 1.0, F1 ≈0.36.
* OP2/OP3: precision 0.5, recall ≈0.62, F1 ≈0.55.

Even with only 13 positive windows, the GCN finds a threshold where about half of the alarms correspond to true falls while retaining more than 60% of falls. This is promising given the limited training data, but it also highlights that URFD alone is too small to estimate clinically realistic false alarm rates; a single misclassified window already changes F1 by several points.

---

#### 6.2.4 MUVIM (large-scale falls from ZED RGB)

MUVIM provides a much larger test set after windowing (7261 test windows, 5269 falls, ≈73% positive). On this dataset both architectures perform strongly.

For the TCN:

* OP2 achieves F1 ≈0.84 with precision ≈0.73 and recall ≈0.997. OP1 is very similar (slightly lower precision, slightly higher recall), and OP3 retains recall ≈0.95 at precision ≈0.72 (F1 ≈0.82).

For the GCN:

* OP1: F1 ≈0.84 with precision ≈0.73, recall 1.0.
* OP2: F1 ≈0.85 with precision ≈0.75, recall ≈0.97 (a slight but consistent improvement over the TCN).
* OP3: precision 1.0 but recall ≈0.04 (very conservative).

With far more training examples, both models learn a clear separation between fall and ADL windows. The GCN has a small but consistent edge at OP2, suggesting that spatial modelling brings marginal gains when there is enough data to learn more subtle patterns of joint coordination. At the same time, the TCN’s performance is sufficiently high that, from an engineering point of view, TCN might still be preferred for on-device deployment if computational budget is tight.

---

### 6.3 Cross-dataset generalisation

To assess robustness to dataset shift, we evaluated TCN models trained on LE2I and CAUCAFall on the URFD test set (70 windows, 13 falls).

* LE2I TCN → URFD: at the fitted OP2, the model achieves precision ≈0.19, recall 1.0 and F1 ≈0.31.
* CAUCAFall TCN → URFD: OP1 gives identical behaviour (precision ≈0.19, recall 1.0, F1 ≈0.31), while the balanced OP2 collapses to a threshold that rejects all falls (F1 = 0).

These results show that:

1. The TCN does learn fall-like temporal patterns that transfer to a new dataset (recall remains 1.0 for a wide range of thresholds).
2. However, threshold calibration is fragile under domain shift: the same OP2 that works on the source validation set can be either overly aggressive (rejecting all falls) or overly permissive (classifying almost everything as a fall) on URFD.
3. The cross-dataset F1 ≈0.31 is clearly lower than the in-domain GCN F1 ≈0.55 on URFD, emphasising that training a dedicated model on the target domain is still beneficial whenever possible.

Overall, the cross-dataset experiments support the idea that training separate models per dataset and then selecting the most appropriate one for the target environment (e.g. home vs hospital vs lab) is safer than trying to learn a single global threshold across all domains.

---

### 6.4 TCN vs GCN: comparative analysis

Across datasets, a consistent pattern emerges:

* On small datasets with complex scenes (LE2I, CAUCAFall), the GCN significantly outperforms the TCN at the balanced OP2. It is able to find thresholds that yield “reasonable” precision and recall simultaneously, whereas the TCN often collapses to trivial high-recall solutions when OPs are fitted automatically.
* On URFD, the GCN achieves F1 ≈0.55 at OP2 despite the tiny test set. There is no directly comparable TCN in-domain result, but cross-dataset TCN performance on URFD (F1 ≈0.31) gives a lower bound on what a non-graph temporal model can do when not tuned specifically for this domain.
* On the large MUVIM dataset both architectures are strong, but the GCN has a small edge at OP2 (F1 ≈0.85 vs ≈0.84). The difference is not dramatic, suggesting that when enough data is available, a well-tuned TCN can approximate the behaviour of a more complex spatio-temporal model.

These findings align with the architectural intuition:

* TCNs see the skeleton as a flat vector and learn temporal filters. They are efficient and work well when there is a lot of data and the temporal pattern is distinctive (e.g. clear falls in MUVIM).
* GCNs explicitly encode the skeleton graph (bones as edges). This spatial bias helps in data-scarce and noisy settings by constraining the model to consider physically plausible joint relations, improving generalisation on LE2I, CAUCAFall and URFD.

From a system design perspective, this suggests a hybrid strategy: use TCN as a strong baseline for large, clean datasets and as an efficient on-device model, while reserving GCNs for scenarios where robustness under limited data and complex environments is more critical than the last few milliseconds of latency.

---

### 6.5 Influence of dataset characteristics

The variability across datasets is not only due to the architectures but also to the underlying data:

* Dataset size:

  * MUVIM provides thousands of windows, allowing both TCN and GCN to learn stable decision boundaries and well-behaved ROC curves.
  * LE2I, CAUCAFall and URFD have very few test windows (hundreds or less), which makes threshold fitting and evaluation noisy. A single mislabelled fall or ADL can change F1 by several percentage points.

* Class balance:

  * MUVIM is heavily skewed towards falls in the test split (≈73% fall windows). This naturally inflates recall and F1 if the model leans towards positive predictions, but the high precision shows that the models still discriminate ADLs reasonably well.
  * URFD is strongly imbalanced the other way (only 13 positive windows). Here, a tendency to “over-call” falls leads to low precision and noisy metrics.

* Scene and motion diversity:

  * LE2I contains realistic home scenes with occlusions and partial views of the person, which challenge pose estimation and make falls look less stereotyped.
  * CAUCAFall includes camera motion and variations in viewpoint; again, skeleton stability becomes important.
  * MUVIM’s falls appear more controlled and visually salient, making the distinction from ADLs easier once the model sees enough examples.

These differences help explain why GCNs bring the largest gains on LE2I and CAUCAFall, where spatial consistency across joints is crucial, and why both models saturate on MUVIM.

---

### 6.6 Implications for deployment and future work

From a practical perspective in the context of a home fall-detection system, the results suggest:

* It is feasible to achieve high window-level recall (≥0.95) on large, curated datasets such as MUVIM with manageable precision, especially around OP2 for both TCN and GCN.
* On smaller, more realistic datasets, achieving both high recall and low false alarms is harder. The GCN improves the situation but cannot fully overcome the limitations of sparse and noisy data.
* Automatically tuned operating points can become unstable on very small validation sets (LE2I, CAUCAFall, URFD). In a real deployment, these thresholds would likely need to be adjusted using additional unlabeled monitoring data (e.g. by directly measuring false alarms per 24 hours in a pilot trial).

Future work could therefore focus on:

* Episode-level evaluation (per-fall metrics rather than window-level) and explicit measurement of false alarms per 24 hours on continuous streams.
* Domain adaptation or multi-domain training to stabilise thresholds across LE2I/URFD/CAUCAFall rather than fitting them independently per dataset.
* Lightweight GCN variants that retain most of the robustness gains while bringing the computational footprint closer to the TCN, making them more attractive for strictly on-device inference.

Overall, the experiments confirm the initial design intuition: a temporal CNN baseline is already a strong model for fall detection from skeletons, but introducing explicit skeletal structure via a GCN consistently improves robustness on challenging, data-scarce home-like datasets, with only marginal computational overhead on modern hardware.
