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
