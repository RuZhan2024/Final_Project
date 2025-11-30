## 1. Setup and notation

You have a **validation set** of (N) windows:

* Ground truth labels
  $$y_i \in {0,1}, \quad i = 1,\dots,N,$$
  where (y_i = 1) = fall window, (y_i = 0) = non-fall.
* Model predicted **probabilities of fall**
  $$p_i \in [0,1], \quad i = 1,\dots,N.$$

For any threshold (\tau \in [0,1]) you turn probabilities into binary predictions:

$$
\hat y_i(\tau) =
\begin{cases}
1, & p_i \ge \tau \
0, & p_i < \tau
\end{cases}
$$

From these, define counts:

[
\begin{aligned}
TP(\tau) &= \sum_{i=1}^N \mathbf{1}\big[y_i = 1 \land \hat y_i(\tau) = 1\big] \
FP(\tau) &= \sum_{i=1}^N \mathbf{1}\big[y_i = 0 \land \hat y_i(\tau) = 1\big] \
FN(\tau) &= \sum_{i=1}^N \mathbf{1}\big[y_i = 1 \land \hat y_i(\tau) = 0\big] \
TN(\tau) &= \sum_{i=1}^N \mathbf{1}\big[y_i = 0 \land \hat y_i(\tau) = 0\big]
\end{aligned}
]

Here (\mathbf{1}[\cdot]) is the indicator function (1 if the condition is true, 0 otherwise).

In the code you don’t consider all real (\tau), you consider a **discrete grid**:

$$
\mathcal{T} = {\tau_1, \dots, \tau_K},
\quad \text{e.g. } \tau_k \in {0.01, 0.01 + \Delta, \dots, 0.99}.
$$


## 2. Metrics as functions of the threshold

For each threshold (\tau) you compute:

**Precision**

$$
P(\tau) =
\frac{TP(\tau)}{TP(\tau) + FP(\tau) + \epsilon}
$$

**Recall**

$$
R(\tau) =
\frac{TP(\tau)}{TP(\tau) + FN(\tau) + \epsilon}
$$

**F1-score**

$$
F_1(\tau) =
\frac{2,P(\tau),R(\tau)}{P(\tau) + R(\tau) + \epsilon}
$$

**False Positive Rate (per window)**

$$
\mathrm{FPR}(\tau) =
\frac{FP(\tau)}{FP(\tau) + TN(\tau) + \epsilon}
$$

(\epsilon) is a tiny constant in the code to avoid division by zero; mathematically you can omit it in the write-up if you assume denominators are non-zero.

## 3. OP2 – **Balanced operating point** (max F1)

This is the easiest:

> Choose the threshold that gives the **highest F1-score** on the validation set.

Formally, over the grid (\mathcal{T}):

$$
\tau_{\mathrm{OP2}}
===================

\arg\max_{\tau \in \mathcal{T}} F_1(\tau)
$$

* Intuition: OP2 is your **“balanced” operating point**, trading off precision and recall symmetrically.

In code: `i2 = argmax(F1)` and `thr = thr[i2]`.

## 4. OP1 – **High-recall operating point**

Here the goal is:

> Achieve **very high recall** (e.g. ≥ 0.95), and among those, prefer higher precision and lower threshold.

Let the recall floor be

$$
r_{\text{floor}}^{(1)} = 0.95
$$

(you can generalise it in text as “a high recall floor, e.g. 0.95”).

1. First define the set of thresholds that meet the recall requirement:

$$
S_1 = \Big{ \tau \in \mathcal{T} ;:; R(\tau) \ge r_{\text{floor}}^{(1)} \Big}.
$$

2. If (S_1) is **not empty**, you pick OP1 by **lexicographic optimisation**:

* maximise recall,
* then (if multiple thresholds have the same recall) maximise precision,
* then (if still tied) choose the **smallest** threshold (i.e. lowest (\tau)).

You can write this as:

$$
\tau_{\mathrm{OP1}}
===================

\arg\max_{\tau \in S_1}
\big( R(\tau),, P(\tau),, -\tau \big)
$$

where the argmax is taken in lexicographic order: first (R), then (P), then (-\tau).

3. If (S_1) is **empty** (your model can’t reach that recall), then OP1 degenerates to:

$$
\tau_{\mathrm{OP1}}
===================

\arg\max_{\tau \in \mathcal{T}} R(\tau)
$$

> Intuition: OP1 is your **“safety-first”** operating point: it prioritises catching as many falls as possible, even if that costs false alarms. Within that constraint, it still prefers better precision and slightly lower thresholds.

## 5. OP3 – **Low-alarm operating point (FA-friendly)**

Here the goal is:

> Keep recall reasonably high (e.g. ≥ 0.90), and among those thresholds choose the one with **lowest false positive rate** (≈ fewest false alarms).

Let the recall floor for OP3 be:

$$
r_{\text{floor}}^{(3)} = 0.90
$$

1. Define the feasible set:

$$
S_3 = \Big{ \tau \in \mathcal{T} ;:; R(\tau) \ge r_{\text{floor}}^{(3)} \Big}.
$$

2. If (S_3) is **not empty**, choose:

$$
\tau_{\mathrm{OP3}}
===================

\arg\min_{\tau \in S_3} \mathrm{FPR}(\tau)
$$

3. If (S_3) is **empty** (model can’t reach 0.90 recall), then fall back to:

$$
\tau_{\mathrm{OP3}}
===================

\arg\min_{\tau \in \mathcal{T}} \mathrm{FPR}(\tau)
$$

> Intuition: OP3 is your **“caregiver-friendly” / low-alarm** setting: it allows a small drop in recall (compared to OP1) in exchange for **significantly fewer false alarms**, which is measured via the false positive rate (and then converted to FA/24h in your report).

## 6. How this matches your code (in one sentence each)

* **OP2**:
  $$\tau_{\mathrm{OP2}} = \arg\max_{\tau} F_1(\tau)$$
  ↔ `i2 = np.argmax(F1)`

* **OP1**:
  $$\tau_{\mathrm{OP1}} = \arg\max_{\tau \in S_1} (R(\tau), P(\tau), -\tau)$$
  ↔ sort indices where `R >= 0.95` by `(-R, -P, thr)` and pick the first.

* **OP3**:
  $$\tau_{\mathrm{OP3}} = \arg\min_{\tau \in S_3} \mathrm{FPR}(\tau)$$
  ↔ among indices where `R >= 0.90`, take the one with smallest `FPR`.


