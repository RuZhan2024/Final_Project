# Day 4: Add Metrics And Checkpointing

## 1. Today's goal

Today we will add:

- basic binary metrics
- average precision
- checkpoint saving and loading

## 2. Why this part exists in the full pipeline

A training run is not useful if it cannot answer two questions:

1. how well is the model doing?
2. can we save and reload exactly what was trained?

The real repository answers those through:

- `src/fall_detection/core/metrics.py`
- `src/fall_detection/core/ckpt.py`

We will now build the teaching version.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/core/ckpt.py`
- `course_project/training/metrics.py`
- a checkpoint contract that Week 3 can later reuse

## 4. File tree snapshot for today

```text
course_project/
├── core/
│   └── ckpt.py
└── training/
    └── metrics.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/core/ckpt.py`

```python
"""Checkpoint helpers for the course project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_ckpt(path: str | Path, bundle: dict[str, Any]) -> None:
    if "state_dict" not in bundle:
        raise ValueError("Checkpoint bundle must contain 'state_dict'")
    payload = {
        "arch": str(bundle.get("arch", "tcn")),
        "state_dict": bundle["state_dict"],
        "model_cfg": dict(bundle.get("model_cfg", {})),
        "feat_cfg": dict(bundle.get("feat_cfg", {})),
        "data_cfg": dict(bundle.get("data_cfg", {})),
        "metrics": dict(bundle.get("metrics", {})),
    }
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, file_path)


def load_ckpt(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path}: invalid checkpoint format")
    payload.setdefault("arch", "tcn")
    payload.setdefault("model_cfg", {})
    payload.setdefault("feat_cfg", {})
    payload.setdefault("data_cfg", {})
    payload.setdefault("metrics", {})
    return payload
```

### File 2: `course_project/training/metrics.py`

```python
"""Training metrics for the Week 2 teaching model."""

from __future__ import annotations

import numpy as np


def average_precision_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if y.size == 0 or int(y.sum()) == 0:
        return 0.0

    order = np.argsort(-p)
    y_sorted = y[order]
    tp = 0
    precisions: list[float] = []
    for idx, label in enumerate(y_sorted, start=1):
        if int(label) == 1:
            tp += 1
            precisions.append(tp / idx)
    return float(sum(precisions) / max(1, int(y.sum())))


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    pred = (p >= float(threshold)).astype(np.int64)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    acc = (tp + tn) / max(1, len(y))
    ap = average_precision_score(y, p)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "ap": float(ap),
    }
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose the model produces probabilities:

`[0.9, 0.8, 0.2, 0.1]`

and the true labels are:

`[1, 1, 0, 0]`

At threshold `0.5`, the predictions are perfect.

That means:

- precision = 1.0
- recall = 1.0
- F1 = 1.0
- accuracy = 1.0

That is the kind of summary `binary_metrics(...)` computes.

### `save_ckpt(...)` line by line

Inputs:

- output path
- checkpoint bundle

Outputs:

- none

Side effects:

- writes a checkpoint file to disk

Why it exists:

- training, evaluation, and deployment need one portable bundle format

Important non-trivial lines:

`if "state_dict" not in bundle:`

- refuse to save a checkpoint that does not actually contain model weights

`"model_cfg": dict(bundle.get("model_cfg", {}))`

- make configuration explicit and serializable

### `load_ckpt(...)`

Inputs:

- checkpoint path

Outputs:

- loaded checkpoint dictionary

Side effects:

- reads a file from disk

Why it exists:

- Week 3 will need to reconstruct the model from this artifact

### `average_precision_score(...)`

This metric is important because it is threshold-free.

It tells us how well the model ranks positive windows above negative windows, which is useful before we choose an operating point in Week 3.

### `binary_metrics(...)`

Inputs:

- true labels
- probabilities
- threshold

Outputs:

- dictionary of human-readable metrics

Side effects:

- none

Why it exists:

- the training loop needs one concise way to summarize predictions

## 7. Exact run commands

Copy today's files into:

- `course_project/core/ckpt.py`
- `course_project/training/metrics.py`

Run a quick metric check:

```bash
PYTHONPATH="$(pwd)" python3 - <<'PY'
import numpy as np
from course_project.training.metrics import binary_metrics

y = np.array([1, 1, 0, 0], dtype=np.int64)
p = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
print(binary_metrics(y, p))
PY
```

## 8. Expected outputs

Output pattern:

```text
{'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0, 'ap': 1.0}
```

## 9. Sanity checks

Check that:

1. AP is between `0.0` and `1.0`
2. F1 is between `0.0` and `1.0`
3. the checkpoint loader returns `arch`, `model_cfg`, `feat_cfg`, `data_cfg`, and `metrics`

## 10. Common bugs and fixes

### Bug: checkpoint loads but is missing config fields

Fix:

- keep the `setdefault(...)` lines in `load_ckpt(...)`

### Bug: AP is always zero

Fix:

- inspect whether `y_true` actually contains positives

## 11. Mapping to the original repository

Today's teaching files map most directly to:

- `src/fall_detection/core/ckpt.py`
- `src/fall_detection/core/metrics.py`

## 12. Tomorrow's preview

Tomorrow we will put everything together in the main training script and run the first complete Week 2 training job.
