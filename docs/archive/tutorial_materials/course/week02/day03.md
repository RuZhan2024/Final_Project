# Day 3: Build The First Temporal Model

## 1. Today's goal

Today we will build the first trainable model of the course: a small temporal convolutional network.

## 2. Why this part exists in the full pipeline

The real repository supports both TCN and GCN models. For teaching, the TCN is the right first step because it is easier to read and still faithful to the real pipeline.

The closest production path is:

- `src/fall_detection/core/models.py`
- `src/fall_detection/training/train_tcn.py`

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/models/__init__.py`
- `course_project/models/tcn.py`
- one working forward pass from `[B, T, C]` to `[B]`

## 4. File tree snapshot for today

```text
course_project/
├── models/
│   ├── __init__.py
│   └── tcn.py
└── ...
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/models/__init__.py`

```python
"""Model definitions for the course project."""
```

### File 2: `course_project/models/tcn.py`

```python
"""A minimal temporal convolutional network for Week 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """One temporal convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCN(nn.Module):
    """Map [B, T, C] input to one logit per sample."""

    def __init__(self, in_ch: int, hidden: int = 64, kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.block1 = TemporalBlock(in_ch, hidden, kernel_size, dropout)
        self.block2 = TemporalBlock(hidden, hidden, kernel_size, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


@dataclass(frozen=True)
class TCNConfig:
    hidden: int = 64
    kernel_size: int = 3
    dropout: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

## 6. Detailed teaching explanation

### Tiny concrete example first

If a training batch has shape:

`[4, 16, 25]`

then:

- `B = 4`
- `T = 16`
- `C = 25`

The model should output:

`[4]`

That means one raw score, or logit, for each window in the batch.

### `TemporalBlock` section by section

Responsibility:

- extract temporal patterns from the sequence

Data flow:

`[B, C, T] -> Conv1d -> BatchNorm -> ReLU -> Dropout`

Relationship to the pipeline:

- this is the small reusable building block of the TCN

Important non-trivial lines:

`padding = kernel_size // 2`

- for odd kernel sizes like `3`, this keeps the temporal length unchanged

`nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)`

- the convolution scans along time

### `TCN` section by section

Responsibility:

- accept one batch of windows and produce one logit per window

Data flow:

`[B, T, C] -> transpose -> temporal blocks -> global pool -> linear head`

Relationship to the pipeline:

- this is the first actual learning model of the course

Important non-trivial lines:

`if x.ndim != 3:`

- fail fast if the input shape is wrong

`x = x.transpose(1, 2)`

- `Conv1d` expects `[B, C, T]`, but our dataloader produces `[B, T, C]`

`self.pool = nn.AdaptiveAvgPool1d(1)`

- compress the temporal axis down to one summary vector

`return self.head(x).squeeze(-1)`

- output one scalar logit per sample

### `TCNConfig`

This tiny config object matters because later we will want to save model settings inside checkpoints.

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/models
```

Copy today's files into:

- `course_project/models/__init__.py`
- `course_project/models/tcn.py`

Run a forward pass:

```bash
PYTHONPATH="$(pwd)" python3 - <<'PY'
import torch
from course_project.models.tcn import TCN

model = TCN(in_ch=25, hidden=32)
x = torch.randn(4, 16, 25)
y = model(x)
print(tuple(y.shape))
PY
```

## 8. Expected outputs

```text
(4,)
```

## 9. Sanity checks

Check that:

1. the input shape is `[B, T, C]`
2. the output shape is `[B]`
3. changing `hidden` changes internal channel width but not output shape

## 10. Common bugs and fixes

### Bug: `Expected [B, T, C]`

Fix:

- confirm the batch tensor still has three dimensions before it reaches the model

### Bug: `Conv1d` shape mismatch

Fix:

- keep the `transpose(1, 2)` line inside `forward()`

## 11. Mapping to the original repository

Today's teaching model maps most directly to:

- the TCN branch inside `src/fall_detection/core/models.py`

Teaching simplification:

- two temporal blocks
- one output head

Full repository version:

- more architectural options
- more feature branches
- more deployment-facing rebuild helpers

## 12. Tomorrow's preview

Tomorrow we will add the two support systems that make training reusable instead of disposable:

- metrics
- checkpoint saving/loading
