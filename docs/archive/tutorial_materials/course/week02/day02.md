# Day 2: Build The Window Dataset And The Dataloader

## 1. Today's goal

Today we will write the dataset layer that turns a folder of saved windows into training batches.

## 2. Why this part exists in the full pipeline

The real repository uses dataset classes inside the training code to:

- discover window files
- decode them
- build features
- batch them consistently

That is the bridge between preprocessing artifacts and model training.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/training/__init__.py`
- `course_project/training/data.py`
- one dataloader that returns `[B, T, C]` tensors and `[B]` labels

## 4. File tree snapshot for today

```text
course_project/
├── training/
│   ├── __init__.py
│   └── data.py
└── ...
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/training/__init__.py`

```python
"""Training utilities for the course project."""
```

### File 2: `course_project/training/data.py`

```python
"""Dataset and dataloader helpers for Week 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from course_project.core.features import FeatCfg, build_canonical_input, build_tcn_input, read_window_npz


def list_window_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(p for p in root_path.glob("*.npz") if p.is_file())


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Read saved window files and return model-ready tensors."""

    def __init__(self, root: str | Path, feat_cfg: FeatCfg) -> None:
        self.files = list_window_files(root)
        self.feat_cfg = feat_cfg
        if not self.files:
            raise ValueError(f"No window files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        path = self.files[index]
        joints, motion, conf, mask, fps, meta = read_window_npz(path)
        x_can, _mask = build_canonical_input(joints, motion, conf, mask, fps, self.feat_cfg)
        x_tcn = build_tcn_input(x_can)
        x = torch.from_numpy(x_tcn.astype(np.float32))
        y = torch.tensor(float(meta.y), dtype=torch.float32)
        info = {
            "path": meta.path,
            "video_id": meta.video_id,
            "w_start": meta.w_start,
            "w_end": meta.w_end,
        }
        return x, y, info


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    xs = torch.stack([row[0] for row in batch], dim=0)
    ys = torch.stack([row[1] for row in batch], dim=0)
    meta = [row[2] for row in batch]
    return xs, ys, meta


def build_loader(root: str | Path, feat_cfg: FeatCfg, batch_size: int, shuffle: bool) -> DataLoader[Any]:
    dataset = WindowDataset(root, feat_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose one window file becomes a tensor of shape:

`[16, 25]`

If we batch 4 such windows together, the dataloader should return:

- `x` shaped `[4, 16, 25]`
- `y` shaped `[4]`

That is exactly the contract we are building today.

### `list_window_files(...)` line by line

Inputs:

- one split directory such as `data/course_demo/windows/train`

Outputs:

- sorted list of `.npz` window files

Side effects:

- none

Why it exists:

- the dataset class should not hide file discovery logic inside its constructor

### `WindowDataset` section by section

Responsibility:

- read one saved window
- build model-ready features
- return one tensor, one label, and a metadata dictionary

Data flow:

`path -> read_window_npz -> build_canonical_input -> build_tcn_input -> torch tensor`

Relationship to the pipeline:

- this is the handoff point between Week 1 artifacts and the Week 2 training loop

#### `__init__(...)`

`self.files = list_window_files(root)`

- discover all window files in the split

`if not self.files: raise ValueError(...)`

- fail fast if the student points to an empty folder

#### `__len__(...)`

- standard dataset length

#### `__getitem__(...)`

Inputs:

- one integer index

Outputs:

- feature tensor
- label tensor
- metadata dictionary

Side effects:

- reads one file from disk

Important non-trivial lines:

`x = torch.from_numpy(x_tcn.astype(np.float32))`

- convert the NumPy feature array into a PyTorch tensor
- keep the dtype explicit

`y = torch.tensor(float(meta.y), dtype=torch.float32)`

- the binary label becomes a float because our later binary loss expects floating targets

`info = {...}`

- keep metadata around for debugging and later evaluation

### `collate_batch(...)` line by line

Inputs:

- a list of individual dataset items

Outputs:

- one batched feature tensor
- one batched label tensor
- one list of metadata dictionaries

Side effects:

- none

Why it exists:

- PyTorch needs to know how to combine individual examples into a batch

`torch.stack([row[0] for row in batch], dim=0)`

- create batch dimension `B`

### `build_loader(...)`

This is a convenience function that:

1. builds the dataset
2. builds the dataloader

It keeps later training code shorter and easier to read.

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/training
```

Copy today's files into:

- `course_project/training/__init__.py`
- `course_project/training/data.py`

Run a quick batch inspection:

```bash
PYTHONPATH="$(pwd)" python3 - <<'PY'
from course_project.core.features import FeatCfg
from course_project.training.data import build_loader

loader = build_loader("data/course_demo/windows/train", FeatCfg(), batch_size=4, shuffle=False)
x, y, meta = next(iter(loader))
print(tuple(x.shape))
print(tuple(y.shape))
print(meta[0])
PY
```

## 8. Expected outputs

Output pattern:

```text
(4, 16, 25)
(4,)
{'path': '...', 'video_id': '...', 'w_start': ..., 'w_end': ...}
```

## 9. Sanity checks

Check that:

1. the first dimension of `x` matches batch size
2. the second dimension is `16`
3. the last dimension is `25`
4. `y` contains only `0.0` and `1.0`

## 10. Common bugs and fixes

### Bug: `No window files found`

Fix:

- confirm Week 1 window generation completed
- confirm you pointed to `train`, `val`, or `test`

### Bug: batch shape is `[16, 25]` instead of `[B, 16, 25]`

Fix:

- check `torch.stack(..., dim=0)` inside `collate_batch(...)`

### Bug: label dtype mismatch later in training

Fix:

- keep `y` as `torch.float32`

## 11. Mapping to the original repository

Today's teaching file maps most directly to:

- `WindowDatasetTCN` inside `src/fall_detection/training/train_tcn.py`

Teaching simplification:

- one dataset class
- one loader path

Full repository version:

- more feature flags
- more augmentation controls
- more architecture-specific loading paths

## 12. Tomorrow's preview

Tomorrow we will build the first actual model: a small teaching TCN that accepts `[B, T, C]` input and returns one fall logit per window.
