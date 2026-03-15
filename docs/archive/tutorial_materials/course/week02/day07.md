# Day 7: Inspect The Trained Artifacts And Prepare For Week 3

## 1. Today's goal

Today we will inspect the outputs of the Week 2 training run and make the handoff to Week 3 explicit.

## 2. Why this part exists in the full pipeline

In the real repository, training does not end the story.

A saved checkpoint is only useful because later stages can read it for:

- threshold fitting
- evaluation
- deployment

So today we will inspect the checkpoint and metrics bundle like engineers, not just celebrate that training ran.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/scripts/inspect_week2_outputs.py`
- a clear understanding of what Week 3 will consume

## 4. File tree snapshot for today

```text
course_project/
└── scripts/
    └── inspect_week2_outputs.py

outputs/
└── course_week2_tcn/
    ├── best.pt
    └── metrics.json
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/scripts/inspect_week2_outputs.py`

```python
"""Inspect the trained Week 2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from course_project.core.ckpt import load_ckpt


def inspect_week2_outputs(root: str | Path = "outputs/course_week2_tcn") -> None:
    root_path = Path(root)
    ckpt = load_ckpt(root_path / "best.pt")
    metrics = json.loads((root_path / "metrics.json").read_text(encoding="utf-8"))

    print(f"[ok] arch: {ckpt['arch']}")
    print(f"[ok] feat_cfg: {ckpt['feat_cfg']}")
    print(f"[ok] model_cfg: {ckpt['model_cfg']}")
    print(f"[ok] data_cfg: {ckpt['data_cfg']}")
    print(f"[ok] best_val_ap: {metrics['best_val_ap']:.4f}")
    print(f"[ok] epochs_logged: {len(metrics['history'])}")


def main() -> None:
    inspect_week2_outputs()


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Week 3 will need to know:

- which architecture was trained
- which feature switches were used
- what input dimension the model expects

That means the checkpoint is not "just weights." It is an artifact bundle.

### `inspect_week2_outputs(...)` line by line

Inputs:

- root output folder from the Week 2 training run

Outputs:

- none directly

Side effects:

- reads checkpoint and metrics files
- prints a summary

Why it exists:

- students should learn to inspect artifacts before building the next stage on top of them

`ckpt = load_ckpt(root_path / "best.pt")`

- load the checkpoint bundle

`metrics = json.loads((root_path / "metrics.json").read_text(encoding="utf-8"))`

- load the training history

`print(f"[ok] feat_cfg: {ckpt['feat_cfg']}")`

- this line is especially important
- Week 3 must rebuild probabilities using the same feature settings

## 7. Exact run commands

Copy today's file into:

- `course_project/scripts/inspect_week2_outputs.py`

Run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/scripts/inspect_week2_outputs.py
```

## 8. Expected outputs

Output pattern:

```text
[ok] arch: tcn
[ok] feat_cfg: {'use_motion': True, 'use_conf_channel': True}
[ok] model_cfg: {'hidden': 64, 'kernel_size': 3, 'dropout': 0.2}
[ok] data_cfg: {'train_dir': '...', 'val_dir': '...', 'window_size': 16, 'input_dim': 25}
[ok] best_val_ap: ...
[ok] epochs_logged: 5
```

## 9. Sanity checks

Check that:

1. `arch` is `tcn`
2. `input_dim` is `25` for the default Week 2 feature layout
3. the metrics file contains the same number of epochs you trained for

## 10. Common bugs and fixes

### Bug: checkpoint loads, but config fields are empty

Fix:

- inspect `save_ckpt(...)` and confirm the bundle included `feat_cfg`, `model_cfg`, and `data_cfg`

### Bug: `metrics.json` missing

Fix:

- confirm the training script wrote the final metrics payload at the end of `main()`

## 11. Mapping to the original repository

Today's inspection step maps most directly to:

- the checkpoint contract in `src/fall_detection/core/ckpt.py`
- the artifact reuse expected by `src/fall_detection/evaluation/fit_ops.py`

## 12. Tomorrow's preview

Week 2 is now complete.

In Week 3, we will start from the checkpoint and the model probabilities it produces. Then we will teach:

- thresholds
- operating points
- alert logic
- event-level evaluation

That means the exact artifacts you inspected today become the raw material for the next stage.
