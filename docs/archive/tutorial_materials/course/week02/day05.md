# Day 5: Write The Full Training And Validation Script

## 1. Today's goal

Today we will connect everything from the first four days into one training script that:

- loads training windows
- loads validation windows
- trains a small TCN
- evaluates on validation data
- saves the best checkpoint
- writes metrics history

## 2. Why this part exists in the full pipeline

The real repository’s TCN training path is orchestrated through:

- `scripts/train_tcn.py`
- `src/fall_detection/training/train_tcn.py`

Our teaching version keeps the same structure, but removes the extra complexity that would distract beginners.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/training/train_tcn.py`
- your first complete training pipeline

## 4. File tree snapshot for today

```text
course_project/
└── training/
    ├── data.py
    ├── metrics.py
    └── train_tcn.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/training/train_tcn.py`

Why this file exists:

- this is the main Week 2 pipeline entrypoint
- it turns Week 1 windows into a saved trained model

```python
"""Train a minimal TCN on Week 1 windows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from course_project.core.ckpt import save_ckpt
from course_project.core.features import FeatCfg
from course_project.models.tcn import TCN, TCNConfig
from course_project.training.data import build_loader
from course_project.training.metrics import binary_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_epoch_metrics(probs_list: list[np.ndarray], labels_list: list[np.ndarray], loss_total: float, n_items: int) -> dict[str, float]:
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,), dtype=np.float32)
    metrics = binary_metrics(labels, probs)
    metrics["loss"] = float(loss_total / max(1, n_items))
    return metrics


@torch.inference_mode()
def evaluate(model: TCN, loader: Any, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    n_items = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for xb, yb, _meta in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.sigmoid(logits)

        batch_size = int(xb.shape[0])
        loss_total += float(loss.item()) * batch_size
        n_items += batch_size
        probs_list.append(probs.cpu().numpy())
        labels_list.append(yb.cpu().numpy())

    return collect_epoch_metrics(probs_list, labels_list, loss_total, n_items)


def train_one_epoch(
    model: TCN,
    loader: Any,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    n_items = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for xb, yb, _meta in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach()
        batch_size = int(xb.shape[0])
        loss_total += float(loss.item()) * batch_size
        n_items += batch_size
        probs_list.append(probs.cpu().numpy())
        labels_list.append(yb.cpu().numpy())

    return collect_epoch_metrics(probs_list, labels_list, loss_total, n_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Week 2 teaching TCN.")
    parser.add_argument("--train_dir", default="data/course_demo/windows/train")
    parser.add_argument("--val_dir", default="data/course_demo/windows/val")
    parser.add_argument("--save_dir", default="outputs/course_week2_tcn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_motion", type=int, default=1)
    parser.add_argument("--use_conf_channel", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feat_cfg = FeatCfg(
        use_motion=bool(args.use_motion),
        use_conf_channel=bool(args.use_conf_channel),
    )
    train_loader = build_loader(args.train_dir, feat_cfg, batch_size=args.batch, shuffle=True)
    val_loader = build_loader(args.val_dir, feat_cfg, batch_size=args.batch, shuffle=False)

    sample_x, _sample_y, _sample_meta = train_loader.dataset[0]
    model_cfg = TCNConfig(hidden=args.hidden, kernel_size=args.kernel_size, dropout=args.dropout)
    model = TCN(
        in_ch=int(sample_x.shape[1]),
        hidden=model_cfg.hidden,
        kernel_size=model_cfg.kernel_size,
        dropout=model_cfg.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ap = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} train_ap={train_metrics['ap']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_ap={val_metrics['ap']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["ap"] >= best_ap:
            best_ap = float(val_metrics["ap"])
            save_ckpt(
                save_dir / "best.pt",
                {
                    "arch": "tcn",
                    "state_dict": model.state_dict(),
                    "model_cfg": model_cfg.to_dict(),
                    "feat_cfg": feat_cfg.to_dict(),
                    "data_cfg": {
                        "train_dir": args.train_dir,
                        "val_dir": args.val_dir,
                        "window_size": int(sample_x.shape[0]),
                        "input_dim": int(sample_x.shape[1]),
                    },
                    "metrics": val_metrics,
                },
            )
            print(f"[ok] saved best checkpoint to {save_dir / 'best.pt'}")

    metrics_payload = {
        "best_val_ap": best_ap,
        "history": history,
        "feat_cfg": feat_cfg.to_dict(),
        "model_cfg": model_cfg.to_dict(),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote metrics to {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

If one batch has:

- `x` shaped `[4, 16, 25]`
- `y` shaped `[4]`

then one training step does:

1. run the model
2. produce four logits
3. compare them to four labels
4. update the weights

That is the core learning cycle this script implements.

### `set_seed(...)`

Inputs:

- one integer seed

Outputs:

- none

Side effects:

- sets random seeds for Python, NumPy, and PyTorch

Why it exists:

- repeated training runs should be reasonably reproducible

### `collect_epoch_metrics(...)`

This helper keeps the epoch logic readable.

It takes the predictions collected across many batches and turns them into one epoch summary.

### `evaluate(...)`

Inputs:

- model
- validation dataloader
- device
- loss function

Outputs:

- validation metrics dictionary

Side effects:

- none beyond running inference

Why it exists:

- training and validation should be separated cleanly

Important non-trivial lines:

`@torch.inference_mode()`

- disable gradient tracking during validation

`probs = torch.sigmoid(logits)`

- validation metrics use probabilities, not raw logits

### `train_one_epoch(...)`

This is the core training loop for one epoch.

Important non-trivial lines:

`optimizer.zero_grad()`

- clear old gradients before computing new ones

`loss.backward()`

- compute gradients by backpropagation

`optimizer.step()`

- update the weights

### `main()` section by section

#### Build configuration and loaders

This reconnects Days 1 and 2 to the training path.

#### Infer input dimension from one sample

`sample_x, _sample_y, _sample_meta = train_loader.dataset[0]`

- instead of hardcoding input size, infer it from the real data

#### Build model, loss, and optimizer

We use:

- TCN model
- `BCEWithLogitsLoss`
- Adam optimizer

This is a clean beginner baseline.

#### Training loop

For each epoch:

- train
- validate
- print metrics
- save best checkpoint if validation AP improved

We save based on validation AP because it is threshold-free and useful before Week 3 threshold selection begins.

## 7. Exact run commands

Copy today's file into:

- `course_project/training/train_tcn.py`

Run training:

```bash
PYTHONPATH="$(pwd)" python3 course_project/training/train_tcn.py \
  --train_dir data/course_demo/windows/train \
  --val_dir data/course_demo/windows/val \
  --save_dir outputs/course_week2_tcn \
  --epochs 5 \
  --batch 4 \
  --lr 1e-3
```

## 8. Expected outputs

Output pattern:

```text
[epoch 1] train_loss=... train_ap=... val_loss=... val_ap=... val_f1=...
[ok] saved best checkpoint to outputs/course_week2_tcn/best.pt
...
[ok] wrote metrics to outputs/course_week2_tcn/metrics.json
```

## 9. Sanity checks

Check that:

1. `outputs/course_week2_tcn/best.pt` exists
2. `outputs/course_week2_tcn/metrics.json` exists
3. `metrics.json` contains `history`
4. validation metrics are being printed every epoch

## 10. Common bugs and fixes

### Bug: loss does not decrease

Fix:

- train for more epochs
- lower the learning rate
- confirm both positive and negative windows exist

### Bug: checkpoint never saves

Fix:

- check that validation AP is being computed and compared against `best_ap`

## 11. Mapping to the original repository

Today's teaching file maps most directly to:

- `src/fall_detection/training/train_tcn.py`
- `scripts/train_tcn.py`

## 12. Tomorrow's preview

Tomorrow we will add a small set of Week 2 tests and debugging checks so students can verify the feature path and checkpoint path before moving into Week 3.
