#!/usr/bin/env python3
"""Plot dataset/split class balance from labels + split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


POS_TOKENS = {"1", "fall", "true", "yes", "pos", "positive"}


def _load_labels(path: Path) -> Dict[str, int]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected labels JSON object")
    out: Dict[str, int] = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[str(k)] = int(float(v) > 0)
        else:
            out[str(k)] = int(str(v).strip().lower() in POS_TOKENS)
    return out


def _read_split(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _count(keys: Iterable[str], labels: Dict[str, int]) -> Dict[str, int]:
    pos = neg = miss = 0
    for k in keys:
        if k not in labels:
            miss += 1
            continue
        if labels[k] == 1:
            pos += 1
        else:
            neg += 1
    return {"fall": pos, "nonfall": neg, "missing": miss}


def plot(dataset: str, labels_json: Path, splits_dir: Path, out_fig: Path) -> None:
    labels = _load_labels(labels_json)
    train = _read_split(splits_dir / f"{dataset}_train.txt")
    val = _read_split(splits_dir / f"{dataset}_val.txt")
    test = _read_split(splits_dir / f"{dataset}_test.txt")

    total = _count(labels.keys(), labels)
    c_train = _count(train, labels)
    c_val = _count(val, labels)
    c_test = _count(test, labels)

    groups = ["total", "train", "val", "test"]
    falls = np.array([total["fall"], c_train["fall"], c_val["fall"], c_test["fall"]], dtype=float)
    nonfalls = np.array([total["nonfall"], c_train["nonfall"], c_val["nonfall"], c_test["nonfall"]], dtype=float)

    x = np.arange(len(groups), dtype=float)
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax1.bar(x - width / 2, falls, width=width, label="Fall")
    ax1.bar(x + width / 2, nonfalls, width=width, label="Non-fall")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("count")
    ax1.set_title(f"{dataset.upper()} class counts")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right")

    totals = falls + nonfalls
    totals[totals == 0] = 1.0
    fall_ratio = falls / totals
    nonfall_ratio = nonfalls / totals
    ax2.bar(x, fall_ratio, width=0.6, label="Fall ratio")
    ax2.bar(x, nonfall_ratio, width=0.6, bottom=fall_ratio, label="Non-fall ratio")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("ratio")
    ax2.set_title("class ratio by split")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="lower right")

    missing_total = total["missing"] + c_train["missing"] + c_val["missing"] + c_test["missing"]
    fig.suptitle(f"Dataset balance ({dataset}) | missing_label_refs={missing_total}", fontsize=11)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser(description="Plot class balance from labels/splits")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels_json", default="")
    ap.add_argument("--splits_dir", default="ops/configs/splits")
    ap.add_argument("--out_fig", default="")
    args = ap.parse_args()

    ds = args.dataset.strip().lower()
    labels_json = Path(args.labels_json) if args.labels_json else Path("ops/configs/labels") / f"{ds}.json"
    out_fig = Path(args.out_fig) if args.out_fig else Path("artifacts/figures/dataset_balance") / f"{ds}_balance.png"

    plot(ds, labels_json, Path(args.splits_dir), out_fig)
    print(f"[ok] saved: {out_fig}")


if __name__ == "__main__":
    main()
