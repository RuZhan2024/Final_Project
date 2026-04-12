#!/usr/bin/env python3
"""Generate a report-ready cross-dataset transfer figure from summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


TRANSFER_ORDER = (
    ("TCN LE2i->CAUCAFall", "TCN: LE2i to CAUCAFall"),
    ("TCN CAUCAFall->LE2i", "TCN: CAUCAFall to LE2i"),
    ("GCN LE2i->CAUCAFall", "GCN: LE2i to CAUCAFall"),
    ("GCN CAUCAFall->LE2i", "GCN: CAUCAFall to LE2i"),
)
BAR_COLORS = ["#5b8ff9", "#d62728", "#5b8ff9", "#d62728"]
METRICS = (
    ("delta_f1", "F1 delta"),
    ("delta_recall", "Recall delta"),
)


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["label"]: row for row in rows}


def plot(rows: List[Dict[str, str]], out_fig: Path) -> None:
    lookup = _row_map(rows)
    labels = [short for _, short in TRANSFER_ORDER]
    x = np.arange(len(labels), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True, sharey=True)

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        vals = np.asarray([float(lookup[label][metric_key]) for label, _ in TRANSFER_ORDER], dtype=float)
        ax.bar(x, vals, color=BAR_COLORS, edgecolor="black", linewidth=0.5, alpha=0.92)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cross-domain minus in-domain")
    fig.suptitle("Cross-dataset transfer is asymmetric", fontsize=15)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot report-ready cross-dataset transfer figure")
    ap.add_argument("--summary_csv", default="artifacts/reports/cross_dataset_summary.csv")
    ap.add_argument("--out_fig", default="artifacts/figures/report/cross_dataset_transfer_summary.png")
    args = ap.parse_args()

    rows = _load_rows(Path(args.summary_csv))
    plot(rows, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
