#!/usr/bin/env python3
"""Generate a report-ready offline comparison figure from stability summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


PRIMARY_DATASETS = ("caucafall", "le2i")
MODEL_ORDER = ("tcn", "gcn")
MODEL_COLORS = {"tcn": "#1f77b4", "gcn": "#ff7f0e"}
DATASET_TITLES = {"caucafall": "CAUCAFall (Primary)", "le2i": "LE2i (Comparative)"}
METRICS = (
    ("f1", "F1"),
    ("recall", "Recall"),
)


def _load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _find_row(rows: List[Dict[str, str]], dataset: str, arch: str) -> Dict[str, str]:
    for row in rows:
        if row.get("dataset") == dataset and row.get("arch") == arch:
            return row
    raise KeyError(f"missing row for dataset={dataset} arch={arch}")


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def plot(rows: List[Dict[str, str]], out_fig: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True, sharey=True)
    width = 0.34
    x = np.arange(len(METRICS), dtype=float)

    for ax, dataset in zip(axes, PRIMARY_DATASETS):
        for idx, arch in enumerate(MODEL_ORDER):
            row = _find_row(rows, dataset=dataset, arch=arch)
            means = np.asarray([_f(row, f"{metric}_mean") for metric, _ in METRICS], dtype=float)
            errs = np.asarray([_f(row, f"{metric}_ci95") for metric, _ in METRICS], dtype=float)
            offset = (-0.5 + idx) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                color=MODEL_COLORS[arch],
                label=arch.upper(),
                yerr=errs,
                capsize=5,
                alpha=0.92,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(DATASET_TITLES[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in METRICS])
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean score")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Frozen five-seed offline comparison (mean ± 95% CI)", fontsize=15)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot report-ready stability comparison")
    ap.add_argument("--summary_csv", default="artifacts/reports/stability_summary.csv")
    ap.add_argument("--out_fig", default="artifacts/figures/report/offline_stability_comparison.png")
    args = ap.parse_args()

    rows = _load_summary(Path(args.summary_csv))
    plot(rows, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
