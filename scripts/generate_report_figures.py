#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "reports"
OUT_DIR = ROOT / "artifacts" / "figures" / "report"
DIAGNOSTIC_OUT_DIR = OUT_DIR / "diagnostic"


def _ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def _save(fig: plt.Figure, name: str) -> Path:
    out = _ensure_out_dir() / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _save_diagnostic(fig: plt.Figure, name: str) -> Path:
    DIAGNOSTIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAGNOSTIC_OUT_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_online_replay_accuracy_heatmap() -> Path:
    df = pd.read_csv(REPORTS_DIR / "online_mc_replay_matrix_20260402.csv")
    if "use_mc" in df.columns:
        df = df[df["use_mc"] == False].copy()
    row_labels = []
    pivot_rows = []
    for (dataset, arch), part in df.groupby(["dataset", "arch"], sort=False):
        row_labels.append(f"{dataset.upper()} {arch}")
        pivot_rows.append(
            [
                float(part.loc[part["op_code"] == op, "accuracy"].iloc[0])
                for op in ["OP-1", "OP-2", "OP-3"]
            ]
        )
    arr = np.array(pivot_rows, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(arr, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3), ["OP-1", "OP-2", "OP-3"])
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("24-Clip Online Replay Accuracy by Dataset, Model, and OP")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=9, fontweight="semibold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    return _save(fig, "online_replay_accuracy_heatmap.png")


def plot_mc_dropout_comparison() -> Path:
    df = pd.read_csv(REPORTS_DIR / "online_mc_replay_matrix_20260402.csv")
    base = df[df["use_mc"] == False].copy()
    mc = df[df["use_mc"] == True].copy()
    merged = base.merge(
        mc,
        on=["dataset", "arch", "spec_key", "op_code"],
        suffixes=("_base", "_mc"),
    )
    merged["acc_delta"] = merged["accuracy_mc"] - merged["accuracy_base"]
    merged["label"] = merged["dataset"].str.upper() + " " + merged["arch"] + " " + merged["op_code"]
    merged = merged.sort_values(["dataset", "arch", "op_code"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    colors = ["#4caf50" if x > 0 else "#ef5350" if x < 0 else "#90a4ae" for x in merged["acc_delta"]]
    ax.bar(range(len(merged)), merged["acc_delta"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(range(len(merged)), merged["label"], rotation=45, ha="right")
    ax.set_ylabel("Accuracy Delta (MC on - MC off)")
    ax.set_title("MC Dropout Effect on Fixed Online Replay Matrix")
    return _save(fig, "online_mc_dropout_delta.png")


def plot_cross_dataset_transfer() -> Path:
    df = pd.read_csv(REPORTS_DIR / "cross_dataset_summary.csv")
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x - width / 2, df["in_f1"], width=width, label="In-dataset F1", color="#1f77b4")
    ax.bar(x + width / 2, df["cross_f1"], width=width, label="Cross-dataset F1", color="#ff7f0e")
    ax.set_xticks(x, df["label"], rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-Dataset Transfer is Strongly Directional")
    ax.legend(loc="upper right")
    return _save(fig, "cross_dataset_f1_comparison.png")


def plot_le2i_per_clip_heatmap() -> Path:
    data = json.loads((REPORTS_DIR / "diagnostic" / "online_replay_le2i_perclip_20260402.json").read_text())
    order = [
        "le2i_tcn:OP-1",
        "le2i_tcn:OP-2",
        "le2i_tcn:OP-3",
        "le2i_gcn:OP-1",
        "le2i_gcn:OP-2",
        "le2i_gcn:OP-3",
    ]
    clip_names = [row["clip"] for row in data[order[0]] if int(row.get("gt", 0)) == 1]
    matrix = []
    for key in order:
        clip_map = {row["clip"]: row for row in data[key]}
        row_vals = []
        for clip in clip_names:
            item = clip_map[clip]
            pred = int(item.get("pred", 0))
            uncertain = bool(item.get("uncertain", False))
            if pred == 1:
                row_vals.append(2)
            elif uncertain:
                row_vals.append(1)
            else:
                row_vals.append(0)
        matrix.append(row_vals)

    arr = np.array(matrix, dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(["#d32f2f", "#fbc02d", "#2e7d32"])
    norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(clip_names)), [c.replace("corridor__", "").replace("kitchen__", "") for c in clip_names], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [k.replace(":", " ") for k in order])
    ax.set_title("LE2I Fall-Clip Outcomes Under the Online Replay Path")
    ax.set_xlabel("Fall clips only")
    legend_handles = [
        plt.matplotlib.patches.Patch(color="#2e7d32", label="Detected fall"),
        plt.matplotlib.patches.Patch(color="#fbc02d", label="Uncertain, not alerted"),
        plt.matplotlib.patches.Patch(color="#d32f2f", label="Missed"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    return _save_diagnostic(fig, "le2i_per_clip_outcome_heatmap.png")


def plot_stability_errorbars() -> Path:
    df = pd.read_csv(REPORTS_DIR / "stability_summary.csv")
    labels = df["dataset"].str.upper() + " " + df["arch"].str.upper()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x, df["f1_mean"], yerr=df["f1_ci95"], color=["#1f77b4", "#6baed6", "#ff7f0e", "#fdae6b"], capsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 mean ± 95% CI")
    ax.set_title("Seed Stability of Final Model Families")
    return _save(fig, "stability_f1_errorbars.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also generate diagnostic-only figures that are not part of the main report pack.",
    )
    args = parser.parse_args()

    outputs = [
        plot_online_replay_accuracy_heatmap(),
        plot_mc_dropout_comparison(),
        plot_cross_dataset_transfer(),
        plot_stability_errorbars(),
    ]
    if args.include_diagnostics:
        outputs.append(plot_le2i_per_clip_heatmap())
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
