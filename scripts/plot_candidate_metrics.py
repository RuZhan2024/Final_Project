#!/usr/bin/env python3
"""Plot final-candidate metric comparison from evaluation JSON artifacts.

This script summarizes AP and selected operating-point metrics (precision, recall,
F1, FA24h) for FC1-FC4 style runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILES = [
    "outputs/metrics/tcn_le2i.json",
    "outputs/metrics/gcn_le2i.json",
    "outputs/metrics/tcn_caucafall.json",
    "outputs/metrics/gcn_caucafall.json",
]


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level object")
    return data


def _first_num(*vals: Any) -> Optional[float]:
    for v in vals:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _selected_op(data: Dict[str, Any]) -> Dict[str, Any]:
    ops = data.get("ops")
    if not isinstance(ops, dict):
        return {}
    sel = data.get("selected")
    if isinstance(sel, dict) and isinstance(sel.get("name"), str):
        key = sel["name"].strip().lower()
        blk = ops.get(key)
        if isinstance(blk, dict):
            return blk
    for key in ("op2", "op1", "op3"):
        blk = ops.get(key)
        if isinstance(blk, dict):
            return blk
    return {}


def _name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "_" not in stem:
        return stem
    arch, ds = stem.split("_", 1)
    return f"{arch.upper()}-{ds.upper()}"


def _extract_row(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    op = _selected_op(data)
    totals = data.get("totals") if isinstance(data.get("totals"), dict) else {}
    ap_auc = data.get("ap_auc") if isinstance(data.get("ap_auc"), dict) else {}

    return {
        "name": _name_from_path(path),
        "ap": _first_num(ap_auc.get("ap"), op.get("ap"), totals.get("ap")),
        "precision": _first_num(op.get("precision"), totals.get("precision")),
        "recall": _first_num(op.get("recall"), totals.get("recall"), totals.get("avg_recall")),
        "f1": _first_num(op.get("f1"), totals.get("f1")),
        "fa24h": _first_num(op.get("fa24h"), totals.get("fa24h")),
    }


def _num(v: Optional[float]) -> float:
    return float("nan") if v is None else float(v)


def plot(rows: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.18

    ap = np.asarray([_num(r["ap"]) for r in rows], dtype=float)
    pr = np.asarray([_num(r["precision"]) for r in rows], dtype=float)
    rc = np.asarray([_num(r["recall"]) for r in rows], dtype=float)
    f1 = np.asarray([_num(r["f1"]) for r in rows], dtype=float)
    fa = np.asarray([_num(r["fa24h"]) for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax1.bar(x - 1.5 * width, ap, width=width, label="AP")
    ax1.bar(x - 0.5 * width, pr, width=width, label="Precision")
    ax1.bar(x + 0.5 * width, rc, width=width, label="Recall")
    ax1.bar(x + 1.5 * width, f1, width=width, label="F1")
    ax1.set_title("Candidate Quality Metrics")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower left")

    ax2.bar(x, fa, width=0.55, color="#cc7a00")
    ax2.set_title("False Alerts per 24h (selected OP)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.grid(True, axis="y", alpha=0.3)
    finite_fa = fa[np.isfinite(fa)]
    if finite_fa.size > 0 and np.nanmax(finite_fa) > 50.0:
        ax2.set_yscale("log")
        ax2.set_ylabel("FA24h (log scale)")
    else:
        ax2.set_ylabel("FA24h")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot FC candidate metrics from eval JSON files")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--out_fig", default="artifacts/figures/pr_curves/fc1_fc4_ap_comparison.png")
    args = ap.parse_args()

    rows = [_extract_row(Path(p)) for p in args.files]
    plot(rows, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
