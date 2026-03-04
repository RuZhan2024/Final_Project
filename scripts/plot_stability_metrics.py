#!/usr/bin/env python3
"""Plot multi-seed stability metrics from evaluation JSON files.

Expected input naming (recommended):
  outputs/metrics/<exp>_seed<seed>.json
or any path list passed via --files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GLOB = "outputs/metrics/*seed*.json"


def _load(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected JSON object")
    return d


def _first_num(*vals: Any) -> Optional[float]:
    for v in vals:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    ops = d.get("ops") if isinstance(d.get("ops"), dict) else {}
    sel = d.get("selected") if isinstance(d.get("selected"), dict) else {}
    key = str(sel.get("name", "op2")).strip().lower()
    op = ops.get(key) if isinstance(ops.get(key), dict) else (ops.get("op2") if isinstance(ops.get("op2"), dict) else {})
    totals = d.get("totals") if isinstance(d.get("totals"), dict) else {}
    return {
        "f1": _first_num(op.get("f1"), totals.get("f1")) or float("nan"),
        "recall": _first_num(op.get("recall"), totals.get("recall"), totals.get("avg_recall")) or float("nan"),
        "fa24h": _first_num(op.get("fa24h"), totals.get("fa24h")) or float("nan"),
    }


def _group_label(path: Path) -> str:
    stem = path.stem.lower()
    # Strip trailing seed suffixes to get candidate group label.
    stem = re.sub(r"([_-]seed\d+)$", "", stem)
    stem = re.sub(r"([_-]s\d+)$", "", stem)
    return stem


def _boxplot(grouped: Dict[str, List[Dict[str, float]]], out_fig: Path) -> None:
    groups = sorted(grouped.keys())
    if not groups:
        raise SystemExit("[ERR] no grouped stability data found")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    keys = ["f1", "recall", "fa24h"]
    titles = ["F1", "Recall", "FA24h"]

    for ax, key, title in zip(axes, keys, titles):
        data = []
        labels = []
        for g in groups:
            vals = np.asarray([row[key] for row in grouped[g]], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(g)
        if not data:
            ax.set_title(f"{title} (no data)")
            continue
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"{title} distribution")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        if key in ("f1", "recall"):
            ax.set_ylim(0.0, 1.05)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot multi-seed stability metrics")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out_fig", default="artifacts/figures/stability/fc_stability_boxplot.png")
    args = ap.parse_args()

    files = [Path(p) for p in args.files] if args.files else sorted(Path(".").glob(args.glob))
    if not files:
        raise SystemExit(f"[ERR] no files found. Provide --files or check --glob ({args.glob})")

    grouped: Dict[str, List[Dict[str, float]]] = {}
    for p in files:
        d = _load(p)
        grouped.setdefault(_group_label(p), []).append(_extract_metrics(d))

    _boxplot(grouped, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")
    print(f"[info] groups={len(grouped)} files={len(files)}")


if __name__ == "__main__":
    main()
