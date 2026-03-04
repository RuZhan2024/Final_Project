#!/usr/bin/env python3
"""Plot latency profile summaries from artifacts/reports/infer_profile_*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected top-level object")
    return d


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _label(d: Dict[str, Any], p: Path) -> str:
    profile = str(d.get("profile", "unknown"))
    arch = str(d.get("arch", "model")).upper()
    wm = "model" if bool(d.get("with_model", False)) else "io"
    return f"{profile}:{arch}:{wm}"


def plot(reports: List[Path], out_fig: Path) -> None:
    rows = []
    for p in reports:
        d = _load(p)
        lat = d.get("latency_ms") if isinstance(d.get("latency_ms"), dict) else {}
        rows.append(
            {
                "label": _label(d, p),
                "median": _f(lat.get("median")),
                "p95": _f(lat.get("p95")),
                "max": _f(lat.get("max")),
                "mean": _f(lat.get("mean")),
            }
        )

    if not rows:
        raise SystemExit("[ERR] no latency reports provided")

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - 1.5 * width, [r["median"] for r in rows], width=width, label="p50")
    ax.bar(x - 0.5 * width, [r["mean"] for r in rows], width=width, label="mean")
    ax.bar(x + 0.5 * width, [r["p95"] for r in rows], width=width, label="p95")
    ax.bar(x + 1.5 * width, [r["max"] for r in rows], width=width, label="max")

    ax.set_title("Latency Profile Summary")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reports",
        nargs="*",
        default=[],
        help="Latency profile JSON files. If omitted, auto-load artifacts/reports/infer_profile_*.json",
    )
    ap.add_argument("--out_fig", default="artifacts/figures/latency/latency_profile_summary.png")
    args = ap.parse_args()

    if args.reports:
        reports = [Path(p) for p in args.reports]
    else:
        reports = sorted(Path("artifacts/reports").glob("infer_profile_*.json"))

    plot(reports, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
