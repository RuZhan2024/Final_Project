#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot FA/24h vs recall curves.

Supports inputs from:
- metrics.py JSON reports (expects a top-level dict with a "sweep" field)
- sweep-only JSON files (either list[dict] or dict-of-arrays)

Makefile compatibility:
- accepts --reports (alias of --sweeps)
- accepts --out_fig (alias of --out)
"""

import os, sys, json, glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# repo root bootstrap (so `from fall_detection.core.*` works)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fall_detection.core.metrics import pareto_frontier


def _load_json(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def _extract_curve(obj):
    """Return (fa24h, recall) arrays from either a report dict or sweep-only JSON."""
    sweep = obj
    if isinstance(obj, dict):
        sweep = obj.get("sweep", None)

    fa = []
    rec = []

    # dict-of-arrays
    if isinstance(sweep, dict):
        fa = list(sweep.get("fa24h") or sweep.get("fa_per_day") or sweep.get("fa") or [])
        rec = list(sweep.get("recall") or sweep.get("avg_recall") or [])
        return np.asarray(fa, dtype=float), np.asarray(rec, dtype=float)

    # list-of-dicts
    if isinstance(sweep, list):
        for r in sweep:
            if not isinstance(r, dict):
                continue
            f = r.get("fa24h", r.get("fa_per_day", r.get("fa", None)))
            y = r.get("recall", r.get("avg_recall", None))
            if f is None or y is None:
                continue
            try:
                fa.append(float(f))
                rec.append(float(y))
            except Exception:
                continue
        return np.asarray(fa, dtype=float), np.asarray(rec, dtype=float)

    return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=None, help="One or more JSON files or globs (sweep or report).")
    ap.add_argument("--reports", nargs="+", default=None, help="Alias for --sweeps (Makefile compatibility).")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for each curve.")
    ap.add_argument("--title", default="FA/24h vs Recall")
    ap.add_argument("--out", default="", help="Output png path (optional).")
    ap.add_argument("--out_fig", default="", help="Alias for --out (Makefile compatibility).")
    ap.add_argument("--plot_pareto", action="store_true")
    ap.add_argument("--xlog", action="store_true")
    args = ap.parse_args()

    patterns = []
    if args.sweeps:
        patterns.extend(list(args.sweeps))
    if args.reports:
        patterns.extend(list(args.reports))
    if not patterns:
        ap.error("Provide --sweeps or --reports")

    sweep_paths = []
    for s in patterns:
        sweep_paths.extend(sorted(glob.glob(s)))
    if not sweep_paths:
        raise FileNotFoundError("No sweep/report files found.")

    labels = args.labels
    if labels is None or len(labels) == 0:
        labels = [os.path.basename(p) for p in sweep_paths]
    if len(labels) != len(sweep_paths):
        labels = [os.path.basename(p) for p in sweep_paths]

    out_path = args.out_fig or args.out

    plt.figure(figsize=(8, 5))

    for p, label in zip(sweep_paths, labels):
        js = _load_json(p)
        x, recall = _extract_curve(js)

        if x.size == 0:
            print(f"[warn] no sweep points in: {p}")
            continue

        if args.plot_pareto:
            xx, rr = pareto_frontier(x, recall, minimize_x=True, maximize_y=True)
            order = np.argsort(xx)

            xx_plot = xx[order]
            if args.xlog:
                pos = xx_plot[xx_plot > 0]
                eps = (float(pos.min()) * 0.1) if pos.size else 1e-6
                xx_plot = np.where(xx_plot > 0, xx_plot, eps)

            plt.plot(xx_plot, rr[order], marker="o", linestyle="-", label=str(label))
        else:
            x_plot = x
            if args.xlog:
                pos = x_plot[x_plot > 0]
                eps = (float(pos.min()) * 0.1) if pos.size else 1e-6
                x_plot = np.where(x_plot > 0, x_plot, eps)
            plt.scatter(x_plot, recall, label=str(label), s=18)

    plt.title(args.title)
    plt.xlabel("False Alarms per 24h (FA/24h)")
    plt.ylabel("Recall")
    plt.grid(True, which="both", alpha=0.35)

    if args.xlog:
        plt.xscale("log")

    plt.legend(loc="best", fontsize=9)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"[ok] saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
