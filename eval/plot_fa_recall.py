#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/plot_fa_recall.py

Plot FA/24h vs Recall from one or more JSON reports produced by eval/metrics.py.

Improvements:
  - Plots Pareto frontier (optional).
  - Avoids connecting points in threshold order when x is not monotonic.
  - Can log-scale x-axis.

Usage:
  python eval/plot_fa_recall.py --reports reports/a.json --reports reports/b.json --out_fig figs/curve.png
"""


from __future__ import annotations

# -------------------------
# Path bootstrap (so `from core.*` works when running as a script)
# -------------------------
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.metrics import pareto_frontier


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_reports_arg(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        out.extend([p.strip() for p in str(v).split(",") if p.strip()])
    # de-dup preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _extract_curve(rep: Dict, *, x_key: str = "fa_per_hour") -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    sweep = rep.get("sweep") or {}
    recall = np.asarray(sweep.get("recall", []), dtype=float)

    # Prefer the requested x metric; fall back sensibly if missing.
    x = np.asarray(sweep.get(x_key, []), dtype=float)
    kind = x_key

    if (x.size == 0) and (x_key == "fa_per_hour"):
        fa24h = np.asarray(sweep.get("fa24h", []), dtype=float)
        if fa24h.size:
            x = fa24h / 24.0
            kind = "fa_per_hour"

    if recall.size and x.size:
        pareto_idx = np.asarray(sweep.get("pareto_idx", []), dtype=int)
        label = "FA/hour" if kind == "fa_per_hour" else ("FA/24h" if kind == "fa24h" else kind)
        return recall, x, label, pareto_idx

    # fallback: FPR (unitless)
    fpr = np.asarray(sweep.get("fpr", []), dtype=float)
    if recall.size and fpr.size:
        return recall, fpr, "FPR", np.asarray([], dtype=int)

    return np.asarray([]), np.asarray([]), "", np.asarray([], dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", action="append", required=True, help="Report JSON path(s). Repeatable / comma-separated.")
    ap.add_argument("--title", default="FA/hour vs Recall")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--x", choices=["fa24h","fa_per_hour"], default="fa_per_hour", help="X-axis metric.")
    ap.add_argument("--out_fig", required=True)
    ap.add_argument("--xlog", action="store_true", help="Log-scale x-axis.")
    ap.add_argument("--plot_pareto", action="store_true", help="Plot Pareto frontier only (if available).")
    args = ap.parse_args()

    report_paths = _split_reports_arg(args.reports)
    if not report_paths:
        raise SystemExit("No report paths provided")

    plt.figure()
    xlab = None

    for rp in report_paths:
        if not os.path.isfile(rp):
            raise SystemExit(f"Report not found: {rp}")
        rep = _read_json(rp)
        recall, x, kind, pareto_idx = _extract_curve(rep, x_key=args.x)
        label = rep.get("label") or rep.get("dataset") or os.path.splitext(os.path.basename(rp))[0]
        if xlab is None:
            xlab = kind or "FA/24h"

        if recall.size == 0 or x.size == 0:
            continue

        if args.plot_pareto:
            if pareto_idx.size:
                rr = recall[pareto_idx]
                xx = x[pareto_idx]
                # ensure sorted by x
                order = np.argsort(xx)
                plt.plot(xx[order], rr[order], marker="o", linestyle="-", label=str(label))
            else:
                # compute pareto on the fly
                keep, rr, xx = pareto_frontier(recall, x)
                if keep.size:
                    order = np.argsort(xx[keep])
                    plt.plot(xx[keep][order], rr[keep][order], marker="o", linestyle="-", label=str(label))
        else:
            plt.scatter(x, recall, label=str(label), s=18)

    plt.xlabel(xlab or "FA/24h")
    plt.ylabel("Recall")
    plt.title(args.title)
    if args.subtitle:
        plt.suptitle(args.subtitle, y=0.98, fontsize=10)

    if args.xlog:
        plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    plt.savefig(args.out_fig, dpi=200, bbox_inches="tight")
    print(f"[ok] wrote figure: {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
