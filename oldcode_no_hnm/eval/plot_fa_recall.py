#!/usr/bin/env python3
"""Plot FA/24h vs Recall from one or more JSON reports produced by eval/metrics.py.

Designed to match the Makefile targets:
  python eval/plot_fa_recall.py --reports reports/le2i_tcn.json --title ... --out_fig ...

Enhancements vs old version
--------------------------
- --reports can be:
    * a single path
    * a comma-separated list
    * repeated flags: --reports a.json --reports b.json
- If FA/24h is unavailable in a report, it falls back to plotting FPR vs Recall.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_reports_arg(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        out.extend(parts)
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _extract_curve(report: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
    sweep = report.get("sweep") or report.get("curve") or {}
    recall = np.asarray(sweep.get("recall", []), dtype=float)
    fa24h = sweep.get("fa24h", sweep.get("fa_per_24h", []))
    fa24h_arr = np.asarray(fa24h, dtype=float) if len(fa24h) else np.asarray([], dtype=float)
    if recall.size and fa24h_arr.size:
        return recall, fa24h_arr, "FA/24h"

    # fallback to FPR
    fpr = np.asarray(sweep.get("fpr", []), dtype=float)
    if recall.size and fpr.size:
        return recall, fpr, "FPR"

    # fallback to ops points only
    ops = report.get("ops", {})
    pts_r: List[float] = []
    pts_x: List[float] = []
    for _, op in ops.items():
        if not isinstance(op, dict):
            continue
        r = op.get("recall")
        x = op.get("fa24h", op.get("fa_per_24h", op.get("fpr")))
        if r is None or x is None:
            continue
        pts_r.append(float(r))
        pts_x.append(float(x))
    if pts_r:
        return np.asarray(pts_r, dtype=float), np.asarray(pts_x, dtype=float), "ops"

    return np.asarray([], dtype=float), np.asarray([], dtype=float), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reports",
        action="append",
        required=True,
        help="Report JSON path(s). Can be repeated or comma-separated.",
    )
    ap.add_argument("--title", default="FA/24h vs Recall")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out_fig", required=True)
    ap.add_argument("--xlog", action="store_true", help="Log-scale x-axis (FA/24h).")
    args = ap.parse_args()

    report_paths = _split_reports_arg(args.reports)
    if not report_paths:
        raise SystemExit("No report paths provided")

    curves: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
    for rp in report_paths:
        if not os.path.isfile(rp):
            raise SystemExit(f"Report not found: {rp}")
        rep = _read_json(rp)
        recall, x, kind = _extract_curve(rep)
        label = rep.get("label") or rep.get("dataset") or os.path.splitext(os.path.basename(rp))[0]
        curves.append((str(label), recall, x, kind))

    # Decide x-axis label
    kinds = {k for (_, _, _, k) in curves if k}
    xlab = "FA per 24h" if "FA/24h" in kinds else ("False Positive Rate" if "FPR" in kinds else "x")

    plt.figure(figsize=(8, 5))
    for (label, recall, x, kind) in curves:
        if recall.size == 0 or x.size == 0:
            continue
        # Sort by x ascending for a nicer curve
        order = np.argsort(x)
        x_s = x[order]
        r_s = recall[order]
        plt.plot(x_s, r_s, marker=".", linewidth=1, label=label)

    plt.xlabel(xlab)
    plt.ylabel("Recall")
    plt.title(args.title)
    if args.subtitle:
        plt.suptitle(args.subtitle, y=0.94, fontsize=10)

    if args.xlog:
        plt.xscale("log")

    if len(curves) > 1:
        plt.legend(loc="best", fontsize=9)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    out_dir = os.path.dirname(args.out_fig)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=160)
    print(f"[OK] wrote: {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
