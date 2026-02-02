#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/plot_fa_recall.py

Plot Recall vs False Alarms per day (FA/24h) from an ops YAML produced by eval/fit_ops.py.

Why this plot matters
---------------------
During deployment, you usually trade off:
- HIGH recall (detect as many real falls as possible)
vs
- LOW false alarms per day (avoid alarm fatigue)

This plot lets you SEE that tradeoff across different tau_high thresholds,
and highlights your chosen operating points OP-1 / OP-2 / OP-3.

Input expected
--------------
This script expects a YAML file like the one produced by eval/fit_ops.py:

  ops:
    op1: { tau_high: ..., tau_low: ..., recall: ..., fa24h: ..., ... }
    op2: { ... }
    op3: { ... }

  sweep:
    - { tau_high: ..., tau_low: ..., recall: ..., fa24h: ..., f1: ... }
    - ...

But we also support older formats where sweep is a dict of lists.

Output
------
- Saves a PNG/PDF if --out is set
- Optionally shows the plot window with --show

No color is hard-coded; matplotlib defaults are used.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.yamlio import yaml_load_simple


# ============================================================
# 1) Helpers: parse sweep data from yaml in a tolerant way
# ============================================================
def _to_float(x: Any, default: float = float("nan")) -> float:
    """Convert x to float safely; return default if conversion fails."""
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def _extract_sweep(ops_yaml: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays from ops_yaml:
      tau_highs, recalls, fa24hs, f1s

    We support two common shapes:

    A) New style (list of dicts):
       ops_yaml["sweep"] = [
         {"tau_high": 0.7, "recall": 0.9, "fa24h": 1.2, "f1": 0.85},
         ...
       ]

    B) Legacy style (dict of lists):
       ops_yaml["sweep"] = {
         "thr": [...], "recall": [...], "fa24h": [...], "f1": [...]
       }

    Returns numpy arrays of same length (may contain NaN where missing).
    """
    sweep = ops_yaml.get("sweep", None)
    if sweep is None:
        raise ValueError("ops yaml missing 'sweep' field. Run eval/fit_ops.py first.")

    # -------------------------
    # Case A: list of dict rows
    # -------------------------
    if isinstance(sweep, list):
        tau = np.array([_to_float(r.get("tau_high")) for r in sweep], dtype=np.float64)
        rec = np.array([_to_float(r.get("recall")) for r in sweep], dtype=np.float64)
        fa = np.array([_to_float(r.get("fa24h")) for r in sweep], dtype=np.float64)
        f1 = np.array([_to_float(r.get("f1")) for r in sweep], dtype=np.float64)
        return tau, rec, fa, f1

    # -------------------------
    # Case B: dict of lists
    # -------------------------
    if isinstance(sweep, dict):
        # Some old scripts used "thr" instead of "tau_high"
        tau_list = sweep.get("tau_high", None) or sweep.get("thr", None) or sweep.get("threshold", None)
        rec_list = sweep.get("recall", None)
        fa_list = sweep.get("fa24h", None) or sweep.get("fa_day", None) or sweep.get("fa_per_day", None)
        f1_list = sweep.get("f1", None)

        if tau_list is None or rec_list is None or fa_list is None:
            raise ValueError("sweep dict missing required keys (need thr/tau_high, recall, fa24h)")

        tau = np.asarray(tau_list, dtype=np.float64)
        rec = np.asarray(rec_list, dtype=np.float64)
        fa = np.asarray(fa_list, dtype=np.float64)

        if f1_list is None:
            f1 = np.full_like(tau, np.nan, dtype=np.float64)
        else:
            f1 = np.asarray(f1_list, dtype=np.float64)

        return tau, rec, fa, f1

    raise ValueError(f"Unsupported sweep type: {type(sweep)}")


def _extract_ops_points(ops_yaml: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract OP-1/2/3 points from ops_yaml.

    We return a dict:
      {"op1": {"tau_high":..., "recall":..., "fa24h":...}, ...}

    If some keys are missing, we just skip them.
    """
    out: Dict[str, Dict[str, float]] = {}

    ops_block = ops_yaml.get("ops", None)
    if not isinstance(ops_block, dict):
        return out

    for op_key in ["op1", "op2", "op3"]:
        row = ops_block.get(op_key, None)
        if not isinstance(row, dict):
            continue

        out[op_key] = {
            "tau_high": _to_float(row.get("tau_high")),
            "recall": _to_float(row.get("recall")),
            "fa24h": _to_float(row.get("fa24h")),
            "f1": _to_float(row.get("f1")),
        }

    return out


# ============================================================
# 2) Plotting utilities
# ============================================================
def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    """Return a boolean mask True where all arrays have finite values."""
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def plot_recall_vs_fa(
    tau_high: np.ndarray,
    recall: np.ndarray,
    fa24h: np.ndarray,
    f1: np.ndarray,
    ops_points: Dict[str, Dict[str, float]],
    *,
    title: str,
    out_path: Optional[str],
    log_x: bool,
    show: bool,
) -> None:
    """
    Create and save/show the Recall vs FA/day plot.

    x-axis: FA/24h (false alerts per day)
    y-axis: recall

    log_x:
      If True, x-axis uses log scale. This is useful when FA ranges widely,
      e.g. 0.1/day to 20/day.
    """
    # Filter to finite sweep points
    m = _finite_mask(recall, fa24h)
    tau2, rec2, fa2, f12 = tau_high[m], recall[m], fa24h[m], f1[m]

    # Sort by FA (x-axis) to create a nicer line
    order = np.argsort(fa2)
    tau2, rec2, fa2, f12 = tau2[order], rec2[order], fa2[order], f12[order]

    plt.figure()
    plt.plot(fa2, rec2, marker="o", linestyle="-", label="sweep")

    # Mark OP points (if present)
    for k, label in [("op1", "OP-1"), ("op2", "OP-2"), ("op3", "OP-3")]:
        p = ops_points.get(k)
        if not p:
            continue
        x = p["fa24h"]
        y = p["recall"]
        if np.isfinite(x) and np.isfinite(y):
            plt.scatter([x], [y], s=80, label=label)

    if log_x:
        # Log scale is common for FA plots because you care about "order of magnitude"
        # improvements (e.g. from 10/day to 1/day).
        plt.xscale("log")

    plt.xlabel("False alarms per day (FA/24h)")
    plt.ylabel("Event recall")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[ok] saved plot: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_f1_vs_threshold(
    tau_high: np.ndarray,
    f1: np.ndarray,
    ops_points: Dict[str, Dict[str, float]],
    *,
    title: str,
    out_path: Optional[str],
    show: bool,
) -> None:
    """
    Optional second plot: F1 vs tau_high.

    Why:
    - Helps you see where the "balanced" operating region is.
    - You can confirm that OP-2 (balanced) is near the global F1 peak.

    If f1 is all NaN (older sweep data), this plot will be skipped.
    """
    if not np.any(np.isfinite(f1)):
        print("[info] f1 is missing/NaN in sweep; skipping F1-vs-threshold plot.")
        return

    m = _finite_mask(tau_high, f1)
    tau2, f12 = tau_high[m], f1[m]

    order = np.argsort(tau2)
    tau2, f12 = tau2[order], f12[order]

    plt.figure()
    plt.plot(tau2, f12, marker="o", linestyle="-", label="sweep")

    for k, label in [("op1", "OP-1"), ("op2", "OP-2"), ("op3", "OP-3")]:
        p = ops_points.get(k)
        if not p:
            continue
        x = p["tau_high"]
        y = p["f1"]
        if np.isfinite(x) and np.isfinite(y):
            plt.scatter([x], [y], s=80, label=label)

    plt.xlabel("tau_high (start threshold)")
    plt.ylabel("Event F1")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[ok] saved plot: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3) Main CLI
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ops_yaml", required=True, help="YAML produced by eval/fit_ops.py")
    ap.add_argument("--out", default="", help="Save Recall-vs-FA plot to this file (png/pdf).")
    ap.add_argument("--out_f1", default="", help="Save F1-vs-threshold plot to this file (png/pdf).")

    ap.add_argument("--title", default="Recall vs False Alarms per day")
    ap.add_argument("--title_f1", default="F1 vs tau_high")
    ap.add_argument("--log_x", action="store_true", help="Use log scale for FA/day axis.")
    ap.add_argument("--show", action="store_true", help="Show plot window instead of closing.")

    args = ap.parse_args()

    # Load ops yaml (our simple yaml loader)
    ops_yaml = yaml_load_simple(args.ops_yaml)

    # Extract arrays from sweep
    tau_high, recall, fa24h, f1 = _extract_sweep(ops_yaml)

    # Extract OP markers if available
    ops_points = _extract_ops_points(ops_yaml)

    # Print a short textual summary (useful when working over SSH)
    if ops_points:
        for k in ["op1", "op2", "op3"]:
            if k in ops_points:
                p = ops_points[k]
                print(f"[{k}] tau_high={p['tau_high']:.4f} recall={p['recall']:.4f} fa24h={p['fa24h']:.4f} f1={p['f1']:.4f}")

    # Main plot
    plot_recall_vs_fa(
        tau_high,
        recall,
        fa24h,
        f1,
        ops_points,
        title=str(args.title),
        out_path=(str(args.out).strip() if str(args.out).strip() else None),
        log_x=bool(args.log_x),
        show=bool(args.show),
    )

    # Optional secondary plot
    plot_f1_vs_threshold(
        tau_high,
        f1,
        ops_points,
        title=str(args.title_f1),
        out_path=(str(args.out_f1).strip() if str(args.out_f1).strip() else None),
        show=bool(args.show),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
