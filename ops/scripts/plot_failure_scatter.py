#!/usr/bin/env python3
"""Plot per-video failure diagnostics from eval metrics JSON.

x-axis: alert fraction (state_counts.alert_frac)
y-axis: false alerts per day (event_metrics.fa24h)
Color/group by video-level status: TP/FP/FN/TN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "TP": "#2ca02c",
    "FP": "#d62728",
    "FN": "#ff7f0e",
    "TN": "#1f77b4",
}
MARKERS = {
    "TP": "o",
    "FP": "X",
    "FN": "^",
    "TN": "s",
}


def _load(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return d


def _rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    detail = d.get("detail") if isinstance(d.get("detail"), dict) else {}
    per_video = detail.get("per_video") if isinstance(detail.get("per_video"), dict) else {}
    out: List[Dict[str, Any]] = []

    for vid, block in per_video.items():
        if not isinstance(block, dict):
            continue
        em = block.get("event_metrics") if isinstance(block.get("event_metrics"), dict) else {}
        sc = block.get("state_counts") if isinstance(block.get("state_counts"), dict) else {}

        gt = int(float(em.get("n_gt_events", 0) or 0) > 0)
        pred = int(float(em.get("n_alert_events", 0) or 0) > 0)
        if gt == 1 and pred == 1:
            status = "TP"
        elif gt == 0 and pred == 1:
            status = "FP"
        elif gt == 1 and pred == 0:
            status = "FN"
        else:
            status = "TN"

        alert_frac = float(sc.get("alert_frac", 0.0) or 0.0)
        fa24h = float(em.get("fa24h", em.get("false_alerts_per_day", 0.0)) or 0.0)
        recall = float(em.get("event_recall", em.get("recall", np.nan)) or np.nan)

        out.append(
            {
                "video": str(vid),
                "status": status,
                "alert_frac": alert_frac,
                "fa24h": fa24h,
                "recall": recall,
            }
        )
    return out


def _plot_scatter(ax1: Any, ax2: Any, rows: List[Dict[str, Any]]) -> None:
    for status in ["TP", "FP", "FN", "TN"]:
        grp = [r for r in rows if r["status"] == status]
        if not grp:
            continue
        x = np.asarray([r["alert_frac"] for r in grp], dtype=float)
        y_fa = np.asarray([r["fa24h"] for r in grp], dtype=float)
        y_rec = np.asarray([r["recall"] for r in grp], dtype=float)
        ax1.scatter(
            x,
            y_fa,
            alpha=0.85,
            s=64,
            c=COLORS[status],
            marker=MARKERS[status],
            edgecolors="white",
            linewidths=0.6,
            label=f"{status} (n={len(grp)})",
        )
        ax2.scatter(
            x,
            y_rec,
            alpha=0.85,
            s=64,
            c=COLORS[status],
            marker=MARKERS[status],
            edgecolors="white",
            linewidths=0.6,
            label=f"{status} (n={len(grp)})",
        )


def _plot_hexbin(ax1: Any, ax2: Any, rows: List[Dict[str, Any]]) -> None:
    x_all = np.asarray([r["alert_frac"] for r in rows], dtype=float)
    y_fa_all = np.asarray([r["fa24h"] for r in rows], dtype=float)
    y_rec_all = np.asarray([r["recall"] for r in rows], dtype=float)
    hb1 = ax1.hexbin(x_all, y_fa_all, gridsize=18, mincnt=1, cmap="Blues", alpha=0.9)
    hb2 = ax2.hexbin(x_all, y_rec_all, gridsize=18, mincnt=1, cmap="Blues", alpha=0.9)
    plt.colorbar(hb1, ax=ax1, label="count")
    plt.colorbar(hb2, ax=ax2, label="count")

    # Overlay status medians for interpretability
    for status in ["TP", "FP", "FN", "TN"]:
        grp = [r for r in rows if r["status"] == status]
        if not grp:
            continue
        x = np.asarray([r["alert_frac"] for r in grp], dtype=float)
        y_fa = np.asarray([r["fa24h"] for r in grp], dtype=float)
        y_rec = np.asarray([r["recall"] for r in grp], dtype=float)
        mx = float(np.nanmedian(x))
        my_fa = float(np.nanmedian(y_fa))
        rec_finite = y_rec[np.isfinite(y_rec)]
        my_rec = float(np.median(rec_finite)) if rec_finite.size else 0.0
        ax1.scatter(
            [mx],
            [my_fa],
            s=140,
            c=COLORS[status],
            marker=MARKERS[status],
            edgecolors="black",
            linewidths=0.8,
            label=f"{status} (n={len(grp)})",
            zorder=5,
        )
        ax2.scatter(
            [mx],
            [my_rec],
            s=140,
            c=COLORS[status],
            marker=MARKERS[status],
            edgecolors="black",
            linewidths=0.8,
            label=f"{status} (n={len(grp)})",
            zorder=5,
        )


def _plot_box(ax1: Any, ax2: Any, rows: List[Dict[str, Any]], fa_log: bool) -> None:
    statuses = ["TP", "FP", "FN", "TN"]
    fa_data: List[np.ndarray] = []
    alert_data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    for s in statuses:
        grp = [r for r in rows if r["status"] == s]
        if not grp:
            continue
        fa = np.asarray([r["fa24h"] for r in grp], dtype=float)
        al = np.asarray([r["alert_frac"] for r in grp], dtype=float)
        fa_data.append(fa)
        alert_data.append(al)
        labels.append(f"{s}\n(n={len(grp)})")
        colors.append(COLORS[s])

    bp1 = ax1.boxplot(fa_data, patch_artist=True, labels=labels, showfliers=True)
    bp2 = ax2.boxplot(alert_data, patch_artist=True, labels=labels, showfliers=True)
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    ax1.set_ylabel("fa24h per video")
    ax1.set_title("Failure (box): FA24h by status")
    ax1.grid(True, axis="y", alpha=0.3)
    if fa_log:
        finite_pos = np.concatenate([x[x > 0] for x in fa_data if x.size]) if fa_data else np.array([])
        if finite_pos.size:
            ax1.set_yscale("log")

    ax2.set_ylabel("alert_frac per video")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Failure (box): alert_frac by status")
    ax2.grid(True, axis="y", alpha=0.3)


def plot(metrics_json: Path, out_fig: Path, fa_log: bool, style: str) -> None:
    d = _load(metrics_json)
    rows = _rows(d)
    if not rows:
        raise SystemExit("[ERR] no per_video rows found in metrics JSON")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    if style == "hexbin":
        _plot_hexbin(ax1, ax2, rows)
    elif style == "box":
        _plot_box(ax1, ax2, rows, fa_log=fa_log)
    else:
        _plot_scatter(ax1, ax2, rows)

    if style in {"hexbin", "scatter"}:
        ax1.set_xlabel("alert_frac per video")
        ax1.set_ylabel("fa24h per video")
        ax1.set_title(f"Failure ({style}): alert_frac vs fa24h")
        ax1.grid(True, alpha=0.3)
        if fa_log:
            finite_pos = [r["fa24h"] for r in rows if r["fa24h"] > 0]
            if finite_pos:
                ax1.set_yscale("log")

        ax2.set_xlabel("alert_frac per video")
        ax2.set_ylabel("event recall per video")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title(f"Failure ({style}): alert_frac vs recall")
        ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(loc="upper right", fontsize=9)
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend(loc="lower right", fontsize=9)

    arch = str(d.get("arch", "model")).upper()
    selected = d.get("selected") if isinstance(d.get("selected"), dict) else {}
    op_name = str(selected.get("name", "op2")).upper()
    fig.suptitle(f"{arch} Failure Analysis ({op_name})", fontsize=12)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-video failure diagnostics")
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--out_fig", required=True)
    ap.add_argument("--fa_log", type=int, default=1)
    ap.add_argument("--style", choices=["scatter", "hexbin", "box"], default="box")
    args = ap.parse_args()

    plot(Path(args.metrics_json), Path(args.out_fig), fa_log=bool(args.fa_log), style=args.style)
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
