#!/usr/bin/env python3
"""Plot video-level confusion matrix from eval metrics JSON.

Expected input: outputs from scripts/eval_metrics.py containing:
- detail.per_video[*].event_metrics.n_gt_events
- detail.per_video[*].event_metrics.n_alert_events
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return d


def _counts(d: Dict[str, Any]) -> Tuple[int, int, int, int]:
    detail = d.get("detail") if isinstance(d.get("detail"), dict) else {}
    per_video = detail.get("per_video") if isinstance(detail.get("per_video"), dict) else {}

    tp = fp = fn = tn = 0
    for _, block in per_video.items():
        if not isinstance(block, dict):
            continue
        em = block.get("event_metrics") if isinstance(block.get("event_metrics"), dict) else {}
        gt = int(float(em.get("n_gt_events", 0) or 0) > 0)
        pred = int(float(em.get("n_alert_events", 0) or 0) > 0)
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def plot(metrics_json: Path, out_fig: Path, normalize: bool) -> None:
    d = _load(metrics_json)
    tp, fp, fn, tn = _counts(d)
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)

    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        shown = mat / row_sum
    else:
        shown = mat

    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(shown, cmap="Blues", vmin=0.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ratio" if normalize else "count")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No-Fall", "Pred: Fall"])
    ax.set_yticklabels(["GT: No-Fall", "GT: Fall"])

    for i in range(2):
        for j in range(2):
            val = shown[i, j]
            raw = int(mat[i, j])
            txt = f"{val:.2f}\n(n={raw})" if normalize else str(raw)
            color = "white" if val > (shown.max() * 0.5 if shown.max() > 0 else 0) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    ap_auc = d.get("ap_auc") if isinstance(d.get("ap_auc"), dict) else {}
    ap = ap_auc.get("ap")
    arch = str(d.get("arch", "model")).upper()
    suffix = f", AP={float(ap):.3f}" if isinstance(ap, (int, float)) else ""
    ax.set_title(f"{arch} Video-level Confusion Matrix{suffix}")

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser(description="Plot video-level confusion matrix from eval JSON")
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--out_fig", required=True)
    ap.add_argument("--normalize", type=int, default=1, help="1=row-normalized, 0=raw counts")
    args = ap.parse_args()

    plot(Path(args.metrics_json), Path(args.out_fig), normalize=bool(args.normalize))
    print(f"[ok] saved: {args.out_fig}")


if __name__ == "__main__":
    main()
