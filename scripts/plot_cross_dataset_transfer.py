#!/usr/bin/env python3
"""Plot in-domain vs cross-dataset transfer metrics.

Input can be a JSON manifest or direct --in_domain/--cross_domain file lists.
Manifest format:
{
  "pairs": [
    {"label": "TCN LE2i->CAUCAFall", "in_domain": "...json", "cross": "...json"},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


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


def _metrics(path: Path) -> Dict[str, float]:
    d = _load(path)
    ops = d.get("ops") if isinstance(d.get("ops"), dict) else {}
    sel = d.get("selected") if isinstance(d.get("selected"), dict) else {}
    key = str(sel.get("name", "op2")).strip().lower()
    op = ops.get(key) if isinstance(ops.get(key), dict) else (ops.get("op2") if isinstance(ops.get("op2"), dict) else {})
    totals = d.get("totals") if isinstance(d.get("totals"), dict) else {}
    ap_auc = d.get("ap_auc") if isinstance(d.get("ap_auc"), dict) else {}
    return {
        "ap": _first_num(ap_auc.get("ap"), op.get("ap"), totals.get("ap")) or float("nan"),
        "f1": _first_num(op.get("f1"), totals.get("f1")) or float("nan"),
        "recall": _first_num(op.get("recall"), totals.get("recall"), totals.get("avg_recall")) or float("nan"),
        "fa24h": _first_num(op.get("fa24h"), totals.get("fa24h")) or float("nan"),
    }


def _load_pairs(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.manifest:
        d = _load(Path(args.manifest))
        pairs = d.get("pairs") if isinstance(d.get("pairs"), list) else []
        out: List[Dict[str, str]] = []
        for i, p in enumerate(pairs):
            if not isinstance(p, dict):
                continue
            lab = str(p.get("label", f"pair_{i+1}"))
            in_dom = str(p.get("in_domain", "")).strip()
            cross = str(p.get("cross", "")).strip()
            if in_dom and cross:
                out.append({"label": lab, "in_domain": in_dom, "cross": cross})
        return out

    n = min(len(args.in_domain), len(args.cross_domain))
    out: List[Dict[str, str]] = []
    for i in range(n):
        label = args.labels[i] if i < len(args.labels) else f"pair_{i+1}"
        out.append({"label": label, "in_domain": args.in_domain[i], "cross": args.cross_domain[i]})
    return out


def plot(pairs: List[Dict[str, str]], out_fig: Path) -> None:
    if not pairs:
        raise SystemExit("[ERR] no valid pairs. Provide --manifest or --in_domain/--cross_domain")

    labels = [p["label"] for p in pairs]
    ind = [_metrics(Path(p["in_domain"])) for p in pairs]
    crs = [_metrics(Path(p["cross"])) for p in pairs]

    x = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # F1
    axes[0].bar(x - width / 2, [m["f1"] for m in ind], width=width, label="in-domain")
    axes[0].bar(x + width / 2, [m["f1"] for m in crs], width=width, label="cross")
    axes[0].set_title("F1 transfer")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="lower left")

    # Recall
    axes[1].bar(x - width / 2, [m["recall"] for m in ind], width=width, label="in-domain")
    axes[1].bar(x + width / 2, [m["recall"] for m in crs], width=width, label="cross")
    axes[1].set_title("Recall transfer")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)

    # FA24h
    fa_in = np.asarray([m["fa24h"] for m in ind], dtype=float)
    fa_cr = np.asarray([m["fa24h"] for m in crs], dtype=float)
    axes[2].bar(x - width / 2, fa_in, width=width, label="in-domain")
    axes[2].bar(x + width / 2, fa_cr, width=width, label="cross")
    axes[2].set_title("FA24h transfer")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].grid(True, axis="y", alpha=0.3)
    finite = np.concatenate([fa_in[np.isfinite(fa_in)], fa_cr[np.isfinite(fa_cr)]])
    if finite.size and np.nanmax(finite) > 50.0:
        axes[2].set_yscale("log")

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cross-dataset transfer bars")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--in_domain", nargs="*", default=[])
    ap.add_argument("--cross_domain", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out_fig", default="artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png")
    args = ap.parse_args()

    pairs = _load_pairs(args)
    plot(pairs, Path(args.out_fig))
    print(f"[ok] saved: {args.out_fig}")
    print(f"[info] pairs={len(pairs)}")


if __name__ == "__main__":
    main()
