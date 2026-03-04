#!/usr/bin/env python3
"""Extract aggregate evaluation metrics and print a Markdown comparison table."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return data


def _pick_aggregate_block(data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    for key in ("aggregate", "summary", "overall", "totals"):
        blk = data.get(key)
        if isinstance(blk, dict):
            return key, blk
    return "none", {}


def _pick_selected_ops_block(data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    ops = data.get("ops")
    if not isinstance(ops, dict):
        return "none", {}
    sel = data.get("selected")
    sel_name = "op2"
    if isinstance(sel, dict) and isinstance(sel.get("name"), str) and sel.get("name"):
        sel_name = str(sel["name"]).strip().lower()
    op_blk = ops.get(sel_name)
    if isinstance(op_blk, dict):
        return sel_name, op_blk
    # fallback
    for k in ("op2", "op1", "op3"):
        if isinstance(ops.get(k), dict):
            return k, ops[k]
    return "none", {}


def _first_num(*vals: Any) -> Optional[float]:
    for v in vals:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def _parse_model_dataset(path: str) -> tuple[str, str]:
    base = os.path.basename(path)
    m = re.match(r"(gcn|tcn)_(.+)\.json$", base, flags=re.IGNORECASE)
    if not m:
        return "unknown", base
    return m.group(1).lower(), m.group(2).lower()


def summarize_one(path: str) -> Dict[str, Any]:
    data = _load_json(path)
    agg_key, agg = _pick_aggregate_block(data)
    op_key, op = _pick_selected_ops_block(data)
    ap_auc = data.get("ap_auc") if isinstance(data.get("ap_auc"), dict) else {}

    model, dataset = _parse_model_dataset(path)

    ap = _first_num(
        ap_auc.get("ap"),
        op.get("ap"),
        agg.get("ap"),
    )
    f1 = _first_num(
        op.get("f1"),
        op.get("event_f1"),
        agg.get("f1"),
        agg.get("event_f1"),
    )
    recall = _first_num(
        op.get("recall"),
        op.get("event_recall"),
        agg.get("recall"),
        agg.get("event_recall"),
        agg.get("avg_recall"),
    )
    precision = _first_num(
        op.get("precision"),
        op.get("event_precision"),
        agg.get("precision"),
        agg.get("event_precision"),
    )
    fa24h = _first_num(
        op.get("fa24h"),
        op.get("fa_per_day"),
        agg.get("fa24h"),
        agg.get("fa_per_day"),
    )

    return {
        "file": path,
        "model": model,
        "dataset": dataset,
        "aggregate_key": agg_key,
        "selected_op": op_key,
        "ap": ap,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "fa24h": fa24h,
    }


def print_markdown(rows: List[Dict[str, Any]]) -> None:
    print("| Model | Dataset | AP | F1/event_f1 | Recall/event_recall | Precision/event_precision | fa24h | Source |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        src = f"ops.{r['selected_op']} + {r['aggregate_key']} + ap_auc"
        print(
            f"| {r['model'].upper()} | {r['dataset']} | {_fmt(r['ap'])} | {_fmt(r['f1'])} | "
            f"{_fmt(r['recall'])} | {_fmt(r['precision'])} | {_fmt(r['fa24h'])} | {src} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        default=[
            "outputs/metrics/gcn_caucafall.json",
            "outputs/metrics/gcn_le2i.json",
            "outputs/metrics/tcn_caucafall.json",
            "outputs/metrics/tcn_le2i.json",
        ],
    )
    args = ap.parse_args()

    rows = [summarize_one(p) for p in args.files]
    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    print_markdown(rows)


if __name__ == "__main__":
    main()
