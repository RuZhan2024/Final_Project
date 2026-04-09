#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v


def _metrics_from_eval(path: Path) -> Dict[str, float]:
    data = _load_json(path)
    totals = data.get("totals", {}) or {}
    ap = _safe_float((data.get("ap_auc") or {}).get("ap"), 0.0)
    recall = _safe_float(totals.get("avg_recall"), 0.0)
    true_alerts = _safe_float(totals.get("n_true_alerts"), 0.0)
    false_alerts = _safe_float(totals.get("n_false_alerts"), 0.0)
    denom = true_alerts + false_alerts
    precision = (true_alerts / denom) if denom > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fa24h = _safe_float(totals.get("fa24h"), 0.0)
    return {
        "ap": ap,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fa24h": fa24h,
    }


def build_summary_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = manifest.get("pairs", []) or []
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        label = str(pair["label"])
        in_domain_path = Path(pair["in_domain"])
        cross_path = Path(pair["cross"])
        in_metrics = _metrics_from_eval(in_domain_path)
        cross_metrics = _metrics_from_eval(cross_path)
        rows.append(
            {
                "label": label,
                "in_ap": in_metrics["ap"],
                "cross_ap": cross_metrics["ap"],
                "delta_ap": cross_metrics["ap"] - in_metrics["ap"],
                "in_f1": in_metrics["f1"],
                "cross_f1": cross_metrics["f1"],
                "delta_f1": cross_metrics["f1"] - in_metrics["f1"],
                "in_recall": in_metrics["recall"],
                "cross_recall": cross_metrics["recall"],
                "delta_recall": cross_metrics["recall"] - in_metrics["recall"],
                "in_fa24h": in_metrics["fa24h"],
                "cross_fa24h": cross_metrics["fa24h"],
                "delta_fa24h": cross_metrics["fa24h"] - in_metrics["fa24h"],
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "in_ap",
        "cross_ap",
        "delta_ap",
        "in_f1",
        "cross_f1",
        "delta_f1",
        "in_recall",
        "cross_recall",
        "delta_recall",
        "in_fa24h",
        "cross_fa24h",
        "delta_fa24h",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-dataset summary CSV from a manifest of eval JSON files")
    ap.add_argument("--manifest", default="artifacts/reports/cross_dataset_manifest.json")
    ap.add_argument("--out_csv", default="artifacts/reports/cross_dataset_summary.csv")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_csv = Path(args.out_csv)
    manifest = _load_json(manifest_path)
    rows = build_summary_rows(manifest)
    write_csv(rows, out_csv)
    print(f"[ok] wrote: {out_csv}")


if __name__ == "__main__":
    main()
