#!/usr/bin/env python3
"""Summarize deployment field validation observations into report JSON files.

Input CSV columns (required):
- clip_id
- has_event (0/1)
- detected_event (0/1)
- false_alert_count (int)
- delay_s (float, optional when no true event detection)
- status (e.g., ok/fail)
- failure_type (string, optional)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median


def _to_int(v: str, default: int = 0) -> int:
    try:
        return int(float((v or "").strip()))
    except Exception:
        return default


def _to_float(v: str) -> float | None:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_csv", required=True, help="Per-clip observations CSV")
    ap.add_argument("--hours", type=float, default=1.0, help="Total monitored hours for FA/day estimate")
    ap.add_argument("--out_eval_json", default="artifacts/reports/deployment_field_eval.json")
    ap.add_argument("--out_failures_json", default="artifacts/reports/deployment_field_failures.json")
    ap.add_argument(
        "--dual_policy_json",
        default="",
        help="Optional dual-policy summary JSON (from tools/summarize_dual_policy_events.py).",
    )
    ap.add_argument(
        "--out_markdown",
        default="artifacts/reports/deployment_field_validation_summary.md",
        help="Optional markdown summary output path.",
    )
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    obs_path = Path(args.obs_csv)
    if not obs_path.exists():
        raise SystemExit(
            "[ERR] observations CSV not found: "
            f"{obs_path}. "
            "Create it from template: "
            "artifacts/templates/deployment_field_observations_template.csv"
        )
    with open(obs_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("[ERR] empty observations CSV")

    total = len(rows)
    gt_events = 0
    detected_events = 0
    true_positives = 0
    false_alerts_total = 0
    delay_vals: list[float] = []
    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()

    for r in rows:
        has_event = _to_int(r.get("has_event", "0"), 0)
        detected = _to_int(r.get("detected_event", "0"), 0)
        fa = _to_int(r.get("false_alert_count", "0"), 0)
        delay = _to_float(r.get("delay_s", ""))

        gt_events += 1 if has_event == 1 else 0
        detected_events += 1 if detected == 1 else 0
        true_positives += 1 if (has_event == 1 and detected == 1) else 0
        false_alerts_total += max(0, fa)

        if has_event == 1 and detected == 1 and delay is not None:
            delay_vals.append(float(delay))

        st = (r.get("status") or "").strip() or "unknown"
        ft = (r.get("failure_type") or "").strip()
        status_counts[st] += 1
        if ft:
            failure_counts[ft] += 1

    recall = (true_positives / gt_events) if gt_events > 0 else 0.0
    precision_proxy = (true_positives / detected_events) if detected_events > 0 else 0.0
    fpr_per_hour = (false_alerts_total / args.hours) if args.hours > 0 else 0.0
    fa24h = fpr_per_hour * 24.0

    eval_report = {
        "schema_version": "1.0",
        "source_csv": str(args.obs_csv),
        "n_clips": total,
        "n_gt_event_clips": gt_events,
        "n_detected_event_clips": detected_events,
        "n_true_positive_clips": true_positives,
        "metrics": {
            "event_recall_proxy": recall,
            "event_precision_proxy": precision_proxy,
            "false_alerts_total": false_alerts_total,
            "false_alerts_per_hour": fpr_per_hour,
            "fa24h_estimate": fa24h,
            "delay_p50_s": median(delay_vals) if delay_vals else 0.0,
            "delay_p95_s": _pct(delay_vals, 95),
        },
        "status_counts": dict(status_counts),
    }

    fail_report = {
        "schema_version": "1.0",
        "source_csv": str(args.obs_csv),
        "failure_type_counts": dict(failure_counts),
        "status_counts": dict(status_counts),
        "n_fail_rows": int(sum(v for k, v in status_counts.items() if k.lower() in {"fail", "error"})),
    }

    dual_report = None
    if args.dual_policy_json:
        p = Path(args.dual_policy_json)
        if p.exists():
            try:
                dual_report = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                dual_report = None
    if isinstance(dual_report, dict):
        eval_report["dual_policy"] = {
            "source_json": str(args.dual_policy_json),
            "window_hours": dual_report.get("window_hours"),
            "rows_scanned": dual_report.get("rows_scanned"),
            "dual_policy_counts": dual_report.get("dual_policy_counts", {}),
            "ratios": dual_report.get("ratios", {}),
        }

    out_eval = Path(args.out_eval_json)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_fail = Path(args.out_failures_json)
    out_fail.parent.mkdir(parents=True, exist_ok=True)
    out_fail.write_text(json.dumps(fail_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_md = Path(args.out_markdown)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Deployment Field Validation Summary",
        "",
        "## Core Metrics",
        "",
        f"- `event_recall_proxy`: {eval_report['metrics']['event_recall_proxy']:.4f}",
        f"- `event_precision_proxy`: {eval_report['metrics']['event_precision_proxy']:.4f}",
        f"- `fa24h_estimate`: {eval_report['metrics']['fa24h_estimate']:.4f}",
        f"- `delay_p50_s`: {eval_report['metrics']['delay_p50_s']:.4f}",
        f"- `delay_p95_s`: {eval_report['metrics']['delay_p95_s']:.4f}",
        "",
        "## Failure Summary",
        "",
        f"- `status_counts`: {json.dumps(eval_report.get('status_counts', {}), ensure_ascii=False)}",
        f"- `failure_type_counts`: {json.dumps(fail_report.get('failure_type_counts', {}), ensure_ascii=False)}",
    ]
    if isinstance(dual_report, dict):
        lines += [
            "",
            "## Dual Policy Runtime Summary",
            "",
            f"- source: `{args.dual_policy_json}`",
            f"- `dual_policy_counts`: {json.dumps(dual_report.get('dual_policy_counts', {}), ensure_ascii=False)}",
            f"- `ratios`: {json.dumps(dual_report.get('ratios', {}), ensure_ascii=False)}",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote eval report: {out_eval}")
    print(f"[ok] wrote failure report: {out_fail}")
    print(f"[ok] wrote markdown summary: {out_md}")


if __name__ == "__main__":
    main()
