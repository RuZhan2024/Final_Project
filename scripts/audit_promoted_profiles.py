#!/usr/bin/env python3
"""Audit promoted profile metrics and unlabeled FA summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[err] missing json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_check(spec: str) -> dict[str, Any]:
    parts = [x.strip() for x in str(spec).split("|")]
    if len(parts) != 8:
        raise SystemExit(
            "[err] invalid --check format. expected: "
            "name|metrics_json|unlabeled_json|min_recall|max_test_false|max_fa24h|max_fa_per_day|max_unlabeled_events"
        )
    return {
        "name": parts[0],
        "metrics_json": parts[1],
        "unlabeled_json": parts[2],
        "min_recall": _to_float(parts[3]),
        "max_test_false": _to_int(parts[4]),
        "max_fa24h": _to_float(parts[5]),
        "max_fa_per_day": _to_float(parts[6]),
        "max_unlabeled_events": _to_int(parts[7]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="append",
        default=[],
        help=(
            "name|metrics_json|unlabeled_json|min_recall|max_test_false|max_fa24h|max_fa_per_day|max_unlabeled_events"
        ),
    )
    ap.add_argument("--out_json", default="artifacts/reports/promoted_profiles_latest.json")
    args = ap.parse_args()

    if not args.check:
        raise SystemExit("[err] provide at least one --check")

    checks = [_parse_check(x) for x in args.check]
    report: dict[str, Any] = {"status": "pass", "checks": {}, "failures": []}

    for c in checks:
        m = _load_json(c["metrics_json"])
        u = _load_json(c["unlabeled_json"])

        mt = m.get("totals", {}) if isinstance(m.get("totals"), dict) else {}
        ut = u.get("total", {}) if isinstance(u.get("total"), dict) else {}

        recall = _to_float(mt.get("avg_recall"))
        test_false = _to_int(mt.get("n_false_alerts"))
        fa24h = _to_float(mt.get("fa24h"))
        unlabeled_fa_day = _to_float(ut.get("fa_per_day"))
        unlabeled_events = _to_int(ut.get("n_alert_events"))

        entry = {
            "metrics_json": c["metrics_json"],
            "unlabeled_json": c["unlabeled_json"],
            "observed": {
                "avg_recall": recall,
                "n_false_alerts": test_false,
                "fa24h": fa24h,
                "unlabeled_fa_per_day": unlabeled_fa_day,
                "unlabeled_n_alert_events": unlabeled_events,
            },
            "limits": {
                "min_recall": c["min_recall"],
                "max_test_false": c["max_test_false"],
                "max_fa24h": c["max_fa24h"],
                "max_fa_per_day": c["max_fa_per_day"],
                "max_unlabeled_events": c["max_unlabeled_events"],
            },
        }
        report["checks"][c["name"]] = entry

        # Min recall gate
        if not math.isfinite(recall) or recall < float(c["min_recall"]):
            report["status"] = "fail"
            report["failures"].append(
                f"{c['name']}: recall {recall} < min_recall {c['min_recall']}"
            )

        # Test false count gate
        if test_false < 0 or test_false > int(c["max_test_false"]):
            report["status"] = "fail"
            report["failures"].append(
                f"{c['name']}: n_false_alerts {test_false} > {c['max_test_false']}"
            )

        # FA/24h gate
        if not math.isfinite(fa24h) or fa24h > float(c["max_fa24h"]):
            report["status"] = "fail"
            report["failures"].append(
                f"{c['name']}: fa24h {fa24h} > {c['max_fa24h']}"
            )

        # Unlabeled FA/day gate
        if not math.isfinite(unlabeled_fa_day) or unlabeled_fa_day > float(c["max_fa_per_day"]):
            report["status"] = "fail"
            report["failures"].append(
                f"{c['name']}: unlabeled fa/day {unlabeled_fa_day} > {c['max_fa_per_day']}"
            )

        # Unlabeled event count gate
        if unlabeled_events < 0 or unlabeled_events > int(c["max_unlabeled_events"]):
            report["status"] = "fail"
            report["failures"].append(
                f"{c['name']}: unlabeled n_alert_events {unlabeled_events} > {c['max_unlabeled_events']}"
            )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] promoted profiles report: {out}")

    if report["status"] != "pass":
        print("[fail] promoted profile audit failed")
        for f in report["failures"]:
            print(f" - {f}")
        raise SystemExit(1)
    print("[ok] promoted profile audit passed")


if __name__ == "__main__":
    main()

