#!/usr/bin/env python3
"""Summarize dual-policy alert fields from persisted events.meta.

Reads recent events from DB and reports:
- count of events carrying safe/recall fields
- safe vs recall fall-state counts
- disagreement counts (safe != recall)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

# Path bootstrap so `from server.db import ...` works when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.db import get_conn_optional


def _to_meta(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            obj = json.loads(v)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident_id", type=int, default=1)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument(
        "--out_json",
        default="artifacts/reports/deployment_dual_policy_events.json",
    )
    args = ap.parse_args()

    rows = []
    with get_conn_optional() as conn:
        if conn is None:
            raise SystemExit("[ERR] DB unavailable. Check DB_* env vars and MySQL.")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, type, model_code, meta
                FROM events
                WHERE resident_id=%s
                  AND ts >= (NOW() - INTERVAL %s HOUR)
                ORDER BY ts DESC
                LIMIT %s
                """,
                (int(args.resident_id), int(args.hours), int(args.limit)),
            )
            rows = cur.fetchall() or []

    c = Counter()
    type_counts = Counter()
    model_counts = Counter()

    for r in rows:
        type_counts[str(r.get("type") or "unknown")] += 1
        model_counts[str(r.get("model_code") or "unknown")] += 1
        meta = _to_meta(r.get("meta"))
        if not meta:
            c["missing_meta"] += 1
            continue

        has_dual = (
            ("safe_alert" in meta)
            or ("recall_alert" in meta)
            or ("policy_alerts" in meta and isinstance(meta.get("policy_alerts"), dict))
        )
        if not has_dual:
            c["legacy_no_dual_fields"] += 1
            continue
        c["dual_rows"] += 1

        safe = _as_bool(meta.get("safe_alert"))
        recall = _as_bool(meta.get("recall_alert"))
        if safe is None and isinstance(meta.get("policy_alerts"), dict):
            safe = _as_bool((meta.get("policy_alerts") or {}).get("safe", {}).get("alert"))
        if recall is None and isinstance(meta.get("policy_alerts"), dict):
            recall = _as_bool((meta.get("policy_alerts") or {}).get("recall", {}).get("alert"))

        if safe is True:
            c["safe_fall"] += 1
        elif safe is False:
            c["safe_not_fall"] += 1
        else:
            c["safe_unknown"] += 1

        if recall is True:
            c["recall_fall"] += 1
        elif recall is False:
            c["recall_not_fall"] += 1
        else:
            c["recall_unknown"] += 1

        if (safe is not None) and (recall is not None):
            if safe == recall:
                c["agreement"] += 1
            else:
                c["disagreement"] += 1
                if safe is False and recall is True:
                    c["disagree_recall_only"] += 1
                elif safe is True and recall is False:
                    c["disagree_safe_only"] += 1

    total = len(rows)
    report = {
        "schema_version": "1.0",
        "resident_id": int(args.resident_id),
        "window_hours": int(args.hours),
        "rows_scanned": total,
        "type_counts": dict(type_counts),
        "model_counts": dict(model_counts),
        "dual_policy_counts": dict(c),
        "ratios": {
            "dual_rows_ratio": (c["dual_rows"] / total) if total else 0.0,
            "disagreement_ratio_on_dual_rows": (c["disagreement"] / c["dual_rows"]) if c["dual_rows"] else 0.0,
            "recall_only_ratio_on_disagreements": (
                c["disagree_recall_only"] / c["disagreement"]
            )
            if c["disagreement"]
            else 0.0,
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out}")
    print(json.dumps(report["dual_policy_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
