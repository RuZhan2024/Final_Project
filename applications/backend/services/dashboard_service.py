from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..repositories.dashboard_repository import load_last_latency_ms, load_system_snapshot, load_today_counts


@dataclass(frozen=True)
class DashboardDeps:
    resident_exists: Callable[[Any, int], bool]
    one_resident_id: Callable[[Any], int]
    table_exists: Callable[[Any, str], bool]
    col_exists: Callable[[Any, str, str], bool]


def base_dashboard_summary(last_pred_latency_ms: int) -> Dict[str, Any]:
    return {
        "status": "normal",
        "today": {"falls_detected": 0, "false_alarms": 0},
        "system": {
            "model_name": "TCN",
            "monitoring_enabled": False,
            "last_latency_ms": int(last_pred_latency_ms or 0),
            "api_online": True,
        },
    }


def resolve_resident_id(conn: Any, resident_id: Optional[int], deps: DashboardDeps) -> int:
    if resident_id and deps.resident_exists(conn, int(resident_id)):
        return int(resident_id)
    return int(deps.one_resident_id(conn))


def build_dashboard_summary_response(
    conn: Any,
    *,
    resident_id: Optional[int],
    deps: DashboardDeps,
    last_pred_latency_ms: int,
) -> Dict[str, Any]:
    summary = base_dashboard_summary(last_pred_latency_ms)
    rid = resolve_resident_id(conn, resident_id, deps)

    system_snapshot = load_system_snapshot(
        conn,
        rid,
        table_exists=deps.table_exists,
        col_exists=deps.col_exists,
    )
    summary["system"].update(system_snapshot)

    counts = load_today_counts(
        conn,
        rid,
        table_exists=deps.table_exists,
        col_exists=deps.col_exists,
    )
    summary["today"].update(counts)
    summary["status"] = "alert" if int(counts.get("falls_detected") or 0) > 0 else "normal"

    latency_ms = load_last_latency_ms(
        conn,
        table_exists=deps.table_exists,
        col_exists=deps.col_exists,
    )
    if latency_ms is not None:
        summary["system"]["last_latency_ms"] = latency_ms

    return summary
