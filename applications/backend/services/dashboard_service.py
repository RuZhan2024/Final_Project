from __future__ import annotations

"""Dashboard summary assembly helpers.

This service translates repository-level snapshots into the stable dashboard API
shape. It owns resident fallback, status derivation, and compatibility across
schema variants; the route layer only handles transport and degraded DB cases.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..repositories.dashboard_repository import load_last_latency_ms, load_system_snapshot, load_today_counts


@dataclass(frozen=True)
class DashboardDeps:
    """Repository hooks required to build a dashboard summary across schema variants."""
    resident_exists: Callable[[Any, int], bool]
    one_resident_id: Callable[[Any], int]
    table_exists: Callable[[Any, str], bool]
    col_exists: Callable[[Any, str, str], bool]


def base_dashboard_summary(last_pred_latency_ms: int) -> Dict[str, Any]:
    """Return the shape the dashboard expects even before DB data is merged in."""
    return {
        "status": "normal",
        "today": {"falls_detected": 0, "false_alarms": 0, "confirmed_falls": 0},
        "system": {
            "model_name": "TCN",
            "monitoring_enabled": False,
            "last_latency_ms": int(last_pred_latency_ms or 0),
            "api_online": True,
        },
    }


def resolve_resident_id(conn: Any, resident_id: Optional[int], deps: DashboardDeps) -> int:
    """Prefer an explicit resident only when it exists; otherwise fall back safely."""
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
    """Assemble the dashboard response from base defaults plus repository snapshots.

    The returned payload always starts from a complete fallback shape, then
    layers in schema-aware system/today snapshots so partial DB feature support
    does not break the frontend contract.
    """
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
    # Dashboard status is intentionally derived from today's confirmed count path
    # rather than raw model state so the overview matches persisted review data.
    summary["status"] = "alert" if int(counts.get("falls_detected") or 0) > 0 else "normal"

    latency_ms = load_last_latency_ms(
        conn,
        table_exists=deps.table_exists,
        col_exists=deps.col_exists,
    )
    if latency_ms is not None:
        summary["system"]["last_latency_ms"] = latency_ms

    return summary
