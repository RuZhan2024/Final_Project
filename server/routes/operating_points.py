from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import _derive_ops_params_from_yaml, _detect_variants, _ensure_system_settings_schema, _table_exists
from ..db import get_conn

router = APIRouter()


@router.get("/api/operating_points")
@router.get("/api/v1/operating_points")
def operating_points(
    model_code: str = Query(..., description="TCN | GCN | HYBRID"),
    dataset_code: str = Query("muvim", description="Dataset code (optional, for YAML fallback)"),
) -> Dict[str, Any]:
    """Return operating point presets for a model.

    - If DB is available, returns DB-backed rows (v1 or v2 schema).
    - If DB is not available, returns YAML-derived OP-1/OP-2/OP-3 thresholds.
    """
    model_code = (model_code or "").upper().strip()
    if model_code not in {"TCN", "GCN", "HYBRID"}:
        raise HTTPException(status_code=400, detail="model_code must be one of: TCN, GCN, HYBRID")

    dataset_code = (dataset_code or "muvim").lower().strip()

    # Try DB path first
    try:
        with get_conn() as conn:
            _ensure_system_settings_schema(conn)
            variants = _detect_variants(conn)
            with conn.cursor() as cur:
                if variants["ops"] == "v2" and _table_exists(conn, "models"):
                    cur.execute("SELECT id FROM models WHERE code=%s", (model_code,))
                    m = cur.fetchone()
                    if not m:
                        raise HTTPException(status_code=404, detail=f"Unknown model_code: {model_code}")
                    model_id = int(m["id"])

                    cur.execute(
                        """
                        SELECT id, name, code,
                               thr_detect, thr_low_conf, thr_high_conf,
                               est_fa24h, est_recall
                        FROM operating_points
                        WHERE model_id=%s
                        ORDER BY code
                        """,
                        (model_id,),
                    )
                    rows = cur.fetchall() or []
                    ops: List[Dict[str, Any]] = []
                    for r in rows:
                        thr_low = float(r.get("thr_low_conf")) if r.get("thr_low_conf") is not None else None
                        thr_high = float(r.get("thr_high_conf")) if r.get("thr_high_conf") is not None else None
                        ops.append(
                            {
                                "id": int(r["id"]),
                                "name": r.get("name"),
                                "code": r.get("code"),
                                "thr_detect": float(r.get("thr_detect")) if r.get("thr_detect") is not None else None,
                                "thr_low_conf": thr_low,
                                "thr_high_conf": thr_high,
                                # v1-compat
                                "threshold_low": thr_low,
                                "threshold_high": thr_high,
                                "cooldown_seconds": 3,
                                "est_fa24h": float(r["est_fa24h"]) if r.get("est_fa24h") is not None else None,
                                "est_recall": float(r["est_recall"]) if r.get("est_recall") is not None else None,
                            }
                        )
                    return {
                        "model_code": model_code,
                        "dataset_code": dataset_code,
                        "operating_points": ops,
                        "db_available": True,
                    }

                # v1 schema
                cur.execute(
                    """
                    SELECT id, model_code, name, threshold_low, threshold_high, cooldown_seconds, code
                    FROM operating_points
                    WHERE model_code=%s
                    ORDER BY id
                    """,
                    (model_code,),
                )
                rows = cur.fetchall() or []
                ops = []
                for r in rows:
                    thr_low = float(r.get("threshold_low")) if r.get("threshold_low") is not None else None
                    thr_high = float(r.get("threshold_high")) if r.get("threshold_high") is not None else None
                    ops.append(
                        {
                            "id": int(r["id"]),
                            "name": r.get("name"),
                            "code": r.get("code") or None,
                            "thr_detect": None,
                            "thr_low_conf": thr_low,
                            "thr_high_conf": thr_high,
                            "threshold_low": thr_low,
                            "threshold_high": thr_high,
                            "cooldown_seconds": int(r.get("cooldown_seconds") or 3),
                            "est_fa24h": None,
                            "est_recall": None,
                        }
                    )
                return {
                    "model_code": model_code,
                    "dataset_code": dataset_code,
                    "operating_points": ops,
                    "db_available": True,
                }
    except HTTPException:
        raise
    except (MySQLError, RuntimeError, TypeError, ValueError):
        # YAML fallback
        ops = []
        for oc in ["OP-1", "OP-2", "OP-3"]:
            dp = _derive_ops_params_from_yaml(dataset_code=dataset_code, model_code=model_code, op_code=oc)
            ui = dp.get("ui") or {}
            ops.append(
                {
                    "id": None,
                    "name": oc,
                    "code": oc,
                    "thr_detect": None,
                    "thr_low_conf": float(ui.get("tau_low", 0.5)),
                    "thr_high_conf": float(ui.get("tau_high", 0.85)),
                    "threshold_low": float(ui.get("tau_low", 0.5)),
                    "threshold_high": float(ui.get("tau_high", 0.85)),
                    "cooldown_seconds": int(round(float(ui.get("cooldown_s", 3.0)))),
                    "est_fa24h": None,
                    "est_recall": None,
                }
            )

        return {
            "model_code": model_code,
            "dataset_code": dataset_code,
            "operating_points": ops,
            "db_available": False,
        }
