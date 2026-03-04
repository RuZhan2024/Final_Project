from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from fastapi import APIRouter
import yaml

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import _jsonable
from ..db import get_conn
from ..deploy_runtime import get_specs as _get_deploy_specs


router = APIRouter()


def _load_deploy_modes_yaml() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "deploy_modes.yaml"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}


@router.get("/api/models/summary")
@router.get("/api/v1/models/summary")
def models_summary() -> Dict[str, Any]:
    """Return deployable model specs.

    The frontend needs dataset-aware models (dataset × arch).
    We also return DB-backed `models` rows for backwards compatibility.
    """
    specs = _get_deploy_specs()
    models = []
    for key, s in specs.items():
        fps_default = None
        try:
            fps_default = float((s.data_cfg or {}).get("fps_default")) if getattr(s, "data_cfg", None) else None
        except (TypeError, ValueError):
            fps_default = None

        op2 = (s.ops or {}).get("OP-2") or (s.ops or {}).get("op2") or {}
        models.append(
            {
                "id": str(key),
                "spec_key": str(key),
                "dataset_code": str(getattr(s, "dataset", "")),
                "arch": str(getattr(s, "arch", "")),
                "name": f"{getattr(s, 'dataset', '')} {getattr(s, 'arch', '')}".strip(),
                "fps_default": fps_default,
                "ckpt": str(getattr(s, "ckpt", "")),
                "ops": (s.ops or {}),
                "alert_cfg": (getattr(s, "alert_cfg", None) or {}),
                "tau_low": op2.get("tau_low"),
                "tau_high": op2.get("tau_high"),
            }
        )

    models.sort(key=lambda d: (d.get("dataset_code") or "", d.get("arch") or ""))

    db_rows = []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM models ORDER BY id ASC")
                db_rows = cur.fetchall() or []
    except (MySQLError, RuntimeError, ValueError, TypeError):
        db_rows = []

    return {"models": models, "db_models": _jsonable(db_rows)}


@router.get("/api/deploy/specs")
@router.get("/api/v1/deploy/specs")
def deploy_specs() -> Dict[str, Any]:
    """Return dataset-specific specs discovered from configs/ops/*.yaml."""
    specs = _get_deploy_specs()
    out = []
    datasets = set()
    models = []

    for key, s in specs.items():
        ds = getattr(s, "dataset", "")
        datasets.add(ds)

        out.append(
            {
                "spec_key": key,
                "dataset_code": ds,
                "arch": s.arch,
                "ckpt": str(s.ckpt),
                "ops": s.ops,
            }
        )

        models.append(
            {
                "key": key,
                "spec_key": key,
                "dataset_code": ds,
                "arch": s.arch,
                "ckpt": str(s.ckpt),
                "alert_cfg": dict(getattr(s, "alert_cfg", {}) or {}),
                "ops": s.ops,
            }
        )

    out.sort(key=lambda d: (d.get("dataset_code") or "", d.get("arch") or ""))
    models.sort(key=lambda d: (d.get("dataset_code") or "", d.get("arch") or ""))
    return {"specs": out, "models": models, "datasets": sorted(datasets)}


@router.get("/api/spec")
@router.get("/api/v1/spec")
def api_spec() -> Dict[str, Any]:
    """Alias for /api/deploy/specs (legacy UI compatibility)."""
    return deploy_specs()


@router.get("/api/deploy/modes")
@router.get("/api/v1/deploy/modes")
def deploy_modes() -> Dict[str, Any]:
    """Expose deploy mode profile config from configs/deploy_modes.yaml."""
    return {"deploy_modes": _load_deploy_modes_yaml()}
