from __future__ import annotations

"""Routes exposing deploy-time model specs, replay clips, and mode config.

These endpoints are read-only views over deploy assets on disk plus the active
spec registry. The route layer keeps file-path safety and legacy aliases local
while leaving spec loading to runtime helpers.
"""

import os
from typing import Any, Dict
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import yaml

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..db import get_conn
from ..deploy_runtime import get_specs as _get_deploy_specs
from ..json_utils import jsonable as _jsonable


router = APIRouter()
_REPLAY_CLIP_EXTS = {".mp4", ".mov", ".webm", ".m4v"}


def _replay_clips_root() -> Path:
    """Resolve the replay clip directory with env override and legacy fallback."""
    raw = os.getenv("REPLAY_CLIPS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    root = Path(__file__).resolve().parents[3]
    canonical = (root / "ops" / "deploy_assets" / "replay_clips").resolve()
    legacy = (root / "data" / "replay_clips").resolve()
    return canonical if canonical.exists() else legacy


def _is_within_root(path: Path, root: Path) -> bool:
    """Reject path traversal by requiring clip paths to stay under the replay root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _list_replay_clips() -> list[Dict[str, Any]]:
    """List replay clips and derive a coarse category from directory/name hints."""
    root = _replay_clips_root()
    if not root.exists() or not root.is_dir():
        return []

    clips: list[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _REPLAY_CLIP_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        rel_l = rel.lower()
        parts_l = [p.lower() for p in path.relative_to(root).parts]
        if any("fall" in p for p in parts_l):
            category = "fall"
        elif any("adl" in p for p in parts_l):
            category = "adl"
        elif any(tok in rel_l for tok in ("nonfall", "non-fall", "non_fall", "normal", "safe")):
            category = "adl"
        elif "fall" in rel_l:
            category = "fall"
        else:
            category = "other"
        clips.append(
            {
                "id": rel,
                "name": path.name,
                "filename": path.name,
                "path": rel,
                "category": category,
                "size_bytes": path.stat().st_size,
                "url": f"/api/replay/clips/{quote(rel)}",
            }
        )
    return clips


def _load_deploy_modes_yaml() -> Dict[str, Any]:
    """Load deploy mode presets from the canonical or legacy config path."""
    root = Path(__file__).resolve().parents[3]
    canonical = root / "ops" / "configs" / "deploy_modes.yaml"
    legacy = root / "configs" / "deploy_modes.yaml"
    path = canonical if canonical.exists() else legacy
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
                # DB-backed rows are kept only for older admin/debug clients;
                # current monitor setup reads the spec-derived `models` array.
                cur.execute("SELECT * FROM models ORDER BY id ASC")
                db_rows = cur.fetchall() or []
    except (MySQLError, RuntimeError, ValueError, TypeError):
        db_rows = []

    return {"models": models, "db_models": _jsonable(db_rows)}


@router.get("/api/deploy/specs")
@router.get("/api/v1/deploy/specs")
def deploy_specs() -> Dict[str, Any]:
    """Return dataset-specific specs discovered from ops/configs/ops/*.yaml."""
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


@router.get("/api/replay/clips")
@router.get("/api/v1/replay/clips")
def replay_clips() -> Dict[str, Any]:
    """Return replay clip metadata plus directory availability diagnostics."""
    root = _replay_clips_root()
    return {
        "clips": _list_replay_clips(),
        "configured_dir": str(root),
        "available": bool(root.exists() and root.is_dir()),
    }


@router.get("/api/replay/clips/{clip_path:path}")
@router.get("/api/v1/replay/clips/{clip_path:path}")
def replay_clip_file(clip_path: str):
    """Serve one replay clip file from the configured replay root."""
    root = _replay_clips_root()
    path = (root / clip_path).resolve()
    if not _is_within_root(path, root):
        raise HTTPException(status_code=404, detail="clip_not_found")
    if not path.exists() or not path.is_file() or path.suffix.lower() not in _REPLAY_CLIP_EXTS:
        raise HTTPException(status_code=404, detail="clip_not_found")
    return FileResponse(path)


@router.get("/api/deploy/modes")
@router.get("/api/v1/deploy/modes")
def deploy_modes() -> Dict[str, Any]:
    """Expose deploy mode profile config from ops/configs/deploy_modes.yaml."""
    return {"deploy_modes": _load_deploy_modes_yaml()}
