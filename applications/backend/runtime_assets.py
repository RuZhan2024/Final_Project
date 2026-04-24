from __future__ import annotations

"""Helpers for event-clip storage and privacy flags at runtime."""

from typing import Tuple

import numpy as np

from .config import get_app_config

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


def read_clip_privacy_flags(
    conn,
    resident_id: int,
    *,
    ensure_system_settings_schema,
    table_exists,
) -> Tuple[bool, bool]:
    """Return `(store_event_clips, anonymize_skeleton_data)` for one resident."""
    store_event_clips = False
    anonymize = True
    try:
        ensure_system_settings_schema(conn)
        row = None
        with conn.cursor() as cur:
            if table_exists(conn, "system_settings"):
                cur.execute(
                    "SELECT store_event_clips, anonymize_skeleton_data FROM system_settings WHERE resident_id=%s LIMIT 1",
                    (resident_id,),
                )
                row = cur.fetchone()
        if isinstance(row, dict):
            if row.get("store_event_clips") is not None:
                val = row.get("store_event_clips")
                store_event_clips = bool(int(val)) if str(val).isdigit() else bool(val)
            if row.get("anonymize_skeleton_data") is not None:
                val = row.get("anonymize_skeleton_data")
                anonymize = bool(int(val)) if str(val).isdigit() else bool(val)
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return store_event_clips, anonymize


def event_clips_dir():
    """Return the clip-output directory, creating it when possible."""
    d = get_app_config().event_clips_dir
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def anonymize_xy_inplace(xy: np.ndarray) -> np.ndarray:
    """Make coordinates relative to pelvis center (privacy-friendly)."""
    try:
        if xy.ndim != 3 or xy.shape[-1] != 2:
            return xy
        _t, joints, _ = xy.shape
        if joints < 25:
            return xy
        # MediaPipe pelvis approximation uses hip joints 23 and 24.
        pelvis = 0.5 * (xy[:, 23, :] + xy[:, 24, :])
        xy = xy - pelvis[:, None, :]
        return xy
    except (AttributeError, TypeError, ValueError, IndexError):
        return xy
