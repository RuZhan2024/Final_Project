#!/usr/bin/env python3
# server/app.py

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List

from models.train_tcn import TCN
from .db import get_conn

# ---------------------------------------------------
# Paths / model checkpoints / metrics files
# ---------------------------------------------------
WIN_W = 48
WIN_S = 12

# Window dirs (for pre-windowed data)
WIN_DIR_LE2I = Path(f"data/processed/le2i/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_URFD = Path(f"data/processed/urfd/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_CAUCA = Path(f"data/processed/caucafall/windows_W{WIN_W}_S{WIN_S}")

# Checkpoints
CKPT_LE2I = Path(f"outputs/le2i_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_URFD = Path(f"outputs/urfd_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_CAUCA = Path(f"outputs/caucafall_tcn_W{WIN_W}S{WIN_S}/best.pt")

# fit-ops yaml (per dataset)
OPS_LE2I = Path("configs/ops_le2i.yaml")
OPS_URFD = Path("configs/ops_urfd.yaml")
OPS_CAUCA = Path("configs/ops_caucafall.yaml")

# metrics reports from your eval commands
REPORT_LE2I = Path("outputs/reports/le2i_in_domain.json")
REPORT_URFD_CROSS = Path("outputs/reports/urfd_cross.json")               # LE2I → URFD
REPORT_CAUCA_IN_DOMAIN = Path("outputs/reports/caucafall_in_domain.json")
REPORT_CAUCA_ON_URFD = Path("outputs/reports/caucafall_on_urfd.json")

DEFAULT_RESIDENT_ID = 1

# ---------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------
app = FastAPI(title="Elder Fall Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Model loading helpers
# ---------------------------------------------------
def load_tcn_from_ckpt(ckpt_path: Path):
    """
    Load a TCN from one of your best.pt files.

    best.pt was saved as:
      {"model": state_dict, "in_ch": C, "best_thr": thr}
    """
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    state_dict = ckpt["model"]
    in_ch = ckpt.get("in_ch", None)
    best_thr = float(ckpt.get("best_thr", 0.5))

    if in_ch is None:
        # Try to infer from first Conv1d weight
        for name, w in state_dict.items():
            if "net.0.weight" in name and w.ndim == 3:
                in_ch = w.shape[1]
                break
        if in_ch is None:
            raise RuntimeError("Could not infer in_ch; set it manually")

    model = TCN(in_ch=in_ch)
    model.load_state_dict(state_dict)
    model.eval()
    return model, best_thr


def load_ops_thr(path: Path, default_thr: float) -> float:
    """
    Load threshold from an ops_*.yaml file.
    Uses OP3_low_alarm.thr if present, otherwise falls back
    to the checkpoint's best_thr.
    """
    if not path.exists():
        return default_thr
    with open(path, "r") as f:
        ops = yaml.safe_load(f)
    if isinstance(ops, dict) and "OP3_low_alarm" in ops:
        op3 = ops["OP3_low_alarm"]
        if isinstance(op3, dict) and "thr" in op3:
            return float(op3["thr"])
    return default_thr


# Load three trained TCN models
model_le2i, thr_ckpt_le2i = load_tcn_from_ckpt(CKPT_LE2I)
model_urfd, thr_ckpt_urfd = load_tcn_from_ckpt(CKPT_URFD)
model_cauca, thr_ckpt_cauca = load_tcn_from_ckpt(CKPT_CAUCA)

# Thresholds (for UI & monitoring)
THR_LE2I = load_ops_thr(OPS_LE2I, thr_ckpt_le2i)
THR_URFD = load_ops_thr(OPS_URFD, thr_ckpt_urfd)
THR_CAUCA = load_ops_thr(OPS_CAUCA, thr_ckpt_cauca)

MODELS_BY_ID = {
    "le2i": model_le2i,
    "urfd": model_urfd,
    "caucafall": model_cauca,
}
THRESHOLDS_BY_MODEL = {
    "le2i": THR_LE2I,
    "urfd": THR_URFD,
    "caucafall": THR_CAUCA,
}
WINDOW_DIR_BY_ID = {
    "le2i": WIN_DIR_LE2I,
    "urfd": WIN_DIR_URFD,
    "caucafall": WIN_DIR_CAUCA,
}

# ---------------------------------------------------
# Metrics report helpers (for models_summary)
# ---------------------------------------------------
def load_report(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def summarise_report(rep):
    """
    Convert the metrics report into a list of operating points with basic metrics.

    Expected format:
      - either: {"ops": {"OP1": {...}, "OP3_low_alarm": {...}}}
      - or: {"OP1": {...}, "OP3_low_alarm": {...}}
    """
    if rep is None or not isinstance(rep, dict):
        return []

    if "ops" in rep and isinstance(rep["ops"], dict):
        ops_dict = rep["ops"]
    else:
        ops_dict = rep

    summary = []
    for name, d in ops_dict.items():
        if not isinstance(d, dict):
            continue
        summary.append(
            {
                "name": name,
                "thr": d.get("thr"),
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "fa24h": d.get("fa24h"),
                "F1": d.get("F1"),
            }
        )
    return summary


# ---------------------------------------------------
# Feature extraction for windowed NPZ (same as training)
# ---------------------------------------------------
def make_window_features(npz_path: Path) -> np.ndarray:
    """
    Re-create the x features exactly like WindowNPZ.__getitem__:
      xy: [W,33,2], conf: [W,33] -> x: [W, 33*2]
    """
    d = np.load(npz_path, allow_pickle=False)
    xy = d["xy"].astype(np.float32)      # [W,33,2]
    conf = d["conf"].astype(np.float32)  # [W,33]

    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    x = xy * conf[..., None]            # [W,33,2]
    x = x.reshape(x.shape[0], -1)       # [W, 33*2] = [T, C]
    return x


def load_sample_windows(root_dir: Path, max_windows: int = 40) -> np.ndarray:
    """
    Returns X with shape [N, T, C] where N = number of windows,
    T = window length (e.g. 48), C = 33*2 = 66.

    Uses the 'test' split of the given dataset.
    """
    root = root_dir / "test"
    npz_files = sorted(root.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz windows found in {root}")

    xs = []
    for f in npz_files[:max_windows]:
        xs.append(make_window_features(f))

    X = np.stack(xs, axis=0)  # [N, T, C]
    return X


def run_sequence(model: TCN, X: np.ndarray, thr: float):
    """
    X: [N, T, C] → list of time steps for the UI:
      [{t: int, p_fall: float, fall: bool}, ...]
    """
    with torch.no_grad():
        inp = torch.from_numpy(X).float()  # [N, T, C]
        logits = model(inp).squeeze(-1)    # [N]
        probs = torch.sigmoid(logits).cpu().numpy()

    points = []
    for i, p in enumerate(probs):
        points.append(
            {
                "t": int(i),
                "p_fall": float(p),
                "fall": bool(p >= thr),
            }
        )
    return points


# ---------------------------------------------------
# Basic health + model summary endpoints
# ---------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models/summary")
def models_summary():
    """
    Summary of all three TCN models, for the Dashboard / Settings pages.
    """
    le2i_rep = summarise_report(load_report(REPORT_LE2I))
    urfd_cross_rep = summarise_report(load_report(REPORT_URFD_CROSS))
    cauca_in_rep = summarise_report(load_report(REPORT_CAUCA_IN_DOMAIN))
    cauca_on_urfd_rep = summarise_report(load_report(REPORT_CAUCA_ON_URFD))

    return {
        "models": [
            {
                "id": "le2i",
                "label": "TCN trained on LE2I",
                "dataset": "LE2I",
                "ckpt": str(CKPT_LE2I),
                "best_thr_from_ckpt": thr_ckpt_le2i,
                "ui_threshold": THR_LE2I,
                "reports": {
                    "in_domain": le2i_rep,
                    "cross": urfd_cross_rep,  # LE2I → URFD
                },
            },
            {
                "id": "urfd",
                "label": "TCN trained on URFD",
                "dataset": "URFD",
                "ckpt": str(CKPT_URFD),
                "best_thr_from_ckpt": thr_ckpt_urfd,
                "ui_threshold": THR_URFD,
                "reports": {
                    "in_domain": [],  # optional: add later
                    "cross": [],
                },
            },
            {
                "id": "caucafall",
                "label": "TCN trained on CAUCAFall",
                "dataset": "CAUCAFall",
                "ckpt": str(CKPT_CAUCA),
                "best_thr_from_ckpt": thr_ckpt_cauca,
                "ui_threshold": THR_CAUCA,
                "reports": {
                    "in_domain": cauca_in_rep,
                    "cross": cauca_on_urfd_rep,  # CAUCA → URFD
                },
            },
        ]
    }


# ---------------------------------------------------
# Monitor: replay of sample window sequences (3 models)
# ---------------------------------------------------
@app.post("/api/demo/{model_id}_fall")
def monitor_sequence(model_id: str):
    """
    Endpoint used by the Monitor page for pre-recorded demo.
    Replays a short pre-recorded sequence through the selected model.

    URL examples (match your existing React code):
      - POST /api/demo/le2i_fall
      - POST /api/demo/urfd_fall
      - POST /api/demo/caucafall_fall
    """
    model_id = model_id.lower()
    if model_id not in MODELS_BY_ID:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    model = MODELS_BY_ID[model_id]
    thr = THRESHOLDS_BY_MODEL[model_id]
    root_dir = WINDOW_DIR_BY_ID[model_id]

    X = load_sample_windows(root_dir, max_windows=40)
    points = run_sequence(model, X, thr)

    return {
        "fps": 5,           # how fast the front-end "plays" the sequence
        "threshold": thr,   # decision threshold used
        "points": points,
    }


# ---------------------------------------------------
# Realtime pose-window inference (frontend → backend)
# ---------------------------------------------------
class PoseWindowPayload(BaseModel):
    """
    JSON shape expected from the frontend for realtime prediction.

    model_id: which trained TCN to use ("le2i" / "urfd" / "caucafall")
    xy:       [T, 33, 2] list → pose keypoints (normalized x,y)
    conf:     [T, 33]    list → confidence/visibility per joint
    """
    model_id: Literal["le2i", "urfd", "caucafall"] = "le2i"
    xy: List[List[List[float]]]
    conf: List[List[float]]


@app.post("/api/monitor/predict_window")
def predict_window(payload: PoseWindowPayload):
    """
    Accept a single pose window from the frontend and run it through
    one of the trained TCN models.

    Example JSON from the frontend:
    {
      "model_id": "le2i",
      "xy":   [[[x,y], ..., [x,y]], ...],  # shape [T,33,2]
      "conf": [[c,...,c], ...]            # shape [T,33]
    }
    """
    model_id = payload.model_id.lower()

    # 1) Choose model + threshold
    if model_id not in MODELS_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown model_id")

    model = MODELS_BY_ID[model_id]
    thr = THRESHOLDS_BY_MODEL[model_id]

    # 2) Convert to numpy
    xy = np.array(payload.xy, dtype=np.float32)      # [T,33,2]
    conf = np.array(payload.conf, dtype=np.float32) # [T,33]

    # 3) Shape checks
    if xy.ndim != 3 or xy.shape[1] != 33 or xy.shape[2] != 2:
        raise HTTPException(
            status_code=400,
            detail=f"xy must have shape [T,33,2], got {xy.shape}",
        )
    if conf.shape != (xy.shape[0], 33):
        raise HTTPException(
            status_code=400,
            detail=f"conf must have shape [T,33], got {conf.shape}",
        )

    # 4) Same preprocessing as training (WindowNPZ)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    x = xy * conf[..., None]           # [T,33,2]
    x = x.reshape(x.shape[0], -1)      # [T, 33*2] = [T,C]

    # Batch of size 1: [1,T,C]
    X = x[None, ...]

    # 5) Run model
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())   # [1,1]
        logit = float(logits.squeeze().item())
        p_fall = float(torch.sigmoid(torch.tensor(logit)).item())

    is_fall = p_fall >= thr

    # (OPTIONAL) You could log an event into DB here if you want.

    return {
        "model_id": model_id,
        "threshold": thr,
        "p_fall": p_fall,
        "fall": is_fall,
    }


# ---------------------------------------------------
# DB-backed endpoints: Dashboard / Events / Settings
# ---------------------------------------------------
@app.get("/api/dashboard/summary")
def dashboard_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    """
    Backend for Dashboard cards:
      - Resident name + status
      - Today: falls detected / false alarms
      - System status: model, monitoring, API health, latency
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resident
            cur.execute(
                "SELECT id, display_name FROM residents WHERE id=%s",
                (resident_id,),
            )
            resident = cur.fetchone()
            if resident is None:
                raise HTTPException(status_code=404, detail="Resident not found")

            # System settings
            cur.execute(
                """
                SELECT monitoring_enabled, api_online, last_latency_ms, active_model_id
                FROM system_settings
                WHERE resident_id = %s
                """,
                (resident_id,),
            )
            settings = cur.fetchone() or {}

            active_model_name = None
            if settings.get("active_model_id"):
                cur.execute(
                    "SELECT name FROM models WHERE id=%s",
                    (settings["active_model_id"],),
                )
                m = cur.fetchone()
                if m:
                    active_model_name = m["name"]

            # Today's events
            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN type='fall'
                         AND status='confirmed_fall' THEN 1 ELSE 0 END) AS falls,
                  SUM(CASE WHEN status='false_alarm' THEN 1 ELSE 0 END) AS false_alarms,
                  SUM(CASE WHEN status IN ('pending_review','confirmed_fall')
                           THEN 1 ELSE 0 END) AS alerts
                FROM events
                WHERE resident_id = %s
                  AND DATE(event_time) = CURDATE()
                """,
                (resident_id,),
            )
            stats = cur.fetchone() or {}
            falls_today = stats.get("falls") or 0
            false_alarms_today = stats.get("false_alarms") or 0
            alerts_today = stats.get("alerts") or 0

    status = "alert" if alerts_today > 0 else "normal"

    return {
        "resident": {
            "id": resident["id"],
            "name": resident["display_name"],
        },
        "status": status,
        "today": {
            "falls_detected": int(falls_today),
            "false_alarms": int(false_alarms_today),
        },
        "system": {
            "model_name": active_model_name,
            "monitoring_enabled": bool(settings.get("monitoring_enabled", 1)),
            "api_online": bool(settings.get("api_online", 1)),
            "last_latency_ms": settings.get("last_latency_ms"),
        },
    }


@app.get("/api/events")
def list_events(
    resident_id: int = DEFAULT_RESIDENT_ID,
    limit: int = Query(50, ge=1, le=500),
):
    """
    Event History table (last N events).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  e.id,
                  e.event_time,
                  e.type,
                  e.p_fall,
                  e.delay_seconds,
                  e.status,
                  m.name AS model_name,
                  m.code AS model_code
                FROM events e
                JOIN models m ON e.model_id = m.id
                WHERE e.resident_id = %s
                ORDER BY e.event_time DESC
                LIMIT %s
                """,
                (resident_id, limit),
            )
            rows = cur.fetchall() or []

    events = []
    for r in rows:
        events.append(
            {
                "id": r["id"],
                "time": r["event_time"].isoformat()
                if isinstance(r["event_time"], datetime)
                else r["event_time"],
                "type": r["type"],
                "model_name": r["model_name"],
                "model_code": r["model_code"],
                "p_fall": float(r["p_fall"]) if r["p_fall"] is not None else None,
                "delay_seconds": float(r["delay_seconds"])
                if r["delay_seconds"] is not None
                else None,
                "status": r["status"],
            }
        )

    return {"events": events}


@app.get("/api/events/summary")
def events_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    """
    Summary cards for the Events page:
      - Falls today
      - False alarms today
      - FA/24h estimate
      - Avg detection delay today
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN type='fall'
                           AND status='confirmed_fall' THEN 1 ELSE 0 END) AS falls_today,
                  SUM(CASE WHEN status='false_alarm'
                           THEN 1 ELSE 0 END) AS false_alarms_today,
                  AVG(delay_seconds) AS avg_delay
                FROM events
                WHERE resident_id = %s
                  AND DATE(event_time) = CURDATE()
                """,
                (resident_id,),
            )
            row = cur.fetchone() or {}

    falls_today = row.get("falls_today") or 0
    false_alarms_today = row.get("false_alarms_today") or 0
    avg_delay = row.get("avg_delay")

    fa24h_estimate = float(false_alarms_today)

    return {
        "falls_today": int(falls_today),
        "false_alarms_today": int(false_alarms_today),
        "fa24h_estimate": fa24h_estimate,
        "avg_detection_delay": float(avg_delay) if avg_delay is not None else None,
    }


@app.get("/api/settings")
def get_settings(resident_id: int = DEFAULT_RESIDENT_ID):
    """
    Settings page data:
      - Resident info
      - Primary caregiver
      - Notification flags
      - Active model + operating point
      - Privacy flags
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  r.id AS resident_id,
                  r.display_name,
                  s.monitoring_enabled,
                  s.notify_on_every_fall,
                  s.require_confirmation,
                  s.store_event_clips,
                  s.anonymize_skeleton_data,
                  s.active_model_id,
                  s.active_operating_point,
                  m.name AS model_name,
                  op.name AS op_name
                FROM residents r
                LEFT JOIN system_settings s ON s.resident_id = r.id
                LEFT JOIN models m ON s.active_model_id = m.id
                LEFT JOIN operating_points op ON s.active_operating_point = op.id
                WHERE r.id = %s
                """,
                (resident_id,),
            )
            main = cur.fetchone()
            if not main:
                raise HTTPException(status_code=404, detail="Resident not found")

            # primary caregiver
            cur.execute(
                """
                SELECT id, name, email, phone
                FROM caregivers
                WHERE resident_id = %s AND is_primary = 1
                ORDER BY id ASC
                LIMIT 1
                """,
                (resident_id,),
            )
            cg = cur.fetchone()

    return {
        "resident": {
            "id": main["resident_id"],
            "name": main["display_name"],
        },
        "caregiver": {
            "id": cg["id"] if cg else None,
            "name": cg["name"] if cg else None,
            "email": cg["email"] if cg else None,
            "phone": cg["phone"] if cg else None,
        },
        "notifications": {
            "notify_on_every_fall": bool(main.get("notify_on_every_fall", 1)),
            "require_confirmation": bool(main.get("require_confirmation", 0)),
        },
        "model": {
            "active_model_id": main.get("active_model_id"),
            "active_model_name": main.get("model_name"),
            "active_operating_point_id": main.get("active_operating_point"),
            "active_operating_point_name": main.get("op_name"),
        },
        "privacy": {
            "store_event_clips": bool(main.get("store_event_clips", 0)),
            "anonymize_skeleton_data": bool(main.get("anonymize_skeleton_data", 1)),
        },
        "monitoring_enabled": bool(main.get("monitoring_enabled", 1)),
    }
