#!/usr/bin/env python3
# server/app.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.db import get_conn

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, build_gcn_input, build_tcn_input
from core.models import build_model, pick_device, p_fall_from_logits
from core.uncertainty import mc_predict_mu_sigma
from core.alerting import (
    TriageCfg,
    SingleModeCfg,
    DualModeCfg,
    SingleTriageStateMachine,
    DualTriageStateMachine,
    triage_state,
    TRIAGE_NOT_FALL,
    TRIAGE_UNCERTAIN,
    TRIAGE_FALL,
    EVENT_POSSIBLE,
    EVENT_CONFIRMED,
)


# ===================================================
# Defaults (project-wide)
# ===================================================
WIN_W = 48
WIN_S = 12

DEFAULT_RESIDENT_ID = 1

# Resolve paths relative to the project root (so `uvicorn server.app:app` works
# even if the working directory is not the repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_CFG_PATH = PROJECT_ROOT / "configs" / "deploy_modes.yaml"


# ===================================================
# FastAPI app + CORS
# ===================================================
app = FastAPI(title="Elder Fall Monitoring API")

# Optional: serve a tiny static demo UI if present (does not affect your React app).
_STATIC_DIR = PROJECT_ROOT / "server" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup_migrate_db() -> None:
    # Best-effort migration; no-op if DB is not configured.
    _db_ensure_event_metadata_column()


# ===================================================
# Device
# ===================================================
DEVICE = pick_device()


# ===================================================
# Model registry
# ===================================================
class ModelSpec(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    label: str
    dataset: str
    arch: Literal["tcn", "gcn"]
    ckpt: str
    ops: str = ""  # optional ops YAML (for default tau_high)
    fps_default: float = 30.0


@dataclass
class ModelRunner:
    spec: ModelSpec
    model: torch.nn.Module
    arch: str
    feat_cfg: FeatCfg
    model_cfg: Dict[str, Any]
    fps_default: float
    two_stream: bool

    # derived thresholds
    tau_low: float
    tau_high: float
    ema_alpha: float
    sigma_max: Optional[float]

    def _build_inputs(self, xy: np.ndarray, conf: np.ndarray, fps: float) -> Tuple[Tuple[torch.Tensor, ...], float]:
        """Build torch inputs for this model.

        Returns (inputs_tuple, fps_used).
        """
        fps_used = float(fps or self.fps_default)
        mask = (conf >= float(self.feat_cfg.conf_gate)).astype(np.float32)
        # motion is derived inside build_* if feat_cfg.use_motion is true
        motion = None

        if self.arch == "tcn":
            Xt, _ = build_tcn_input(xy, motion, conf, mask, fps_used, self.feat_cfg)  # [T,C]
            x = torch.from_numpy(Xt[None, ...]).float()  # [1,T,C]
            return (x,), fps_used

        # gcn: use build_gcn_input then split if two_stream
        Xg, _ = build_gcn_input(xy, motion, conf, mask, fps_used, self.feat_cfg)  # [T,V,F]
        if self.two_stream:
            xy2 = Xg[..., 0:2]
            conf1 = Xg[..., -1:] if self.feat_cfg.use_conf_channel else None
            xj = np.concatenate([xy2, conf1], axis=-1) if conf1 is not None else xy2
            if self.feat_cfg.use_motion:
                xm = Xg[..., 2:4]
            else:
                xm = np.zeros_like(xy2, dtype=np.float32)
            tj = torch.from_numpy(xj[None, ...]).float()
            tm = torch.from_numpy(xm[None, ...]).float()
            return (tj, tm), fps_used

        x = torch.from_numpy(Xg[None, ...]).float()
        return (x,), fps_used

    @torch.no_grad()
    def predict_p(self, xy: np.ndarray, conf: np.ndarray, fps: float) -> Tuple[float, float]:
        """Deterministic probability."""
        inputs, fps_used = self._build_inputs(xy, conf, fps)
        inputs = tuple(t.to(DEVICE) for t in inputs)
        logits = self.model(*inputs)
        p = float(p_fall_from_logits(logits)[0].detach().cpu().item())
        return p, fps_used

    @torch.no_grad()
    def predict_mu_sigma(self, xy: np.ndarray, conf: np.ndarray, fps: float, M: int) -> Tuple[float, float, float]:
        """MC-dropout mean/std. Returns (mu, sigma, fps_used)."""
        inputs, fps_used = self._build_inputs(xy, conf, fps)
        inputs = tuple(t.to(DEVICE) for t in inputs)

        def forward() -> torch.Tensor:
            logits = self.model(*inputs)
            return p_fall_from_logits(logits)

        mu, sigma, _ = mc_predict_mu_sigma(self.model, forward, M=int(M))
        return float(mu), float(sigma), fps_used


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file.

    Important: This must *never* crash the server on startup.
    If the YAML is missing or malformed, we return an empty dict and the server
    will fall back to safe defaults.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        print(f"[startup] warn: invalid YAML in {path}: {e}")
        return {}
    except Exception as e:
        print(f"[startup] warn: failed to read {path}: {e}")
        return {}


def _load_tau_high_from_ops(ops_path: str) -> Optional[float]:
    """Try to read tau_high from an ops YAML.

    Supported shapes:
    - {ops: {OP2: {thr: ...}}}
    - {OP2: {thr: ...}}
    - {thr: ...}
    """
    if not ops_path:
        return None
    p = Path(ops_path)
    if not p.exists():
        return None
    d = _load_yaml(p)
    if not isinstance(d, dict):
        return None
    if "ops" in d and isinstance(d["ops"], dict):
        d = d["ops"]
    # prefer OP2 if present, else OP_bal/OP1/OP3
    for k in ["OP2", "OP_BAL", "OP1", "OP3"]:
        if k in d and isinstance(d[k], dict) and "thr" in d[k]:
            try:
                return float(d[k]["thr"])
            except Exception:
                pass
    if "thr" in d:
        try:
            return float(d["thr"])
        except Exception:
            return None
    return None


# Default model specs (edit these paths to match your outputs)
MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(id="le2i_tcn", label="LE2I (TCN)", dataset="le2i", arch="tcn", ckpt=f"outputs/le2i_tcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="le2i_gcn", label="LE2I (GCN)", dataset="le2i", arch="gcn", ckpt=f"outputs/le2i_gcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="urfd_tcn", label="URFD (TCN)", dataset="urfd", arch="tcn", ckpt=f"outputs/urfd_tcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="urfd_gcn", label="URFD (GCN)", dataset="urfd", arch="gcn", ckpt=f"outputs/urfd_gcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="caucafall_tcn", label="CAUCAFall (TCN)", dataset="caucafall", arch="tcn", ckpt=f"outputs/caucafall_tcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="caucafall_gcn", label="CAUCAFall (GCN)", dataset="caucafall", arch="gcn", ckpt=f"outputs/caucafall_gcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="muvim_tcn", label="MUVIM (TCN)", dataset="muvim", arch="tcn", ckpt=f"outputs/muvim_tcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
    ModelSpec(id="muvim_gcn", label="MUVIM (GCN)", dataset="muvim", arch="gcn", ckpt=f"outputs/muvim_gcn_W{WIN_W}S{WIN_S}/best.pt", ops=""),
]


# Runtime registry (filled at startup)
RUNNERS: Dict[str, ModelRunner] = {}
DEPLOY_CFG: Dict[str, Any] = {}


@app.on_event("startup")
def _startup_load() -> None:
    global RUNNERS, DEPLOY_CFG
    DEPLOY_CFG = _load_yaml(DEPLOY_CFG_PATH)
    if not DEPLOY_CFG:
        print(f"[startup] warn: missing {DEPLOY_CFG_PATH} (using defaults)")

    # per-arch deploy defaults
    tcn_cfg = DEPLOY_CFG.get("tcn", {}) if isinstance(DEPLOY_CFG, dict) else {}
    gcn_cfg = DEPLOY_CFG.get("gcn", {}) if isinstance(DEPLOY_CFG, dict) else {}

    def triage_defaults(arch: str) -> Tuple[float, float, float, Optional[float]]:
        d = tcn_cfg if arch == "tcn" else gcn_cfg
        tau_low = float(d.get("tau_low", 0.05))
        tau_high = float(d.get("tau_high", 0.90))
        ema_alpha = float(d.get("ema_alpha", 0.20))
        sigma_max = d.get("sigma_max", None)
        sigma_max = float(sigma_max) if sigma_max is not None else None
        return tau_low, tau_high, ema_alpha, sigma_max

    RUNNERS = {}
    for spec in MODEL_SPECS:
        ckpt_path = Path(spec.ckpt)
        if not ckpt_path.exists():
            print(f"[startup] skip missing ckpt: {ckpt_path}")
            continue

        try:
            bundle = load_ckpt(str(ckpt_path), map_location="cpu")
            model_cfg = get_cfg(bundle, "model_cfg", default={}) or {}
            raw_feat = get_cfg(bundle, "feat_cfg", default={}) or {}
            if hasattr(raw_feat, "to_dict"):
                try:
                    raw_feat = raw_feat.to_dict()
                except Exception:
                    pass
            feat_cfg = FeatCfg.from_dict(raw_feat)

            ckpt_arch = str(get_cfg(bundle, "arch", default=model_cfg.get("arch", spec.arch)) or spec.arch).lower()
            arch = ckpt_arch if ckpt_arch in ("tcn", "gcn") else str(spec.arch)
            data_cfg = get_cfg(bundle, "data_cfg", default={}) or {}
            fps_default = float(data_cfg.get("fps_default", spec.fps_default))

            tau_low, tau_high, ema_alpha, sigma_max = triage_defaults(arch)
            # If an ops YAML exists, prefer its OP2 threshold as tau_high.
            tau_from_ops = _load_tau_high_from_ops(spec.ops)
            if tau_from_ops is not None:
                tau_high = float(tau_from_ops)

            # IMPORTANT: use the checkpoint's feature config to rebuild the model
            model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(DEVICE)
            model.load_state_dict(bundle["state_dict"], strict=True)
            model.eval()

            two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False

            runner = ModelRunner(
                spec=spec,
                model=model,
                arch=arch,
                feat_cfg=feat_cfg,
                model_cfg=model_cfg,
                fps_default=fps_default,
                two_stream=two_stream,
                tau_low=tau_low,
                tau_high=tau_high,
                ema_alpha=ema_alpha,
                sigma_max=sigma_max,
            )
            RUNNERS[spec.id.lower()] = runner
            print(f"[startup] loaded {spec.id} arch={arch} device={DEVICE} tau_high={tau_high:.3f}")
        except Exception as e:
            print(f"[startup] FAILED loading {spec.id} ({spec.ckpt}): {e}")


# ===================================================
# Session state machines (in-memory)
# ===================================================

@dataclass
class _Session:
    t0_mono: float
    machines: Dict[str, Any]


SESSIONS: Dict[str, _Session] = {}


def _norm_mode(mode: str) -> str:
    """Normalise a front-end mode string to: 'tcn' | 'gcn' | 'dual'."""
    m = (mode or "").strip().lower()
    if m in ("tcn", "mode1", "single_tcn"):
        return "tcn"
    if m in ("gcn", "mode2", "single_gcn"):
        return "gcn"
    if m in ("dual", "hybrid", "tcn+gcn", "tcn_gcn", "tcn-gcn", "mode3"):
        return "dual"
    return m


def _get_time_sec(session_id: str, timestamp_ms: Optional[int]) -> float:
    if timestamp_ms is not None:
        return float(timestamp_ms) / 1000.0
    now = time.monotonic()
    sess = SESSIONS.get(session_id)
    if sess is None:
        sess = _Session(t0_mono=now, machines={})
        SESSIONS[session_id] = sess
    return now - sess.t0_mono


def _single_mode_cfg() -> SingleModeCfg:
    d = DEPLOY_CFG.get("single", {}) if isinstance(DEPLOY_CFG, dict) else {}
    return SingleModeCfg(
        possible_k=int(d.get("possible_k", 3)),
        possible_T_s=float(d.get("possible_T_s", 2.0)),
        confirm_T_s=float(d.get("confirm_T_s", 3.6)),
        confirm_k_fall=int(d.get("confirm_k_fall", 2)),
        cooldown_possible_s=float(d.get("cooldown_possible_s", 15.0)),
        cooldown_confirmed_s=float(d.get("cooldown_confirmed_s", 60.0)),
    )


def _dual_mode_cfg() -> DualModeCfg:
    d = DEPLOY_CFG.get("dual", {}) if isinstance(DEPLOY_CFG, dict) else {}
    return DualModeCfg(
        possible_k=int(d.get("possible_k", 3)),
        possible_T_s=float(d.get("possible_T_s", 2.0)),
        confirm_T_s=float(d.get("confirm_T_s", 3.6)),
        confirm_k_tcn=int(d.get("confirm_k_tcn", 1)),
        confirm_k_gcn=int(d.get("confirm_k_gcn", 1)),
        require_both=bool(d.get("require_both", True)),
        cooldown_possible_s=float(d.get("cooldown_possible_s", 15.0)),
        cooldown_confirmed_s=float(d.get("cooldown_confirmed_s", 60.0)),
    )


def _mc_cfg() -> Tuple[int, int]:
    d = DEPLOY_CFG.get("mc", {}) if isinstance(DEPLOY_CFG, dict) else {}
    M = int(d.get("M", 1))
    M_confirm = int(d.get("M_confirm", max(1, M)))
    return max(1, M), max(1, M_confirm)


def _get_or_create_sm(session_id: str, key: str, factory) -> Any:
    sess = SESSIONS.get(session_id)
    if sess is None:
        sess = _Session(t0_mono=time.monotonic(), machines={})
        SESSIONS[session_id] = sess
    if key not in sess.machines:
        sess.machines[key] = factory()
    return sess.machines[key]


def _triage_cfg_from_runner(r: ModelRunner) -> TriageCfg:
    return TriageCfg(
        tau_low=float(r.tau_low),
        tau_high=float(r.tau_high),
        ema_alpha=float(r.ema_alpha),
        sigma_max=r.sigma_max,
    )


# ===================================================
# Payload models
# ===================================================

class PoseWindowPayload(BaseModel):
    model_config = {"protected_namespaces": ()}
    # stream/session identity
    session_id: str = Field(default="default")
    resident_id: int = Field(default=DEFAULT_RESIDENT_ID)

    # mode selection
    # NOTE: keep this as a plain string so the front-end can send
    # 'dual', 'hybrid', 'tcn+gcn', etc. We normalise it server-side.
    mode: str = Field(default="tcn")
    model_id: Optional[str] = Field(default=None, description="Back-compat for single-model mode")
    model_tcn: Optional[str] = None
    model_gcn: Optional[str] = None

    # timing
    fps: Optional[float] = None
    timestamp_ms: Optional[int] = None

    # if true, also compute MC-dropout uncertainty when needed
    use_mc: bool = True

    # if true, attempt to store alerts into DB
    persist: bool = False

    # data
    xy: List[List[List[float]]]
    conf: List[List[float]]


# ===================================================
# DB helpers (optional)
# ===================================================

def _db_model_id_for_code(model_code: str) -> Optional[int]:
    """Return models.id for a model 'code'. If missing, create a minimal row.

    We use *stable* codes for DB grouping: 'TCN', 'GCN', 'HYBRID'.
    The runner/spec id (e.g. 'muvim_gcn_W48S12') is stored in event_metadata.
    """
    code = (model_code or "").strip().upper()
    if code not in ("TCN", "GCN", "HYBRID"):
        # fallback: keep what caller passed
        code = (model_code or "").strip()
        if not code:
            return None

    family = "Hybrid" if code == "HYBRID" else code
    name = code

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM models WHERE code=%s", (code,))
                row = cur.fetchone()
                if row and "id" in row:
                    return int(row["id"])

                # Create a minimal row; other fields can remain NULL.
                cur.execute(
                    """
                    INSERT INTO models (code, name, family)
                    VALUES (%s, %s, %s)
                    """,
                    (code, name, family),
                )
                cur.execute("SELECT id FROM models WHERE code=%s", (code,))
                row2 = cur.fetchone()
                return int(row2["id"]) if row2 and "id" in row2 else None
    except Exception:
        return None



def _db_has_column(table: str, col: str) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE %s", (col,))
                row = cur.fetchone()
                return row is not None
    except Exception:
        return False


HAS_EVENT_METADATA_COL: Optional[bool] = None


def _db_ensure_event_metadata_column() -> None:
    """Best-effort migration: add events.event_metadata JSON if missing."""
    global HAS_EVENT_METADATA_COL
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW COLUMNS FROM `events` LIKE %s", ("event_metadata",))
                row = cur.fetchone()
                if row is None:
                    # JSON supported in MySQL 5.7+. If your DB is older, change to TEXT.
                    cur.execute("ALTER TABLE `events` ADD COLUMN `event_metadata` JSON NULL")
                HAS_EVENT_METADATA_COL = True
    except Exception:
        # If DB unavailable, keep as None; inserts will be skipped or fall back.
        HAS_EVENT_METADATA_COL = False


def _db_insert_event(
    resident_id: int,
    model_code: str,
    event_type: str,
    status: str,
    p_fall: float,
    delay_seconds: float | None,
    fa24h_snapshot: float | None = None,
    operating_point_id: int | None = None,
    event_metadata: dict | None = None,
) -> None:
    """Insert an event row.

    - model_code: 'TCN' | 'GCN' | 'HYBRID'
    - event_metadata: stores spec_id(s), thresholds, mode, etc.
    """
    model_db_id = _db_model_id_for_code(model_code)
    if model_db_id is None:
        return

    meta_json = None
    if event_metadata is not None:
        try:
            meta_json = json.dumps(event_metadata, ensure_ascii=False)
        except Exception:
            meta_json = None

    global HAS_EVENT_METADATA_COL
    if HAS_EVENT_METADATA_COL is None:
        # lazy detect
        HAS_EVENT_METADATA_COL = _db_has_column("events", "event_metadata")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if HAS_EVENT_METADATA_COL:
                    cur.execute(
                        """
                        INSERT INTO events
                          (resident_id, model_id, operating_point_id, event_time, type, p_fall, delay_seconds, fa24h_snapshot, status, event_metadata)
                        VALUES
                          (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, CAST(%s AS JSON))
                        """,
                        (
                            int(resident_id),
                            int(model_db_id),
                            int(operating_point_id) if operating_point_id is not None else None,
                            str(event_type),
                            float(p_fall),
                            float(delay_seconds) if delay_seconds is not None else None,
                            float(fa24h_snapshot) if fa24h_snapshot is not None else None,
                            str(status),
                            meta_json,
                        ),
                    )
                else:
                    # Fallback: store metadata JSON in notes
                    cur.execute(
                        """
                        INSERT INTO events
                          (resident_id, model_id, operating_point_id, event_time, type, p_fall, delay_seconds, fa24h_snapshot, status, notes)
                        VALUES
                          (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            int(resident_id),
                            int(model_db_id),
                            int(operating_point_id) if operating_point_id is not None else None,
                            str(event_type),
                            float(p_fall),
                            float(delay_seconds) if delay_seconds is not None else None,
                            float(fa24h_snapshot) if fa24h_snapshot is not None else None,
                            str(status),
                            meta_json,
                        ),
                    )
    except Exception:
        return



@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_models": sorted(list(RUNNERS.keys())),
    }


@app.get("/")
def root():
    """Tiny landing page for the FastAPI server.

    Your React front-end will typically run separately (e.g., Vite on :5173).
    This page is just to make it obvious the API is online.
    """
    return {
        "name": "Elder Fall Monitoring API",
        "health": "/api/health",
        "models": "/api/models/summary",
        "docs": "/docs",
    }


@app.get("/monitor-demo", response_class=HTMLResponse)
def monitor_demo_page():
    """Optional static demo page.

    This does NOT replace your React UI. If you create
    server/static/monitor-demo.html it will be served here.
    """
    p = _STATIC_DIR / "monitor-demo.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return """
    <!doctype html>
    <html><head><meta charset='utf-8'><title>monitor-demo</title></head>
    <body style='font-family: system-ui; padding: 24px;'>
      <h2>monitor-demo</h2>
      <p>No static demo page found. (Create <code>server/static/monitor-demo.html</code> if you want one.)</p>
      <p>Your React front-end should call <code>/api/monitor/predict_window</code> for real-time inference.</p>
    </body></html>
    """


@app.get("/api/models/summary")
def models_summary():
    models = []
    for mid, r in RUNNERS.items():
        models.append(
            {
                "id": mid,
                "label": r.spec.label,
                "dataset": r.spec.dataset,
                "arch": r.arch,
                "ckpt": r.spec.ckpt,
                "fps_default": r.fps_default,
                "tau_low": r.tau_low,
                "tau_high": r.tau_high,
                "ema_alpha": r.ema_alpha,
                "sigma_max": r.sigma_max,
            }
        )
    # expose triage timings for front-end UI
    single = _single_mode_cfg().__dict__
    dual = _dual_mode_cfg().__dict__
    M, M_confirm = _mc_cfg()
    return {
        "models": models,
        "modes": {
            "tcn": {"latency_targets_s": {"possible": float(single["possible_T_s"]), "confirmed": float(single["possible_T_s"] + single["confirm_T_s"]) }},
            "gcn": {"latency_targets_s": {"possible": float(single["possible_T_s"]), "confirmed": float(single["possible_T_s"] + single["confirm_T_s"]) }},
            "dual": {"latency_targets_s": {"possible": float(dual["possible_T_s"]), "confirmed": float(dual["possible_T_s"] + dual["confirm_T_s"]) }},
        },
        "mc": {"M": int(M), "M_confirm": int(M_confirm)},
        "deploy_cfg_path": str(DEPLOY_CFG_PATH),
    }


@app.post("/api/monitor/predict_window")
def predict_window(payload: PoseWindowPayload):
    # ---- pick models by mode ----
    mode = _norm_mode(payload.mode)
    if mode not in ("tcn", "gcn", "dual"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {payload.mode}")

    # Back-compat: model_id selects the single model.
    model_id = (payload.model_id or "").lower().strip()
    model_tcn = (payload.model_tcn or "").lower().strip()
    model_gcn = (payload.model_gcn or "").lower().strip()

    if mode in ("tcn", "gcn"):
        chosen = model_id or (model_tcn if mode == "tcn" else model_gcn)
        if not chosen:
            # pick the first loaded model of that arch
            for mid, r in RUNNERS.items():
                if r.arch == mode:
                    chosen = mid
                    break
        if not chosen or chosen not in RUNNERS:
            raise HTTPException(status_code=400, detail=f"Unknown model_id: {chosen}")
        r = RUNNERS[chosen]
        if r.arch != mode:
            raise HTTPException(status_code=400, detail=f"Model {chosen} is arch={r.arch}, but mode={mode}")
        runners = {mode: r}
    else:
        # dual
        if not model_tcn:
            model_tcn = next((mid for mid, rr in RUNNERS.items() if rr.arch == "tcn"), "")
        if not model_gcn:
            model_gcn = next((mid for mid, rr in RUNNERS.items() if rr.arch == "gcn"), "")
        if not model_tcn or model_tcn not in RUNNERS:
            raise HTTPException(status_code=400, detail=f"Unknown model_tcn: {model_tcn}")
        if not model_gcn or model_gcn not in RUNNERS:
            raise HTTPException(status_code=400, detail=f"Unknown model_gcn: {model_gcn}")
        rt = RUNNERS[model_tcn]
        rg = RUNNERS[model_gcn]
        if rt.arch != "tcn" or rg.arch != "gcn":
            raise HTTPException(status_code=400, detail="model_tcn must be a TCN and model_gcn must be a GCN")
        runners = {"tcn": rt, "gcn": rg}

    # ---- parse arrays ----
    xy = np.array(payload.xy, dtype=np.float32)
    conf = np.array(payload.conf, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise HTTPException(status_code=400, detail=f"xy must have shape [T,33,2], got {xy.shape}")
    if conf.ndim != 2 or conf.shape[0] != xy.shape[0] or conf.shape[1] != xy.shape[1]:
        raise HTTPException(status_code=400, detail=f"conf must have shape [T,33], got {conf.shape}")

    fps = float(payload.fps) if payload.fps is not None else 0.0
    t_sec = _get_time_sec(payload.session_id, payload.timestamp_ms)

    # MC settings
    M, M_confirm = _mc_cfg()
    use_mc = bool(payload.use_mc)

    single_cfg = _single_mode_cfg()
    dual_cfg = _dual_mode_cfg()

    out_models: Dict[str, Any] = {}
    alert_level = "none"
    triage_out = {}

    # ---- single-mode ----
    if mode in ("tcn", "gcn"):
        r = runners[mode]
        tri_cfg = _triage_cfg_from_runner(r)

        sm_key = f"single:{mode}:{r.spec.id.lower()}"
        sm: SingleTriageStateMachine = _get_or_create_sm(
            payload.session_id,
            sm_key,
            lambda: SingleTriageStateMachine(triage_cfg=tri_cfg, mode_cfg=single_cfg),
        )

        # quick deterministic pass
        p_det, fps_used = r.predict_p(xy, conf, fps)
        mu = p_det
        sigma = None

        # compute MC-dropout when enabled and configured with M>1
        # (sigma_max only affects how triage uses sigma; front-end may still want sigma for display)
        if use_mc and (M > 1 or M_confirm > 1):
            # during confirm state -> heavier MC
            state_now = getattr(sm, "_state", "idle")
            m_use = M_confirm if state_now == "confirm" else M
            if m_use > 1:
                mu, sigma, fps_used = r.predict_mu_sigma(xy, conf, fps, m_use)

        # step the state machine
        evs = sm.step(t_sec, float(mu), sigma=float(sigma) if sigma is not None else None)

        tri = triage_state(
            float(getattr(sm, "_ema", mu)),
            tri_cfg.tau_low,
            tri_cfg.tau_high,
            sigma=sigma,
            sigma_max=tri_cfg.sigma_max,
        )
        triage_out = {
            "state": tri,
            "tau_low": tri_cfg.tau_low,
            "tau_high": tri_cfg.tau_high,
            "ema": float(getattr(sm, "_ema", mu)),
            "sigma_max": tri_cfg.sigma_max,
            "sm_state": getattr(sm, "_state", None),
        }

        out_models[mode] = {
            "model_id": r.spec.id,
            "p_det": float(p_det),
            "mu": float(mu),
            "sigma": float(sigma) if sigma is not None else None,
            "triage": triage_out,
            "fps_used": float(fps_used),
        }

        # events -> alert level
        for e in evs:
            if e.kind == EVENT_POSSIBLE:
                alert_level = "possible_fall"
                if payload.persist:
                    _db_insert_event(payload.resident_id, ("TCN" if mode=="tcn" else "GCN"), "possible_fall", "pending_review", float(mu), 0.0, event_metadata={"spec_id": r.spec.id, "mode": mode, "triage": triage_out})
            elif e.kind == EVENT_CONFIRMED:
                alert_level = "fall_detected"
                if payload.persist:
                    _db_insert_event(payload.resident_id, ("TCN" if mode=="tcn" else "GCN"), "fall", "confirmed_fall", float(mu), 0.0, event_metadata={"spec_id": r.spec.id, "mode": mode, "triage": triage_out})

        # Top-level triage for convenience (what most UIs want).
        return {
            "mode": mode,
            "session_id": payload.session_id,
            "t_sec": float(t_sec),
            "triage_state": tri,
            "alert_level": alert_level,
            "models": out_models,
            "device": str(DEVICE),
            "mc": {"enabled": bool(use_mc), "M": int(M), "M_confirm": int(M_confirm)},
            # camelCase aliases for front-ends that prefer it
            "alertLevel": alert_level,
            "triageState": tri,
        }

    # ---- dual-mode ----
    rt = runners["tcn"]
    rg = runners["gcn"]
    tri_t = _triage_cfg_from_runner(rt)
    tri_g = _triage_cfg_from_runner(rg)

    sm_key = f"dual:{rt.spec.id.lower()}+{rg.spec.id.lower()}"
    smd: DualTriageStateMachine = _get_or_create_sm(
        payload.session_id,
        sm_key,
        lambda: DualTriageStateMachine(triage_tcn=tri_t, triage_gcn=tri_g, mode_cfg=dual_cfg),
    )

    p_t_det, fps_t = rt.predict_p(xy, conf, fps)
    p_g_det, fps_g = rg.predict_p(xy, conf, fps)
    mu_t, mu_g = p_t_det, p_g_det
    sig_t = None
    sig_g = None

    if use_mc:
        state_now = getattr(smd, "_state", "idle")
        m_use = M_confirm if state_now == "confirm" else M
        if m_use > 1:
            mu_t, sig_t, fps_t = rt.predict_mu_sigma(xy, conf, fps, m_use)
            mu_g, sig_g, fps_g = rg.predict_mu_sigma(xy, conf, fps, m_use)

    evs = smd.step(t_sec, float(mu_t), float(mu_g), sigma_tcn=sig_t, sigma_gcn=sig_g)

    out_models["tcn"] = {
        "model_id": rt.spec.id,
        "p_det": float(p_t_det),
        "mu": float(mu_t),
        "sigma": float(sig_t) if sig_t is not None else None,
        "triage": {
            "state": triage_state(float(getattr(smd, "_ema_tcn", mu_t)), tri_t.tau_low, tri_t.tau_high, sigma=sig_t, sigma_max=tri_t.sigma_max),
            "tau_low": tri_t.tau_low,
            "tau_high": tri_t.tau_high,
            "ema": float(getattr(smd, "_ema_tcn", mu_t)),
        },
        "fps_used": float(fps_t),
    }
    out_models["gcn"] = {
        "model_id": rg.spec.id,
        "p_det": float(p_g_det),
        "mu": float(mu_g),
        "sigma": float(sig_g) if sig_g is not None else None,
        "triage": {
            "state": triage_state(float(getattr(smd, "_ema_gcn", mu_g)), tri_g.tau_low, tri_g.tau_high, sigma=sig_g, sigma_max=tri_g.sigma_max),
            "tau_low": tri_g.tau_low,
            "tau_high": tri_g.tau_high,
            "ema": float(getattr(smd, "_ema_gcn", mu_g)),
        },
        "fps_used": float(fps_g),
    }

    triage_out = {"tcn": out_models["tcn"]["triage"], "gcn": out_models["gcn"]["triage"], "sm_state": getattr(smd, "_state", None)}

    # Overall triage (fall / uncertain / not_fall) for the fusion mode.
    tri_t = out_models["tcn"]["triage"]["state"]
    tri_g = out_models["gcn"]["triage"]["state"]
    if tri_t == TRIAGE_FALL and tri_g == TRIAGE_FALL:
        tri_overall = TRIAGE_FALL
    elif tri_t == TRIAGE_NOT_FALL and tri_g == TRIAGE_NOT_FALL:
        tri_overall = TRIAGE_NOT_FALL
    else:
        tri_overall = TRIAGE_UNCERTAIN


    for e in evs:
        if e.kind == EVENT_POSSIBLE:
            alert_level = "possible_fall"
            if payload.persist:
                _db_insert_event(payload.resident_id, "HYBRID", "possible_fall", "pending_review", float(max(mu_t, mu_g)), 0.0, event_metadata={"spec_tcn": rt.spec.id, "spec_gcn": rg.spec.id, "mode": "dual", "triage": triage_out})
        elif e.kind == EVENT_CONFIRMED:
            alert_level = "fall_detected"
            if payload.persist:
                _db_insert_event(payload.resident_id, "HYBRID", "fall", "confirmed_fall", float(max(mu_t, mu_g)), 0.0, event_metadata={"spec_tcn": rt.spec.id, "spec_gcn": rg.spec.id, "mode": "dual", "triage": triage_out})

    return {
        "mode": "dual",
        "session_id": payload.session_id,
        "t_sec": float(t_sec),
        "triage_state": tri_overall,
        "alert_level": alert_level,
        "models": out_models,
        "device": str(DEVICE),
        "mc": {"enabled": bool(use_mc), "M": int(M), "M_confirm": int(M_confirm)},
        "alertLevel": alert_level,
        "triageState": tri_overall,
    }


@app.post("/api/monitor/reset_session")
def reset_session(session_id: str = "default"):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"ok": True, "session_id": session_id}


# ===================================================
# DB-backed endpoints (kept; optional)
# ===================================================

@app.get("/api/dashboard/summary")
def dashboard_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    if not _db_available():
        raise HTTPException(status_code=503, detail='DB is not configured or unavailable.')

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, display_name FROM residents WHERE id=%s", (resident_id,))
            resident = cur.fetchone()
            if resident is None:
                raise HTTPException(status_code=404, detail="Resident not found")

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
                cur.execute("SELECT name FROM models WHERE id=%s", (settings["active_model_id"],))
                m = cur.fetchone()
                if m:
                    active_model_name = m.get("name")

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
        "resident": {"id": resident["id"], "name": resident.get("display_name")},
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
    if not _db_available():
        raise HTTPException(status_code=503, detail='DB is not configured or unavailable.')

    # Decide whether DB has events.event_metadata
    global HAS_EVENT_METADATA_COL
    if HAS_EVENT_METADATA_COL is None:
        HAS_EVENT_METADATA_COL = _db_has_column("events", "event_metadata")

    meta_sel = "e.event_metadata AS event_metadata" if HAS_EVENT_METADATA_COL else "e.notes AS event_metadata"
    q = f"""
        SELECT
          e.id,
          e.event_time,
          e.type,
          e.p_fall,
          e.delay_seconds,
          e.status,
          m.name AS model_name,
          m.code AS model_code,
          {meta_sel}
        FROM events e
        JOIN models m ON e.model_id = m.id
        WHERE e.resident_id = %s
        ORDER BY e.event_time DESC
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (resident_id, int(limit)))
            rows = cur.fetchall() or []

    # Parse event_metadata JSON if present
    for r in rows:
        meta = r.get("event_metadata")
        if isinstance(meta, (str, bytes)):
            try:
                r["event_metadata"] = json.loads(meta)
            except Exception:
                r["event_metadata"] = meta
        else:
            r["event_metadata"] = meta

    return {"resident_id": resident_id, "events": rows}


# ===================================================
# Optional CRUD endpoints used by the front-end Settings pages
# (These endpoints are DB-backed. If DB is not configured, they return
# safe empty/default responses instead of crashing the app.)
# ===================================================


def _db_available() -> bool:
    try:
        with get_conn() as _conn:
            return True
    except Exception:
        return False


class ResidentCreatePayload(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: Optional[str] = None  # YYYY-MM-DD
    notes: Optional[str] = None


@app.get("/api/residents")
def list_residents():
    if not _db_available():
        return {"enabled": False, "residents": []}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, display_name, date_of_birth, notes FROM residents ORDER BY id ASC")
            rows = cur.fetchall() or []
    return {"enabled": True, "residents": rows}


@app.post("/api/residents")
def create_resident(payload: ResidentCreatePayload):
    if not _db_available():
        raise HTTPException(status_code=503, detail="DB is not configured")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO residents (display_name, date_of_birth, notes) VALUES (%s, %s, %s)",
                (payload.display_name, payload.date_of_birth, payload.notes),
            )
            cur.execute("SELECT LAST_INSERT_ID() AS id")
            rid = int((cur.fetchone() or {}).get("id") or 0)
    return {"ok": True, "id": rid}


class CaregiverPayload(BaseModel):
    resident_id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    phone: Optional[str] = None
    is_primary: bool = True


@app.get("/api/caregivers")
def list_caregivers(resident_id: int = DEFAULT_RESIDENT_ID):
    if not _db_available():
        return {"enabled": False, "caregivers": []}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, resident_id, name, email, phone, is_primary FROM caregivers WHERE resident_id=%s ORDER BY id ASC",
                (int(resident_id),),
            )
            rows = cur.fetchall() or []
    return {"enabled": True, "caregivers": rows}


@app.post("/api/caregivers")
def create_caregiver(payload: CaregiverPayload):
    if not _db_available():
        raise HTTPException(status_code=503, detail="DB is not configured")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO caregivers (resident_id, name, email, phone, is_primary) VALUES (%s, %s, %s, %s, %s)",
                (int(payload.resident_id), payload.name, payload.email, payload.phone, 1 if payload.is_primary else 0),
            )
            cur.execute("SELECT LAST_INSERT_ID() AS id")
            cid = int((cur.fetchone() or {}).get("id") or 0)
    return {"ok": True, "id": cid}


@app.put("/api/caregivers/{caregiver_id}")
def update_caregiver(caregiver_id: int, payload: CaregiverPayload):
    if not _db_available():
        raise HTTPException(status_code=503, detail="DB is not configured")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE caregivers SET resident_id=%s, name=%s, email=%s, phone=%s, is_primary=%s WHERE id=%s",
                (int(payload.resident_id), payload.name, payload.email, payload.phone, 1 if payload.is_primary else 0, int(caregiver_id)),
            )
    return {"ok": True, "id": int(caregiver_id)}


@app.delete("/api/caregivers/{caregiver_id}")
def delete_caregiver(caregiver_id: int):
    if not _db_available():
        raise HTTPException(status_code=503, detail="DB is not configured")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM caregivers WHERE id=%s", (int(caregiver_id),))
    return {"ok": True}


@app.get("/api/operating_points")
def list_operating_points(model_code: Optional[str] = None, resident_id: int = DEFAULT_RESIDENT_ID):
    """Return operating point presets.

    If model_code is provided, it should be 'TCN'|'GCN'|'HYBRID' (stable DB codes).
    """
    if not _db_available():
        return {"enabled": False, "operating_points": []}

    where = ""
    params: List[Any] = []
    if model_code:
        where = "WHERE m.code=%s"
        params.append(str(model_code).upper())

    q = f"""
      SELECT op.id, op.model_id, op.name, op.code, op.thr_detect, op.thr_low_conf, op.thr_high_conf, op.est_fa24h, op.est_recall
      FROM operating_points op
      JOIN models m ON op.model_id=m.id
      {where}
      ORDER BY op.model_id ASC, op.id ASC
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, tuple(params))
            rows = cur.fetchall() or []
    return {"enabled": True, "operating_points": rows}


class TestNotificationPayload(BaseModel):
    resident_id: int = Field(default=DEFAULT_RESIDENT_ID, ge=1)
    caregiver_id: Optional[int] = None
    channel: Literal["email", "sms", "push"] = "email"
    message: Optional[str] = None


@app.post("/api/notifications/test")
def send_test_notification(payload: TestNotificationPayload):
    """Record a test notification in the DB.

    This repo does not implement real SMS/email sending. The UI can still
    use this endpoint to demonstrate the workflow.
    """
    if not _db_available():
        return {"enabled": False, "ok": False, "detail": "DB is not configured"}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO notifications_log (resident_id, caregiver_id, event_id, channel, type, success, error_message)
                VALUES (%s, %s, NULL, %s, 'test_alert', 1, %s)
                """,
                (int(payload.resident_id), payload.caregiver_id, payload.channel, payload.message),
            )
    return {"enabled": True, "ok": True}


class EventStatusUpdatePayload(BaseModel):
    status: Literal["pending_review", "confirmed_fall", "false_alarm", "dismissed"] = Field(...)
    notes: Optional[str] = None


@app.post("/api/events/{event_id}/status")
def update_event_status(event_id: int, payload: EventStatusUpdatePayload, resident_id: int = DEFAULT_RESIDENT_ID):
    """Mark an event as confirmed/false alarm/etc."""
    if not _db_available():
        raise HTTPException(status_code=503, detail='DB is not configured or unavailable.')
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ensure event belongs to resident
            cur.execute("SELECT id FROM events WHERE id=%s AND resident_id=%s", (int(event_id), int(resident_id)))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Event not found")

            if payload.notes is None:
                cur.execute(
                    "UPDATE events SET status=%s, reviewed_at=NOW() WHERE id=%s",
                    (payload.status, int(event_id)),
                )
            else:
                cur.execute(
                    "UPDATE events SET status=%s, reviewed_at=NOW(), notes=%s WHERE id=%s",
                    (payload.status, payload.notes, int(event_id)),
                )
    return {"ok": True, "event_id": int(event_id), "status": payload.status}



class TestFallPayload(BaseModel):
    resident_id: int = 1
    model_code: Optional[str] = None  # 'TCN'|'GCN'|'HYBRID'
    p_fall: float = 0.99
    status: str = 'pending_review'


@app.post("/api/events/test_fall")
def test_fall(payload: TestFallPayload):
    """
    Helper endpoint for the front-end "Test Fall" button (Monitor-demo page).

    Inserts a synthetic fall event into the DB and returns the inserted event_id.
    If DB is not configured / unavailable, returns enabled=False without raising.
    """
    if not _db_available():
        return {"ok": False, "enabled": False, "detail": "DB is not configured"}

    model_code = (payload.model_code or "GCN").upper().strip()
    if model_code not in ("TCN", "GCN", "HYBRID"):
        model_code = "GCN"

    status = (payload.status or "pending_review").lower().strip()
    if status not in ("pending_review", "confirmed_fall", "false_alarm", "dismissed"):
        status = "pending_review"

    # Make sure events.event_metadata exists (best-effort)
    _db_ensure_event_metadata_column()

    model_db_id = _db_model_id_for_code(model_code)
    if model_db_id is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_code: {model_code}")

    meta = {
        "manual": True,
        "source": "test_button",
        "note": "Synthetic test event inserted from the UI",
    }
    meta_json = json.dumps(meta, ensure_ascii=False)

    global HAS_EVENT_METADATA_COL
    if HAS_EVENT_METADATA_COL is None:
        HAS_EVENT_METADATA_COL = _db_has_column("events", "event_metadata")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if HAS_EVENT_METADATA_COL:
                    cur.execute(
                        """
                        INSERT INTO events
                          (resident_id, model_id, operating_point_id, event_time, type, p_fall, delay_seconds, fa24h_snapshot, status, event_metadata)
                        VALUES
                          (%s, %s, NULL, NOW(), 'fall', %s, 0.0, NULL, %s, CAST(%s AS JSON))
                        """,
                        (int(payload.resident_id), int(model_db_id), float(payload.p_fall), status, meta_json),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO events
                          (resident_id, model_id, operating_point_id, event_time, type, p_fall, delay_seconds, fa24h_snapshot, status, notes)
                        VALUES
                          (%s, %s, NULL, NOW(), 'fall', %s, 0.0, NULL, %s, %s)
                        """,
                        (int(payload.resident_id), int(model_db_id), float(payload.p_fall), status, meta_json),
                    )
                event_id = int(cur.lastrowid)
        return {"ok": True, "enabled": True, "event_id": event_id}
    except Exception as e:
        # Return 500 so the UI can show an error, but keep message readable
        raise HTTPException(status_code=500, detail=f"Failed to insert test event: {e}")



@app.get("/api/events/summary")
def events_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN status='confirmed_fall' THEN 1 ELSE 0 END) AS falls,
                  SUM(CASE WHEN status='pending_review' THEN 1 ELSE 0 END) AS pending,
                  SUM(CASE WHEN status='false_alarm' THEN 1 ELSE 0 END) AS false_alarms
                FROM events
                WHERE resident_id = %s
                  AND DATE(event_time) = CURDATE()
                """,
                (resident_id,),
            )
            row = cur.fetchone() or {}
    return {
        "resident_id": resident_id,
        "today": {
            "falls": int(row.get("falls") or 0),
            "pending": int(row.get("pending") or 0),
            "false_alarms": int(row.get("false_alarms") or 0),
        },
    }


@app.get("/api/settings")
def settings(resident_id: int = DEFAULT_RESIDENT_ID):
    """Return both:
    - deploy settings used by the server (triage + timing)
    - system settings stored in DB (monitoring toggle, active model, etc.)
    """
    # ---- deploy settings (YAML) ----
    M, M_confirm = _mc_cfg()
    deploy = {
        "window": {"W": WIN_W, "S": WIN_S},
        "latency_targets_s": {
            "possible": float(_single_mode_cfg().possible_T_s),
            "confirmed": float(_single_mode_cfg().possible_T_s + _single_mode_cfg().confirm_T_s),
        },
        "triage_defaults": {"tcn": DEPLOY_CFG.get("tcn", {}), "gcn": DEPLOY_CFG.get("gcn", {})},
        "mode_cfg": {"single": _single_mode_cfg().__dict__, "dual": _dual_mode_cfg().__dict__},
        "mc": {"M": int(M), "M_confirm": int(M_confirm)},
    }

    # ---- system settings (DB) ----
    sys = {
        "monitoring_enabled": None,
        "notify_on_every_fall": None,
        "require_confirmation": None,
        "active_model_code": None,  # 'TCN'|'GCN'|'HYBRID'
        "active_operating_point": None,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ss.monitoring_enabled, ss.notify_on_every_fall, ss.require_confirmation,
                           ss.active_operating_point, m.code AS active_model_code
                    FROM system_settings ss
                    LEFT JOIN models m ON ss.active_model_id = m.id
                    WHERE ss.resident_id=%s
                    """,
                    (int(resident_id),),
                )
                row = cur.fetchone()
                if row is None:
                    # create default settings row
                    cur.execute(
                        """
                        INSERT INTO system_settings (resident_id)
                        VALUES (%s)
                        """,
                        (int(resident_id),),
                    )
                    row = {
                        "monitoring_enabled": 1,
                        "notify_on_every_fall": 1,
                        "require_confirmation": 0,
                        "active_operating_point": None,
                        "active_model_code": None,
                    }

                sys = {
                    "monitoring_enabled": bool(row.get("monitoring_enabled")) if row.get("monitoring_enabled") is not None else None,
                    "notify_on_every_fall": bool(row.get("notify_on_every_fall")) if row.get("notify_on_every_fall") is not None else None,
                    "require_confirmation": bool(row.get("require_confirmation")) if row.get("require_confirmation") is not None else None,
                    "active_operating_point": row.get("active_operating_point"),
                    "active_model_code": row.get("active_model_code"),
                }
    except Exception:
        # DB not configured -> keep None fields
        pass

    return {"resident_id": resident_id, "deploy": deploy, "system": sys}


class SettingsUpdatePayload(BaseModel):
    monitoring_enabled: Optional[bool] = None
    notify_on_every_fall: Optional[bool] = None
    require_confirmation: Optional[bool] = None
    active_model_code: Optional[Literal["TCN", "GCN", "HYBRID"]] = None
    active_operating_point: Optional[int] = None


@app.put("/api/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: int = DEFAULT_RESIDENT_ID):
    """Update system_settings row (DB)."""
    updates = []
    params = []

    if payload.monitoring_enabled is not None:
        updates.append("monitoring_enabled=%s")
        params.append(1 if payload.monitoring_enabled else 0)

    if payload.notify_on_every_fall is not None:
        updates.append("notify_on_every_fall=%s")
        params.append(1 if payload.notify_on_every_fall else 0)

    if payload.require_confirmation is not None:
        updates.append("require_confirmation=%s")
        params.append(1 if payload.require_confirmation else 0)

    if payload.active_operating_point is not None:
        updates.append("active_operating_point=%s")
        params.append(int(payload.active_operating_point))

    if payload.active_model_code is not None:
        mid = _db_model_id_for_code(payload.active_model_code)
        updates.append("active_model_id=%s")
        params.append(int(mid) if mid is not None else None)

    if not updates:
        return {"ok": True}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # ensure row exists
                cur.execute("SELECT id FROM system_settings WHERE resident_id=%s", (int(resident_id),))
                row = cur.fetchone()
                if row is None:
                    cur.execute("INSERT INTO system_settings (resident_id) VALUES (%s)", (int(resident_id),))

                q = "UPDATE system_settings SET " + ", ".join(updates) + " WHERE resident_id=%s"
                params2 = params + [int(resident_id)]
                cur.execute(q, tuple(params2))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


