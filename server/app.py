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

DEPLOY_CFG_PATH = Path("configs/deploy_modes.yaml")


# ===================================================
# FastAPI app + CORS
# ===================================================
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
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    mode: Literal["tcn", "gcn", "dual"] = Field(default="tcn")
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
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM models WHERE code=%s", (model_code,))
                row = cur.fetchone()
                return int(row["id"]) if row and "id" in row else None
    except Exception:
        return None


def _db_insert_event(
    resident_id: int,
    model_code: str,
    status: str,
    p_fall: float,
    delay_seconds: float,
) -> None:
    model_db_id = _db_model_id_for_code(model_code)
    if model_db_id is None:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (resident_id, model_id, event_time, type, p_fall, delay_seconds, status)
                    VALUES (%s, %s, NOW(), 'fall', %s, %s, %s)
                    """,
                    (resident_id, model_db_id, float(p_fall), float(delay_seconds), status),
                )
    except Exception:
        # best-effort; don't fail inference
        return


# ===================================================
# API endpoints
# ===================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "loaded_models": sorted(list(RUNNERS.keys())),
    }


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
    mode = payload.mode.lower().strip()
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

        # compute MC only when enabled and helpful
        if use_mc and (M > 1 or M_confirm > 1) and (r.sigma_max is not None):
            # during confirm state -> heavier MC
            state_now = getattr(sm, "_state", "idle")
            m_use = M_confirm if state_now == "confirm" else M
            if m_use > 1:
                mu, sigma, fps_used = r.predict_mu_sigma(xy, conf, fps, m_use)

        # step the state machine
        evs = sm.step(t_sec, float(mu), sigma=float(sigma) if sigma is not None else None)

        tri = triage_state(float(getattr(sm, "_ema", mu)), tri_cfg.tau_low, tri_cfg.tau_high, sigma=sigma, sigma_max=tri_cfg.sigma_max)
        triage_out = {"state": tri, "tau_low": tri_cfg.tau_low, "tau_high": tri_cfg.tau_high, "ema": float(getattr(sm, "_ema", mu))}

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
                    _db_insert_event(payload.resident_id, r.spec.id, "pending_review", float(mu), 0.0)
            elif e.kind == EVENT_CONFIRMED:
                alert_level = "fall_detected"
                if payload.persist:
                    _db_insert_event(payload.resident_id, r.spec.id, "confirmed_fall", float(mu), 0.0)

        return {
            "mode": mode,
            "session_id": payload.session_id,
            "t_sec": float(t_sec),
            "alert_level": alert_level,
            "models": out_models,
            "device": str(DEVICE),
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
            if rt.sigma_max is not None:
                mu_t, sig_t, fps_t = rt.predict_mu_sigma(xy, conf, fps, m_use)
            if rg.sigma_max is not None:
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

    for e in evs:
        if e.kind == EVENT_POSSIBLE:
            alert_level = "possible_fall"
            if payload.persist:
                _db_insert_event(payload.resident_id, f"{rt.spec.id}+{rg.spec.id}", "pending_review", float(max(mu_t, mu_g)), 0.0)
        elif e.kind == EVENT_CONFIRMED:
            alert_level = "fall_detected"
            if payload.persist:
                _db_insert_event(payload.resident_id, f"{rt.spec.id}+{rg.spec.id}", "confirmed_fall", float(max(mu_t, mu_g)), 0.0)

    return {
        "mode": "dual",
        "session_id": payload.session_id,
        "t_sec": float(t_sec),
        "alert_level": alert_level,
        "models": out_models,
        "device": str(DEVICE),
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
                (resident_id, int(limit)),
            )
            rows = cur.fetchall() or []
    return {"resident_id": resident_id, "events": rows}


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
    """Return deploy settings used by the server (triage + timing).

    This is useful for the front-end UI.
    """
    M, M_confirm = _mc_cfg()
    return {
        "resident_id": resident_id,
        "window": {"W": WIN_W, "S": WIN_S},
        "latency_targets_s": {
            "possible": float(_single_mode_cfg().possible_T_s),
            "confirmed": float(_single_mode_cfg().possible_T_s + _single_mode_cfg().confirm_T_s),
        },
        "triage_defaults": {
            "tcn": DEPLOY_CFG.get("tcn", {}),
            "gcn": DEPLOY_CFG.get("gcn", {}),
        },
        "mode_cfg": {
            "single": _single_mode_cfg().__dict__,
            "dual": _dual_mode_cfg().__dict__,
        },
        "mc": {"M": int(M), "M_confirm": int(M_confirm)},
    }
