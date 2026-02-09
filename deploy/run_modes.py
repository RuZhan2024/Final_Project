#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy/run_modes.py

A CPU-friendly "deployment simulation" runner.

It replays window NPZ files in time order and runs a triage state machine that emits:
- POSSIBLE fall
- CONFIRMED fall
- RESOLVED

Modes:
- tcn  : single-model triage using TCN checkpoint
- gcn  : single-model triage using GCN checkpoint
- dual : two-model triage using both checkpoints (agreement confirmation)

Why this exists
---------------
This is NOT part of training.
It is a convenient bridge between "offline ML" and a future real server:
- you can validate thresholds
- you can test latency behavior (possible -> confirmed)
- you can test MC dropout usage during confirm
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.ckpt import load_ckpt, get_cfg, get_state_dict
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_gcn_input
from core.models import build_model, pick_device, logits_1d
from core.uncertainty import mc_predict_mu_sigma
from core.yamlio import yaml_load_simple
from core.alerting import (
    TriageCfg,
    SingleModeCfg,
    DualModeCfg,
    SingleTriageStateMachine,
    DualTriageStateMachine,
    EVENT_POSSIBLE,
    EVENT_CONFIRMED,
    EVENT_RESOLVED,
)


# ============================================================
# 1) Small helpers for robust NPZ key reading
# ============================================================
def _npz_scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    """
    Read a scalar-like NPZ entry safely.

    NPZ stores values as numpy arrays, often 0-d arrays for scalars/strings.
    This helper:
    - returns default if key missing
    - unwraps 0-d arrays using .item()
    - decodes bytes to str when needed
    """
    if key not in z.files:
        return default
    v = z[key]
    try:
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="replace")
        return v
    except Exception:
        return default


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid converts logits -> probabilities in [0,1]."""
    return torch.sigmoid(x)


# ============================================================
# 2) Load a checkpoint bundle into a model (EMA-safe)
# ============================================================
def load_model(
    ckpt_path: str,
    device: torch.device,
    *,
    prefer_ema: bool,
) -> Tuple[torch.nn.Module, str, Dict[str, Any], FeatCfg, bool, float]:
    """
    Returns:
      model        : torch model in eval mode on device
      arch         : "tcn" or "gcn"
      model_cfg    : model hyperparameters dict (used for rebuilding)
      feat_cfg     : FeatCfg (controls feature packing)
      two_stream   : True if this GCN is two-stream
      fps_default  : fallback fps stored in checkpoint
    """
    b = load_ckpt(ckpt_path, map_location="cpu")

    arch = str(b.get("arch", "unknown")).lower()
    model_cfg = get_cfg(b, "model_cfg", {}) or {}
    feat_cfg_d = get_cfg(b, "feat_cfg", {}) or {}
    data_cfg = get_cfg(b, "data_cfg", {}) or {}

    fps_default = float(data_cfg.get("fps_default", 30.0))
    two_stream = bool(model_cfg.get("two_stream", False))

    feat_cfg = FeatCfg.from_dict(feat_cfg_d)

    # Build the model architecture from config.
    # This MUST match training, or state_dict loading will fail.
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default)

    # EMA-safe loading:
    # - base state dict includes BN buffers
    # - ema state dict often only includes params
    sd = get_state_dict(b, prefer_ema=False)
    if prefer_ema:
        ema_sd = b.get("ema_state_dict", None) or b.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    return model, arch, model_cfg, feat_cfg, two_stream, fps_default


# ============================================================
# 3) Iterate windows grouped by video_id, sorted by w_start
# ============================================================
def iter_windows(win_dir: str) -> Dict[str, List[str]]:
    """
    Return:
      { video_id: [path1, path2, ...] }

    Windows are sorted by w_start so they replay in time order.

    NOTE:
    We use a cheap NPZ header read here (np.load) to avoid loading full arrays.
    """
    files = sorted(glob.glob(os.path.join(win_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"No .npz windows found in: {win_dir}")

    by_vid: Dict[str, List[Tuple[int, str]]] = {}

    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                vid = _npz_scalar(z, "video_id", None)
                if vid is None:
                    vid = _npz_scalar(z, "seq_id", None)
                if vid is None:
                    vid = _npz_scalar(z, "clip_id", None)
                if vid is None:
                    vid = os.path.splitext(os.path.basename(p))[0]
                vid = str(vid)

                ws = _npz_scalar(z, "w_start", 0)
                ws = int(ws) if ws is not None else 0
        except Exception:
            vid = os.path.splitext(os.path.basename(p))[0]
            ws = 0

        by_vid.setdefault(vid, []).append((ws, p))

    out: Dict[str, List[str]] = {}
    for vid, lst in by_vid.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        out[vid] = [path for _, path in lst_sorted]
    return out


# ============================================================
# 4) Convert a window NPZ into the right model input tensor
# ============================================================
def load_one_input(
    path: str,
    arch: str,
    feat_cfg: FeatCfg,
    two_stream: bool,
    fps_default: float,
):
    """
    Reads one NPZ window and returns (X, meta).

    X shape depends on model:
    - TCN: X is [T, C]
    - GCN single-stream: X is [T, V, F]
    - GCN two-stream: X is (xj[T,V,Fj], xm[T,V,2])
    """
    joints, motion, conf, mask, fps, meta = read_window_npz(path, fps_default=fps_default)

    if arch == "tcn":
        Xt, _ = build_tcn_input(joints, motion, conf, mask, float(fps), feat_cfg)
        return Xt.astype(np.float32), meta

    Xg, _ = build_gcn_input(joints, motion, conf, mask, float(fps), feat_cfg)

    if two_stream:
        xy = Xg[..., 0:2]
        if feat_cfg.use_conf_channel:
            c = Xg[..., -1:]
            xj = np.concatenate([xy, c], axis=-1)
        else:
            xj = xy

        if feat_cfg.use_motion and Xg.shape[-1] >= 4:
            xm = Xg[..., 2:4]
        else:
            xm = np.zeros_like(xy, dtype=np.float32)

        return (xj.astype(np.float32), xm.astype(np.float32)), meta

    return Xg.astype(np.float32), meta


def window_time_seconds(meta, time_mode: str) -> float:
    """
    Convert window indices -> a timestamp in seconds.

    time_mode:
      start  -> w_start / fps
      center -> (w_start + w_end)/2 / fps
      end    -> w_end / fps
    """
    ws = float(meta.w_start)
    we = float(meta.w_end)
    fps = float(meta.fps) if meta.fps and meta.fps > 0 else 30.0

    if time_mode == "start":
        return ws / fps
    if time_mode == "end":
        return we / fps
    return 0.5 * (ws + we) / fps


# ============================================================
# 5) Prediction helpers (single pass vs MC dropout)
# ============================================================
@torch.no_grad()
def predict_prob(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
) -> float:
    """
    Single forward pass probability.

    We wrap X into batch dimension [1, ...] then:
    logits_1d ensures output shape is [B] => [1]
    """
    if arch == "tcn":
        xb = torch.from_numpy(np.asarray(X, dtype=np.float32))[None, ...].to(device)
        logits = logits_1d(model(xb))
    else:
        if two_stream:
            xj = torch.from_numpy(np.asarray(X[0], dtype=np.float32))[None, ...].to(device)
            xm = torch.from_numpy(np.asarray(X[1], dtype=np.float32))[None, ...].to(device)
            logits = logits_1d(model(xj, xm))
        else:
            xb = torch.from_numpy(np.asarray(X, dtype=np.float32))[None, ...].to(device)
            logits = logits_1d(model(xb))

    p = _sigmoid(logits).detach().cpu().numpy().reshape(-1)[0]
    return float(p)


def predict_mu_sigma(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
    M: int,
) -> Tuple[float, float]:
    """
    MC Dropout estimate:
      mu    = mean probability
      sigma = std probability

    We pass a forward_fn that returns probabilities, not logits.
    """
    def forward_fn() -> torch.Tensor:
        if arch == "tcn":
            xb = torch.from_numpy(np.asarray(X, dtype=np.float32))[None, ...].to(device)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj = torch.from_numpy(np.asarray(X[0], dtype=np.float32))[None, ...].to(device)
                xm = torch.from_numpy(np.asarray(X[1], dtype=np.float32))[None, ...].to(device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = torch.from_numpy(np.asarray(X, dtype=np.float32))[None, ...].to(device)
                logits = logits_1d(model(xb))
        return _sigmoid(logits).detach().float()

    mu, sigma = mc_predict_mu_sigma(model, forward_fn, M=int(M), return_samples=False)
    mu_f = float(mu.detach().cpu().numpy().reshape(-1)[0])
    sigma_f = float(sigma.detach().cpu().numpy().reshape(-1)[0])
    return mu_f, sigma_f


# ============================================================
# 6) YAML config loader
# ============================================================
def load_cfg(path: str) -> Dict[str, Any]:
    """Load YAML config if provided, else return {}.

    If the file path is missing or does not exist, return {} and continue.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[warn] cfg not found: {path} (using defaults)")
        return {}
    return yaml_load_simple(path) or {}


# ============================================================
# 7) Main loop
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["tcn", "gcn", "dual"], required=True)
    ap.add_argument("--win_dir", required=True, help="Directory containing window .npz files (recursive).")

    ap.add_argument("--ckpt_tcn", default="", help="TCN checkpoint (required for mode=tcn or dual).")
    ap.add_argument("--ckpt_gcn", default="", help="GCN checkpoint (required for mode=gcn or dual).")

    ap.add_argument("--cfg", default="", help="YAML config for triage timings/thresholds.")
    ap.add_argument("--device", default="", help="cpu|cuda|mps (default: auto).")
    ap.add_argument("--prefer_ema", type=int, default=1, help="1 => use EMA weights if present in checkpoint.")
    ap.add_argument("--time_mode", default="center", choices=["start", "center", "end"])

    ap.add_argument("--out_json", default="", help="If set, write emitted events to this JSON file.")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    cfg = load_cfg(args.cfg)

    prefer_ema = bool(int(args.prefer_ema))

    # -------------------------
    # Triaging configs (read from YAML with safe defaults)
    # -------------------------
    tcn_tri = TriageCfg(
        tau_low=float(cfg.get("tcn", {}).get("tau_low", 0.05)),
        tau_high=float(cfg.get("tcn", {}).get("tau_high", 0.90)),
        ema_alpha=float(cfg.get("tcn", {}).get("ema_alpha", 0.20)),
        sigma_max=cfg.get("tcn", {}).get("sigma_max", None),
    )
    gcn_tri = TriageCfg(
        tau_low=float(cfg.get("gcn", {}).get("tau_low", 0.05)),
        tau_high=float(cfg.get("gcn", {}).get("tau_high", 0.90)),
        ema_alpha=float(cfg.get("gcn", {}).get("ema_alpha", 0.20)),
        sigma_max=cfg.get("gcn", {}).get("sigma_max", None),
    )

    single_cfg = SingleModeCfg(
        possible_k=int(cfg.get("single", {}).get("possible_k", 3)),
        possible_T_s=float(cfg.get("single", {}).get("possible_T_s", 2.0)),
        confirm_T_s=float(cfg.get("single", {}).get("confirm_T_s", 3.6)),
        confirm_k_fall=int(cfg.get("single", {}).get("confirm_k_fall", 2)),
        cooldown_possible_s=float(cfg.get("single", {}).get("cooldown_possible_s", 15.0)),
        cooldown_confirmed_s=float(cfg.get("single", {}).get("cooldown_confirmed_s", 60.0)),
    )
    dual_cfg = DualModeCfg(
        possible_k=int(cfg.get("dual", {}).get("possible_k", 3)),
        possible_T_s=float(cfg.get("dual", {}).get("possible_T_s", 2.0)),
        confirm_T_s=float(cfg.get("dual", {}).get("confirm_T_s", 3.6)),
        confirm_k_tcn=int(cfg.get("dual", {}).get("confirm_k_tcn", 1)),
        confirm_k_gcn=int(cfg.get("dual", {}).get("confirm_k_gcn", 1)),
        require_both=bool(cfg.get("dual", {}).get("require_both", True)),
        cooldown_possible_s=float(cfg.get("dual", {}).get("cooldown_possible_s", 15.0)),
        cooldown_confirmed_s=float(cfg.get("dual", {}).get("cooldown_confirmed_s", 60.0)),
    )

    # MC Dropout settings:
    # - M used for normal operation (keep 1 for speed)
    # - M_confirm used only during confirm state (common CPU-friendly trick)
    mc_M = int(cfg.get("mc", {}).get("M", 1))
    mc_M_confirm = int(cfg.get("mc", {}).get("M_confirm", 12))

    # -------------------------
    # Load required model(s)
    # -------------------------
    model_tcn = model_gcn = None
    feat_tcn = feat_gcn = None
    two_stream_tcn = two_stream_gcn = False
    fps_tcn = fps_gcn = 30.0

    if args.mode in ("tcn", "dual"):
        if not args.ckpt_tcn:
            raise SystemExit("--ckpt_tcn is required for mode=tcn or dual")
        model_tcn, arch_tcn, _m_cfg, feat_tcn, two_stream_tcn, fps_tcn = load_model(
            args.ckpt_tcn, device, prefer_ema=prefer_ema
        )
        if arch_tcn != "tcn":
            print(f"[warn] --ckpt_tcn arch={arch_tcn} (expected tcn)")

    if args.mode in ("gcn", "dual"):
        if not args.ckpt_gcn:
            raise SystemExit("--ckpt_gcn is required for mode=gcn or dual")
        model_gcn, arch_gcn, _m_cfg, feat_gcn, two_stream_gcn, fps_gcn = load_model(
            args.ckpt_gcn, device, prefer_ema=prefer_ema
        )
        if arch_gcn != "gcn":
            print(f"[warn] --ckpt_gcn arch={arch_gcn} (expected gcn)")

    # -------------------------
    # Group windows by video and replay in time order
    # -------------------------
    by_vid = iter_windows(args.win_dir)

    events_out: List[Dict[str, Any]] = []

    for vid, paths in by_vid.items():
        # Create a fresh state machine per video (just like deployment per stream)
        if args.mode == "tcn":
            sm = SingleTriageStateMachine(tcn_tri, single_cfg)
        elif args.mode == "gcn":
            sm = SingleTriageStateMachine(gcn_tri, single_cfg)
        else:
            sm = DualTriageStateMachine(tcn_tri, gcn_tri, dual_cfg)

        for p in paths:
            # Determine whether we are in the confirm phase.
            # This uses a private attribute, but it’s practical for this runner.
            in_confirm = (getattr(sm, "_state", "idle") == "confirm")

            # Decide MC sample count:
            # - use M_confirm during confirm
            # - otherwise use M (usually 1)
            M_now = mc_M_confirm if (in_confirm and mc_M_confirm > 1) else mc_M

            if args.mode == "tcn":
                X, meta = load_one_input(p, "tcn", feat_tcn, two_stream_tcn, fps_tcn)
                t = window_time_seconds(meta, args.time_mode)

                if M_now > 1 or (tcn_tri.sigma_max is not None):
                    mu, sigma = predict_mu_sigma(model_tcn, "tcn", X, device, two_stream_tcn, M_now)
                    evs = sm.step(t, mu, sigma=sigma)
                else:
                    p1 = predict_prob(model_tcn, "tcn", X, device, two_stream_tcn)
                    evs = sm.step(t, p1, sigma=None)

            elif args.mode == "gcn":
                X, meta = load_one_input(p, "gcn", feat_gcn, two_stream_gcn, fps_gcn)
                t = window_time_seconds(meta, args.time_mode)

                if M_now > 1 or (gcn_tri.sigma_max is not None):
                    mu, sigma = predict_mu_sigma(model_gcn, "gcn", X, device, two_stream_gcn, M_now)
                    evs = sm.step(t, mu, sigma=sigma)
                else:
                    p1 = predict_prob(model_gcn, "gcn", X, device, two_stream_gcn)
                    evs = sm.step(t, p1, sigma=None)

            else:
                # Dual mode: compute both model probs then feed into dual state machine.
                Xt, meta = load_one_input(p, "tcn", feat_tcn, two_stream_tcn, fps_tcn)
                Xg, _meta2 = load_one_input(p, "gcn", feat_gcn, two_stream_gcn, fps_gcn)
                t = window_time_seconds(meta, args.time_mode)

                if M_now > 1 or (tcn_tri.sigma_max is not None) or (gcn_tri.sigma_max is not None):
                    mu_t, sig_t = predict_mu_sigma(model_tcn, "tcn", Xt, device, two_stream_tcn, M_now)
                    mu_g, sig_g = predict_mu_sigma(model_gcn, "gcn", Xg, device, two_stream_gcn, M_now)
                    evs = sm.step(t, mu_t, mu_g, sigma_tcn=sig_t, sigma_gcn=sig_g)
                else:
                    pt = predict_prob(model_tcn, "tcn", Xt, device, two_stream_tcn)
                    pg = predict_prob(model_gcn, "gcn", Xg, device, two_stream_gcn)
                    evs = sm.step(t, pt, pg, sigma_tcn=None, sigma_gcn=None)

            # Print and record emitted events
            for e in evs:
                rec = {"video_id": vid, "kind": e.kind, "t_sec": float(e.t_sec), **(e.info or {})}
                events_out.append(rec)

                if e.kind == EVENT_POSSIBLE:
                    tag = "POSSIBLE"
                elif e.kind == EVENT_CONFIRMED:
                    tag = "CONFIRMED"
                else:
                    tag = "RESOLVED"

                print(f"[{tag}] vid={vid} t={e.t_sec:.2f}s info={rec}")

    # Optional: write all events to JSON
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(events_out, f, indent=2)
        print(f"[ok] wrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
