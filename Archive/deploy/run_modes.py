#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/run_modes.py

Run the real-time triage logic in three modes:

Mode 1: tcn   (baseline, conservative)
Mode 2: gcn   (sensitive)
Mode 3: dual  (tcn + gcn, agreement-confirm)

Inputs
------
Option A: --win_dir PATH
    PATH contains window .npz files created by windows/make_windows.py or make_unlabeled_windows.py.

(Option B: JSONL stream is not included in this v1 runner to keep it minimal on CPU.
           You can still deploy by producing windows on the fly on your backend and
           calling the same state machine per window.)

Outputs
-------
Prints events to stdout and can optionally write a JSON list with --out_json.

CPU-friendly recommendation
---------------------------
- Normal: single forward pass (M=1).
- Confirm window: MC Dropout sampling (M_confirm=12) if enabled in YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.ckpt import load_ckpt, get_cfg
from core.models import build_model, pick_device, logits_1d
from core.features import read_window_npz, build_tcn_input, build_gcn_input, FeatCfg
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
from core.uncertainty import mc_predict_mu_sigma


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, str, Dict[str, Any], FeatCfg, bool, float]:
    b = load_ckpt(ckpt_path, map_location="cpu")
    arch = str(b.get("arch", "unknown")).lower()
    model_cfg = get_cfg(b, "model_cfg", {}) or {}
    feat_cfg_d = get_cfg(b, "feat_cfg", {}) or {}
    data_cfg = get_cfg(b, "data_cfg", {}) or {}

    fps_default = float(data_cfg.get("fps_default", 30.0))
    two_stream = bool(model_cfg.get("two_stream", False))

    model = build_model(arch=arch, model_cfg=model_cfg, feat_cfg=feat_cfg_d, fps_default=fps_default)
    sd = b.get("state_dict", b.get("model", None))
    if isinstance(sd, dict):
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    return model, arch, model_cfg, FeatCfg.from_dict(feat_cfg_d), two_stream, fps_default


def iter_windows(win_dir: str) -> Dict[str, List[str]]:
    files = sorted(glob.glob(os.path.join(win_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz windows found in: {win_dir}")
    by_vid: Dict[str, List[str]] = {}
    for p in files:
        # read meta cheaply (np.load is ok; files are small)
        try:
            with np.load(p, allow_pickle=True) as z:
                vid = z.get("video_id", None)
                seq = z.get("seq_id", None)
                if vid is None:
                    vid = seq
                if isinstance(vid, np.ndarray) and vid.shape == ():
                    vid = vid.item()
                vid = str(vid) if vid is not None else os.path.splitext(os.path.basename(p))[0]
                ws = int(z.get("w_start", 0))
        except Exception:
            vid = os.path.splitext(os.path.basename(p))[0]
            ws = 0

        by_vid.setdefault(vid, []).append((ws, p))

    # sort by w_start per video
    out: Dict[str, List[str]] = {}
    for vid, lst in by_vid.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        out[vid] = [p for _, p in lst_sorted]
    return out


def _get_t(meta, time_mode: str) -> float:
    ws = float(meta.w_start)
    we = float(meta.w_end)
    fps = float(meta.fps) if meta.fps and meta.fps > 0 else 30.0
    if time_mode == "start":
        return ws / fps
    if time_mode == "end":
        return we / fps
    return 0.5 * (ws + we) / fps


def _load_one_input(path: str, arch: str, feat_cfg: FeatCfg, two_stream: bool, fps_default: float):
    joints, motion, conf, mask, fps, meta = read_window_npz(path, fps_default=fps_default)
    if arch == "tcn":
        Xt, _m = build_tcn_input(joints, motion, conf, mask, float(fps), feat_cfg)
        return Xt, meta
    # gcn
    Xg, _m = build_gcn_input(joints, motion, conf, mask, float(fps), feat_cfg)
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
            xm = np.zeros_like(xy)
        return (xj, xm), meta
    return Xg, meta


@torch.no_grad()
def _predict_prob(model: torch.nn.Module, arch: str, X, device: torch.device, two_stream: bool) -> float:
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
    p = _sigmoid(logits).detach().float().cpu().numpy().reshape(-1)[0]
    return float(p)


def _predict_mu_sigma(
    model: torch.nn.Module,
    arch: str,
    X,
    device: torch.device,
    two_stream: bool,
    M: int,
) -> Tuple[float, float]:
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

    mu, sigma, _ = mc_predict_mu_sigma(model, forward_fn, M=int(M), return_samples=False)
    mu_f = float(mu.detach().cpu().numpy().reshape(-1)[0])
    sigma_f = float(sigma.detach().cpu().numpy().reshape(-1)[0])
    return mu_f, sigma_f


def load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return yaml_load_simple(path) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tcn", "gcn", "dual"], required=True)
    ap.add_argument("--win_dir", required=True, help="Directory of window .npz files (e.g., .../test_unlabeled)")
    ap.add_argument("--ckpt_tcn", default="", help="Path to TCN checkpoint (required for mode=tcn or dual)")
    ap.add_argument("--ckpt_gcn", default="", help="Path to GCN checkpoint (required for mode=gcn or dual)")
    ap.add_argument("--cfg", default="", help="YAML config for thresholds + timings")
    ap.add_argument("--device", default="", help="cpu|cuda|mps (default: auto)")
    ap.add_argument("--time_mode", default="center", choices=["start", "center", "end"])
    ap.add_argument("--out_json", default="", help="Optional: write events json to this file")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()

    cfg = load_cfg(args.cfg)

    # defaults (CPU-friendly)
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

    mc_M = int(cfg.get("mc", {}).get("M", 1))
    mc_M_confirm = int(cfg.get("mc", {}).get("M_confirm", 12))

    # load models
    model_tcn = model_gcn = None
    arch_tcn = arch_gcn = ""
    feat_tcn = feat_gcn = None
    two_stream_tcn = two_stream_gcn = False
    fps_tcn = fps_gcn = 30.0

    if args.mode in ("tcn", "dual"):
        if not args.ckpt_tcn:
            raise SystemExit("--ckpt_tcn is required for mode=tcn or dual")
        model_tcn, arch_tcn, _m_cfg, feat_tcn, two_stream_tcn, fps_tcn = load_model(args.ckpt_tcn, device)
        if arch_tcn != "tcn":
            print(f"[warn] ckpt_tcn arch={arch_tcn} (expected tcn)")
    if args.mode in ("gcn", "dual"):
        if not args.ckpt_gcn:
            raise SystemExit("--ckpt_gcn is required for mode=gcn or dual")
        model_gcn, arch_gcn, _m_cfg, feat_gcn, two_stream_gcn, fps_gcn = load_model(args.ckpt_gcn, device)
        if arch_gcn != "gcn":
            print(f"[warn] ckpt_gcn arch={arch_gcn} (expected gcn)")

    # group windows by video
    by_vid = iter_windows(args.win_dir)

    events_out: List[Dict[str, Any]] = []

    for vid, paths in by_vid.items():
        if args.mode == "tcn":
            sm = SingleTriageStateMachine(tcn_tri, single_cfg)
        elif args.mode == "gcn":
            sm = SingleTriageStateMachine(gcn_tri, single_cfg)
        else:
            sm = DualTriageStateMachine(tcn_tri, gcn_tri, dual_cfg)

        for p in paths:
            if args.mode == "tcn":
                X, meta = _load_one_input(p, "tcn", feat_tcn, two_stream_tcn, fps_tcn)
                t = _get_t(meta, args.time_mode)
                # normal / confirm MC
                if mc_M_confirm > 1 and getattr(sm, "_state", "idle") == "confirm":
                    mu, sigma = _predict_mu_sigma(model_tcn, "tcn", X, device, two_stream_tcn, mc_M_confirm)
                    evs = sm.step(t, mu, sigma=sigma)
                else:
                    p1 = _predict_prob(model_tcn, "tcn", X, device, two_stream_tcn)
                    evs = sm.step(t, p1, sigma=None)
            elif args.mode == "gcn":
                X, meta = _load_one_input(p, "gcn", feat_gcn, two_stream_gcn, fps_gcn)
                t = _get_t(meta, args.time_mode)
                if mc_M_confirm > 1 and getattr(sm, "_state", "idle") == "confirm":
                    mu, sigma = _predict_mu_sigma(model_gcn, "gcn", X, device, two_stream_gcn, mc_M_confirm)
                    evs = sm.step(t, mu, sigma=sigma)
                else:
                    p1 = _predict_prob(model_gcn, "gcn", X, device, two_stream_gcn)
                    evs = sm.step(t, p1, sigma=None)
            else:
                # dual: tcn always; gcn always for now (you can gate it later for speed)
                Xt, meta = _load_one_input(p, "tcn", feat_tcn, two_stream_tcn, fps_tcn)
                Xg, _meta2 = _load_one_input(p, "gcn", feat_gcn, two_stream_gcn, fps_gcn)
                t = _get_t(meta, args.time_mode)

                in_confirm = getattr(sm, "_state", "idle") == "confirm"
                if mc_M_confirm > 1 and in_confirm:
                    mu_t, sig_t = _predict_mu_sigma(model_tcn, "tcn", Xt, device, two_stream_tcn, mc_M_confirm)
                    mu_g, sig_g = _predict_mu_sigma(model_gcn, "gcn", Xg, device, two_stream_gcn, mc_M_confirm)
                    evs = sm.step(t, mu_t, mu_g, sigma_tcn=sig_t, sigma_gcn=sig_g)
                else:
                    pt = _predict_prob(model_tcn, "tcn", Xt, device, two_stream_tcn)
                    pg = _predict_prob(model_gcn, "gcn", Xg, device, two_stream_gcn)
                    evs = sm.step(t, pt, pg, sigma_tcn=None, sigma_gcn=None)

            for e in evs:
                rec = {"video_id": vid, "kind": e.kind, "t_sec": e.t_sec, **e.info}
                events_out.append(rec)
                tag = "POSSIBLE" if e.kind == EVENT_POSSIBLE else ("CONFIRMED" if e.kind == EVENT_CONFIRMED else "RESOLVED")
                print(f"[{tag}] vid={vid} t={e.t_sec:.2f}s info={{{', '.join([k+'='+str(v) for k,v in e.info.items()])}}}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(events_out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
