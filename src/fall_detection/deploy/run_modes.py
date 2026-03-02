#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/run_modes.py

Offline deploy-time runner for *triage state machines* (Mode 1/2/3):

- Mode 1: TCN-only triage
- Mode 2: GCN-only triage
- Mode 3: Dual-model triage (TCN + GCN)

This script is used to sanity-check real-time behaviour on window streams.

Key fixes vs previous version:
- Single NPZ load per window in dual mode (no double I/O)
- Centralised two-stream splitting logic (shared with evaluation)
- Optional lightweight confirm gate using skeleton-derived scores:
    - lying_score (torso horizontalness)
    - motion_score (post-impact stillness proxy)
- Safer np.load usage (allow_pickle=False)
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fall_detection.core.models import pick_device
from fall_detection.core.yamlio import yaml_load_simple
from fall_detection.core.alerting import (
    TriageCfg,
    SingleModeCfg,
    DualModeCfg,
    SingleTriageStateMachine,
    DualTriageStateMachine,
    EVENT_POSSIBLE,
    EVENT_CONFIRMED,
    EVENT_RESOLVED,
)

from fall_detection.deploy.common import (
    load_model_bundle,
    load_window_raw,
    build_input_from_raw,
    compute_confirm_scores,
    predict_prob,
    predict_mu_sigma,
)


def iter_windows(win_dir: str) -> Dict[str, List[str]]:
    """Group window NPZs by video id and sort by w_start."""
    # ✅ CHANGED: recursive glob so nested window dirs work
    paths = sorted(glob.glob(os.path.join(win_dir, "**", "*.npz"), recursive=True))

    by_vid: Dict[str, List[Tuple[int, str]]] = {}
    for p in paths:
        vid = os.path.splitext(os.path.basename(p))[0]
        ws = 0
        try:
            z = np.load(p, allow_pickle=False)
            with z:
                v = z.get("video_id", None)
                if v is not None:
                    if isinstance(v, np.ndarray) and v.shape == ():
                        v = v.item()
                    vid = str(v)
                ws = int(z.get("w_start", 0))
        except Exception:
            pass
        by_vid.setdefault(vid, []).append((ws, p))

    out: Dict[str, List[str]] = {}
    for vid, lst in by_vid.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        out[vid] = [p for _, p in lst_sorted]
    return out


def _get_t(meta, time_mode: str) -> float:
    ws = float(meta.w_start)
    we = float(meta.w_end)  # ✅ CHANGED: inclusive frame index
    fps = float(meta.fps) if meta.fps and meta.fps > 0 else 30.0
    if time_mode == "start":
        return ws / fps
    if time_mode == "end":
        return we / fps
    return 0.5 * (ws + we) / fps


def load_cfg(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return yaml_load_simple(path) or {}


def _tag(kind: str) -> str:
    if kind == EVENT_POSSIBLE:
        return "POSSIBLE"
    if kind == EVENT_CONFIRMED:
        return "CONFIRMED"
    if kind == EVENT_RESOLVED:
        return "RESOLVED"
    return kind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tcn", "gcn", "dual", "mode1", "mode2", "mode3"], required=True)
    ap.add_argument("--win_dir", "--window_dir", dest="win_dir", required=True, help="Directory of window .npz files")
    ap.add_argument("--ckpt_tcn", default="", help="Path to TCN checkpoint (required for mode=tcn or dual)")
    ap.add_argument("--ckpt_gcn", default="", help="Path to GCN checkpoint (required for mode=gcn or dual)")
    ap.add_argument("--cfg", default="", help="YAML config for thresholds + timings")
    ap.add_argument("--device", default="", help="cpu|cuda|mps (default: auto)")
    ap.add_argument("--time_mode", default="center", choices=["start", "center", "end"])
    ap.add_argument("--out_json", "--save_events", dest="out_json", default="", help="Optional: write events json to this file")
    args = ap.parse_args()

    # Backward-compatible mode names
    mode_map = {"mode1": "tcn", "mode2": "gcn", "mode3": "dual"}
    args.mode = mode_map.get(args.mode, args.mode)

    device = torch.device(args.device) if args.device else pick_device()
    cfg = load_cfg(args.cfg)

    # triage thresholds
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

    # mode logic (+ optional confirm gate)
    single_cfg = SingleModeCfg(
        possible_k=int(cfg.get("single", {}).get("possible_k", 3)),
        possible_T_s=float(cfg.get("single", {}).get("possible_T_s", 2.0)),
        confirm_T_s=float(cfg.get("single", {}).get("confirm_T_s", 3.6)),
        confirm_k_fall=int(cfg.get("single", {}).get("confirm_k_fall", 2)),
        cooldown_possible_s=float(cfg.get("single", {}).get("cooldown_possible_s", 15.0)),
        cooldown_confirmed_s=float(cfg.get("single", {}).get("cooldown_confirmed_s", 60.0)),
        confirm_use_scores=bool(cfg.get("single", {}).get("confirm_use_scores", False)),
        confirm_min_lying=float(cfg.get("single", {}).get("confirm_min_lying", 0.65)),
        confirm_max_motion=float(cfg.get("single", {}).get("confirm_max_motion", 0.08)),
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
        confirm_use_scores=bool(cfg.get("dual", {}).get("confirm_use_scores", False)),
        confirm_min_lying=float(cfg.get("dual", {}).get("confirm_min_lying", 0.65)),
        confirm_max_motion=float(cfg.get("dual", {}).get("confirm_max_motion", 0.08)),
    )

    # MC dropout settings
    mc_M_confirm = int(cfg.get("mc", {}).get("M_confirm", 12))

    # Optional dual speed gate: skip GCN when TCN is confidently low.
    gate_gcn = bool(cfg.get("dual", {}).get("gate_gcn", False))
    gate_gcn_tau = float(cfg.get("dual", {}).get("gate_gcn_tau", 0.20))

    # Load models
    model_tcn = model_gcn = None
    arch_tcn = arch_gcn = ""
    feat_tcn = feat_gcn = None
    two_stream_tcn = two_stream_gcn = False
    fps_tcn = fps_gcn = 30.0

    if args.mode in ("tcn", "dual"):
        if not args.ckpt_tcn:
            raise SystemExit("--ckpt_tcn is required for mode=tcn or dual")
        model_tcn, arch_tcn, _m_cfg, feat_tcn, two_stream_tcn, fps_tcn = load_model_bundle(args.ckpt_tcn, device)
        if arch_tcn != "tcn":
            print(f"[warn] ckpt_tcn arch={arch_tcn} (expected tcn)")
    if args.mode in ("gcn", "dual"):
        if not args.ckpt_gcn:
            raise SystemExit("--ckpt_gcn is required for mode=gcn or dual")
        model_gcn, arch_gcn, _m_cfg, feat_gcn, two_stream_gcn, fps_gcn = load_model_bundle(args.ckpt_gcn, device)
        if arch_gcn != "gcn":
            print(f"[warn] ckpt_gcn arch={arch_gcn} (expected gcn)")

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
            # In dual mode we load once and build both inputs from the same raw arrays.
            if args.mode == "dual":
                raw = load_window_raw(p, fps_default=fps_tcn)
                Xt, _ = build_input_from_raw(raw, feat_tcn, "tcn", two_stream=two_stream_tcn)
                Xg, _ = build_input_from_raw(raw, feat_gcn, "gcn", two_stream=two_stream_gcn)
                t = _get_t(raw.meta, args.time_mode)

                in_confirm = getattr(sm, "_state", "idle") == "confirm"
                lying = motion = None
                if in_confirm and bool(dual_cfg.confirm_use_scores):
                    lying, motion = compute_confirm_scores(raw)

                if mc_M_confirm > 1 and in_confirm:
                    mu_t, sig_t = predict_mu_sigma(model_tcn, "tcn", Xt, device, two_stream_tcn, mc_M_confirm)
                    mu_g, sig_g = predict_mu_sigma(model_gcn, "gcn", Xg, device, two_stream_gcn, mc_M_confirm)
                    evs = sm.step(t, mu_t, mu_g, sigma_tcn=sig_t, sigma_gcn=sig_g, lying=lying, motion=motion)
                else:
                    pt = predict_prob(model_tcn, "tcn", Xt, device, two_stream_tcn)
                    if gate_gcn and (not in_confirm) and (pt < gate_gcn_tau):
                        pg = 0.0
                    else:
                        pg = predict_prob(model_gcn, "gcn", Xg, device, two_stream_gcn)
                    evs = sm.step(t, pt, pg, sigma_tcn=None, sigma_gcn=None, lying=lying, motion=motion)

            else:
                # single-model modes
                raw = load_window_raw(p, fps_default=(fps_tcn if args.mode == "tcn" else fps_gcn))
                t = _get_t(raw.meta, args.time_mode)

                in_confirm = getattr(sm, "_state", "idle") == "confirm"
                lying = motion = None
                if in_confirm and bool(single_cfg.confirm_use_scores):
                    lying, motion = compute_confirm_scores(raw)

                if args.mode == "tcn":
                    X, _ = build_input_from_raw(raw, feat_tcn, "tcn", two_stream=two_stream_tcn)
                    if mc_M_confirm > 1 and in_confirm:
                        mu, sigma = predict_mu_sigma(model_tcn, "tcn", X, device, two_stream_tcn, mc_M_confirm)
                        evs = sm.step(t, mu, sigma=sigma, lying=lying, motion=motion)
                    else:
                        p1 = predict_prob(model_tcn, "tcn", X, device, two_stream_tcn)
                        evs = sm.step(t, p1, sigma=None, lying=lying, motion=motion)
                else:
                    X, _ = build_input_from_raw(raw, feat_gcn, "gcn", two_stream=two_stream_gcn)
                    if mc_M_confirm > 1 and in_confirm:
                        mu, sigma = predict_mu_sigma(model_gcn, "gcn", X, device, two_stream_gcn, mc_M_confirm)
                        evs = sm.step(t, mu, sigma=sigma, lying=lying, motion=motion)
                    else:
                        p1 = predict_prob(model_gcn, "gcn", X, device, two_stream_gcn)
                        evs = sm.step(t, p1, sigma=None, lying=lying, motion=motion)

            for e in evs:
                rec = {"video_id": vid, "kind": e.kind, "t_sec": e.t_sec, **e.info}
                events_out.append(rec)
                info_str = ", ".join([f"{k}={v}" for k, v in e.info.items()])
                print(f"[{_tag(e.kind)}] vid={vid} t={e.t_sec:.2f}s {info_str}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(events_out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
