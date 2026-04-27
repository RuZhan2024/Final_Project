#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/run_modes.py

Offline deploy-time runner for *triage state machines* (Mode 1/2/3):

- Mode 1: TCN-only triage
- Mode 2: GCN-only triage
- Mode 3: Dual-model triage (TCN + GCN)

This script is used to sanity-check real-time behaviour on window streams.

Key fixes vs previous version:
- Single NPZ load per window across all modes (no double I/O)
- Centralised two-stream splitting logic (shared with evaluation)
- Optional lightweight confirm gate using skeleton-derived scores:
    - lying_score (torso horizontalness)
    - motion_score (post-impact stillness proxy)
"""

from __future__ import annotations

import argparse
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

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
    build_dual_inputs_from_raw,
    compute_confirm_scores,
    predict_prob,
    predict_mu_sigma,
)


def iter_window_raws(win_dir: str, *, fps_default: float) -> Dict[str, List[Any]]:
    """Group decoded window payloads by video id and sort by w_start."""
    paths = sorted(glob.glob(f"{win_dir}/**/*.npz", recursive=True))

    by_vid: Dict[str, List[Tuple[int, Any]]] = {}
    for p in paths:
        try:
            raw = load_window_raw(p, fps_default=fps_default)
            vid = str(raw.meta.video_id)
            ws = int(raw.meta.w_start)
        except Exception:
            continue
        by_vid.setdefault(vid, []).append((ws, raw))

    out: Dict[str, List[Any]] = {}
    for vid, lst in by_vid.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        out[vid] = [raw for _, raw in lst_sorted]
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

    def _score_window(model, arch: str, X, two_stream: bool, in_confirm: bool) -> Tuple[float, Optional[float]]:
        """Return (p_or_mu, sigma_or_none) for one model/window."""
        if mc_M_confirm > 1 and in_confirm:
            mu, sigma = predict_mu_sigma(model, arch, X, device, two_stream, mc_M_confirm)
            return float(mu), float(sigma)
        p1 = predict_prob(model, arch, X, device, two_stream)
        return float(p1), None

    mode_fps_default = float(fps_tcn if args.mode in {"tcn", "dual"} else fps_gcn)
    by_vid = iter_window_raws(args.win_dir, fps_default=mode_fps_default)
    events_out: List[Dict[str, Any]] = []

    for vid, raws in by_vid.items():
        if args.mode == "tcn":
            sm = SingleTriageStateMachine(tcn_tri, single_cfg)
        elif args.mode == "gcn":
            sm = SingleTriageStateMachine(gcn_tri, single_cfg)
        else:
            sm = DualTriageStateMachine(tcn_tri, gcn_tri, dual_cfg)

        for raw in raws:
            # In dual mode we load once and build both inputs from the same raw arrays.
            if args.mode == "dual":
                Xt, Xg, _ = build_dual_inputs_from_raw(
                    raw,
                    feat_tcn,
                    feat_gcn,
                    two_stream_gcn=two_stream_gcn,
                )
                t = _get_t(raw.meta, args.time_mode)

                in_confirm = getattr(sm, "_state", "idle") == "confirm"
                lying = motion = None
                if in_confirm and bool(dual_cfg.confirm_use_scores):
                    lying, motion = compute_confirm_scores(raw)

                pt, sig_t = _score_window(model_tcn, "tcn", Xt, two_stream_tcn, in_confirm)
                if gate_gcn and (not in_confirm) and (pt < gate_gcn_tau):
                    pg, sig_g = 0.0, None
                else:
                    pg, sig_g = _score_window(model_gcn, "gcn", Xg, two_stream_gcn, in_confirm)
                evs = sm.step(t, pt, pg, sigma_tcn=sig_t, sigma_gcn=sig_g, lying=lying, motion=motion)

            else:
                # single-model modes
                t = _get_t(raw.meta, args.time_mode)

                in_confirm = getattr(sm, "_state", "idle") == "confirm"
                lying = motion = None
                if in_confirm and bool(single_cfg.confirm_use_scores):
                    lying, motion = compute_confirm_scores(raw)

                if args.mode == "tcn":
                    X, _ = build_input_from_raw(raw, feat_tcn, "tcn", two_stream=two_stream_tcn)
                    p1, sigma = _score_window(model_tcn, "tcn", X, two_stream_tcn, in_confirm)
                    evs = sm.step(t, p1, sigma=sigma, lying=lying, motion=motion)
                else:
                    X, _ = build_input_from_raw(raw, feat_gcn, "gcn", two_stream=two_stream_gcn)
                    p1, sigma = _score_window(model_gcn, "gcn", X, two_stream_gcn, in_confirm)
                    evs = sm.step(t, p1, sigma=sigma, lying=lying, motion=motion)

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
