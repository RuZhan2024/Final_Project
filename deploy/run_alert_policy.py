#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy/run_alert_policy.py

Offline runner for the alert policy (core.alerting.detect_alert_events).

IMPORTANT:
- detect_alert_events signature is (probs, times_s, cfg) and returns (active_mask, events).
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from core.models import pick_device
from core.yamlio import yaml_load_simple
from core.alerting import AlertCfg, detect_alert_events

from deploy.common import (
    load_model_bundle,
    load_window_raw,
    build_input_from_raw,
    compute_confirm_scores,
    predict_prob,
)


def iter_windows(win_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    # Recursive so nested window dirs work.
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

    for vid in list(by_vid.keys()):
        by_vid[vid] = sorted(by_vid[vid], key=lambda x: x[0])
    return by_vid


def _get_t(meta, time_mode: str) -> float:
    ws = float(meta.w_start)
    we = float(meta.w_end)  # inclusive frame index
    fps = float(meta.fps) if meta.fps and meta.fps > 0 else 30.0
    if time_mode == "start":
        return ws / fps
    if time_mode == "end":
        return we / fps
    return 0.5 * (ws + we) / fps


def load_alert_cfg(path: str) -> AlertCfg:
    d = yaml_load_simple(path) or {}
    if "alert" in d and isinstance(d["alert"], dict):
        d = d["alert"]
    return AlertCfg.from_dict(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alert_cfg", required=True)
    ap.add_argument("--device", default="", help="cpu|cuda|mps (default: auto)")
    ap.add_argument("--time_mode", default="center", choices=["start", "center", "end"])
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    alert_cfg = load_alert_cfg(args.alert_cfg)

    model, arch_ckpt, _m_cfg, feat_cfg, two_stream, fps_default = load_model_bundle(args.ckpt, device)
    if arch_ckpt != args.arch:
        print(f"[warn] ckpt arch={arch_ckpt} (expected {args.arch})")

    by_vid = iter_windows(args.win_dir)
    events_out: List[Dict[str, Any]] = []

    for vid, lst in by_vid.items():
        ts: List[float] = []
        ps: List[float] = []
        lying: List[float] = []
        motion: List[float] = []

        for _, p in lst:
            raw = load_window_raw(p, fps_default=fps_default)
            X, _ = build_input_from_raw(raw, feat_cfg, args.arch, two_stream=two_stream)
            t = _get_t(raw.meta, args.time_mode)
            prob = predict_prob(model, args.arch, X, device, two_stream)

            ts.append(t)
            ps.append(prob)

            ls, ms = compute_confirm_scores(raw)
            lying.append(np.nan if ls is None else float(ls))
            motion.append(np.nan if ms is None else float(ms))

        ts_a = np.asarray(ts, dtype=np.float32)
        ps_a = np.asarray(ps, dtype=np.float32)
        lying_a = np.asarray(lying, dtype=np.float32)
        motion_a = np.asarray(motion, dtype=np.float32)

        # Correct order + correct unpack
        if alert_cfg.confirm:
            _active, evs = detect_alert_events(ps_a, ts_a, alert_cfg, lying_score=lying_a, motion_score=motion_a)
        else:
            _active, evs = detect_alert_events(ps_a, ts_a, alert_cfg, lying_score=None, motion_score=None)

        for e in evs:
            rec = {"video_id": vid, "kind": e.kind, "t_sec": e.t_sec, **e.info}
            events_out.append(rec)
            info_str = ", ".join([f"{k}={v}" for k, v in e.info.items()])
            print(f"[{e.kind}] vid={vid} t={e.t_sec:.2f}s {info_str}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(events_out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
