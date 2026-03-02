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
import glob
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from fall_detection.core.models import pick_device
from fall_detection.core.yamlio import yaml_load_simple
from fall_detection.core.alerting import AlertCfg, detect_alert_events

from fall_detection.deploy.common import (
    load_model_bundle,
    load_window_raw,
    build_input_from_raw,
    compute_confirm_scores,
    predict_prob,
)


def list_windows(win_dir: str) -> List[str]:
    # Recursive so nested window dirs work.
    return sorted(glob.glob(f"{win_dir}/**/*.npz", recursive=True))


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

    paths = list_windows(args.win_dir)
    events_out: List[Dict[str, Any]] = []
    # Single-pass inference + grouping to avoid opening each NPZ twice.
    by_vid: Dict[str, List[Tuple[Any, ...]]] = defaultdict(list)
    for p in paths:
        raw = load_window_raw(p, fps_default=fps_default)
        X, _ = build_input_from_raw(raw, feat_cfg, args.arch, two_stream=two_stream)
        t = _get_t(raw.meta, args.time_mode)
        prob = predict_prob(model, args.arch, X, device, two_stream)
        ws = int(raw.meta.w_start)
        vid = str(raw.meta.video_id)

        if alert_cfg.confirm:
            ls, ms = compute_confirm_scores(raw)
            ls_v = np.nan if ls is None else float(ls)
            ms_v = np.nan if ms is None else float(ms)
            by_vid[vid].append((ws, t, float(prob), float(ls_v), float(ms_v)))
        else:
            by_vid[vid].append((ws, t, float(prob)))

    for vid, rows in by_vid.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        n = len(rows)
        ts_a = np.fromiter((r[1] for r in rows), dtype=np.float32, count=n)
        ps_a = np.fromiter((r[2] for r in rows), dtype=np.float32, count=n)
        if alert_cfg.confirm:
            ls_a = np.fromiter((r[3] for r in rows), dtype=np.float32, count=n)
            ms_a = np.fromiter((r[4] for r in rows), dtype=np.float32, count=n)
            _active, evs = detect_alert_events(ps_a, ts_a, alert_cfg, lying_score=ls_a, motion_score=ms_a)
        else:
            _active, evs = detect_alert_events(ps_a, ts_a, alert_cfg, lying_score=None, motion_score=None)

        for e in evs:
            rec = {"video_id": vid, **e.to_dict()}
            events_out.append(rec)
            print(f"[ALERT] vid={vid} start={rec['start_time_s']:.2f}s end={rec['end_time_s']:.2f}s peak_p={rec['peak_p']:.3f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(events_out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
