#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/score_unlabeled_alert_rate.py

Score alert rate on UNLABELED (or negative-only) windows.

Why this exists
---------------
To estimate deployment false alarms realistically, you often need long "normal life"
footage with no falls. Many such datasets are unlabeled.

So instead of recall/F1, we measure:
- false alerts per hour/day
- how often the system enters SUSPECT / ALERT state

This script:
1) loads a checkpoint (TCN or GCN)
2) runs inference on windows_dir
3) applies temperature scaling
4) applies alert policy (core/alerting)
5) counts alert events per duration

Important assumption
--------------------
- windows_dir should contain windows covering full clips with your deploy stride S.
- If windows are mined/subsampled, duration estimation is not reliable, so FA/day is not reliable.

Outputs
-------
Writes a JSON file with:
- aggregated FA/hour, FA/day
- state occupancy totals
- per-video breakdown (duration, alerts, state counts)
"""

from __future__ import annotations

# ------------------------------------------------------------
# Path bootstrap so running directly works:
#   python eval/score_unlabeled_alert_rate.py ...
# ------------------------------------------------------------
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.alerting import AlertCfg, classify_states, detect_alert_events, times_from_windows
from core.calibration import apply_temperature, load_temperature
from core.ckpt import get_cfg, get_state_dict, load_ckpt
from core.features import FeatCfg, build_gcn_input, build_tcn_input, read_window_npz
from core.models import build_model, logits_1d, pick_device
from core.signals import compute_window_signals
from core.yamlio import yaml_load_simple


# ============================================================
# 1) Stable sigmoid
# ============================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Dataset
# ============================================================
@dataclass
class MetaRow:
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float


class UnlabeledWindows(Dataset):
    """
    Loads window NPZ files, returns model input + meta + signals.

    We ignore y completely because this is intended for unlabeled streams.
    """

    def __init__(self, root: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool):
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {root}")

        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower().strip()
        self.two_stream = bool(two_stream)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)

        # Signals computed from same representation as model consumes
        sig = compute_window_signals(joints, motion, conf, mask, float(fps), self.feat_cfg)

        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)
        else:
            X, _ = build_gcn_input(joints, motion, conf, mask, float(fps), self.feat_cfg)
            if self.two_stream:
                xy = X[..., 0:2]
                if self.feat_cfg.use_conf_channel:
                    c = X[..., -1:]
                    xj = np.concatenate([xy, c], axis=-1)
                else:
                    xj = xy

                if self.feat_cfg.use_motion and X.shape[-1] >= 4:
                    xm = X[..., 2:4]
                else:
                    xm = np.zeros_like(xy, dtype=np.float32)

                X = (xj.astype(np.float32), xm.astype(np.float32))

        vid = meta.video_id or meta.seq_id or os.path.splitext(os.path.basename(p))[0]
        m = MetaRow(
            path=str(p),
            video_id=str(vid),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
        )

        return X, m, np.float32(sig.quality), np.float32(sig.lying), np.float32(sig.motion)


def _collate(batch):
    Xs, metas, qs, ls, ms = zip(*batch)
    return (
        list(Xs),
        list(metas),
        np.asarray(qs, dtype=np.float32),
        np.asarray(ls, dtype=np.float32),
        np.asarray(ms, dtype=np.float32),
    )


# ============================================================
# 3) Inference
# ============================================================
@torch.no_grad()
def infer_logits(model, loader, device, arch: str, two_stream: bool):
    arch = str(arch).lower().strip()

    logits_all: List[np.ndarray] = []
    vids: List[str] = []
    ws: List[int] = []
    we: List[int] = []
    fps_list: List[float] = []
    q_all: List[np.ndarray] = []
    l_all: List[np.ndarray] = []
    m_all: List[np.ndarray] = []

    for Xs, metas, qs, ls, ms in loader:
        if arch == "tcn":
            xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)
            logits = logits_1d(model(xb))
        else:
            if two_stream:
                xj = torch.from_numpy(np.stack([x[0] for x in Xs], axis=0)).to(device)
                xm = torch.from_numpy(np.stack([x[1] for x in Xs], axis=0)).to(device)
                logits = logits_1d(model(xj, xm))
            else:
                xb = torch.from_numpy(np.stack(Xs, axis=0)).to(device)
                logits = logits_1d(model(xb))

        logits_all.append(logits.detach().cpu().numpy().reshape(-1))

        q_all.append(qs.reshape(-1))
        l_all.append(ls.reshape(-1))
        m_all.append(ms.reshape(-1))

        for meta in metas:
            vids.append(meta.video_id)
            ws.append(int(meta.w_start))
            we.append(int(meta.w_end))
            fps_list.append(float(meta.fps))

    return (
        np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32),
        vids,
        ws,
        we,
        fps_list,
        np.concatenate(q_all) if q_all else np.array([], dtype=np.float32),
        np.concatenate(l_all) if l_all else np.array([], dtype=np.float32),
        np.concatenate(m_all) if m_all else np.array([], dtype=np.float32),
    )


# ============================================================
# 4) Alert-rate aggregation per video
# ============================================================
def score_alert_rate(
    probs: np.ndarray,
    vids: List[str],
    ws: List[int],
    we: List[int],
    fps_list: List[float],
    qs: np.ndarray,
    ls: np.ndarray,
    ms: np.ndarray,
    *,
    alert_cfg: AlertCfg,
    time_mode: str,
    fps_default: float,
) -> Dict[str, Any]:
    vids_arr = np.asarray(vids, dtype=str)
    ws_arr = np.asarray(ws, dtype=np.int32)
    we_arr = np.asarray(we, dtype=np.int32)
    fps_arr = np.asarray(fps_list, dtype=np.float32)

    unique_vids = list(dict.fromkeys(list(vids_arr)))

    total_duration_s = 0.0
    total_alerts = 0

    state_totals = {"n_windows": 0, "clear": 0, "suspect": 0, "pending": 0, "alert": 0}
    per_video: Dict[str, Any] = {}

    for v in unique_vids:
        mv = vids_arr == v
        if not mv.any():
            continue

        idx = np.argsort(ws_arr[mv])

        p_v = probs[mv][idx]
        ws_v = ws_arr[mv][idx]
        we_v = we_arr[mv][idx]

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        fps_v = fps_v if fps_v > 0 else float(fps_default)

        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)

        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)

        q_v = qs[mv][idx] if qs is not None else None
        l_v = ls[mv][idx] if ls is not None else None
        m_v = ms[mv][idx] if ms is not None else None

        # Alert events (these are all "false alerts" by assumption)
        _, events = detect_alert_events(
            p_v,
            t_v,
            alert_cfg,
            lying_score=l_v,
            motion_score=m_v,
            quality_score=q_v,
        )

        # State occupancy (clear/suspect/alert)
        st = classify_states(
            p_v,
            t_v,
            alert_cfg,
            lying_score=l_v,
            motion_score=m_v,
            quality_score=q_v,
        )

        n_alert = int(len(events))
        total_alerts += n_alert

        per_video[v] = {
            "duration_s": float(duration_s),
            "n_alert_events": int(n_alert),
            "alerts": [e.to_dict() for e in events],
            "state_counts": {
                "n_windows": int(t_v.size),
                "clear": int(np.sum(st["clear"])),
                "suspect": int(np.sum(st["suspect"])),
                "pending": int(np.sum(st.get("pending", np.zeros_like(st["alert"], dtype=bool)))),
                "alert": int(np.sum(st["alert"])),
            },
        }

        state_totals["n_windows"] += int(t_v.size)
        state_totals["clear"] += int(np.sum(st["clear"]))
        state_totals["suspect"] += int(np.sum(st["suspect"]))
        state_totals["pending"] += int(np.sum(st.get("pending", np.zeros_like(st["alert"], dtype=bool))))
        state_totals["alert"] += int(np.sum(st["alert"]))

    dur_h = total_duration_s / 3600.0 if total_duration_s > 0 else float("nan")
    dur_d = total_duration_s / 86400.0 if total_duration_s > 0 else float("nan")
    fa_h = float(total_alerts / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
    fa_d = float(total_alerts / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

    return {
        "summary": {
            "n_alert_events": int(total_alerts),
            "total_duration_s": float(total_duration_s),
            "false_alerts_per_hour": float(fa_h),
            "false_alerts_per_day": float(fa_d),
        },
        "state_totals": state_totals,
        "per_video": per_video,
    }


# ============================================================
# 5) Main CLI
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", choices=["tcn", "gcn"], required=True)
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prefer_ema", type=int, default=1)

    # Optional ops yaml + op selection
    ap.add_argument("--ops_yaml", default="")
    ap.add_argument("--op", choices=["op1", "op2", "op3"], default="op2")

    # Calibration
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--calibration_yaml", default="")

    # Alert cfg overrides (if not using ops_yaml)
    ap.add_argument("--tau_high", type=float, default=0.0)
    ap.add_argument("--tau_low", type=float, default=0.0)
    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cooldown_s", type=float, default=30.0)

    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)

    ap.add_argument("--quality_adapt", type=int, default=0)
    ap.add_argument("--quality_min", type=float, default=0.0)
    ap.add_argument("--quality_boost", type=float, default=0.15)
    ap.add_argument("--quality_boost_low", type=float, default=0.05)

    ap.add_argument("--time_mode", choices=["start", "center", "end"], default="center")
    ap.add_argument("--fps_default", type=float, default=30.0)

    args = ap.parse_args()
    arch = str(args.arch).lower().strip()

    ops = None
    if str(args.ops_yaml).strip():
        ops = yaml_load_simple(args.ops_yaml)

    # Load ckpt bundle + model rebuild
    bundle = load_ckpt(args.ckpt, map_location="cpu")
    model_cfg = get_cfg(bundle, "model_cfg", default={})

    raw_feat = get_cfg(bundle, "feat_cfg", default={})
    if hasattr(raw_feat, "to_dict"):
        try:
            raw_feat = raw_feat.to_dict()
        except Exception:
            pass
    feat_cfg = FeatCfg.from_dict(raw_feat if isinstance(raw_feat, dict) else {})

    two_stream = bool(model_cfg.get("two_stream", False)) if arch == "gcn" else False
    fps_default = float(get_cfg(bundle, "data_cfg", default={}).get("fps_default", args.fps_default))

    device = pick_device()
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default).to(device)

    # EMA-safe load
    sd = get_state_dict(bundle, prefer_ema=False)
    if bool(int(args.prefer_ema)):
        ema_sd = bundle.get("ema_state_dict", None) or bundle.get("ema", None)
        if isinstance(ema_sd, dict) and len(ema_sd) > 0:
            sd = dict(sd)
            sd.update(ema_sd)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Temperature scaling
    if float(args.temperature) > 0:
        T = float(args.temperature)
        cal_source = "cli"
    elif str(args.calibration_yaml).strip():
        T = load_temperature(str(args.calibration_yaml).strip(), default=1.0)
        cal_source = "yaml"
    elif isinstance(ops, dict) and isinstance(ops.get("calibration"), dict) and "temperature" in ops["calibration"]:
        T = float(ops["calibration"]["temperature"])
        cal_source = "ops_yaml"
    else:
        T = 1.0
        cal_source = "none"

    # Alert config base
    if isinstance(ops, dict) and isinstance(ops.get("alert_cfg"), dict):
        alert_cfg = AlertCfg.from_dict(ops["alert_cfg"])
    else:
        alert_cfg = AlertCfg(
            ema_alpha=float(args.ema_alpha),
            k=int(args.k),
            n=int(args.n),
            tau_high=0.90,
            tau_low=0.70,
            cooldown_s=float(args.cooldown_s),
            confirm=bool(int(args.confirm)),
            confirm_s=float(args.confirm_s),
            confirm_min_lying=float(args.confirm_min_lying),
            confirm_max_motion=float(args.confirm_max_motion),
            confirm_require_low=bool(int(args.confirm_require_low)),
            quality_adapt=bool(int(args.quality_adapt)),
            quality_min=float(args.quality_min),
            quality_boost=float(args.quality_boost),
            quality_boost_low=float(args.quality_boost_low),
        )

    # Override thresholds from ops_yaml selected op
    if isinstance(ops, dict) and isinstance(ops.get("ops"), dict) and isinstance(ops["ops"].get(args.op), dict):
        oc = ops["ops"][args.op]
        d = alert_cfg.to_dict()
        d["tau_high"] = float(oc.get("tau_high", d.get("tau_high")))
        d["tau_low"] = float(oc.get("tau_low", d.get("tau_low")))
        alert_cfg = AlertCfg.from_dict(d)

    # CLI explicit overrides
    if float(args.tau_high) > 0:
        d = alert_cfg.to_dict()
        d["tau_high"] = float(args.tau_high)
        alert_cfg = AlertCfg.from_dict(d)
    if float(args.tau_low) > 0:
        d = alert_cfg.to_dict()
        d["tau_low"] = float(args.tau_low)
        alert_cfg = AlertCfg.from_dict(d)

    # Dataset + inference
    ds = UnlabeledWindows(args.windows_dir, feat_cfg, fps_default, arch, two_stream)
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=0, collate_fn=_collate)

    logits, vids, ws, we, fps_list, qs, ls, ms = infer_logits(model, loader, device, arch, two_stream)

    probs = _sigmoid_np(apply_temperature(logits, float(T)))

    report = score_alert_rate(
        probs,
        vids,
        ws,
        we,
        fps_list,
        qs,
        ls,
        ms,
        alert_cfg=alert_cfg,
        time_mode=str(args.time_mode),
        fps_default=float(fps_default),
    )

    out = {
        "arch": arch,
        "ckpt": str(args.ckpt),
        "windows_dir": str(args.windows_dir),
        "prefer_ema": bool(int(args.prefer_ema)),
        "calibration": {"temperature": float(T), "source": str(cal_source)},
        "alert_cfg": alert_cfg.to_dict(),
        "time_mode": str(args.time_mode),
        **report,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    s = report["summary"]
    print(
        f"[ok] wrote: {args.out_json}\n"
        f"[summary] alerts={s['n_alert_events']} FA/day={s['false_alerts_per_day']:.3f} "
        f"duration_s={s['total_duration_s']:.1f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
