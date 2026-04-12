#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fall_detection.core.alerting import AlertCfg, classify_states, detect_alert_events, times_from_windows
from fall_detection.core.ckpt import get_cfg, load_ckpt
from fall_detection.core.confirm import confirm_scores_window
from fall_detection.core.features import FeatCfg, build_canonical_input, build_tcn_input, read_window_npz, split_gcn_two_stream
from fall_detection.core.models import build_model, logits_1d, p_fall_from_logits, pick_device


def _collect_video_windows(win_dir: Path, video_id: str, fps_default: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(win_dir.glob("*.npz")):
        try:
            joints, motion, conf, mask, fps, meta = read_window_npz(str(fp), fps_default=fps_default)
        except Exception:
            continue
        vid = str(meta.video_id or "")
        if vid != video_id:
            continue
        rows.append(
            {
                "path": str(fp),
                "joints": joints,
                "motion": motion,
                "conf": conf,
                "mask": mask,
                "fps": float(fps),
                "w_start": int(meta.w_start),
                "w_end": int(meta.w_end),
            }
        )
    rows.sort(key=lambda r: (r["w_start"], r["w_end"]))
    return rows


def _infer_probs(
    rows: list[dict[str, Any]],
    arch: str,
    model: torch.nn.Module,
    feat_cfg: FeatCfg,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs: list[float] = []
    lying: list[float] = []
    motion_s: list[float] = []

    model.eval()
    with torch.no_grad():
        for r in rows:
            X, m = build_canonical_input(
                joints_xy=r["joints"],
                motion_xy=r["motion"],
                conf=r["conf"],
                mask=r["mask"],
                fps=float(r["fps"]),
                feat_cfg=feat_cfg,
            )
            ls, ms = confirm_scores_window(r["joints"], m, fps=float(r["fps"]), tail_s=1.0)
            ls = float(ls) if np.isfinite(ls) else 0.0
            ms = float(ms) if np.isfinite(ms) else float("inf")
            lying.append(ls)
            motion_s.append(ms)

            if arch == "tcn":
                x = build_tcn_input(X, feat_cfg)
                xb = torch.from_numpy(x).to(torch.float32).unsqueeze(0).to(device)
                p = float(p_fall_from_logits(model(xb)).detach().cpu().numpy().reshape(-1)[0])
            else:
                two_stream = bool(getattr(model, "j_enc", None) is not None and getattr(model, "m_enc", None) is not None)
                if two_stream:
                    xj, xm = split_gcn_two_stream(X, feat_cfg)
                    xjb = torch.from_numpy(xj).to(torch.float32).unsqueeze(0).to(device)
                    xmb = torch.from_numpy(xm).to(torch.float32).unsqueeze(0).to(device)
                    p = float(p_fall_from_logits(model(xjb, xmb)).detach().cpu().numpy().reshape(-1)[0])
                else:
                    xb = torch.from_numpy(X).to(torch.float32).unsqueeze(0).to(device)
                    p = float(p_fall_from_logits(model(xb)).detach().cpu().numpy().reshape(-1)[0])
            probs.append(p)

    return (
        np.asarray(probs, dtype=np.float32),
        np.asarray(lying, dtype=np.float32),
        np.asarray(motion_s, dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--fps_default", type=float, default=25.0)

    ap.add_argument("--ema_alpha", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tau_high", type=float, required=True)
    ap.add_argument("--tau_low", type=float, required=True)
    ap.add_argument("--cooldown_s", type=float, default=30.0)
    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=2.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.65)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=int, default=1)
    ap.add_argument("--start_guard_max_lying", type=float, default=-1.0)
    ap.add_argument("--start_guard_prefixes", type=str, default="")
    args = ap.parse_args()

    win_dir = Path(args.win_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_ckpt(args.ckpt, map_location="cpu")
    arch, model_cfg, feat_cfg, data_cfg = get_cfg(bundle)
    fps_default = float(data_cfg.get("fps_default", args.fps_default))
    device = pick_device()

    feat_cfg_obj = FeatCfg.from_dict(feat_cfg)
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default)
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.to(device)

    rows = _collect_video_windows(win_dir, args.video_id, fps_default=fps_default)
    if not rows:
        raise SystemExit(f"[err] no windows found for video_id={args.video_id}")

    probs, lying, motion_s = _infer_probs(rows, arch, model, feat_cfg_obj, device)
    ws = np.asarray([r["w_start"] for r in rows], dtype=np.int32)
    we = np.asarray([r["w_end"] for r in rows], dtype=np.int32)
    fps = float(np.median(np.asarray([r["fps"] for r in rows], dtype=np.float32)))
    t = times_from_windows(ws, we, fps, mode="center")

    alert_cfg = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=float(args.tau_high),
        tau_low=float(args.tau_low),
        cooldown_s=float(args.cooldown_s),
        confirm=bool(int(args.confirm)),
        confirm_s=float(args.confirm_s),
        confirm_min_lying=float(args.confirm_min_lying),
        confirm_max_motion=float(args.confirm_max_motion),
        confirm_require_low=bool(int(args.confirm_require_low)),
        start_guard_max_lying=(None if float(args.start_guard_max_lying) < 0.0 else float(args.start_guard_max_lying)),
        start_guard_prefixes=([x.strip() for x in str(args.start_guard_prefixes).split(",") if x.strip()] or None),
    )
    alert_mask, events = detect_alert_events(
        probs, t, alert_cfg, lying_score=lying, motion_score=motion_s, video_id=str(args.video_id)
    )
    state_obj = classify_states(
        probs, t, alert_cfg, lying_score=lying, motion_score=motion_s, video_id=str(args.video_id)
    )
    clear_m = np.asarray(state_obj.get("clear", np.zeros_like(probs, dtype=bool)), dtype=bool)
    suspect_m = np.asarray(state_obj.get("suspect", np.zeros_like(probs, dtype=bool)), dtype=bool)
    alert_m = np.asarray(state_obj.get("alert", np.zeros_like(probs, dtype=bool)), dtype=bool)

    window_rows: list[dict[str, Any]] = []
    for i in range(len(rows)):
        window_rows.append(
            {
                "idx": int(i),
                "path": rows[i]["path"],
                "w_start": int(ws[i]),
                "w_end": int(we[i]),
                "t_center_s": float(t[i]),
                "p_fall": float(probs[i]),
                "lying_score": float(lying[i]),
                "motion_score": float(motion_s[i]),
                "state": ("alert" if bool(alert_m[i]) else ("suspect" if bool(suspect_m[i]) else "clear")),
                "alert_active": bool(alert_mask[i]),
            }
        )

    payload = {
        "video_id": args.video_id,
        "arch": str(arch),
        "ckpt": str(args.ckpt),
        "win_dir": str(win_dir),
        "fps": float(fps),
        "n_windows": int(len(rows)),
        "alert_cfg": asdict(alert_cfg),
        "events": [
            {
                "start_idx": int(e.start_idx),
                "end_idx": int(e.end_idx),
                "start_time_s": float(e.start_time_s),
                "end_time_s": float(e.end_time_s),
                "peak_p": float(e.peak_p),
            }
            for e in events
        ],
        "window_trace": window_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_json}")

    out_csv = args.out_csv.strip()
    if out_csv:
        out_csv_p = Path(out_csv)
        out_csv_p.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "idx",
            "w_start",
            "w_end",
            "t_center_s",
            "p_fall",
            "lying_score",
            "motion_score",
            "state",
            "alert_active",
            "path",
        ]
        with out_csv_p.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in window_rows:
                wr.writerow(r)
        print(f"[ok] wrote: {out_csv_p}")


if __name__ == "__main__":
    main()
