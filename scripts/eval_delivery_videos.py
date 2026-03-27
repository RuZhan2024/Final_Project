#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fall_detection.core.alerting import AlertCfg, detect_alert_events, times_from_windows
from fall_detection.core.ckpt import get_cfg, load_ckpt
from fall_detection.core.confirm import confirm_scores_window
from fall_detection.core.features import FeatCfg, build_canonical_input, build_tcn_input, read_window_npz, split_gcn_two_stream
from fall_detection.core.models import build_model, p_fall_from_logits, pick_device


def _npz_scalar_str(path: str, key: str, default: str = "") -> str:
    try:
        with np.load(path, allow_pickle=False) as z:
            if key not in z.files:
                return default
            arr = z[key]
            if isinstance(arr, np.ndarray) and arr.shape == ():
                return str(arr.item())
            if isinstance(arr, np.ndarray) and arr.size == 1:
                return str(arr.reshape(-1)[0].item())
            return str(arr)
    except Exception:
        return default


def _truth_from_src(src: str, fall_contains: list[str], nonfall_contains: list[str]) -> int | None:
    src_l = src.lower()
    for needle in nonfall_contains:
        if needle.lower() in src_l:
            return 0
    for needle in fall_contains:
        if needle.lower() in src_l:
            return 1
    return None


def _build_alert_cfg(ops_obj: dict[str, Any], op_code: str) -> AlertCfg:
    op = ops_obj["ops"][op_code]
    alert_base = ops_obj["alert_base"]
    return AlertCfg(
        ema_alpha=float(alert_base["ema_alpha"]),
        k=int(alert_base["k"]),
        n=int(alert_base["n"]),
        tau_high=float(op["tau_high"]),
        tau_low=float(op["tau_low"]),
        cooldown_s=float(alert_base["cooldown_s"]),
        confirm=bool(alert_base["confirm"]),
        confirm_s=float(alert_base["confirm_s"]),
        confirm_min_lying=float(alert_base["confirm_min_lying"]),
        confirm_max_motion=float(alert_base["confirm_max_motion"]),
        confirm_require_low=bool(alert_base["confirm_require_low"]),
        start_guard_max_lying=alert_base.get("start_guard_max_lying"),
        start_guard_prefixes=alert_base.get("start_guard_prefixes"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate deliverable video-level metrics on window NPZs.")
    ap.add_argument("--config_yaml", default="", help="Optional delivery config YAML.")
    ap.add_argument("--windows_dir", default="", help="Directory containing window NPZs.")
    ap.add_argument("--ops_yaml", default="", help="Ops YAML with model checkpoint and alert settings.")
    ap.add_argument("--op_code", default="OP2", help="Operating point inside ops_yaml.")
    ap.add_argument("--out_prefix", default="", help="Output prefix for .csv/.json metrics.")
    ap.add_argument("--fall_contains", nargs="+", default=[], help="Path substrings that identify fall videos.")
    ap.add_argument("--nonfall_contains", nargs="+", default=[], help="Path substrings that identify non-fall videos.")
    ap.add_argument("--fps_default", type=float, default=15.0)
    ap.add_argument("--override_ema_alpha", type=float, default=-1.0)
    ap.add_argument("--override_tau_high", type=float, default=-1.0)
    ap.add_argument("--override_tau_low", type=float, default=-1.0)
    ap.add_argument("--override_k", type=int, default=-1)
    ap.add_argument("--override_n", type=int, default=-1)
    ap.add_argument("--gate_max_lying", type=float, default=-1.0, help="Reject predicted fall if max lying exceeds this.")
    ap.add_argument("--gate_min_mean_motion_high", type=float, default=-1.0, help="Reject predicted fall if mean motion over high-prob windows is below this.")
    ap.add_argument("--gate_max_event_start_s", type=float, default=-1.0, help="Reject predicted fall if the first event starts later than this.")
    args = ap.parse_args()

    cfg_obj: dict[str, Any] = {}
    if args.config_yaml:
        cfg_obj = yaml.safe_load(Path(args.config_yaml).read_text(encoding="utf-8")) or {}

    windows_dir = Path(args.windows_dir or cfg_obj.get("windows_dir", ""))
    ops_path = Path(args.ops_yaml or cfg_obj.get("ops_yaml", ""))
    out_prefix = Path(args.out_prefix or cfg_obj.get("out_prefix", ""))
    if not str(windows_dir):
        raise SystemExit("[err] --windows_dir or config_yaml:windows_dir is required")
    if not str(ops_path):
        raise SystemExit("[err] --ops_yaml or config_yaml:ops_yaml is required")
    if not str(out_prefix):
        raise SystemExit("[err] --out_prefix or config_yaml:out_prefix is required")
    if not args.fall_contains:
        args.fall_contains = list(cfg_obj.get("fall_contains", []))
    if not args.nonfall_contains:
        args.nonfall_contains = list(cfg_obj.get("nonfall_contains", []))
    if not args.fall_contains or not args.nonfall_contains:
        raise SystemExit("[err] fall/non-fall path matchers are required")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    ops_obj = yaml.safe_load(ops_path.read_text(encoding="utf-8"))
    op_code = str(cfg_obj.get("op_code", args.op_code))
    alert_cfg = _build_alert_cfg(ops_obj, op_code)

    alert_override = dict(cfg_obj.get("alert_override", {}) or {})
    delivery_gate = dict(cfg_obj.get("delivery_gate", {}) or {})
    if args.override_ema_alpha < 0.0 and "ema_alpha" in alert_override:
        args.override_ema_alpha = float(alert_override["ema_alpha"])
    if args.override_tau_high < 0.0 and "tau_high" in alert_override:
        args.override_tau_high = float(alert_override["tau_high"])
    if args.override_tau_low < 0.0 and "tau_low" in alert_override:
        args.override_tau_low = float(alert_override["tau_low"])
    if args.override_k < 0 and "k" in alert_override:
        args.override_k = int(alert_override["k"])
    if args.override_n < 0 and "n" in alert_override:
        args.override_n = int(alert_override["n"])
    if args.gate_max_lying < 0.0 and "max_lying" in delivery_gate:
        args.gate_max_lying = float(delivery_gate["max_lying"])
    if args.gate_min_mean_motion_high < 0.0 and "min_mean_motion_high" in delivery_gate:
        args.gate_min_mean_motion_high = float(delivery_gate["min_mean_motion_high"])
    if args.gate_max_event_start_s < 0.0 and "max_event_start_s" in delivery_gate:
        args.gate_max_event_start_s = float(delivery_gate["max_event_start_s"])

    if args.override_ema_alpha >= 0.0:
        alert_cfg = AlertCfg(**{**alert_cfg.to_dict(), "ema_alpha": float(args.override_ema_alpha)})
    if args.override_tau_high >= 0.0:
        alert_cfg = AlertCfg(**{**alert_cfg.to_dict(), "tau_high": float(args.override_tau_high)})
    if args.override_tau_low >= 0.0:
        alert_cfg = AlertCfg(**{**alert_cfg.to_dict(), "tau_low": float(args.override_tau_low)})
    if args.override_k > 0:
        alert_cfg = AlertCfg(**{**alert_cfg.to_dict(), "k": int(args.override_k)})
    if args.override_n > 0:
        alert_cfg = AlertCfg(**{**alert_cfg.to_dict(), "n": int(args.override_n)})

    ckpt = (ops_path.parent / ops_obj["model"]["ckpt"]).resolve()
    bundle = load_ckpt(str(ckpt), map_location="cpu")
    arch, model_cfg, feat_cfg, data_cfg = get_cfg(bundle)
    feat_cfg_obj = FeatCfg.from_dict(feat_cfg)
    fps_default = float(data_cfg.get("fps_default", args.fps_default))
    model = build_model(arch, model_cfg, feat_cfg, fps_default=fps_default)
    model.load_state_dict(bundle["state_dict"], strict=False)
    device = pick_device()
    model.to(device)
    model.eval()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp in sorted(windows_dir.glob("*.npz")):
        joints, motion, conf, mask, fps, meta = read_window_npz(str(fp), fps_default=fps_default)
        groups[str(meta.video_id)].append(
            {
                "path": str(fp),
                "joints": joints,
                "motion": motion,
                "conf": conf,
                "mask": mask,
                "fps": float(fps),
                "w_start": int(meta.w_start),
                "w_end": int(meta.w_end),
                "src": _npz_scalar_str(str(fp), "src", ""),
            }
        )

    rows_out: list[dict[str, Any]] = []
    with torch.no_grad():
        for video_id, rows in sorted(groups.items()):
            rows.sort(key=lambda r: (r["w_start"], r["w_end"]))
            probs: list[float] = []
            lying: list[float] = []
            motion_s: list[float] = []
            ws: list[int] = []
            we: list[int] = []
            fpss: list[float] = []

            for r in rows:
                X, m = build_canonical_input(
                    joints_xy=r["joints"],
                    motion_xy=r["motion"],
                    conf=r["conf"],
                    mask=r["mask"],
                    fps=float(r["fps"]),
                    feat_cfg=feat_cfg_obj,
                )
                ls, ms = confirm_scores_window(r["joints"], m, fps=float(r["fps"]), tail_s=1.0)
                lying.append(float(ls) if np.isfinite(ls) else 0.0)
                motion_s.append(float(ms) if np.isfinite(ms) else float("inf"))
                if arch == "tcn":
                    x = build_tcn_input(X, feat_cfg_obj)
                    xb = torch.from_numpy(x).to(torch.float32).unsqueeze(0).to(device)
                    p = float(p_fall_from_logits(model(xb)).detach().cpu().numpy().reshape(-1)[0])
                else:
                    xj, xm = split_gcn_two_stream(X, feat_cfg_obj)
                    xjb = torch.from_numpy(xj).to(torch.float32).unsqueeze(0).to(device)
                    xmb = torch.from_numpy(xm).to(torch.float32).unsqueeze(0).to(device)
                    p = float(p_fall_from_logits(model(xjb, xmb)).detach().cpu().numpy().reshape(-1)[0])
                probs.append(p)
                ws.append(r["w_start"])
                we.append(r["w_end"])
                fpss.append(r["fps"])

            probs_a = np.asarray(probs, dtype=np.float32)
            lying_a = np.asarray(lying, dtype=np.float32)
            motion_a = np.asarray(motion_s, dtype=np.float32)
            t = times_from_windows(
                np.asarray(ws, dtype=np.int32),
                np.asarray(we, dtype=np.int32),
                float(np.median(np.asarray(fpss, dtype=np.float32))),
                mode="center",
            )
            _mask, events = detect_alert_events(probs_a, t, alert_cfg, lying_score=lying_a, motion_score=motion_a, video_id=video_id)
            pred = 1 if events else 0
            high = probs_a >= float(alert_cfg.tau_high)
            max_lying = float(lying_a.max()) if lying_a.size else 0.0
            mean_motion_high = float(motion_a[high].mean()) if high.any() else 0.0
            first_event_start_s = float(events[0].start_time_s) if events else None

            if pred == 1:
                if args.gate_max_lying >= 0.0 and max_lying > float(args.gate_max_lying):
                    pred = 0
                if pred == 1 and args.gate_min_mean_motion_high >= 0.0 and mean_motion_high < float(args.gate_min_mean_motion_high):
                    pred = 0
                if pred == 1 and args.gate_max_event_start_s >= 0.0 and first_event_start_s is not None and first_event_start_s > float(args.gate_max_event_start_s):
                    pred = 0

            src = rows[0]["src"]
            truth = _truth_from_src(src, args.fall_contains, args.nonfall_contains)
            rows_out.append(
                {
                    "video_id": video_id,
                    "src": src,
                    "true_label": truth,
                    "pred_label": pred,
                    "match": (int(truth == pred) if truth is not None else ""),
                    "event_count": len(events),
                    "max_p_fall": float(probs_a.max()) if probs_a.size else 0.0,
                    "mean_p_fall": float(probs_a.mean()) if probs_a.size else 0.0,
                    "max_lying": max_lying,
                    "mean_motion_high": mean_motion_high,
                    "first_event_start_s": (first_event_start_s if first_event_start_s is not None else ""),
                    "n_windows": int(len(rows)),
                }
            )

    rows_out.sort(key=lambda r: r["src"])
    summary_json = out_prefix.with_suffix(".json")
    summary_csv = out_prefix.with_suffix(".csv")
    metrics_json = out_prefix.parent / f"{out_prefix.name}_metrics.json"
    metrics = {
        "n_videos": len(rows_out),
        "n_true_fall": sum(1 for r in rows_out if r["true_label"] == 1),
        "n_true_nonfall": sum(1 for r in rows_out if r["true_label"] == 0),
        "tp": sum(1 for r in rows_out if r["true_label"] == 1 and r["pred_label"] == 1),
        "tn": sum(1 for r in rows_out if r["true_label"] == 0 and r["pred_label"] == 0),
        "fp": sum(1 for r in rows_out if r["true_label"] == 0 and r["pred_label"] == 1),
        "fn": sum(1 for r in rows_out if r["true_label"] == 1 and r["pred_label"] == 0),
        "alert_cfg": alert_cfg.to_dict(),
        "delivery_gate": {
            "gate_max_lying": args.gate_max_lying,
            "gate_min_mean_motion_high": args.gate_min_mean_motion_high,
            "gate_max_event_start_s": args.gate_max_event_start_s,
        },
        "config_yaml": args.config_yaml,
        "ops_yaml": str(ops_path),
        "op_code": op_code,
        "checkpoint": str(ckpt),
    }

    summary_json.write_text(json.dumps(rows_out, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wr.writeheader()
        wr.writerows(rows_out)
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[ok] wrote {summary_csv}")
    print(f"[ok] wrote {summary_json}")
    print(f"[ok] wrote {metrics_json}")


if __name__ == "__main__":
    main()
