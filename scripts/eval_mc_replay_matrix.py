#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fall_detection.core.alerting import AlertCfg, detect_alert_events, times_from_windows
from fall_detection.core.confirm import confirm_scores_window
from fall_detection.core.features import read_window_npz
from server.deploy_runtime import get_specs, predict_spec


def truth_from_src(src: str, fall_contains: list[str], nonfall_contains: list[str]) -> int | None:
    src_l = src.lower()
    for needle in nonfall_contains:
        if needle.lower() in src_l:
            return 0
    for needle in fall_contains:
        if needle.lower() in src_l:
            return 1
    return None


def build_alert_cfg(spec: Any, op_code: str) -> AlertCfg:
    op = (spec.ops or {}).get(op_code) or {}
    base = dict(spec.alert_cfg or {})
    return AlertCfg(
        ema_alpha=float(base.get("ema_alpha", 0.2)),
        k=int(base.get("k", 2)),
        n=int(base.get("n", 3)),
        tau_high=float(op.get("tau_high", base.get("tau_high", 0.85))),
        tau_low=float(op.get("tau_low", base.get("tau_low", 0.5))),
        cooldown_s=float(base.get("cooldown_s", 30.0)),
        confirm=bool(base.get("confirm", False)),
        confirm_s=float(base.get("confirm_s", 2.0)),
        confirm_min_lying=float(base.get("confirm_min_lying", 0.65)),
        confirm_max_motion=float(base.get("confirm_max_motion", 0.08)),
        confirm_require_low=bool(base.get("confirm_require_low", True)),
        start_guard_max_lying=base.get("start_guard_max_lying"),
        start_guard_prefixes=base.get("start_guard_prefixes"),
    )


def load_groups(windows_dir: Path, fps_default: float) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp in sorted(windows_dir.glob("*.npz")):
        joints, motion, conf, mask, fps, meta = read_window_npz(str(fp), fps_default=fps_default)
        with np.load(str(fp), allow_pickle=False) as z:
            src = str(z["src"].item()) if "src" in z.files else ""
        groups[str(meta.video_id)].append(
            {
                "joints": joints,
                "mask": mask,
                "conf": conf,
                "fps": float(fps),
                "w_start": int(meta.w_start),
                "w_end": int(meta.w_end),
                "src": src,
            }
        )
    return groups


def evaluate_combo(
    *,
    spec_key: str,
    op_code: str,
    use_mc: bool,
    mc_M: int,
    groups: dict[str, list[dict[str, Any]]],
    fall_contains: list[str],
    nonfall_contains: list[str],
) -> dict[str, Any]:
    spec = get_specs()[spec_key]
    alert_cfg = build_alert_cfg(spec, op_code)

    rows_out: list[dict[str, Any]] = []
    mc_applied_windows = 0
    total_windows = 0

    for video_id, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (r["w_start"], r["w_end"]))
        probs = []
        lying = []
        motion_s = []
        ws, we, fpss = [], [], []

        for r in rows:
            out = predict_spec(
                spec_key=spec_key,
                joints_xy=r["joints"],
                conf=r["conf"],
                fps=float(r["fps"]),
                target_T=int(r["joints"].shape[0]),
                op_code=op_code,
                use_mc=use_mc,
                mc_M=mc_M,
            )
            probs.append(float(out["mu"] if use_mc else out["p_det"]))
            mc_applied_windows += int(bool(out.get("mc_applied", False)))
            total_windows += 1

            ls, ms = confirm_scores_window(r["joints"], r["mask"], fps=float(r["fps"]), tail_s=1.0)
            lying.append(float(ls) if np.isfinite(ls) else 0.0)
            motion_s.append(float(ms) if np.isfinite(ms) else float("inf"))
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
        _mask, events = detect_alert_events(
            probs_a, t, alert_cfg, lying_score=lying_a, motion_score=motion_a, video_id=video_id
        )
        pred = 1 if events else 0
        src = rows[0]["src"]
        truth = truth_from_src(src, fall_contains, nonfall_contains)
        rows_out.append({"src": src, "truth": truth, "pred": pred})

    tp = sum(1 for r in rows_out if r["truth"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in rows_out if r["truth"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in rows_out if r["truth"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in rows_out if r["truth"] == 1 and r["pred"] == 0)
    correct = tp + tn
    total = len(rows_out)

    return {
        "spec_key": spec_key,
        "dataset": str(spec.dataset),
        "arch": str(spec.arch).upper(),
        "op_code": op_code,
        "use_mc": bool(use_mc),
        "mc_M": int(mc_M),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "correct": correct,
        "n_videos": total,
        "accuracy": (float(correct) / float(total)) if total else 0.0,
        "recall": (float(tp) / float(tp + fn)) if (tp + fn) else 0.0,
        "specificity": (float(tn) / float(tn + fp)) if (tn + fp) else 0.0,
        "mc_applied_windows": int(mc_applied_windows),
        "total_windows": int(total_windows),
        "mc_applied_ratio": (float(mc_applied_windows) / float(total_windows)) if total_windows else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MC on/off matrix on replay custom clips.")
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--mc_M", type=int, default=10)
    ap.add_argument("--fps_default", type=float, default=15.0)
    ap.add_argument("--fall_contains", nargs="+", required=True)
    ap.add_argument("--nonfall_contains", nargs="+", required=True)
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    groups = load_groups(windows_dir, fps_default=float(args.fps_default))
    specs = get_specs()

    target_specs = [
        ("caucafall_tcn", "OP-1"),
        ("caucafall_tcn", "OP-2"),
        ("caucafall_tcn", "OP-3"),
        ("caucafall_gcn", "OP-1"),
        ("caucafall_gcn", "OP-2"),
        ("caucafall_gcn", "OP-3"),
        ("le2i_tcn", "OP-1"),
        ("le2i_tcn", "OP-2"),
        ("le2i_tcn", "OP-3"),
        ("le2i_gcn", "OP-1"),
        ("le2i_gcn", "OP-2"),
        ("le2i_gcn", "OP-3"),
    ]

    rows = []
    for spec_key, op_code in target_specs:
        if spec_key not in specs:
            continue
        for use_mc in (False, True):
            rows.append(
                evaluate_combo(
                    spec_key=spec_key,
                    op_code=op_code,
                    use_mc=use_mc,
                    mc_M=int(args.mc_M),
                    groups=groups,
                    fall_contains=list(args.fall_contains),
                    nonfall_contains=list(args.nonfall_contains),
                )
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "arch",
                "spec_key",
                "op_code",
                "use_mc",
                "mc_M",
                "tp",
                "tn",
                "fp",
                "fn",
                "correct",
                "n_videos",
                "accuracy",
                "recall",
                "specificity",
                "mc_applied_windows",
                "total_windows",
                "mc_applied_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "out_json": str(out_json), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
