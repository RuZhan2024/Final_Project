#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import mediapipe as mp

from fall_detection.pose.extract_2d_videos import make_safe_stem, run_one_video


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _iter_windows(
    xy: np.ndarray,
    conf: np.ndarray,
    fps: float,
    *,
    target_t: int,
    stride: int,
) -> tuple[int, list[float], list[Any], list[Any]]:
    total = int(xy.shape[0])
    if total < target_t:
        return 0, [], [], []
    for start in range(0, total - target_t + 1, stride):
        end = start + target_t
        t_ms = [float((i * 1000.0) / fps) for i in range(start, end)]
        yield start, t_ms, xy[start:end].tolist(), conf[start:end].tolist()


def _guess_expected_label(name: str) -> str:
    n = name.lower()
    if any(tok in n for tok in ("walk", "sit", "squat", "bend", "lie")):
        return "nonfall"
    return "fall"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-evaluate local replay clips against the local backend.")
    ap.add_argument("--clips_dir", default="deploy_assets/replay_clips")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/api/monitor/predict_window")
    ap.add_argument("--dataset_code", default="caucafall")
    ap.add_argument("--mode", default="tcn")
    ap.add_argument("--op_code", default="OP-2")
    ap.add_argument("--target_t", type=int, default=48)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--model_complexity", type=int, default=2)
    ap.add_argument("--min_det_conf", type=float, default=0.35)
    ap.add_argument("--min_track_conf", type=float, default=0.35)
    ap.add_argument("--fps_default", type=float, default=23.0)
    ap.add_argument("--timeout_s", type=float, default=30.0)
    ap.add_argument("--out_json", default="artifacts/replay_eval/local_replay_eval.json")
    ap.add_argument("--pose_cache_dir", default=None)
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pose_cache_dir = Path(args.pose_cache_dir) if args.pose_cache_dir else None
    if pose_cache_dir is not None:
        pose_cache_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in clips_dir.rglob("*.mp4") if p.is_file())
    if not clips:
        raise SystemExit(f"[err] no mp4 clips found under {clips_dir}")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="replay_eval_pose_") as td:
        tmp_dir = Path(td)
        root = str(clips_dir.resolve())
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(args.model_complexity),
            enable_segmentation=False,
            min_detection_confidence=float(args.min_det_conf),
            min_tracking_confidence=float(args.min_track_conf),
        ) as pose:
            for clip in clips:
                stem = make_safe_stem(str(clip), root)
                out_npz = (pose_cache_dir / f"{stem}.npz") if pose_cache_dir is not None else (tmp_dir / f"{stem}.npz")
                ok = True
                if not out_npz.exists():
                    ok = run_one_video(
                        str(clip),
                        str(out_npz),
                        pose,
                        fps_default=float(args.fps_default),
                        force_fps=None,
                        log_every_s=0.0,
                        max_proc_fps=0.0,
                    )
                if not ok or (not out_npz.exists()):
                    rows.append(
                        {
                            "clip": clip.name,
                            "expected": _guess_expected_label(clip.name),
                            "status": "extract_failed",
                        }
                    )
                    continue

                z = np.load(out_npz, allow_pickle=False)
                xy = np.asarray(z["xy"], dtype=np.float32)
                conf = np.asarray(z["conf"], dtype=np.float32)
                fps = float(z["fps"]) if "fps" in z.files else float(args.fps_default)
                session_id = f"batch:{clip.stem}"

                seen_fall = False
                seen_uncertain = False
                first_fall_window = None
                max_p_fall = 0.0
                window_count = 0
                error_text = None
                first_fall_diag = None

                for window_seq, (start, t_ms, xy_win, conf_win) in enumerate(
                    _iter_windows(
                        xy,
                        conf,
                        fps,
                        target_t=int(args.target_t),
                        stride=int(args.stride),
                    )
                ):
                    payload = {
                        "session_id": session_id,
                        "dataset_code": args.dataset_code,
                        "mode": args.mode,
                        "op_code": args.op_code,
                        "target_T": int(args.target_t),
                        "window_seq": int(window_seq),
                        "window_end_t_ms": float(t_ms[-1]),
                        "raw_t_ms": t_ms,
                        "raw_xy": xy_win,
                        "raw_conf": conf_win,
                        "input_source": "replay",
                        "persist": False,
                    }
                    try:
                        out = _post_json(str(args.endpoint), payload, timeout_s=float(args.timeout_s))
                    except urllib.error.HTTPError as exc:
                        error_text = f"HTTP {exc.code}"
                        break
                    except Exception as exc:  # noqa: BLE001
                        error_text = str(exc)
                        break

                    window_count += 1
                    triage = str(out.get("triage_state") or "")
                    p_fall = 0.0
                    models = out.get("models") or {}
                    if isinstance(models, dict):
                        primary = models.get(str(args.mode).lower()) or models.get("tcn") or {}
                        if isinstance(primary, dict):
                            p_fall = float(primary.get("p_alert_in", primary.get("mu", 0.0)) or 0.0)
                    max_p_fall = max(max_p_fall, p_fall)
                    if triage == "uncertain":
                        seen_uncertain = True
                    if triage == "fall":
                        seen_fall = True
                        first_fall_window = int(window_seq)
                        first_fall_diag = {
                            "delivery_gate": out.get("delivery_gate"),
                            "uncertain_promoted": out.get("uncertain_promoted"),
                            "replay_relaxed_promoted": out.get("replay_relaxed_promoted"),
                            "lying_score": out.get("lying_score"),
                            "confirm_motion_score": out.get("confirm_motion_score"),
                            "motion_score": out.get("motion_score"),
                            "quality_diag": out.get("quality_diag"),
                        }
                        break

                final_state = "fall" if seen_fall else "uncertain" if seen_uncertain else "not_fall"
                rows.append(
                    {
                        "clip": clip.name,
                        "expected": _guess_expected_label(clip.name),
                        "status": "ok" if error_text is None else "request_failed",
                        "final_state": final_state,
                        "seen_fall": bool(seen_fall),
                        "seen_uncertain": bool(seen_uncertain),
                        "first_fall_window": first_fall_window,
                        "first_fall_diag": first_fall_diag,
                        "max_p_fall": float(max_p_fall),
                        "windows_sent": int(window_count),
                        "fps": float(fps),
                        "error": error_text,
                    }
                )

    summary = {
        "n_clips": len(rows),
        "falls_detected": sum(1 for r in rows if r.get("seen_fall")),
        "uncertain_only": sum(
            1
            for r in rows
            if (not r.get("seen_fall")) and r.get("seen_uncertain")
        ),
        "not_fall": sum(
            1
            for r in rows
            if (not r.get("seen_fall")) and (not r.get("seen_uncertain")) and r.get("status") == "ok"
        ),
        "request_failed": sum(1 for r in rows if r.get("status") == "request_failed"),
        "extract_failed": sum(1 for r in rows if r.get("status") == "extract_failed"),
    }

    out_payload = {
        "endpoint": str(args.endpoint),
        "mode": str(args.mode),
        "dataset_code": str(args.dataset_code),
        "op_code": str(args.op_code),
        "target_t": int(args.target_t),
        "stride": int(args.stride),
        "summary": summary,
        "rows": rows,
    }
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[ok] wrote {out_json}")


if __name__ == "__main__":
    main()
