#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi.testclient import TestClient

from applications.backend import core
from applications.backend.app import app


def _group_windows(windows_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(windows_dir.glob("*.npz")):
        video_id = path.stem.split("__w", 1)[0]
        groups[video_id].append(path)
    return groups


def _gt_for_video(video_id: str) -> int:
    return 0 if "_adl__" in video_id else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a directory of fixed windows through /api/monitor/predict_window.")
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--dataset_code", default="caucafall")
    ap.add_argument("--mode", default="tcn")
    ap.add_argument("--op_code", default="OP-2")
    ap.add_argument("--target_T", type=int, default=48)
    ap.add_argument("--session_prefix", default="replay")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    groups = _group_windows(windows_dir)

    tp = tn = fp = fn = 0
    rows: List[Dict[str, Any]] = []

    for video_id, files in sorted(groups.items()):
        session_id = f"{args.session_prefix}:{video_id}"
        core._SESSION_STATE.pop(session_id, None)
        gt = _gt_for_video(video_id)
        pred = 0
        first_fall_idx = None
        first_gate: Dict[str, Any] | None = None
        for idx, path in enumerate(files):
            x = np.load(path, allow_pickle=True)
            payload = {
                "session_id": session_id,
                "dataset_code": args.dataset_code,
                "mode": args.mode,
                "op_code": args.op_code,
                "target_T": int(args.target_T),
                "xy": x["joints"].tolist(),
                "conf": x["conf"].tolist(),
            }
            out = client.post("/api/monitor/predict_window", json=payload).json()
            if out.get("triage_state") == "fall":
                pred = 1
                first_fall_idx = idx
                first_gate = out.get("delivery_gate")
                break

        if gt and pred:
            tp += 1
            status = "TP"
        elif (not gt) and (not pred):
            tn += 1
            status = "TN"
        elif pred:
            fp += 1
            status = "FP"
        else:
            fn += 1
            status = "FN"

        rows.append(
            {
                "video_id": video_id,
                "gt": int(gt),
                "pred": int(pred),
                "status": status,
                "first_fall_idx": first_fall_idx,
                "delivery_gate": first_gate,
                "src": str(files[0]) if files else None,
            }
        )

    out = {
        "run": args.mode,
        "op": args.op_code,
        "summary": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "n_videos": len(rows),
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
