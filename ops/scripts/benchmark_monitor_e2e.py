#!/usr/bin/env python3
"""Benchmark monitor end-to-end API latency by replaying raw skeleton windows."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _post_http(base: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=f"{base.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return {"status": int(resp.status), "json": json.loads(raw) if raw else {}}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        body: Any
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"raw": raw}
        return {"status": int(e.code), "json": body}


def _synthetic_window(start_ms: float, dt_ms: float, n_frames: int, n_joints: int) -> Dict[str, Any]:
    t = [start_ms + i * dt_ms for i in range(n_frames)]
    xy = []
    conf = []
    for i in range(n_frames):
        frame_xy = []
        frame_conf = []
        for j in range(n_joints):
            frame_xy.append([0.1 + 0.001 * i + 0.0001 * j, 0.2 + 0.0008 * i + 0.0001 * j])
            frame_conf.append(1.0)
        xy.append(frame_xy)
        conf.append(frame_conf)
    return {"raw_t_ms": t, "raw_xy": xy, "raw_conf": conf, "window_end_t_ms": t[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="/api/monitor/predict_window")
    ap.add_argument("--dataset_code", default="le2i")
    ap.add_argument("--mode", default="tcn", choices=["tcn", "gcn", "dual"])
    ap.add_argument("--op_code", default="OP-2")
    ap.add_argument("--n_windows", type=int, default=40)
    ap.add_argument("--window_frames", type=int, default=48)
    ap.add_argument("--target_fps", type=float, default=25.0)
    ap.add_argument("--joints", type=int, default=33)
    ap.add_argument("--sleep_ms", type=float, default=0.0)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    dt_ms = 1000.0 / max(1.0, float(args.target_fps))
    t0 = time.time() * 1000.0

    api_lat_ms: List[float] = []
    decision_delay_ms: List[float] = []
    send_gap_ms: List[float] = []
    statuses: Dict[str, int] = {}
    last_send_end_ms: float | None = None

    for w in range(max(1, int(args.n_windows))):
        win = _synthetic_window(
            start_ms=t0 + w * dt_ms * (args.window_frames // 4),
            dt_ms=dt_ms,
            n_frames=int(args.window_frames),
            n_joints=int(args.joints),
        )
        payload = {
            "session_id": "benchmark-session",
            "mode": str(args.mode),
            "dataset_code": str(args.dataset_code),
            "op_code": str(args.op_code),
            "target_T": int(args.window_frames),
            "target_fps": float(args.target_fps),
            **win,
        }

        send_start = time.time() * 1000.0
        if last_send_end_ms is not None:
            send_gap_ms.append(float(send_start - last_send_end_ms))
        resp = _post_http(args.api_base, args.endpoint, payload)
        send_end = time.time() * 1000.0
        last_send_end_ms = send_end

        api_lat_ms.append(float(send_end - send_start))
        decision_delay_ms.append(float(send_end - float(win["window_end_t_ms"])))
        sc = str(resp.get("status", 0))
        statuses[sc] = statuses.get(sc, 0) + 1

        if float(args.sleep_ms) > 0:
            time.sleep(float(args.sleep_ms) / 1000.0)

    report = {
        "schema_version": "1.0",
        "endpoint": f"{args.api_base.rstrip('/')}{args.endpoint}",
        "dataset_code": args.dataset_code,
        "mode": args.mode,
        "n_windows": int(args.n_windows),
        "target_fps": float(args.target_fps),
        "window_frames": int(args.window_frames),
        "status_counts": statuses,
        "latency_ms": summarize(api_lat_ms),
        "decision_delay_ms": summarize(decision_delay_ms),
        "send_gap_ms": summarize(send_gap_ms),
    }

    out_json = args.out_json.strip() or f"artifacts/reports/monitor_e2e_benchmark_{int(time.time())}.json"
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] benchmark report: {out}")
    print(json.dumps(report["latency_ms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

