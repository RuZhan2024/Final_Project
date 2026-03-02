#!/usr/bin/env python3
"""Populate performance_baseline.json from a pinned metrics artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_metrics(report: dict[str, Any], op_name: str) -> dict[str, float]:
    ops = report.get("ops", {})
    key = str(op_name).lower()
    m = ops.get(key, {})
    return {
        "f1": float(m.get("f1", 0.0)),
        "recall": float(m.get("recall", 0.0)),
        "fa24h": float(m.get("fa24h", 0.0)),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_perf_json", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--op", default="op2")
    ap.add_argument("--metrics_source", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    perf_path = Path(args.baseline_perf_json)
    if not perf_path.is_file():
        raise SystemExit(f"[ERR] missing baseline performance file: {perf_path}")
    metrics_path = Path(args.metrics_json)
    if not metrics_path.is_file():
        raise SystemExit(f"[ERR] missing metrics file: {metrics_path}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"[ERR] missing checkpoint file: {ckpt_path}")

    base = _load_json(perf_path)
    rep = _load_json(metrics_path)
    cur = _selected_metrics(rep, args.op)

    base.setdefault("targets", {})
    base["targets"]["f1"] = float(cur["f1"])
    base["targets"]["recall"] = float(cur["recall"])
    base["targets"]["fa24h"] = float(cur["fa24h"])
    base["status"] = "captured"

    base.setdefault("provenance", {})
    base["provenance"]["checkpoint"] = str(ckpt_path)
    base["provenance"]["checkpoint_sha256"] = _sha256_file(ckpt_path)
    base["provenance"]["metrics_source"] = str(args.metrics_source or metrics_path)
    if args.notes.strip():
        base["provenance"]["notes"] = str(args.notes)
    base["provenance"]["op"] = str(args.op).lower()

    perf_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] updated baseline performance: {perf_path}")
    print(json.dumps(base["targets"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
