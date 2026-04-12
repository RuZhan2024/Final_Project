#!/usr/bin/env python3
"""Audit ops sweep sanity: fail when sweep has no meaningful alert region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _arr(d: dict[str, Any], k: str) -> np.ndarray:
    x = d.get(k, [])
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)
    if isinstance(x, list):
        vals = [np.nan if v is None else float(v) for v in x]
        return np.asarray(vals, dtype=np.float32).reshape(-1)
    return np.asarray([], dtype=np.float32)


def _check_one(fp: Path, eps: float) -> tuple[bool, str]:
    data = json.loads(fp.read_text(encoding="utf-8"))
    sw = data.get("sweep", {})
    if not isinstance(sw, dict):
        return False, "missing sweep dict"

    rec = _arr(sw, "recall")
    f1 = _arr(sw, "f1")
    n_alert = _arr(sw, "n_alert_events")

    has_rec = bool(rec.size and np.isfinite(rec).any() and float(np.nanmax(rec)) > eps)
    has_f1 = bool(f1.size and np.isfinite(f1).any() and float(np.nanmax(f1)) > eps)
    has_alert = bool(n_alert.size and np.isfinite(n_alert).any() and float(np.nanmax(n_alert)) > eps)

    if has_rec or has_f1 or has_alert:
        return True, ""
    return False, "degenerate sweep: no recall/F1/alert-events across all thresholds"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops_dir", default="ops/configs/ops")
    ap.add_argument("--pattern", default="*.sweep.json")
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    root = Path(args.ops_dir)
    files = sorted(root.glob(args.pattern))
    if not files:
        raise SystemExit(f"[err] no sweep files found in {root} matching {args.pattern}")

    failures: list[str] = []
    for fp in files:
        ok, msg = _check_one(fp, float(args.eps))
        if ok:
            print(f"[ok] {fp}")
        else:
            failures.append(f"{fp}: {msg}")

    if failures:
        print("[fail] ops sanity failures:")
        for f in failures:
            print(f" - {f}")
        raise SystemExit(1)
    print("[ok] ops sanity passed")


if __name__ == "__main__":
    main()

