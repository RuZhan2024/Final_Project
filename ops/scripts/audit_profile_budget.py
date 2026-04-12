#!/usr/bin/env python3
"""Validate inference profile report against latency budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[err] profile report not found: {path}") from None
    except json.JSONDecodeError as e:
        raise SystemExit(f"[err] invalid profile report json: {path}: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit(f"[err] profile report must be a JSON object: {path}")
    return data


def _load_budget(gates_json: Path, profile: str, with_model: bool) -> float:
    try:
        cfg = json.loads(gates_json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[err] gates config not found: {gates_json}") from None
    except json.JSONDecodeError as e:
        raise SystemExit(f"[err] invalid gates config json: {gates_json}: {e}") from e

    prof = ((cfg.get("profile") or {}).get(profile) or {})
    key = "with_model_p95_ms_max" if with_model else "io_only_p95_ms_max"
    val = prof.get(key)
    if val is None:
        raise SystemExit(f"[err] missing profile budget '{profile}.{key}' in {gates_json}")
    return float(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--gates_json", default="ops/configs/audit_gates.json")
    args = ap.parse_args()

    report_path = Path(args.report_json)
    gates_path = Path(args.gates_json)
    rep = _load_json(report_path)

    profile = str(rep.get("profile", "")).strip() or "cpu_local"
    with_model = bool(rep.get("with_model", False))
    lat = rep.get("latency_ms") or {}
    try:
        p95 = float(lat.get("p95"))
    except Exception:
        raise SystemExit(f"[err] missing/invalid latency_ms.p95 in {report_path}") from None

    budget = _load_budget(gates_path, profile, with_model)
    mode = "with_model" if with_model else "io_only"
    print(f"[info] profile={profile} mode={mode} p95_ms={p95:.4f} budget_ms={budget:.4f}")
    if p95 > budget:
        raise SystemExit(f"[fail] p95 {p95:.4f}ms exceeds budget {budget:.4f}ms")
    print("[ok] profile budget passed")


if __name__ == "__main__":
    main()

