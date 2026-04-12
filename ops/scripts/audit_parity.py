#!/usr/bin/env python3
"""Baseline parity gate using committed baseline bundle under baselines/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_labels(pattern: str) -> dict[str, float]:
    files = sorted(Path().glob(pattern))
    pos = 0
    neg = 0
    fps_vals: list[float] = []
    for fp in files:
        with np.load(fp, allow_pickle=False) as z:
            y = int(np.asarray(z["y"]).reshape(-1)[0]) if "y" in z.files else -1
            pos += int(y == 1)
            neg += int(y == 0)
            if "fps" in z.files:
                fps_vals.append(float(np.asarray(z["fps"]).reshape(-1)[0]))
    return {
        "total": float(len(files)),
        "pos": float(pos),
        "neg": float(neg),
        "fps_mean": float(sum(fps_vals) / len(fps_vals)) if fps_vals else 0.0,
    }


def _current_contract() -> dict[str, Any]:
    return {
        "le2i_eval": _count_labels("data/processed/le2i/windows_eval_W48_S12/*/*.npz"),
        "cauc_eval": _count_labels("data/processed/caucafall/windows_eval_W48_S12/*/*.npz"),
    }


def _selected_metrics(report: dict[str, Any], op_name: str) -> dict[str, float]:
    ops = report.get("ops", {})
    key = str(op_name).lower()
    m = ops.get(key, {})
    return {
        "f1": float(m.get("f1", 0.0)),
        "recall": float(m.get("recall", 0.0)),
        "fa24h": float(m.get("fa24h", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--out_json", default="artifacts/reports/parity_le2i_latest.json")
    ap.add_argument("--count_tol", type=int, default=0)
    ap.add_argument("--fps_tol", type=float, default=0.25)
    ap.add_argument("--op", default="op2")
    ap.add_argument("--current_metrics_json", default="")
    ap.add_argument("--allow_missing_perf_baseline", type=int, default=1)
    ap.add_argument("--require_perf_targets", type=int, default=0)
    args = ap.parse_args()

    bdir = Path(args.baseline_dir)
    contract_path = bdir / "dataset_contract.json"
    if not contract_path.is_file():
        raise SystemExit(f"[ERR] missing baseline dataset contract: {contract_path}")

    baseline = _load_json(contract_path)
    current = _current_contract()

    report: dict[str, Any] = {
        "baseline_dir": str(bdir),
        "status": "pass",
        "checks": {},
        "failures": [],
        "warnings": [],
    }

    for key in ("le2i_eval", "cauc_eval"):
        b = baseline.get(key)
        c = current.get(key)
        if not isinstance(b, dict) or not isinstance(c, dict):
            report["status"] = "fail"
            report["failures"].append(f"missing contract key: {key}")
            continue

        total_delta = abs(int(c["total"]) - int(b["total"]))
        pos_delta = abs(int(c["pos"]) - int(b["pos"]))
        neg_delta = abs(int(c["neg"]) - int(b["neg"]))
        fps_delta = abs(float(c["fps_mean"]) - float(b["fps_mean"]))
        report["checks"][key] = {
            "baseline": b,
            "current": c,
            "delta": {
                "total": total_delta,
                "pos": pos_delta,
                "neg": neg_delta,
                "fps_mean": fps_delta,
            },
        }
        if total_delta > int(args.count_tol):
            report["status"] = "fail"
            report["failures"].append(f"{key}: total delta {total_delta} > {args.count_tol}")
        if pos_delta > int(args.count_tol):
            report["status"] = "fail"
            report["failures"].append(f"{key}: pos delta {pos_delta} > {args.count_tol}")
        if neg_delta > int(args.count_tol):
            report["status"] = "fail"
            report["failures"].append(f"{key}: neg delta {neg_delta} > {args.count_tol}")
        if fps_delta > float(args.fps_tol):
            report["status"] = "fail"
            report["failures"].append(f"{key}: fps_mean delta {fps_delta:.6f} > {args.fps_tol}")

    # Optional performance parity when a pinned baseline performance file exists.
    perf_base = bdir / "performance_baseline.json"
    cur_metrics_json = Path(args.current_metrics_json) if args.current_metrics_json.strip() else None
    if perf_base.is_file() and cur_metrics_json is not None and cur_metrics_json.is_file():
        pb = _load_json(perf_base)
        cm = _load_json(cur_metrics_json)
        cur = _selected_metrics(cm, args.op)
        tgt = pb.get("targets", {}) if isinstance(pb.get("targets"), dict) else {}
        tols = pb.get("tolerances", {}) if isinstance(pb.get("tolerances"), dict) else {}
        prov = pb.get("provenance", {}) if isinstance(pb.get("provenance"), dict) else {}
        status = str(pb.get("status", "")).strip().lower()
        if int(args.require_perf_targets):
            if status != "captured":
                report["status"] = "fail"
                report["failures"].append(f"performance baseline status is '{status}', expected 'captured'")
            for req in ("checkpoint", "checkpoint_sha256", "metrics_source", "op"):
                if not str(prov.get(req, "")).strip():
                    report["status"] = "fail"
                    report["failures"].append(f"missing performance provenance field: {req}")
        pending = [k for k in ("f1", "recall", "fa24h") if tgt.get(k) is None]
        if pending:
            msg = f"performance targets unset: {', '.join(pending)}"
            if int(args.require_perf_targets):
                report["status"] = "fail"
                report["failures"].append(msg)
            else:
                report["warnings"].append(msg)
        for k, tol_key, default_tol in (
            ("f1", "f1_abs_delta_max", 0.02),
            ("recall", "recall_abs_delta_max", 0.02),
            ("fa24h", "fa24h_abs_delta_max", 0.2),
        ):
            if tgt.get(k) is None:
                continue
            delta = abs(float(cur[k]) - float(tgt[k]))
            tol = float(tols.get(tol_key, default_tol))
            report["checks"][f"perf_{k}"] = {
                "current": float(cur[k]),
                "target": float(tgt[k]),
                "delta": delta,
                "tol": tol,
            }
            if delta > tol:
                report["status"] = "fail"
                report["failures"].append(f"perf {k}: delta {delta:.6f} > {tol:.6f}")
    else:
        msg = "performance baseline not enforced (missing performance_baseline.json or current metrics input)"
        if int(args.allow_missing_perf_baseline):
            report["warnings"].append(msg)
        else:
            report["status"] = "fail"
            report["failures"].append(msg)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] parity report: {out}")

    if report["status"] == "fail":
        print("[fail] parity gate failed")
        for f in report["failures"]:
            print(f" - {f}")
        raise SystemExit(1)
    if report["warnings"]:
        print("[warn] parity gate warnings:")
        for w in report["warnings"]:
            print(f" - {w}")
    print("[ok] parity gate passed")


if __name__ == "__main__":
    main()
