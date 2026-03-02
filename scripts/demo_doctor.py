#!/usr/bin/env python3
"""Local demo readiness checker for examiner-friendly setup validation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return int(e.returncode), (e.output or "").strip()
    except Exception as e:
        return 1, str(e)


def _check_import(mod: str) -> tuple[bool, str]:
    try:
        importlib.import_module(mod)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="le2i")
    ap.add_argument("--model", default="tcn", choices=["tcn", "gcn"])
    ap.add_argument("--strict", type=int, default=1, help="exit non-zero when critical checks fail")
    ap.add_argument("--out_json", default="artifacts/reports/demo_doctor.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, critical: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "critical": bool(critical)})

    add("python_version", sys.version_info >= (3, 10), f"{sys.version.split()[0]}", critical=True)
    add("venv_exists", (root / ".venv").is_dir(), str(root / ".venv"), critical=False)

    for mod in ("fall_detection", "server.app", "fastapi"):
        ok, detail = _check_import(mod)
        add(f"import:{mod}", ok, detail, critical=True)

    add("frontend_dir", (root / "apps").is_dir(), str(root / "apps"), critical=True)
    add("frontend_config", (root / "apps" / "src" / "lib" / "config.js").is_file(), "apps/src/lib/config.js", critical=True)

    npm_path = shutil.which("npm")
    add("npm_present", npm_path is not None, npm_path or "npm not found in PATH", critical=False)
    if npm_path:
        rc, out = _run(["npm", "--version"])
        add("npm_version", rc == 0, out if out else "unknown", critical=False)

    uvicorn_path = shutil.which("uvicorn")
    add("uvicorn_present", uvicorn_path is not None, uvicorn_path or "uvicorn not found in PATH", critical=False)

    ckpt = root / "outputs" / f"{args.dataset}_{args.model}_W48S12" / "best.pt"
    ops = root / "configs" / "ops" / f"{args.model}_{args.dataset}.yaml"
    add("checkpoint_exists", ckpt.is_file(), str(ckpt), critical=True)
    add("ops_yaml_exists", ops.is_file(), str(ops), critical=True)

    add("raw_data_dir", (root / "data" / "raw").is_dir(), str(root / "data" / "raw"), critical=False)
    add("camera_permission_note", True, "Manual check required: browser/camera permissions during demo.", critical=False)

    critical_failures = [c for c in checks if c["critical"] and not c["ok"]]
    ready = len(critical_failures) == 0

    report = {
        "schema_version": "1.0",
        "ready": ready,
        "dataset": args.dataset,
        "model": args.model,
        "checks": checks,
        "critical_failures": [c["name"] for c in critical_failures],
        "quick_start": [
            "source .venv/bin/activate",
            "uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload",
            "cd apps && npm install && npm start",
        ],
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[ok] demo doctor report: {out}")
    print(f"[status] {'READY' if ready else 'NOT READY'}")
    if critical_failures:
        for c in critical_failures:
            print(f"[fail] {c['name']}: {c['detail']}")

    if int(args.strict) == 1 and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

