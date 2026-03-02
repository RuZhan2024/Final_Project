#!/usr/bin/env python3
"""Validate artifact bundle integrity and portability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _is_abs_like(s: str) -> bool:
    return s.startswith("/") or (len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_json", default="artifacts/artifact_bundle.json")
    args = ap.parse_args()

    bpath = Path(args.bundle_json)
    if not bpath.exists():
        raise SystemExit(f"[err] bundle not found: {bpath}")
    try:
        data = json.loads(bpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[err] invalid bundle json: {e}") from e

    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit("[err] bundle must contain non-empty 'files' list")

    root = bpath.parent
    failures: list[str] = []
    for i, rec in enumerate(files):
        if not isinstance(rec, dict):
            failures.append(f"files[{i}] is not an object")
            continue
        p = rec.get("path")
        if not isinstance(p, str) or not p.strip():
            failures.append(f"files[{i}] missing path")
            continue
        if _is_abs_like(p):
            failures.append(f"files[{i}] absolute path forbidden: {p}")
            continue
        target = (root / p).resolve()
        if not target.exists():
            failures.append(f"files[{i}] missing target: {p}")

    if failures:
        print("[fail] artifact bundle validation failed:")
        for f in failures:
            print(f" - {f}")
        raise SystemExit(1)
    print(f"[ok] artifact bundle valid: {bpath}")


if __name__ == "__main__":
    main()

