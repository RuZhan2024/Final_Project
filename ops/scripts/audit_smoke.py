#!/usr/bin/env python3
"""Project smoke audit: import contracts, resolver config, and test presence."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


def _check_imports(mods: list[str]) -> list[str]:
    failures: list[str] = []
    for mod in mods:
        try:
            importlib.import_module(mod)
            print(f"[ok] import {mod}")
        except Exception as exc:
            failures.append(f"{mod}: {type(exc).__name__}: {exc}")
            print(f"[fail] import {mod}: {type(exc).__name__}: {exc}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    failures: list[str] = []

    modules = [
        "fall_detection",
        "fall_detection.data.datamodule",
        "fall_detection.data.pipeline",
        "fall_detection.data.transforms",
    ]
    failures.extend(_check_imports(modules))

    cfg = root / "ops/configs/experiments/data_sources.yaml"
    if cfg.is_file():
        print(f"[ok] config exists: {cfg}")
    else:
        print(f"[fail] missing config: {cfg}")
        failures.append(f"missing config: {cfg}")

    tests_dir = root / "tests"
    test_files = sorted(tests_dir.rglob("test_*.py")) if tests_dir.exists() else []
    if test_files:
        print(f"[ok] tests detected: {len(test_files)}")
    else:
        print("[fail] no test_*.py files found under qa/tests/")
        failures.append("missing tests")

    if failures:
        print("\n[summary] smoke audit failed:")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

    print("\n[summary] smoke audit passed")


if __name__ == "__main__":
    main()

