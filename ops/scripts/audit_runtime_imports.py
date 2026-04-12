#!/usr/bin/env python3
"""Audit deploy/runtime import graph for training-module leakage."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys


def _imports_from_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and mod:
                out.append("." * node.level + mod)
            elif mod:
                out.append(mod)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="ml/src/fall_detection/deploy,applications/backend/deploy_runtime.py")
    ap.add_argument("--forbidden_prefix", default="fall_detection.training")
    args = ap.parse_args()

    targets: list[Path] = []
    for raw in [p.strip() for p in args.paths.split(",") if p.strip()]:
        p = Path(raw)
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            targets.append(p)

    forbidden = str(args.forbidden_prefix).strip()
    hits: list[str] = []
    for path in targets:
        for imp in _imports_from_file(path):
            norm = imp.lstrip(".")
            if norm == forbidden or norm.startswith(forbidden + "."):
                hits.append(f"{path}: import {imp}")

    if hits:
        print("[fail] runtime import audit failed:")
        for h in hits:
            print(f" - {h}")
        sys.exit(1)

    print("[ok] runtime import audit passed")


if __name__ == "__main__":
    main()
