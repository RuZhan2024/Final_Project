#!/usr/bin/env python3
"""Audit that every /api/* route has a matching /api/v1/* alias."""

from __future__ import annotations

import re
from pathlib import Path


ROUTE_RE = re.compile(r'@router\.(?:get|post|put|delete|patch)\("([^"]+)"\)')


def collect_api_paths(routes_dir: Path) -> set[str]:
    out: set[str] = set()
    for fp in sorted(routes_dir.glob("*.py")):
        txt = fp.read_text(encoding="utf-8")
        for p in ROUTE_RE.findall(txt):
            if p.startswith("/api/"):
                out.add(p)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    routes_dir = root / "applications" / "backend" / "routes"
    paths = collect_api_paths(routes_dir)

    missing: list[tuple[str, str]] = []
    for p in sorted(paths):
        if p.startswith("/api/v1/"):
            continue
        v1 = "/api/v1/" + p[len("/api/") :]
        if v1 not in paths:
            missing.append((p, v1))

    print(f"[info] api_routes={len(paths)}")
    if missing:
        print("[fail] missing /api/v1 aliases:")
        for p, v1 in missing:
            print(f" - {p} -> {v1}")
        raise SystemExit(1)

    print("[ok] /api and /api/v1 route parity passed")


if __name__ == "__main__":
    main()
