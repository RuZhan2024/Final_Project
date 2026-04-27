#!/usr/bin/env python3
"""Static API contract audit: frontend calls vs backend routes."""

from __future__ import annotations

import re
from pathlib import Path


ROUTE_RE = re.compile(r'@router\.(get|post|put|delete|patch)\("([^"]+)"\)')
API_STR_RE = re.compile(r"""["'`](/api[^"'`]+)["'`]""")
API_REQ_RE = re.compile(r"""apiRequest\([^,]+,\s*["'`](/api[^"'`]+)["'`]\s*(?:,\s*\{(?P<opts>.*?)\})?\s*\)""", re.DOTALL)
METHOD_RE = re.compile(r"""method\s*:\s*["'](GET|POST|PUT|DELETE|PATCH)["']""", re.IGNORECASE)
TEMPLATE_PARAM_RE = re.compile(r"\$\{[^}]+\}")
ROUTE_PARAM_RE = re.compile(r"\{[^}]+\}")


def _norm_path(path: str) -> str:
    p = path.split("?", 1)[0]
    p = TEMPLATE_PARAM_RE.sub("{param}", p)
    p = ROUTE_PARAM_RE.sub("{param}", p)
    return p


def collect_backend_routes(root: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    routes_dir = root / "applications" / "backend" / "routes"
    for fp in sorted(routes_dir.glob("*.py")):
        txt = fp.read_text(encoding="utf-8")
        for m in ROUTE_RE.finditer(txt):
            method = m.group(1).upper()
            path = _norm_path(m.group(2).strip())
            out.add((method, path))
    return out


def collect_frontend_calls(root: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    apps_dir = root / "applications" / "frontend" / "src"
    patterns = ("*.js", "*.jsx", "*.ts", "*.tsx")
    for pat in patterns:
        for fp in sorted(apps_dir.rglob(pat)):
            txt = fp.read_text(encoding="utf-8")
            for m in API_REQ_RE.finditer(txt):
                path = _norm_path(m.group(1).strip())
                opts = m.group("opts") or ""
                mm = METHOD_RE.search(opts)
                method = (mm.group(1).upper() if mm else "GET")
                out.add((method, path))

            # Fallback: plain /api strings not seen in apiRequest() calls, default GET.
            for m in API_STR_RE.finditer(txt):
                path = _norm_path(m.group(1).strip())
                if not any(p == path for _meth, p in out):
                    out.add(("GET", path))
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    backend = collect_backend_routes(root)
    frontend = collect_frontend_calls(root)
    missing = sorted([f"{m} {p}" for (m, p) in frontend if (m, p) not in backend])

    print(f"[info] backend_routes={len(backend)} frontend_calls={len(frontend)}")
    if missing:
        print("[fail] frontend API paths missing on backend:")
        for p in missing:
            print(f" - {p}")
        raise SystemExit(1)

    print("[ok] static API contract paths aligned")


if __name__ == "__main__":
    main()
