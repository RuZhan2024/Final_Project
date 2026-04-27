from __future__ import annotations

from pathlib import Path

from ops.scripts.audit_api_v1_parity import collect_api_paths


def test_collect_api_paths(tmp_path: Path) -> None:
    # The parity audit scans route decorators only, so this test guards the
    # minimum contract: both `/api/*` and `/api/v1/*` variants stay discoverable.
    routes = tmp_path / "server" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "x.py").write_text(
        '\n'.join(
            [
                "from fastapi import APIRouter",
                "router = APIRouter()",
                '@router.get("/api/foo")',
                '@router.get("/api/v1/foo")',
                '@router.post("/api/bar")',
                "def _x():",
                "    return {}",
            ]
        ),
        encoding="utf-8",
    )
    paths = collect_api_paths(routes)
    assert "/api/foo" in paths
    assert "/api/v1/foo" in paths
    assert "/api/bar" in paths
