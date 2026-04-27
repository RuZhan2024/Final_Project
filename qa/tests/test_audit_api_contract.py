from __future__ import annotations

from pathlib import Path

from ops.scripts.audit_api_contract import collect_backend_routes, collect_frontend_calls


def test_api_contract_collectors_normalize_query_and_template_params(tmp_path: Path) -> None:
    # The audit tools intentionally ignore concrete query values and template
    # names so route drift is detected by endpoint shape rather than by local
    # variable spelling in the frontend.
    root = tmp_path
    routes_dir = root / "applications" / "backend" / "routes"
    app_dir = root / "applications" / "frontend" / "src"
    routes_dir.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    (routes_dir / "events.py").write_text(
        '\n'.join(
            [
                'from fastapi import APIRouter',
                'router = APIRouter()',
                '@router.get("/api/events/{event_id}/status")',
                'def x():',
                '    return {}',
            ]
        ),
        encoding="utf-8",
    )

    (app_dir / "events.js").write_text(
        '\n'.join(
            [
                'const p = `/api/events/${eventId}/status?resident_id=${rid}`;',
                "const x = '/api/events/summary?resident_id=1';",
            ]
        ),
        encoding="utf-8",
    )

    backend = collect_backend_routes(root)
    frontend = collect_frontend_calls(root)

    assert ("GET", "/api/events/{param}/status") in backend
    assert ("GET", "/api/events/{param}/status") in frontend
    assert ("GET", "/api/events/summary") in frontend


def test_api_contract_collectors_extract_http_methods_from_api_request(tmp_path: Path) -> None:
    # `apiRequest(...)` is the shared frontend transport. If method extraction
    # breaks here, the contract audit silently misreports write endpoints as GETs.
    root = tmp_path
    routes_dir = root / "applications" / "backend" / "routes"
    app_dir = root / "applications" / "frontend" / "src"
    routes_dir.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)

    (routes_dir / "events.py").write_text(
        '\n'.join(
            [
                "from fastapi import APIRouter",
                "router = APIRouter()",
                '@router.put("/api/events/{event_id}/status")',
                '@router.post("/api/events/test_fall")',
                "def x():",
                "    return {}",
            ]
        ),
        encoding="utf-8",
    )

    (app_dir / "events.js").write_text(
        '\n'.join(
            [
                "apiRequest(API_BASE, `/api/events/${eventId}/status`, {",
                '  method: "PUT",',
                "  body: { status },",
                "});",
                "apiRequest(API_BASE, '/api/events/test_fall', { method: 'POST' });",
            ]
        ),
        encoding="utf-8",
    )

    backend = collect_backend_routes(root)
    frontend = collect_frontend_calls(root)

    assert ("PUT", "/api/events/{param}/status") in backend
    assert ("POST", "/api/events/test_fall") in backend
    assert ("PUT", "/api/events/{param}/status") in frontend
    assert ("POST", "/api/events/test_fall") in frontend
