#!/usr/bin/env python3
"""No-training backend API contract smoke checks."""

from __future__ import annotations

from fastapi.testclient import TestClient

from applications.backend.app import app


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[fail] {msg}")


def main() -> None:
    c = TestClient(app)

    r = c.get("/api/health")
    _check(r.status_code == 200 and bool(r.json().get("ok")), "GET /api/health")

    r = c.get("/api/v1/health")
    _check(r.status_code == 200 and bool(r.json().get("ok")), "GET /api/v1/health")

    r = c.get("/api/deploy/specs")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("specs"), list), "GET /api/deploy/specs")

    r = c.get("/api/v1/deploy/specs")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("specs"), list), "GET /api/v1/deploy/specs")

    r = c.get("/api/v1/spec")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("specs"), list), "GET /api/v1/spec")

    r = c.get("/api/v1/models/summary")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("models"), list), "GET /api/v1/models/summary")

    r = c.get("/api/deploy/modes")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("deploy_modes"), dict), "GET /api/deploy/modes")

    r = c.get("/api/v1/deploy/modes")
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    _check(r.status_code == 200 and isinstance(body.get("deploy_modes"), dict), "GET /api/v1/deploy/modes")

    r = c.post("/api/notifications/test", json={"message": "smoke"})
    _check(r.status_code == 200 and bool(r.json().get("ok")), "POST /api/notifications/test")

    r = c.post("/api/v1/notifications/test", json={"message": "smoke"})
    _check(r.status_code == 200 and bool(r.json().get("ok")), "POST /api/v1/notifications/test")

    r = c.put("/api/events/123/status", json={"status": "confirmed_fall"})
    _check(r.status_code == 200 and bool(r.json().get("ok")), "PUT /api/events/{id}/status")

    r = c.put("/api/v1/events/123/status", json={"status": "confirmed_fall"})
    _check(r.status_code == 200 and bool(r.json().get("ok")), "PUT /api/v1/events/{id}/status")

    r = c.post("/api/v1/monitor/reset_session?session_id=smoke")
    _check(r.status_code == 200 and bool(r.json().get("ok")), "POST /api/v1/monitor/reset_session")

    r = c.get("/api/v1/settings")
    _check(r.status_code == 200 and isinstance(r.json().get("system"), dict), "GET /api/v1/settings")

    r = c.get("/api/v1/events?resident_id=1&limit=5")
    _check(r.status_code == 200 and isinstance(r.json().get("events"), list), "GET /api/v1/events")

    r = c.get("/api/v1/events/summary?resident_id=1")
    _check(r.status_code == 200 and isinstance(r.json().get("today"), dict), "GET /api/v1/events/summary")

    r = c.get("/api/v1/caregivers?resident_id=1")
    _check(r.status_code == 200 and isinstance(r.json().get("caregivers"), list), "GET /api/v1/caregivers")

    r = c.get("/api/v1/operating_points?model_code=TCN")
    _check(r.status_code == 200 and isinstance(r.json().get("operating_points"), list), "GET /api/v1/operating_points")

    r = c.get("/api/v1/summary")
    _check(r.status_code == 200 and isinstance(r.json().get("today"), dict), "GET /api/v1/summary")

    r = c.get("/api/v1/dashboard/summary")
    _check(r.status_code == 200 and isinstance(r.json().get("today"), dict), "GET /api/v1/dashboard/summary")

    r = c.post("/api/v1/events/test_fall")
    _check(r.status_code == 200 and ("ok" in r.json()), "POST /api/v1/events/test_fall")

    print("[ok] api contract smoke passed")


if __name__ == "__main__":
    main()
