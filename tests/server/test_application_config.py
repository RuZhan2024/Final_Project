from __future__ import annotations

from server.application import create_app
from server.config import get_app_config


def test_app_config_uses_env_for_cors_and_runtime_paths(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,https://example.com")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", "tmp/test_app.sqlite3")
    monkeypatch.setenv("EVENT_CLIPS_DIR", "tmp/test_event_clips")
    monkeypatch.setenv("SAFE_GUARD_SQLITE_PATH", "tmp/test_notifications.sqlite3")
    get_app_config.cache_clear()

    cfg = get_app_config()

    assert cfg.cors_allowed_origins == ("http://localhost:3000", "https://example.com")
    assert cfg.db_backend == "sqlite"
    assert str(cfg.sqlite_path).endswith("tmp/test_app.sqlite3")
    assert str(cfg.event_clips_dir).endswith("tmp/test_event_clips")
    assert str(cfg.notification_sqlite_path).endswith("tmp/test_notifications.sqlite3")

    get_app_config.cache_clear()


def test_create_app_without_runtime_routes_is_still_assemblable(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    get_app_config.cache_clear()

    app = create_app(include_runtime_routes=False)

    assert hasattr(app.state, "app_config")
    paths = {route.path for route in app.router.routes}
    assert "/api/health" in paths
    assert "/api/events" in paths
    assert "/api/monitor/predict_window" not in paths

    get_app_config.cache_clear()
