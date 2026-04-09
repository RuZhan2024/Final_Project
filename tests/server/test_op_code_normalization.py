from __future__ import annotations

from server import core as core_mod
from server.repositories import settings_repository as repo
from server.services import monitor_context_service as ctx_service
from server.services import settings_service


class _NoopCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        self.current = self.rows.pop(0) if self.rows else None

    def fetchone(self):
        return self.current or {}


class _Conn:
    def __init__(self, rows):
        self.rows = list(rows)

    def cursor(self):
        return _NoopCursor(self.rows)


def test_apply_settings_update_inmem_normalizes_op_code() -> None:
    payload = core_mod.SettingsUpdatePayload(active_op_code="op2")
    core_mod.apply_settings_update_inmem(payload, resident_id=99)
    system, _deploy = core_mod.get_inmem_settings(99).values()
    assert system["active_op_code"] == "OP-2"


def test_build_settings_response_normalizes_active_op_code() -> None:
    system = {"active_dataset_code": "caucafall", "active_model_code": "TCN", "active_op_code": "op3"}
    deploy = {}
    body = settings_service.build_settings_response(1, system, deploy, db_available=False)
    assert body["system"]["active_op_code"] == "OP-3"
    assert body["active_op_code"] == "OP-3"


def test_monitor_request_context_normalizes_requested_and_db_op_code() -> None:
    conn = _Conn([{"active_op_code": "op3"}])

    ctx = ctx_service.load_monitor_request_context(
        conn=conn,
        resident_id=1,
        input_source="camera",
        mode="tcn",
        dataset_code="caucafall",
        op_code="op2",
        event_location=None,
        active_model_code="TCN",
        requested_use_mc=False,
        requested_mc_M=10,
        is_replay=False,
        ensure_system_settings_schema=lambda _conn: None,
        detect_variants=lambda _conn: {"settings": "v2"},
        table_exists=lambda _conn, name: name == "system_settings",
    )

    assert ctx.op_code == "OP-2"
    assert ctx.runtime.op_code == "OP-2"


def test_load_settings_snapshot_normalizes_active_op_code(monkeypatch) -> None:
    system = {}
    deploy = {}
    conn = _Conn([{"active_op_code": "op3"}])
    monkeypatch.setattr(repo, "_ensure_system_settings_schema", lambda _conn: None)
    monkeypatch.setattr(repo, "_table_exists", lambda _conn, name: name == "settings")

    repo.load_settings_snapshot(conn, 1, system, deploy)

    assert system["active_op_code"] == "OP-3"
