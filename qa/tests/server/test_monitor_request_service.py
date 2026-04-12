import logging

from applications.backend.schemas import MonitorPredictPayload
from applications.backend.services.monitor_request_service import MySQLError, prepare_monitor_request


def test_prepare_monitor_request_falls_back_when_db_defaults_fail():
    payload = MonitorPredictPayload.model_validate(
        {
            "session_id": "test-session",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "resident_id": 1,
            "xy": [[[0.1, 0.2]] for _ in range(48)],
            "conf": [[0.9] for _ in range(48)],
        }
    )

    class _FailingConn:
        def __enter__(self):
            raise MySQLError("db unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    prepared = prepare_monitor_request(
        payload=payload,
        logger=logging.getLogger("test"),
        get_conn=lambda: _FailingConn(),
        normalize_dataset_code=lambda code: str(code or "caucafall"),
        coerce_bool=lambda value, default=False: bool(default if value is None else value),
        decode_quantized_raw_window=lambda *_a, **_k: (None, None),
        raw_window_stats=lambda *_a, **_k: {"raw_len": 0, "raw_fps_est": None},
        resolve_runtime_fps=lambda **_k: (23.0, 23.0),
        resolve_monitor_specs=lambda **_k: {
            "mode": "tcn",
            "tcn_key": "tcn:caucafall",
            "gcn_key": "gcn:caucafall",
            "guard_spec_key": "tcn:caucafall",
            "primary_spec_key": "tcn:caucafall",
            "primary_model_key": "tcn",
        },
        get_pose_preprocess_cfg=lambda *_a, **_k: None,
        resample_pose_window=lambda **_k: ([], [], None, None, None),
        preprocess_online_raw_window=lambda xy, conf, **_k: (xy, conf),
        direct_window_stats=lambda xy, conf, effective_fps: {
            "raw_len": len(xy),
            "raw_fps_est": float(effective_fps),
        },
        get_deploy_specs=lambda: {"models": {}},
        ensure_system_settings_schema=lambda _conn: None,
        detect_variants=lambda _conn: {},
        table_exists=lambda _conn, _table: False,
    )

    assert prepared.dataset_code == "caucafall"
    assert prepared.mode == "tcn"
    assert prepared.session_id == "test-session"
    assert prepared.xy
