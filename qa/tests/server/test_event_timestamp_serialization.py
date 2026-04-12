from __future__ import annotations

from applications.backend.services.events_service import map_v1_event_row, map_v2_event_row


def test_map_v2_event_row_normalizes_naive_db_timestamp_to_utc_iso():
    row = {
        "id": 1,
        "ts": "2026-04-11 16:59:14",
        "type": "fall",
        "status": "pending_review",
        "score": 0.91,
        "operating_point_id": 1,
        "model_code": "TCN",
        "model_family": "TCN",
        "meta": "{\"event_source\":\"realtime\"}",
    }

    mapped = map_v2_event_row(row)

    assert mapped["event_time"] == "2026-04-11T16:59:14+00:00"
    assert mapped["ts"] == "2026-04-11T16:59:14+00:00"


def test_map_v1_event_row_normalizes_naive_db_timestamp_to_utc_iso():
    row = {
        "id": 2,
        "ts": "2026-04-11 16:50:01",
        "type": "fall",
        "severity": "high",
        "model_code": "TCN",
        "score": 0.98,
        "meta": "{\"source\":\"monitor\"}",
    }

    mapped = map_v1_event_row(row)

    assert mapped["event_time"] == "2026-04-11T16:50:01+00:00"
    assert mapped["ts"] == "2026-04-11T16:50:01+00:00"
