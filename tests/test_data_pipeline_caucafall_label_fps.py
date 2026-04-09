from __future__ import annotations

from pathlib import Path

from fall_detection.data.pipeline import discover_sequences_with_labels


def test_discover_caucafall_sequences_uses_caucafall_nominal_fps_for_raw_label_rebuild(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "caucafall" / "fall_scene"
    raw_dir.mkdir(parents=True)

    (raw_dir / "clip.mp4").write_bytes(b"")
    (raw_dir / "classes.txt").write_text("adl\nfall\n", encoding="utf-8")
    (raw_dir / "clip_0001.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (raw_dir / "clip_0002.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    records = discover_sequences_with_labels(raw_dir=raw_dir.parent)

    assert len(records) == 1
    span = records[0].fall_spans_ms[0]
    assert abs(span[0] - (1000.0 / 23.0)) < 1e-4
    assert abs(span[1] - (3000.0 / 23.0)) < 1e-4
