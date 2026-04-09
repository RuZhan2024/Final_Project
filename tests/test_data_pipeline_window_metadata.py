from __future__ import annotations

import numpy as np

from fall_detection.data.pipeline import export_windows


def test_export_windows_writes_frame_index_metadata(tmp_path) -> None:
    pose_path = tmp_path / "vid.npz"
    out_dir = tmp_path / "windows"

    fps = 25.0
    n_frames = 100
    t_ms = (np.arange(n_frames, dtype=np.float32) / fps) * 1000.0
    xy = np.zeros((n_frames, 33, 2), dtype=np.float32)
    conf = np.ones((n_frames, 33), dtype=np.float32)

    np.savez_compressed(
        pose_path,
        sequence_id=np.asarray(["vid"]),
        source_path=np.asarray([str(tmp_path / "vid.mp4")]),
        t_ms=t_ms,
        xy=xy,
        conf=conf,
        fps=np.asarray([fps], dtype=np.float32),
    )

    outputs = export_windows(
        pose_sequence_paths=(pose_path,),
        splits={"train": ("vid",), "val": (), "test": (), "unlabeled": ()},
        labels={"vid": ()},
        out_dir=out_dir,
        target_fps=fps,
        window_frames=48,
        stride_frames=12,
        conf_gate=0.2,
        overwrite=True,
    )

    assert len(outputs) >= 2

    with np.load(outputs[0], allow_pickle=False) as z0:
        assert int(np.asarray(z0["w_start"]).reshape(-1)[0]) == 0
        assert int(np.asarray(z0["w_end"]).reshape(-1)[0]) == 47

    with np.load(outputs[1], allow_pickle=False) as z1:
        assert int(np.asarray(z1["w_start"]).reshape(-1)[0]) == 12
        assert int(np.asarray(z1["w_end"]).reshape(-1)[0]) == 59
