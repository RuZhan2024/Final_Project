from __future__ import annotations

import numpy as np

from fall_detection.data.adapters import build_adapter


def test_adapter_output_contract(tmp_path) -> None:
    fp = tmp_path / "seq.npz"
    T = 10
    joints = np.zeros((T, 33, 2), dtype=np.float32)
    conf = np.ones((T, 33), dtype=np.float32)
    np.savez_compressed(
        fp,
        joints=joints,
        conf=conf,
        fps=np.float32(25.0),
        y=np.int32(0),
        seq_id=np.array("seq_1"),
        video_id=np.array("video_1"),
    )

    adapter = build_adapter("le2i")
    out = adapter.load_sequence(str(fp), fps_default=25.0)
    assert out.joints_xy.shape == (T, 33, 2)
    assert out.conf is not None and out.conf.shape == (T, 33)
    assert out.mask is None or out.mask.shape == (T, 33)
    assert float(out.fps) == 25.0
    assert int(out.meta["y"]) == 0

    adapter17 = build_adapter("le2i", joint_layout="internal17")
    out17 = adapter17.load_sequence(str(fp), fps_default=25.0)
    assert out17.joints_xy.shape == (T, 17, 2)
    assert out17.conf is not None and out17.conf.shape == (T, 17)
