from __future__ import annotations

import numpy as np

from fall_detection.core.features import read_window_npz


def test_read_window_npz_contract(tmp_path) -> None:
    fp = tmp_path / "w.npz"
    T, V = 48, 17
    np.savez_compressed(
        fp,
        joints=np.zeros((T, V, 2), dtype=np.float32),
        motion=np.zeros((T, V, 2), dtype=np.float32),
        conf=np.ones((T, V), dtype=np.float32),
        mask=np.ones((T, V), dtype=np.uint8),
        y=np.int32(0),
        fps=np.float32(25.0),
        video_id=np.array("vid"),
        w_start=np.int32(0),
        w_end=np.int32(T - 1),
    )
    joints, motion, conf, mask, fps, meta = read_window_npz(str(fp), fps_default=25.0)
    assert joints.shape == (T, V, 2)
    assert motion is not None and motion.shape == (T, V, 2)
    assert conf is not None and conf.shape == (T, V)
    assert mask is not None and mask.shape == (T, V)
    assert float(fps) == 25.0
    assert int(meta.y) == 0

