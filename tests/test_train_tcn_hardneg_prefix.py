from __future__ import annotations

import numpy as np

from fall_detection.core.features import FeatCfg
from fall_detection.training.train_tcn import WindowDatasetTCN


def _write_window(path, *, y: int) -> None:
    T, V = 48, 17
    np.savez_compressed(
        path,
        joints=np.zeros((T, V, 2), dtype=np.float32),
        motion=np.zeros((T, V, 2), dtype=np.float32),
        conf=np.ones((T, V), dtype=np.float32),
        mask=np.ones((T, V), dtype=np.uint8),
        y=np.int32(y),
        fps=np.float32(25.0),
        video_id=np.array("vid"),
        w_start=np.int32(0),
        w_end=np.int32(T - 1),
    )


def test_hard_negative_prefix_multiplier_applies_only_to_matching_files(tmp_path) -> None:
    root = tmp_path / "train"
    root.mkdir(parents=True)

    # Base dataset file.
    _write_window(root / "Home_01__Videos__video__base___w000000_000047.npz", y=0)

    # Extra negatives (one matching prefix, one not), plus one positive to ensure skip.
    fp_match = tmp_path / "Coffee_room_02__Videos__video__52___w000000_000047.npz"
    fp_other = tmp_path / "Home_02__Videos__video__37___w000000_000047.npz"
    fp_pos = tmp_path / "Coffee_room_02__Videos__video__59___w000000_000047.npz"
    _write_window(fp_match, y=0)
    _write_window(fp_other, y=0)
    _write_window(fp_pos, y=1)

    ds = WindowDatasetTCN(
        str(root),
        split="train",
        feat_cfg=FeatCfg(),
        fps_default=25.0,
        skip_unlabeled=True,
        mask_joint_p=0.0,
        mask_frame_p=0.0,
        seed=1,
        extra_neg_files=[str(fp_match), str(fp_other), str(fp_pos)],
        extra_neg_mult=2,
        extra_neg_prefixes=["Coffee_room_02__Videos__"],
        extra_neg_prefix_mult=3,
    )

    # Counts:
    # - base root neg: 1
    # - matching extra neg: 2 * 3 = 6
    # - non-matching extra neg: 2
    # - positive extra is skipped
    assert len(ds) == 9
    assert int((ds.labels01 == 0).sum()) == 9
