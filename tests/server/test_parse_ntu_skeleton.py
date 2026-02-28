import numpy as np

from pose.parse_ntu_skeleton import read_ntu_skeleton


def _write_tmp(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_read_ntu_skeleton_handles_single_body_canonical_meta(tmp_path):
    # 1 frame, 1 body, canonical one-line metadata (10 fields), 25 joints.
    joints = "\n".join(["1.0 2.0 3.0 0 0 0 0 0 0 0 0 0"] * 25)
    txt = "\n".join(
        [
            "1",          # frames
            "1",          # bodies
            "0 0 0 0 0 0 0 0 0 0",  # body metadata (10 fields)
            "25",         # joints
            joints,
        ]
    )
    path = _write_tmp(tmp_path, "single_body.skeleton", txt)
    arr = read_ntu_skeleton(path)
    assert arr.shape == (1, 25, 3)
    assert np.allclose(arr[0, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_read_ntu_skeleton_picks_primary_body_in_multi_body_frame(tmp_path):
    # Frame has 2 bodies; second one has larger non-zero magnitude and should be selected.
    body0_joints = "\n".join(["0.0 0.0 0.0 0 0 0 0 0 0 0 0 0"] * 25)
    body1_joints = "\n".join(["4.0 5.0 6.0 0 0 0 0 0 0 0 0 0"] * 25)
    txt = "\n".join(
        [
            "1",  # frames
            "2",  # bodies
            "0 0 0 0 0 0 0 0 0 0",
            "25",
            body0_joints,
            "0 0 0 0 0 0 0 0 0 0",
            "25",
            body1_joints,
        ]
    )
    path = _write_tmp(tmp_path, "multi_body.skeleton", txt)
    arr = read_ntu_skeleton(path)
    assert arr.shape == (1, 25, 3)
    assert np.allclose(arr[0, 0], np.array([4.0, 5.0, 6.0], dtype=np.float32))


def test_read_ntu_skeleton_accepts_legacy_10_line_body_metadata(tmp_path):
    # Metadata encoded as 10 lines instead of one 10-field line.
    meta_lines = "\n".join(["0"] * 10)
    joints = "\n".join(["7.0 8.0 9.0 0 0 0 0 0 0 0 0 0"] * 25)
    txt = "\n".join(
        [
            "1",          # frames
            "1",          # bodies
            meta_lines,   # 10 metadata lines
            "25",         # joints
            joints,
        ]
    )
    path = _write_tmp(tmp_path, "legacy_meta.skeleton", txt)
    arr = read_ntu_skeleton(path)
    assert arr.shape == (1, 25, 3)
    assert np.allclose(arr[0, 0], np.array([7.0, 8.0, 9.0], dtype=np.float32))
