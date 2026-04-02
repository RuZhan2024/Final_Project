from fall_detection.pose.preprocess_config import (
    DEFAULT_POSE_PREPROCESS_CFG,
    get_pose_preprocess_cfg_from_data_cfg,
    normalize_pose_preprocess_cfg,
)


def test_normalize_pose_preprocess_cfg_sanitizes_and_falls_back():
    cfg = normalize_pose_preprocess_cfg(
        {
            "conf_thr": "0.35",
            "smooth_k": "7",
            "max_gap": -4,
            "fill_conf": "linear",
            "normalize": "shoulder",
            "pelvis_fill": "forward",
            "rotate": "shoulders",
            "ignored": 123,
        }
    )

    assert cfg["conf_thr"] == 0.35
    assert cfg["smooth_k"] == 7
    assert cfg["max_gap"] == 0
    assert cfg["fill_conf"] == "linear"
    assert cfg["normalize"] == "shoulder"
    assert cfg["pelvis_fill"] == "forward"
    assert cfg["rotate"] == "shoulders"


def test_get_pose_preprocess_cfg_from_data_cfg_merges_with_fallback():
    fallback = dict(DEFAULT_POSE_PREPROCESS_CFG)
    fallback["smooth_k"] = 9
    fallback["normalize"] = "none"

    cfg = get_pose_preprocess_cfg_from_data_cfg(
        {
            "pose_preprocess": {
                "smooth_k": 3,
                "fill_conf": "min_neighbors",
            }
        },
        fallback=fallback,
    )

    assert cfg["smooth_k"] == 3
    assert cfg["fill_conf"] == "min_neighbors"
    assert cfg["normalize"] == "none"
    assert cfg["pelvis_fill"] == DEFAULT_POSE_PREPROCESS_CFG["pelvis_fill"]
