from __future__ import annotations

import torch

from fall_detection.core.ctr_gcn import CTRGCNConfig, build_mediapipe_adjacency, normalize_adjacency
from fall_detection.core.features import FeatCfg, split_ctr_gcn_two_stream
from fall_detection.core.models import build_model
from fall_detection.training.train_ctr_gcn import _select_monitor_metric


def test_project_adapted_ctr_gcn_forward_shape_and_refinement_terms() -> None:
    A = normalize_adjacency(build_mediapipe_adjacency(num_joints=6))
    assert A.shape == (6, 6)

    model_cfg = CTRGCNConfig(
        num_joints=6,
        channels=(8, 8),
        rel_channels=4,
        ctr_rank=3,
        temporal_kernel=3,
        dropout=0.0,
    ).to_dict()
    model_cfg["in_feats"] = 5

    model = build_model(
        "ctr_gcn",
        model_cfg,
        FeatCfg(use_motion=True, use_conf_channel=True).to_dict(),
    )
    model.eval()

    x = torch.randn(2, 10, 6, 5)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2,)
    keys = set(model.state_dict().keys())
    assert any("graph.theta" in k for k in keys)
    assert any("ctr_u" in k for k in keys)
    assert any("ctr_v" in k for k in keys)


def test_ctr_gcn_config_accepts_string_schedule() -> None:
    cfg = CTRGCNConfig.from_dict(
        {
            "channel_schedule": "8,16,32",
            "rel_channels": 4,
            "ctr_rank": 3,
        }
    )

    assert cfg.channels == (8, 16, 32)
    assert cfg.rel_channels == 4
    assert cfg.ctr_rank == 3


def test_two_stream_ctr_gcn_forward_shape_and_split_contract() -> None:
    feat_cfg = FeatCfg(use_motion=True, use_bone=True, use_conf_channel=True)
    X = torch.randn(10, 6, 7).numpy().astype("float32")
    xj, xb = split_ctr_gcn_two_stream(X, feat_cfg)

    assert xj.shape == (10, 6, 5)
    assert xb.shape == (10, 6, 3)

    model_cfg = CTRGCNConfig(
        num_joints=6,
        channels=(8, 8),
        rel_channels=4,
        ctr_rank=3,
        temporal_kernel=3,
        dropout=0.0,
        two_stream=True,
        fuse="concat",
    ).to_dict()
    model_cfg["in_feats_j"] = 5
    model_cfg["in_feats_b"] = 3

    model = build_model("ctr_gcn", model_cfg, feat_cfg.to_dict())
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(xj).unsqueeze(0), torch.from_numpy(xb).unsqueeze(0))

    assert logits.shape == (1,)


def test_ctr_gcn_monitor_falls_back_when_ap_is_nan() -> None:
    metric, effective_monitor = _select_monitor_metric(
        "ap",
        float("nan"),
        f1=0.42,
        train_loss=0.9,
    )

    assert metric == 0.42
    assert effective_monitor == "f1"
