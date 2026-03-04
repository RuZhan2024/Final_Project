#!/usr/bin/env python3
"""Smoke test for checkpoint schema + EMA compatibility.

Covers:
A) save/load with ema_state
B) load old checkpoint without ema_state
C) save_ckpt kwargs style vs legacy bundle style normalization
D) EMA vs raw weight divergence sanity
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

from fall_detection.core.ckpt import load_ckpt, save_ckpt
from fall_detection.core.ema import EMA


def _first_float_tensor(sd: dict[str, torch.Tensor]) -> torch.Tensor:
    for v in sd.values():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            return v
    raise RuntimeError("no float tensor found")


def _any_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    keys = sorted(set(a.keys()) & set(b.keys()))
    for k in keys:
        ta, tb = a[k], b[k]
        if isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor):
            if ta.shape == tb.shape and torch.is_floating_point(ta) and torch.is_floating_point(tb):
                if not torch.allclose(ta, tb):
                    return True
    return False


def main() -> None:
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    ema = EMA(model, decay=0.9)

    with tempfile.TemporaryDirectory(prefix="smoke_ckpt_ema_") as td:
        best_p = os.path.join(td, "best.pt")
        last_p = os.path.join(td, "last.pt")
        old_p = os.path.join(td, "old_no_ema.pt")
        legacy_p = os.path.join(td, "legacy_style.pt")
        canon_p = os.path.join(td, "canonical_style.pt")

        # Move model weights, then update EMA so shadow differs from raw.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.update(model)

        with ema.use(model):
            ema_sd_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        assert _any_diff(raw_sd, ema_sd_weights), "EMA weights should differ from raw weights in smoke setup"

        # D) Save last/raw and best/EMA style checkpoints.
        save_ckpt(
            last_p,
            arch="gcn",
            state_dict=raw_sd,
            model_cfg={"hidden": 8},
            feat_cfg={"use_motion": True},
            data_cfg={"fps_default": 25.0},
            meta={"kind": "last_raw"},
        )
        save_ckpt(
            best_p,
            arch="gcn",
            state_dict=ema_sd_weights,
            model_cfg={"hidden": 8},
            feat_cfg={"use_motion": True},
            data_cfg={"fps_default": 25.0},
            meta={"kind": "best_ema"},
            ema_state=ema.state_dict(),
        )

        b_best = load_ckpt(best_p)
        b_last = load_ckpt(last_p)
        assert "ema_state" in b_best, "best checkpoint should contain ema_state in this smoke case"
        assert "ema_state" not in b_last, "raw last checkpoint should not contain ema_state in this smoke case"
        assert _any_diff(b_best["state_dict"], b_last["state_dict"]), "best(EMA) and last(raw) should differ"

        # B) Old ckpt without ema_state should load.
        torch.save({"state_dict": raw_sd, "arch": "gcn"}, old_p)
        b_old = load_ckpt(old_p)
        assert isinstance(b_old, dict) and "state_dict" in b_old
        assert "ema_state" not in b_old

        # C) save_ckpt kwargs style vs legacy bundle style normalization.
        save_ckpt(
            canon_p,
            arch="gcn",
            state_dict=raw_sd,
            model_cfg={"a": 1},
            feat_cfg={"b": 2},
            data_cfg={"fps_default": 25.0},
            meta={"via": "kwargs"},
        )
        save_ckpt(
            legacy_p,
            {
                "arch": "gcn",
                "state_dict": raw_sd,
                "model_cfg": {"a": 1},
                "feat_cfg": {"b": 2},
                "data_cfg": {"fps_default": 25.0},
                "meta": {"via": "legacy"},
            },
        )
        b_canon = load_ckpt(canon_p)
        b_legacy = load_ckpt(legacy_p)

        req_keys = {"ckpt_version", "version", "arch", "state_dict", "model_cfg", "feat_cfg", "data_cfg", "meta"}
        assert req_keys.issubset(set(b_canon.keys()))
        assert req_keys.issubset(set(b_legacy.keys()))
        assert set(b_canon.keys()) == set(b_legacy.keys())

        # A) explicit with ema_state already checked above.
        print("[ok] smoke_ckpt_ema passed")


if __name__ == "__main__":
    main()
