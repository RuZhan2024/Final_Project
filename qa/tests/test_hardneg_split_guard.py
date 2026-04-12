from __future__ import annotations

from pathlib import Path

import pytest

from fall_detection.training.train_gcn import _validate_hard_neg_paths as _validate_gcn
from fall_detection.training.train_tcn import _validate_hard_neg_paths as _validate_tcn


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return str(path)


def test_tcn_hardneg_guard_rejects_val_and_test_paths(tmp_path: Path) -> None:
    train_dir = tmp_path / "windows" / "train"
    p_train = _touch(train_dir / "ok.npz")
    p_val = _touch(tmp_path / "windows" / "val" / "bad.npz")
    p_test = _touch(tmp_path / "windows" / "test" / "bad2.npz")

    _validate_tcn([p_train], train_dir=str(train_dir), allow_nontrain=False)

    with pytest.raises(ValueError):
        _validate_tcn([p_val], train_dir=str(train_dir), allow_nontrain=False)

    with pytest.raises(ValueError):
        _validate_tcn([p_test], train_dir=str(train_dir), allow_nontrain=False)


def test_gcn_hardneg_guard_rejects_nontrain_unknown_paths(tmp_path: Path) -> None:
    train_dir = tmp_path / "windows" / "train"
    p_train = _touch(train_dir / "ok.npz")
    p_other = _touch(tmp_path / "hardneg_pool" / "candidate.npz")

    _validate_gcn([p_train], train_dir=str(train_dir), allow_nontrain=False)

    with pytest.raises(ValueError):
        _validate_gcn([p_other], train_dir=str(train_dir), allow_nontrain=False)

    # Explicit override keeps backward compatibility for intentionally custom workflows.
    _validate_gcn([p_other], train_dir=str(train_dir), allow_nontrain=True)

