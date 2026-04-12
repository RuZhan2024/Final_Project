from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fall_detection.data.datamodule import DataModuleConfig, UnifiedWindowDataModule
from fall_detection.data.contracts import DataContractError


@dataclass
class _LayoutStub:
    interim: Path


class _ResolverStub:
    def __init__(self, interim: Path, *, ids: dict[str, tuple[str, ...]] | None = None, raise_for: set[str] | None = None):
        self.layout = _LayoutStub(interim=interim)
        self._ids = ids or {}
        self._raise_for = raise_for or set()

    def split_ids(self, dataset: str, split: str) -> tuple[str, ...]:
        if split in self._raise_for:
            raise DataContractError(f"missing split {split}")
        return self._ids.get(split, ())


def test_resolve_split_non_strict_does_not_fallback_train_val_test_to_all_windows(tmp_path) -> None:
    resolver = _ResolverStub(tmp_path, raise_for={"train", "val", "test"})
    dm = UnifiedWindowDataModule(
        resolver=resolver,
        config=DataModuleConfig(dataset="demo", strict_split_enforcement=False),
    )
    windows = (tmp_path / "a_w0.npz", tmp_path / "b_w1.npz")

    assert dm._resolve_split("train", windows) == ()
    assert dm._resolve_split("val", windows) == ()
    assert dm._resolve_split("test", windows) == ()


def test_resolve_split_predict_can_still_use_all_windows_when_ids_are_missing(tmp_path) -> None:
    resolver = _ResolverStub(tmp_path, raise_for={"predict"})
    dm = UnifiedWindowDataModule(
        resolver=resolver,
        config=DataModuleConfig(dataset="demo", strict_split_enforcement=False),
    )
    windows = (tmp_path / "a_w0.npz", tmp_path / "b_w1.npz")

    assert dm._resolve_split("predict", windows) == windows
