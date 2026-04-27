"""Unified DataModule implementation for training/eval/serving parity.

This module provides a deterministic, split-aware window loader with optional
transform and cache layers. It is designed to satisfy `DataModuleProtocol`
without binding the project to a specific deep-learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..preprocessing import FeatCfg, build_canonical_input, read_window_npz
from ..training.contracts import Batch, DataModuleProtocol, Stage, TransformPipelineProtocol
from .cache import CacheKeyParts, InMemoryWindowCache, WindowCacheProtocol, make_cache_key
from .contracts import DataContractError
from .resolver import DataPathResolver
from .transforms import IdentityTransformPipeline

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataModuleConfig:
    """Typed runtime configuration for `UnifiedWindowDataModule`.

    Complexity:
        O(1) immutable config container.
    """

    dataset: str
    batch_size: int = 8
    num_workers: int = 0
    drop_last: bool = False
    strict_split_enforcement: bool = True


class UnifiedWindowDataModule(DataModuleProtocol):
    """Deterministic NPZ window data module.

    Notes:
        - Split assignments are sourced from `DataPathResolver`.
        - In strict mode, missing split IDs raise to prevent train/val/test leakage.

    Complexity:
        Setup is O(n) in discovered windows plus optional metadata scans.
    """

    def __init__(
        self,
        *,
        resolver: DataPathResolver,
        config: DataModuleConfig,
        feat_cfg: FeatCfg | None = None,
        transforms: TransformPipelineProtocol | None = None,
        cache: WindowCacheProtocol | None = None,
    ) -> None:
        self._resolver = resolver
        self._config = config
        self._feat_cfg = feat_cfg or FeatCfg()
        self._transforms = transforms or IdentityTransformPipeline()
        self._cache = cache or InMemoryWindowCache()
        self._splits: dict[str, tuple[Path, ...]] = {"train": (), "val": (), "test": (), "predict": ()}
        self._setup = False

    @property
    def dataset_name(self) -> str:
        return self._config.dataset

    @property
    def preprocess_signature(self) -> str:
        payload = {
            "dataset": self._config.dataset,
            "manifest_hash": self._resolver.dataset_manifest_hash(self._config.dataset),
            "feat_cfg": self._feat_cfg.to_dict(),
            "transforms": {
                "name": self._transforms.name,
                "version": self._transforms.version,
                "signature": self._transforms.signature(),
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def setup(self, stage: Stage | None = None) -> None:
        all_windows = self._resolver.list_processed_window_files(self._config.dataset)
        if not all_windows:
            self._splits = {"train": (), "val": (), "test": (), "predict": ()}
            self._setup = True
            return

        if stage in {None, "fit"}:
            self._splits["train"] = self._resolve_split("train", all_windows)
            self._splits["val"] = self._resolve_split("val", all_windows)
        if stage in {None, "validate"}:
            self._splits["val"] = self._resolve_split("val", all_windows)
        if stage in {None, "test"}:
            self._splits["test"] = self._resolve_split("test", all_windows)
        if stage in {None, "predict"}:
            self._splits["predict"] = self._resolve_split("predict", all_windows)

        self._assert_non_empty_positive_labels(stage)
        self._setup = True

    def train_dataloader(self) -> Iterable[Batch]:
        return self._iter_batches("train")

    def val_dataloader(self) -> Iterable[Batch]:
        return self._iter_batches("val")

    def test_dataloader(self) -> Iterable[Batch]:
        return self._iter_batches("test")

    def predict_dataloader(self) -> Iterable[Batch]:
        return self._iter_batches("predict")

    def _iter_batches(self, split: str) -> Iterator[Batch]:
        if not self._setup:
            self.setup(_stage_for_split(split))

        paths = self._splits.get(split, ())
        if not paths:
            return

        batch_size = max(1, int(self._config.batch_size))
        bucket: list[Mapping[str, Any]] = []
        for path in paths:
            bucket.append(self._load_window(split=split, path=path))
            if len(bucket) >= batch_size:
                yield _collate_batch(bucket)
                bucket = []

        if bucket and not bool(self._config.drop_last):
            yield _collate_batch(bucket)

    def _resolve_split(self, split: str, windows: Sequence[Path]) -> tuple[Path, ...]:
        declared_ids: tuple[str, ...] | None = None
        try:
            declared_ids = self._resolver.split_ids(self._config.dataset, split)
        except Exception:
            declared_ids = None

        split_ids = self._safe_split_ids(split)
        if not split_ids:
            explicitly_empty_split = declared_ids is not None and len(declared_ids) == 0
            if (
                bool(self._config.strict_split_enforcement)
                and split in {"train", "val", "test"}
                and not explicitly_empty_split
            ):
                raise DataContractError(
                    f"Missing split IDs for dataset '{self._config.dataset}' split '{split}' under strict enforcement."
                )
            # Only predict is allowed to fall back to the full window set. Returning
            # all windows for train/val/test would silently reintroduce split leakage.
            return tuple(windows) if split == "predict" else ()

        selected = [p for p in windows if _path_matches_split_ids(p, split_ids)]
        if selected:
            return tuple(selected)

        by_meta: list[Path] = []
        for path in windows:
            try:
                _j, _m, _c, _mask, _fps, meta = read_window_npz(str(path))
            except Exception:
                continue
            if meta.video_id and meta.video_id in split_ids:
                by_meta.append(path)
        return tuple(by_meta)

    def _safe_split_ids(self, split: str) -> tuple[str, ...]:
        try:
            ids = self._resolver.split_ids(self._config.dataset, split)
            if ids:
                return ids
        except Exception:
            pass

        # This keeps local rebuilds usable when config-level split IDs are absent, but
        # still anchors the split contract to a concrete generated artifact.
        split_path = self._resolver.layout.interim / str(self._config.dataset).strip().lower() / "splits.json"
        if not split_path.is_file():
            return ()
        try:
            payload = json.loads(split_path.read_text(encoding="utf-8"))
        except Exception:
            return ()
        if not isinstance(payload, Mapping):
            return ()
        raw_ids = payload.get(str(split), [])
        if not isinstance(raw_ids, Sequence):
            return ()
        out = tuple(str(x).strip() for x in raw_ids if str(x).strip())
        return out

    def _load_window(self, *, split: str, path: Path) -> Mapping[str, Any]:
        key = make_cache_key(
            CacheKeyParts(
                dataset=self._config.dataset,
                split=split,
                window_path=str(path),
                preprocess_signature=self.preprocess_signature,
            )
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        joints, motion, conf, mask, fps, meta = read_window_npz(str(path))
        x, m = build_canonical_input(
            joints,
            motion,
            conf,
            mask,
            fps,
            self._feat_cfg,
        )

        sample: Mapping[str, Any] = {
            "x": x,
            "mask": m,
            "y": int(meta.y),
            "fps": float(fps),
            "meta": {
                "path": str(path),
                "video_id": meta.video_id,
                "w_start": int(meta.w_start),
                "w_end": int(meta.w_end),
            },
        }
        transformed = self._transforms.apply(sample)
        self._cache.set(key, transformed)
        return transformed

    def _assert_non_empty_positive_labels(self, stage: Stage | None) -> None:
        split_order = ("train", "val", "test", "predict")
        active_splits = [name for name in split_order if self._splits.get(name)]
        if not active_splits:
            return

        total_pos = 0
        for split in active_splits:
            pos, neg, unlabeled = _count_labels_from_paths(self._splits[split])
            total_pos += pos
            LOGGER.warning(
                "Label distribution dataset=%s split=%s positive=%d negative=%d unlabeled=%d total=%d",
                self._config.dataset,
                split,
                pos,
                neg,
                unlabeled,
                pos + neg + unlabeled,
            )

        require_positive = stage in {None, "fit", "validate", "test"}
        if require_positive and total_pos == 0:
            raise ValueError(
                f"Zero positive labels detected for dataset '{self._config.dataset}' "
                f"across splits {active_splits}. Check label generation/annotation parsing."
            )


def _collate_batch(samples: Sequence[Mapping[str, Any]]) -> Batch:
    xs = [np.asarray(sample["x"], dtype=np.float32) for sample in samples]
    ms = [np.asarray(sample["mask"], dtype=bool) for sample in samples]
    ys = np.asarray([int(sample.get("y", -1)) for sample in samples], dtype=np.int64)

    x_stacked = _try_stack(xs)
    m_stacked = _try_stack(ms)

    return {
        "x": x_stacked if x_stacked is not None else xs,
        "mask": m_stacked if m_stacked is not None else ms,
        "y": ys,
        "meta": [sample.get("meta", {}) for sample in samples],
    }


def _try_stack(arrays: Sequence[np.ndarray]) -> np.ndarray | None:
    if not arrays:
        return None
    first_shape = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != first_shape:
            return None
    return np.stack(arrays, axis=0)


def _count_labels_from_paths(paths: Sequence[Path]) -> tuple[int, int, int]:
    pos = 0
    neg = 0
    unlabeled = 0
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as z:
                y = int(np.asarray(z["y"]).reshape(-1)[0]) if "y" in z.files else -1
        except Exception:
            y = -1
        if y == 1:
            pos += 1
        elif y == 0:
            neg += 1
        else:
            unlabeled += 1
    return pos, neg, unlabeled


def _path_matches_split_ids(path: Path, split_ids: Sequence[str]) -> bool:
    stem = path.stem
    seq_token = _sequence_token_from_stem(stem).lower()
    for sid in split_ids:
        token = str(sid).strip().lower()
        if not token:
            continue
        if seq_token == token:
            return True
    return False


def _sequence_token_from_stem(stem: str) -> str:
    """Extract canonical sequence token from window filename stem."""

    raw = str(stem).strip()
    m = re.match(r"^(?P<seq>.+)_w\d+$", raw)
    if m is not None:
        return m.group("seq")
    m2 = re.match(r"^(?P<seq>.+)_win_\d+$", raw)
    if m2 is not None:
        return m2.group("seq")
    return raw


def _stage_for_split(split: str) -> Stage | None:
    if split == "train":
        return "fit"
    if split == "val":
        return "validate"
    if split == "test":
        return "test"
    if split == "predict":
        return "predict"
    return None


__all__ = ["DataModuleConfig", "UnifiedWindowDataModule"]
