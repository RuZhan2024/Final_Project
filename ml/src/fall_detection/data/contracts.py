"""Typed contracts for immutable dataset access in the refactored pipeline.

This module defines protocol-first interfaces for reading raw pose streams and
processed model windows from the canonical `data/` storage hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class DataContractError(ValueError):
    """Raised when data contracts are violated."""


@dataclass(frozen=True)
class RawPoseSample:
    """Metadata contract for a raw pose sequence.

    Attributes:
        sequence_id: Stable sequence identifier.
        source_path: Read-only path to source raw pose artifact.
        frame_count: Number of frames in the sequence.
        fps: Nominal sequence frame rate.
        layout: Skeleton layout identifier.
        metadata: Optional extensible metadata payload.

    Complexity:
        O(1) wrapper around immutable metadata.
    """

    sequence_id: str
    source_path: Path
    frame_count: int
    fps: float
    layout: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessedWindowSample:
    """Metadata contract for one processed model window.

    Attributes:
        sequence_id: Parent sequence identifier.
        window_id: Stable window identifier.
        feature_path: Path to canonical feature artifact.
        label: Binary target label.
        start_frame: Inclusive window start frame.
        end_frame: Exclusive window end frame.
        metadata: Optional metadata payload.

    Complexity:
        O(1) wrapper around immutable metadata.
    """

    sequence_id: str
    window_id: str
    feature_path: Path
    label: int
    start_frame: int
    end_frame: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetIndexProtocol(Protocol):
    """Protocol for split/index access with disjointness guarantees.

    Complexity:
        Method complexity depends on backend/index representation.
    """

    def list_sequence_ids(self, split: str) -> tuple[str, ...]:
        """Return sequence IDs for one split."""

    def validate_disjoint(self) -> None:
        """Raise `DataContractError` if split overlap exists."""


@runtime_checkable
class RawPoseDatasetProtocol(Protocol):
    """Protocol for read-only access to raw pose datasets.

    Complexity:
        Method complexity depends on backing storage/index.
    """

    @property
    def dataset_name(self) -> str:
        """Return stable dataset key."""

    @property
    def root_dir(self) -> Path:
        """Return dataset root under immutable `data/raw` hierarchy."""

    def get_sample(self, sequence_id: str) -> RawPoseSample:
        """Resolve one raw pose sample by sequence identifier."""

    def list_samples(self, split: str) -> tuple[RawPoseSample, ...]:
        """Return split-resolved raw samples."""


@runtime_checkable
class ProcessedWindowDatasetProtocol(Protocol):
    """Protocol for read-only access to processed training/eval windows.

    Complexity:
        Method complexity depends on backing storage/index.
    """

    @property
    def dataset_name(self) -> str:
        """Return stable dataset key."""

    @property
    def root_dir(self) -> Path:
        """Return processed-window root under `data/processed`."""

    def get_window(self, window_id: str) -> ProcessedWindowSample:
        """Resolve one processed window by window identifier."""

    def list_windows(self, split: str) -> tuple[ProcessedWindowSample, ...]:
        """Return split-resolved windows."""


def assert_path_is_under(path: Path, parent: Path) -> None:
    """Validate that `path` is contained under `parent`.

    Args:
        path: Candidate path.
        parent: Expected parent directory.

    Raises:
        DataContractError: If path escapes expected parent.

    Complexity:
        O(k) in path component count.
    """

    resolved_parent = parent.resolve()
    resolved_path = path.resolve()
    if resolved_parent == resolved_path:
        return
    if resolved_parent not in resolved_path.parents:
        raise DataContractError(f"Path '{resolved_path}' escapes immutable root '{resolved_parent}'.")


def validate_split_disjointness(splits: Mapping[str, Sequence[str]]) -> None:
    """Validate that split sequence IDs are pairwise disjoint.

    Args:
        splits: Mapping of split name to sequence IDs.

    Raises:
        DataContractError: If any ID exists in more than one split.

    Complexity:
        O(n) where n is total number of IDs across splits.
    """

    owner: dict[str, str] = {}
    for split_name, sequence_ids in splits.items():
        for seq_id in sequence_ids:
            key = str(seq_id)
            prior = owner.get(key)
            if prior is not None and prior != split_name:
                raise DataContractError(
                    f"Split leakage detected for sequence '{key}': '{prior}' and '{split_name}'."
                )
            owner[key] = split_name
