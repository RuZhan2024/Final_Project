"""Schema models and validators for immutable data-source configuration.

This module validates YAML/JSON payloads that describe how datasets are
located under project-level `data/` storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import DataContractError, validate_split_disjointness

DATA_SOURCES_SCHEMA_VERSION = "data_sources.v1"


@dataclass(frozen=True)
class DataPathLayout:
    """Resolved immutable data-path layout.

    Attributes:
        root: Canonical data root directory.
        raw: Directory containing raw pose artifacts.
        interim: Directory containing intermediate derived assets.
        processed: Directory containing model-ready windows/features.

    Complexity:
        O(1) data container.
    """

    root: Path
    raw: Path
    interim: Path
    processed: Path

    @classmethod
    def from_root(cls, root: Path) -> "DataPathLayout":
        """Build canonical path layout from one root.

        Args:
            root: Project data root.

        Returns:
            DataPathLayout with resolved child directories.

        Complexity:
            O(1).
        """

        resolved_root = root.expanduser().resolve()
        return cls(
            root=resolved_root,
            raw=(resolved_root / "raw"),
            interim=(resolved_root / "interim"),
            processed=(resolved_root / "processed"),
        )


@dataclass(frozen=True)
class DatasetSource:
    """One dataset source definition within data-source config.

    Attributes:
        name: Stable dataset key.
        raw_dir: Raw source directory relative to/under `data/raw`.
        processed_dir: Processed window directory relative to/under `data/processed`.
        splits: Mapping split name to sequence IDs.

    Complexity:
        O(1) immutable metadata wrapper.
    """

    name: str
    raw_dir: Path
    processed_dir: Path
    splits: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class DataSourcesConfig:
    """Validated root configuration for dataset source bindings.

    Attributes:
        schema_version: Config schema version literal.
        layout: Canonical data-path layout.
        datasets: Dataset definitions keyed by dataset name.

    Complexity:
        O(1) wrapper; validation is separate.
    """

    schema_version: str
    layout: DataPathLayout
    datasets: Mapping[str, DatasetSource]


def validate_data_sources_payload(payload: Mapping[str, Any], *, base_dir: Path) -> DataSourcesConfig:
    """Validate and normalize data-source payload.

    Args:
        payload: Parsed YAML/JSON mapping.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Validated and normalized `DataSourcesConfig`.

    Raises:
        DataContractError: If schema or dataset definitions are invalid.

    Complexity:
        O(d + n), where d is dataset count and n is total split IDs.
    """

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != DATA_SOURCES_SCHEMA_VERSION:
        raise DataContractError(
            f"Unsupported data sources schema '{schema_version}'. Expected '{DATA_SOURCES_SCHEMA_VERSION}'."
        )

    data_root_value = payload.get("data_root", "data")
    data_root = _resolve_path(base_dir=base_dir, candidate=data_root_value)
    layout = DataPathLayout.from_root(data_root)

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, Mapping) or not datasets_raw:
        raise DataContractError("`datasets` must be a non-empty mapping.")

    datasets: dict[str, DatasetSource] = {}
    for dataset_name, raw_value in datasets_raw.items():
        name = str(dataset_name).strip()
        if not name:
            raise DataContractError("Dataset name keys must be non-empty strings.")
        if not isinstance(raw_value, Mapping):
            raise DataContractError(f"Dataset '{name}' payload must be a mapping.")

        raw_dir = _resolve_path(base_dir=layout.raw, candidate=raw_value.get("raw_dir", name))
        processed_dir = _resolve_path(
            base_dir=layout.processed,
            candidate=raw_value.get("processed_dir", name),
        )

        splits_value = raw_value.get("splits", {})
        if not isinstance(splits_value, Mapping):
            raise DataContractError(f"Dataset '{name}' splits must be a mapping.")

        normalized_splits: dict[str, tuple[str, ...]] = {}
        for split_name, ids_value in splits_value.items():
            split_key = str(split_name).strip()
            if not split_key:
                raise DataContractError(f"Dataset '{name}' has empty split name.")
            if not isinstance(ids_value, (list, tuple)):
                raise DataContractError(f"Dataset '{name}' split '{split_key}' must be a list/tuple.")
            normalized_splits[split_key] = tuple(str(item) for item in ids_value)

        validate_split_disjointness(normalized_splits)

        datasets[name] = DatasetSource(
            name=name,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            splits=normalized_splits,
        )

    return DataSourcesConfig(
        schema_version=schema_version,
        layout=layout,
        datasets=datasets,
    )


def _resolve_path(*, base_dir: Path, candidate: Any) -> Path:
    """Resolve one candidate path relative to a base directory.

    Args:
        base_dir: Directory used for relative resolution.
        candidate: Raw path value.

    Returns:
        Resolved absolute `Path`.

    Complexity:
        O(k) in path component count.
    """

    path = Path(str(candidate))
    if path.is_absolute():
        return path.expanduser().resolve()
    return (base_dir / path).expanduser().resolve()
