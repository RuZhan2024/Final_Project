"""Path resolution and immutability guards for dataset storage.

This module centralizes canonical `data/` path resolution and provides explicit
write guards so raw datasets remain immutable across training/evaluation flows.
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from .contracts import DataContractError, assert_path_is_under
from .schema import DataPathLayout, DataSourcesConfig, validate_data_sources_payload

DEFAULT_DATA_SOURCES_CONFIG = Path("configs/experiments/data_sources.yaml")
DATA_ROOT_ENV = "FD_DATA_ROOT"


class DataResolutionError(ValueError):
    """Raised when data-source config resolution fails."""


class DataPathResolver:
    """Resolver for canonical data paths and dataset source bindings.

    Args:
        config: Validated data-sources configuration.

    Complexity:
        O(1) for accessors; O(n) for split/index traversals.
    """

    def __init__(self, config: DataSourcesConfig) -> None:
        self._config = config

    @property
    def config(self) -> DataSourcesConfig:
        """Return immutable data-sources configuration.

        Complexity:
            O(1).
        """

        return self._config

    @property
    def layout(self) -> DataPathLayout:
        """Return canonical root/raw/interim/processed layout.

        Complexity:
            O(1).
        """

        return self._config.layout

    def dataset(self, name: str) -> Mapping[str, Any]:
        """Return one dataset entry as a typed mapping view.

        Args:
            name: Dataset key.

        Raises:
            DataResolutionError: If dataset key does not exist.

        Complexity:
            O(1).
        """

        key = str(name).strip()
        dataset = self._config.datasets.get(key)
        if dataset is None:
            raise DataResolutionError(f"Unknown dataset '{key}'.")
        return {
            "name": dataset.name,
            "raw_dir": dataset.raw_dir,
            "processed_dir": dataset.processed_dir,
            "splits": dataset.splits,
        }

    def split_ids(self, dataset_name: str, split: str) -> tuple[str, ...]:
        """Return sequence IDs for one dataset split.

        Args:
            dataset_name: Dataset key.
            split: Split name (`train`, `val`, `test`, etc.).

        Raises:
            DataResolutionError: If dataset/split is unknown.

        Complexity:
            O(1).
        """

        ds = self._config.datasets.get(str(dataset_name).strip())
        if ds is None:
            raise DataResolutionError(f"Unknown dataset '{dataset_name}'.")
        split_key = str(split).strip()
        ids = ds.splits.get(split_key)
        if ids is None:
            raise DataResolutionError(f"Unknown split '{split_key}' for dataset '{dataset_name}'.")
        return ids

    def dataset_manifest_hash(self, dataset_name: str | None = None) -> str:
        """Return deterministic SHA256 hash for dataset manifest bindings.

        Args:
            dataset_name: Optional dataset key. When omitted, hashes all datasets.

        Returns:
            Hex SHA256 digest of normalized manifest payload.

        Raises:
            DataResolutionError: If selected dataset key is unknown.

        Complexity:
            O(d + n), where d is dataset count and n is total split IDs.
        """

        if dataset_name is None:
            dataset_keys = sorted(self._config.datasets.keys())
        else:
            key = str(dataset_name).strip()
            if key not in self._config.datasets:
                raise DataResolutionError(f"Unknown dataset '{key}'.")
            dataset_keys = [key]

        datasets_payload: dict[str, dict[str, Any]] = {}
        for key in dataset_keys:
            ds = self._config.datasets[key]
            datasets_payload[key] = {
                "raw_dir": str(ds.raw_dir),
                "processed_dir": str(ds.processed_dir),
                "splits": {split: list(ids) for split, ids in sorted(ds.splits.items())},
            }

        payload = {
            "schema_version": self._config.schema_version,
            "layout": {
                "root": str(self.layout.root),
                "raw": str(self.layout.raw),
                "interim": str(self.layout.interim),
                "processed": str(self.layout.processed),
            },
            "datasets": datasets_payload,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return _sha256_bytes(raw)

    def processed_dir(self, dataset_name: str) -> Path:
        """Return processed directory for one dataset key.

        Args:
            dataset_name: Dataset key.

        Returns:
            Canonical processed directory path.

        Raises:
            DataResolutionError: If dataset key is unknown.

        Complexity:
            O(1).
        """

        ds = self._config.datasets.get(str(dataset_name).strip())
        if ds is None:
            raise DataResolutionError(f"Unknown dataset '{dataset_name}'.")
        return ds.processed_dir

    def raw_dir(self, dataset_name: str) -> Path:
        """Return raw directory for one dataset key.

        Complexity:
            O(1).
        """

        ds = self._config.datasets.get(str(dataset_name).strip())
        if ds is None:
            raise DataResolutionError(f"Unknown dataset '{dataset_name}'.")
        return ds.raw_dir

    def list_processed_window_files(
        self,
        dataset_name: str,
        *,
        suffixes: tuple[str, ...] = (".npz",),
        require_window_like_names: bool = True,
    ) -> tuple[Path, ...]:
        """Return deterministic list of processed window files for one dataset.

        Args:
            dataset_name: Dataset key.
            suffixes: Allowed file suffixes.
            require_window_like_names: When true, include only likely window artifacts.

        Returns:
            Sorted tuple of window artifact paths.

        Complexity:
            O(n log n) in number of discovered files.
        """

        base = self.processed_dir(dataset_name)
        if not base.exists():
            return ()
        allowed = {s.lower() for s in suffixes}
        paths: list[Path] = []
        window_like_paths: list[Path] = []
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            paths.append(p)
            if not require_window_like_names:
                continue
            stem = p.stem.lower()
            parent = p.parent.name.lower()
            if "window" in stem or "_w" in stem or parent.startswith("windows"):
                window_like_paths.append(p)
        if require_window_like_names and window_like_paths:
            return tuple(sorted(window_like_paths, key=lambda p: str(p)))
        return tuple(sorted(paths, key=lambda p: str(p)))

    def resolve_in_raw(self, relative_or_absolute: str | Path) -> Path:
        """Resolve a path under immutable raw root.

        Args:
            relative_or_absolute: Relative path under raw root or absolute path.

        Returns:
            Canonical absolute path.

        Raises:
            DataContractError: If resolved path escapes raw root.

        Complexity:
            O(k) in path component count.
        """

        path = _resolve_under(relative_or_absolute, base=self.layout.raw)
        assert_path_is_under(path, self.layout.raw)
        return path

    def resolve_in_interim(self, relative_or_absolute: str | Path) -> Path:
        """Resolve a path under interim root.

        Complexity:
            O(k) in path component count.
        """

        path = _resolve_under(relative_or_absolute, base=self.layout.interim)
        assert_path_is_under(path, self.layout.interim)
        return path

    def resolve_in_processed(self, relative_or_absolute: str | Path) -> Path:
        """Resolve a path under processed root.

        Complexity:
            O(k) in path component count.
        """

        path = _resolve_under(relative_or_absolute, base=self.layout.processed)
        assert_path_is_under(path, self.layout.processed)
        return path

    def guard_writable_target(self, path: str | Path) -> Path:
        """Guard that a write target does not point under immutable raw root.

        Args:
            path: Candidate write target path.

        Returns:
            Canonical absolute path if writable by contract.

        Raises:
            DataContractError: If path points under `data/raw`.

        Complexity:
            O(k) in path component count.
        """

        candidate = Path(path).expanduser().resolve()
        if candidate == self.layout.raw or self.layout.raw in candidate.parents:
            raise DataContractError(
                f"Write target '{candidate}' is under immutable raw root '{self.layout.raw}'."
            )
        return candidate


def load_data_sources_config(
    source: str | Path = DEFAULT_DATA_SOURCES_CONFIG,
    *,
    env: Mapping[str, str] | None = None,
) -> DataSourcesConfig:
    """Load and validate the data-sources config file.

    Resolution precedence for data root:
      1) `FD_DATA_ROOT` environment override
      2) `data_root` from config file

    Args:
        source: YAML/JSON config path.
        env: Optional environment mapping override for deterministic tests.

    Returns:
        Validated `DataSourcesConfig`.

    Raises:
        DataResolutionError: If file cannot be decoded or schema is invalid.

    Complexity:
        O(n) in config size.
    """

    path = Path(source).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise DataResolutionError(f"Data sources config not found: {path}")

    payload = _decode_mapping(path)
    if not isinstance(payload, Mapping):
        raise DataResolutionError(f"Data sources payload must be a mapping: {path}")

    mutable_payload = dict(payload)
    active_env = dict(os.environ if env is None else env)
    root_override = str(active_env.get(DATA_ROOT_ENV, "")).strip()
    if root_override:
        mutable_payload["data_root"] = root_override

    try:
        return validate_data_sources_payload(mutable_payload, base_dir=path.parent)
    except DataContractError as exc:
        raise DataResolutionError(f"Invalid data sources config: {path}") from exc


def build_data_path_resolver(
    source: str | Path = DEFAULT_DATA_SOURCES_CONFIG,
    *,
    env: Mapping[str, str] | None = None,
) -> DataPathResolver:
    """Build a ready-to-use data path resolver.

    Args:
        source: Data sources YAML/JSON config path.
        env: Optional env mapping for deterministic override handling.

    Returns:
        Initialized `DataPathResolver`.

    Complexity:
        O(n) in config size.
    """

    return DataPathResolver(load_data_sources_config(source=source, env=env))


def _decode_mapping(path: Path) -> Mapping[str, Any]:
    """Decode YAML/JSON mapping from disk.

    Args:
        path: Config file path.

    Returns:
        Parsed mapping payload.

    Raises:
        DataResolutionError: On unsupported extension or decode failure.

    Complexity:
        O(n) in file size.
    """

    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise DataResolutionError(f"Unsupported data sources config extension: {path.suffix}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataResolutionError(f"Failed to read data sources config: {path}") from exc

    try:
        if suffix == ".json":
            decoded = json.loads(text)
        else:
            decoded = yaml.safe_load(text)
    except Exception as exc:
        raise DataResolutionError(f"Failed to decode data sources config: {path}") from exc

    if not isinstance(decoded, Mapping):
        raise DataResolutionError(f"Data sources config must decode to mapping: {path}")
    return decoded


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_under(candidate: str | Path, *, base: Path) -> Path:
    """Resolve candidate path under base when relative.

    Args:
        candidate: Relative or absolute candidate path.
        base: Base directory for relative candidates.

    Returns:
        Canonical absolute path.

    Complexity:
        O(k) in path component count.
    """

    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()
