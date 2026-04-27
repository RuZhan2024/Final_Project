"""Window-level cache utilities for unified datamodule loading.

The cache layer is optional and designed to keep DataModule logic free from
backend-specific persistence details.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class WindowCacheProtocol(Protocol):
    """Protocol for cache backends storing preprocessed window samples.

    Complexity:
        O(1) average-case for lookup/set by key in concrete implementations.
    """

    def get(self, key: str) -> Mapping[str, Any] | None:
        """Return cached sample payload for one key, or None when absent."""

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        """Store one sample payload by key."""


@dataclass(frozen=True)
class CacheKeyParts:
    """Components used to derive stable cache keys.

    Complexity:
        O(1) immutable metadata container.
    """

    dataset: str
    split: str
    window_path: str
    preprocess_signature: str


class InMemoryWindowCache(WindowCacheProtocol):
    """In-memory dictionary cache for preprocessed windows."""

    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Any]] = {}

    def get(self, key: str) -> Mapping[str, Any] | None:
        return self._store.get(key)

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        self._store[str(key)] = value


class DiskWindowCache(WindowCacheProtocol):
    """Filesystem-backed cache using pickle payloads.

    Notes:
        This backend prioritizes deterministic key-to-path mapping over
        serialization portability.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Mapping[str, Any] | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            payload = pickle.loads(path.read_bytes())
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
        return None

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(dict(value), protocol=pickle.HIGHEST_PROTOCOL))

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.pkl"


def make_cache_key(parts: CacheKeyParts) -> str:
    """Build stable cache key for one window sample.

    Complexity:
        O(n) in joined string length.
    """

    raw = "|".join(
        [
            str(parts.dataset).strip().lower(),
            str(parts.split).strip().lower(),
            str(parts.window_path),
            str(parts.preprocess_signature),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "CacheKeyParts",
    "DiskWindowCache",
    "InMemoryWindowCache",
    "WindowCacheProtocol",
    "make_cache_key",
]
