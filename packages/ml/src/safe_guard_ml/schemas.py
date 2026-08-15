from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping


def _ensure_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _clamp_probability(value: float) -> float:
    _ensure_finite("probability", value)
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _ensure_finite("x", self.x)
        _ensure_finite("y", self.y)
        object.__setattr__(self, "confidence", _clamp_probability(self.confidence))


@dataclass(frozen=True)
class SkeletonFrame:
    timestamp_ms: int
    keypoints: Mapping[str, Keypoint]

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not self.keypoints:
            raise ValueError("keypoints must not be empty")


@dataclass(frozen=True)
class SkeletonWindow:
    frames: tuple[SkeletonFrame, ...]
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frames must not be empty")

        previous_timestamp = -1
        for frame in self.frames:
            if frame.timestamp_ms < previous_timestamp:
                raise ValueError("frames must be ordered by timestamp_ms")
            previous_timestamp = frame.timestamp_ms


@dataclass(frozen=True)
class Prediction:
    label: Literal["fall", "non_fall"]
    fall_probability: float
    confidence: float
    model_name: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fall_probability", _clamp_probability(self.fall_probability))
        object.__setattr__(self, "confidence", _clamp_probability(self.confidence))
        if not self.model_name:
            raise ValueError("model_name must not be empty")
