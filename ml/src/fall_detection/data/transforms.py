"""Deterministic transform pipeline implementations for DataModule flows."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from ..training.contracts import TransformPipelineProtocol

TransformFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class TransformStep:
    """One named transform step with serializable config metadata.

    Complexity:
        O(1) immutable metadata container.
    """

    name: str
    fn: TransformFn
    config: Mapping[str, Any] = field(default_factory=dict)


class IdentityTransformPipeline(TransformPipelineProtocol):
    """No-op transform pipeline with stable signature."""

    @property
    def name(self) -> str:
        return "identity"

    @property
    def version(self) -> str:
        return "v1"

    def signature(self) -> str:
        return _signature_payload({"name": self.name, "version": self.version, "steps": []})

    def apply(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        return sample


class ComposeTransformPipeline(TransformPipelineProtocol):
    """Composable deterministic transform pipeline."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        steps: Sequence[TransformStep],
    ) -> None:
        self._name = str(name)
        self._version = str(version)
        self._steps = tuple(steps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def signature(self) -> str:
        payload = {
            "name": self._name,
            "version": self._version,
            "steps": [
                {
                    "name": step.name,
                    "config": _to_jsonable(step.config),
                }
                for step in self._steps
            ],
        }
        return _signature_payload(payload)

    def apply(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        out: Mapping[str, Any] = sample
        for step in self._steps:
            out = step.fn(out)
        return out


def _signature_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ComposeTransformPipeline",
    "IdentityTransformPipeline",
    "TransformStep",
]
