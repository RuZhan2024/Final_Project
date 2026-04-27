"""Lightweight protocol contracts for refactored data modules."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Protocol, TypedDict


Stage = Literal["fit", "validate", "test", "predict"]


class Batch(TypedDict):
    x: Any
    mask: Any
    y: Any
    meta: list[Mapping[str, Any]]


class TransformPipelineProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def signature(self) -> str:
        ...

    def apply(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class DataModuleProtocol(Protocol):
    @property
    def dataset_name(self) -> str:
        ...

    def setup(self, stage: Stage | None = None) -> None:
        ...

    def train_dataloader(self) -> Iterable[Batch]:
        ...

    def val_dataloader(self) -> Iterable[Batch]:
        ...

    def test_dataloader(self) -> Iterable[Batch]:
        ...

    def predict_dataloader(self) -> Iterable[Batch]:
        ...

