from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KeypointRequest(BaseModel):
    x: float
    y: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SkeletonFrameRequest(BaseModel):
    timestamp_ms: int = Field(ge=0)
    keypoints: dict[str, KeypointRequest] = Field(min_length=1)


class PredictionRequest(BaseModel):
    source_id: str | None = None
    frames: list[SkeletonFrameRequest] = Field(min_length=1)


class PredictionResponse(BaseModel):
    label: Literal["fall", "non_fall"]
    fall_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    reasons: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class RuntimeResponse(BaseModel):
    model_name: str
    input_type: Literal["skeleton_window"]
