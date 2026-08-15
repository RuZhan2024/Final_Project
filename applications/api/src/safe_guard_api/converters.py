from __future__ import annotations

from safe_guard_api.schemas import PredictionRequest, PredictionResponse
from safe_guard_ml import Keypoint, Prediction, SkeletonFrame, SkeletonWindow


def request_to_window(request: PredictionRequest) -> SkeletonWindow:
    return SkeletonWindow(
        source_id=request.source_id,
        frames=tuple(
            SkeletonFrame(
                timestamp_ms=frame.timestamp_ms,
                keypoints={
                    name: Keypoint(
                        x=keypoint.x,
                        y=keypoint.y,
                        confidence=keypoint.confidence,
                    )
                    for name, keypoint in frame.keypoints.items()
                },
            )
            for frame in request.frames
        ),
    )


def prediction_to_response(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        label=prediction.label,
        fall_probability=prediction.fall_probability,
        confidence=prediction.confidence,
        model_name=prediction.model_name,
        reasons=list(prediction.reasons),
    )
