from __future__ import annotations

from dataclasses import dataclass

from safe_guard_ml.schemas import Keypoint, Prediction, SkeletonFrame, SkeletonWindow


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class HeuristicConfig:
    confidence_threshold: float = 0.3
    fall_threshold: float = 0.65
    displacement_scale: float = 0.45
    velocity_scale: float = 0.9
    min_usable_frames: int = 2


@dataclass(frozen=True)
class FrameSummary:
    timestamp_ms: int
    center_y: float
    body_height: float
    coverage: float


class HeuristicFallPredictor:
    model_name = "heuristic-baseline-v1"

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    def predict(self, window: SkeletonWindow) -> Prediction:
        summaries = [summary for frame in window.frames if (summary := self._summarize(frame))]

        if len(summaries) < self.config.min_usable_frames:
            return Prediction(
                label="non_fall",
                fall_probability=0.0,
                confidence=0.15,
                model_name=self.model_name,
                reasons=("insufficient confident skeleton frames",),
            )

        first = summaries[0]
        last = summaries[-1]
        elapsed_s = max((last.timestamp_ms - first.timestamp_ms) / 1000.0, 0.001)

        downward_displacement = max(0.0, last.center_y - first.center_y)
        downward_velocity = downward_displacement / elapsed_s
        posture_change = max(0.0, 1.0 - (last.body_height / max(first.body_height, 0.001)))

        displacement_score = _clamp(downward_displacement / self.config.displacement_scale)
        velocity_score = _clamp(downward_velocity / self.config.velocity_scale)
        posture_score = _clamp(posture_change)

        probability = _clamp(
            0.50 * displacement_score
            + 0.35 * velocity_score
            + 0.15 * posture_score
        )
        coverage = sum(summary.coverage for summary in summaries) / len(summaries)
        confidence = _clamp((0.45 + abs(probability - self.config.fall_threshold)) * coverage)
        label = "fall" if probability >= self.config.fall_threshold else "non_fall"

        return Prediction(
            label=label,
            fall_probability=probability,
            confidence=confidence,
            model_name=self.model_name,
            reasons=self._reasons(displacement_score, velocity_score, posture_score),
        )

    def _summarize(self, frame: SkeletonFrame) -> FrameSummary | None:
        points = self._confident_points(frame)
        if not points:
            return None

        y_values = [point.y for point in points]
        coverage = len(points) / len(frame.keypoints)

        return FrameSummary(
            timestamp_ms=frame.timestamp_ms,
            center_y=sum(y_values) / len(y_values),
            body_height=max(y_values) - min(y_values),
            coverage=coverage,
        )

    def _confident_points(self, frame: SkeletonFrame) -> list[Keypoint]:
        return [
            point
            for point in frame.keypoints.values()
            if point.confidence >= self.config.confidence_threshold
        ]

    def _reasons(
        self,
        displacement_score: float,
        velocity_score: float,
        posture_score: float,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if displacement_score >= 0.5:
            reasons.append("downward displacement")
        if velocity_score >= 0.5:
            reasons.append("downward velocity")
        if posture_score >= 0.5:
            reasons.append("body height collapse")
        if not reasons:
            reasons.append("stable pose window")
        return tuple(reasons)
