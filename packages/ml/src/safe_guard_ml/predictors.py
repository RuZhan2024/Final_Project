from __future__ import annotations

from typing import Protocol

from safe_guard_ml.schemas import Prediction, SkeletonWindow


class FallPredictor(Protocol):
    model_name: str

    def predict(self, window: SkeletonWindow) -> Prediction:
        """Return a fall prediction for a skeleton window."""
