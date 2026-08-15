from __future__ import annotations

from dataclasses import dataclass

from safe_guard_ml.baseline import HeuristicFallPredictor
from safe_guard_ml.predictors import FallPredictor
from safe_guard_ml.schemas import Prediction, SkeletonWindow


@dataclass(frozen=True)
class PredictionRuntime:
    predictor: FallPredictor

    @classmethod
    def baseline(cls) -> "PredictionRuntime":
        return cls(predictor=HeuristicFallPredictor())

    def predict(self, window: SkeletonWindow) -> Prediction:
        return self.predictor.predict(window)
