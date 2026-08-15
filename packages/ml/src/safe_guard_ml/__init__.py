"""Clean ML package for Safe Guard."""

from safe_guard_ml.baseline import HeuristicFallPredictor
from safe_guard_ml.runtime import PredictionRuntime
from safe_guard_ml.schemas import Keypoint, Prediction, SkeletonFrame, SkeletonWindow

__version__ = "0.1.0"

__all__ = [
    "HeuristicFallPredictor",
    "Keypoint",
    "Prediction",
    "PredictionRuntime",
    "SkeletonFrame",
    "SkeletonWindow",
    "__version__",
]
