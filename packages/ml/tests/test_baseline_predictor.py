import unittest

from safe_guard_ml import HeuristicFallPredictor, Keypoint, SkeletonFrame, SkeletonWindow


def frame(timestamp_ms: int, y_offset: float, height: float = 0.30) -> SkeletonFrame:
    top = y_offset
    middle = y_offset + height / 2
    bottom = y_offset + height
    return SkeletonFrame(
        timestamp_ms=timestamp_ms,
        keypoints={
            "head": Keypoint(x=0.5, y=top, confidence=0.95),
            "hip": Keypoint(x=0.5, y=middle, confidence=0.95),
            "ankle": Keypoint(x=0.5, y=bottom, confidence=0.95),
        },
    )


class HeuristicFallPredictorTest(unittest.TestCase):
    def test_stable_window_is_not_a_fall(self) -> None:
        window = SkeletonWindow(
            frames=(
                frame(timestamp_ms=0, y_offset=0.20),
                frame(timestamp_ms=500, y_offset=0.20),
                frame(timestamp_ms=1000, y_offset=0.20),
            )
        )

        prediction = HeuristicFallPredictor().predict(window)

        self.assertEqual(prediction.label, "non_fall")
        self.assertLess(prediction.fall_probability, 0.10)
        self.assertIn("stable pose window", prediction.reasons)

    def test_fast_downward_motion_is_a_fall(self) -> None:
        window = SkeletonWindow(
            frames=(
                frame(timestamp_ms=0, y_offset=0.10, height=0.30),
                frame(timestamp_ms=500, y_offset=0.45, height=0.20),
                frame(timestamp_ms=1000, y_offset=0.72, height=0.10),
            )
        )

        prediction = HeuristicFallPredictor().predict(window)

        self.assertEqual(prediction.label, "fall")
        self.assertGreaterEqual(prediction.fall_probability, 0.65)
        self.assertIn("downward displacement", prediction.reasons)
        self.assertIn("downward velocity", prediction.reasons)

    def test_low_confidence_frames_are_ignored(self) -> None:
        window = SkeletonWindow(
            frames=(
                SkeletonFrame(
                    timestamp_ms=0,
                    keypoints={"head": Keypoint(x=0.5, y=0.1, confidence=0.1)},
                ),
                SkeletonFrame(
                    timestamp_ms=1000,
                    keypoints={"head": Keypoint(x=0.5, y=0.8, confidence=0.1)},
                ),
            )
        )

        prediction = HeuristicFallPredictor().predict(window)

        self.assertEqual(prediction.label, "non_fall")
        self.assertEqual(prediction.fall_probability, 0.0)
        self.assertIn("insufficient confident skeleton frames", prediction.reasons)


if __name__ == "__main__":
    unittest.main()
