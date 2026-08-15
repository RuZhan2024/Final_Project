import unittest

from safe_guard_ml import Keypoint, Prediction, SkeletonFrame, SkeletonWindow


class SchemaValidationTest(unittest.TestCase):
    def test_empty_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frames must not be empty"):
            SkeletonWindow(frames=())

    def test_frames_must_be_time_ordered(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordered by timestamp_ms"):
            SkeletonWindow(
                frames=(
                    SkeletonFrame(2, {"head": Keypoint(0.5, 0.2)}),
                    SkeletonFrame(1, {"head": Keypoint(0.5, 0.3)}),
                )
            )

    def test_prediction_probabilities_are_clamped(self) -> None:
        prediction = Prediction(
            label="fall",
            fall_probability=2.0,
            confidence=-1.0,
            model_name="test-model",
        )

        self.assertEqual(prediction.fall_probability, 1.0)
        self.assertEqual(prediction.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
