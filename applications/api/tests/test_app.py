import unittest

from fastapi.testclient import TestClient
from safe_guard_api import create_app


def frame(timestamp_ms: int, y_offset: float, height: float = 0.3) -> dict:
    return {
        "timestamp_ms": timestamp_ms,
        "keypoints": {
            "head": {"x": 0.5, "y": y_offset, "confidence": 0.95},
            "hip": {"x": 0.5, "y": y_offset + height / 2, "confidence": 0.95},
            "ankle": {"x": 0.5, "y": y_offset + height, "confidence": 0.95},
        },
    }


class ApiAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_runtime_endpoint_exposes_model_name(self) -> None:
        response = self.client.get("/api/v1/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "heuristic-baseline-v1")

    def test_prediction_endpoint_returns_fall(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            json={
                "source_id": "unit-test",
                "frames": [
                    frame(timestamp_ms=0, y_offset=0.10, height=0.30),
                    frame(timestamp_ms=500, y_offset=0.45, height=0.20),
                    frame(timestamp_ms=1000, y_offset=0.72, height=0.10),
                ],
            },
        )

        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["label"], "fall")
        self.assertGreaterEqual(payload["fall_probability"], 0.65)

    def test_prediction_endpoint_validates_empty_frames(self) -> None:
        response = self.client.post(
            "/api/v1/predictions",
            json={"source_id": "unit-test", "frames": []},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
