from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import BASE_MODELS_CONFIG, app, biased_multimodal_score, comparison_text


class AppTests(unittest.TestCase):
    def test_root_reports_api_status_and_dashboard_url(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["dashboard_url"], "http://localhost:7863/sandbox")

    def test_favicon_returns_no_content(self) -> None:
        response = TestClient(app).get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_biased_multimodal_score_blends_face_and_multimodal(self) -> None:
        self.assertEqual(
            biased_multimodal_score(40.0, 80.0, face_weight=0.5),
            60.0,
        )
        self.assertIsNone(biased_multimodal_score(None, 80.0, face_weight=0.5))
        self.assertIsNone(biased_multimodal_score(40.0, None, face_weight=0.5))

    def test_comparison_text_uses_ridge_as_reference(self) -> None:
        self.assertEqual(comparison_text(70.0, 55.0), "+15.0 vs Ridge")
        self.assertEqual(comparison_text(40.0, 55.0), "-15.0 vs Ridge")
        self.assertEqual(comparison_text(None, 55.0), "no comparison")

    def test_chandra_7k_model_uses_separate_artifacts(self) -> None:
        config = BASE_MODELS_CONFIG["chandra_7k"]

        self.assertEqual(config["store"], "embeddings/reference_store_chandra_7k.npz")
        self.assertEqual(config["regressor"], "models/rating_regressor_chandra_7k.joblib")
        self.assertEqual(
            config["multimodal"],
            "models/rating_regressor_multimodal_chandra_7k.joblib",
        )
        self.assertEqual(config["face_weight"], 0.50)


if __name__ == "__main__":
    unittest.main()
