from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

import swipe_app


class SwipeAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.cropped = self.root / "cropped"
        self.labels = self.root / "labels.csv"
        self.scores = self.root / "scores.csv"

        self.source.mkdir()
        self.cropped.mkdir()

        # Create 5 dummy test images
        self.filenames = []
        for i in range(5):
            fn = f"test_{i:02d}.webp"
            (self.source / fn).write_bytes(b"dummy_image_data")
            (self.cropped / fn).write_bytes(b"dummy_cropped_data")
            self.filenames.append(fn)

        # Write dummy scores.csv
        with open(self.scores, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["screenshot", "score"])
            writer.writeheader()
            for fn in self.filenames:
                writer.writerow({"screenshot": fn, "score": "50.0"})

        # Override module paths
        swipe_app.SOURCE_DIR = self.source
        swipe_app.CROPPED_DIR = self.cropped
        swipe_app.LABELS_CSV = self.labels
        swipe_app.SCORES_CSV = self.scores

        swipe_app.load_dataset()
        self.client = TestClient(swipe_app.app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_initial_state(self) -> None:
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["labeled"], 0)
        self.assertEqual(data["remaining"], 5)
        self.assertEqual(data["can_undo"], False)

    def test_get_cards(self) -> None:
        res = self.client.get("/api/cards?count=3")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["cards"]), 3)
        self.assertEqual(data["cards"][0]["filename"], "test_00.webp")
        self.assertEqual(data["cards"][0]["image_url"], "/images/cropped/test_00.webp")

    def test_swipe_right_and_left(self) -> None:
        # Swipe right on first card
        res = self.client.post("/api/swipe", json={"filename": "test_00.webp", "action": "right"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["labeled"], 1)
        self.assertEqual(data["likes"], 1)
        self.assertEqual(data["nopes"], 0)
        self.assertEqual(data["can_undo"], True)

        # Swipe left on second card
        res = self.client.post("/api/swipe", json={"filename": "test_01.webp", "action": "left"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["labeled"], 2)
        self.assertEqual(data["likes"], 1)
        self.assertEqual(data["nopes"], 1)

        # Verify CSV contents
        rows = list(csv.DictReader(self.labels.open(encoding="utf-8")))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "test_00.webp")
        self.assertEqual(rows[0]["action"], "right")
        self.assertEqual(rows[0]["rating"], "100")
        self.assertEqual(rows[1]["filename"], "test_01.webp")
        self.assertEqual(rows[1]["action"], "left")
        self.assertEqual(rows[1]["rating"], "0")

    def test_undo_swipe(self) -> None:
        # Swipe one card
        self.client.post("/api/swipe", json={"filename": "test_00.webp", "action": "right"})
        state = self.client.get("/api/state").json()
        self.assertEqual(state["labeled"], 1)

        # Undo
        res = self.client.post("/api/undo")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["restored_card"]["filename"], "test_00.webp")
        self.assertEqual(data["labeled"], 0)
        self.assertEqual(data["likes"], 0)

        # Next card should now be test_00.webp again
        cards_res = self.client.get("/api/cards?count=1").json()
        self.assertEqual(cards_res["cards"][0]["filename"], "test_00.webp")

    def test_serve_index_html(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("BumbleClaw Swipe", res.text)
        self.assertIn("deck-container", res.text)
        self.assertIn("undo-btn", res.text)


if __name__ == "__main__":
    unittest.main()
