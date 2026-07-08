from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prepare_packaged_labels import prepare_training_labels


class PreparePackagedLabelsTests(unittest.TestCase):
    def test_writes_absolute_training_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            image = images / "bc7k_000001.jpg"
            image.write_bytes(b"image")
            labels = root / "labels.csv"
            labels.write_text(
                "filename,rating_1_5,rating\nbc7k_000001.jpg,4,75\n",
                encoding="utf-8",
            )
            output = root / "training.csv"

            count = prepare_training_labels(labels, images, output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(count, 1)
            self.assertEqual(rows, [{"path": str(image.resolve()), "rating": "75"}])

    def test_rejects_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.csv"
            labels.write_text("filename,rating\nmissing.jpg,50\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "image does not exist"):
                prepare_training_labels(labels, root / "images", root / "training.csv")

    def test_rejects_inconsistent_rating_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"image")
            labels = root / "labels.csv"
            labels.write_text("filename,rating_1_5,rating\na.jpg,5,75\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_training_labels(labels, images, root / "training.csv")


if __name__ == "__main__":
    unittest.main()
