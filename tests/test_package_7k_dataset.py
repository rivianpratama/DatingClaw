from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from package_7k_dataset import package_dataset


class Package7kDatasetTests(unittest.TestCase):
    def test_packages_flat_images_with_deterministic_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "same.jpg"
            second = root / "b" / "same.jpg"
            first.parent.mkdir()
            second.parent.mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(first)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(second)
            labels = root / "dataset_labels.csv"
            labels.write_text(
                "path,rating_1_5,rating\n"
                "a/same.jpg,1,0\n"
                "b/same.jpg,5,100\n",
                encoding="utf-8",
            )

            output = root / "package"
            manifest = package_dataset(labels, output)

            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(manifest["label_distribution"], {"1": 1, "2": 0, "3": 0, "4": 0, "5": 1})
            self.assertTrue((output / "images" / "bc7k_000001.jpg").exists())
            self.assertTrue((output / "images" / "bc7k_000002.jpg").exists())
            with (output / "labels.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["image_id"], "bc7k_000001")
            self.assertEqual(rows[0]["filename"], "bc7k_000001.jpg")
            self.assertEqual(rows[1]["image_id"], "bc7k_000002")

    def test_rejects_missing_source_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "dataset_labels.csv"
            labels.write_text("path,rating_1_5,rating\nmissing.jpg,3,50\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Missing source image"):
                package_dataset(labels, root / "package")

    def test_rejects_invalid_rating_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            Image.new("RGB", (8, 8)).save(image)
            labels = root / "dataset_labels.csv"
            labels.write_text("path,rating_1_5,rating\na.jpg,4,100\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid rating mapping"):
                package_dataset(labels, root / "package")

    def test_refuses_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.jpg"
            Image.new("RGB", (8, 8)).save(image)
            labels = root / "dataset_labels.csv"
            labels.write_text("path,rating_1_5,rating\na.jpg,3,50\n", encoding="utf-8")
            output = root / "package"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "Pass --overwrite"):
                package_dataset(labels, output)


if __name__ == "__main__":
    unittest.main()
