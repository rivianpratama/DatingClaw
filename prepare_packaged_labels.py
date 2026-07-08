from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = ["path", "rating"]
RATING_TO_SCORE = {
    "1": "0",
    "2": "25",
    "3": "50",
    "4": "75",
    "5": "100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert packaged labels into a training label CSV.")
    parser.add_argument("--labels", required=True, help="CSV with filename and rating columns")
    parser.add_argument("--images", required=True, help="Directory containing the packaged images")
    parser.add_argument("--output", required=True, help="Output CSV with path and rating columns")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = prepare_training_labels(Path(args.labels), Path(args.images), Path(args.output))
    print(f"Saved {count} training label(s) to {args.output}")
    return 0


def prepare_training_labels(labels_path: Path, images_dir: Path, output_path: Path) -> int:
    rows = read_rows(labels_path)
    prepared = []
    for row_number, row in enumerate(rows, start=2):
        filename = (row.get("filename") or "").strip()
        rating = normalize_rating(row.get("rating", ""), row_number)
        if not filename:
            raise ValueError(f"Row {row_number}: filename is required")
        validate_rating_mapping(row, rating, row_number)

        image_path = (images_dir / filename).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Row {row_number}: image does not exist: {image_path}")
        prepared.append({"path": str(image_path), "rating": rating})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(prepared)
    return len(prepared)


def read_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "rating"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV missing required column(s): {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Labels CSV must contain at least one row")
    return rows


def normalize_rating(value: str, row_number: int) -> str:
    try:
        rating = float((value or "").strip())
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: rating must be a number") from exc
    if not 0 <= rating <= 100:
        raise ValueError(f"Row {row_number}: rating must be between 0 and 100")
    return f"{rating:g}"


def validate_rating_mapping(row: dict[str, str], rating: str, row_number: int) -> None:
    rating_1_5 = (row.get("rating_1_5") or "").strip()
    if rating_1_5 and RATING_TO_SCORE.get(rating_1_5) != rating:
        raise ValueError(f"Row {row_number}: rating does not match rating_1_5")


if __name__ == "__main__":
    raise SystemExit(main())
