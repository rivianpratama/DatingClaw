from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_LABELS = Path("dataset_labels.csv")
DEFAULT_OUTPUT = Path(r"D:\BumbleLabel")
LABEL_FIELDS = ["image_id", "filename", "original_path", "rating_1_5", "rating"]
RATING_TO_SCORE = {
    "1": "0",
    "2": "25",
    "3": "50",
    "4": "75",
    "5": "100",
}


@dataclass(frozen=True)
class PackagedRow:
    image_id: str
    filename: str
    original_path: str
    rating_1_5: str
    rating: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the 7k labeled dataset into one flat image directory.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Input dataset_labels.csv path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output package directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output package")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dataset(Path(args.labels), Path(args.output), overwrite=args.overwrite)
    return 0


def package_dataset(labels_path: Path, output_dir: Path, *, overwrite: bool = False) -> dict[str, object]:
    labels_path = labels_path.resolve()
    output_dir = output_dir.resolve()
    rows = read_label_rows(labels_path)
    validate_rows(rows, labels_path.parent)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    packaged_rows = copy_images(rows, labels_path.parent, images_dir)
    write_labels(output_dir / "labels.csv", packaged_rows)
    manifest = write_manifest(output_dir / "manifest.json", labels_path, packaged_rows)

    print(f"Packaged {manifest['row_count']} image(s)")
    print(f"Images: {images_dir}")
    print(f"Labels: {output_dir / 'labels.csv'}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return manifest


def read_label_rows(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "rating_1_5", "rating"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV missing required column(s): {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def validate_rows(rows: Iterable[dict[str, str]], base_dir: Path) -> None:
    missing_paths = []
    invalid_rows = []
    for index, row in enumerate(rows, start=2):
        source = resolve_source_path(row.get("path", ""), base_dir)
        rating_1_5 = (row.get("rating_1_5") or "").strip()
        rating = normalize_rating(row.get("rating", ""))
        expected = RATING_TO_SCORE.get(rating_1_5)
        if expected is None or rating != expected:
            invalid_rows.append(index)
        if not source.exists():
            missing_paths.append(row.get("path", ""))

    if invalid_rows:
        preview = ", ".join(str(row) for row in invalid_rows[:10])
        raise ValueError(f"Invalid rating mapping in row(s): {preview}")
    if missing_paths:
        preview = ", ".join(missing_paths[:10])
        raise FileNotFoundError(f"Missing source image(s): {preview}")


def copy_images(rows: list[dict[str, str]], base_dir: Path, images_dir: Path) -> list[PackagedRow]:
    packaged = []
    seen_filenames: set[str] = set()
    for index, row in enumerate(rows, start=1):
        source = resolve_source_path(row["path"], base_dir)
        image_id = f"bc7k_{index:06d}"
        filename = f"{image_id}{source.suffix.lower()}"
        if filename in seen_filenames:
            raise ValueError(f"Generated duplicate output filename: {filename}")
        seen_filenames.add(filename)
        shutil.copy2(source, images_dir / filename)
        packaged.append(
            PackagedRow(
                image_id=image_id,
                filename=filename,
                original_path=Path(row["path"]).as_posix(),
                rating_1_5=(row.get("rating_1_5") or "").strip(),
                rating=normalize_rating(row.get("rating", "")),
            )
        )
    return packaged


def write_labels(path: Path, rows: list[PackagedRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "filename": row.filename,
                    "original_path": row.original_path,
                    "rating_1_5": row.rating_1_5,
                    "rating": row.rating,
                }
            )


def write_manifest(path: Path, source_csv: Path, rows: list[PackagedRow]) -> dict[str, object]:
    distribution = Counter(row.rating_1_5 for row in rows)
    manifest: dict[str, object] = {
        "dataset": "bumbleclaw_7k_public_label_seed",
        "row_count": len(rows),
        "image_count": len(rows),
        "task_count": len(rows),
        "label_distribution": {rating: distribution.get(rating, 0) for rating in sorted(RATING_TO_SCORE)},
        "rating_to_score": {rating: int(score) for rating, score in RATING_TO_SCORE.items()},
        "source_csv": source_csv.name,
        "source_csv_sha256": sha256_file(source_csv),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_prefix": "images/",
        "label_fields": LABEL_FIELDS,
        "tasks": [{"image_id": row.image_id, "filename": row.filename} for row in rows],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def resolve_source_path(value: str, base_dir: Path) -> Path:
    source = Path((value or "").strip())
    if not source.is_absolute():
        source = base_dir / source
    return source


def normalize_rating(value: str) -> str:
    return str(int(float((value or "").strip())))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
