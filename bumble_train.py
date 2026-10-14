from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
SCORE_FIELDS = ["score", "face_biased", "multimodal", "ridge", "knn"]
SELECTION_FIELDS = [
    "split",
    "selection_reason",
    "timestamp",
    "screenshot",
    "raw_path",
    "crop_path",
    "selected_path",
    "method",
    "action",
    "score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
    "component_spread",
    "score_band",
]
COMBINED_LABEL_FIELDS = ["path", "rating_1_5", "rating"]
EVALUATION_FIELDS = [
    "metric",
    "count",
    "mae",
    "rmse",
    "bias",
    "exact_1_5_error_rate",
    "off_by_one_error_rate",
    "swipe_error_rate",
]


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    raw_path: Path
    crop_path: Path
    score: float
    component_spread: float
    score_band: str
    image_hash: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, combine, and evaluate Bumble training data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Copy logs, back up current artifacts, crop, and select images")
    prepare.add_argument("--source", default=r"D:\BumbleLog", help="Source Bumble log folder")
    prepare.add_argument("--output", default=r"D:\BumbleTrain", help="Output training workspace")
    prepare.add_argument("--target-count", type=int, default=1200, help="Total selected image count")
    prepare.add_argument("--val-count", type=int, default=200, help="Locked validation image count")
    prepare.add_argument(
        "--selection-profile",
        choices=("balanced", "swipe-recall", "calibration"),
        default="balanced",
        help="Candidate selection profile",
    )
    prepare.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Existing selection manifest to exclude from the new selection",
    )
    prepare.add_argument("--threshold", type=float, default=54.0, help="Current swipe threshold")
    prepare.add_argument("--seed", type=int, default=42, help="Deterministic selection seed")
    prepare.add_argument("--crop-left", type=float, default=0.02, help="Photo crop left ratio")
    prepare.add_argument("--crop-top", type=float, default=0.06, help="Photo crop top ratio")
    prepare.add_argument("--crop-right", type=float, default=0.98, help="Photo crop right ratio")
    prepare.add_argument("--crop-bottom", type=float, default=0.68, help="Photo crop bottom ratio")
    prepare.add_argument("--dedupe-hamming", type=int, default=0, help="Average-hash distance treated as duplicate")
    prepare.add_argument("--no-mask-share-icon", action="store_true", help="Do not mask the fixed top-right share icon area")

    recrop = subparsers.add_parser("recrop-selected", help="Re-crop the current selected set without changing labels or selection")
    recrop.add_argument("--manifest", default=r"D:\BumbleTrain\manifests\selection.csv", help="Selection manifest")
    recrop.add_argument("--crop-left", type=float, default=0.02, help="Photo crop left ratio")
    recrop.add_argument("--crop-top", type=float, default=0.06, help="Photo crop top ratio")
    recrop.add_argument("--crop-right", type=float, default=0.98, help="Photo crop right ratio")
    recrop.add_argument("--crop-bottom", type=float, default=0.68, help="Photo crop bottom ratio")
    recrop.add_argument("--no-mask-share-icon", action="store_true", help="Do not mask the fixed top-right share icon area")

    combine = subparsers.add_parser("combine-labels", help="Combine base labels with Bumble train labels")
    combine.add_argument("--base", default="dataset_labels.csv", help="Existing base labels")
    combine.add_argument("--bumble-labels", default=r"D:\BumbleTrain\labels\bumble_labels.csv", help="Bumble labels from label_app.py")
    combine.add_argument("--manifest", default=r"D:\BumbleTrain\manifests\selection.csv", help="Selection manifest")
    combine.add_argument("--output", default=r"D:\BumbleTrain\labels\combined_train_labels.csv", help="Combined labels output")

    evaluate = subparsers.add_parser("evaluate", help="Compare Bumble labels against logged predictions")
    evaluate.add_argument("--labels", default=r"D:\BumbleTrain\labels\bumble_labels.csv", help="Bumble labels from label_app.py")
    evaluate.add_argument("--manifest", default=r"D:\BumbleTrain\manifests\selection.csv", help="Selection manifest")
    evaluate.add_argument("--predictions", help="Optional score.py CSV output to evaluate instead of logged manifest scores")
    evaluate.add_argument("--split", choices=("all", "train", "validation"), default="validation", help="Split to evaluate")
    evaluate.add_argument("--threshold", type=float, default=54.0, help="Prediction swipe threshold")
    evaluate.add_argument("--output", default=r"D:\BumbleTrain\reports\evaluation.csv", help="Evaluation report CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare_workspace(args)
    elif args.command == "recrop-selected":
        recrop_selected(args)
    elif args.command == "combine-labels":
        combine_labels(args)
    elif args.command == "evaluate":
        evaluate_labels(args)
    return 0


def prepare_workspace(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    if args.val_count >= args.target_count:
        raise ValueError("--val-count must be smaller than --target-count")
    if not source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    scores_path = source / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv does not exist: {scores_path}")

    output.mkdir(parents=True, exist_ok=True)
    backup_current_artifacts(output)
    raw_dir = output / "raw"
    crop_dir = output / "cropped_all"
    selected_dir = output / "selected"
    manifest_dir = output / "manifests"
    report_dir = output / "reports"
    for directory in (crop_dir, selected_dir, manifest_dir):
        reset_directory(directory)
    for directory in (raw_dir, crop_dir, selected_dir, manifest_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied = copy_bumble_log(source, raw_dir)
    crop_rows = crop_all_images(
        raw_dir,
        crop_dir,
        crop_box=(args.crop_left, args.crop_top, args.crop_right, args.crop_bottom),
        mask_share_icon=not args.no_mask_share_icon,
    )
    rows = read_csv(raw_dir / "scores.csv")
    candidates = build_candidates(
        rows,
        raw_dir=raw_dir,
        crop_dir=crop_dir,
        dedupe_hamming=args.dedupe_hamming,
        excluded_screenshots=excluded_screenshots(getattr(args, "exclude_manifest", [])),
    )
    selected = select_candidates(
        candidates,
        target_count=args.target_count,
        val_count=args.val_count,
        threshold=args.threshold,
        seed=args.seed,
        profile=getattr(args, "selection_profile", "balanced"),
    )
    write_selection_manifest(selected, selected_dir=selected_dir, manifest_path=manifest_dir / "selection.csv")
    write_rows(manifest_dir / "all_crops.csv", crop_rows, ["screenshot", "raw_path", "crop_path", "width", "height"])

    print(f"Copied {copied} log image(s) to {raw_dir}")
    print(f"Cropped {len(crop_rows)} image(s) to {crop_dir}")
    print(f"Selected {len(selected)} image(s): {args.target_count - args.val_count} train, {args.val_count} validation")
    print(f"Selection manifest: {manifest_dir / 'selection.csv'}")
    print("Label with:")
    print(
        "  python label_app.py "
        f"--source-dir {selected_dir} "
        f"--output-csv {output / 'labels' / 'bumble_labels.csv'} "
        "--port 7863"
    )
    print("Train with the CLIP venv and cache:")
    print(r"  $env:HF_HOME='D:\DatingClawHFCache'; $env:TRANSFORMERS_CACHE='D:\DatingClawHFCache'")
    print(r"  D:\DatingClawClipVenv\Scripts\python.exe build_references.py --labels D:\BumbleTrain\labels\combined_train_labels.csv --output embeddings\reference_store_bumble_combined.npz --provider cuda --det-thresh 0.25")


def recrop_selected(args: argparse.Namespace) -> None:
    rows = read_csv(args.manifest)
    crop_box = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    count = 0
    for row in rows:
        raw_path = Path(row.get("raw_path", ""))
        crop_path = Path(row.get("crop_path", ""))
        selected_path = Path(row.get("selected_path", ""))
        if not raw_path.exists() or not selected_path:
            continue
        if crop_path:
            crop_image(
                raw_path,
                crop_path,
                crop_box=crop_box,
                mask_share_icon=not args.no_mask_share_icon,
            )
        crop_image(
            raw_path,
            selected_path,
            crop_box=crop_box,
            mask_share_icon=not args.no_mask_share_icon,
        )
        count += 1
    print(f"Re-cropped {count} selected image(s) from {args.manifest}")


def backup_current_artifacts(output: Path) -> Path:
    backup_dir = output / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for directory_name in ("embeddings", "models"):
        source = Path(directory_name)
        if source.exists():
            shutil.copytree(source, backup_dir / directory_name, dirs_exist_ok=True)

    for pattern in ("dataset_labels*.csv", "labels.csv", "results/*eval*.csv", "results/label_*.csv"):
        for source in Path(".").glob(pattern):
            if source.is_file():
                destination = backup_dir / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    return backup_dir


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_bumble_log(source: Path, raw_dir: Path) -> int:
    shutil.copy2(source / "scores.csv", raw_dir / "scores.csv")
    count = 0
    for image_path in image_files(source):
        shutil.copy2(image_path, raw_dir / image_path.name)
        count += 1
    return count


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_all_images(
    raw_dir: Path,
    crop_dir: Path,
    *,
    crop_box: tuple[float, float, float, float],
    mask_share_icon: bool,
) -> list[dict[str, str]]:
    rows = []
    for image_path in image_files(raw_dir):
        output_path = crop_dir / f"{image_path.stem}.jpg"
        width, height = crop_image(image_path, output_path, crop_box=crop_box, mask_share_icon=mask_share_icon)
        rows.append(
            {
                "screenshot": image_path.name,
                "raw_path": image_path.as_posix(),
                "crop_path": output_path.as_posix(),
                "width": str(width),
                "height": str(height),
            }
        )
    return rows


def crop_image(
    image_path: Path,
    output_path: Path,
    *,
    crop_box: tuple[float, float, float, float],
    mask_share_icon: bool,
) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        left_ratio, top_ratio, right_ratio, bottom_ratio = crop_box
        width, height = image.size
        left = clamp_int(width * left_ratio, 0, width - 1)
        top = clamp_int(height * top_ratio, 0, height - 1)
        right = clamp_int(width * right_ratio, left + 1, width)
        bottom = clamp_int(height * bottom_ratio, top + 1, height)
        cropped = image.crop((left, top, right, bottom))
        if mask_share_icon:
            mask_top_right_artifact(cropped)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, format="JPEG", quality=95, optimize=True)
        return cropped.size


def mask_top_right_artifact(image: Image.Image) -> None:
    width, height = image.size
    left = int(width * 0.865)
    top = int(height * 0.02)
    right = int(width * 0.965)
    bottom = int(height * 0.115)
    if right <= left or bottom <= top:
        return
    sample_x = max(0, left - int(width * 0.02))
    sample_y = min(height - 1, bottom + int(height * 0.02))
    color = image.getpixel((sample_x, sample_y))
    ImageDraw.Draw(image).rounded_rectangle((left, top, right, bottom), radius=max(4, (right - left) // 3), fill=color)


def clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(int(round(value)), upper))


def build_candidates(
    rows: list[dict[str, str]],
    *,
    raw_dir: Path,
    crop_dir: Path,
    dedupe_hamming: int = 0,
    excluded_screenshots: set[str] | None = None,
) -> list[Candidate]:
    candidates = []
    hashes: list[int] = []
    excluded_screenshots = excluded_screenshots or set()
    for row in rows:
        screenshot = (row.get("screenshot") or "").strip()
        score = parse_float(row.get("score"))
        if not screenshot or score is None:
            continue
        if screenshot.casefold() in excluded_screenshots:
            continue
        raw_path = raw_dir / screenshot
        crop_path = crop_dir / f"{Path(screenshot).stem}.jpg"
        if not raw_path.exists() or not crop_path.exists():
            continue
        image_hash = average_hash(crop_path)
        if any(hamming_distance(image_hash, existing) <= dedupe_hamming for existing in hashes):
            continue
        hashes.append(image_hash)
        candidates.append(
            Candidate(
                row=row,
                raw_path=raw_path,
                crop_path=crop_path,
                score=score,
                component_spread=component_spread(row),
                score_band=score_band(score),
                image_hash=image_hash,
            )
        )
    return candidates


def average_hash(image_path: Path) -> int:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def select_candidates(
    candidates: list[Candidate],
    *,
    target_count: int,
    val_count: int,
    threshold: float,
    seed: int,
    profile: str = "balanced",
) -> list[tuple[str, str, Candidate]]:
    if len(candidates) < target_count:
        raise ValueError(f"Only {len(candidates)} usable unique candidate(s), cannot select {target_count}")
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    selected: list[tuple[str, str, Candidate]] = []

    val_quotas = validation_quotas(val_count)
    for band, quota in val_quotas.items():
        pool = [candidate for candidate in candidates if candidate.score_band == band and candidate.row["screenshot"] not in selected_ids]
        rng.shuffle(pool)
        for candidate in pool[:quota]:
            selected.append(("validation", f"validation_{band}", candidate))
            selected_ids.add(candidate.row["screenshot"])

    while len([item for item in selected if item[0] == "validation"]) < val_count:
        remaining = [candidate for candidate in candidates if candidate.row["screenshot"] not in selected_ids]
        rng.shuffle(remaining)
        candidate = remaining[0]
        selected.append(("validation", "validation_fill", candidate))
        selected_ids.add(candidate.row["screenshot"])

    train_count = target_count - val_count
    if profile == "swipe-recall":
        train_rules = swipe_recall_train_rules(train_count)
    elif profile == "calibration":
        train_rules = calibration_train_rules(train_count)
    else:
        train_rules = balanced_train_rules(train_count, threshold)

    train_selected = 0
    for reason, quota, predicate, sort_key in train_rules:
        needed = min(quota, train_count - train_selected)
        pool = [
            candidate
            for candidate in candidates
            if candidate.row["screenshot"] not in selected_ids and predicate(candidate)
        ]
        rng.shuffle(pool)
        pool.sort(key=sort_key)
        for candidate in pool[:needed]:
            selected.append(("train", reason, candidate))
            selected_ids.add(candidate.row["screenshot"])
            train_selected += 1
        if train_selected >= train_count:
            break

    while train_selected < train_count:
        remaining = [candidate for candidate in candidates if candidate.row["screenshot"] not in selected_ids]
        rng.shuffle(remaining)
        candidate = remaining[0]
        selected.append(("train", "train_diversity_fill", candidate))
        selected_ids.add(candidate.row["screenshot"])
        train_selected += 1

    return selected


def balanced_train_rules(
    train_count: int,
    threshold: float,
) -> list[tuple[str, int, Callable[[Candidate], bool], Callable[[Candidate], float]]]:
    return [
        (
            "train_near_threshold",
            round(train_count * 0.30),
            lambda candidate: abs(candidate.score - threshold) <= 10,
            lambda candidate: abs(candidate.score - threshold),
        ),
        (
            "train_model_disagreement",
            round(train_count * 0.30),
            lambda candidate: candidate.component_spread >= 15,
            lambda candidate: -candidate.component_spread,
        ),
        (
            "train_clear_left",
            round(train_count * 0.25),
            lambda candidate: candidate.score < 45,
            lambda candidate: candidate.score,
        ),
        (
            "train_clear_right",
            train_count,
            lambda candidate: candidate.score > 59,
            lambda candidate: -candidate.score,
        ),
    ]


def swipe_recall_train_rules(
    train_count: int,
) -> list[tuple[str, int, Callable[[Candidate], bool], Callable[[Candidate], float]]]:
    return [
        (
            "train_swipe_recall_70_plus",
            train_count,
            lambda candidate: candidate.score >= 70,
            lambda candidate: -candidate.score,
        ),
        (
            "train_swipe_recall_55_70",
            train_count,
            lambda candidate: 55 <= candidate.score < 70,
            lambda candidate: abs(candidate.score - 54),
        ),
        (
            "train_swipe_recall_45_55",
            train_count,
            lambda candidate: 45 <= candidate.score < 55,
            lambda candidate: -candidate.component_spread,
        ),
        (
            "train_swipe_recall_disagreement_fill",
            train_count,
            lambda candidate: candidate.component_spread >= 15,
            lambda candidate: -candidate.component_spread,
        ),
    ]


def calibration_train_rules(
    train_count: int,
) -> list[tuple[str, int, Callable[[Candidate], bool], Callable[[Candidate], float]]]:
    return [
        (
            "train_calibration_30_50",
            round(train_count * 0.30),
            lambda candidate: 30 <= candidate.score < 50,
            lambda candidate: abs(candidate.score - 40),
        ),
        (
            "train_calibration_50_60",
            round(train_count * 0.25),
            lambda candidate: 50 <= candidate.score < 60,
            lambda candidate: abs(candidate.score - 54),
        ),
        (
            "train_calibration_60_75",
            round(train_count * 0.30),
            lambda candidate: 60 <= candidate.score <= 75,
            lambda candidate: abs(candidate.score - 67.5),
        ),
        (
            "train_calibration_disagreement_fill",
            train_count,
            lambda candidate: 30 <= candidate.score <= 75 and candidate.component_spread >= 15,
            lambda candidate: -candidate.component_spread,
        ),
    ]


def excluded_screenshots(manifest_paths: Iterable[str]) -> set[str]:
    screenshots: set[str] = set()
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude manifest does not exist: {path}")
        for row in read_csv(path):
            screenshot = (row.get("screenshot") or Path(row.get("raw_path", "")).name).strip()
            if screenshot:
                screenshots.add(screenshot.casefold())
    return screenshots


def validation_quotas(val_count: int) -> dict[str, int]:
    bands = ["low", "mid_low", "threshold", "high"]
    base = val_count // len(bands)
    remainder = val_count % len(bands)
    return {band: base + int(index < remainder) for index, band in enumerate(bands)}


def write_selection_manifest(
    selected: list[tuple[str, str, Candidate]],
    *,
    selected_dir: Path,
    manifest_path: Path,
) -> None:
    rows = []
    for split, reason, candidate in selected:
        destination = selected_dir / split / candidate.crop_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.crop_path, destination)
        row = {
            "split": split,
            "selection_reason": reason,
            "timestamp": candidate.row.get("timestamp", ""),
            "screenshot": candidate.row.get("screenshot", ""),
            "raw_path": candidate.raw_path.resolve().as_posix(),
            "crop_path": candidate.crop_path.resolve().as_posix(),
            "selected_path": destination.resolve().as_posix(),
            "method": candidate.row.get("method", ""),
            "action": candidate.row.get("action", ""),
            "score": format_float(candidate.score),
            "face_biased": candidate.row.get("face_biased", ""),
            "multimodal": candidate.row.get("multimodal", ""),
            "ridge": candidate.row.get("ridge", ""),
            "knn": candidate.row.get("knn", ""),
            "component_spread": format_float(candidate.component_spread),
            "score_band": candidate.score_band,
        }
        rows.append(row)
    write_rows(manifest_path, rows, SELECTION_FIELDS)


def combine_labels(args: argparse.Namespace) -> None:
    manifest_rows = read_csv(args.manifest)
    train_paths = {
        normalize_path(row["selected_path"])
        for row in manifest_rows
        if row.get("split") == "train" and row.get("selected_path")
    }
    base_path = Path(args.base)
    bumble_labels_path = Path(args.bumble_labels)
    base_rows = read_csv(base_path)
    bumble_rows = read_csv(bumble_labels_path)

    output_rows = [
        normalized
        for row in base_rows
        if (normalized := normalize_label_row(row, base_dir=base_path.parent)) is not None
    ]
    labeled_train = []
    for row in bumble_rows:
        normalized = normalize_label_row(row, base_dir=bumble_labels_path.parent)
        if normalized is None:
            continue
        if normalize_path(normalized["path"]) in train_paths:
            labeled_train.append(normalized)

    output_rows.extend(labeled_train)
    write_rows(Path(args.output), output_rows, COMBINED_LABEL_FIELDS)
    print(f"Base labels: {len(output_rows) - len(labeled_train)}")
    print(f"Bumble train labels included: {len(labeled_train)}")
    print(f"Saved combined labels to {args.output}")


def evaluate_labels(args: argparse.Namespace) -> None:
    labels = {normalize_path(row["path"]): row for row in read_csv(args.labels)}
    if getattr(args, "predictions", None):
        evaluate_prediction_csv(args, labels)
        return

    manifest_rows = [
        row
        for row in read_csv(args.manifest)
        if args.split == "all" or row.get("split") == args.split
    ]

    report_rows = []
    for metric in SCORE_FIELDS:
        pairs = []
        for row in manifest_rows:
            label = labels.get(normalize_path(row.get("selected_path", "")))
            prediction = parse_float(row.get(metric))
            if label is None or prediction is None:
                continue
            rating = parse_float(label.get("rating"))
            rating_1_5 = parse_float(label.get("rating_1_5"))
            if rating is None or rating_1_5 is None:
                continue
            pairs.append((prediction, rating, int(rating_1_5)))
        report_rows.append(evaluation_row(metric, pairs, threshold=args.threshold))

    write_rows(Path(args.output), report_rows, EVALUATION_FIELDS)
    for row in report_rows:
        print(
            f"{row['metric']}: n={row['count']} MAE={row['mae']} RMSE={row['rmse']} "
            f"1-5 error={row['exact_1_5_error_rate']} swipe error={row['swipe_error_rate']}"
        )
    print(f"Saved evaluation report to {args.output}")


def evaluate_prediction_csv(args: argparse.Namespace, labels: dict[str, dict[str, str]]) -> None:
    pairs = []
    for row in read_csv(args.predictions):
        path = normalize_path(row.get("file", ""))
        label = labels.get(path)
        prediction = parse_float(row.get("rating"))
        if label is None or prediction is None:
            continue
        rating = parse_float(label.get("rating"))
        rating_1_5 = parse_float(label.get("rating_1_5"))
        if rating is None or rating_1_5 is None:
            continue
        pairs.append((prediction, rating, int(rating_1_5)))
    report_rows = [evaluation_row("prediction", pairs, threshold=args.threshold)]
    write_rows(Path(args.output), report_rows, EVALUATION_FIELDS)
    row = report_rows[0]
    print(
        f"prediction: n={row['count']} MAE={row['mae']} RMSE={row['rmse']} "
        f"1-5 error={row['exact_1_5_error_rate']} swipe error={row['swipe_error_rate']}"
    )
    print(f"Saved evaluation report to {args.output}")


def evaluation_row(metric: str, pairs: list[tuple[float, float, int]], *, threshold: float) -> dict[str, str]:
    if not pairs:
        return {
            "metric": metric,
            "count": "0",
            "mae": "",
            "rmse": "",
            "bias": "",
            "exact_1_5_error_rate": "",
            "off_by_one_error_rate": "",
            "swipe_error_rate": "",
        }
    errors = [prediction - rating for prediction, rating, _ in pairs]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    exact_errors = 0
    off_by_one_errors = 0
    swipe_errors = 0
    for prediction, rating, rating_1_5 in pairs:
        predicted_1_5 = score_to_rating_1_5(prediction)
        if predicted_1_5 != rating_1_5:
            exact_errors += 1
        if abs(predicted_1_5 - rating_1_5) > 1:
            off_by_one_errors += 1
        if (prediction >= threshold) != (rating >= 75):
            swipe_errors += 1
    count = len(pairs)
    return {
        "metric": metric,
        "count": str(count),
        "mae": format_float(sum(abs_errors) / count),
        "rmse": format_float(math.sqrt(sum(squared_errors) / count)),
        "bias": format_float(sum(errors) / count),
        "exact_1_5_error_rate": format_float(exact_errors / count),
        "off_by_one_error_rate": format_float(off_by_one_errors / count),
        "swipe_error_rate": format_float(swipe_errors / count),
    }


def score_to_rating_1_5(score: float) -> int:
    return min(5, max(1, round(score / 25) + 1))


def normalize_label_row(row: dict[str, str], *, base_dir: Path | None = None) -> dict[str, str] | None:
    path = (row.get("path") or "").strip()
    rating = parse_float(row.get("rating"))
    rating_1_5 = parse_float(row.get("rating_1_5"))
    if not path or rating is None:
        return None
    if rating_1_5 is None:
        rating_1_5 = score_to_rating_1_5(rating)
    image_path = Path(path)
    if base_dir is not None and not image_path.is_absolute():
        image_path = base_dir / image_path
    return {
        "path": normalize_path(image_path.resolve()),
        "rating_1_5": str(int(rating_1_5)),
        "rating": format_float(rating),
    }


def score_band(score: float) -> str:
    if score < 45:
        return "low"
    if score < 49:
        return "mid_low"
    if score <= 59:
        return "threshold"
    return "high"


def component_spread(row: dict[str, str]) -> float:
    values = [value for value in (parse_float(row.get(field)) for field in ("multimodal", "ridge", "knn")) if value is not None]
    if not values:
        return 0.0
    return max(values) - min(values)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str | Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_float(value: float) -> str:
    return f"{value:.6f}"


def normalize_path(value: str | Path) -> str:
    return Path(value).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
