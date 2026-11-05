from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_SOURCE = Path(r"D:\BumbleLog")
DEFAULT_OUTPUT = Path(r"D:\BumbleLog\cropped")
DEFAULT_SCORES = Path(r"D:\BumbleLog\scores.csv")
# Preserves full photo, bio and lower details while cleanly removing Android top/bottom bars and white side gutters
DEFAULT_CROP_BOX = (0.022, 0.078, 0.978, 0.920)
MATCH_TEMPLATE_PATH = Path("assets") / "match_template.png"

# Module-level template cache for workers
_MATCH_TEMPLATE: np.ndarray | None = None


def get_match_template() -> np.ndarray | None:
    global _MATCH_TEMPLATE
    if _MATCH_TEMPLATE is None and MATCH_TEMPLATE_PATH.exists():
        _MATCH_TEMPLATE = cv2.imread(str(MATCH_TEMPLATE_PATH))
    return _MATCH_TEMPLATE


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(round(value)), maximum))


def is_what_a_match(image_path: Path, template: np.ndarray | None) -> bool:
    """Fast detection of Bumble 'What a match!' popup screens."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        h, w = img.shape[:2]
        # Match screens have an almost entirely white top header (>80% white in the top 20% of the image)
        top_band = img[:int(h * 0.20), :]
        white_ratio = float(np.mean((top_band > 235).all(axis=2)))
        if white_ratio < 0.80:
            return False

        # If top band is white, verify with template match on the 'What a match!' text banner
        if template is not None:
            th, tw = template.shape[:2]
            if h >= th and w >= tw:
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                if float(res.max()) > 0.65:
                    return True
        else:
            # Fallback heuristic if template missing: top 25% is >90% white and middle has white > 70%
            bottom_band = img[int(h * 0.70):int(h * 0.90), :]
            bottom_white = float(np.mean((bottom_band > 235).all(axis=2)))
            if white_ratio > 0.90 and bottom_white > 0.65:
                return True
    except Exception:
        pass
    return False


def crop_single_image(
    source_path_str: str,
    output_path_str: str,
    crop_box: tuple[float, float, float, float],
    overwrite: bool,
    filter_matches: bool,
) -> tuple[bool, str]:
    output_path = Path(output_path_str)
    if not overwrite and output_path.exists() and output_path.stat().st_size > 0:
        return True, "skipped"

    source_path = Path(source_path_str)
    if not source_path.exists():
        return False, f"Missing: {source_path.name}"

    # Filter out 'What a match!' screens
    if filter_matches:
        template = get_match_template()
        if is_what_a_match(source_path, template):
            return False, "match_screen"

    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            left_r, top_r, right_r, bottom_r = crop_box
            w, h = image.size
            left = clamp_int(w * left_r, 0, w - 1)
            top = clamp_int(h * top_r, 0, h - 1)
            right = clamp_int(w * right_r, left + 1, w)
            bottom = clamp_int(h * bottom_r, top + 1, h)

            cropped = image.crop((left, top, right, bottom))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = "WEBP" if output_path.suffix.lower() == ".webp" else "JPEG"
            cropped.save(output_path, format=fmt, quality=90, optimize=True)
            return True, "cropped"
    except Exception as exc:
        return False, f"Error processing {source_path.name}: {exc}"


def _worker_wrapper(item: tuple[str, str, tuple[float, float, float, float], bool, bool]) -> tuple[bool, str]:
    return crop_single_image(*item)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-core batch pre-cropper for BumbleLog screenshots.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source directory containing screenshots")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination directory for cropped images")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES, help="Path to scores.csv")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Re-crop and overwrite existing files")
    parser.add_argument("--no-filter-matches", action="store_true", help="Do not filter out 'What a match' screens")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing (0 = all)")
    return parser.parse_args(argv)


def discover_tasks(source_dir: Path, output_dir: Path, scores_csv: Path | None) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    seen = set()

    if scores_csv and scores_csv.exists():
        with open(scores_csv, mode="r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                screenshot = (row.get("screenshot") or "").strip()
                if not screenshot or screenshot in seen:
                    continue
                seen.add(screenshot)
                src = source_dir / screenshot
                dst = output_dir / screenshot
                tasks.append((str(src), str(dst)))

    # Also check any image files directly in source folder that might not be in CSV
    valid_exts = {".webp", ".jpg", ".jpeg", ".png"}
    for img_p in source_dir.iterdir():
        if img_p.is_file() and img_p.suffix.lower() in valid_exts and img_p.name not in seen:
            seen.add(img_p.name)
            tasks.append((str(img_p), str(output_dir / img_p.name)))

    return tasks


def run_batch_crop(
    tasks: list[tuple[str, str]],
    crop_box: tuple[float, float, float, float] = DEFAULT_CROP_BOX,
    overwrite: bool = False,
    filter_matches: bool = True,
    workers: int = 8,
) -> dict[str, int]:
    total = len(tasks)
    print(f"Preparing to crop {total} images using {workers} processes (crop_box={crop_box})...")

    items = [
        (src, dst, crop_box, overwrite, filter_matches)
        for src, dst in tasks
    ]

    stats = {"cropped": 0, "skipped": 0, "match_screens": 0, "failed": 0}
    errors = []

    chunksize = max(50, total // (workers * 20))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(_worker_wrapper, items, chunksize=chunksize)
        with tqdm(total=total, desc="Cropping cards", unit="img") as pbar:
            for ok, msg in futures:
                if ok:
                    if msg == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["cropped"] += 1
                else:
                    if msg == "match_screen":
                        stats["match_screens"] += 1
                    else:
                        stats["failed"] += 1
                        errors.append(msg)
                pbar.update(1)

    print("\nBatch crop complete:")
    print(f"  Cropped: {stats['cropped']}")
    print(f"  Skipped (already cached): {stats['skipped']}")
    print(f"  Removed Match Screens: {stats['match_screens']}")
    print(f"  Failed/Missing: {stats['failed']}")
    if errors:
        print(f"First few errors:\n  " + "\n  ".join(errors[:5]))

    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(args.source, output_dir, args.scores)
    if args.limit > 0:
        tasks = tasks[:args.limit]

    run_batch_crop(
        tasks=tasks,
        crop_box=DEFAULT_CROP_BOX,
        overwrite=args.overwrite,
        filter_matches=not args.no_filter_matches,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
