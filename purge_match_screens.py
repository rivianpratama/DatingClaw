from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import shutil

LOG_DIR = Path(r"D:\BumbleLog")
CROPPED_DIR = LOG_DIR / "cropped"
SCORES_CSV = LOG_DIR / "scores.csv"
LABELS_CSV = LOG_DIR / "binary_swipe_labels.csv"
MATCH_LIST_PATH = Path("match_screens.txt")


def main() -> None:
    if not MATCH_LIST_PATH.exists():
        print("match_screens.txt not found!")
        return

    with open(MATCH_LIST_PATH, encoding="utf-8") as f:
        targets = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(targets)} match screen filenames to purge.")

    # 1. Delete image files
    deleted_raw = 0
    deleted_cropped = 0
    for name in targets:
        raw_p = LOG_DIR / name
        if raw_p.exists():
            raw_p.unlink()
            deleted_raw += 1
        
        crop_p = CROPPED_DIR / name
        if crop_p.exists():
            crop_p.unlink()
            deleted_cropped += 1

    print(f"Deleted {deleted_raw} raw images from {LOG_DIR}")
    print(f"Deleted {deleted_cropped} cropped images from {CROPPED_DIR}")

    # 2. Back up and clean scores.csv
    if SCORES_CSV.exists():
        bak_name = f"scores.csv.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(SCORES_CSV, LOG_DIR / bak_name)
        print(f"Backed up scores.csv to {bak_name}")

        with open(SCORES_CSV, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            all_rows = list(reader)

        original_count = len(all_rows)
        clean_rows = [r for r in all_rows if (r.get("screenshot") or "").strip() not in targets]
        removed_count = original_count - len(clean_rows)

        with open(SCORES_CSV, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

        print(f"scores.csv: Removed {removed_count} rows ({original_count} -> {len(clean_rows)})")

    # 3. Clean binary_swipe_labels.csv if any match screen was swiped
    if LABELS_CSV.exists():
        with open(LABELS_CSV, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            label_rows = list(reader)

        original_labels = len(label_rows)
        clean_labels = [r for r in label_rows if (r.get("filename") or "").strip() not in targets]
        if len(clean_labels) < original_labels:
            with open(LABELS_CSV, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(clean_labels)
            print(f"binary_swipe_labels.csv: Removed {original_labels - len(clean_labels)} swiped match screens")
        else:
            print(f"binary_swipe_labels.csv: No match screens were swiped ({original_labels} labels intact)")

    print("\nPurge completed successfully!")


if __name__ == "__main__":
    main()
