from __future__ import annotations

import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

MATCH_TEMPLATE = cv2.imread(str(Path("assets") / "match_template.png"))
SCORES_CSV = Path(r"D:\BumbleLog\scores.csv")
SOURCE_DIR = Path(r"D:\BumbleLog")


def check_file(filename: str) -> dict[str, object] | None:
    p = SOURCE_DIR / filename
    if not p.exists():
        return None
    try:
        img = cv2.imread(str(p))
        if img is None:
            return None
        h, w = img.shape[:2]

        # Top 20% white check
        top_band = img[:int(h * 0.20), :]
        white_top = float(np.mean((top_band > 235).all(axis=2)))
        if white_top < 0.70:
            return None

        # Bottom band white check
        bot_band = img[int(h * 0.65):int(h * 0.90), :]
        white_bot = float(np.mean((bot_band > 235).all(axis=2)))

        # Scaled template match
        scale = w / 540.0
        if abs(scale - 1.0) > 0.05:
            scaled_tmpl = cv2.resize(MATCH_TEMPLATE, (0, 0), fx=scale, fy=scale)
        else:
            scaled_tmpl = MATCH_TEMPLATE

        th, tw = scaled_tmpl.shape[:2]
        if h >= th and w >= tw:
            res = cv2.matchTemplate(img, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
        else:
            score = 0.0

        # Also check whether center between cards has high white
        mid_col = img[int(h * 0.35):int(h * 0.65), int(w * 0.45):int(w * 0.55)]
        white_mid = float(np.mean((mid_col > 235).all(axis=2)))

        # If it's a match screen, it has high white_top (>75%) AND (template_match > 0.50 OR (white_bot > 0.65 and white_mid > 0.40))
        is_match = (white_top > 0.75) and (score > 0.50 or (white_bot > 0.65 and white_mid > 0.35))

        return {
            "filename": filename,
            "is_match": is_match,
            "white_top": white_top,
            "white_bot": white_bot,
            "white_mid": white_mid,
            "match_score": score,
            "shape": f"{w}x{h}",
        }
    except Exception:
        return None


def main() -> None:
    seen = set()
    filenames = []
    if SCORES_CSV.exists():
        with open(SCORES_CSV, mode="r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                s = (row.get("screenshot") or "").strip()
                if s and s not in seen:
                    seen.add(s)
                    filenames.append(s)

    print(f"Scanning {len(filenames)} images across 8 CPU processes...")
    matches = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for res in ex.map(check_file, filenames, chunksize=150):
            if res and res["is_match"]:
                matches.append(res)

    print(f"\nScan complete. Total match screens detected: {len(matches)}")
    output_txt = Path("match_screens.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        for m in matches:
            f.write(f"{m['filename']}\n")
            print(f"  {m['filename']} (score={m['match_score']:.2f}, top_w={m['white_top']:.2f}, bot_w={m['white_bot']:.2f})")
    print(f"Saved list to {output_txt}")


if __name__ == "__main__":
    main()
