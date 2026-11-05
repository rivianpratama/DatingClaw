"""Paths, protocol constants, and explicit environment setup for the v6 revision."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Read-only legacy sources.
LABELS_CSV = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
SCORES_CSV = Path(r"D:\BumbleLog\scores.csv")
TRAIN_D2_NPZ = Path(r"D:\master_50k_store.npz")
TRAIN_D3_NPZ = Path(r"D:\master_50k_dinov3_store.npz")
HOLDOUT_NPZ = Path(r"D:\holdout_4540_store.npz")
VAL_1582_CSV = Path(r"D:\BumbleLog\binary_swipe_labels_val_1582.csv")
VAL_1582_ARC = REPO_ROOT / "embeddings" / "reference_store_val_1582.npz"
IMAGE_ROOT = Path(r"D:\BumbleLog")

# New artifacts must not live inside the repository or the read-only source dirs.
V6R1_DIR = Path(os.environ.get("BUMBLECLAW_V6R1_DIR", r"D:\bumbleclaw_v6r1"))
DATASETS_DIR = V6R1_DIR / "datasets"
RESULTS_DIR = V6R1_DIR / "results"
LABELING_DIR = V6R1_DIR / "labeling"
RELEASES_DIR = V6R1_DIR / "releases"

# Historical protocol constants. Counts are reproduced, not used to fabricate data.
N_HISTORICAL_SAMPLES = 54_610
N_HISTORICAL_LIKES = 5_091
FOLD_LEN = 4_540
FOLD_STARTS = (31_910, 36_450, 40_990, 45_530, 50_070)
PRIMARY_FOLDS = ("F2", "F3", "F4", "F5")
QUOTA = 0.208
DIAGNOSTIC_BUDGET_PER_DECK = 10
DIAGNOSTIC_DECK_SIZE = 4_540
CAT_FP_THRESHOLD = 50.0
CUTE_THRESHOLD = 60.0
SEEDS = (0, 1, 2)
WD_GRID = (0.03, 0.1, 0.3, 1.0)
LAMBDA_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0)

# Protocol schema versions.
SNAPSHOT_SCHEMA = "bumbleclaw_snapshot_v1"
SPLIT_SCHEMA = "matched_pipeline_v1"
HISTORICAL_SCHEMA = "historical_reproduction_v1"
RUN_SCHEMA = "bumbleclaw_run_v1"
CALIBRATOR_SCHEMA = "platt_v1"
POLICY_SCHEMA = "expected_cost_policy_v1"

LEGACY_FEATURE_RECIPE = "legacy_store_v0"
MATCHED_PURGE_POLICY = "conservative-group-purge-v1"


def ensure_artifact_dirs() -> None:
    """Create only the new artifact tree. Never call at import time."""
    for path in (V6R1_DIR, DATASETS_DIR, RESULTS_DIR, LABELING_DIR, RELEASES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_env() -> None:
    """Put temporary/cache files on D: when a command explicitly opts in."""
    temp = Path(r"D:\BumbleTemp")
    temp.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = str(temp)
    os.environ.setdefault("HF_HOME", r"D:\BumbleClawHFCache")
    ensure_artifact_dirs()
