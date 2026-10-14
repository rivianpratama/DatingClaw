from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from bumble_train import crop_image
from face_similarity.clip_embedding import get_clip_embedding
from face_similarity.clip_runtime import ensure_clip_runtime
from face_similarity.dynamic_threshold import threshold_from_scores
from face_similarity.embedding import get_face_embedding
from face_similarity.experimental_setup import EXPERIMENTAL1, MULTIMODALX, MULTIMODALX2
from face_similarity.labeling import load_label_rows, natural_sort_key
from face_similarity.multimodalx import METHOD as MULTIMODALX_METHOD
from face_similarity.multimodalx import METHOD2 as MULTIMODALX2_METHOD
from face_similarity.multimodalx import score as multimodalx_score
from face_similarity.preference import (
    FEATURE_FIELDS,
    BucketLikeRateClassifier,
    feature_vector,
    load_preference_model,
    make_features,
    parse_float,
    preference_probability,
    save_preference_model,
)
from face_similarity.prediction import biased_multimodal_score, predict_image_rating
from face_similarity.regressor import load_regressor, predict_multimodal_rating, predict_rating
from face_similarity.scoring import score_embedding
from face_similarity.store import load_store
from face_similarity.warnings import suppress_known_third_party_warnings


BUCKET_QUOTAS = [
    (0.0, 20.0, 200),
    (20.0, 30.0, 300),
    (30.0, 40.0, 500),
    (40.0, 50.0, 650),
    (50.0, 60.0, 650),
    (60.0, 70.0, 450),
    (70.0, 80.0, 200),
    (80.0, 100.0001, 50),
]

MANIFEST_FIELDS = [
    "selection_reason",
    "timestamp",
    "screenshot",
    "action",
    "source_path",
    "selected_path",
    "score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
    "threshold",
    "face_weight",
    "regressor_path",
    "multimodal_regressor_path",
    "component_spread",
    "ridge_minus_multimodal",
    "distance_from_threshold",
    "score_bucket",
]

REPORT_FIELDS = [
    "model",
    "split",
    "count",
    "probability_threshold",
    "swipe_errors",
    "swipe_error_rate",
    "false_positive",
    "false_negative",
    "right_swipe_rate",
    "precision",
    "recall",
    "accuracy",
    "auc",
    "log_loss",
]

CALIBRATION_FIELDS = [
    "model",
    "score_bucket",
    "count",
    "actual_like_rate",
    "predicted_like_rate",
    "right_swipe_rate",
    "precision",
    "recall",
]

FACE_WEIGHT_SWEEP = [0.0, 0.1, 0.2, 0.22, 0.3, 0.4, 0.44, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

COMPONENT_SCORE_FIELDS = [
    "timestamp",
    "selected_path",
    "source_path",
    "generation",
    "store_path",
    "regressor_path",
    "multimodal_regressor_path",
    "ridge",
    "multimodal",
    "knn",
    "error",
]

MODEL_SCORE_FIELDS = [
    "timestamp",
    "selected_path",
    "generation",
    "method",
    "face_weight",
    "score",
    "face_biased",
    "multimodal",
    "ridge",
    "knn",
    "threshold",
    "regressor_path",
    "multimodal_regressor_path",
    "component_spread",
    "ridge_minus_multimodal",
    "distance_from_threshold",
    "score_bucket",
]

MODEL_BENCHMARK_FIELDS = [
    "strategy",
    "generation",
    "method",
    "face_weight",
    "config_id",
    "preference_model",
    "probability_threshold",
    "count",
    "swipe_errors",
    "swipe_error_rate",
    "false_positive",
    "false_negative",
    "right_swipe_rate",
    "precision",
    "recall",
    "accuracy",
    "auc",
    "log_loss",
    "store_path",
    "regressor_path",
    "multimodal_regressor_path",
    "score_cache",
]

BEST_MODEL_FIELDS = ["selection", *MODEL_BENCHMARK_FIELDS]

VETO_COMPONENT_FIELDS = [
    "timestamp",
    "screenshot",
    "source_path",
    "generation",
    "store_path",
    "regressor_path",
    "multimodal_regressor_path",
    "ridge",
    "multimodal",
    "knn_k9",
    "knn_k11",
    "error",
]

VETO_MANIFEST_FIELDS = [
    "timestamp",
    "screenshot",
    "source_path",
    "selected_path",
    "disagreement_pattern",
    "round3_score",
    "round3_threshold",
    "round3_action",
    "round3_error",
    "multimodalx_score",
    "multimodalx_preference_probability",
    "multimodalx_threshold",
    "multimodalx_action",
    "multimodalx_error",
    "multimodalx2_score",
    "multimodalx2_preference_probability",
    "multimodalx2_threshold",
    "multimodalx2_action",
    "multimodalx2_error",
    "experimental1_score",
    "experimental1_probability",
    "experimental1_threshold",
    "experimental1_action",
    "experimental1_error",
]

VETO_PREPARATION_STATS_FIELDS = [
    "eligible_unseen_count",
    "scored_count",
    "round3_scoring_failure_count",
    "multimodalx_scoring_failure_count",
    "multimodalx2_scoring_failure_count",
    "experimental1_scoring_failure_count",
    "unanimous_left_excluded_count",
    "unanimous_right_excluded_count",
    "disagreement_selected_count",
]

VETO_REPORT_FIELDS = [
    "timestamp",
    "screenshot",
    "selected_path",
    "disagreement_pattern",
    "veto_action",
    "round3_action",
    "round3_agrees",
    "multimodalx_action",
    "multimodalx_agrees",
    "multimodalx2_action",
    "multimodalx2_agrees",
    "experimental1_action",
    "experimental1_agrees",
]

VETO_SUMMARY_FIELDS = [
    "model",
    "labeled_disagreement_count",
    "agreement_count",
    "agreement_rate",
    "false_positive",
    "false_negative",
    "model_right_swipe_rate",
    "veto_right_swipe_rate",
    *VETO_PREPARATION_STATS_FIELDS,
    "note",
]

VETO_LAYER_BENCHMARK_FIELDS = [
    "lane",
    "strategy",
    "target_right_rate",
    *REPORT_FIELDS,
]

VETO_LAYER_BEST_FIELDS = ["selection", *VETO_LAYER_BENCHMARK_FIELDS]

VETO_X3_OLD_P_LIKE_FIELD = "old_p_like"
VETO_X3_FORMULA_SWEEP_FIELDS = [
    "config_id",
    "knn_k",
    "explicit_old_p_like",
    "ridge_weight",
    "multimodal_weight",
    "knn_weight",
    "old_p_like_weight",
    *REPORT_FIELDS,
]
VETO_X3_FINAL_FIELDS = ["selection", *VETO_X3_FORMULA_SWEEP_FIELDS]

VETO_LANES = [
    ("Round3", "round3"),
    ("MultimodalX", "multimodalx"),
    ("MultimodalX2", "multimodalx2"),
    ("Experimental1", "experimental1"),
]
VETO_DISAGREEMENT_NOTE = "Agreement metrics are measured on the disagreement-only pool, not the full unseen pool."


@dataclass(frozen=True)
class PreparedCandidate:
    row: dict[str, str]
    image_path: Path
    score: float


@dataclass(frozen=True)
class VetoCandidate:
    row: dict[str, str]
    image_path: Path
    screenshot_stem: str


@dataclass(frozen=True)
class EvalResult:
    model: str
    split: str
    count: int
    probability_threshold: float
    swipe_errors: int
    swipe_error_rate: float
    false_positive: int
    false_negative: int
    right_swipe_rate: float
    precision: float
    recall: float
    accuracy: float
    auc: float
    log_loss: float


@dataclass(frozen=True)
class ModelBundle:
    generation: str
    store_path: Path
    regressor_path: Path | None
    multimodal_regressor_path: Path | None
    default_face_weight: float
    deploy_rank: int


@dataclass(frozen=True)
class ModelConfig:
    bundle: ModelBundle
    method: str
    face_weight: float

    @property
    def config_id(self) -> str:
        if self.method == "face_biased":
            return f"{self.bundle.generation}__{self.method}__w{self.face_weight:g}".replace(".", "p")
        return f"{self.bundle.generation}__{self.method}"


@dataclass(frozen=True)
class VetoX3Formula:
    ridge_weight: float
    multimodal_weight: float
    knn_weight: float
    old_p_like_weight: float
    knn_k: int
    explicit_old_p_like: bool

    @property
    def config_id(self) -> str:
        weights = (
            f"r{self.ridge_weight:g}",
            f"m{self.multimodal_weight:g}",
            f"k{self.knn_weight:g}",
            f"p{self.old_p_like_weight:g}",
        )
        suffix = "with_plike_feature" if self.explicit_old_p_like else "score_plike_only"
        return f"x3__k{self.knn_k}__{'_'.join(weights)}__{suffix}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and train binary Bumble preference models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare balanced historical screenshots for binary labeling")
    prepare.add_argument("--source", default=r"D:\BumbleLog", help="Bumble log folder with scores.csv and screenshots")
    prepare.add_argument("--output", default=r"D:\BumblePreferenceTrain", help="Output preference workspace")
    prepare.add_argument("--seed", type=int, default=42, help="Deterministic selection seed")
    prepare.add_argument("--crop-left", type=float, default=0.0)
    prepare.add_argument("--crop-top", type=float, default=0.075)
    prepare.add_argument("--crop-right", type=float, default=1.0)
    prepare.add_argument("--crop-bottom", type=float, default=0.925)
    prepare.add_argument("--mask-share-icon", action="store_true", help="Mask fixed share icon in prepared images")

    train = subparsers.add_parser("train", help="Train and benchmark a binary preference classifier")
    train.add_argument("--manifest", default=r"D:\BumblePreferenceTrain\manifests\selection.csv")
    train.add_argument("--labels", default=r"D:\BumblePreferenceTrain\labels\binary_preference_labels.csv")
    train.add_argument("--output", default=r"models\bumble_preference_classifier.joblib")
    train.add_argument("--report", default=r"results\bumble_preference_benchmark.csv")
    train.add_argument("--target-right-rate", type=float, default=0.225)
    train.add_argument("--score-threshold", type=float, default=55.0)
    train.add_argument("--dynamic-window", type=int, default=200)
    train.add_argument("--dynamic-min-threshold", type=float, default=48.0)
    train.add_argument("--dynamic-max-threshold", type=float, default=70.0)
    train.add_argument("--train-fraction", type=float, default=0.8)
    train.add_argument("--random-state", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark-models", help="Benchmark all scoring generations with preference models on top")
    benchmark.add_argument("--manifest", default=r"D:\BumblePreferenceTrain\manifests\selection.csv")
    benchmark.add_argument("--labels", default=r"D:\BumblePreferenceTrain\labels\binary_preference_labels.csv")
    benchmark.add_argument("--preference-model", default=r"models\bumble_preference_classifier.joblib")
    benchmark.add_argument("--output", default=r"results\preference_top_model_benchmark.csv")
    benchmark.add_argument("--best-output", default=r"results\preference_top_model_best.csv")
    benchmark.add_argument("--calibration-output", default=r"results\preference_top_model_bucket_calibration.csv")
    benchmark.add_argument("--cache-dir", default=r"D:\BumblePreferenceTrain\cache\model_scores")
    benchmark.add_argument("--target-right-rate", type=float, default=0.20)
    benchmark.add_argument("--score-threshold", type=float, default=55.0)
    benchmark.add_argument("--train-fraction", type=float, default=0.8)
    benchmark.add_argument("--random-state", type=int, default=42)
    benchmark.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    benchmark.add_argument("--k", type=int, default=11)
    benchmark.add_argument("--force-rescore", action="store_true")

    veto_prepare = subparsers.add_parser(
        "prepare-veto-eval",
        help="Score unseen Bumble logs and prepare disagreement-only binary veto labeling",
    )
    veto_prepare.add_argument("--source", default=r"D:\BumbleLog", help="Bumble log folder with scores.csv and screenshots")
    veto_prepare.add_argument("--output", default=r"D:\BumbleDecisionEval", help="Output veto evaluation workspace")
    veto_prepare.add_argument(
        "--preference-manifest",
        default=r"D:\BumblePreferenceTrain\manifests\selection.csv",
        help="Binary preference training manifest to exclude",
    )
    veto_prepare.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="cuda")
    veto_prepare.add_argument("--dynamic-window", type=int, default=200)
    veto_prepare.add_argument("--dynamic-min-history", type=int, default=50)
    veto_prepare.add_argument("--crop-left", type=float, default=0.0)
    veto_prepare.add_argument("--crop-top", type=float, default=0.075)
    veto_prepare.add_argument("--crop-right", type=float, default=1.0)
    veto_prepare.add_argument("--crop-bottom", type=float, default=0.925)
    veto_prepare.add_argument("--mask-share-icon", action="store_true", help="Mask fixed share icon in prepared images")
    veto_prepare.add_argument("--force-rescore", action="store_true", help="Ignore existing veto component caches")

    veto_report = subparsers.add_parser(
        "report-veto-eval",
        help="Compare disagreement veto labels against each prepared model decision",
    )
    veto_report.add_argument(
        "--manifest",
        default=r"D:\BumbleDecisionEval\manifests\disagreement_selection.csv",
        help="Disagreement manifest from prepare-veto-eval",
    )
    veto_report.add_argument(
        "--labels",
        default=r"D:\BumbleDecisionEval\labels\veto_labels.csv",
        help="Binary veto labels from label_app.py",
    )
    veto_report.add_argument(
        "--stats",
        default=r"D:\BumbleDecisionEval\reports\preparation_stats.csv",
        help="Preparation stats CSV from prepare-veto-eval",
    )
    veto_report.add_argument(
        "--report",
        default=r"D:\BumbleDecisionEval\reports\model_agreement_report.csv",
        help="Per-picture agreement report output",
    )
    veto_report.add_argument(
        "--summary",
        default=r"D:\BumbleDecisionEval\reports\model_agreement_summary.csv",
        help="Per-model agreement summary output",
    )

    veto_benchmark = subparsers.add_parser(
        "benchmark-veto-layers",
        help="Benchmark new Round2, Round3, and MultimodalX2 decision layers from veto labels",
    )
    veto_benchmark.add_argument(
        "--manifest",
        default=r"D:\BumbleDecisionEval\manifests\disagreement_selection.csv",
        help="Disagreement manifest from prepare-veto-eval",
    )
    veto_benchmark.add_argument(
        "--labels",
        default=r"D:\BumbleDecisionEval\labels\veto_labels.csv",
        help="Binary veto labels from label_app.py",
    )
    veto_benchmark.add_argument(
        "--original-components",
        default=r"D:\BumbleDecisionEval\cache\original__veto_components.csv",
        help="Original veto component cache for labeled veto screenshots",
    )
    veto_benchmark.add_argument(
        "--round2-components",
        default=r"D:\BumbleDecisionEval\cache\bumble_combined_round2__veto_components.csv",
        help="Round2 veto component cache from prepare-veto-eval",
    )
    veto_benchmark.add_argument(
        "--round3-components",
        default=r"D:\BumbleDecisionEval\cache\bumble_combined_round3__veto_components.csv",
        help="Round3 veto component cache from prepare-veto-eval",
    )
    veto_benchmark.add_argument(
        "--output",
        default=r"D:\BumbleDecisionEval\reports\veto_decision_layer_benchmark.csv",
        help="Decision layer benchmark output",
    )
    veto_benchmark.add_argument(
        "--best-output",
        default=r"D:\BumbleDecisionEval\reports\veto_decision_layer_best.csv",
        help="Best decision layer output",
    )
    veto_benchmark.add_argument(
        "--calibration-output",
        default=r"D:\BumbleDecisionEval\reports\veto_decision_layer_bucket_calibration.csv",
        help="Winning decision layer bucket calibration output",
    )
    veto_benchmark.add_argument(
        "--original-model-output",
        default=r"models\bumble_preference_experimental3.joblib",
        help="Deployable Original veto-layer model output",
    )
    veto_benchmark.add_argument(
        "--round2-model-output",
        default=r"models\bumble_preference_round2_veto.joblib",
        help="Deployable Round2 veto-layer model output",
    )
    veto_benchmark.add_argument(
        "--model-output",
        default=r"models\bumble_preference_multimodalx3.joblib",
        help="Deployable Round3 veto-layer model output",
    )
    veto_benchmark.add_argument(
        "--multimodalx4-model-output",
        default=r"models\bumble_preference_multimodalx4.joblib",
        help="Deployable MultimodalX2 veto-layer model output",
    )
    veto_benchmark.add_argument(
        "--target-right-rate",
        type=float,
        default=0.20,
        help="Validation decision target; defaults to the runtime P80 right-swipe rate",
    )
    veto_benchmark.add_argument("--train-fraction", type=float, default=0.8)
    veto_benchmark.add_argument("--random-state", type=int, default=42)

    veto_x3 = subparsers.add_parser(
        "tune-veto-x3",
        help="Tune MultimodalX3 score formulas on veto labels before deployment",
    )
    veto_x3.add_argument(
        "--manifest",
        default=r"D:\BumbleDecisionEval\manifests\disagreement_selection.csv",
        help="Disagreement manifest from prepare-veto-eval",
    )
    veto_x3.add_argument(
        "--labels",
        default=r"D:\BumbleDecisionEval\labels\veto_labels.csv",
        help="Binary veto labels from label_app.py",
    )
    veto_x3.add_argument(
        "--round3-components",
        default=r"D:\BumbleDecisionEval\cache\bumble_combined_round3__veto_components.csv",
        help="Round3 veto component cache from prepare-veto-eval",
    )
    veto_x3.add_argument(
        "--sweep-output",
        default=r"D:\BumbleDecisionEval\reports\veto_x3_formula_sweep.csv",
        help="Inner tuning sweep over MultimodalX3 formulas",
    )
    veto_x3.add_argument(
        "--final-output",
        default=r"D:\BumbleDecisionEval\reports\veto_x3_formula_finalists.csv",
        help="Held-out final evaluation for tuned MultimodalX3 finalists",
    )
    veto_x3.add_argument(
        "--model-output",
        default=r"models\bumble_preference_multimodalx5.joblib",
        help="Deployable veto spline for the held-out winning formula",
    )
    veto_x3.add_argument("--target-right-rate", type=float, default=0.20)
    veto_x3.add_argument("--train-fraction", type=float, default=0.8)
    veto_x3.add_argument("--tune-fraction", type=float, default=0.8)
    veto_x3.add_argument("--weight-step", type=float, default=0.10)
    veto_x3.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    suppress_known_third_party_warnings()
    args = parse_args()
    if args.command == "prepare":
        prepare_workspace(args)
    elif args.command == "train":
        train_preference_model(args)
    elif args.command == "benchmark-models":
        benchmark_models(args)
    elif args.command == "prepare-veto-eval":
        ensure_clip_runtime("face_biased")
        prepare_veto_eval(args)
    elif args.command == "report-veto-eval":
        report_veto_eval(args)
    elif args.command == "benchmark-veto-layers":
        benchmark_veto_layers(args)
    elif args.command == "tune-veto-x3":
        tune_veto_x3(args)
    return 0


def prepare_workspace(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    scores_path = source / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv does not exist: {scores_path}")

    selected_dir = output / "selected"
    manifest_dir = output / "manifests"
    for directory in (selected_dir, manifest_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(scores_path, source)
    selected = select_balanced(candidates, seed=args.seed)
    rows = []
    crop_box = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    for index, candidate in enumerate(selected, start=1):
        selected_path = selected_dir / f"{Path(candidate.row['screenshot']).stem}.jpg"
        crop_image(
            candidate.image_path,
            selected_path,
            crop_box=crop_box,
            mask_share_icon=args.mask_share_icon,
        )
        features = make_features(candidate.row)
        rows.append(
            {
                "selection_reason": bucket_reason(candidate.score),
                "timestamp": candidate.row.get("timestamp", ""),
                "screenshot": candidate.row.get("screenshot", ""),
                "action": candidate.row.get("action", ""),
                "source_path": candidate.image_path.resolve().as_posix(),
                "selected_path": selected_path.resolve().as_posix(),
                "score": format_float(candidate.score),
                "face_biased": candidate.row.get("face_biased", ""),
                "multimodal": candidate.row.get("multimodal", ""),
                "ridge": candidate.row.get("ridge", ""),
                "knn": candidate.row.get("knn", ""),
                "threshold": candidate.row.get("threshold", ""),
                "face_weight": candidate.row.get("face_weight", ""),
                "regressor_path": candidate.row.get("regressor_path", ""),
                "multimodal_regressor_path": candidate.row.get("multimodal_regressor_path", ""),
                "component_spread": format_float(features["component_spread"]),
                "ridge_minus_multimodal": format_float(features["ridge_minus_multimodal"]),
                "distance_from_threshold": format_float(features["distance_from_threshold"]),
                "score_bucket": format_float(features["score_bucket"]),
            }
        )
        if index % 250 == 0:
            print(f"Prepared {index} image(s)")

    manifest_path = manifest_dir / "selection.csv"
    write_rows(manifest_path, rows, MANIFEST_FIELDS)
    print(f"Selected {len(rows)} image(s)")
    print(f"Selection manifest: {manifest_path}")
    print("Label with:")
    print(
        "  D:\\DatingClawClipVenv\\Scripts\\python.exe label_app.py "
        f"--binary --source-dir {selected_dir} "
        f"--output-csv {output / 'labels' / 'binary_preference_labels.csv'} "
        "--port 7863"
    )


def read_candidates(scores_path: Path, source: Path) -> list[PreparedCandidate]:
    candidates = []
    for row in read_csv(scores_path):
        screenshot = (row.get("screenshot") or "").strip()
        score = parse_float(row.get("score"))
        if not screenshot or score is None:
            continue
        image_path = source / screenshot
        if not image_path.exists():
            continue
        candidates.append(PreparedCandidate(row=row, image_path=image_path, score=score))
    return candidates


def select_balanced(candidates: list[PreparedCandidate], *, seed: int) -> list[PreparedCandidate]:
    rng = random.Random(seed)
    selected: list[PreparedCandidate] = []
    selected_names: set[str] = set()
    for lower, upper, quota in BUCKET_QUOTAS:
        pool = [
            candidate
            for candidate in candidates
            if lower <= candidate.score < upper and candidate.row["screenshot"] not in selected_names
        ]
        pool.sort(key=lambda candidate: natural_sort_key(candidate.row["screenshot"]))
        rng.shuffle(pool)
        if len(pool) < quota:
            raise ValueError(f"Only {len(pool)} candidate(s) for {lower:g}-{upper:g}, need {quota}")
        chosen = pool[:quota]
        selected.extend(chosen)
        selected_names.update(candidate.row["screenshot"] for candidate in chosen)
    rng.shuffle(selected)
    return selected


def train_preference_model(args: argparse.Namespace) -> None:
    if not 0 < args.target_right_rate < 1:
        raise ValueError("--target-right-rate must be between 0 and 1")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")

    rows = joined_labeled_rows(Path(args.manifest), Path(args.labels))
    rows.sort(key=lambda row: row.get("timestamp", ""))
    split_at = max(1, min(len(rows) - 1, int(len(rows) * args.train_fraction)))
    train_rows = rows[:split_at]
    validation_rows = rows[split_at:]

    x_train, y_train = matrix(train_rows)
    x_validation, y_validation = matrix(validation_rows)

    results = [
        evaluate_baseline("score_threshold", validation_rows, y_validation, args.score_threshold),
        evaluate_dynamic_baseline(
            "dynamic_threshold",
            train_rows,
            validation_rows,
            y_validation,
            target_right_rate=args.target_right_rate,
            window=args.dynamic_window,
            min_threshold=args.dynamic_min_threshold,
            max_threshold=args.dynamic_max_threshold,
        ),
        evaluate_baseline("logged_action", validation_rows, y_validation, None),
    ]

    estimators = {
        "bucket_like_rate": BucketLikeRateClassifier(score_index=FEATURE_FIELDS.index("score")),
        "spline_logistic": spline_logistic(random_state=args.random_state),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, random_state=args.random_state),
        ),
    }

    trained = []
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        probability_threshold = probability_threshold_for_rate(probabilities, args.target_right_rate)
        result = evaluate_probabilities(name, y_validation, probabilities, probability_threshold)
        results.append(result)
        trained.append((result, estimator, probabilities))

    best_result, best_estimator, best_probabilities = min(
        trained,
        key=lambda item: (
            item[0].swipe_errors,
            item[0].false_negative,
            abs(item[0].right_swipe_rate - args.target_right_rate),
            item[0].log_loss,
        ),
    )
    save_preference_model(
        args.output,
        estimator=best_estimator,
        model_name=best_result.model,
        threshold=best_result.probability_threshold,
        metrics={
            "swipe_error_rate": best_result.swipe_error_rate,
            "right_swipe_rate": best_result.right_swipe_rate,
            "precision": best_result.precision,
            "recall": best_result.recall,
            "auc": best_result.auc,
            "log_loss": best_result.log_loss,
            "validation_count": float(best_result.count),
        },
    )
    write_report(Path(args.report), results)
    calibration_path = Path(args.report).with_name("bumble_preference_bucket_calibration.csv")
    write_bucket_calibration(
        calibration_path,
        validation_rows,
        y_validation,
        best_probabilities,
        best_result.probability_threshold,
        model_name=best_result.model,
    )
    print(f"Labeled examples: {len(rows)}")
    print(f"Train: {len(train_rows)} Validation: {len(validation_rows)}")
    print(
        f"Best model: {best_result.model} threshold={best_result.probability_threshold:.4f} "
        f"swipe_error={best_result.swipe_errors}/{best_result.count} "
        f"precision={best_result.precision:.4f} recall={best_result.recall:.4f}"
    )
    print(f"Saved model to {args.output}")
    print(f"Saved benchmark to {args.report}")
    print(f"Saved bucket calibration to {calibration_path}")


def benchmark_models(args: argparse.Namespace) -> None:
    if not 0 < args.target_right_rate < 1:
        raise ValueError("--target-right-rate must be between 0 and 1")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.k < 1:
        raise ValueError("--k must be at least 1")

    rows = joined_labeled_rows(Path(args.manifest), Path(args.labels))
    train_rows, validation_rows = chronological_split(rows, train_fraction=args.train_fraction)
    _, y_train = matrix(train_rows)
    _, y_validation = matrix(validation_rows)
    current_preference = load_preference_model(args.preference_model)
    cache_dir = Path(args.cache_dir)
    configs = model_grid()
    if not configs:
        raise ValueError("No benchmarkable model artifacts found")

    benchmark_rows: list[dict[str, object]] = []
    best_by_strategy: dict[str, tuple[dict[str, object], np.ndarray, list[dict[str, str]]]] = {}
    component_cache: dict[str, list[dict[str, str]]] = {}

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.config_id}")
        bundle = config.bundle
        if bundle.generation not in component_cache:
            component_cache[bundle.generation] = ensure_component_score_cache(
                bundle,
                rows,
                cache_dir=cache_dir,
                provider=args.provider,
                k=args.k,
                force=args.force_rescore,
            )
        score_rows = ensure_model_score_cache(
            config,
            component_cache[bundle.generation],
            cache_dir=cache_dir,
            threshold=args.score_threshold,
            force=args.force_rescore,
        )
        config_train_rows, config_validation_rows = chronological_split(score_rows, train_fraction=args.train_fraction)

        x_validation_reuse = feature_matrix(config_validation_rows, current_preference.feature_fields)
        reuse_probabilities = current_preference.estimator.predict_proba(x_validation_reuse)[:, 1]
        reuse_threshold = probability_threshold_for_rate(reuse_probabilities, args.target_right_rate)
        reuse_result = evaluate_probabilities("reuse_current_preference", y_validation, reuse_probabilities, reuse_threshold)
        reuse_row = benchmark_row(
            config,
            strategy="reuse_current",
            preference_model=current_preference.model_name,
            result=reuse_result,
            score_cache=cache_dir / f"{config.config_id}.csv",
        )
        benchmark_rows.append(reuse_row)
        update_best(best_by_strategy, reuse_row, reuse_probabilities, config_validation_rows, target_right_rate=args.target_right_rate)

        retrain_result, retrained_name, retrain_probabilities = train_config_preference(
            config_train_rows,
            y_train,
            config_validation_rows,
            y_validation,
            target_right_rate=args.target_right_rate,
            random_state=args.random_state,
        )
        retrain_row = benchmark_row(
            config,
            strategy="retrained_per_config",
            preference_model=retrained_name,
            result=retrain_result,
            score_cache=cache_dir / f"{config.config_id}.csv",
        )
        benchmark_rows.append(retrain_row)
        update_best(best_by_strategy, retrain_row, retrain_probabilities, config_validation_rows, target_right_rate=args.target_right_rate)

    write_rows(Path(args.output), benchmark_rows, MODEL_BENCHMARK_FIELDS)
    best_rows = best_report_rows(best_by_strategy)
    write_rows(Path(args.best_output), best_rows, BEST_MODEL_FIELDS)
    recommended = next((row for row in best_rows if row["selection"] == "overall_recommendation"), None)
    if recommended is not None:
        _, probabilities, winner_validation_rows = best_by_strategy[str(recommended["strategy"])]
        write_bucket_calibration(
            Path(args.calibration_output),
            winner_validation_rows,
            y_validation,
            probabilities,
            float(recommended["probability_threshold"]),
            model_name=f"{recommended['generation']} {recommended['method']} {recommended['strategy']}",
        )

    print(f"Benchmarked {len(configs)} configs")
    print(f"Saved benchmark to {args.output}")
    print(f"Saved best configs to {args.best_output}")
    print(f"Saved winner calibration to {args.calibration_output}")


def prepare_veto_eval(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    if args.dynamic_window < 1:
        raise ValueError("--dynamic-window must be at least 1")
    if args.dynamic_min_history < 1:
        raise ValueError("--dynamic-min-history must be at least 1")
    scores_path = source / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv does not exist: {scores_path}")

    selected_dir = output / "selected"
    manifest_path = output / "manifests" / "disagreement_selection.csv"
    labels_dir = output / "labels"
    cache_dir = output / "cache"
    reports_dir = output / "reports"
    for directory in (labels_dir, cache_dir, reports_dir, manifest_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    round2_bundle = required_bundle("bumble_combined_round2")
    round3_bundle = required_bundle("bumble_combined_round3")
    excluded_stems = excluded_veto_training_stems(
        [round2_bundle.store_path, round3_bundle.store_path],
        Path(args.preference_manifest),
    )
    candidates = eligible_veto_candidates(read_csv(scores_path), source=source, excluded_stems=excluded_stems)
    if not candidates:
        raise ValueError("No unseen Bumble log screenshots are eligible for veto evaluation")

    print(f"Eligible unseen screenshot(s): {len(candidates)}")
    round3_components = ensure_veto_component_cache(
        round3_bundle,
        candidates,
        cache_path=cache_dir / "bumble_combined_round3__veto_components.csv",
        provider=args.provider,
        knn_values=(9, 11),
        force=args.force_rescore,
    )
    round2_components = ensure_veto_component_cache(
        round2_bundle,
        candidates,
        cache_path=cache_dir / "bumble_combined_round2__veto_components.csv",
        provider=args.provider,
        knn_values=(11,),
        force=args.force_rescore,
    )
    model_rows = build_veto_model_rows(
        candidates,
        round3_components=round3_components,
        round2_components=round2_components,
    )
    apply_veto_dynamic_decisions(
        model_rows,
        dynamic_window=args.dynamic_window,
        dynamic_min_history=args.dynamic_min_history,
    )
    disagreement_rows = disagreement_veto_rows(model_rows)

    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    crop_box = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
    selected_rows = []
    for index, row in enumerate(disagreement_rows, start=1):
        selected_path = selected_dir / f"{Path(row['screenshot']).stem}.jpg"
        crop_image(
            Path(row["source_path"]),
            selected_path,
            crop_box=crop_box,
            mask_share_icon=args.mask_share_icon,
        )
        selected_row = dict(row)
        selected_row["selected_path"] = selected_path.resolve().as_posix()
        selected_rows.append(selected_row)
        if index % 250 == 0:
            print(f"Prepared {index}/{len(disagreement_rows)} disagreement image(s)")

    write_rows(manifest_path, selected_rows, VETO_MANIFEST_FIELDS)
    stats = veto_preparation_stats(candidates, model_rows, selected_rows)
    write_rows(reports_dir / "preparation_stats.csv", [stats], VETO_PREPARATION_STATS_FIELDS)
    print(f"Scored {len(model_rows)} unseen screenshot(s)")
    print(f"Selected {len(selected_rows)} disagreement screenshot(s)")
    print(f"Disagreement manifest: {manifest_path}")
    print("Label with:")
    print(
        "  D:\\DatingClawClipVenv\\Scripts\\python.exe label_app.py "
        f"--binary --source-dir {selected_dir} "
        f"--output-csv {labels_dir / 'veto_labels.csv'} "
        "--port 7863"
    )
    print("Report with:")
    print("  D:\\DatingClawClipVenv\\Scripts\\python.exe bumble_preference.py report-veto-eval")


def report_veto_eval(args: argparse.Namespace) -> None:
    manifest_rows = read_csv(Path(args.manifest))
    labels = load_label_rows(Path(args.labels))
    stats_rows = read_csv(Path(args.stats)) if Path(args.stats).exists() else []
    detail_rows, summary_rows = build_veto_report_rows(
        manifest_rows,
        labels=labels,
        stats=stats_rows[0] if stats_rows else {},
    )
    if not detail_rows:
        raise ValueError("No veto labels match the disagreement manifest")
    write_rows(Path(args.report), detail_rows, VETO_REPORT_FIELDS)
    write_rows(Path(args.summary), summary_rows, VETO_SUMMARY_FIELDS)
    print(f"Labeled disagreement screenshot(s): {len(detail_rows)}")
    print(f"Saved veto agreement report to {args.report}")
    print(f"Saved veto agreement summary to {args.summary}")


def benchmark_veto_layers(args: argparse.Namespace) -> None:
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.target_right_rate is not None and not 0 < args.target_right_rate < 1:
        raise ValueError("--target-right-rate must be between 0 and 1")

    manifest_rows = joined_labeled_rows(Path(args.manifest), Path(args.labels))
    original_component_rows = read_csv(Path(args.original_components))
    round2_component_rows = read_csv(Path(args.round2_components))
    round3_component_rows = read_csv(Path(args.round3_components))
    benchmark_rows = []
    best_layers: dict[str, tuple[dict[str, object], np.ndarray, list[dict[str, str]], np.ndarray]] = {}

    for lane, component_rows in (
        ("Original", original_component_rows),
        ("Round2", round2_component_rows),
        ("Round3", round3_component_rows),
        ("MultimodalX2", round3_component_rows),
    ):
        rows = veto_layer_rows(manifest_rows, component_rows, lane=lane)
        train_rows, validation_rows = chronological_split(rows, train_fraction=args.train_fraction)
        _, y_train = matrix(train_rows)
        _, y_validation = matrix(validation_rows)
        target_right_rate = args.target_right_rate if args.target_right_rate is not None else float(y_train.mean())

        baseline = evaluate_baseline("logged_dynamic_action", validation_rows, y_validation, None)
        benchmark_rows.append(veto_layer_benchmark_row(lane, "baseline", target_right_rate, baseline))

        trained = train_preference_candidates(
            train_rows,
            y_train,
            validation_rows,
            y_validation,
            target_right_rate=target_right_rate,
            random_state=args.random_state,
        )
        for result, probabilities in trained:
            row = veto_layer_benchmark_row(lane, "retrained_layer", target_right_rate, result)
            benchmark_rows.append(row)
            if best_veto_layer_row(row, best_layers.get(lane), target_right_rate=target_right_rate):
                best_layers[lane] = (row, probabilities, validation_rows, y_validation)

    if not best_layers:
        raise ValueError("No veto decision layers were benchmarked")

    write_rows(Path(args.output), benchmark_rows, VETO_LAYER_BENCHMARK_FIELDS)
    best_rows = veto_layer_best_rows(best_layers)
    write_rows(Path(args.best_output), best_rows, VETO_LAYER_BEST_FIELDS)
    winner = next(row for row in best_rows if row["selection"] == "overall_recommendation")
    _, probabilities, validation_rows, y_validation = best_layers[str(winner["lane"])]
    write_bucket_calibration(
        Path(args.calibration_output),
        validation_rows,
        y_validation,
        probabilities,
        float(winner["probability_threshold"]),
        model_name=f"{winner['lane']} {winner['model']}",
    )
    round3_row = next(row for row in best_rows if row["selection"] == "best_round3")
    round3_rows = veto_layer_rows(manifest_rows, round3_component_rows, lane="Round3")
    save_veto_spline_model(
        Path(args.model_output),
        round3_rows,
        round3_row,
        model_name="multimodalx3_round3_veto_spline",
        random_state=args.random_state,
    )
    multimodalx2_row = next(row for row in best_rows if row["selection"] == "best_multimodalx2")
    multimodalx2_rows = veto_layer_rows(manifest_rows, round3_component_rows, lane="MultimodalX2")
    save_veto_spline_model(
        Path(args.multimodalx4_model_output),
        multimodalx2_rows,
        multimodalx2_row,
        model_name="multimodalx4_multimodalx2_veto_spline",
        random_state=args.random_state,
    )
    round2_row = next(row for row in best_rows if row["selection"] == "best_round2")
    round2_rows = veto_layer_rows(manifest_rows, round2_component_rows, lane="Round2")
    save_veto_spline_model(
        Path(args.round2_model_output),
        round2_rows,
        round2_row,
        model_name="round2_veto_spline",
        random_state=args.random_state,
    )
    original_row = next(
        row
        for row in benchmark_rows
        if row["lane"] == "Original"
        and row["strategy"] == "retrained_layer"
        and row["model"] == "spline_logistic"
    )
    original_rows = veto_layer_rows(manifest_rows, original_component_rows, lane="Original")
    save_veto_spline_model(
        Path(args.original_model_output),
        original_rows,
        original_row,
        model_name="experimental3_original_veto_spline",
        random_state=args.random_state,
    )

    print(f"Labeled veto disagreement screenshot(s): {len(manifest_rows)}")
    print(f"Saved veto decision layer benchmark to {args.output}")
    print(f"Saved best veto decision layers to {args.best_output}")
    print(f"Saved winning layer calibration to {args.calibration_output}")
    print(f"Saved Experimental3 Original spline veto layer to {args.original_model_output}")
    print(f"Saved Round2 veto layer to {args.round2_model_output}")
    print(f"Saved MultimodalX3 Round3 veto layer to {args.model_output}")
    print(f"Saved MultimodalX4 MultimodalX2 veto layer to {args.multimodalx4_model_output}")


def tune_veto_x3(args: argparse.Namespace) -> None:
    if not 0 < args.target_right_rate < 1:
        raise ValueError("--target-right-rate must be between 0 and 1")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0 < args.tune_fraction < 1:
        raise ValueError("--tune-fraction must be between 0 and 1")

    manifest_rows = joined_labeled_rows(Path(args.manifest), Path(args.labels))
    component_rows = read_csv(Path(args.round3_components))
    formulas = veto_x3_formula_grid(args.weight_step)
    if not formulas:
        raise ValueError("No MultimodalX3 formulas to tune")

    sweep_rows = []
    best_by_variant: dict[tuple[int, bool], tuple[VetoX3Formula, dict[str, object]]] = {}
    for index, formula in enumerate(formulas, start=1):
        print(f"[{index}/{len(formulas)}] {formula.config_id}")
        formula_rows = veto_x3_formula_rows(manifest_rows, component_rows, formula=formula)
        formula_outer_train, _ = chronological_split(formula_rows, train_fraction=args.train_fraction)
        train_rows, heldout_rows = chronological_split(formula_outer_train, train_fraction=args.tune_fraction)
        result, _ = fit_veto_x3_spline(
            train_rows,
            heldout_rows,
            formula=formula,
            target_right_rate=args.target_right_rate,
            random_state=args.random_state,
        )
        row = veto_x3_formula_row(formula, result)
        sweep_rows.append(row)
        variant = (formula.knn_k, formula.explicit_old_p_like)
        current = best_by_variant.get(variant)
        if current is None or veto_x3_sort_key(row, target_right_rate=args.target_right_rate) < veto_x3_sort_key(
            current[1],
            target_right_rate=args.target_right_rate,
        ):
            best_by_variant[variant] = (formula, row)

    finalists = [formula for formula, _ in best_by_variant.values()]
    finalists.extend(veto_x3_reference_formulas())
    finalist_rows = []
    seen = set()
    for formula in finalists:
        if formula.config_id in seen:
            continue
        seen.add(formula.config_id)
        formula_rows = veto_x3_formula_rows(manifest_rows, component_rows, formula=formula)
        train_rows, heldout_rows = chronological_split(formula_rows, train_fraction=args.train_fraction)
        result, _ = fit_veto_x3_spline(
            train_rows,
            heldout_rows,
            formula=formula,
            target_right_rate=args.target_right_rate,
            random_state=args.random_state,
        )
        finalist_rows.append(veto_x3_formula_row(formula, result))

    best_final = min(finalist_rows, key=lambda row: veto_x3_sort_key(row, target_right_rate=args.target_right_rate))
    final_rows = []
    for row in sorted(finalist_rows, key=lambda item: veto_x3_sort_key(item, target_right_rate=args.target_right_rate)):
        selected = dict(row)
        selected["selection"] = "overall_recommendation" if row["config_id"] == best_final["config_id"] else "finalist"
        final_rows.append(selected)

    write_rows(Path(args.sweep_output), sweep_rows, VETO_X3_FORMULA_SWEEP_FIELDS)
    write_rows(Path(args.final_output), final_rows, VETO_X3_FINAL_FIELDS)
    best_formula = next(formula for formula in finalists if formula.config_id == best_final["config_id"])
    best_formula_rows = veto_x3_formula_rows(manifest_rows, component_rows, formula=best_formula)
    save_veto_x3_spline_model(
        Path(args.model_output),
        best_formula_rows,
        best_formula,
        best_final,
        model_name="multimodalx5_tuned_veto_spline",
        random_state=args.random_state,
    )
    print(f"Tuned MultimodalX3 formula(s): {len(sweep_rows)}")
    print(f"Held-out finalist(s): {len(final_rows)}")
    print(f"Saved MultimodalX3 tuning sweep to {args.sweep_output}")
    print(f"Saved MultimodalX3 held-out finalists to {args.final_output}")
    print(f"Saved tuned veto spline to {args.model_output}")


def required_bundle(generation: str) -> ModelBundle:
    bundle = next((item for item in available_model_bundles() if item.generation == generation), None)
    if bundle is None or bundle.regressor_path is None or bundle.multimodal_regressor_path is None:
        raise FileNotFoundError(f"Required veto evaluation artifacts are unavailable for {generation}")
    return bundle


def excluded_veto_training_stems(store_paths: Sequence[Path], preference_manifest: Path) -> set[str]:
    stems = set()
    for store_path in store_paths:
        stems.update(reference_store_path_stems(store_path))
    if preference_manifest.exists():
        for row in read_csv(preference_manifest):
            for field in ("screenshot", "source_path", "selected_path"):
                stem = normalized_screenshot_stem(row.get(field, ""))
                if stem:
                    stems.add(stem)
    return stems


def reference_store_path_stems(store_path: Path) -> set[str]:
    if not store_path.exists():
        raise FileNotFoundError(f"Reference store does not exist: {store_path}")
    with np.load(store_path, allow_pickle=False) as data:
        if "paths" not in data.files:
            raise ValueError(f"Reference store is missing paths: {store_path}")
        return {
            stem
            for stem in (normalized_screenshot_stem(str(path)) for path in data["paths"])
            if stem
        }


def eligible_veto_candidates(
    score_rows: list[dict[str, str]],
    *,
    source: Path,
    excluded_stems: set[str],
) -> list[VetoCandidate]:
    candidates = []
    seen_stems = set()
    for row in score_rows:
        screenshot = (row.get("screenshot") or "").strip()
        stem = normalized_screenshot_stem(screenshot)
        image_path = source / screenshot
        if not screenshot or not stem or stem in seen_stems or stem in excluded_stems or not image_path.exists():
            continue
        seen_stems.add(stem)
        candidates.append(VetoCandidate(row=row, image_path=image_path, screenshot_stem=stem))
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.row.get("timestamp", ""),
            natural_sort_key(candidate.row.get("screenshot", "")),
        ),
    )


def normalized_screenshot_stem(value: str) -> str:
    value = str(value or "").strip()
    return "" if not value else Path(value).stem.casefold()


def ensure_veto_component_cache(
    bundle: ModelBundle,
    candidates: list[VetoCandidate],
    *,
    cache_path: Path,
    provider: str,
    knn_values: Sequence[int],
    force: bool,
) -> list[dict[str, str]]:
    cached_rows = [] if force or not cache_path.exists() else read_csv(cache_path)
    cached = {
        veto_row_key(row): row
        for row in cached_rows
        if veto_row_key(row)
    }
    missing = [candidate for candidate in candidates if veto_candidate_key(candidate) not in cached]
    if missing:
        print(f"Scoring {len(missing)} missing {bundle.generation} veto component row(s)")
        store = load_store(bundle.store_path)
        face_regressor = load_regressor(bundle.regressor_path)
        multimodal_regressor = load_regressor(bundle.multimodal_regressor_path)
        for index, candidate in enumerate(missing, start=1):
            row = score_veto_component_row(
                candidate,
                bundle=bundle,
                store=store,
                face_regressor=face_regressor,
                multimodal_regressor=multimodal_regressor,
                provider=provider,
                knn_values=knn_values,
            )
            cached[veto_candidate_key(candidate)] = row
            if index % 25 == 0 or index == len(missing):
                write_ordered_veto_component_cache(cache_path, candidates, cached)
                print(f"  scored {index}/{len(missing)} for {bundle.generation}")
    return write_ordered_veto_component_cache(cache_path, candidates, cached)


def write_ordered_veto_component_cache(
    cache_path: Path,
    candidates: list[VetoCandidate],
    cached: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    ordered = [
        cached[veto_candidate_key(candidate)]
        for candidate in candidates
        if veto_candidate_key(candidate) in cached
    ]
    write_rows(cache_path, ordered, VETO_COMPONENT_FIELDS)
    return ordered


def score_veto_component_row(
    candidate: VetoCandidate,
    *,
    bundle: ModelBundle,
    store,
    face_regressor,
    multimodal_regressor,
    provider: str,
    knn_values: Sequence[int],
) -> dict[str, str]:
    ridge = multimodal = 0.0
    knn_scores = {9: "", 11: ""}
    error = ""
    try:
        embedding = get_face_embedding(
            candidate.image_path,
            model_name=store.model_name,
            provider=provider,
            det_size=store.det_size,
            det_thresh=store.det_thresh,
        )
        ridge = predict_rating(face_regressor, embedding)
        clip_model = multimodal_regressor.metadata.get("clip_model", "openai/clip-vit-base-patch32")
        clip_embedding = get_clip_embedding(candidate.image_path, model_name=clip_model, provider=provider)
        multimodal = predict_multimodal_rating(
            multimodal_regressor,
            face_embedding=embedding,
            clip_embedding=clip_embedding,
        )
        for value in knn_values:
            knn_scores[value] = format_float(score_embedding(embedding, store.embeddings, store.ratings, k=value).rating)
    except Exception as exc:
        error = str(exc)
    return {
        "timestamp": candidate.row.get("timestamp", ""),
        "screenshot": candidate.row.get("screenshot", ""),
        "source_path": candidate.image_path.resolve().as_posix(),
        "generation": bundle.generation,
        "store_path": bundle.store_path.as_posix(),
        "regressor_path": bundle.regressor_path.as_posix() if bundle.regressor_path else "",
        "multimodal_regressor_path": bundle.multimodal_regressor_path.as_posix() if bundle.multimodal_regressor_path else "",
        "ridge": format_float(ridge),
        "multimodal": format_float(multimodal),
        "knn_k9": knn_scores[9],
        "knn_k11": knn_scores[11],
        "error": error,
    }


def veto_candidate_key(candidate: VetoCandidate) -> str:
    return f"{candidate.row.get('timestamp', '')}|{candidate.row.get('screenshot', '')}"


def veto_row_key(row: dict[str, str]) -> str:
    if not row.get("timestamp") or not row.get("screenshot"):
        return ""
    return f"{row['timestamp']}|{row['screenshot']}"


def build_veto_model_rows(
    candidates: list[VetoCandidate],
    *,
    round3_components: list[dict[str, str]],
    round2_components: list[dict[str, str]],
) -> list[dict[str, str]]:
    round3_by_key = {veto_row_key(row): row for row in round3_components}
    round2_by_key = {veto_row_key(row): row for row in round2_components}
    multimodalx_preference = load_preference_model(MULTIMODALX.preference_model)
    experimental1_preference = load_preference_model(EXPERIMENTAL1.preference_model)
    rows = []
    for candidate in candidates:
        key = veto_candidate_key(candidate)
        round3 = round3_by_key.get(key, {})
        round2 = round2_by_key.get(key, {})
        row = {
            "timestamp": candidate.row.get("timestamp", ""),
            "screenshot": candidate.row.get("screenshot", ""),
            "source_path": candidate.image_path.resolve().as_posix(),
            "selected_path": "",
        }
        row.update(round3_veto_lane_values(round3, multimodalx_preference))
        row.update(experimental1_veto_lane_values(round2, experimental1_preference))
        rows.append(row)
    return rows


def round3_veto_lane_values(component: dict[str, str], preference_model) -> dict[str, str]:
    error = component.get("error", "") or ("Missing Round3 component row" if not component else "")
    ridge = component_number(component, "ridge")
    multimodal = component_number(component, "multimodal")
    knn_k9 = component_number(component, "knn_k9")
    knn_k11 = component_number(component, "knn_k11")
    round3_score = biased_multimodal_score(ridge, multimodal, face_weight=0.44) if not error else 0.0
    multimodalx_probability = 0.0
    multimodalx2_probability = 0.0
    if not error:
        multimodalx_probability = component_preference_probability(
            preference_model,
            component,
            ridge=ridge,
            multimodal=multimodal,
            knn=knn_k11,
            face_weight=MULTIMODALX.face_weight,
            threshold=55.0,
        )
        multimodalx2_probability = component_preference_probability(
            preference_model,
            component,
            ridge=ridge,
            multimodal=multimodal,
            knn=knn_k9,
            face_weight=MULTIMODALX2.face_weight,
            threshold=55.0,
        )
    return {
        "round3_score": format_float(round3_score),
        "round3_error": error,
        "multimodalx_score": format_float(
            0.0 if error else multimodalx_score(ridge, multimodal, multimodalx_probability, knn=knn_k11, method=MULTIMODALX_METHOD)
        ),
        "multimodalx_preference_probability": format_float(multimodalx_probability),
        "multimodalx_error": error,
        "multimodalx2_score": format_float(
            0.0 if error else multimodalx_score(ridge, multimodal, multimodalx2_probability, knn=knn_k9, method=MULTIMODALX2_METHOD)
        ),
        "multimodalx2_preference_probability": format_float(multimodalx2_probability),
        "multimodalx2_error": error,
    }


def experimental1_veto_lane_values(component: dict[str, str], preference_model) -> dict[str, str]:
    error = component.get("error", "") or ("Missing Round2 component row" if not component else "")
    ridge = component_number(component, "ridge")
    multimodal = component_number(component, "multimodal")
    knn = component_number(component, "knn_k11")
    score = biased_multimodal_score(ridge, multimodal, face_weight=EXPERIMENTAL1.face_weight) if not error else 0.0
    probability = 0.0 if error else component_preference_probability(
        preference_model,
        component,
        ridge=ridge,
        multimodal=multimodal,
        knn=knn,
        face_weight=EXPERIMENTAL1.face_weight,
        threshold=EXPERIMENTAL1.threshold,
    )
    return {
        "experimental1_score": format_float(score),
        "experimental1_probability": format_float(probability),
        "experimental1_error": error,
    }


def component_preference_probability(
    model,
    component: dict[str, str],
    *,
    ridge: float,
    multimodal: float,
    knn: float,
    face_weight: float,
    threshold: float,
) -> float:
    face_biased = biased_multimodal_score(ridge, multimodal, face_weight=face_weight)
    features = make_features(
        {
            "score": face_biased,
            "face_biased": face_biased,
            "multimodal": multimodal,
            "ridge": ridge,
            "knn": knn,
            "threshold": threshold,
            "face_weight": face_weight,
            "regressor_path": component.get("regressor_path", ""),
            "multimodal_regressor_path": component.get("multimodal_regressor_path", ""),
        }
    )
    return preference_probability(model, features)


def component_number(row: dict[str, str], field: str) -> float:
    return parse_float(row.get(field), 0.0) or 0.0


def apply_veto_dynamic_decisions(
    rows: list[dict[str, str]],
    *,
    dynamic_window: int,
    dynamic_min_history: int,
) -> None:
    apply_veto_lane_dynamic_decisions(
        rows,
        value_field="round3_score",
        threshold_field="round3_threshold",
        action_field="round3_action",
        fixed_threshold=55.0,
        min_threshold=48.0,
        max_threshold=70.0,
        dynamic_window=dynamic_window,
        dynamic_min_history=dynamic_min_history,
    )
    apply_veto_lane_dynamic_decisions(
        rows,
        value_field="multimodalx_score",
        threshold_field="multimodalx_threshold",
        action_field="multimodalx_action",
        fixed_threshold=MULTIMODALX.threshold,
        min_threshold=MULTIMODALX.dynamic_min_threshold,
        max_threshold=MULTIMODALX.dynamic_max_threshold,
        dynamic_window=dynamic_window,
        dynamic_min_history=dynamic_min_history,
    )
    apply_veto_lane_dynamic_decisions(
        rows,
        value_field="multimodalx2_score",
        threshold_field="multimodalx2_threshold",
        action_field="multimodalx2_action",
        fixed_threshold=MULTIMODALX2.threshold,
        min_threshold=MULTIMODALX2.dynamic_min_threshold,
        max_threshold=MULTIMODALX2.dynamic_max_threshold,
        dynamic_window=dynamic_window,
        dynamic_min_history=dynamic_min_history,
    )
    apply_veto_lane_dynamic_decisions(
        rows,
        value_field="experimental1_probability",
        threshold_field="experimental1_threshold",
        action_field="experimental1_action",
        fixed_threshold=EXPERIMENTAL1.preference_threshold,
        min_threshold=EXPERIMENTAL1.dynamic_preference_min_threshold,
        max_threshold=EXPERIMENTAL1.dynamic_preference_max_threshold,
        dynamic_window=dynamic_window,
        dynamic_min_history=dynamic_min_history,
    )


def apply_veto_lane_dynamic_decisions(
    rows: list[dict[str, str]],
    *,
    value_field: str,
    threshold_field: str,
    action_field: str,
    fixed_threshold: float,
    min_threshold: float,
    max_threshold: float,
    dynamic_window: int,
    dynamic_min_history: int,
) -> None:
    values = [component_number(row, value_field) for row in rows]
    decisions = simulate_dynamic_decisions(
        values,
        fixed_threshold=fixed_threshold,
        dynamic_window=dynamic_window,
        dynamic_min_history=dynamic_min_history,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    for row, (threshold, action) in zip(rows, decisions):
        row[threshold_field] = format_float(threshold)
        row[action_field] = action


def simulate_dynamic_decisions(
    values: Sequence[float],
    *,
    fixed_threshold: float,
    dynamic_window: int,
    dynamic_min_history: int,
    min_threshold: float,
    max_threshold: float,
) -> list[tuple[float, str]]:
    history = []
    decisions = []
    for value in values:
        threshold = threshold_from_scores(
            history[-dynamic_window:],
            fixed_threshold=fixed_threshold,
            target_right_rate=0.20,
            min_history=dynamic_min_history,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        decisions.append((threshold, "right" if value >= threshold else "left"))
        history.append(value)
    return decisions


def disagreement_veto_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    disagreements = []
    for row in rows:
        pattern = veto_disagreement_pattern(row)
        if len(set(pattern)) == 1:
            continue
        selected = dict(row)
        selected["disagreement_pattern"] = pattern
        disagreements.append(selected)
    return disagreements


def veto_disagreement_pattern(row: dict[str, str]) -> str:
    return "".join(
        "R" if str(row.get(f"{prefix}_action", "")).lower() == "right" else "L"
        for _, prefix in VETO_LANES
    )


def veto_preparation_stats(
    candidates: list[VetoCandidate],
    model_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> dict[str, object]:
    patterns = [veto_disagreement_pattern(row) for row in model_rows]
    return {
        "eligible_unseen_count": len(candidates),
        "scored_count": len(model_rows),
        "round3_scoring_failure_count": count_veto_errors(model_rows, "round3_error"),
        "multimodalx_scoring_failure_count": count_veto_errors(model_rows, "multimodalx_error"),
        "multimodalx2_scoring_failure_count": count_veto_errors(model_rows, "multimodalx2_error"),
        "experimental1_scoring_failure_count": count_veto_errors(model_rows, "experimental1_error"),
        "unanimous_left_excluded_count": sum(1 for pattern in patterns if pattern == "LLLL"),
        "unanimous_right_excluded_count": sum(1 for pattern in patterns if pattern == "RRRR"),
        "disagreement_selected_count": len(selected_rows),
    }


def count_veto_errors(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if row.get(field, ""))


def build_veto_report_rows(
    manifest_rows: list[dict[str, str]],
    *,
    labels,
    stats: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels_by_path = {normalize_path(label.path): label for label in labels}
    detail_rows = []
    for row in manifest_rows:
        label = labels_by_path.get(normalize_path(row.get("selected_path", "")))
        if label is None:
            continue
        veto_action = "right" if label.rating_1_5 >= 4 else "left"
        detail = {
            "timestamp": row.get("timestamp", ""),
            "screenshot": row.get("screenshot", ""),
            "selected_path": row.get("selected_path", ""),
            "disagreement_pattern": row.get("disagreement_pattern", veto_disagreement_pattern(row)),
            "veto_action": veto_action,
        }
        for _, prefix in VETO_LANES:
            action = row.get(f"{prefix}_action", "")
            detail[f"{prefix}_action"] = action
            detail[f"{prefix}_agrees"] = str(action == veto_action)
        detail_rows.append(detail)

    summary_rows = []
    for name, prefix in VETO_LANES:
        actions = [str(row.get(f"{prefix}_action", "")).lower() for row in detail_rows]
        veto_actions = [str(row["veto_action"]).lower() for row in detail_rows]
        agreements = [action == veto for action, veto in zip(actions, veto_actions)]
        false_positive = sum(1 for action, veto in zip(actions, veto_actions) if action == "right" and veto == "left")
        false_negative = sum(1 for action, veto in zip(actions, veto_actions) if action == "left" and veto == "right")
        row = {
            "model": name,
            "labeled_disagreement_count": len(detail_rows),
            "agreement_count": sum(agreements),
            "agreement_rate": format_float(sum(agreements) / max(1, len(detail_rows))),
            "false_positive": false_positive,
            "false_negative": false_negative,
            "model_right_swipe_rate": format_float(sum(action == "right" for action in actions) / max(1, len(actions))),
            "veto_right_swipe_rate": format_float(sum(action == "right" for action in veto_actions) / max(1, len(veto_actions))),
            "note": VETO_DISAGREEMENT_NOTE,
        }
        row.update({field: stats.get(field, "") for field in VETO_PREPARATION_STATS_FIELDS})
        summary_rows.append(row)
    return detail_rows, summary_rows


def veto_layer_rows(
    manifest_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    *,
    lane: str,
) -> list[dict[str, str]]:
    component_by_key = {veto_row_key(row): row for row in component_rows}
    if lane == "Round2":
        score_field = "experimental1_score"
        threshold_field = ""
        action_field = "experimental1_action"
        face_weight = EXPERIMENTAL1.face_weight
        knn_field = "knn_k11"
    elif lane == "Round3":
        score_field = "round3_score"
        threshold_field = "round3_threshold"
        action_field = "round3_action"
        face_weight = 0.44
        knn_field = "knn_k11"
    elif lane == "MultimodalX2":
        score_field = "multimodalx2_score"
        threshold_field = "multimodalx2_threshold"
        action_field = "multimodalx2_action"
        face_weight = MULTIMODALX2.face_weight
        knn_field = "knn_k9"
    elif lane == "Original":
        score_field = ""
        threshold_field = ""
        action_field = ""
        face_weight = 0.50
        knn_field = "knn_k11"
    else:
        raise ValueError(f"Unknown veto layer lane: {lane}")

    rows = []
    for row in manifest_rows:
        component = component_by_key.get(veto_row_key(row))
        if component is None:
            continue
        ridge = component_number(component, "ridge")
        multimodal = component_number(component, "multimodal")
        knn = component_number(component, knn_field)
        face_biased = biased_multimodal_score(ridge, multimodal, face_weight=face_weight)
        score = face_biased if lane == "Original" else component_number(row, score_field)
        threshold = (
            55.0
            if lane == "Original"
            else EXPERIMENTAL1.threshold
            if lane == "Round2"
            else component_number(row, threshold_field)
        )
        action = ("right" if score >= threshold else "left") if lane == "Original" else row.get(action_field, "")
        features = make_features(
            {
                "score": score,
                "face_biased": face_biased,
                "multimodal": multimodal,
                "ridge": ridge,
                "knn": knn,
                "threshold": threshold,
                "face_weight": face_weight,
                "regressor_path": component.get("regressor_path", ""),
                "multimodal_regressor_path": component.get("multimodal_regressor_path", ""),
            }
        )
        rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "screenshot": row.get("screenshot", ""),
                "selected_path": row.get("selected_path", ""),
                "action": action,
                "like": row["like"],
                "score": format_float(score),
                "face_biased": format_float(face_biased),
                "multimodal": format_float(multimodal),
                "ridge": format_float(ridge),
                "knn": format_float(knn),
                "threshold": format_float(threshold),
                "face_weight": format_float(face_weight),
                "regressor_path": component.get("regressor_path", ""),
                "multimodal_regressor_path": component.get("multimodal_regressor_path", ""),
                "component_spread": format_float(features["component_spread"]),
                "ridge_minus_multimodal": format_float(features["ridge_minus_multimodal"]),
                "distance_from_threshold": format_float(features["distance_from_threshold"]),
                "score_bucket": format_float(features["score_bucket"]),
            }
        )
    if not rows:
        raise ValueError(f"No veto rows can be built for {lane}")
    return rows


def veto_x3_formula_grid(step: float) -> list[VetoX3Formula]:
    if not 0 < step <= 1:
        raise ValueError("--weight-step must be between 0 and 1")
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--weight-step must divide 1.0 exactly")

    formulas = []
    for ridge_units in range(units + 1):
        for multimodal_units in range(units - ridge_units + 1):
            for knn_units in range(units - ridge_units - multimodal_units + 1):
                old_p_like_units = units - ridge_units - multimodal_units - knn_units
                weights = (
                    round(ridge_units * step, 6),
                    round(multimodal_units * step, 6),
                    round(knn_units * step, 6),
                    round(old_p_like_units * step, 6),
                )
                for knn_k in (9, 11):
                    for explicit_old_p_like in (False, True):
                        formulas.append(VetoX3Formula(*weights, knn_k=knn_k, explicit_old_p_like=explicit_old_p_like))
    unique = {formula.config_id: formula for formula in [*formulas, *veto_x3_reference_formulas()]}
    return [unique[key] for key in sorted(unique)]


def veto_x3_reference_formulas() -> list[VetoX3Formula]:
    references = [
        (0.44, 0.56, 0.0, 0.0),
        (0.73, 0.20, 0.0, 0.07),
        (0.12, 0.05, 0.30, 0.53),
    ]
    return [
        VetoX3Formula(*weights, knn_k=knn_k, explicit_old_p_like=explicit_old_p_like)
        for weights in references
        for knn_k in (9, 11)
        for explicit_old_p_like in (False, True)
    ]


def veto_x3_formula_rows(
    manifest_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    *,
    formula: VetoX3Formula,
) -> list[dict[str, str]]:
    component_by_key = {veto_row_key(row): row for row in component_rows}
    rows = []
    for row in sorted(manifest_rows, key=lambda item: item.get("timestamp", "")):
        component = component_by_key.get(veto_row_key(row))
        if component is None:
            continue
        ridge = component_number(component, "ridge")
        multimodal = component_number(component, "multimodal")
        knn = component_number(component, f"knn_k{formula.knn_k}")
        old_p_like = component_number(
            row,
            "multimodalx2_preference_probability" if formula.knn_k == 9 else "multimodalx_preference_probability",
        )
        score = (
            formula.ridge_weight * ridge
            + formula.multimodal_weight * multimodal
            + formula.knn_weight * knn
            + formula.old_p_like_weight * old_p_like * 100.0
        )
        ridge_multimodal_weight = formula.ridge_weight + formula.multimodal_weight
        face_weight = formula.ridge_weight / ridge_multimodal_weight if ridge_multimodal_weight else 0.44
        face_biased = biased_multimodal_score(ridge, multimodal, face_weight=face_weight)
        features = make_features(
            {
                "score": score,
                "face_biased": face_biased,
                "multimodal": multimodal,
                "ridge": ridge,
                "knn": knn,
                "threshold": 55.0,
                "face_weight": face_weight,
                "regressor_path": component.get("regressor_path", ""),
                "multimodal_regressor_path": component.get("multimodal_regressor_path", ""),
            }
        )
        rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "screenshot": row.get("screenshot", ""),
                "selected_path": row.get("selected_path", ""),
                "like": row["like"],
                "score": format_float(score),
                "face_biased": format_float(face_biased),
                "multimodal": format_float(multimodal),
                "ridge": format_float(ridge),
                "knn": format_float(knn),
                "threshold": "55.000000",
                "face_weight": format_float(face_weight),
                VETO_X3_OLD_P_LIKE_FIELD: format_float(old_p_like),
                "regressor_path": component.get("regressor_path", ""),
                "multimodal_regressor_path": component.get("multimodal_regressor_path", ""),
                "component_spread": format_float(features["component_spread"]),
                "ridge_minus_multimodal": format_float(features["ridge_minus_multimodal"]),
                "distance_from_threshold": format_float(features["distance_from_threshold"]),
                "score_bucket": format_float(features["score_bucket"]),
            }
        )
    if not rows:
        raise ValueError(f"No MultimodalX3 formula rows can be built for {formula.config_id}")
    thresholds = simulate_dynamic_decisions(
        [component_number(row, "score") for row in rows],
        fixed_threshold=55.0,
        dynamic_window=200,
        dynamic_min_history=50,
        min_threshold=48.0,
        max_threshold=70.0,
    )
    for row, (threshold, _) in zip(rows, thresholds):
        row["threshold"] = format_float(threshold)
        features = make_features(row)
        row["distance_from_threshold"] = format_float(features["distance_from_threshold"])
        row["score_bucket"] = format_float(features["score_bucket"])
    return rows


def available_model_bundles(root: str | Path = ".") -> list[ModelBundle]:
    root = Path(root)
    candidates = [
        ModelBundle("original", root / "embeddings/reference_store.npz", root / "models/rating_regressor.joblib", root / "models/rating_regressor_multimodal.joblib", 0.44, 3),
        ModelBundle("normalized", root / "embeddings/reference_store_normalized.npz", root / "models/rating_regressor_normalized.joblib", None, 0.44, 4),
        ModelBundle("bumble_combined", root / "embeddings/reference_store_bumble_combined.npz", root / "models/rating_regressor_bumble_combined.joblib", root / "models/rating_regressor_multimodal_bumble_combined.joblib", 0.22, 2),
        ModelBundle("bumble_combined_round2", root / "embeddings/reference_store_bumble_combined_round2.npz", root / "models/rating_regressor_bumble_combined_round2.joblib", root / "models/rating_regressor_multimodal_bumble_combined_round2.joblib", 0.22, 1),
        ModelBundle("bumble_combined_round3", root / "embeddings/reference_store_bumble_combined_round3.npz", root / "models/rating_regressor_bumble_combined_round3.joblib", root / "models/rating_regressor_multimodal_bumble_combined_round3.joblib", 0.44, 0),
        ModelBundle("bumble_only", root / "embeddings/reference_store_bumble_only.npz", root / "models/rating_regressor_bumble_only.joblib", root / "models/rating_regressor_multimodal_bumble_only.joblib", 0.22, 6),
        ModelBundle("bumble_only_round2", root / "embeddings/reference_store_bumble_only_round2.npz", root / "models/rating_regressor_bumble_only_round2.joblib", root / "models/rating_regressor_multimodal_bumble_only_round2.joblib", 0.22, 5),
    ]
    bundles = []
    for bundle in candidates:
        if not bundle.store_path.exists():
            continue
        regressor_path = (
            bundle.regressor_path
            if bundle.regressor_path and bundle.regressor_path.exists() and can_load_regressor(bundle.regressor_path)
            else None
        )
        multimodal_path = (
            bundle.multimodal_regressor_path
            if bundle.multimodal_regressor_path
            and bundle.multimodal_regressor_path.exists()
            and can_load_regressor(bundle.multimodal_regressor_path)
            else None
        )
        bundles.append(
            ModelBundle(
                generation=bundle.generation,
                store_path=bundle.store_path,
                regressor_path=regressor_path,
                multimodal_regressor_path=multimodal_path,
                default_face_weight=bundle.default_face_weight,
                deploy_rank=bundle.deploy_rank,
            )
        )
    return bundles


def can_load_regressor(path: Path) -> bool:
    try:
        load_regressor(path)
    except Exception:
        return False
    return True


def model_grid(root: str | Path = ".") -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for bundle in available_model_bundles(root):
        configs.append(ModelConfig(bundle, "knn", bundle.default_face_weight))
        if bundle.regressor_path is not None:
            configs.append(ModelConfig(bundle, "ridge", bundle.default_face_weight))
        if bundle.multimodal_regressor_path is not None:
            configs.append(ModelConfig(bundle, "multimodal", 0.0))
        if bundle.regressor_path is not None and bundle.multimodal_regressor_path is not None:
            configs.extend(ModelConfig(bundle, "face_biased", weight) for weight in FACE_WEIGHT_SWEEP)
    return configs


def ensure_component_score_cache(
    bundle: ModelBundle,
    rows: list[dict[str, str]],
    *,
    cache_dir: Path,
    provider: str,
    k: int,
    force: bool,
) -> list[dict[str, str]]:
    cache_path = cache_dir / f"{bundle.generation}__components.csv"
    if cache_path.exists() and not force:
        cached = read_csv(cache_path)
        if len(cached) == len(rows):
            return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    store = load_store(bundle.store_path)
    face_regressor = load_regressor(bundle.regressor_path) if bundle.regressor_path is not None else None
    multimodal_regressor = load_regressor(bundle.multimodal_regressor_path) if bundle.multimodal_regressor_path is not None else None
    output_rows = []
    for index, row in enumerate(rows, start=1):
        if index % 100 == 0:
            print(f"  scored {index}/{len(rows)} for {bundle.generation}")
        image_path = Path(row["selected_path"])
        error = ""
        ridge = multimodal = knn = 0.0
        try:
            prediction = predict_image_rating(
                image_path,
                store=store,
                method="knn",
                k=k,
                provider=provider,
                face_regressor=face_regressor,
                multimodal_regressor=multimodal_regressor,
                face_weight=bundle.default_face_weight,
                include_components=True,
            )
            ridge = prediction.face_rating if prediction.face_rating is not None else 0.0
            multimodal = prediction.multimodal_rating if prediction.multimodal_rating is not None else 0.0
            knn = prediction.knn_rating
        except Exception as exc:
            error = str(exc)
        output_rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "selected_path": row.get("selected_path", ""),
                "source_path": row.get("source_path", ""),
                "generation": bundle.generation,
                "store_path": bundle.store_path.as_posix(),
                "regressor_path": bundle.regressor_path.as_posix() if bundle.regressor_path else "",
                "multimodal_regressor_path": bundle.multimodal_regressor_path.as_posix() if bundle.multimodal_regressor_path else "",
                "ridge": format_float(float(ridge)),
                "multimodal": format_float(float(multimodal)),
                "knn": format_float(float(knn)),
                "error": error,
            }
        )
    write_rows(cache_path, output_rows, COMPONENT_SCORE_FIELDS)
    return output_rows


def ensure_model_score_cache(
    config: ModelConfig,
    component_rows: list[dict[str, str]],
    *,
    cache_dir: Path,
    threshold: float,
    force: bool,
) -> list[dict[str, str]]:
    cache_path = cache_dir / f"{config.config_id}.csv"
    if cache_path.exists() and not force:
        cached = read_csv(cache_path)
        if len(cached) == len(component_rows):
            return cached
    rows = derive_config_score_rows(config, component_rows, threshold=threshold)
    write_rows(cache_path, rows, MODEL_SCORE_FIELDS)
    return rows


def derive_config_score_rows(
    config: ModelConfig,
    component_rows: list[dict[str, str]],
    *,
    threshold: float,
) -> list[dict[str, str]]:
    rows = []
    for row in component_rows:
        ridge = parse_float(row.get("ridge"), 0.0) or 0.0
        multimodal = parse_float(row.get("multimodal"), 0.0) or 0.0
        knn = parse_float(row.get("knn"), 0.0) or 0.0
        face_biased = biased_multimodal_score(ridge, multimodal, face_weight=config.face_weight)
        if config.method == "knn":
            score = knn
        elif config.method == "ridge":
            score = ridge
        elif config.method == "multimodal":
            score = multimodal
        elif config.method == "face_biased":
            score = face_biased
        else:
            raise ValueError(f"Unknown benchmark method: {config.method}")
        feature_source = {
            "score": score,
            "face_biased": face_biased,
            "multimodal": multimodal,
            "ridge": ridge,
            "knn": knn,
            "threshold": threshold,
            "face_weight": config.face_weight,
            "regressor_path": row.get("regressor_path", ""),
            "multimodal_regressor_path": row.get("multimodal_regressor_path", ""),
        }
        features = make_features(feature_source)
        rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "selected_path": row.get("selected_path", ""),
                "generation": config.bundle.generation,
                "method": config.method,
                "face_weight": format_float(config.face_weight),
                "score": format_float(score),
                "face_biased": format_float(face_biased),
                "multimodal": format_float(multimodal),
                "ridge": format_float(ridge),
                "knn": format_float(knn),
                "threshold": format_float(threshold),
                "regressor_path": row.get("regressor_path", ""),
                "multimodal_regressor_path": row.get("multimodal_regressor_path", ""),
                "component_spread": format_float(features["component_spread"]),
                "ridge_minus_multimodal": format_float(features["ridge_minus_multimodal"]),
                "distance_from_threshold": format_float(features["distance_from_threshold"]),
                "score_bucket": format_float(features["score_bucket"]),
            }
        )
    return rows


def chronological_split(rows: list[dict[str, str]], *, train_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sorted_rows = sorted(rows, key=lambda row: row.get("timestamp", ""))
    split_at = max(1, min(len(sorted_rows) - 1, int(len(sorted_rows) * train_fraction)))
    return sorted_rows[:split_at], sorted_rows[split_at:]


def feature_matrix(rows: list[dict[str, str]], fields: list[str] | None = None) -> np.ndarray:
    return np.asarray([feature_vector(row_features(row), fields=fields) for row in rows], dtype=np.float32)


def train_config_preference(
    train_rows: list[dict[str, str]],
    y_train: np.ndarray,
    validation_rows: list[dict[str, str]],
    y_validation: np.ndarray,
    *,
    target_right_rate: float,
    random_state: int,
) -> tuple[EvalResult, str, np.ndarray]:
    x_train = feature_matrix(train_rows)
    x_validation = feature_matrix(validation_rows)
    estimators = {
        "bucket_like_rate": BucketLikeRateClassifier(score_index=FEATURE_FIELDS.index("score")),
        "spline_logistic": spline_logistic(random_state=random_state),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, random_state=random_state),
        ),
    }
    results = []
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        threshold = probability_threshold_for_rate(probabilities, target_right_rate)
        result = evaluate_probabilities(name, y_validation, probabilities, threshold)
        results.append((result, name, probabilities))
    return min(
        results,
        key=lambda item: (
            item[0].swipe_errors,
            abs(item[0].right_swipe_rate - target_right_rate),
            item[0].false_positive,
            -item[0].precision,
            -safe_auc(item[0].auc),
        ),
    )


def train_preference_candidates(
    train_rows: list[dict[str, str]],
    y_train: np.ndarray,
    validation_rows: list[dict[str, str]],
    y_validation: np.ndarray,
    *,
    target_right_rate: float,
    random_state: int,
) -> list[tuple[EvalResult, np.ndarray]]:
    x_train = feature_matrix(train_rows)
    x_validation = feature_matrix(validation_rows)
    results = []
    for name in ("bucket_like_rate", "spline_logistic", "gradient_boosting"):
        estimator = veto_candidate_estimator(name, random_state=random_state)
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        threshold = probability_threshold_for_rate(probabilities, target_right_rate)
        results.append((evaluate_probabilities(name, y_validation, probabilities, threshold), probabilities))
    return results


def veto_candidate_estimator(model: str, *, random_state: int):
    if model == "bucket_like_rate":
        return BucketLikeRateClassifier(score_index=FEATURE_FIELDS.index("score"))
    if model == "spline_logistic":
        return spline_logistic(random_state=random_state)
    if model == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, random_state=random_state),
        )
    raise ValueError(f"Unknown veto candidate model: {model}")


def fit_veto_x3_spline(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    *,
    formula: VetoX3Formula,
    target_right_rate: float,
    random_state: int,
) -> tuple[EvalResult, np.ndarray]:
    fields = veto_x3_feature_fields(formula)
    x_train = feature_matrix(train_rows, fields=fields)
    y_train = np.asarray([int(row["like"]) for row in train_rows], dtype=np.int32)
    x_validation = feature_matrix(validation_rows, fields=fields)
    y_validation = np.asarray([int(row["like"]) for row in validation_rows], dtype=np.int32)
    estimator = spline_logistic(random_state=random_state, fields=fields)
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_validation)[:, 1]
    threshold = probability_threshold_for_rate(probabilities, target_right_rate)
    return evaluate_probabilities("spline_logistic", y_validation, probabilities, threshold), probabilities


def save_veto_spline_model(
    output: Path,
    rows: list[dict[str, str]],
    validation_row: dict[str, object],
    *,
    model_name: str,
    random_state: int,
) -> None:
    estimator = spline_logistic(random_state=random_state)
    x, y = matrix(rows)
    estimator.fit(x, y)
    save_preference_model(
        output,
        estimator=estimator,
        model_name=model_name,
        threshold=parse_float(validation_row.get("probability_threshold"), 0.493593) or 0.493593,
        metrics={
            "swipe_error_rate": parse_float(validation_row.get("swipe_error_rate"), 0.0) or 0.0,
            "right_swipe_rate": parse_float(validation_row.get("right_swipe_rate"), 0.0) or 0.0,
            "precision": parse_float(validation_row.get("precision"), 0.0) or 0.0,
            "recall": parse_float(validation_row.get("recall"), 0.0) or 0.0,
            "auc": parse_float(validation_row.get("auc"), 0.0) or 0.0,
            "log_loss": parse_float(validation_row.get("log_loss"), 0.0) or 0.0,
            "validation_count": parse_float(validation_row.get("count"), 0.0) or 0.0,
            "training_count": float(len(rows)),
        },
    )


def save_veto_candidate_model(
    output: Path,
    rows: list[dict[str, str]],
    validation_row: dict[str, object],
    *,
    model_name: str,
    random_state: int,
) -> None:
    estimator = veto_candidate_estimator(str(validation_row.get("model", "")), random_state=random_state)
    x, y = matrix(rows)
    estimator.fit(x, y)
    save_preference_model(
        output,
        estimator=estimator,
        model_name=model_name,
        threshold=parse_float(validation_row.get("probability_threshold"), 0.493593) or 0.493593,
        metrics={
            "swipe_error_rate": parse_float(validation_row.get("swipe_error_rate"), 0.0) or 0.0,
            "right_swipe_rate": parse_float(validation_row.get("right_swipe_rate"), 0.0) or 0.0,
            "precision": parse_float(validation_row.get("precision"), 0.0) or 0.0,
            "recall": parse_float(validation_row.get("recall"), 0.0) or 0.0,
            "auc": parse_float(validation_row.get("auc"), 0.0) or 0.0,
            "log_loss": parse_float(validation_row.get("log_loss"), 0.0) or 0.0,
            "validation_count": parse_float(validation_row.get("count"), 0.0) or 0.0,
            "training_count": float(len(rows)),
        },
    )


def save_veto_x3_spline_model(
    output: Path,
    rows: list[dict[str, str]],
    formula: VetoX3Formula,
    validation_row: dict[str, object],
    *,
    model_name: str,
    random_state: int,
) -> None:
    fields = veto_x3_feature_fields(formula)
    estimator = spline_logistic(random_state=random_state, fields=fields)
    x = feature_matrix(rows, fields=fields)
    y = np.asarray([int(row["like"]) for row in rows], dtype=np.int32)
    estimator.fit(x, y)
    save_preference_model(
        output,
        estimator=estimator,
        model_name=model_name,
        threshold=parse_float(validation_row.get("probability_threshold"), 0.489530) or 0.489530,
        feature_fields=fields,
        metrics={
            "swipe_error_rate": parse_float(validation_row.get("swipe_error_rate"), 0.0) or 0.0,
            "right_swipe_rate": parse_float(validation_row.get("right_swipe_rate"), 0.0) or 0.0,
            "precision": parse_float(validation_row.get("precision"), 0.0) or 0.0,
            "recall": parse_float(validation_row.get("recall"), 0.0) or 0.0,
            "auc": parse_float(validation_row.get("auc"), 0.0) or 0.0,
            "log_loss": parse_float(validation_row.get("log_loss"), 0.0) or 0.0,
            "validation_count": parse_float(validation_row.get("count"), 0.0) or 0.0,
            "training_count": float(len(rows)),
        },
    )


def veto_x3_feature_fields(formula: VetoX3Formula) -> list[str]:
    if not formula.explicit_old_p_like:
        return list(FEATURE_FIELDS)
    return [*FEATURE_FIELDS, VETO_X3_OLD_P_LIKE_FIELD]


def veto_x3_formula_row(formula: VetoX3Formula, result: EvalResult) -> dict[str, object]:
    row = {
        "config_id": formula.config_id,
        "knn_k": formula.knn_k,
        "explicit_old_p_like": str(formula.explicit_old_p_like),
        "ridge_weight": format_float(formula.ridge_weight),
        "multimodal_weight": format_float(formula.multimodal_weight),
        "knn_weight": format_float(formula.knn_weight),
        "old_p_like_weight": format_float(formula.old_p_like_weight),
    }
    row.update(report_row(result))
    return row


def veto_x3_sort_key(row: dict[str, object], *, target_right_rate: float) -> tuple[float, float, float, float, float]:
    return (
        float(row["swipe_errors"]),
        abs((parse_float(row["right_swipe_rate"], 0.0) or 0.0) - target_right_rate),
        float(row["false_positive"]),
        -(parse_float(row["precision"], 0.0) or 0.0),
        -safe_auc(parse_float(row["auc"], 0.0) or 0.0),
    )


def veto_layer_benchmark_row(
    lane: str,
    strategy: str,
    target_right_rate: float,
    result: EvalResult,
) -> dict[str, object]:
    row = {
        "lane": lane,
        "strategy": strategy,
        "target_right_rate": format_float(target_right_rate),
    }
    row.update(report_row(result))
    return row


def veto_layer_best_rows(
    best_layers: dict[str, tuple[dict[str, object], np.ndarray, list[dict[str, str]], np.ndarray]]
) -> list[dict[str, object]]:
    rows = []
    for lane in sorted(best_layers):
        row = dict(best_layers[lane][0])
        row["selection"] = f"best_{lane.lower()}"
        rows.append(row)
    recommended = dict(
        min(
            (item[0] for item in best_layers.values()),
            key=lambda row: veto_layer_sort_key(row, target_right_rate=parse_float(row["target_right_rate"], 0.0) or 0.0),
        )
    )
    recommended["selection"] = "overall_recommendation"
    rows.append(recommended)
    return rows


def best_veto_layer_row(
    row: dict[str, object],
    current: tuple[dict[str, object], np.ndarray, list[dict[str, str]], np.ndarray] | None,
    *,
    target_right_rate: float,
) -> bool:
    return current is None or veto_layer_sort_key(row, target_right_rate=target_right_rate) < veto_layer_sort_key(
        current[0],
        target_right_rate=target_right_rate,
    )


def veto_layer_sort_key(row: dict[str, object], *, target_right_rate: float) -> tuple[float, float, float, float, float]:
    return (
        float(row["swipe_errors"]),
        abs((parse_float(row["right_swipe_rate"], 0.0) or 0.0) - target_right_rate),
        float(row["false_positive"]),
        -(parse_float(row["precision"], 0.0) or 0.0),
        -safe_auc(parse_float(row["auc"], 0.0) or 0.0),
    )


def benchmark_row(
    config: ModelConfig,
    *,
    strategy: str,
    preference_model: str,
    result: EvalResult,
    score_cache: Path,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "generation": config.bundle.generation,
        "method": config.method,
        "face_weight": format_float(config.face_weight),
        "config_id": config.config_id,
        "preference_model": preference_model,
        "probability_threshold": format_float(result.probability_threshold),
        "count": result.count,
        "swipe_errors": result.swipe_errors,
        "swipe_error_rate": format_float(result.swipe_error_rate),
        "false_positive": result.false_positive,
        "false_negative": result.false_negative,
        "right_swipe_rate": format_float(result.right_swipe_rate),
        "precision": format_float(result.precision),
        "recall": format_float(result.recall),
        "accuracy": format_float(result.accuracy),
        "auc": format_float(result.auc),
        "log_loss": format_float(result.log_loss),
        "store_path": config.bundle.store_path.as_posix(),
        "regressor_path": config.bundle.regressor_path.as_posix() if config.bundle.regressor_path else "",
        "multimodal_regressor_path": config.bundle.multimodal_regressor_path.as_posix() if config.bundle.multimodal_regressor_path else "",
        "score_cache": score_cache.as_posix(),
    }


def update_best(
    best_by_strategy: dict[str, tuple[dict[str, object], np.ndarray, list[dict[str, str]]]],
    row: dict[str, object],
    probabilities: np.ndarray,
    validation_rows: list[dict[str, str]],
    *,
    target_right_rate: float,
) -> None:
    strategy = str(row["strategy"])
    current = best_by_strategy.get(strategy)
    if current is None or best_sort_key(row, target_right_rate=target_right_rate) < best_sort_key(current[0], target_right_rate=target_right_rate):
        best_by_strategy[strategy] = (row, probabilities, validation_rows)


def best_report_rows(
    best_by_strategy: dict[str, tuple[dict[str, object], np.ndarray, list[dict[str, str]]]]
) -> list[dict[str, object]]:
    rows = []
    for strategy in sorted(best_by_strategy):
        row = dict(best_by_strategy[strategy][0])
        row["selection"] = f"best_{strategy}"
        rows.append(row)
    if "retrained_per_config" in best_by_strategy:
        recommended = dict(best_by_strategy["retrained_per_config"][0])
    else:
        recommended = dict(min((item[0] for item in best_by_strategy.values()), key=lambda row: best_sort_key(row, target_right_rate=0.20)))
    recommended["selection"] = "overall_recommendation"
    rows.append(recommended)
    return rows


def best_sort_key(row: dict[str, object], *, target_right_rate: float) -> tuple[float, float, float, float, float, int]:
    generation = str(row["generation"])
    return (
        float(row["swipe_errors"]),
        abs(parse_float(row["right_swipe_rate"], 0.0) - target_right_rate),
        float(row["false_positive"]),
        -(parse_float(row["precision"], 0.0) or 0.0),
        -safe_auc(parse_float(row["auc"], 0.0) or 0.0),
        generation_deploy_rank(generation),
    )


def generation_deploy_rank(generation: str) -> int:
    ranks = {bundle.generation: bundle.deploy_rank for bundle in available_model_bundles()}
    return ranks.get(generation, 99)


def safe_auc(value: float) -> float:
    return 0.0 if math.isnan(value) else value


def spline_logistic(*, random_state: int, fields: list[str] | None = None):
    fields = fields or FEATURE_FIELDS
    score_index = fields.index("score")
    numeric_indices = list(range(len(fields)))
    features = ColumnTransformer(
        [
            (
                "score_spline",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    SplineTransformer(n_knots=6, degree=3, include_bias=False),
                ),
                [score_index],
            ),
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_indices,
            ),
        ]
    )
    return make_pipeline(
        features,
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )


def joined_labeled_rows(manifest_path: Path, labels_path: Path) -> list[dict[str, str]]:
    labels = {normalize_path(row.path): row for row in load_label_rows(labels_path)}
    rows = []
    for row in read_csv(manifest_path):
        label = labels.get(normalize_path(row.get("selected_path", "")))
        if label is None:
            continue
        merged = dict(row)
        merged["like"] = "1" if label.rating_1_5 >= 4 else "0"
        rows.append(merged)
    if not rows:
        raise ValueError("No labeled manifest rows found")
    return rows


def matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([feature_vector(row_features(row)) for row in rows], dtype=np.float32)
    y = np.asarray([int(row["like"]) for row in rows], dtype=np.int32)
    return x, y


def row_features(row: dict[str, str]) -> dict[str, float]:
    features = make_features(row)
    features[VETO_X3_OLD_P_LIKE_FIELD] = parse_float(row.get(VETO_X3_OLD_P_LIKE_FIELD), 0.0) or 0.0
    return features


def evaluate_baseline(
    name: str,
    rows: list[dict[str, str]],
    y_true: np.ndarray,
    threshold: float | None,
) -> EvalResult:
    if threshold is None:
        y_pred = np.asarray([1 if str(row.get("action", "")).lower() in {"right", "arrowright"} else 0 for row in rows])
        probability_threshold = 0.5
    else:
        y_pred = np.asarray([1 if (parse_float(row.get("score"), 0.0) or 0.0) >= threshold else 0 for row in rows])
        probability_threshold = threshold
    return metrics_from_predictions(name, y_true, y_pred, y_pred.astype(np.float32), probability_threshold)


def evaluate_dynamic_baseline(
    name: str,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    y_true: np.ndarray,
    *,
    target_right_rate: float,
    window: int,
    min_threshold: float,
    max_threshold: float,
) -> EvalResult:
    history = [
        score
        for score in (parse_float(row.get("score")) for row in train_rows)
        if score is not None
    ]
    y_pred = []
    thresholds = []
    percentile = (1.0 - target_right_rate) * 100.0
    for row in validation_rows:
        recent = history[-window:] if window > 0 else history
        if recent:
            threshold = float(np.percentile(np.asarray(recent, dtype=np.float32), percentile))
            threshold = min(max(threshold, min_threshold), max_threshold)
        else:
            threshold = parse_float(row.get("threshold"), 55.0) or 55.0
        score = parse_float(row.get("score"), 0.0) or 0.0
        y_pred.append(1 if score >= threshold else 0)
        thresholds.append(threshold)
        history.append(score)
    y_pred_array = np.asarray(y_pred, dtype=np.int32)
    probability_threshold = float(np.mean(thresholds)) if thresholds else 0.0
    return metrics_from_predictions(name, y_true, y_pred_array, y_pred_array.astype(np.float32), probability_threshold)


def evaluate_probabilities(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    probability_threshold: float,
) -> EvalResult:
    y_pred = (probabilities >= probability_threshold).astype(np.int32)
    return metrics_from_predictions(name, y_true, y_pred, probabilities, probability_threshold)


def metrics_from_predictions(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    probability_threshold: float,
) -> EvalResult:
    count = int(len(y_true))
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    swipe_errors = false_positive + false_negative
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    try:
        auc = float(roc_auc_score(y_true, probabilities)) if len(set(y_true.tolist())) > 1 else math.nan
    except ValueError:
        auc = math.nan
    try:
        loss = float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except ValueError:
        loss = math.nan
    return EvalResult(
        model=name,
        split="validation",
        count=count,
        probability_threshold=float(probability_threshold),
        swipe_errors=swipe_errors,
        swipe_error_rate=swipe_errors / max(1, count),
        false_positive=false_positive,
        false_negative=false_negative,
        right_swipe_rate=float(y_pred.mean()) if count else 0.0,
        precision=precision,
        recall=recall,
        accuracy=(true_positive + true_negative) / max(1, count),
        auc=auc,
        log_loss=loss,
    )


def probability_threshold_for_rate(probabilities: np.ndarray, target_right_rate: float) -> float:
    if len(probabilities) == 0:
        return 1.0
    right_count = max(1, int(round(len(probabilities) * target_right_rate)))
    descending = np.sort(probabilities)[::-1]
    return float(descending[min(right_count - 1, len(descending) - 1)])


def write_report(path: Path, results: Iterable[EvalResult]) -> None:
    rows = []
    for result in results:
        rows.append(report_row(result))
    write_rows(path, rows, REPORT_FIELDS)


def report_row(result: EvalResult) -> dict[str, object]:
    return {
        "model": result.model,
        "split": result.split,
        "count": result.count,
        "probability_threshold": format_float(result.probability_threshold),
        "swipe_errors": result.swipe_errors,
        "swipe_error_rate": format_float(result.swipe_error_rate),
        "false_positive": result.false_positive,
        "false_negative": result.false_negative,
        "right_swipe_rate": format_float(result.right_swipe_rate),
        "precision": format_float(result.precision),
        "recall": format_float(result.recall),
        "accuracy": format_float(result.accuracy),
        "auc": format_float(result.auc),
        "log_loss": format_float(result.log_loss),
    }


def write_bucket_calibration(
    path: Path,
    validation_rows: list[dict[str, str]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    probability_threshold: float,
    *,
    model_name: str,
) -> None:
    y_pred = (probabilities >= probability_threshold).astype(np.int32)
    rows = []
    buckets = sorted({row_score_bucket(row) for row in validation_rows})
    for bucket in buckets:
        indices = [
            index
            for index, row in enumerate(validation_rows)
            if row_score_bucket(row) == bucket
        ]
        if not indices:
            continue
        actual = y_true[indices]
        predicted = probabilities[indices]
        decided = y_pred[indices]
        true_positive = int(((decided == 1) & (actual == 1)).sum())
        false_positive = int(((decided == 1) & (actual == 0)).sum())
        false_negative = int(((decided == 0) & (actual == 1)).sum())
        rows.append(
            {
                "model": model_name,
                "score_bucket": bucket_label(bucket),
                "count": len(indices),
                "actual_like_rate": format_float(float(actual.mean())),
                "predicted_like_rate": format_float(float(predicted.mean())),
                "right_swipe_rate": format_float(float(decided.mean())),
                "precision": format_float(true_positive / max(1, true_positive + false_positive)),
                "recall": format_float(true_positive / max(1, true_positive + false_negative)),
            }
        )
    write_rows(path, rows, CALIBRATION_FIELDS)


def row_score_bucket(row: dict[str, str]) -> int:
    value = parse_float(row.get("score_bucket"))
    return -1 if value is None else int(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_path(path: str) -> str:
    return Path(path).resolve().as_posix().casefold()


def bucket_reason(score: float) -> str:
    for lower, upper, _ in BUCKET_QUOTAS:
        if lower <= score < upper:
            return f"score_{lower:g}_{upper:g}"
    return "score_other"


def bucket_label(bucket: int) -> str:
    if 0 <= bucket < len(BUCKET_QUOTAS):
        lower, upper, _ = BUCKET_QUOTAS[bucket]
        return f"{lower:g}-{min(upper, 100):g}"
    return "unknown"


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
