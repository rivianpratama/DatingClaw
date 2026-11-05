from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    fbeta_score,
    precision_score,
    recall_score,
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    base_rate: float = 0.144,
    beta: float = 0.5,
) -> dict[str, float]:
    """
    Computes ranking and operational metrics optimized for dating app class imbalance:
    1. PR-AUC (Average Precision)
    2. Precision@14.4% and Precision@10% (R-Precision under fixed swipe budgets)
    3. F0.5 score (Precision weighted 2x over Recall) & optimal threshold
    4. Matthews Correlation Coefficient (MCC)
    """
    y_true = (np.asarray(y_true) >= 50.0).astype(np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # 1. PR-AUC
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc_score = float(auc(recalls, precisions))

    # 2. Precision@Fixed Budget (top k% highest predicted scores)
    n_total = len(y_true)
    k_14_4 = max(1, int(n_total * base_rate))
    k_10_0 = max(1, int(n_total * 0.10))

    sorted_indices = np.argsort(-y_prob)
    top_14_4_idx = sorted_indices[:k_14_4]
    top_10_0_idx = sorted_indices[:k_10_0]

    prec_at_14_4 = float(np.mean(y_true[top_14_4_idx]))
    prec_at_10_0 = float(np.mean(y_true[top_10_0_idx]))

    # 3. Optimal Threshold for F0.5
    best_f_beta = -1.0
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_mcc = 0.0

    # Search across candidate thresholds
    cand_threshs = np.linspace(0.05, 0.95, 91)
    for t in cand_threshs:
        preds = (y_prob >= t).astype(np.int32)
        if np.sum(preds) == 0:
            continue
        fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if fb > best_f_beta:
            best_f_beta = float(fb)
            best_thresh = float(t)
            best_prec = float(precision_score(y_true, preds, zero_division=0))
            best_rec = float(recall_score(y_true, preds, zero_division=0))
            best_mcc = float(matthews_corrcoef(y_true, preds))

    return {
        "pr_auc": pr_auc_score,
        "precision_at_base_rate": prec_at_14_4,
        "precision_at_10_pct": prec_at_10_0,
        "optimal_threshold": best_thresh,
        "f_0_5": best_f_beta,
        "precision_at_optimal": best_prec,
        "recall_at_optimal": best_rec,
        "mcc": best_mcc,
    }
