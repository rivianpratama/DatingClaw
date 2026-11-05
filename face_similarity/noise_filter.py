from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity

def compute_noise_weights(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int = 15,
    noise_weight: float = 0.15,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Computes sample-level confidence weights w_i based on Out-of-Fold predictions
    and Base-Rate-Aware 15-NN graph neighborhood agreement.

    Flags suspected accidental mis-swipes (thumbslips):
    1. Suspected False Like (y_i = 1): p_oof <= 0.15 and at most 1 of 15 NN is a Like (<= 6.7%).
    2. Suspected False Pass (y_i = 0): p_oof >= 0.80 and at least 8 of 15 NN are Likes (>= 53.3%).
    """
    n_samples = len(labels)
    weights = np.ones(n_samples, dtype=np.float32)

    # 1. Compute Out-of-Fold (OOF) continuous predictions
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    binary_labels = (labels >= 0.5).astype(np.int32) if np.max(labels) <= 1.0 else (labels >= 50.0).astype(np.int32)

    for train_idx, val_idx in skf.split(embeddings, binary_labels):
        x_tr, y_tr = embeddings[train_idx], binary_labels[train_idx]
        x_val = embeddings[val_idx]

        reg = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        reg.fit(x_tr, y_tr)
        oof_preds[val_idx] = np.clip(reg.predict(x_val), 0.0, 1.0)

    # 2. Compute 15-NN graph disagreement in embedding space
    # Normalize embeddings for fast cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_emb = embeddings / norms
    sim_matrix = normed_emb @ normed_emb.T
    np.fill_diagonal(sim_matrix, -1.0) # Ignore self-match

    flagged_false_likes = 0
    flagged_false_passes = 0

    for i in range(n_samples):
        # Top-K nearest neighbors
        nn_indices = np.argpartition(-sim_matrix[i], k_neighbors)[:k_neighbors]
        nn_likes_count = int(np.sum(binary_labels[nn_indices] == 1))
        
        y_i = binary_labels[i]
        p_oof = oof_preds[i]

        # Check conditions
        if y_i == 1:
            # Suspected Accidental Right-Swipe
            if p_oof <= 0.15 and nn_likes_count <= 1:
                weights[i] = noise_weight
                flagged_false_likes += 1
        else:
            # Suspected Accidental Left-Swipe
            if p_oof >= 0.80 and nn_likes_count >= 8:
                weights[i] = noise_weight
                flagged_false_passes += 1

    stats = {
        "total_samples": n_samples,
        "flagged_false_likes": flagged_false_likes,
        "flagged_false_passes": flagged_false_passes,
        "total_flagged": flagged_false_likes + flagged_false_passes,
    }
    return weights, stats
