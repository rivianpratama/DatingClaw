from __future__ import annotations

import numpy as np

def extract_memory_features(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_labels: np.ndarray,
    ref_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes 4 stacked memory features for each query sample against the historical reference store:
    1. mu_pos@5: mean similarity to top-5 Likes
    2. mu_pos@20: mean similarity to top-20 Likes
    3. mu_neg@20: mean similarity to top-20 Passes
    4. density_ratio: mu_pos@20 / (mu_pos@20 + mu_neg@20)

    If ref_weights is provided, neighbor similarities are modulated by w_i.
    """
    if ref_weights is None:
        ref_weights = np.ones(len(ref_labels), dtype=np.float32)

    ref_binary = (ref_labels >= 0.5).astype(np.int32) if np.max(ref_labels) <= 1.0 else (ref_labels >= 50.0).astype(np.int32)
    pos_mask = (ref_binary == 1)
    neg_mask = (ref_binary == 0)

    pos_embeddings = ref_embeddings[pos_mask]
    pos_weights = ref_weights[pos_mask]

    neg_embeddings = ref_embeddings[neg_mask]
    neg_weights = ref_weights[neg_mask]

    # Normalize
    q_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True).clip(1e-12)
    pos_norm = pos_embeddings / np.linalg.norm(pos_embeddings, axis=1, keepdims=True).clip(1e-12)
    neg_norm = neg_embeddings / np.linalg.norm(neg_embeddings, axis=1, keepdims=True).clip(1e-12)

    # Compute cosine similarity matrices
    sim_pos = q_norm @ pos_norm.T * pos_weights[np.newaxis, :]
    sim_neg = q_norm @ neg_norm.T * neg_weights[np.newaxis, :]

    n_queries = len(query_embeddings)
    memory_feats = np.zeros((n_queries, 4), dtype=np.float32)

    for i in range(n_queries):
        row_pos = sim_pos[i]
        row_neg = sim_neg[i]

        # Top-5 and Top-20 Likes
        top5_pos = np.partition(-row_pos, min(5, len(row_pos) - 1))[:5] if len(row_pos) >= 5 else -row_pos
        top20_pos = np.partition(-row_pos, min(20, len(row_pos) - 1))[:20] if len(row_pos) >= 20 else -row_pos

        # Top-20 Passes
        top20_neg = np.partition(-row_neg, min(20, len(row_neg) - 1))[:20] if len(row_neg) >= 20 else -row_neg

        mu_pos_5 = float(-np.mean(top5_pos))
        mu_pos_20 = float(-np.mean(top20_pos))
        mu_neg_20 = float(-np.mean(top20_neg))

        denom = mu_pos_20 + mu_neg_20
        density_ratio = mu_pos_20 / denom if denom > 1e-6 else 0.5

        memory_feats[i, 0] = mu_pos_5
        memory_feats[i, 1] = mu_pos_20
        memory_feats[i, 2] = mu_neg_20
        memory_feats[i, 3] = density_ratio

    return memory_feats
