from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from face_similarity.neural_fusion import ContextGatedBilinearNet, SessionPairwiseSmoothedLoss
from face_similarity.noise_filter import compute_noise_weights
from face_similarity.memory_stats import extract_memory_features
from face_similarity.metrics import evaluate_predictions
from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Head Walk-Forward Phase Validation for BumbleClaw")
    parser.add_argument("--train-end", type=int, default=10000, help="Approximate training split end")
    parser.add_argument("--val-end", type=int, default=20000, help="Approximate validation split end")
    parser.add_argument("--labels", default=r"D:\BumbleLog\binary_swipe_labels.csv", help="Swipe labels CSV")
    parser.add_argument("--face-store", default="embeddings/reference_store_bumble_10k_pure.npz", help="Train face store")
    parser.add_argument("--clip-store", default="embeddings/clip_store_bumble_10k_pure.npz", help="Train CLIP store")
    parser.add_argument("--report", default="results/phase1_leaderboard.csv", help="Leaderboard output path")
    parser.add_argument("--epochs", type=int, default=30, help="Neural head training epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()

def find_session_boundary(timestamps: list[str], target_index: int, max_gap_seconds: float = 900.0) -> int:
    """Finds nearest timestamp gap > 15 minutes near target_index to avoid splitting sessions."""
    n = len(timestamps)
    if target_index >= n:
        return n
    
    start_search = max(1, target_index - 500)
    end_search = min(n - 1, target_index + 500)
    
    best_index = target_index
    max_gap = 0.0

    for i in range(start_search, end_search):
        try:
            t1 = datetime.fromisoformat(timestamps[i - 1].replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
            gap = abs((t2 - t1).total_seconds())
            if gap >= max_gap_seconds and gap > max_gap:
                max_gap = gap
                best_index = i
        except Exception:
            continue
            
    return best_index

def extract_session_pairs(timestamps: list[str], labels: np.ndarray, max_gap_seconds: float = 900.0) -> list[tuple[int, int]]:
    """Generates pairwise indices (pos_i, neg_j) within the same contiguous session."""
    pairs = []
    n = len(timestamps)
    curr_pos = []
    curr_neg = []
    
    for i in range(n):
        if i > 0:
            try:
                t1 = datetime.fromisoformat(timestamps[i - 1].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
                if abs((t2 - t1).total_seconds()) > max_gap_seconds:
                    for p in curr_pos:
                        for neg in curr_neg:
                            pairs.append((p, neg))
                    curr_pos = []
                    curr_neg = []
            except Exception:
                pass
                
        if labels[i] >= 50.0:
            curr_pos.append(i)
        else:
            curr_neg.append(i)

    for p in curr_pos:
        for neg in curr_neg:
            pairs.append((p, neg))
            
    return pairs

def main():
    args = parse_args()
    print("================================================================================")
    print("BUMBLECLAW DUAL-HEAD WALK-FORWARD VALIDATION HARNESS")
    print(f"Device: {args.device.upper()} | Evaluating Head A (Neural) vs Head B (OOF Stack) vs Blend")
    print("================================================================================")

    # 1. Load historical stores
    face_store = load_store(args.face_store)
    clip_store = load_clip_store(args.clip_store)
    print(f"Loaded training reference store: {len(face_store.embeddings)} samples")

    # 2. Check full labels CSV
    df = pd.read_csv(args.labels)
    df = df.drop_duplicates(subset=['filename'], keep='last').reset_index(drop=True)
    total_labels = len(df)
    print(f"Total labeled profiles: {total_labels}")

    if total_labels < args.train_end + 100:
        print(f"Waiting for more validation swipes (currently {total_labels}, target {args.val_end}).")
        return 0

    # 3. Determine session-aware split boundaries
    train_split_idx = find_session_boundary(df['timestamp'].tolist(), args.train_end)
    val_split_idx = min(total_labels, find_session_boundary(df['timestamp'].tolist(), args.val_end))
    print(f"Session-aligned Train Split: [0:{train_split_idx}] | Validation Split: [{train_split_idx}:{val_split_idx}]")

    train_df = df.iloc[:train_split_idx]
    val_df = df.iloc[train_split_idx:val_split_idx]
    print(f"Train profiles: {len(train_df)} (Likes: {sum(train_df['action'] == 'right')})")
    print(f"Validation profiles: {len(val_df)} (Likes: {sum(val_df['action'] == 'right')})")

    # 4. Filter Noise on Train Split
    X_train_face = face_store.embeddings
    X_train_clip = clip_store.embeddings
    y_train = (face_store.ratings >= 50.0).astype(np.float32)

    X_train_concat = np.hstack([X_train_face, X_train_clip])
    print("Computing base-rate-aware 15-NN noise weights on training split...")
    noise_weights, noise_stats = compute_noise_weights(X_train_concat, y_train, k_neighbors=15)
    print(f"Noise Filter: {noise_stats['total_flagged']} suspicious mis-swipes down-weighted (w=0.15)")

    # 5. Extract Non-Parametric Memory Features
    print("Computing stacked k-NN memory features...")
    mem_train = extract_memory_features(X_train_concat, X_train_concat, y_train, ref_weights=noise_weights)
    print("Validation extraction will run when validation embeddings are computed.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
