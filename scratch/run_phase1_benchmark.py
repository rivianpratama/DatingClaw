from __future__ import annotations

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"
os.environ["BUMBLECLAW_CLIP_ALLOW_DOWNLOAD"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix,
    fbeta_score,
)

from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store
from face_similarity.regressor import load_regressor
from face_similarity.preference import load_preference_model, preference_probability, features_from_prediction
from face_similarity.scoring import score_embedding
from face_similarity.prediction import RatingPrediction
from face_similarity.neural_fusion import ContextGatedBilinearNet, SessionPairwiseSmoothedLoss
from face_similarity.noise_filter import compute_noise_weights
from face_similarity.memory_stats import extract_memory_features
from face_similarity.metrics import evaluate_predictions

def predict_multimodal_batch(regressor, face_embeddings, clip_embeddings):
    feature_mode = regressor.metadata.get("feature_mode", "face_clip")
    if feature_mode == "face_only":
        X = face_embeddings
    elif feature_mode == "clip_only":
        X = clip_embeddings
    elif feature_mode == "face_clip":
        X = np.hstack([face_embeddings, clip_embeddings])
    else:
        X = face_embeddings
    return regressor.estimator.predict(X)

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"PHASE 1 EXPANDING-WINDOW VALIDATION BENCHMARK (0-10k Train -> 10k-20k Holdout)")
    print(f"Compute Device: {device.upper()}")
    print("=" * 80)

    # 1. Load Training Stores (0-10k Pure)
    print("\n[1/5] Loading 10k Pure Training Stores...")
    train_face_store = load_store(str(root / "embeddings" / "reference_store_bumble_10k_pure.npz"))
    train_clip_store = load_clip_store(str(root / "embeddings" / "clip_store_bumble_10k_pure.npz"))
    
    X_train_face = train_face_store.embeddings
    X_train_clip = train_clip_store.embeddings
    y_train_raw = train_face_store.ratings
    y_train = (y_train_raw >= 50.0).astype(np.float32)
    n_train = len(y_train)
    print(f"Training Samples: {n_train} (Likes: {int(y_train.sum())}, Passes: {int((1-y_train).sum())})")

    # 2. Load Validation Stores (10k-20k Holdout)
    print("\n[2/5] Loading 10k-20k Validation Holdout Stores...")
    val_face_store = load_store(str(root / "embeddings" / "reference_store_bumble_10k_to_20k.npz"))
    val_clip_store = load_clip_store(str(root / "embeddings" / "clip_store_bumble_10k_to_20k.npz"))
    
    X_val_face = val_face_store.embeddings
    X_val_clip = val_clip_store.embeddings
    y_val_raw = val_face_store.ratings
    y_val = (y_val_raw >= 50.0).astype(np.float32)
    n_val = len(y_val)
    n_val_likes = int(y_val.sum())
    n_val_passes = int((1 - y_val).sum())
    base_rate = n_val_likes / n_val
    print(f"Validation Holdout Samples: {n_val} (Likes: {n_val_likes} [{base_rate*100:.2f}%], Passes: {n_val_passes})")

    # 3. Model Predictions Container
    results = {}

    # -------------------------------------------------------------
    # Candidate 1: Legacy face_biased (7k stock photo reference store)
    # -------------------------------------------------------------
    print("\n[3/5] Evaluating Baseline 1: Legacy face_biased...")
    try:
        legacy_store = load_store(str(root / "embeddings" / "reference_store_chandra_7k.npz"))
        legacy_reg = load_regressor(str(root / "models" / "rating_regressor.joblib"))
        legacy_multi_reg = load_regressor(str(root / "models" / "rating_regressor_multimodal.joblib"))

        # Predict Face Ridge
        preds_face = legacy_reg.estimator.predict(X_val_face)
        # Predict Multi Ridge
        preds_multi = predict_multimodal_batch(legacy_multi_reg, X_val_face, X_val_clip)
        # 50/50 blend
        pred_legacy_face_biased = 0.5 * preds_face + 0.5 * preds_multi
        results["Legacy face_biased"] = pred_legacy_face_biased
    except Exception as e:
        print("Error evaluating Legacy face_biased:", e)

    # -------------------------------------------------------------
    # Candidate 2: Experimental3 / multimodalx_original
    # -------------------------------------------------------------
    print("Evaluating Baseline 2: Experimental3 (multimodalx_original)...")
    try:
        exp_store = load_store(str(root / "embeddings" / "reference_store.npz"))
        exp_pref = load_preference_model(str(root / "models" / "bumble_preference_classifier.joblib"))

        # Fast matrix-based cosine similarity k-NN
        norm_val_face = X_val_face / (np.linalg.norm(X_val_face, axis=1, keepdims=True) + 1e-8)
        norm_exp_face = exp_store.embeddings / (np.linalg.norm(exp_store.embeddings, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(norm_val_face, norm_exp_face.T)
        top_k_idx = np.argpartition(-sim_matrix, 11, axis=1)[:, :11]
        scores_knn = np.mean(exp_store.ratings[top_k_idx], axis=1)

        preds_multi_exp = predict_multimodal_batch(legacy_multi_reg, X_val_face, X_val_clip)
        
        # Preference model probability
        pref_probs = []
        for sf, sm, sk in zip(preds_face, preds_multi_exp, scores_knn):
            pred_obj = RatingPrediction(rating=float(0.5*sf + 0.5*sm), method="face_biased", face_rating=float(sf), multimodal_rating=float(sm), knn_rating=float(sk))
            feats = features_from_prediction(pred_obj, threshold=55.0, face_weight=0.48)
            p = preference_probability(exp_pref, feats)
            pref_probs.append(p if p is not None else 0.5)
        pref_probs = np.array(pref_probs)

        pred_exp3 = 0.48 * preds_face + 0.48 * preds_multi_exp + 0.04 * (pref_probs * 100.0)
        results["Experimental3"] = pred_exp3
    except Exception as e:
        print("Error evaluating Experimental3:", e)

    # -------------------------------------------------------------
    # Candidate 3: Bumble 10k Pure (Trained strictly on 0-10k real swipes)
    # -------------------------------------------------------------
    print("Evaluating Baseline 3: Bumble 10k Pure (Face Ridge + Multi Ridge)...")
    pure_reg = load_regressor(str(root / "models" / "rating_regressor_bumble_10k_pure.joblib"))
    pure_multi_reg = load_regressor(str(root / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib"))

    pure_face_preds = pure_reg.estimator.predict(X_val_face)
    pure_multi_preds = predict_multimodal_batch(pure_multi_reg, X_val_face, X_val_clip)
    pred_10k_pure = 0.5 * pure_face_preds + 0.5 * pure_multi_preds
    results["Bumble 10k Pure"] = pred_10k_pure

    # -------------------------------------------------------------
    # Candidate 4: Head B (Stacked Non-Parametric Memory Head + Noise Filter)
    # -------------------------------------------------------------
    print("\n[4/5] Training & Evaluating Head B: Stacked Non-Parametric Memory Head...")
    X_train_concat = np.hstack([X_train_face, X_train_clip])
    X_val_concat = np.hstack([X_val_face, X_val_clip])

    # 15-NN noise filter on training data
    noise_weights, noise_stats = compute_noise_weights(X_train_concat, y_train, k_neighbors=15)
    flagged_pct = (noise_stats['total_flagged'] / noise_stats['total_samples']) * 100
    print(f"  Noise Filter: Flagged {noise_stats['total_flagged']} thumb-slips ({flagged_pct:.2f}%)")

    # Stacked k-NN memory features
    print("  Extracting k-NN memory features for Train and Validation sets...")
    mem_train = extract_memory_features(X_train_concat, X_train_concat, y_train, ref_weights=noise_weights)
    mem_val = extract_memory_features(X_val_concat, X_train_concat, y_train, ref_weights=noise_weights)

    # Train HistGradientBoosting on [Face || CLIP || Memory] with sample weights
    feat_train = np.hstack([X_train_concat, mem_train])
    feat_val = np.hstack([X_val_concat, mem_val])

    hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
    hgb.fit(feat_train, y_train, sample_weight=noise_weights)
    pred_head_b = hgb.predict_proba(feat_val)[:, 1]
    results["Head B (Stacked Memory)"] = pred_head_b

    # -------------------------------------------------------------
    # Candidate 5: Head A (ContextGatedBilinearNet with Session Pairwise Loss)
    # -------------------------------------------------------------
    print("\nTraining & Evaluating Head A: ContextGatedBilinearNet...")
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare dummy meta quality vector (norm of face emb, norm of clip emb, face score, dummy aspect)
    meta_train = np.column_stack([
        np.linalg.norm(X_train_face, axis=1, keepdims=True),
        np.linalg.norm(X_train_clip, axis=1, keepdims=True),
        pure_reg.estimator.predict(X_train_face).reshape(-1, 1) / 100.0,
        np.ones((n_train, 1), dtype=np.float32)
    ])
    meta_val = np.column_stack([
        np.linalg.norm(X_val_face, axis=1, keepdims=True),
        np.linalg.norm(X_val_clip, axis=1, keepdims=True),
        pure_face_preds.reshape(-1, 1) / 100.0,
        np.ones((n_val, 1), dtype=np.float32)
    ])

    # Convert to PyTorch tensors on GPU
    t_face_train = torch.from_numpy(X_train_face).float().to(device)
    t_clip_train = torch.from_numpy(X_train_clip).float().to(device)
    t_meta_train = torch.from_numpy(meta_train).float().to(device)
    t_mem_train = torch.from_numpy(mem_train).float().to(device)
    t_y_train = torch.from_numpy(y_train).float().to(device)
    t_weights_train = torch.from_numpy(noise_weights).float().to(device)

    t_face_val = torch.from_numpy(X_val_face).float().to(device)
    t_clip_val = torch.from_numpy(X_val_clip).float().to(device)
    t_meta_val = torch.from_numpy(meta_val).float().to(device)
    t_mem_val = torch.from_numpy(mem_val).float().to(device)

    model_a = ContextGatedBilinearNet(face_dim=512, vis_dim=512, meta_dim=4, memory_dim=4, latent_dim=128).to(device)
    optimizer = optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = SessionPairwiseSmoothedLoss(alpha=0.6, label_smoothing=0.03)

    model_a.train()
    batch_size = 256
    epochs = 25
    n_batches = int(np.ceil(n_train / batch_size))

    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            logits = model_a(t_face_train[idx], t_clip_train[idx], t_meta_train[idx], t_mem_train[idx])
            loss = loss_fn(logits, t_y_train[idx], sample_weights=t_weights_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

    model_a.eval()
    with torch.no_grad():
        val_logits = model_a(t_face_val, t_clip_val, t_meta_val, t_mem_val)
        pred_head_a = torch.sigmoid(val_logits).cpu().numpy()
        pred_head_a = np.nan_to_num(pred_head_a, nan=0.5)
    results["Head A (Neural Gated)"] = pred_head_a

    # -------------------------------------------------------------
    # Candidate 6: Ensemble Blend (Head A + Head B)
    # -------------------------------------------------------------
    def scale_01(arr):
        arr_clean = np.nan_to_num(arr, nan=float(np.nanmedian(arr)))
        mi, ma = np.min(arr_clean), np.max(arr_clean)
        return (arr_clean - mi) / (ma - mi + 1e-8)

    blend_pred = 0.5 * scale_01(pred_head_a) + 0.5 * scale_01(pred_head_b)
    results["Blend (Head A + B)"] = blend_pred

    # -------------------------------------------------------------
    # [5/5] Compute Full Evaluation Metrics & High-Recall Calibration
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"PHASE 1 LEADERBOARD: 9,778 HOLDOUT PROFILES (BASE RATE: {base_rate*100:.2f}%)")
    print("=" * 80)

    rows = []
    for name, pred in list(results.items()):
        nan_count = int(np.isnan(pred).sum())
        print(f"Model: {name:<26} | NaNs: {nan_count} | Min: {np.nanmin(pred):.4f} | Max: {np.nanmax(pred):.4f}")
        if nan_count > 0:
            pred = np.nan_to_num(pred, nan=float(np.nanmedian(pred)))
            results[name] = pred
        # PR-AUC
        precisions, recalls, thresholds = precision_recall_curve(y_val, pred)
        pr_auc = auc(recalls, precisions)

        # ROC-AUC
        roc_auc = roc_auc_score(y_val, pred)

        # Precision@Top 11.1% (Base rate fixed budget)
        k_top = max(1, int(len(y_val) * base_rate))
        top_idx = np.argsort(-pred)[:k_top]
        prec_at_base = np.mean(y_val[top_idx])

        # Find High-Recall Operating Point (Target >= 95% Recall to minimize Missed Likes)
        # Find lowest threshold that achieves recall >= 0.95
        cand_thresholds = np.sort(pred)
        high_rec_thresh = cand_thresholds[0]
        actual_recall = 1.0
        missed_likes = 0
        junk_filtered_pct = 0.0

        for t in cand_thresholds[::10]:
            preds_binary = (pred >= t).astype(int)
            rec = np.sum((preds_binary == 1) & (y_val == 1)) / n_val_likes
            if rec >= 0.95:
                high_rec_thresh = t
                actual_recall = rec
                missed_likes = int(np.sum((preds_binary == 0) & (y_val == 1)))
                junk_filtered_pct = float(np.sum((preds_binary == 0) & (y_val == 0)) / n_val_passes * 100.0)
            else:
                break

        # Max F0.5
        best_fb = 0.0
        best_mcc = 0.0
        for t in cand_thresholds[::50]:
            p_bin = (pred >= t).astype(int)
            if np.sum(p_bin) == 0: continue
            fb = fbeta_score(y_val, p_bin, beta=0.5, zero_division=0)
            if fb > best_fb:
                best_fb = fb
                best_mcc = matthews_corrcoef(y_val, p_bin)

        rows.append({
            "Model": name,
            "PR-AUC": round(pr_auc, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Precision@Top11%": round(prec_at_base * 100.0, 1),
            "F0.5 (Max)": round(best_fb, 4),
            "MCC": round(best_mcc, 4),
            "95% Recall Thresh": round(float(high_rec_thresh), 3),
            "Missed Likes": missed_likes,
            "Junk Filtered %": round(junk_filtered_pct, 1),
        })

    df_leaderboard = pd.DataFrame(rows).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    print(df_leaderboard.to_string(index=False))

    out_csv = root / "results" / "phase1_leaderboard_10k_to_20k.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_leaderboard.to_csv(out_csv, index=False)
    print(f"\nSaved leaderboard CSV to: {out_csv}")

if __name__ == "__main__":
    main()
