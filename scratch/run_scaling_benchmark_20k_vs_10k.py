import os
from pathlib import Path
import sys

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    matthews_corrcoef,
    fbeta_score,
)

from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store
from face_similarity.regressor import load_regressor
from face_similarity.preference import load_preference_model, preference_probability, features_from_prediction
from face_similarity.prediction import RatingPrediction

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
    print("=" * 80)
    print("SCALING BENCHMARK: 10K vs 20K PURE ON UNTRAINED 20k-22k HOLDOUT")
    print("=" * 80)

    # 1. Load Holdout Data (20k-22k)
    val_face_store = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    val_clip_store = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_to_22k.npz"))

    X_val_face = val_face_store.embeddings
    X_val_clip = val_clip_store.embeddings
    y_val = (val_face_store.ratings >= 50.0).astype(np.float32)

    n_val = len(y_val)
    n_likes = int(y_val.sum())
    n_passes = int((1 - y_val).sum())
    base_rate = n_likes / n_val
    print(f"Holdout Size: {n_val} profiles (Likes: {n_likes} [{base_rate*100:.2f}%], Passes: {n_passes})")

    results = {}

    # -------------------------------------------------------------
    # 1. Legacy face_biased (7k Stock)
    # -------------------------------------------------------------
    print("\n[1/4] Evaluating Legacy face_biased...")
    legacy_reg = load_regressor(str(root / "models" / "rating_regressor.joblib"))
    legacy_multi_reg = load_regressor(str(root / "models" / "rating_regressor_multimodal.joblib"))

    preds_legacy_face = legacy_reg.estimator.predict(X_val_face)
    preds_legacy_multi = predict_multimodal_batch(legacy_multi_reg, X_val_face, X_val_clip)
    results["Legacy face_biased (7k)"] = 0.5 * preds_legacy_face + 0.5 * preds_legacy_multi

    # -------------------------------------------------------------
    # 2. Experimental3 / multimodalx_original
    # -------------------------------------------------------------
    print("[2/4] Evaluating Experimental3 (multimodalx_original)...")
    exp_store = load_store(str(root / "embeddings" / "reference_store.npz"))
    exp_pref = load_preference_model(str(root / "models" / "bumble_preference_classifier.joblib"))

    norm_val_face = X_val_face / (np.linalg.norm(X_val_face, axis=1, keepdims=True) + 1e-8)
    norm_exp_face = exp_store.embeddings / (np.linalg.norm(exp_store.embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(norm_val_face, norm_exp_face.T)
    top_k_idx = np.argpartition(-sim_matrix, 11, axis=1)[:, :11]
    scores_knn = np.mean(exp_store.ratings[top_k_idx], axis=1)

    pref_probs = []
    for sf, sm, sk in zip(preds_legacy_face, preds_legacy_multi, scores_knn):
        pred_obj = RatingPrediction(rating=float(0.5*sf + 0.5*sm), method="face_biased", face_rating=float(sf), multimodal_rating=float(sm), knn_rating=float(sk))
        feats = features_from_prediction(pred_obj, threshold=55.0, face_weight=0.48)
        p = preference_probability(exp_pref, feats)
        pref_probs.append(p if p is not None else 0.5)
    pref_probs = np.array(pref_probs)
    results["Experimental3 (Production)"] = 0.48 * preds_legacy_face + 0.48 * preds_legacy_multi + 0.04 * (pref_probs * 100.0)

    # -------------------------------------------------------------
    # 3. Bumble 10k Pure (Trained on 0-10k)
    # -------------------------------------------------------------
    print("[3/4] Evaluating Bumble 10k Pure (Trained on 10,000 real swipes)...")
    pure_10k_reg = load_regressor(str(root / "models" / "rating_regressor_bumble_10k_pure.joblib"))
    pure_10k_multi = load_regressor(str(root / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib"))

    p10k_face = pure_10k_reg.estimator.predict(X_val_face)
    p10k_multi = predict_multimodal_batch(pure_10k_multi, X_val_face, X_val_clip)
    results["Bumble 10k Pure (10k data)"] = 0.5 * p10k_face + 0.5 * p10k_multi

    # -------------------------------------------------------------
    # 4. Bumble 20k Pure (Trained on 0-20k)
    # -------------------------------------------------------------
    print("[4/4] Evaluating Bumble 20k Pure (Trained on 20,000 real swipes - DOUBLED DATASET)...")
    pure_20k_reg = load_regressor(str(root / "models" / "rating_regressor_bumble_20k_pure.joblib"))
    pure_20k_multi = load_regressor(str(root / "models" / "rating_regressor_multimodal_bumble_20k_pure.joblib"))

    p20k_face = pure_20k_reg.estimator.predict(X_val_face)
    p20k_multi = predict_multimodal_batch(pure_20k_multi, X_val_face, X_val_clip)

    results["Bumble 20k Pure (60/40 Aligned)"] = 0.6 * p20k_face + 0.4 * p20k_multi
    results["Bumble 20k Pure (50/50 Blend)"] = 0.5 * p20k_face + 0.5 * p20k_multi
    results["Bumble 20k Pure (Multimodal Only)"] = p20k_multi
    results["Bumble 20k Pure (Face Only)"] = p20k_face

    # -------------------------------------------------------------
    # Compute Metrics & Leaderboard
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCALING LEADERBOARD ON 2,000 UNSEEN HOLDOUT PROFILES")
    print("=" * 80)

    rows = []
    for name, pred in results.items():
        precisions, recalls, thresholds = precision_recall_curve(y_val, pred)
        pr_auc = auc(recalls, precisions)
        roc_auc = roc_auc_score(y_val, pred)

        # Precision@Top 8% (budget matching base rate)
        k_top = max(1, int(len(y_val) * base_rate))
        top_idx = np.argsort(-pred)[:k_top]
        prec_at_base = np.mean(y_val[top_idx])

        # Find 95% Recall Threshold
        cand_t = np.sort(pred)
        high_rec_thresh = cand_t[0]
        actual_rec = 1.0
        missed_likes = 0
        junk_filtered_pct = 0.0

        for t in cand_t[::5]:
            p_bin = (pred >= t).astype(int)
            rec = np.sum((p_bin == 1) & (y_val == 1)) / n_likes
            if rec >= 0.95:
                high_rec_thresh = t
                actual_rec = rec
                missed_likes = int(np.sum((p_bin == 0) & (y_val == 1)))
                junk_filtered_pct = float(np.sum((p_bin == 0) & (y_val == 0)) / n_passes * 100.0)
            else:
                break

        # Max F0.5
        best_fb = 0.0
        best_mcc = 0.0
        for t in cand_t[::20]:
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
            "Precision@Top8%": round(prec_at_base * 100.0, 1),
            "Max F0.5": round(best_fb, 4),
            "MCC": round(best_mcc, 4),
            "95% Recall Thresh": round(float(high_rec_thresh), 3),
            "Missed Likes": missed_likes,
            "Junk Filtered %": round(junk_filtered_pct, 1),
        })

    df_out = pd.DataFrame(rows).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    print(df_out.to_string(index=False))

    out_csv = root / "results" / "scaling_benchmark_20k_vs_10k.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved scaling benchmark report to: {out_csv}")

if __name__ == "__main__":
    main()
