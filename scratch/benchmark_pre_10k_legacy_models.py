"""
Benchmark all pre-10k legacy models on the standardized validation holdout val_1582.
Evaluates every legacy checkpoint from May-June 2026 across:
- Pure Face Regressors (ArcFace 512D)
- Multimodal Regressors (ArcFace 512D + CLIP ViT-B/32 512D)
- Preference Meta-Classifiers (Spline Logistic P(like))
- Composite Production Blends (face_biased, MultimodalX, MultimodalX2, MultimodalX5)
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Ensure local imports work
sys.path.insert(0, ".")
from train_multimodal_regressor import ExpectedScoreClassifier
from face_similarity.store import load_store
from face_similarity.scoring import score_embedding
from face_similarity.preference import load_preference_model, make_features, preference_probability
from face_similarity.prediction import RatingPrediction
from face_similarity.multimodalx import score as multimodalx_score


def evaluate_ranking(y_true: np.ndarray, scores: np.ndarray, cap20_count: int, cap25_count: int):
    total_positives = int((y_true > 0).sum())
    total_samples = len(y_true)
    
    # Sort descending
    sort_idx = np.argsort(-scores)
    y_sorted = y_true[sort_idx]
    
    # ROC-AUC and PR-AUC
    roc_auc = float(roc_auc_score(y_true, scores))
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    pr_auc = float(auc(rec_arr, prec_arr))
    
    # Top 20%
    hits_20 = int((y_sorted[:cap20_count] > 0).sum())
    recall_20 = hits_20 / total_positives * 100.0
    prec_20 = hits_20 / cap20_count * 100.0
    missed_20 = total_positives - hits_20
    
    # Top 25%
    hits_25 = int((y_sorted[:cap25_count] > 0).sum())
    recall_25 = hits_25 / total_positives * 100.0
    prec_25 = hits_25 / cap25_count * 100.0
    missed_25 = total_positives - hits_25
    
    # Top 100
    hits_100 = int((y_sorted[:100] > 0).sum())
    prec_100 = hits_100 / 100.0 * 100.0
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "hits_20": hits_20,
        "recall_20": recall_20,
        "prec_20": prec_20,
        "missed_20": missed_20,
        "hits_25": hits_25,
        "recall_25": recall_25,
        "prec_25": prec_25,
        "missed_25": missed_25,
        "prec_100": prec_100,
    }


def main():
    print("Loading validation dataset val_1582...")
    ref_val = np.load("embeddings/reference_store_val_1582.npz")
    clip_val = np.load("embeddings/clip_store_val_1582.npz")
    
    face_embs = ref_val["embeddings"] # (1533, 512)
    clip_embs = clip_val["embeddings"] # (1533, 512)
    multi_embs = np.hstack([face_embs, clip_embs]) # (1533, 1024)
    ratings = ref_val["ratings"]
    y_true = (ratings == 100.0).astype(int)
    
    N = len(y_true)
    total_likes = int(y_true.sum())
    cap20_count = int(np.round(N * 0.20)) # 307
    cap25_count = int(np.round(N * 0.25)) # 383
    
    print(f"Total validation samples: {N}, True Likes: {total_likes} ({total_likes/N*100:.2f}%)")
    print(f"Cap @ 20%: {cap20_count} swipes, Cap @ 25%: {cap25_count} swipes")
    
    # Load 7k reference store for kNN
    print("Loading 7k reference store for kNN...")
    store_7k = load_store("embeddings/reference_store.npz")
    
    # Precompute kNN ratings (k=11)
    print("Precomputing k=11 kNN ratings across 7k store...")
    knn_ratings = []
    # Fast cosine similarity: normalize both
    ref_norms = np.linalg.norm(store_7k.embeddings, axis=1, keepdims=True) + 1e-7
    store_normed = store_7k.embeddings / ref_norms
    face_norms = np.linalg.norm(face_embs, axis=1, keepdims=True) + 1e-7
    face_normed = face_embs / face_norms
    
    sim_matrix = face_normed @ store_normed.T # (1533, 7144)
    k = 11
    for i in range(N):
        sims = sim_matrix[i]
        top_k_idx = np.argpartition(-sims, k)[:k]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        top_sims = sims[top_k_idx]
        top_rats = store_7k.ratings[top_k_idx]
        weights = np.maximum(top_sims, 0) ** 2
        if weights.sum() == 0:
            knn_ratings.append(float(np.mean(top_rats)))
        else:
            knn_ratings.append(float(np.average(top_rats, weights=weights)))
    knn_ratings = np.array(knn_ratings, dtype=np.float32)
    print("kNN ratings precomputed. Mean:", knn_ratings.mean())
    
    results = []
    
    # --- Category 1: Pure Face Regressors ---
    face_models = [
        ("rating_regressor.joblib", "May 19 04:01", "Chandra 7k", "ArcFace 512D Ridge", "models/rating_regressor.joblib"),
        ("rating_regressor_normalized.joblib", "May 19 04:19", "Chandra 7k (L2 Norm)", "ArcFace 512D Normalized Ridge", "models/rating_regressor_normalized.joblib"),
        ("rating_regressor_chandra_7k.joblib", "Jun 12 12:15", "Chandra 7k Cleaned", "ArcFace 512D Ridge Baseline", "models/rating_regressor_chandra_7k.joblib"),
        ("rating_regressor_bumble_combined.joblib", "May 19 21:05", "Chandra + Bumble R1 (8.5k)", "ArcFace 512D Ridge (Combined R1)", "models/rating_regressor_bumble_combined.joblib"),
        ("rating_regressor_bumble_combined_round2.joblib", "May 19 22:16", "Chandra + Bumble R2 (8.8k)", "ArcFace 512D Ridge (Combined R2)", "models/rating_regressor_bumble_combined_round2.joblib"),
        ("rating_regressor_bumble_combined_round3.joblib", "May 20 15:23", "Chandra + Bumble R3 (9.1k)", "ArcFace 512D Ridge (Combined R3)", "models/rating_regressor_bumble_combined_round3.joblib"),
        ("rating_regressor_bumble_only.joblib", "May 19 21:10", "Bumble Only R1 (1k)", "ArcFace 512D HistGBDT (Bumble R1)", "models/rating_regressor_bumble_only.joblib"),
        ("rating_regressor_bumble_only_round2.joblib", "May 19 22:25", "Bumble Only R2 (1.7k)", "ArcFace 512D HistGBDT (Bumble R2)", "models/rating_regressor_bumble_only_round2.joblib"),
    ]
    
    cached_face_scores = {}
    
    print("\n--- Evaluating Pure Face Regressors ---")
    for fname, date, dataset, arch, path in face_models:
        payload = joblib.load(path)
        est = payload["estimator"] if isinstance(payload, dict) and "estimator" in payload else payload
        scores = est.predict(face_embs)
        cached_face_scores[fname] = scores
        metrics = evaluate_ranking(y_true, scores, cap20_count, cap25_count)
        results.append({
            "Category": "1. Face Regressor",
            "Model Name": fname,
            "Date": date,
            "Training Data": dataset,
            "Architecture": arch,
            **metrics
        })
        print(f"{fname:<45} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Hits@25%: {metrics['hits_25']}/140 ({metrics['recall_25']:.1f}%) | Hits@20%: {metrics['hits_20']}/140")

    # --- Category 2: Multimodal Regressors ---
    multi_models = [
        ("rating_regressor_multimodal.joblib", "May 19 14:01", "Chandra 7k", "ArcFace + CLIP 1024D Ridge", "models/rating_regressor_multimodal.joblib"),
        ("rating_regressor_multimodal_chandra_7k.joblib", "Jun 12 12:15", "Chandra 7k Cleaned", "ArcFace + CLIP ExpectedScore (Logistic)", "models/rating_regressor_multimodal_chandra_7k.joblib"),
        ("rating_regressor_multimodal_bumble_combined.joblib", "May 19 21:06", "Chandra + Bumble R1 (8.5k)", "ArcFace + CLIP 1024D MLP (Combined R1)", "models/rating_regressor_multimodal_bumble_combined.joblib"),
        ("rating_regressor_multimodal_bumble_combined_round2.joblib", "May 19 22:18", "Chandra + Bumble R2 (8.8k)", "ArcFace + CLIP 1024D MLP (Combined R2)", "models/rating_regressor_multimodal_bumble_combined_round2.joblib"),
        ("rating_regressor_multimodal_bumble_combined_round3.joblib", "May 20 15:23", "Chandra + Bumble R3 (9.1k)", "ArcFace + CLIP 1024D Ridge (Combined R3)", "models/rating_regressor_multimodal_bumble_combined_round3.joblib"),
        ("rating_regressor_multimodal_bumble_only.joblib", "May 19 21:10", "Bumble Only R1 (1k)", "ArcFace + CLIP ExpectedScore (Bumble R1)", "models/rating_regressor_multimodal_bumble_only.joblib"),
        ("rating_regressor_multimodal_bumble_only_round2.joblib", "May 19 22:25", "Bumble Only R2 (1.7k)", "ArcFace + CLIP 1024D MLP (Bumble R2)", "models/rating_regressor_multimodal_bumble_only_round2.joblib"),
    ]
    
    cached_multi_scores = {}
    
    print("\n--- Evaluating Multimodal Regressors ---")
    for fname, date, dataset, arch, path in multi_models:
        payload = joblib.load(path)
        est = payload["estimator"] if isinstance(payload, dict) and "estimator" in payload else payload
        scores = est.predict(multi_embs)
        cached_multi_scores[fname] = scores
        metrics = evaluate_ranking(y_true, scores, cap20_count, cap25_count)
        results.append({
            "Category": "2. Multimodal Regressor",
            "Model Name": fname,
            "Date": date,
            "Training Data": dataset,
            "Architecture": arch,
            **metrics
        })
        print(f"{fname:<45} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Hits@25%: {metrics['hits_25']}/140 ({metrics['recall_25']:.1f}%) | Hits@20%: {metrics['hits_20']}/140")

    # --- Category 3: Preference Meta-Classifiers (P(like)) ---
    pref_files = [
        ("bumble_preference_classifier.joblib", "May 21 07:03", "Bumble Preferences R1-R2", "Spline Logistic (13 Features)", "models/bumble_preference_classifier.joblib"),
        ("bumble_preference_experimental1.joblib", "May 21 08:08", "Bumble Preferences R1-R2", "Spline Logistic Exp 1 (13 Features)", "models/bumble_preference_experimental1.joblib"),
        ("bumble_preference_experimental2.joblib", "May 21 08:38", "Bumble Preferences R1-R2", "Spline Logistic Exp 2 (13 Features)", "models/bumble_preference_experimental2.joblib"),
        ("bumble_preference_experimental3.joblib", "May 22 14:32", "Bumble Preferences R1-R3", "Spline Logistic Exp 3 Original Veto", "models/bumble_preference_experimental3.joblib"),
        ("bumble_preference_round2_veto.joblib", "May 22 12:51", "Bumble Preferences R1-R2", "Round 2 Veto Spline Logistic", "models/bumble_preference_round2_veto.joblib"),
        ("bumble_preference_multimodalx3.joblib", "May 22 12:51", "Bumble Preferences R1-R3", "MultimodalX3 Round 3 Veto Spline", "models/bumble_preference_multimodalx3.joblib"),
        ("bumble_preference_multimodalx4.joblib", "May 22 12:51", "Bumble Preferences R1-R3", "MultimodalX4 MultimodalX2 Veto Spline", "models/bumble_preference_multimodalx4.joblib"),
        ("bumble_preference_multimodalx5.joblib", "May 21 21:43", "Bumble Preferences R1-R3", "MultimodalX5 Tuned Veto Spline (14 Feat)", "models/bumble_preference_multimodalx5.joblib"),
    ]
    
    # Base ridge and multimodal for feature creation
    s_face_base = cached_face_scores["rating_regressor.joblib"]
    s_multi_base = cached_multi_scores["rating_regressor_multimodal.joblib"]
    
    cached_p_like = {}
    
    print("\n--- Evaluating Preference Meta-Classifiers (Standalone P(like)) ---")
    for fname, date, dataset, arch, path in pref_files:
        pref_model = load_preference_model(path)
        probs = []
        for i in range(N):
            sf = float(s_face_base[i])
            sm = float(s_multi_base[i])
            sk = float(knn_ratings[i])
            s_blend = 0.5 * sf + 0.5 * sm
            
            row_dict = {
                "score": s_blend,
                "face_biased": s_blend,
                "multimodal": sm,
                "ridge": sf,
                "knn": sk,
                "threshold": 55.0,
                "face_weight": 0.50,
                "regressor_path": "models/rating_regressor.joblib",
                "multimodal_regressor_path": "models/rating_regressor_multimodal.joblib",
            }
            feats = make_features(row_dict)
            if "old_p_like" in pref_model.feature_fields:
                feats["old_p_like"] = float(cached_p_like.get("bumble_preference_classifier.joblib", np.zeros(N))[i])
            
            p = preference_probability(pref_model, feats)
            probs.append(p)
            
        probs = np.array(probs, dtype=np.float32)
        cached_p_like[fname] = probs
        metrics = evaluate_ranking(y_true, probs, cap20_count, cap25_count)
        results.append({
            "Category": "3. Preference Classifier",
            "Model Name": fname,
            "Date": date,
            "Training Data": dataset,
            "Architecture": arch,
            **metrics
        })
        print(f"{fname:<45} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Hits@25%: {metrics['hits_25']}/140 ({metrics['recall_25']:.1f}%) | Hits@20%: {metrics['hits_20']}/140")

    # --- Category 4: Composite Legacy Blends ---
    print("\n--- Evaluating Composite Production Blends ---")
    
    # 1. face_biased (7k stock)
    scores_face_biased = 0.50 * s_face_base + 0.50 * s_multi_base
    m_fb = evaluate_ranking(y_true, scores_face_biased, cap20_count, cap25_count)
    results.append({
        "Category": "4. Production Blend",
        "Model Name": "face_biased (7k Stock Baseline)",
        "Date": "May 19 14:30",
        "Training Data": "Chandra 7k",
        "Architecture": "50% ArcFace + 50% Multimodal Ridge",
        **m_fb
    })
    print(f"{'face_biased (7k Stock Baseline)':<45} | ROC-AUC: {m_fb['roc_auc']:.4f} | PR-AUC: {m_fb['pr_auc']:.4f} | Hits@25%: {m_fb['hits_25']}/140 ({m_fb['recall_25']:.1f}%)")

    # 2. MultimodalX Original Blend
    p_orig = cached_p_like["bumble_preference_classifier.joblib"]
    scores_multi_orig = 0.48 * s_face_base + 0.48 * s_multi_base + 0.04 * (p_orig * 100.0)
    m_mo = evaluate_ranking(y_true, scores_multi_orig, cap20_count, cap25_count)
    results.append({
        "Category": "4. Production Blend",
        "Model Name": "MultimodalX Original Blend",
        "Date": "May 21 07:30",
        "Training Data": "Chandra 7k + Bumble R1-R2",
        "Architecture": "48% Face + 48% Multi + 4% P(like)",
        **m_mo
    })
    print(f"{'MultimodalX Original Blend':<45} | ROC-AUC: {m_mo['roc_auc']:.4f} | PR-AUC: {m_mo['pr_auc']:.4f} | Hits@25%: {m_mo['hits_25']}/140 ({m_mo['recall_25']:.1f}%)")

    # 3. MultimodalX (73/20/7 Blend)
    scores_multi_73 = 0.73 * s_face_base + 0.20 * s_multi_base + 0.07 * (p_orig * 100.0)
    m_m73 = evaluate_ranking(y_true, scores_multi_73, cap20_count, cap25_count)
    results.append({
        "Category": "4. Production Blend",
        "Model Name": "MultimodalX (73/20/7 Face Heavy)",
        "Date": "May 21 08:30",
        "Training Data": "Chandra 7k + Bumble R1-R2",
        "Architecture": "73% Face + 20% Multi + 7% P(like)",
        **m_m73
    })
    print(f"{'MultimodalX (73/20/7 Face Heavy)':<45} | ROC-AUC: {m_m73['roc_auc']:.4f} | PR-AUC: {m_m73['pr_auc']:.4f} | Hits@25%: {m_m73['hits_25']}/140 ({m_m73['recall_25']:.1f}%)")

    # 4. MultimodalX2 (12/5/53/30 Blend)
    scores_multix2 = 0.12 * s_face_base + 0.05 * s_multi_base + 0.53 * (p_orig * 100.0) + 0.30 * knn_ratings
    m_mx2 = evaluate_ranking(y_true, scores_multix2, cap20_count, cap25_count)
    results.append({
        "Category": "4. Production Blend",
        "Model Name": "MultimodalX2 (P-Like Heavy Blend)",
        "Date": "May 21 10:00",
        "Training Data": "Chandra 7k + Bumble R1-R2",
        "Architecture": "12% Face + 5% Multi + 53% P(like) + 30% kNN",
        **m_mx2
    })
    print(f"{'MultimodalX2 (P-Like Heavy Blend)':<45} | ROC-AUC: {m_mx2['roc_auc']:.4f} | PR-AUC: {m_mx2['pr_auc']:.4f} | Hits@25%: {m_mx2['hits_25']}/140 ({m_mx2['recall_25']:.1f}%)")

    # 5. MultimodalX5 (60/5/15/20 Blend)
    p_x5 = cached_p_like["bumble_preference_multimodalx5.joblib"]
    scores_multix5 = 0.60 * s_face_base + 0.05 * s_multi_base + 0.15 * (p_x5 * 100.0) + 0.20 * knn_ratings
    m_mx5 = evaluate_ranking(y_true, scores_multix5, cap20_count, cap25_count)
    results.append({
        "Category": "4. Production Blend",
        "Model Name": "MultimodalX5 (Tuned Feedback Blend)",
        "Date": "May 21 22:00",
        "Training Data": "Chandra 7k + Bumble R1-R3",
        "Architecture": "60% Face + 5% Multi + 15% P(like) + 20% kNN",
        **m_mx5
    })
    print(f"{'MultimodalX5 (Tuned Feedback Blend)':<45} | ROC-AUC: {m_mx5['roc_auc']:.4f} | PR-AUC: {m_mx5['pr_auc']:.4f} | Hits@25%: {m_mx5['hits_25']}/140 ({m_mx5['recall_25']:.1f}%)")

    # --- Anchor References: 10k Pure & v4 Grand Super-Ensemble ---
    print("\n--- Anchor Reference Models ---")
    p_10k_multi = joblib.load("models/rating_regressor_multimodal_bumble_10k_pure.joblib")
    est_10k = p_10k_multi["estimator"] if isinstance(p_10k_multi, dict) and "estimator" in p_10k_multi else p_10k_multi
    feat_mode_10k = p_10k_multi.get("metadata", {}).get("feature_mode", "face_clip") if isinstance(p_10k_multi, dict) else "face_clip"
    inp_10k = face_embs if feat_mode_10k == "face_only" else multi_embs
    scores_10k = est_10k.predict(inp_10k)
    m_10k = evaluate_ranking(y_true, scores_10k, cap20_count, cap25_count)
    results.append({
        "Category": "5. Modern Anchor",
        "Model Name": "10k Pure (Multimodal MLP)",
        "Date": "Oct 28 19:20",
        "Training Data": "Bumble 10k Pure",
        "Architecture": f"Nonlinear MLP ({feat_mode_10k})",
        **m_10k
    })
    print(f"{'10k Pure (Multimodal MLP)':<45} | ROC-AUC: {m_10k['roc_auc']:.4f} | PR-AUC: {m_10k['pr_auc']:.4f} | Hits@25%: {m_10k['hits_25']}/140 ({m_10k['recall_25']:.1f}%)")

    # Read v4 master results from previous benchmark
    results.append({
        "Category": "5. Modern Anchor",
        "Model Name": "v4 Specialist Meta-Blender (Winner)",
        "Date": "Nov 2 13:20",
        "Training Data": "Bumble 40k Pure (DINO+SigLIP+Geom)",
        "Architecture": "4-Pillar Meta-Blender (LogSumExp)",
        "roc_auc": 0.9147,
        "pr_auc": 0.5188,
        "hits_20": 113,
        "recall_20": 80.7,
        "prec_20": 36.8,
        "missed_20": 27,
        "hits_25": 125,
        "recall_25": 89.3,
        "prec_25": 32.6,
        "missed_25": 15,
        "prec_100": 63.0,
    })

    # Save to CSV
    df_results = pd.DataFrame(results)
    out_csv = "results/benchmark_pre_10k_legacy_models_on_val_1582.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved complete benchmark report to {out_csv}")


if __name__ == "__main__":
    main()
