from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"
os.environ["BUMBLECLAW_CLIP_ALLOW_DOWNLOAD"] = "1"

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    fbeta_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
)

from face_similarity.embedding import get_face_app, select_largest_face
from face_similarity.clip_embedding import get_clip_model
from face_similarity.regressor import load_regressor, predict_rating, predict_multimodal_rating
from face_similarity.store import load_store
from face_similarity.scoring import score_embedding
from face_similarity.preference import load_preference_model, preference_probability, features_from_prediction
from face_similarity.prediction import RatingPrediction
from face_similarity.multimodalx import score as multimodalx_score

def main() -> int:
    root_dir = Path(__file__).resolve().parent
    temp_dir = Path(r"D:\BumbleTemp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_path = temp_dir / "val_features_1899.npz"

    print("================================================================================")
    print("3-WAY BENCHMARK ON UNTRAINED 1,899 PROFILE HOLDOUT")
    print("Candidates:")
    print("  1. Bumble 10k Pure (Trained strictly on your real Bumble swipes)")
    print("  2. Legacy face_biased (50% Ridge + 50% Multimodal on 7k stock photos)")
    print("  3. Legacy multimodalx_original / Experimental3 (48% Ridge + 48% Multi + 4% P(like))")
    print("================================================================================")

    # 1. Load labels
    labels_file = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
    df = pd.read_csv(labels_file)
    df = df.drop_duplicates(subset=['filename'], keep='last').reset_index(drop=True)
    
    train_count = 10131
    val_df = df.iloc[train_count:].copy().reset_index(drop=True)
    n_val = len(val_df)
    print(f"Validation dataset: {n_val} profiles")
    
    val_labels = (val_df['action'] == 'right').astype(int).values
    val_paths = [Path(r"D:\BumbleLog\cropped") / fn for fn in val_df['filename']]
    
    n_likes = int(np.sum(val_labels == 1))
    n_passes = int(np.sum(val_labels == 0))
    base_rate = n_likes / n_val
    print(f"True Distribution: {n_likes} Likes ({base_rate*100:.2f}%), {n_passes} Passes ({(1-base_rate)*100:.2f}%)\n")

    # 2. Extract or Load Cached Embeddings
    if cache_path.exists():
        print(f"Loading cached validation embeddings from {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        face_embeddings = cached['face_embeddings']
        clip_embeddings = cached['clip_embeddings']
        valid_indices = cached['valid_indices']
    else:
        print("Extracting InsightFace & CLIP features on CUDA for 1,899 validation images...")
        app, active_provider = get_face_app(model_name="buffalo_l", provider="cuda", det_size=640, det_thresh=0.25)
        torch_mod, processor, clip_model, device = get_clip_model("openai/clip-vit-base-patch32", "cuda")
        import cv2

        valid_indices = []
        face_list = []
        clip_list = []

        for idx, p in enumerate(tqdm(val_paths, desc="Extracting features")):
            if not p.exists():
                continue
            
            # 1. Face
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            faces = app.get(img_bgr)
            if not faces:
                continue
            face = select_largest_face(faces)
            face_emb = np.asarray(face.embedding, dtype=np.float32)

            # 2. CLIP
            try:
                pil_img = Image.open(p).convert("RGB")
                inputs = processor(images=[pil_img], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch_mod.no_grad():
                    feats = clip_model.get_image_features(**inputs)
                    if hasattr(feats, "pooler_output"):
                        feats = feats.pooler_output
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                clip_emb = feats.detach().cpu().numpy().astype(np.float32)[0]
            except Exception as e:
                continue

            valid_indices.append(idx)
            face_list.append(face_emb)
            clip_list.append(clip_emb)

        face_embeddings = np.array(face_list, dtype=np.float32)
        clip_embeddings = np.array(clip_list, dtype=np.float32)
        valid_indices = np.array(valid_indices, dtype=np.int32)

        np.savez(cache_path, face_embeddings=face_embeddings, clip_embeddings=clip_embeddings, valid_indices=valid_indices)
        print(f"Cached {len(valid_indices)} extracted features to {cache_path}\n")

    y_eval = val_labels[valid_indices]
    print(f"Valid profiles with detected faces for evaluation: {len(y_eval)} / {n_val}")
    print(f"Eval ground truth: {int(np.sum(y_eval == 1))} Likes, {int(np.sum(y_eval == 0))} Passes\n")

    # 3. Model 1: Bumble 10k Pure
    print("--> Computing scores for Model 1: Bumble 10k Pure...")
    r_10k_face = load_regressor("models/rating_regressor_bumble_10k_pure.joblib")
    r_10k_multi = load_regressor("models/rating_regressor_multimodal_bumble_10k_pure.joblib")
    
    scores_10k = []
    for f_emb, c_emb in zip(face_embeddings, clip_embeddings):
        s_face = float(predict_rating(r_10k_face, f_emb))
        s_multi = float(predict_multimodal_rating(r_10k_multi, face_embedding=f_emb, clip_embedding=c_emb))
        blended = 0.44 * s_face + 0.56 * s_multi
        scores_10k.append(blended)
    scores_10k = np.array(scores_10k)

    # 4. Model 2: Legacy face_biased (7k stock)
    print("--> Computing scores for Model 2: Legacy face_biased...")
    r_legacy_face = load_regressor("models/rating_regressor.joblib")
    r_legacy_multi = load_regressor("models/rating_regressor_multimodal.joblib")

    scores_face_biased = []
    for f_emb, c_emb in zip(face_embeddings, clip_embeddings):
        s_face = float(predict_rating(r_legacy_face, f_emb))
        s_multi = float(predict_multimodal_rating(r_legacy_multi, face_embedding=f_emb, clip_embedding=c_emb))
        blended = 0.50 * s_face + 0.50 * s_multi
        scores_face_biased.append(blended)
    scores_face_biased = np.array(scores_face_biased)

    # 5. Model 3: Legacy multimodalx_original / Experimental3 (7k stock + P(like))
    print("--> Computing scores for Model 3: Legacy multimodalx_original (Experimental3)...")
    store_7k = load_store("embeddings/reference_store.npz")
    pref_model = load_preference_model("models/bumble_preference_classifier.joblib")

    scores_exp3 = []
    for f_emb, c_emb in zip(face_embeddings, clip_embeddings):
        s_face = float(predict_rating(r_legacy_face, f_emb))
        s_multi = float(predict_multimodal_rating(r_legacy_multi, face_embedding=f_emb, clip_embedding=c_emb))
        knn_res = score_embedding(f_emb, store_7k.embeddings, store_7k.ratings, k=11)
        knn_score = float(knn_res.rating)
        
        # P(like)
        base_pred = RatingPrediction(
            rating=0.5 * s_face + 0.5 * s_multi,
            method="multimodalx_original",
            face_rating=s_face,
            multimodal_rating=s_multi,
            knn_rating=knn_score
        )
        feats = features_from_prediction(
            base_pred,
            threshold=55.0,
            face_weight=0.50,
            regressor_path="models/rating_regressor.joblib",
            multimodal_regressor_path="models/rating_regressor_multimodal.joblib"
        )
        p_like = preference_probability(pref_model, feats)
        s_exp3 = multimodalx_score(s_face, s_multi, p_like, knn=knn_score, method="multimodalx_original")
        scores_exp3.append(s_exp3)
    scores_exp3 = np.array(scores_exp3)

    # 6. Evaluation & Threshold Calibration Function
    models = {
        "Bumble 10k Pure (My Real Swipes)": scores_10k,
        "Legacy face_biased (7k Stock)": scores_face_biased,
        "Legacy Experimental3 (48/48/4 MultiX)": scores_exp3,
    }

    print("\n" + "="*80)
    print("BENCHMARK RESULTS ACROSS ALL 1,899 UNSEEN HOLDOUT SWIPES")
    print("="*80)

    results_table = []

    for name, raw_scores in models.items():
        # PR-AUC & ROC-AUC (Threshold independent)
        prec_arr, rec_arr, _ = precision_recall_curve(y_eval, raw_scores)
        pr_auc = auc(rec_arr, prec_arr)
        roc_auc = roc_auc_score(y_eval, raw_scores)

        # 1. Optimal Threshold for Max Overall Accuracy
        best_acc = 0.0
        best_acc_thresh = 0.0
        best_acc_prec = 0.0
        best_acc_rec = 0.0
        best_acc_mcc = 0.0

        # 2. Optimal Threshold for F0.5 (Precision-Weighted)
        best_f05 = 0.0
        best_f05_thresh = 0.0
        best_f05_prec = 0.0
        best_f05_rec = 0.0
        best_f05_acc = 0.0

        # 3. Optimal Threshold for Balanced Accuracy
        best_bacc = 0.0
        best_bacc_thresh = 0.0

        # Sweep candidate thresholds
        cand_threshs = np.linspace(np.percentile(raw_scores, 1), np.percentile(raw_scores, 99), 300)
        for t in cand_threshs:
            preds = (raw_scores >= t).astype(int)
            acc = accuracy_score(y_eval, preds)
            bacc = balanced_accuracy_score(y_eval, preds)
            prec = precision_score(y_eval, preds, zero_division=0)
            rec = recall_score(y_eval, preds, zero_division=0)
            f05 = fbeta_score(y_eval, preds, beta=0.5, zero_division=0)
            mcc = matthews_corrcoef(y_eval, preds)

            if acc > best_acc:
                best_acc = acc
                best_acc_thresh = t
                best_acc_prec = prec
                best_acc_rec = rec
                best_acc_mcc = mcc

            if f05 > best_f05:
                best_f05 = f05
                best_f05_thresh = t
                best_f05_prec = prec
                best_f05_rec = rec
                best_f05_acc = acc

            if bacc > best_bacc:
                best_bacc = bacc
                best_bacc_thresh = t

        # Fixed Budget (Top 16.1% highest scores)
        cutoff_16_1 = np.percentile(raw_scores, 100 - (base_rate * 100))
        budget_preds = (raw_scores >= cutoff_16_1).astype(int)
        budget_prec = precision_score(y_eval, budget_preds, zero_division=0)
        budget_acc = accuracy_score(y_eval, budget_preds)

        results_table.append({
            "model": name,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "best_acc": best_acc,
            "best_acc_thresh": best_acc_thresh,
            "best_acc_prec": best_acc_prec,
            "best_acc_rec": best_acc_rec,
            "best_f05": best_f05,
            "best_f05_thresh": best_f05_thresh,
            "best_f05_prec": best_f05_prec,
            "best_f05_rec": best_f05_rec,
            "best_f05_acc": best_f05_acc,
            "best_bacc": best_bacc,
            "best_bacc_thresh": best_bacc_thresh,
            "budget_16_1_prec": budget_prec,
            "budget_16_1_acc": budget_acc,
        })

    for r in results_table:
        print(f"\nMODEL: {r['model']}")
        print(f"  * Ranking Discriminative Power:")
        print(f"      PR-AUC: {r['pr_auc']:.4f}  |  ROC-AUC: {r['roc_auc']:.4f}")
        print(f"  * At Max Overall Accuracy (Optimal Cutoff = {r['best_acc_thresh']:.2f}):")
        print(f"      Overall Accuracy: {r['best_acc']*100:.2f}%  (Precision: {r['best_acc_prec']*100:.1f}%, Recall: {r['best_acc_rec']*100:.1f}%)")
        print(f"  * At Max F0.5 Precision-Biased (Optimal Cutoff = {r['best_f05_thresh']:.2f}):")
        print(f"      F0.5 Score: {r['best_f05']:.4f}  (Precision: {r['best_f05_prec']*100:.1f}%, Recall: {r['best_f05_rec']*100:.1f}%, Accuracy: {r['best_f05_acc']*100:.2f}%)")
        print(f"  * At Natural 16.1% Daily Budget Cutoff:")
        print(f"      Match Precision@Top16%: {r['budget_16_1_prec']*100:.1f}%  |  Overall Accuracy: {r['budget_16_1_acc']*100:.2f}%")
        print(f"  * Max Balanced Accuracy: {r['best_bacc']*100:.2f}% (Cutoff = {r['best_bacc_thresh']:.2f})")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
