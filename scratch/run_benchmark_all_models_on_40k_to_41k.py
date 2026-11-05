import os
import sys
import time
import subprocess
from pathlib import Path

# Lock caches and temp to D: drive
os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import cv2
import torch
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store
from scratch.build_siglip_portrait_crop_store import extract_portrait_crop_384
from scratch.train_dual_crop_subcenter_pauc import DualCropOrthogonalFusionNet, predict_dual_crop, predict_visual_logits

root = Path(__file__).resolve().parent.parent
python_clip = r"D:\BumbleClawClipVenv\Scripts\python.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_40k_to_41k():
    print("=" * 80)
    print("STEP 1: FEATURE EXTRACTION FOR 40K TO 41K UNTRAINED SLICE")
    print("=" * 80)

    csv_full = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
    csv_slice = Path(r"D:\BumbleLog\binary_swipe_labels_40k_to_41k.csv")
    ref_store_path = root / "embeddings" / "reference_store_bumble_40k_to_41k.npz"
    clip_store_path = root / "embeddings" / "clip_store_bumble_40k_to_41k.npz"
    siglip_store_path = root / "embeddings" / "siglip_store_bumble_40k_to_41k.npz"
    portrait_store_path = root / "embeddings" / "siglip_portrait_store_bumble_40k_to_41k.npz"

    # Slice CSV
    df = pd.read_csv(csv_full)
    df_new = df.iloc[40086:].copy()
    df_new.to_csv(csv_slice, index=False)
    print(f"Created {csv_slice} with {len(df_new)} rows (Likes: {(df_new['rating'] >= 50).sum()}, Passes: {(df_new['rating'] < 50).sum()}).")

    # ArcFace Reference Store
    if not ref_store_path.exists():
        print(f"\n[1/4] Extracting InsightFace embeddings -> {ref_store_path}...")
        cmd_ref = [
            python_clip, "build_references.py",
            "--labels", str(csv_slice),
            "--output", str(ref_store_path),
            "--provider", "cuda"
        ]
        subprocess.run(cmd_ref, check=True)
        print(f"[OK] ArcFace store saved.")
    else:
        print(f"[EXISTS] {ref_store_path}")

    # CLIP ViT-B/32 Store
    if not clip_store_path.exists():
        print(f"\n[2/4] Extracting CLIP ViT-B/32 -> {clip_store_path}...")
        cmd_clip = [
            python_clip, "build_clip_store.py",
            "--labels", str(csv_slice),
            "--store", str(ref_store_path),
            "--output", str(clip_store_path),
            "--batch-size", "64",
            "--provider", "cuda"
        ]
        subprocess.run(cmd_clip, check=True)
        print(f"[OK] CLIP store saved.")
    else:
        print(f"[EXISTS] {clip_store_path}")

    # SigLIP-384 Full Scene Store
    if not siglip_store_path.exists():
        print(f"\n[3/4] Extracting SigLIP-384 Full Scene -> {siglip_store_path}...")
        cmd_siglip = [
            python_clip, "build_siglip_store.py",
            "--store", str(ref_store_path),
            "--output", str(siglip_store_path),
            "--batch-size", "32",
            "--workers", "8"
        ]
        subprocess.run(cmd_siglip, check=True)
        print(f"[OK] SigLIP Full store saved.")
    else:
        print(f"[EXISTS] {siglip_store_path}")

    # SigLIP-384 Portrait Store
    if not portrait_store_path.exists():
        print(f"\n[4/4] Extracting SigLIP-384 Portrait Crops -> {portrait_store_path}...")
        from transformers import SiglipVisionModel
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        siglip = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            cache_dir=r"D:\BumbleClawHFCache",
            torch_dtype=torch.float16
        ).to(device)
        siglip.eval()

        ref_store = load_store(str(ref_store_path))
        paths = ref_store.paths
        n_total = len(paths)
        batch_size = 32

        all_embs = []
        all_face_valids = []

        for i in range(0, n_total, batch_size):
            batch_paths = paths[i : min(i + batch_size, n_total)]
            batch_crops = []
            batch_valids = []

            for p in batch_paths:
                img = cv2.imread(p)
                if img is None:
                    dummy = np.zeros((3, 384, 384), dtype=np.float32)
                    batch_crops.append(dummy)
                    batch_valids.append(0.0)
                    continue

                try:
                    bboxes, _ = app.det_model.detect(img, max_num=1)
                    bbox = bboxes[0][:4].tolist() if bboxes is not None and len(bboxes) > 0 else None
                except Exception:
                    bbox = None

                crop_arr, f_valid = extract_portrait_crop_384(img, bbox)
                batch_crops.append(crop_arr)
                batch_valids.append(f_valid)

            batch_np = np.stack(batch_crops, axis=0)
            with torch.no_grad():
                pixel_values = torch.tensor(batch_np, dtype=torch.float16, device=device)
                out = siglip(pixel_values=pixel_values).pooler_output
                out = out / out.norm(dim=-1, keepdim=True)
                batch_emb = out.cpu().to(torch.float32).numpy()

            all_embs.append(batch_emb)
            all_face_valids.extend(batch_valids)

        all_embs = np.concatenate(all_embs, axis=0)
        all_face_valids = np.array(all_face_valids, dtype=np.float32).reshape(-1, 1)

        np.savez_compressed(
            portrait_store_path,
            embeddings=all_embs,
            face_valid=all_face_valids,
            paths=np.array(paths),
            model_name="google/siglip-so400m-patch14-384"
        )
        print(f"[OK] Portrait store saved: {portrait_store_path}")
    else:
        print(f"[EXISTS] {portrait_store_path}")


def evaluate_metrics(preds, y_true):
    n_total = len(y_true)
    n_likes = int(np.sum(y_true))
    roc = roc_auc_score(y_true, preds)
    p_curve, r_curve, _ = precision_recall_curve(y_true, preds)
    pr_auc = auc(r_curve, p_curve)

    sorted_idx = np.argsort(preds)[::-1]
    sorted_y = y_true[sorted_idx]

    # Likes @ 20% Cap
    k_20 = int(np.round(0.20 * n_total))
    likes_20 = int(np.sum(sorted_y[:k_20]))
    rec_20 = likes_20 / n_likes * 100
    hit_rate_20 = likes_20 / k_20 * 100

    # Likes @ 25% Cap
    k_25 = int(np.round(0.25 * n_total))
    likes_25 = int(np.sum(sorted_y[:k_25]))
    rec_25 = likes_25 / n_likes * 100
    hit_rate_25 = likes_25 / k_25 * 100

    # Swipe Rate @ 90% and 95%
    cum_likes = np.cumsum(sorted_y)
    target_90 = 0.90 * n_likes
    idx_90 = np.where(cum_likes >= target_90)[0]
    sr_90 = (idx_90[0] + 1) / n_total * 100 if len(idx_90) > 0 else 100.0

    target_95 = 0.95 * n_likes
    idx_95 = np.where(cum_likes >= target_95)[0]
    sr_95 = (idx_95[0] + 1) / n_total * 100 if len(idx_95) > 0 else 100.0

    return {
        "ROC-AUC": roc,
        "PR-AUC": pr_auc,
        "Likes@20%Cap": f"{likes_20}/{n_likes} ({rec_20:.1f}%)",
        "HitRate@20%": f"{hit_rate_20:.1f}%",
        "Likes@25%Cap": f"{likes_25}/{n_likes} ({rec_25:.1f}%)",
        "HitRate@25%": f"{hit_rate_25:.1f}%",
        "Likes_25_int": likes_25,
        "SwipeRate@90%": f"{sr_90:.1f}%",
        "SwipeRate@95%": f"{sr_95:.1f}%",
    }


def to_rank_norm(arr):
    r = rankdata(arr) / (len(arr) + 1.0)
    r = np.clip(r, 1e-4, 1.0 - 1e-4)
    return norm.ppf(r)


def main():
    # 1. Feature Extraction
    extract_40k_to_41k()

    # 2. Load Features
    print("\n" + "=" * 80)
    print("STEP 2: LOADING 40K-41K FEATURES FOR ALL-MODEL TOURNAMENT")
    print("=" * 80)

    r_test = load_store(str(root / "embeddings" / "reference_store_bumble_40k_to_41k.npz"))
    c_test = np.load(str(root / "embeddings" / "clip_store_bumble_40k_to_41k.npz"))['embeddings']
    s_test = np.load(str(root / "embeddings" / "siglip_store_bumble_40k_to_41k.npz"))['embeddings']
    p_test = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_40k_to_41k.npz"))

    y_test = (r_test.ratings >= 50.0).astype(int)
    X_fa = r_test.embeddings
    X_cl = c_test
    X_fu = s_test
    X_po = p_test['embeddings']
    FV = p_test['face_valid']

    X_mlp_512 = X_fa # for 10k MLP if 512D
    X_mlp_1024 = np.hstack([X_fa, X_cl]) # for 20k/30k/40k MLP
    X_dc_mlp = np.hstack([X_fu, X_po, X_fa]) # for 40k Dual-Crop MLP

    N_test = len(y_test)
    N_likes = int(y_test.sum())
    print(f"Untrained 40k-41k Slice: {N_test} profiles | Likes: {N_likes} ({N_likes/N_test*100:.2f}%)")
    print(f"20% Swipe Budget: {int(np.round(0.20 * N_test))} swipes | 25% Swipe Budget: {int(np.round(0.25 * N_test))} swipes\n")

    # Helper function to safely load joblib estimator
    def load_joblib_model(path):
        obj = joblib.load(path)
        return obj['estimator'] if isinstance(obj, dict) else obj

    predictions = {}

    # --- 10k Generation ---
    try:
        m_r10 = load_joblib_model(root / "models" / "rating_regressor_bumble_10k_pure.joblib")
        predictions["10k Pure Ridge Face"] = m_r10.predict(X_fa)
    except Exception as e:
        print(f"10k Ridge load error: {e}")

    try:
        m_m10 = load_joblib_model(root / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib")
        feat_10 = X_mlp_1024 if m_m10.n_features_in_ == 1024 else X_fa
        predictions["10k Pure Multimodal MLP"] = m_m10.predict(feat_10)
    except Exception as e:
        print(f"10k MLP load error: {e}")

    try:
        ckpt_pauc_10k = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_10k.pt"
        if ckpt_pauc_10k.exists():
            predictions["10k Single-Crop SigLIP pAUC"] = predict_visual_logits(str(ckpt_pauc_10k), X_fu, X_fa)
    except Exception as e:
        print(f"10k pAUC load error: {e}")

    # --- 20k Generation ---
    try:
        m_r20 = load_joblib_model(root / "models" / "rating_regressor_bumble_20k_pure.joblib")
        predictions["20k Pure Ridge Face"] = m_r20.predict(X_fa)
    except Exception as e:
        print(f"20k Ridge load error: {e}")

    try:
        m_m20 = load_joblib_model(root / "models" / "rating_regressor_multimodal_bumble_20k_pure.joblib")
        feat_20 = X_mlp_1024 if m_m20.n_features_in_ == 1024 else X_fa
        predictions["20k Pure Multimodal MLP"] = m_m20.predict(feat_20)
    except Exception as e:
        print(f"20k MLP load error: {e}")

    try:
        ckpt_pauc_20k = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_20k.pt"
        if ckpt_pauc_20k.exists():
            predictions["20k Single-Crop SigLIP pAUC"] = predict_visual_logits(str(ckpt_pauc_20k), X_fu, X_fa)
    except Exception as e:
        print(f"20k pAUC load error: {e}")

    # --- 30k Generation ---
    try:
        m_r30 = load_joblib_model(root / "models" / "rating_regressor_bumble_30k_pure.joblib")
        predictions["30k Pure Ridge Face"] = m_r30.predict(X_fa)
    except Exception as e:
        print(f"30k Ridge load error: {e}")

    try:
        m_m30 = load_joblib_model(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
        predictions["30k Pure Multimodal MLP"] = m_m30.predict(X_mlp_1024)
    except Exception as e:
        print(f"30k MLP load error: {e}")

    try:
        ckpt_pauc_sc30 = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt"
        predictions["30k Single-Crop SigLIP pAUC"] = predict_visual_logits(str(ckpt_pauc_sc30), X_fu, X_fa)
    except Exception as e:
        print(f"30k Single-Crop pAUC load error: {e}")

    try:
        ckpt_pauc_dc30 = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_30k_full.pt"
        predictions["30k Dual-Crop SigLIP pAUC"] = predict_dual_crop(str(ckpt_pauc_dc30), X_fu, X_po, X_fa, FV)
    except Exception as e:
        print(f"30k Dual-Crop pAUC load error: {e}")

    # 30k Grand Super-Ensemble (37.5 / 25 / 37.5)
    if "30k Pure Multimodal MLP" in predictions and "30k Single-Crop SigLIP pAUC" in predictions and "30k Dual-Crop SigLIP pAUC" in predictions:
        r_m30 = rankdata(predictions["30k Pure Multimodal MLP"]) / N_test
        r_sc30 = rankdata(predictions["30k Single-Crop SigLIP pAUC"]) / N_test
        r_dc30 = rankdata(predictions["30k Dual-Crop SigLIP pAUC"]) / N_test
        predictions["30k Grand Super-Ensemble (37.5/25/37.5)"] = 0.375 * r_m30 + 0.250 * r_sc30 + 0.375 * r_dc30

    # --- 40k Generation (The 4 Pillars) ---
    try:
        m_m40 = load_joblib_model(root / "models" / "rating_regressor_multimodal_bumble_40k.joblib")
        predictions["40k Pillar 1: Single-Crop Multimodal MLP"] = m_m40.predict(X_mlp_1024)
    except Exception as e:
        print(f"40k MLP load error: {e}")

    try:
        m_dc_mlp40 = load_joblib_model(root / "models" / "rating_regressor_dual_crop_mlp_40k.joblib")
        predictions["40k Pillar 2: Dual-Crop SigLIP MLP"] = m_dc_mlp40.predict(X_dc_mlp)
    except Exception as e:
        print(f"40k DC MLP load error: {e}")

    try:
        ckpt_pauc_sc40 = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_40k.pt"
        predictions["40k Pillar 3: Single-Crop Regularized pAUC"] = predict_visual_logits(str(ckpt_pauc_sc40), X_fu, X_fa)
    except Exception as e:
        print(f"40k Single-Crop pAUC load error: {e}")

    try:
        ckpt_pauc_dc40 = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_40k.pt"
        predictions["40k Pillar 4: Dual-Crop Regularized pAUC"] = predict_dual_crop(str(ckpt_pauc_dc40), X_fu, X_po, X_fa, FV)
    except Exception as e:
        print(f"40k Dual-Crop pAUC load error: {e}")

    # 40k Ensembles
    p1 = predictions.get("40k Pillar 1: Single-Crop Multimodal MLP")
    p2 = predictions.get("40k Pillar 2: Dual-Crop SigLIP MLP")
    p3 = predictions.get("40k Pillar 3: Single-Crop Regularized pAUC")
    p4 = predictions.get("40k Pillar 4: Dual-Crop Regularized pAUC")

    if p1 is not None and p2 is not None and p3 is not None and p4 is not None:
        # Equal Rank Norm Quad
        rn1, rn2, rn3, rn4 = to_rank_norm(p1), to_rank_norm(p2), to_rank_norm(p3), to_rank_norm(p4)
        predictions["40k Quad Ensemble (Equal Rank Norm)"] = (rn1 + rn2 + rn3 + rn4) / 4.0

        # Meta-Blended Ensemble using learned weights:
        # 1. Single-Crop MLP: +0.1404
        # 2. Dual-Crop MLP:   +0.8868
        # 3. Single-Crop pAUC:+0.3079
        # 4. Dual-Crop pAUC:  +0.7871
        predictions["40k Grand Super-Ensemble (Meta-Blended)"] = (
            0.1404 * rn1 + 0.8868 * rn2 + 0.3079 * rn3 + 0.7871 * rn4
        )

    # 3. Benchmark Evaluation
    print("\n" + "=" * 80)
    print("STEP 3: TOURNAMENT LEADERBOARD ON UNTRAINED 40K-41K SLICE")
    print("=" * 80)

    results = []
    for name, p in predictions.items():
        m = evaluate_metrics(p, y_test)
        results.append({"Model": name, **m})

    df_res = pd.DataFrame(results)
    # Sort by Likes@25%Cap descending, then PR-AUC descending
    df_res = df_res.sort_values(by=["Likes_25_int", "PR-AUC"], ascending=[False, False])

    print(f"\n{'Model Architecture':<46} | {'Likes@20%Cap':<17} | {'Likes@25%Cap':<17} | {'HitRate@25%':<11} | {'PR-AUC':<7} | {'ROC':<7} | {'Swipe@90%'}")
    print("-" * 125)
    for _, r in df_res.iterrows():
        print(f"{r['Model']:<46} | {r['Likes@20%Cap']:<17} | {r['Likes@25%Cap']:<17} | {r['HitRate@25%']:<11} | {r['PR-AUC']:.4f}  | {r['ROC-AUC']:.4f} | {r['SwipeRate@90%']}")

    out_csv = root / "results" / "benchmark_all_models_on_40k_to_41k.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[COMPLETE] Saved all-models leaderboard to {out_csv}")


if __name__ == "__main__":
    main()
