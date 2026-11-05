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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import rankdata, norm

from face_similarity.store import load_store, save_store
from scratch.build_siglip_portrait_crop_store import extract_portrait_crop_384
from scratch.train_dual_crop_subcenter_pauc import DualCropOrthogonalFusionNet, predict_dual_crop, predict_visual_logits

root = Path(__file__).resolve().parent.parent
python_clip = r"D:\BumbleClawClipVenv\Scripts\python.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step1_extract_33k_to_40k():
    print("=" * 80)
    print("STEP 1: BATCH FEATURE EXTRACTION FOR 33k TO 40k SLICE")
    print("=" * 80)

    csv_full = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
    csv_slice = Path(r"D:\BumbleLog\binary_swipe_labels_33k_to_40k.csv")
    ref_store_path = root / "embeddings" / "reference_store_bumble_33k_to_40k.npz"
    clip_store_path = root / "embeddings" / "clip_store_bumble_33k_to_40k.npz"
    siglip_store_path = root / "embeddings" / "siglip_store_bumble_33k_to_40k.npz"
    portrait_store_path = root / "embeddings" / "siglip_portrait_store_bumble_33k_to_40k.npz"

    # 1.1 Create Slice CSV if needed
    if not csv_slice.exists():
        df = pd.read_csv(csv_full)
        df_new = df.iloc[33629:40086].copy()
        df_new.to_csv(csv_slice, index=False)
        print(f"Created {csv_slice} with {len(df_new)} rows.")
    else:
        print(f"[EXISTS] {csv_slice}")

    # 1.2 ArcFace Reference Store
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

    # 1.3 CLIP ViT-B/32 Store
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

    # 1.4 SigLIP-384 Full Scene Store
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

    # 1.5 SigLIP-384 1.8x Portrait Crop Store
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
        t0 = time.time()

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

            if (i + batch_size) % 320 == 0 or (i + batch_size) >= n_total:
                done = min(i + batch_size, n_total)
                print(f"Portrait extraction: {done}/{n_total} ({done/n_total*100:.1f}%) | Valid faces: {sum(all_face_valids):.0f}/{done}")

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


def main():
    # 1. Feature Extraction
    step1_extract_33k_to_40k()

    # 2. Load and Consolidate all stores
    print("\n" + "=" * 80)
    print("STEP 2: CONSOLIDATING UNIFIED 40K EMBEDDINGS & DEFINING PARTITIONS")
    print("=" * 80)

    # Load 30k pure (29,582)
    r_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    c_30k = np.load(str(root / "embeddings" / "clip_store_bumble_30k_pure.npz"))['embeddings']
    s_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))['embeddings']
    p_30k = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_pure.npz"))
    fv_30k = p_30k['face_valid']
    port_30k = p_30k['embeddings']

    # Load Holdout 2 (3,481)
    r_h2 = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    c_h2 = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))['embeddings']
    s_h2 = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))['embeddings']
    p_h2 = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_to_33k.npz"))
    fv_h2 = p_h2['face_valid']
    port_h2 = p_h2['embeddings']

    # Load New 33k-40k slice
    r_new = load_store(str(root / "embeddings" / "reference_store_bumble_33k_to_40k.npz"))
    c_new = np.load(str(root / "embeddings" / "clip_store_bumble_33k_to_40k.npz"))['embeddings']
    s_new = np.load(str(root / "embeddings" / "siglip_store_bumble_33k_to_40k.npz"))['embeddings']
    p_new = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_33k_to_40k.npz"))
    fv_new = p_new['face_valid']
    port_new = p_new['embeddings']

    print(f"Loaded Slice 1 (0-30k pure): {len(r_30k.paths)} samples")
    print(f"Loaded Slice 2 (Holdout 2):  {len(r_h2.paths)} samples")
    print(f"Loaded Slice 3 (33k to 40k): {len(r_new.paths)} samples")

    # Combine into unified dataset
    paths_all = np.concatenate([r_30k.paths, r_h2.paths, r_new.paths])
    ratings_all = np.concatenate([r_30k.ratings, r_h2.ratings, r_new.ratings])
    y_all = (ratings_all >= 50.0).astype(int)

    X_fa_all = np.concatenate([r_30k.embeddings, r_h2.embeddings, r_new.embeddings], axis=0)
    X_cl_all = np.concatenate([c_30k, c_h2, c_new], axis=0)
    X_fu_all = np.concatenate([s_30k, s_h2, s_new], axis=0)
    X_po_all = np.concatenate([port_30k, port_h2, port_new], axis=0)
    FV_all = np.concatenate([fv_30k, fv_h2, fv_new], axis=0)

    N_total = len(y_all)
    total_likes = int(y_all.sum())
    print(f"Unified 40k Dataset: {N_total} profiles | Total Likes: {total_likes} ({total_likes/N_total*100:.2f}%)")

    # Partitions:
    # Train: 0 to 33,000 (all of 30k pure + Holdout 2)
    # Val: 33,000 to 36,000 (for early stopping & meta-blending calibration)
    # Locked Holdout 3: 36,000 to N_total (untouched out-of-sample benchmark)
    idx_train_end = len(r_30k.paths) + len(r_h2.paths) # 29,582 + 3,481 = 33,063
    n_new = len(r_new.paths)
    val_size = int(np.round(0.45 * n_new)) # ~2,900 profiles for val
    idx_val_end = idx_train_end + val_size # ~36,000

    print(f"\nExact Partition Slices:")
    print(f"  Train Split:        [0 : {idx_train_end}] ({idx_train_end} profiles, {y_all[:idx_train_end].sum()} likes)")
    print(f"  Val Split:          [{idx_train_end} : {idx_val_end}] ({idx_val_end - idx_train_end} profiles, {y_all[idx_train_end:idx_val_end].sum()} likes)")
    print(f"  Locked Holdout 3:   [{idx_val_end} : {N_total}] ({N_total - idx_val_end} profiles, {y_all[idx_val_end:].sum()} likes)")
    print(f"  Full New 33k-40k:   [{idx_train_end} : {N_total}] ({N_total - idx_train_end} profiles, {y_all[idx_train_end:].sum()} likes)")

    # 3. Strategy 1 Audit (Zero-Shot Verification on New Slices)
    print("\n" + "=" * 80)
    print("STEP 3: STRATEGY 1 AUDIT - ZERO-SHOT VERIFICATION OF FROZEN 30K MODELS")
    print("=" * 80)

    # Load 30k models
    mlp_30k_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    mlp_30k = mlp_30k_obj['estimator'] if isinstance(mlp_30k_obj, dict) else mlp_30k_obj

    sc_30k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt"
    dc_30k_path = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_30k_full.pt"

    # Evaluate on:
    # A) Full New 33k-40k Slice
    # B) Locked Holdout 3
    test_slices = {
        "Full New 33k-40k Slice": (slice(idx_train_end, N_total), y_all[idx_train_end:N_total]),
        "Locked Holdout 3 (36k-40k)": (slice(idx_val_end, N_total), y_all[idx_val_end:N_total])
    }

    strat1_results = []
    for s_name, (s_idx, s_y) in test_slices.items():
        print(f"\n--- Strategy 1 Zero-Shot Leaderboard: {s_name} ({len(s_y)} profiles, {s_y.sum()} likes) ---")
        X_fa_s = X_fa_all[s_idx]
        X_cl_s = X_cl_all[s_idx]
        X_fu_s = X_fu_all[s_idx]
        X_po_s = X_po_all[s_idx]
        FV_s = FV_all[s_idx]
        X_mlp_s = np.hstack([X_fa_s, X_cl_s])

        p_mlp = mlp_30k.predict(X_mlp_s)
        p_sc = predict_visual_logits(str(sc_30k_path), X_fu_s, X_fa_s)
        p_dc = predict_dual_crop(str(dc_30k_path), X_fu_s, X_po_s, X_fa_s, FV_s)

        r_mlp = rankdata(p_mlp) / len(s_y)
        r_sc = rankdata(p_sc) / len(s_y)
        r_dc = rankdata(p_dc) / len(s_y)

        # 30k Grand Super-Ensemble (37.5 / 25 / 37.5)
        p_grand = 0.375 * r_mlp + 0.250 * r_sc + 0.375 * r_dc

        models_eval = {
            "30k Multimodal MLP (Pure)": p_mlp,
            "30k Single-Crop SigLIP pAUC": p_sc,
            "30k Dual-Crop SigLIP pAUC": p_dc,
            "30k Grand Super-Ensemble (37.5/25/37.5)": p_grand
        }

        for m_name, m_preds in models_eval.items():
            met = evaluate_metrics(m_preds, s_y)
            strat1_results.append({"Evaluation Set": s_name, "Model": m_name, **met})
            print(f"{m_name:<45} | Likes@25%: {met['Likes@25%Cap']:<18} | PR-AUC: {met['PR-AUC']:.4f} | ROC: {met['ROC-AUC']:.4f}")

    df_s1 = pd.DataFrame(strat1_results)
    csv_s1 = root / "results" / "strategy1_zero_shot_40k_benchmark.csv"
    csv_s1.parent.mkdir(parents=True, exist_ok=True)
    df_s1.to_csv(csv_s1, index=False)
    print(f"\nSaved Strategy 1 benchmark to {csv_s1}")

    # 4. Strategy 2 Training (4-Pillar Grand Super-Ensemble on 40k)
    print("\n" + "=" * 80)
    print("STEP 4: STRATEGY 2 - TRAINING 40K 4-PILLAR GRAND SUPER-ENSEMBLE")
    print("=" * 80)

    # Recency Sample Weighting (Upgrade 1)
    # 0-10k: 0.60, 10k-20k: 0.85, 20k-30k: 1.00, 30k-33.5k: 1.15
    w_recency = np.ones(idx_train_end, dtype=np.float32)
    w_recency[:9800] = 0.60
    w_recency[9800:19600] = 0.85
    w_recency[19600:29582] = 1.00
    w_recency[29582:idx_train_end] = 1.15

    # Data slices for training and validation
    X_fa_tr = X_fa_all[:idx_train_end]
    X_cl_tr = X_cl_all[:idx_train_end]
    X_fu_tr = X_fu_all[:idx_train_end]
    X_po_tr = X_po_all[:idx_train_end]
    FV_tr = FV_all[:idx_train_end]
    y_tr = y_all[:idx_train_end]

    X_fa_v = X_fa_all[idx_train_end:idx_val_end]
    X_cl_v = X_cl_all[idx_train_end:idx_val_end]
    X_fu_v = X_fu_all[idx_train_end:idx_val_end]
    X_po_v = X_po_all[idx_train_end:idx_val_end]
    FV_v = FV_all[idx_train_end:idx_val_end]
    y_v = y_all[idx_train_end:idx_val_end]

    X_fa_te = X_fa_all[idx_val_end:]
    X_cl_te = X_cl_all[idx_val_end:]
    X_fu_te = X_fu_all[idx_val_end:]
    X_po_te = X_po_all[idx_val_end:]
    FV_te = FV_all[idx_val_end:]
    y_te = y_all[idx_val_end:]

    # PILLAR 1: Single-Crop Multimodal MLP (40k)
    print("\n--- Training Pillar 1: Single-Crop Multimodal MLP (40k) ---")
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_mlp_tr = np.hstack([X_fa_tr, X_cl_tr])
    X_mlp_v = np.hstack([X_fa_v, X_cl_v])
    X_mlp_te = np.hstack([X_fa_te, X_cl_te])

    pipe_mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), activation='relu', max_iter=25, random_state=42, early_stopping=True))
    ])
    pipe_mlp.fit(X_mlp_tr, y_tr)
    preds_mlp_v = pipe_mlp.predict(X_mlp_v)
    preds_mlp_te = pipe_mlp.predict(X_mlp_te)
    joblib.dump({'estimator': pipe_mlp}, root / "models" / "rating_regressor_multimodal_bumble_40k.joblib")
    print(f"Pillar 1 Val ROC: {roc_auc_score(y_v, preds_mlp_v):.4f}")

    # PILLAR 2: Dual-Crop SigLIP Multimodal MLP (40k) (Upgrade 2)
    print("\n--- Training Pillar 2: Dual-Crop SigLIP Multimodal MLP (40k) ---")
    X_dc_tr = np.hstack([X_fu_tr, X_po_tr, X_fa_tr])
    X_dc_v = np.hstack([X_fu_v, X_po_v, X_fa_v])
    X_dc_te = np.hstack([X_fu_te, X_po_te, X_fa_te])

    pipe_dc_mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(384, 96), activation='relu', max_iter=20, random_state=42, early_stopping=True))
    ])
    pipe_dc_mlp.fit(X_dc_tr, y_tr)
    preds_dc_mlp_v = pipe_dc_mlp.predict(X_dc_v)
    preds_dc_mlp_te = pipe_dc_mlp.predict(X_dc_te)
    joblib.dump({'estimator': pipe_dc_mlp}, root / "models" / "rating_regressor_dual_crop_mlp_40k.joblib")
    print(f"Pillar 2 Val ROC: {roc_auc_score(y_v, preds_dc_mlp_v):.4f}")

    # PILLAR 3: Single-Crop Regularized pAUC Subcenter (40k)
    print("\n--- Training Pillar 3: Single-Crop Regularized pAUC Subcenter (40k) ---")
    # For speed and optimal scaling, we use the proven subcenter training function
    from scratch.train_subcenter_pauc_regularized import train_regularized_subcenter, evaluate_metrics as eval_m
    C_tr = np.zeros((len(y_tr), 4), dtype=np.float32)
    C_v = np.zeros((len(y_v), 4), dtype=np.float32)

    save_p3 = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_40k.pt"
    train_regularized_subcenter(
        X_fu_tr, X_fa_tr, y_tr, C_tr,
        X_fu_v, X_fa_v, y_v, C_v,
        K=16, M=32, epochs=8, batch_size=512,
        save_path=str(save_p3)
    )
    preds_p3_v = predict_visual_logits(str(save_p3), X_fu_v, X_fa_v)
    preds_p3_te = predict_visual_logits(str(save_p3), X_fu_te, X_fa_te)

    # PILLAR 4: Dual-Crop Regularized pAUC Subcenter (40k)
    print("\n--- Training Pillar 4: Dual-Crop Regularized pAUC Subcenter (40k) ---")
    from scratch.train_dual_crop_subcenter_pauc import train_dual_crop_model
    save_p4 = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_40k.pt"
    train_dual_crop_model(
        X_fu_tr, X_po_tr, X_fa_tr, FV_tr, y_tr, C_tr,
        X_fu_v, X_po_v, X_fa_v, FV_v, y_v, C_v,
        K=16, M=32, epochs=6, batch_size=512,
        save_path=str(save_p4)
    )
    preds_p4_v = predict_dual_crop(str(save_p4), X_fu_v, X_po_v, X_fa_v, FV_v)
    preds_p4_te = predict_dual_crop(str(save_p4), X_fu_te, X_po_te, X_fa_te, FV_te)

    # 5. Out-of-Fold Rank-Logit Meta-Blender on Validation Split (Upgrade 3)
    print("\n" + "=" * 80)
    print("STEP 5: OUT-OF-FOLD RANK-LOGIT META-BLENDER CALIBRATION (UPGRADE 3)")
    print("=" * 80)

    # Gaussian Rank-Normal Quantile Transformation
    def to_rank_norm(arr):
        r = rankdata(arr) / (len(arr) + 1.0)
        # clip to prevent infinite values
        r = np.clip(r, 1e-4, 1.0 - 1e-4)
        return norm.ppf(r)

    Z_v = np.column_stack([
        to_rank_norm(preds_mlp_v),
        to_rank_norm(preds_dc_mlp_v),
        to_rank_norm(preds_p3_v),
        to_rank_norm(preds_p4_v)
    ])

    Z_te = np.column_stack([
        to_rank_norm(preds_mlp_te),
        to_rank_norm(preds_dc_mlp_te),
        to_rank_norm(preds_p3_te),
        to_rank_norm(preds_p4_te)
    ])

    # Fit L2 Regularized Logistic Regression CV on Validation Block
    meta_blender = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=500, random_state=42)
    meta_blender.fit(Z_v, y_v)
    print("Meta-Blender Pillar Coefficients:")
    pillar_names = ["1. Single-Crop MLP", "2. Dual-Crop MLP", "3. Single-Crop pAUC", "4. Dual-Crop pAUC"]
    for p_name, coef in zip(pillar_names, meta_blender.coef_[0]):
        print(f"  {p_name:<25}: {coef:+.4f}")

    preds_40k_grand_v = meta_blender.predict_proba(Z_v)[:, 1]
    preds_40k_grand_te = meta_blender.predict_proba(Z_te)[:, 1]

    # Equal Triad / Quad for baseline
    preds_quad_avg_te = (to_rank_norm(preds_mlp_te) + to_rank_norm(preds_dc_mlp_te) + to_rank_norm(preds_p3_te) + to_rank_norm(preds_p4_te)) / 4.0

    # 6. Final Tournament Benchmark on Locked Holdout 3 (36k to 40k)
    print("\n" + "=" * 80)
    print("STEP 6: FINAL TOURNAMENT BENCHMARK ON LOCKED HOLDOUT 3")
    print("=" * 80)
    print(f"Locked Holdout 3: {len(y_te)} profiles | Likes: {y_te.sum()} ({y_te.sum()/len(y_te)*100:.2f}%)")

    # Evaluate 30k Champion Grand Super-Ensemble on Locked Holdout 3
    p_mlp_30k_te = mlp_30k.predict(X_mlp_te)
    p_sc_30k_te = predict_visual_logits(str(sc_30k_path), X_fu_te, X_fa_te)
    p_dc_30k_te = predict_dual_crop(str(dc_30k_path), X_fu_te, X_po_te, X_fa_te, FV_te)
    r_mlp_30k_te = rankdata(p_mlp_30k_te) / len(y_te)
    r_sc_30k_te = rankdata(p_sc_30k_te) / len(y_te)
    r_dc_30k_te = rankdata(p_dc_30k_te) / len(y_te)
    p_30k_grand_te = 0.375 * r_mlp_30k_te + 0.250 * r_sc_30k_te + 0.375 * r_dc_30k_te

    final_models = {
        "30k Grand Super-Ensemble (Prior SOTA)": p_30k_grand_te,
        "40k Pillar 1: Single-Crop Multimodal MLP": preds_mlp_te,
        "40k Pillar 2: Dual-Crop SigLIP MLP": preds_dc_mlp_te,
        "40k Pillar 3: Single-Crop Regularized pAUC": preds_p3_te,
        "40k Pillar 4: Dual-Crop Regularized pAUC": preds_p4_te,
        "40k Quad Ensemble (Equal Rank Norm)": preds_quad_avg_te,
        "40k Grand Super-Ensemble (Meta-Blended)": preds_40k_grand_te
    }

    final_results = []
    print("\n--- Final Leaderboard on Locked Holdout 3 ---")
    for name, p in final_models.items():
        m = evaluate_metrics(p, y_te)
        final_results.append({"Model": name, **m})
        print(f"{name:<45} | Likes@25%: {m['Likes@25%Cap']:<18} | PR-AUC: {m['PR-AUC']:.4f} | ROC: {m['ROC-AUC']:.4f} | Swipe@90%: {m['SwipeRate@90%']}")

    df_final = pd.DataFrame(final_results)
    csv_final = root / "results" / "benchmark_40k_tournament_locked_holdout3.csv"
    df_final.to_csv(csv_final, index=False)
    print(f"\n[COMPLETE] Saved final 40k benchmark table to {csv_final}")


if __name__ == "__main__":
    main()
