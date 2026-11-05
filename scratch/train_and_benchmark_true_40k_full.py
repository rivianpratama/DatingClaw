import os
import sys
import time
from pathlib import Path

# Lock caches to D: drive
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import rankdata, norm

from face_similarity.store import load_store
from scratch.train_dual_crop_subcenter_pauc import DualCropOrthogonalFusionNet, predict_dual_crop
from scratch.train_subcenter_pauc_regularized import (
    StratifiedSwipeSampler,
    ShiftInvariantTopQuartilePAUCLoss,
    compute_multi_proxy_logit_and_reg,
    predict_visual_logits
)

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_predictions(scores, y_true, model_name="Model"):
    N = len(y_true)
    total_pos = int(y_true.sum())
    
    # 20% cap & 25% cap
    k_20 = int(np.round(0.20 * N))
    k_25 = int(np.round(0.25 * N))

    ranks = np.argsort(-scores)
    top_20_idx = ranks[:k_20]
    top_25_idx = ranks[:k_25]

    hits_20 = int(y_true[top_20_idx].sum())
    hits_25 = int(y_true[top_25_idx].sum())

    recall_20 = hits_20 / total_pos if total_pos > 0 else 0
    precision_20 = hits_20 / k_20 if k_20 > 0 else 0

    recall_25 = hits_25 / total_pos if total_pos > 0 else 0
    precision_25 = hits_25 / k_25 if k_25 > 0 else 0

    roc = roc_auc_score(y_true, scores)
    p_curve, r_curve, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(r_curve, p_curve)

    return {
        "model": model_name,
        "hits_20": hits_20,
        "k_20": k_20,
        "prec_20": precision_20 * 100,
        "rec_20": recall_20 * 100,
        "hits_25": hits_25,
        "k_25": k_25,
        "prec_25": precision_25 * 100,
        "rec_25": recall_25 * 100,
        "pr_auc": pr_auc,
        "roc_auc": roc
    }

def main():
    print("=" * 80)
    print("TRAINING TRUE 40,000 FULL MASTER ENSEMBLE & BENCHMARKING ON 1,582 HOLDOUT")
    print("=" * 80)

    # 1. Load Training Data (0 to 40,000)
    print("\n[1/5] Loading and consolidating full 0..40,000 training features...")
    r_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    c_30k = np.load(str(root / "embeddings" / "clip_store_bumble_30k_pure.npz"))['embeddings']
    s_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))['embeddings']
    p_30k = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_pure.npz"))
    port_30k = p_30k['embeddings']
    fv_30k = p_30k['face_valid']

    r_h2 = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    c_h2 = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))['embeddings']
    s_h2 = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))['embeddings']
    p_h2 = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_to_33k.npz"))
    port_h2 = p_h2['embeddings']
    fv_h2 = p_h2['face_valid']

    r_new = load_store(str(root / "embeddings" / "reference_store_bumble_33k_to_40k.npz"))
    c_new = np.load(str(root / "embeddings" / "clip_store_bumble_33k_to_40k.npz"))['embeddings']
    s_new = np.load(str(root / "embeddings" / "siglip_store_bumble_33k_to_40k.npz"))['embeddings']
    p_new = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_33k_to_40k.npz"))
    port_new = p_new['embeddings']
    fv_new = p_new['face_valid']

    X_fa_train = np.concatenate([r_30k.embeddings, r_h2.embeddings, r_new.embeddings], axis=0)
    X_cl_train = np.concatenate([c_30k, c_h2, c_new], axis=0)
    X_fu_train = np.concatenate([s_30k, s_h2, s_new], axis=0)
    X_po_train = np.concatenate([port_30k, port_h2, port_new], axis=0)
    FV_train = np.concatenate([fv_30k, fv_h2, fv_new], axis=0)

    ratings_train = np.concatenate([r_30k.ratings, r_h2.ratings, r_new.ratings])
    y_train = (ratings_train >= 50.0).astype(int)

    N_train = len(y_train)
    n_train_likes = int(y_train.sum())
    print(f"True 40k Training Corpus: {N_train:,} profiles | {n_train_likes:,} Likes ({n_train_likes/N_train*100:.2f}%)")

    # 2. Load 1,582 Holdout Validation Set
    print("\n[2/5] Loading 1,582 Holdout Validation features...")
    r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
    c_val = np.load(str(root / "embeddings" / "clip_store_val_1582.npz"))['embeddings']
    s_val = np.load(str(root / "embeddings" / "siglip_store_val_1582.npz"))['embeddings']
    p_val = np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))

    X_fa_val = r_val.embeddings
    X_cl_val = c_val
    X_fu_val = s_val
    X_po_val = p_val['embeddings']
    FV_val = p_val['face_valid']
    y_val = (r_val.ratings >= 50.0).astype(int)

    N_val = len(y_val)
    n_val_likes = int(y_val.sum())
    print(f"Holdout Validation Corpus: {N_val:,} profiles | {n_val_likes:,} Likes ({n_val_likes/N_val*100:.2f}%)")

    # Normalize
    X_fa_train = X_fa_train / np.linalg.norm(X_fa_train, axis=1, keepdims=True)
    X_cl_train = X_cl_train / np.linalg.norm(X_cl_train, axis=1, keepdims=True)
    X_fu_train = X_fu_train / np.linalg.norm(X_fu_train, axis=1, keepdims=True)
    X_po_train = X_po_train / np.linalg.norm(X_po_train, axis=1, keepdims=True)

    X_fa_val = X_fa_val / np.linalg.norm(X_fa_val, axis=1, keepdims=True)
    X_cl_val = X_cl_val / np.linalg.norm(X_cl_val, axis=1, keepdims=True)
    X_fu_val = X_fu_val / np.linalg.norm(X_fu_val, axis=1, keepdims=True)
    X_po_val = X_po_val / np.linalg.norm(X_po_val, axis=1, keepdims=True)

    # Composite Features
    X_mlp_single_train = np.concatenate([X_fa_train, X_cl_train], axis=1) # 1024D
    X_mlp_single_val = np.concatenate([X_fa_val, X_cl_val], axis=1)

    X_mlp_dual_train = np.concatenate([X_fa_train, X_fu_train, X_po_train], axis=1) # 2816D
    X_mlp_dual_val = np.concatenate([X_fa_val, X_fu_val, X_po_val], axis=1)

    # 3. Train Models
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # --- Pillar 1: Single-Crop Multimodal MLP (1024D) on Full 40k ---
    p1_model_path = models_dir / "rating_regressor_multimodal_bumble_40k_full.joblib"
    if p1_model_path.exists():
        print(f"\n[EXISTS] Loading pre-trained Pillar 1 from {p1_model_path}...")
        pipe_p1 = joblib.load(p1_model_path)
    else:
        print("\n--- [3/5] Training Pillar 1: Single-Crop MLP (1024D) on Full 40,000 cards ---")
        pipe_p1 = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(256, 64),
                activation='relu',
                alpha=1e-4,
                max_iter=30,
                batch_size=256,
                random_state=42,
                verbose=False
            ))
        ])
        pipe_p1.fit(X_mlp_single_train, y_train)
        joblib.dump(pipe_p1, p1_model_path)
        print(f"Pillar 1 Trained & Saved -> {p1_model_path}")
    score_p1_val = pipe_p1.predict_proba(X_mlp_single_val)[:, 1]

    # --- Pillar 2: Dual-Crop SigLIP MLP (2816D) on Full 40k ---
    p2_model_path = models_dir / "rating_regressor_dual_crop_mlp_40k_full.joblib"
    if p2_model_path.exists():
        print(f"[EXISTS] Loading pre-trained Pillar 2 from {p2_model_path}...")
        pipe_p2 = joblib.load(p2_model_path)
    else:
        print("\n--- Training Pillar 2: Dual-Crop SigLIP MLP (2816D) on Full 40,000 cards ---")
        pipe_p2 = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(512, 128),
                activation='relu',
                alpha=1e-3,
                max_iter=30,
                batch_size=256,
                random_state=42,
                verbose=False
            ))
        ])
        pipe_p2.fit(X_mlp_dual_train, y_train)
        joblib.dump(pipe_p2, p2_model_path)
        print(f"Pillar 2 Trained & Saved -> {p2_model_path}")
    score_p2_val = pipe_p2.predict_proba(X_mlp_dual_val)[:, 1]

    # --- Pillar 3: Single-Crop Regularized pAUC Subcenter (512D) on Full 40k ---
    p3_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_40k_full.pt"
    if p3_path.exists():
        print(f"[EXISTS] Loading pre-trained Pillar 3 from {p3_path}...")
        ckpt_p3 = torch.load(p3_path, map_location=device)
        proxies_p3 = ckpt_p3['pos_proxies'].to(device)
        c_neg_p3 = ckpt_p3['neg_proxy'].to(device)
    else:
        print("\n--- Training Pillar 3: Single-Crop pAUC Subcenter (512D, 12 Proxies) on Full 40k ---")
        pos_idx = np.where(y_train == 1)[0]
        X_pos = X_fa_train[pos_idx]
        kmeans = KMeans(n_clusters=12, random_state=42, n_init=10).fit(X_pos)
        init_proxies = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        init_proxies = F.normalize(init_proxies, p=2, dim=1)

        neg_idx = np.where(y_train == 0)[0]
        X_neg = X_fa_train[neg_idx]
        c_neg = torch.tensor(X_neg.mean(axis=0, keepdims=True), dtype=torch.float32, device=device)
        c_neg = F.normalize(c_neg, p=2, dim=1)

        proxies_p3 = nn.Parameter(init_proxies.clone())
        c_neg_p3 = nn.Parameter(c_neg.clone(), requires_grad=False)
        opt_p3 = optim.AdamW([proxies_p3], lr=0.005, weight_decay=1e-4)

        for epoch in range(15):
            p_sub = np.random.choice(pos_idx, size=64, replace=False)
            n_sub = np.random.choice(neg_idx, size=192, replace=False)
            b_idx = np.concatenate([p_sub, n_sub])
            b_x = torch.tensor(X_fa_train[b_idx], dtype=torch.float32, device=device)
            b_y = torch.tensor(y_train[b_idx], dtype=torch.float32, device=device)

            opt_p3.zero_grad()
            p_norm = F.normalize(proxies_p3, p=2, dim=1)
            sim_pos = torch.matmul(b_x, p_norm.T)
            logits_pos = (1.0 / 20.0) * torch.logsumexp(20.0 * sim_pos, dim=1)
            sim_neg = torch.matmul(b_x, c_neg_p3.T).squeeze(-1)
            diff = logits_pos - sim_neg

            loss = F.binary_cross_entropy_with_logits(diff * 10.0, b_y)
            loss.backward()
            opt_p3.step()

        p3_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'pos_proxies': F.normalize(proxies_p3, p=2, dim=1).detach().cpu(),
            'neg_proxy': c_neg_p3.detach().cpu(),
            'gamma': 0.15,
            'tau': 20.0
        }, p3_path)
        print(f"Pillar 3 Saved -> {p3_path}")

    # Predict Pillar 3 on Val
    with torch.no_grad():
        v_x = torch.tensor(X_fa_val, dtype=torch.float32, device=device)
        v_p = F.normalize(proxies_p3, p=2, dim=1)
        v_sim = torch.matmul(v_x, v_p.T)
        v_l = (1.0 / 20.0) * torch.logsumexp(20.0 * v_sim, dim=1)
        v_n = torch.matmul(v_x, c_neg_p3.T).squeeze(-1)
        score_p3_val = (v_l - v_n).cpu().numpy()

    # --- Pillar 4: Dual-Crop Gated pAUC Subcenter (1664D) on Full 40k ---
    p4_path = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_40k_full.pt"
    model_dc = DualCropOrthogonalFusionNet(sig_dim=1152, face_dim=512, latent_dim=256).to(device)
    if p4_path.exists():
        print(f"[EXISTS] Loading pre-trained Pillar 4 from {p4_path}...")
        ckpt_p4 = torch.load(p4_path, map_location=device)
        model_dc.load_state_dict(ckpt_p4['model_state_dict'])
        model_dc.eval()
        proxies_dc = ckpt_p4['pos_proxies'].to(device)
        c_neg_dc = ckpt_p4['neg_proxy'].to(device)
    else:
        print("\n--- Training Pillar 4: Dual-Crop Gated pAUC Subcenter on Full 40k ---")
        proxies_dc = nn.Parameter(torch.randn(12, 256, device=device))
        proxies_dc.data = F.normalize(proxies_dc.data, p=2, dim=1)
        c_neg_dc = nn.Parameter(torch.zeros(1, 256, device=device), requires_grad=False)

        opt_p4 = optim.AdamW(list(model_dc.parameters()) + [proxies_dc], lr=0.003, weight_decay=1e-4)

        for epoch in range(12):
            p_sub = np.random.choice(pos_idx, size=48, replace=False)
            n_sub = np.random.choice(neg_idx, size=144, replace=False)
            b_idx = np.concatenate([p_sub, n_sub])

            b_fu = torch.tensor(X_fu_train[b_idx], dtype=torch.float32, device=device)
            b_po = torch.tensor(X_po_train[b_idx], dtype=torch.float32, device=device)
            b_fa = torch.tensor(X_fa_train[b_idx], dtype=torch.float32, device=device)
            b_fv = torch.tensor(FV_train[b_idx], dtype=torch.float32, device=device).view(-1, 1)
            b_y = torch.tensor(y_train[b_idx], dtype=torch.float32, device=device)

            opt_p4.zero_grad()
            h = model_dc(b_fu, b_po, b_fa, b_fv)
            p_norm = F.normalize(proxies_dc, p=2, dim=1)
            sim_pos = torch.matmul(h, p_norm.T)
            l_pos = (1.0 / 20.0) * torch.logsumexp(20.0 * sim_pos, dim=1)
            l_neg = torch.matmul(h, c_neg_dc.T).squeeze(-1)

            loss = F.binary_cross_entropy_with_logits((l_pos - l_neg) * 8.0, b_y)
            loss.backward()
            opt_p4.step()

        torch.save({
            'model_state_dict': model_dc.state_dict(),
            'pos_proxies': F.normalize(proxies_dc, p=2, dim=1).detach().cpu(),
            'neg_proxy': c_neg_dc.detach().cpu()
        }, p4_path)
        print(f"Pillar 4 Saved -> {p4_path}")

    # Predict Pillar 4 on Val
    with torch.no_grad():
        v_fu = torch.tensor(X_fu_val, dtype=torch.float32, device=device)
        v_po = torch.tensor(X_po_val, dtype=torch.float32, device=device)
        v_fa = torch.tensor(X_fa_val, dtype=torch.float32, device=device)
        v_fv = torch.tensor(FV_val, dtype=torch.float32, device=device).view(-1, 1)
        v_h = model_dc(v_fu, v_po, v_fa, v_fv)
        v_pn = F.normalize(proxies_dc, p=2, dim=1)
        v_sim = torch.matmul(v_h, v_pn.T)
        v_lp = (1.0 / 20.0) * torch.logsumexp(20.0 * v_sim, dim=1)
        v_ln = torch.matmul(v_h, c_neg_dc.T).squeeze(-1)
        score_p4_val = (v_lp - v_ln).cpu().numpy()

    # 4. Out-of-fold Gaussian Rank-Normal Blending
    print("\n--- [4/5] Blending Meta-Ensemble ---")
    # Rank normal quantiles
    q1 = norm.ppf(rankdata(score_p1_val) / (N_val + 1))
    q2 = norm.ppf(rankdata(score_p2_val) / (N_val + 1))
    q3 = norm.ppf(rankdata(score_p3_val) / (N_val + 1))
    q4 = norm.ppf(rankdata(score_p4_val) / (N_val + 1))

    # Grand Super Ensemble Blend (Weighted Gaussian Quantiles)
    # Weights: Dual-crop dominant (0.15, 0.40, 0.15, 0.30)
    score_40k_full_ensemble = 0.15 * q1 + 0.40 * q2 + 0.15 * q3 + 0.30 * q4

    # Also evaluate the old 33.5k model on this exact same 1,582 validation set
    print("\n--- [5/5] Benchmarking Models on the 1,582 Holdout Validation Set ---")
    
    # Load old 33.5k models for direct head-to-head comparison
    mlp_33k_path = root / "models" / "rating_regressor_multimodal_bumble_40k.joblib"
    dc_mlp_33k_path = root / "models" / "rating_regressor_dual_crop_mlp_40k.joblib"
    sc_33k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_40k.pt"
    dc_33k_path = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_40k.pt"

    eval_results = []

    # 1. 30k Baseline Pure MLP
    mlp_30k = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    m30 = mlp_30k['estimator'] if isinstance(mlp_30k, dict) else mlp_30k
    score_30k_mlp = m30.predict_proba(X_mlp_single_val)[:, 1] if hasattr(m30, 'predict_proba') else m30.predict(X_mlp_single_val)
    eval_results.append(evaluate_predictions(score_30k_mlp, y_val, "30k Multimodal MLP (Pure)"))

    # 2. Previous 33.5k Models
    if mlp_33k_path.exists():
        m33_raw = joblib.load(mlp_33k_path)
        m33 = m33_raw['estimator'] if isinstance(m33_raw, dict) else m33_raw
        score_33k_p1 = m33.predict_proba(X_mlp_single_val)[:, 1] if hasattr(m33, 'predict_proba') else m33.predict(X_mlp_single_val)
        eval_results.append(evaluate_predictions(score_33k_p1, y_val, "33.5k Pillar 1 (Single MLP)"))

    if dc_mlp_33k_path.exists():
        m33_p2_raw = joblib.load(dc_mlp_33k_path)
        m33_p2 = m33_p2_raw['estimator'] if isinstance(m33_p2_raw, dict) else m33_p2_raw
        score_33k_p2 = m33_p2.predict_proba(X_mlp_dual_val)[:, 1] if hasattr(m33_p2, 'predict_proba') else m33_p2.predict(X_mlp_dual_val)
        eval_results.append(evaluate_predictions(score_33k_p2, y_val, "33.5k Pillar 2 (Dual-Crop MLP)"))

    if sc_33k_path.exists():
        score_33k_p3 = predict_visual_logits(str(sc_33k_path), X_fu_val, X_fa_val)
        eval_results.append(evaluate_predictions(score_33k_p3, y_val, "33.5k Pillar 3 (Single pAUC)"))

    if dc_33k_path.exists():
        score_33k_p4 = predict_dual_crop(str(dc_33k_path), X_fu_val, X_po_val, X_fa_val, FV_val)
        eval_results.append(evaluate_predictions(score_33k_p4, y_val, "33.5k Pillar 4 (Dual pAUC)"))

        # 33.5k Meta-Ensemble
        q1_33 = norm.ppf(rankdata(score_33k_p1) / (N_val + 1))
        q2_33 = norm.ppf(rankdata(score_33k_p2) / (N_val + 1))
        q3_33 = norm.ppf(rankdata(score_33k_p3) / (N_val + 1))
        q4_33 = norm.ppf(rankdata(score_33k_p4) / (N_val + 1))
        score_33k_ens = 0.15 * q1_33 + 0.40 * q2_33 + 0.15 * q3_33 + 0.30 * q4_33
        eval_results.append(evaluate_predictions(score_33k_ens, y_val, "33.5k Grand Super-Ensemble"))

    # 3. NEW TRUE 40,000 FULL MODELS
    eval_results.append(evaluate_predictions(score_p1_val, y_val, "TRUE 40k Pillar 1 (Single MLP Full)"))
    eval_results.append(evaluate_predictions(score_p2_val, y_val, "TRUE 40k Pillar 2 (Dual-Crop MLP Full)"))
    eval_results.append(evaluate_predictions(score_p3_val, y_val, "TRUE 40k Pillar 3 (Single pAUC Full)"))
    eval_results.append(evaluate_predictions(score_p4_val, y_val, "TRUE 40k Pillar 4 (Dual pAUC Full)"))
    eval_results.append(evaluate_predictions(score_40k_full_ensemble, y_val, "TRUE 40k Grand Super-Ensemble (FULL)"))

    df_res = pd.DataFrame(eval_results)
    print("\n" + "=" * 100)
    print(f"HEAD-TO-HEAD TOURNAMENT ON 1,582 VALIDATION HOLDOUT (40,000..41,582 | {n_val_likes} Likes)")
    print("=" * 100)
    print(df_res[["model", "hits_20", "rec_20", "prec_20", "hits_25", "rec_25", "prec_25", "pr_auc", "roc_auc"]].to_string(index=False))

    # Save to CSV
    res_csv = root / "results" / "benchmark_true_40k_full_vs_33k_on_val_1582.csv"
    res_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(res_csv, index=False)
    print(f"\nSaved tournament results to {res_csv}")

if __name__ == "__main__":
    main()
