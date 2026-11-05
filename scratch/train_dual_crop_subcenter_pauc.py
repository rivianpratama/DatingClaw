import os
import sys
from pathlib import Path

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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import rankdata

from face_similarity.store import load_store
from scratch.train_subcenter_pauc_regularized import (
    StratifiedSwipeSampler,
    ShiftInvariantTopQuartilePAUCLoss,
    CircadianContextBranch,
    project_proxy_gradients_to_tangent_space,
    retract_proxies_to_hypersphere,
    compute_multi_proxy_logit_and_reg,
    extract_context_features,
    evaluate_metrics,
    predict_visual_logits
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# Dual-Crop Orthogonal Fusion Net (Full Scene + Portrait Crop + ArcFace)
# ==============================================================================

class DualCropOrthogonalFusionNet(nn.Module):
    """
    Inputs:
      x_sig_full: [B, 1152] (L2-normalized full letterboxed SigLIP-384)
      x_sig_port: [B, 1152] (L2-normalized 1.8x portrait crop SigLIP-384)
      x_face:     [B, 512]  (L2-normalized InsightFace ArcFace)
      face_valid: [B, 1]    (1.0 if face detected, 0.0 otherwise)
    Output:
      h_fused:    [B, 256]  (Unit-norm embedding on S^255)
    """
    def __init__(self, sig_dim: int = 1152, face_dim: int = 512, latent_dim: int = 256, rank: int = 64):
        super().__init__()
        self.proj_sig_full = nn.Sequential(
            nn.Linear(sig_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.25)
        )
        self.proj_sig_port = nn.Sequential(
            nn.Linear(sig_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.25)
        )
        self.proj_face = nn.Sequential(
            nn.Linear(face_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.25)
        )

        # Stage 1: Hierarchical Gate between Full Scene SigLIP and Portrait Crop SigLIP
        self.port_gate = nn.Linear(latent_dim * 2 + 1, latent_dim)
        nn.init.constant_(self.port_gate.bias, 0.0)  # Starts 50/50 balanced

        # Stage 2: Low-Rank Factorized Bilinear Interaction (Rank = 64)
        self.U_sig = nn.Linear(sig_dim * 2, rank, bias=False)
        self.U_face = nn.Linear(face_dim, rank, bias=False)
        self.W_up = nn.Sequential(nn.Linear(rank, latent_dim, bias=False), nn.LayerNorm(latent_dim))
        nn.init.normal_(self.W_up[0].weight, std=0.01)

        # Stage 3: Dynamic Gates + Channel Scaling for Orthogonal Face & Bilinear Streams
        self.gate_ortho = nn.Linear(latent_dim * 2 + 1, latent_dim)
        self.gate_bilin = nn.Linear(latent_dim * 2 + 1, latent_dim)
        nn.init.constant_(self.gate_ortho.bias, -1.5)
        nn.init.constant_(self.gate_bilin.bias, -2.0)

        self.gamma_1 = nn.Parameter(torch.full((1, latent_dim), 0.20))
        self.gamma_2 = nn.Parameter(torch.full((1, latent_dim), 0.15))

    def forward(self, x_sig_full, x_sig_port, x_face, face_valid):
        # 1. Fuse Full-Scene SigLIP and Portrait-Crop SigLIP into unified aesthetic anchor
        h_full = F.normalize(self.proj_sig_full(x_sig_full), p=2, dim=-1)
        h_port = F.normalize(self.proj_sig_port(x_sig_port), p=2, dim=-1)
        beta = torch.sigmoid(self.port_gate(torch.cat([h_full, h_port, face_valid], dim=-1)))
        h_siglip = F.normalize((1.0 - beta) * h_full + beta * h_port, p=2, dim=-1)  # [B, 256]

        # 2. Gram-Schmidt Orthogonal Projection of ArcFace (keep raw sin(theta) magnitude!)
        h_face = F.normalize(self.proj_face(x_face), p=2, dim=-1) * face_valid
        proj_coeff = (h_face * h_siglip).sum(dim=-1, keepdim=True)
        h_face_ortho = h_face - proj_coeff * h_siglip  # Strictly orthogonal to h_siglip

        # 3. Low-Rank Factorized Bilinear Pooling
        x_sig_cat = torch.cat([x_sig_full, x_sig_port], dim=-1)  # [B, 2304]
        h_bilinear = self.W_up(F.gelu(self.U_sig(x_sig_cat)) * F.gelu(self.U_face(x_face))) * face_valid

        # 4. Context-Gated Orthogonal Residual Combination on S^255
        alpha_1 = self.gamma_1 * torch.sigmoid(self.gate_ortho(torch.cat([h_siglip, h_face_ortho, face_valid], dim=-1))) * face_valid
        alpha_2 = self.gamma_2 * torch.sigmoid(self.gate_bilin(torch.cat([h_siglip, h_bilinear, face_valid], dim=-1))) * face_valid

        h_fused = F.normalize(h_siglip + alpha_1 * h_face_ortho + alpha_2 * h_bilinear, p=2, dim=-1)
        return h_fused


class DualCropSwipeDataset(Dataset):
    def __init__(self, x_full, x_port, x_face, face_valid, context, labels):
        self.x_full = torch.tensor(x_full, dtype=torch.float32)
        self.x_port = torch.tensor(x_port, dtype=torch.float32)
        self.x_face = torch.tensor(x_face, dtype=torch.float32)
        self.face_valid = torch.tensor(face_valid, dtype=torch.float32)
        self.context = torch.tensor(context, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.x_full[idx], self.x_port[idx], self.x_face[idx],
            self.face_valid[idx], self.context[idx], self.labels[idx]
        )


def train_dual_crop_model(
    X_full_tr, X_port_tr, X_face_tr, FV_tr, y_tr, C_tr,
    X_full_val, X_port_val, X_face_val, FV_val, y_val, C_val,
    K=12, M=20, epochs=12, batch_size=512,
    save_path="models/subcenter_phase2/best_dual_crop_subcenter_pauc.pt"
):
    train_dataset = DualCropSwipeDataset(X_full_tr, X_port_tr, X_face_tr, FV_tr, C_tr, y_tr)
    val_dataset = DualCropSwipeDataset(X_full_val, X_port_val, X_face_val, FV_val, C_val, y_val)

    sampler = StratifiedSwipeSampler(y_tr, batch_size=batch_size, pos_per_batch=64)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    print("--- Phase 1: Warm-starting Dual-Crop Orthogonal Fusion Net (2 Epochs) ---")
    fusion_net = DualCropOrthogonalFusionNet(sig_dim=1152, face_dim=512, latent_dim=256, rank=64).to(device)
    warm_head = nn.Linear(256, 1).to(device)
    warm_opt = optim.AdamW(list(fusion_net.parameters()) + list(warm_head.parameters()), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(2):
        fusion_net.train()
        warm_head.train()
        for b_full, b_port, b_face, b_fv, _, b_y in train_loader:
            b_full, b_port = b_full.to(device), b_port.to(device)
            b_face, b_fv, b_y = b_face.to(device), b_fv.to(device), b_y.to(device)
            warm_opt.zero_grad()
            emb = fusion_net(b_full, b_port, b_face, b_fv)
            loss = bce_loss_fn(warm_head(emb).view(-1), b_y)
            loss.backward()
            warm_opt.step()

    print(f"\n--- Phase 2: Seeding Proxies (K={K}, M={M}) via Spherical K-Means ---")
    fusion_net.eval()
    all_embs = []
    extract_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for b_full, b_port, b_face, b_fv, _, _ in extract_loader:
            b_full, b_port = b_full.to(device), b_port.to(device)
            b_face, b_fv = b_face.to(device), b_fv.to(device)
            emb = fusion_net(b_full, b_port, b_face, b_fv)
            all_embs.append(emb.cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)

    pos_embs = all_embs[y_tr == 1.0]
    neg_embs = all_embs[y_tr == 0.0]

    km_pos = KMeans(n_clusters=K, n_init=10, random_state=42).fit(pos_embs)
    pos_centroids = km_pos.cluster_centers_ / np.linalg.norm(km_pos.cluster_centers_, axis=-1, keepdims=True)

    km_neg = KMeans(n_clusters=M, n_init=10, random_state=42).fit(neg_embs)
    neg_centroids = km_neg.cluster_centers_ / np.linalg.norm(km_neg.cluster_centers_, axis=-1, keepdims=True)

    proxies_pos = nn.Parameter(torch.tensor(pos_centroids, dtype=torch.float32, device=device))
    proxies_neg = nn.Parameter(torch.tensor(neg_centroids, dtype=torch.float32, device=device))
    proxies_pos_init = torch.tensor(pos_centroids, dtype=torch.float32, device=device)

    bias_b0 = nn.Parameter(torch.tensor([-2.04], dtype=torch.float32, device=device))
    context_net = CircadianContextBranch(in_features=4, hidden=16).to(device)

    optimizer = optim.AdamW([
        {'params': [proxies_pos, proxies_neg], 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': list(fusion_net.parameters()) + list(context_net.parameters()) + [bias_b0], 'lr': 5e-4, 'weight_decay': 0.05}
    ])

    loss_fn = ShiftInvariantTopQuartilePAUCLoss(top_neg_frac=0.30, beta=3.0, margin=0.30, smooth_eps=0.03)

    print(f"\n--- Phase 3: Regularized Dual-Crop Training ({epochs} Epochs) ---")
    best_val_score = -float('inf')
    best_metrics = {}
    patience = 4
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        fusion_net.train()
        context_net.train()

        if epoch <= 5:
            tau = 0.07 + 0.5 * (0.15 - 0.07) * (1.0 + np.cos(np.pi * (epoch - 1) / 5.0))
        else:
            tau = 0.07

        total_loss_sum = 0.0
        pauc_loss_sum = 0.0
        bce_loss_sum = 0.0
        reg_loss_sum = 0.0
        n_batches = 0

        for b_full, b_port, b_face, b_fv, b_c, b_y in train_loader:
            b_full, b_port = b_full.to(device), b_port.to(device)
            b_face, b_fv, b_c, b_y = b_face.to(device), b_fv.to(device), b_c.to(device), b_y.to(device)

            optimizer.zero_grad()
            h_fused = fusion_net(b_full, b_port, b_face, b_fv)
            z_proxy, loss_reg = compute_multi_proxy_logit_and_reg(
                h_fused, proxies_pos, proxies_neg, proxies_pos_init,
                tau=tau, margin=0.15, gamma=16.0, bias_b0=bias_b0,
                p_drop=0.20, training=True
            )
            g_context = context_net(b_c).squeeze(-1)
            z_train = z_proxy + g_context

            total_loss, l_pauc, l_bce = loss_fn(z_proxy, z_train, b_y)
            loss_with_reg = total_loss + 0.05 * loss_reg
            loss_with_reg.backward()

            project_proxy_gradients_to_tangent_space(proxies_pos, proxies_neg)
            optimizer.step()
            retract_proxies_to_hypersphere(proxies_pos, proxies_neg)

            total_loss_sum += loss_with_reg.item()
            pauc_loss_sum += l_pauc.item()
            bce_loss_sum += l_bce.item()
            reg_loss_sum += loss_reg.item()
            n_batches += 1

        fusion_net.eval()
        val_preds = []
        with torch.no_grad():
            for b_full, b_port, b_face, b_fv, _, _ in val_loader:
                b_full, b_port = b_full.to(device), b_port.to(device)
                b_face, b_fv = b_face.to(device), b_fv.to(device)
                h_fused = fusion_net(b_full, b_port, b_face, b_fv)
                z_proxy, _ = compute_multi_proxy_logit_and_reg(
                    h_fused, proxies_pos, proxies_neg, proxies_pos_init,
                    tau=0.07, margin=0.15, gamma=16.0, bias_b0=bias_b0,
                    p_drop=0.0, training=False
                )
                val_preds.append(z_proxy.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_m = evaluate_metrics(val_preds, y_val)
        val_score = val_m["Likes_25_int"] + 10.0 * val_m["PR-AUC"]

        print(
            f"Epoch {epoch:2d}/{epochs} (tau={tau:.3f}) | "
            f"Loss: {total_loss_sum/n_batches:.4f} (pAUC: {pauc_loss_sum/n_batches:.4f}, BCE: {bce_loss_sum/n_batches:.4f}) | "
            f"Val Likes@25%: {val_m['Likes@25%Cap']} | Val PR-AUC: {val_m['PR-AUC']:.4f} | ROC: {val_m['ROC-AUC']:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_metrics = val_m
            best_epoch = epoch
            patience_counter = 0

            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'fusion_net': fusion_net.state_dict(),
                'context_net': context_net.state_dict(),
                'proxies_pos': proxies_pos.data,
                'proxies_neg': proxies_neg.data,
                'bias_b0': bias_b0.data,
                'proxies_pos_init': proxies_pos_init,
                'val_metrics': val_m,
                'K': K, 'M': M,
                'tau': 0.07, 'margin': 0.15, 'gamma': 16.0
            }, save_path)
            print(f"  >>> Best checkpoint saved at epoch {epoch} (Val Likes@25%: {val_m['Likes@25%Cap']}, PR-AUC: {val_m['PR-AUC']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Triggered at epoch {epoch}. Best was {best_epoch}.")
                break

    return best_epoch, best_metrics


def predict_dual_crop(model_path, X_full, X_port, X_face, FV, batch_size=1024):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    fn = DualCropOrthogonalFusionNet(sig_dim=1152, face_dim=512, latent_dim=256, rank=64).to(device)
    fn.load_state_dict(ckpt['fusion_net'])
    fn.eval()

    proxies_pos = ckpt['proxies_pos'].to(device)
    proxies_neg = ckpt['proxies_neg'].to(device)
    proxies_pos_init = ckpt['proxies_pos_init'].to(device)
    bias_b0 = ckpt['bias_b0'].to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_full, dtype=torch.float32),
        torch.tensor(X_port, dtype=torch.float32),
        torch.tensor(X_face, dtype=torch.float32),
        torch.tensor(FV, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for b_fu, b_po, b_fa, b_fv in loader:
            b_fu, b_po = b_fu.to(device), b_po.to(device)
            b_fa, b_fv = b_fa.to(device), b_fv.to(device)
            h_fused = fn(b_fu, b_po, b_fa, b_fv)
            z_proxy, _ = compute_multi_proxy_logit_and_reg(
                h_fused, proxies_pos, proxies_neg, proxies_pos_init,
                tau=0.07, margin=0.15, gamma=16.0, bias_b0=bias_b0,
                p_drop=0.0, training=False
            )
            preds.append(z_proxy.cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"

    print("=" * 80)
    print("STEP 2: TRAINING DUAL-CROP ORTHOGONAL RESIDUAL FUSION ON 30K DATASET")
    print("=" * 80)

    # 1. Load Stores for 30k
    ref_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    sig_full_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))
    sig_port_30k = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_pure.npz"))

    y_30k = (ref_30k.ratings >= 50.0).astype(int)
    X_face_30k = ref_30k.embeddings
    X_full_30k = sig_full_30k['embeddings']
    X_port_30k = sig_port_30k['embeddings']
    FV_30k = sig_port_30k['face_valid']
    paths_30k = ref_30k.paths

    print(f"Total 30k samples: {len(y_30k)} | Full Sig: {X_full_30k.shape} | Port Sig: {X_port_30k.shape} | Face Valid: {FV_30k.sum():.0f}/{len(FV_30k)}")

    # Extract context
    C_30k = extract_context_features(paths_30k, csv_path)

    # 2. Chronological Split: 0 to 26,500 Train (26.5k), 26,500 to 29,582 Val (3.08k)
    train_cutoff = 26500
    X_fu_tr, X_po_tr, X_fa_tr, FV_tr, y_tr, C_tr = (
        X_full_30k[:train_cutoff], X_port_30k[:train_cutoff], X_face_30k[:train_cutoff],
        FV_30k[:train_cutoff], y_30k[:train_cutoff], C_30k[:train_cutoff]
    )
    X_fu_val, X_po_val, X_fa_val, FV_val, y_val, C_val = (
        X_full_30k[train_cutoff:], X_port_30k[train_cutoff:], X_face_30k[train_cutoff:],
        FV_30k[train_cutoff:], y_30k[train_cutoff:], C_30k[train_cutoff:]
    )

    print(f"Train Block: {len(y_tr)} profiles | Likes: {int(y_tr.sum())} ({y_tr.sum()/len(y_tr)*100:.2f}%)")
    print(f"Val Block: {len(y_val)} profiles | Likes: {int(y_val.sum())} ({y_val.sum()/len(y_val)*100:.2f}%)")

    # 3. Train Dual-Crop 26.5k Model
    save_path_26k = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_26k.pt"
    best_epoch_26k, best_m_26k = train_dual_crop_model(
        X_fu_tr, X_po_tr, X_fa_tr, FV_tr, y_tr, C_tr,
        X_fu_val, X_po_val, X_fa_val, FV_val, y_val, C_val,
        K=12, M=20, epochs=12, batch_size=512,
        save_path=str(save_path_26k)
    )

    # 4. Train Dual-Crop Companion Model on Full 30k for exact best_epoch_26k epochs
    print("\n" + "=" * 80)
    print(f"TRAINING COMPANION DUAL-CROP MODEL ON FULL 0–29.5k FOR EXACTLY E*={best_epoch_26k} EPOCHS")
    print("=" * 80)
    save_path_30k = root / "models" / "subcenter_phase2" / "best_dual_crop_pauc_30k_full.pt"
    train_dual_crop_model(
        X_full_30k, X_port_30k, X_face_30k, FV_30k, y_30k, C_30k,
        X_fu_val, X_po_val, X_fa_val, FV_val, y_val, C_val,
        K=12, M=20, epochs=best_epoch_26k, batch_size=512,
        save_path=str(save_path_30k)
    )

    # 5. Tournament Benchmark on Untouched Holdout 2 (30,055 to 33,629)
    print("\n" + "=" * 80)
    print("TOURNAMENT BENCHMARK ON UNTOUCHED HOLDOUT 2 (30k to 33.6k)")
    print("=" * 80)

    ref_h2 = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    sig_full_h2 = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))
    sig_port_h2 = np.load(str(root / "embeddings" / "siglip_portrait_store_bumble_30k_to_33k.npz"))
    clip_h2 = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))

    y_test = (ref_h2.ratings >= 50.0).astype(int)
    X_fa_test = ref_h2.embeddings
    X_fu_test = sig_full_h2['embeddings']
    X_po_test = sig_port_h2['embeddings']
    FV_test = sig_port_h2['face_valid']
    X_clip_test = clip_h2['embeddings']
    X_mlp_test = np.hstack([X_fa_test, X_clip_test])

    n_test = len(y_test)
    n_test_likes = int(y_test.sum())
    k_25 = int(np.round(0.25 * n_test))
    print(f"Holdout 2: {n_test} profiles | Likes: {n_test_likes} ({n_test_likes/n_test*100:.2f}%) | 25% Budget: {k_25} swipes\n")

    # Dual-Crop Predictions
    preds_dc_26k = predict_dual_crop(str(save_path_26k), X_fu_test, X_po_test, X_fa_test, FV_test)
    preds_dc_30k = predict_dual_crop(str(save_path_30k), X_fu_test, X_po_test, X_fa_test, FV_test)

    # Single-Crop pAUC (30k) for comparison
    sc_30k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt"
    preds_sc_30k = predict_visual_logits(str(sc_30k_path), X_fu_test, X_fa_test)

    # Multimodal MLP (30k)
    mlp_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    mlp = mlp_obj['estimator'] if isinstance(mlp_obj, dict) else mlp_obj
    preds_mlp = mlp.predict(X_mlp_test)

    # Rank data for Ensembles
    r_mlp = rankdata(preds_mlp) / n_test
    r_sc_30k = rankdata(preds_sc_30k) / n_test
    r_dc_26k = rankdata(preds_dc_26k) / n_test
    r_dc_30k = rankdata(preds_dc_30k) / n_test

    models = {
        "Multimodal MLP (30k Pure)": preds_mlp,
        "Single-Crop SigLIP pAUC (30k)": preds_sc_30k,
        "Dual-Crop SigLIP pAUC (26.5k)": preds_dc_26k,
        "Dual-Crop SigLIP pAUC (30k Full)": preds_dc_30k,
        "Hybrid Ensemble 1: MLP + Single-Crop pAUC (50/50)": 0.50 * r_mlp + 0.50 * r_sc_30k,
        "Hybrid Ensemble 2: MLP + Dual-Crop pAUC 30k (50/50)": 0.50 * r_mlp + 0.50 * r_dc_30k,
        "Hybrid Ensemble 3: Dual-Crop 30k + Single-Crop 30k (50/50)": 0.50 * r_dc_30k + 0.50 * r_sc_30k,
        "Grand Super-Ensemble: MLP (34%) + Single-Crop (33%) + Dual-Crop (33%)": (
            0.34 * r_mlp + 0.33 * r_sc_30k + 0.33 * r_dc_30k
        ),
        "Grand Super-Ensemble: MLP (40%) + Dual-Crop 26k (30%) + Dual-Crop 30k (30%)": (
            0.40 * r_mlp + 0.30 * r_dc_26k + 0.30 * r_dc_30k
        ),
    }

    results = []
    print("--- Leaderboard on Untouched Holdout 2 (3,481 profiles, 443 likes) ---")
    for name, p in models.items():
        m = evaluate_metrics(p, y_test)
        results.append({"Model": name, **m})
        print(f"{name:<60} | Likes@25%: {m['Likes@25%Cap']:<18} | PR-AUC: {m['PR-AUC']:.4f} | ROC: {m['ROC-AUC']:.4f}")

    df_res = pd.DataFrame(results)
    out_csv = root / "results" / "benchmark_dual_crop_siglip_holdout2.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark table to {out_csv}")


if __name__ == "__main__":
    main()
