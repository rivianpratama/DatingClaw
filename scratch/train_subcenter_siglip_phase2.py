import os
import sys
import argparse
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store
from scratch.train_subcenter_multiproxy_phase1 import (
    CircadianContextBranch,
    Top25BoundaryLoss,
    extract_context_features,
    evaluate_holdout
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# 1. Asymmetric Gated Residual Fusion for SigLIP (1152D) + ArcFace (512D)
# ------------------------------------------------------------------------------
class SiglipGatedResidualFusion(nn.Module):
    """
    Asymmetric Gated Residual Fusion:
    SigLIP-384 (1152D) captures rich visual aesthetics (hair, smile, styling, lighting).
    ArcFace (512D) captures facial bone geometry and identity.
    A gating network with multiplicative interaction modulates how much facial geometry contributes.
    """
    def __init__(self, siglip_dim=1152, face_dim=512, proj_dim=256, mod_drop_prob=0.20):
        super().__init__()
        self.proj_dim = proj_dim
        self.mod_drop_prob = mod_drop_prob

        self.siglip_proj = nn.Sequential(
            nn.Linear(siglip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.face_proj = nn.Sequential(
            nn.Linear(face_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # Gate with multiplicative interaction: [h_siglip, h_face, h_siglip * h_face] -> 256D
        self.gate_fc = nn.Linear(proj_dim * 3, proj_dim)
        # Initialize gate bias to -1.5 so sigmoid(b_g) ~= 0.18 initially
        nn.init.constant_(self.gate_fc.bias, -1.5)
        nn.init.xavier_uniform_(self.gate_fc.weight)

        self.post_norm = nn.LayerNorm(proj_dim)

    def forward(self, x_siglip, x_face, training=True):
        x_siglip = F.normalize(x_siglip, p=2, dim=-1)
        x_face = F.normalize(x_face, p=2, dim=-1)

        # Modality DropPath on face during training
        if training and self.mod_drop_prob > 0.0:
            mask = (torch.rand(x_face.size(0), 1, device=x_face.device) > self.mod_drop_prob).float()
            x_face = x_face * mask

        h_siglip = self.siglip_proj(x_siglip)
        h_face = self.face_proj(x_face)

        interact = h_siglip * h_face
        gate_input = torch.cat([h_siglip, h_face, interact], dim=-1)
        alpha = torch.sigmoid(self.gate_fc(gate_input))

        fused = h_siglip + alpha * h_face
        fused = self.post_norm(fused)
        fused = F.normalize(fused, p=2, dim=-1) # Project to S^255 unit hypersphere
        return fused

# ------------------------------------------------------------------------------
# 2. Multi-Proxy Sub-Center Classifier (SoftTriple Architecture)
# ------------------------------------------------------------------------------
class MultiProxyClassifier(nn.Module):
    def __init__(self, in_dim=256, num_pos_proxies=12, num_neg_proxies=20, tau=0.07, margin=0.15, gamma=16.0):
        super().__init__()
        self.K = num_pos_proxies
        self.M = num_neg_proxies
        self.tau = tau
        self.margin = margin
        self.gamma = gamma

        # Proxies live in projected 256D unit hypersphere
        self.pos_proxies = nn.Parameter(torch.randn(num_pos_proxies, in_dim))
        self.neg_proxies = nn.Parameter(torch.randn(num_neg_proxies, in_dim))
        self.b_0 = nn.Parameter(torch.zeros(1))

    def init_proxies(self, pos_centroids_np, neg_centroids_np):
        with torch.no_grad():
            self.pos_proxies.copy_(torch.tensor(pos_centroids_np, dtype=torch.float32))
            self.neg_proxies.copy_(torch.tensor(neg_centroids_np, dtype=torch.float32))

    def forward(self, fused_emb):
        P_pos = F.normalize(self.pos_proxies, p=2, dim=-1) # [K, D]
        P_neg = F.normalize(self.neg_proxies, p=2, dim=-1) # [M, D]

        cos_pos = torch.matmul(fused_emb, P_pos.t()) # [B, K]
        cos_neg = torch.matmul(fused_emb, P_neg.t()) # [B, M]

        # Smooth-max aggregations
        s_pos = self.tau * torch.logsumexp((cos_pos - self.margin) / self.tau, dim=-1, keepdim=True)
        s_neg = self.tau * torch.logsumexp(cos_neg / self.tau, dim=-1, keepdim=True)

        z_proxy = self.gamma * (s_pos - s_neg) + self.b_0
        return z_proxy

    def ortho_loss(self):
        """Orthogonality penalty between positive proxies to prevent sub-center collapse."""
        P_pos = F.normalize(self.pos_proxies, p=2, dim=-1)
        cos_matrix = torch.matmul(P_pos, P_pos.t()) # [K, K]
        mask = ~torch.eye(self.K, dtype=torch.bool, device=P_pos.device)
        off_diag = cos_matrix[mask]
        penalty = torch.clamp(off_diag - 0.25, min=0.0) ** 2
        return penalty.mean()

import copy

# ------------------------------------------------------------------------------
# 3. Training Function
# ------------------------------------------------------------------------------
def train_siglip_subcenter(
    X_siglip_tr, X_face_tr, y_tr, C_tr,
    X_siglip_val, X_face_val, y_val,
    K=12, M=20, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384"
):
    print(f"\n--- Training {exp_name} (K={K}, M={M}, Epochs={epochs}, LR={lr}) ---")
    t_tr_siglip = torch.tensor(X_siglip_tr, dtype=torch.float32)
    t_tr_face = torch.tensor(X_face_tr, dtype=torch.float32)
    t_tr_c = torch.tensor(C_tr, dtype=torch.float32)
    t_tr_y = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    t_val_siglip = torch.tensor(X_siglip_val, dtype=torch.float32).to(device)
    t_val_face = torch.tensor(X_face_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(t_tr_siglip, t_tr_face, t_tr_c, t_tr_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 1. Warm-start Gated Fusion (2 epochs)
    fusion_net = SiglipGatedResidualFusion(siglip_dim=1152, face_dim=512, proj_dim=256, mod_drop_prob=0.20).to(device)
    warm_head = nn.Linear(256, 1).to(device)
    warm_opt = optim.AdamW(list(fusion_net.parameters()) + list(warm_head.parameters()), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    print("  Warm-starting Gated Fusion representation (2 epochs)...")
    for epoch in range(2):
        fusion_net.train()
        warm_head.train()
        for b_siglip, b_face, _, b_y in train_loader:
            b_siglip, b_face, b_y = b_siglip.to(device), b_face.to(device), b_y.to(device)
            warm_opt.zero_grad()
            emb = fusion_net(b_siglip, b_face, training=True)
            loss = bce_loss_fn(warm_head(emb), b_y)
            loss.backward()
            warm_opt.step()

    # 2. Extract embeddings & cluster proxies using Spherical K-Means
    fusion_net.eval()
    with torch.no_grad():
        all_embs = []
        eval_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        for b_siglip, b_face, _, _ in eval_loader:
            b_siglip, b_face = b_siglip.to(device), b_face.to(device)
            emb = fusion_net(b_siglip, b_face, training=False)
            all_embs.append(emb.cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)

    pos_embs = all_embs[y_tr == 1.0]
    neg_embs = all_embs[y_tr == 0.0]

    print(f"  Clustering {len(pos_embs)} Likes into K={K} archetypes via Spherical K-Means...")
    km_pos = KMeans(n_clusters=K, n_init=10, random_state=42).fit(pos_embs)
    pos_centroids = km_pos.cluster_centers_ / np.linalg.norm(km_pos.cluster_centers_, axis=-1, keepdims=True)

    print(f"  Clustering {len(neg_embs)} Passes into M={M} clusters via Spherical K-Means...")
    km_neg = KMeans(n_clusters=M, n_init=10, random_state=42).fit(neg_embs)
    neg_centroids = km_neg.cluster_centers_ / np.linalg.norm(km_neg.cluster_centers_, axis=-1, keepdims=True)

    # 3. Full Training with Quantile Boundary Loss & Circadian Branch
    proxy_model = MultiProxyClassifier(
        in_dim=256,
        num_pos_proxies=K,
        num_neg_proxies=M,
        tau=0.07,
        margin=0.15,
        gamma=16.0
    ).to(device)
    proxy_model.init_proxies(pos_centroids, neg_centroids)

    context_net = CircadianContextBranch(in_features=4, hidden=16).to(device)
    boundary_loss_fn = Top25BoundaryLoss(w_pos=6.0, beta=4.0, m_pos=0.25, m_neg=0.15)

    optimizer = optim.AdamW([
        {'params': fusion_net.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': proxy_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': context_net.parameters(), 'lr': 1e-3, 'weight_decay': 0.1}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    tau_hat = 0.0
    best_likes_25 = 0
    best_stats = None
    best_preds = None
    best_fusion_state = None
    best_proxy_state = None
    best_context_state = None
    best_tau_hat = 0.0

    for epoch in range(epochs):
        fusion_net.train()
        proxy_model.train()
        context_net.train()

        w_boundary = min(0.40, 0.40 * (epoch / 5.0))
        w_bce = 0.55
        w_ortho = 0.05

        for b_siglip, b_face, b_c, b_y in train_loader:
            b_siglip, b_face = b_siglip.to(device), b_face.to(device)
            b_c, b_y = b_c.to(device), b_y.to(device)

            optimizer.zero_grad()
            emb = fusion_net(b_siglip, b_face, training=True)
            z_proxy = proxy_model(emb)
            g_c = context_net(b_c)
            z_train = z_proxy + g_c

            with torch.no_grad():
                batch_q78 = torch.quantile(z_proxy.detach(), 0.78).item()
                if epoch == 0 and tau_hat == 0.0:
                    tau_hat = batch_q78
                else:
                    tau_hat = 0.98 * tau_hat + 0.02 * batch_q78

            y_smoothed = b_y * (1.0 - 0.03) + 0.5 * 0.03
            loss_bce = F.binary_cross_entropy_with_logits(z_train, y_smoothed)
            loss_boundary = boundary_loss_fn(z_proxy, b_y, tau_hat)
            loss_ortho = proxy_model.ortho_loss()

            total_loss = w_bce * loss_bce + w_boundary * loss_boundary + w_ortho * loss_ortho
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(fusion_net.parameters()) + list(proxy_model.parameters()), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation (clamping context g_c = 0)
        fusion_net.eval()
        proxy_model.eval()
        with torch.no_grad():
            val_emb = fusion_net(t_val_siglip, t_val_face, training=False)
            val_preds = proxy_model(val_emb).cpu().numpy().flatten()

        stats = evaluate_holdout(val_preds, y_val)
        likes_count = int(stats['Likes@25%Cap'].split('/')[0])
        print(f"  Epoch {epoch+1:2d}/{epochs} | Tau78: {tau_hat:5.2f} | Likes@25%Cap: {stats['Likes@25%Cap']} | PR-AUC: {stats['PR-AUC']:.4f} | ROC: {stats['ROC-AUC']:.4f} | Swipe@95%: {stats['SwipeRate@95%Likes']}")

        if likes_count > best_likes_25 or (likes_count == best_likes_25 and stats['PR-AUC'] > (best_stats['PR-AUC'] if best_stats else 0)):
            best_likes_25 = likes_count
            best_stats = stats
            best_preds = val_preds
            best_fusion_state = copy.deepcopy(fusion_net.state_dict())
            best_proxy_state = copy.deepcopy(proxy_model.state_dict())
            best_context_state = copy.deepcopy(context_net.state_dict())
            best_tau_hat = tau_hat

    # Restore peak state
    fusion_net.load_state_dict(best_fusion_state)
    proxy_model.load_state_dict(best_proxy_state)
    context_net.load_state_dict(best_context_state)

    return best_stats, best_preds, fusion_net, proxy_model, context_net, best_tau_hat

def main():
    parser = argparse.ArgumentParser(description="Phase 2 SigLIP-384 Multi-Proxy Training & Benchmark")
    parser.add_argument("--mode", choices=["20k", "28k", "30k", "all"], default="20k", help="Evaluation mode")
    args = parser.parse_args()

    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    models_dir = root / "models" / "subcenter_phase2"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PHASE 2: SigLIP-So400m-384 HIGH-RESOLUTION MULTI-PROXY (MODE: {args.mode.upper()})")
    print("=" * 80)

    # 1. Load Common Holdout (20k-22k)
    s_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    sig_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_to_22k.npz"))

    y_val = (s_holdout.ratings >= 50.0).astype(int)
    X_val_face = s_holdout.embeddings
    X_val_siglip = sig_holdout['embeddings']
    print(f"Holdout: {len(y_val)} profiles (Likes: {int(y_val.sum())}, Passes: {len(y_val)-int(y_val.sum())})")

    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"

    rows = [
        {"Model": "ViT-B/32 Pure Multimodal (Baseline)", "Likes@25%": "123/157 (78.3%)", "HitRate@25%": "24.9%", "PR-AUC": 0.4576, "Swipe@95%": "50.3%"},
        {"Model": "Phase 1 Multi-Proxy (ViT-B/32, K=8)", "Likes@25%": "130/157 (82.8%)", "HitRate@25%": "26.4%", "PR-AUC": 0.4632, "Swipe@95%": "46.0%"},
        {"Model": "Phase 1 Multi-Proxy (ViT-B/32, K=12, 28k)", "Likes@25%": "129/157 (82.2%)", "HitRate@25%": "26.2%", "PR-AUC": 0.4802, "Swipe@95%": "48.2%"},
    ]

    # --- Mode 20k or all ---
    if args.mode in ["20k", "all"]:
        s_20k = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
        sig_20k = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_pure.npz"))

        y_20k = (s_20k.ratings >= 50.0).astype(np.float32)
        X_face_20k = s_20k.embeddings
        X_sig_20k = sig_20k['embeddings']
        C_20k = extract_context_features(s_20k.paths, csv_path)
        print(f"\n[20K BENCHMARK] Training set: {len(y_20k)} profiles (Likes: {int(y_20k.sum())})")

        stats_k8, _, f_k8, p_k8, c_k8, _ = train_siglip_subcenter(
            X_sig_20k, X_face_20k, y_20k, C_20k,
            X_val_siglip, X_val_face, y_val,
            K=8, M=16, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384 20k (K=8)"
        )
        rows.append({
            "Model": "Phase 2 SigLIP-384 (20k, K=8)",
            "Likes@25%": stats_k8['Likes@25%Cap'],
            "HitRate@25%": stats_k8['HitRate@25%'],
            "PR-AUC": stats_k8['PR-AUC'],
            "Swipe@95%": stats_k8['SwipeRate@95%Likes']
        })

        stats_k12, _, f_k12, p_k12, c_k12, _ = train_siglip_subcenter(
            X_sig_20k, X_face_20k, y_20k, C_20k,
            X_val_siglip, X_val_face, y_val,
            K=12, M=20, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384 20k (K=12)"
        )
        rows.append({
            "Model": "Phase 2 SigLIP-384 (20k, K=12)",
            "Likes@25%": stats_k12['Likes@25%Cap'],
            "HitRate@25%": stats_k12['HitRate@25%'],
            "PR-AUC": stats_k12['PR-AUC'],
            "Swipe@95%": stats_k12['SwipeRate@95%Likes']
        })

        # Checkpoint
        torch.save({
            'fusion_net': f_k12.state_dict(),
            'proxy_model': p_k12.state_dict(),
            'context_net': c_k12.state_dict(),
            'stats': stats_k12,
        }, str(models_dir / "best_subcenter_siglip_20k.pt"))

    # --- Mode 28k or all ---
    if args.mode in ["28k", "all"]:
        s_20k = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
        sig_20k = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_pure.npz"))
        s_tail = load_store(str(root / "embeddings" / "reference_store_bumble_22k_to_30k.npz"))
        sig_tail = np.load(str(root / "embeddings" / "siglip_store_bumble_22k_to_30k.npz"))

        X_train_face_leakfree = np.vstack([s_20k.embeddings, s_tail.embeddings])
        X_train_siglip_leakfree = np.vstack([sig_20k['embeddings'], sig_tail['embeddings']])
        y_train_leakfree = np.concatenate([
            (s_20k.ratings >= 50.0).astype(np.float32),
            (s_tail.ratings >= 50.0).astype(np.float32)
        ])
        paths_leakfree = list(s_20k.paths) + list(s_tail.paths)
        C_train_leakfree = extract_context_features(paths_leakfree, csv_path)

        print(f"\n[28K LEAK-FREE BENCHMARK] Training set: {len(y_train_leakfree)} profiles (Likes: {int(y_train_leakfree.sum())})")
        stats_28k, _, f_28k, p_28k, c_28k, _ = train_siglip_subcenter(
            X_train_siglip_leakfree, X_train_face_leakfree, y_train_leakfree, C_train_leakfree,
            X_val_siglip, X_val_face, y_val,
            K=12, M=20, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384 28k (K=12)"
        )
        rows.append({
            "Model": "Phase 2 SigLIP-384 (28k Leak-Free, K=12)",
            "Likes@25%": stats_28k['Likes@25%Cap'],
            "HitRate@25%": stats_28k['HitRate@25%'],
            "PR-AUC": stats_28k['PR-AUC'],
            "Swipe@95%": stats_28k['SwipeRate@95%Likes']
        })

    # --- Mode 30k or all ---
    if args.mode in ["30k", "all"]:
        sig_30k_file = root / "embeddings" / "siglip_store_bumble_30k_pure.npz"
        s_20k = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
        sig_20k = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_pure.npz"))
        s_hold = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
        sig_hold = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_to_22k.npz"))
        s_tail = load_store(str(root / "embeddings" / "reference_store_bumble_22k_to_30k.npz"))
        sig_tail = np.load(str(root / "embeddings" / "siglip_store_bumble_22k_to_30k.npz"))

        if not sig_30k_file.exists():
            print(f"\nConsolidating {sig_30k_file}...")
            all_siglip = np.vstack([sig_20k['embeddings'], sig_hold['embeddings'], sig_tail['embeddings']])
            all_paths = list(sig_20k['paths']) + list(sig_hold['paths']) + list(sig_tail['paths'])
            np.savez_compressed(
                sig_30k_file,
                embeddings=all_siglip,
                paths=np.array(all_paths),
                model_name="google/siglip-so400m-patch14-384",
                resolution=384,
                patch_size=14
            )
            print(f"Saved {len(all_siglip)} embeddings to {sig_30k_file}")

        s_all = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
        sig_all = np.load(str(sig_30k_file))
        y_all = (s_all.ratings >= 50.0).astype(np.float32)
        C_all = extract_context_features(s_all.paths, csv_path)

        print(f"\n[30K PRODUCTION TRAINING] Dataset: {len(y_all)} profiles (Likes: {int(y_all.sum())})")
        stats_30k, _, f_30k, p_30k, c_30k, tau_30k = train_siglip_subcenter(
            sig_all['embeddings'], s_all.embeddings, y_all, C_all,
            X_val_siglip, X_val_face, y_val,
            K=12, M=20, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384 30k Production (K=12)"
        )

        prod_ckpt = models_dir / "best_subcenter_siglip_30k_pure.pt"
        torch.save({
            'fusion_net': f_30k.state_dict(),
            'proxy_model': p_30k.state_dict(),
            'context_net': c_30k.state_dict(),
            'tau_hat': tau_30k,
            'stats': stats_30k,
            'train_samples': len(y_all)
        }, str(prod_ckpt))
        print(f"\nSaved production checkpoint to {prod_ckpt}")

        rows.append({
            "Model": "Phase 2 SigLIP-384 (30k Production, K=12)",
            "Likes@25%": stats_30k['Likes@25%Cap'],
            "HitRate@25%": stats_30k['HitRate@25%'],
            "PR-AUC": stats_30k['PR-AUC'],
            "Swipe@95%": stats_30k['SwipeRate@95%Likes']
        })

    # Save benchmark dataframe
    df_res = pd.DataFrame(rows)
    df_res.to_csv(str(results_dir / "phase2_siglip_benchmark_results.csv"), index=False)
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK RESULTS")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print(f"\nSaved benchmark results to {results_dir / 'phase2_siglip_benchmark_results.csv'}")

if __name__ == "__main__":
    main()
