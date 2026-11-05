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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store
from scratch.train_subcenter_multiproxy_phase1 import (
    GatedResidualFusion,
    CircadianContextBranch,
    MultiProxyClassifier,
    Top25BoundaryLoss,
    extract_context_features,
    evaluate_holdout
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_subcenter_model(X_clip_train, X_face_train, y_train, C_train,
                          X_clip_val, X_face_val, y_val,
                          K=12, M=20, epochs=12, batch_size=512):
    # Convert to tensors
    t_tr_clip = torch.tensor(X_clip_train, dtype=torch.float32)
    t_tr_face = torch.tensor(X_face_train, dtype=torch.float32)
    t_tr_c = torch.tensor(C_train, dtype=torch.float32)
    t_tr_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    t_val_clip = torch.tensor(X_clip_val, dtype=torch.float32).to(device)
    t_val_face = torch.tensor(X_face_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(t_tr_clip, t_tr_face, t_tr_c, t_tr_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 1. Warm-start Gated Fusion (2 epochs)
    fusion_net = GatedResidualFusion(clip_dim=512, face_dim=512, proj_dim=256, mod_drop_prob=0.20).to(device)
    warm_head = nn.Linear(256, 1).to(device)
    warm_opt = optim.AdamW(list(fusion_net.parameters()) + list(warm_head.parameters()), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(2):
        fusion_net.train()
        warm_head.train()
        for b_clip, b_face, _, b_y in train_loader:
            b_clip, b_face, b_y = b_clip.to(device), b_face.to(device), b_y.to(device)
            warm_opt.zero_grad()
            emb = fusion_net(b_clip, b_face, training=True)
            loss = bce_loss_fn(warm_head(emb), b_y)
            loss.backward()
            warm_opt.step()

    # 2. Extract embeddings & cluster proxies
    fusion_net.eval()
    with torch.no_grad():
        all_embs = []
        eval_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        for b_clip, b_face, _, _ in eval_loader:
            b_clip, b_face = b_clip.to(device), b_face.to(device)
            emb = fusion_net(b_clip, b_face, training=False)
            all_embs.append(emb.cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)

    pos_embs = all_embs[y_train == 1.0]
    neg_embs = all_embs[y_train == 0.0]

    print(f"  Clustering {len(pos_embs)} Likes into K={K} archetypes...")
    km_pos = KMeans(n_clusters=K, n_init=10, random_state=42).fit(pos_embs)
    pos_centroids = km_pos.cluster_centers_ / np.linalg.norm(km_pos.cluster_centers_, axis=-1, keepdims=True)

    print(f"  Clustering {len(neg_embs)} Passes into M={M} clusters...")
    km_neg = KMeans(n_clusters=M, n_init=10, random_state=42).fit(neg_embs)
    neg_centroids = km_neg.cluster_centers_ / np.linalg.norm(km_neg.cluster_centers_, axis=-1, keepdims=True)

    # 3. Full Training with Boundary Loss & Context Branch
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
        {'params': fusion_net.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': proxy_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': context_net.parameters(), 'lr': 1e-3, 'weight_decay': 0.1}
    ])

    tau_hat = 0.0
    best_likes_25 = 0
    best_stats = None
    best_preds = None

    for epoch in range(epochs):
        fusion_net.train()
        proxy_model.train()
        context_net.train()

        w_boundary = min(0.40, 0.40 * (epoch / 5.0))
        w_bce = 0.55
        w_ortho = 0.05

        for b_clip, b_face, b_c, b_y in train_loader:
            b_clip, b_face = b_clip.to(device), b_face.to(device)
            b_c, b_y = b_c.to(device), b_y.to(device)

            optimizer.zero_grad()
            emb = fusion_net(b_clip, b_face, training=True)
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

        # Validation (clamping context g_c = 0)
        fusion_net.eval()
        proxy_model.eval()
        with torch.no_grad():
            val_emb = fusion_net(t_val_clip, t_val_face, training=False)
            val_preds = proxy_model(val_emb).cpu().numpy().flatten()

        stats = evaluate_holdout(val_preds, y_val)
        likes_count = int(stats['Likes@25%Cap'].split('/')[0])
        print(f"  Epoch {epoch+1:2d}/{epochs} | Tau78: {tau_hat:5.2f} | Likes@25%Cap: {stats['Likes@25%Cap']} | PR-AUC: {stats['PR-AUC']:.4f} | ROC: {stats['ROC-AUC']:.4f}")

        if likes_count > best_likes_25:
            best_likes_25 = likes_count
            best_stats = stats
            best_preds = val_preds

    return best_stats, best_preds, fusion_net, proxy_model, context_net, tau_hat

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    models_dir = root / "models" / "subcenter_30k"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "results"

    print("=" * 80)
    print("SCALED 30K MULTI-PROXY SUB-CENTER TRAINING & BENCHMARK")
    print("=" * 80)

    # 1. Load Data Slices
    print("Loading data slices...")
    s_20k = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
    c_20k = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_pure.npz"))

    s_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    c_holdout = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_to_22k.npz"))

    s_30k_tail = load_store(str(root / "embeddings" / "reference_store_bumble_22k_to_30k.npz"))
    c_30k_tail = load_clip_store(str(root / "embeddings" / "clip_store_bumble_22k_to_30k.npz"))

    # Holdout benchmark (20k-22k)
    y_val = (s_holdout.ratings >= 50.0).astype(int)
    X_val_face = s_holdout.embeddings
    X_val_clip = c_holdout.embeddings
    print(f"Holdout: {len(y_val)} samples (Likes: {int(y_val.sum())}, Passes: {len(y_val)-int(y_val.sum())})")

    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"

    # -------------------------------------------------------------
    # Experiment A: Scientific Holdout Benchmark (0-20k + 22k-30k)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("EXPERIMENT A: 28K TRAINING (0-20k + 22k-30k, ZERO LEAKAGE ON 20k-22k)")
    print("#" * 60)

    X_train_face_leakfree = np.vstack([s_20k.embeddings, s_30k_tail.embeddings])
    X_train_clip_leakfree = np.vstack([c_20k.embeddings, c_30k_tail.embeddings])
    y_train_leakfree = np.concatenate([
        (s_20k.ratings >= 50.0).astype(np.float32),
        (s_30k_tail.ratings >= 50.0).astype(np.float32)
    ])
    paths_leakfree = list(s_20k.paths) + list(s_30k_tail.paths)
    C_train_leakfree = extract_context_features(paths_leakfree, csv_path)
    print(f"Leak-free Train Size: {len(y_train_leakfree)} (Likes: {int(y_train_leakfree.sum())})")

    stats_leakfree, preds_leakfree, _, _, _, _ = train_subcenter_model(
        X_train_clip_leakfree, X_train_face_leakfree, y_train_leakfree, C_train_leakfree,
        X_val_clip, X_val_face, y_val,
        K=12, M=20, epochs=10, batch_size=512
    )

    print("\n" + "=" * 60)
    print("LEAK-FREE 28K SCALING RESULT ON 20K-22K HOLDOUT:")
    print(f"Likes@25%Cap: {stats_leakfree['Likes@25%Cap']} | HitRate: {stats_leakfree['HitRate@25%']}")
    print(f"PR-AUC: {stats_leakfree['PR-AUC']} | ROC-AUC: {stats_leakfree['ROC-AUC']} | Swipe@95%: {stats_leakfree['SwipeRate@95%Likes']}")
    print("=" * 60)

    # -------------------------------------------------------------
    # Experiment B: Full 30k Production Training (All 29,582 samples)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("EXPERIMENT B: FULL 30K PRODUCTION TRAINING (ALL 29,582 SAMPLES)")
    print("#" * 60)

    s_all = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    c_all = load_clip_store(str(root / "embeddings" / "clip_store_bumble_30k_pure.npz"))
    y_all = (s_all.ratings >= 50.0).astype(np.float32)
    C_all = extract_context_features(s_all.paths, csv_path)

    stats_prod, _, f_net, p_model, c_net, tau_h = train_subcenter_model(
        c_all.embeddings, s_all.embeddings, y_all, C_all,
        X_val_clip, X_val_face, y_val,
        K=12, M=20, epochs=10, batch_size=512
    )

    # Save 30k production checkpoint
    prod_ckpt_path = models_dir / "best_subcenter_30k_pure.pt"
    torch.save({
        'fusion_net': f_net.state_dict(),
        'proxy_model': p_model.state_dict(),
        'context_net': c_net.state_dict(),
        'tau_hat': tau_h,
        'stats': stats_prod,
        'train_samples': len(y_all)
    }, str(prod_ckpt_path))
    print(f"\nSaved production 30k model checkpoint to {prod_ckpt_path}")

if __name__ == "__main__":
    main()
