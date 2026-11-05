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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# 1. Feature Representation: Asymmetric Gated Residual Fusion
# ------------------------------------------------------------------------------

class GatedResidualFusion(nn.Module):
    """
    Asymmetric Gated Residual Fusion:
    CLIP acts as primary anchor and gates when ArcFace facial geometry is allowed to contribute.
    """
    def __init__(self, clip_dim=512, face_dim=512, proj_dim=256, mod_drop_prob=0.20):
        super().__init__()
        self.proj_dim = proj_dim
        self.mod_drop_prob = mod_drop_prob

        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.face_proj = nn.Sequential(
            nn.Linear(face_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # Gate with multiplicative interaction: [h_clip, h_face, h_clip * h_face] -> 256D
        self.gate_fc = nn.Linear(proj_dim * 3, proj_dim)
        # Initialize gate bias to -1.5 so sigmoid(b_g) ~= 0.18 initially
        nn.init.constant_(self.gate_fc.bias, -1.5)
        nn.init.xavier_uniform_(self.gate_fc.weight)

        self.post_norm = nn.LayerNorm(proj_dim)

    def forward(self, x_clip, x_face, training=True):
        # L2-normalize inputs
        x_clip = F.normalize(x_clip, p=2, dim=-1)
        x_face = F.normalize(x_face, p=2, dim=-1)

        # Modality DropPath on face during training
        if training and self.mod_drop_prob > 0.0:
            mask = (torch.rand(x_face.size(0), 1, device=x_face.device) > self.mod_drop_prob).float()
            x_face = x_face * mask

        h_clip = self.clip_proj(x_clip)
        h_face = self.face_proj(x_face)

        interact = h_clip * h_face
        gate_input = torch.cat([h_clip, h_face, interact], dim=-1)
        alpha = torch.sigmoid(self.gate_fc(gate_input))

        fused = h_clip + alpha * h_face
        fused = self.post_norm(fused)
        fused = F.normalize(fused, p=2, dim=-1) # Project to S^255 unit hypersphere
        return fused

# ------------------------------------------------------------------------------
# 2. Circadian & Fatigue Context Offset Branch
# ------------------------------------------------------------------------------

class CircadianContextBranch(nn.Module):
    """
    Additive logit decomposition:
    g_phi(c) absorbs time-of-day and card fatigue variance during training.
    """
    def __init__(self, in_features=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, c):
        return self.net(c)

# ------------------------------------------------------------------------------
# 3. Multi-Proxy Sub-Center Classifier (SoftTriple Architecture)
# ------------------------------------------------------------------------------

class MultiProxyClassifier(nn.Module):
    def __init__(self, in_dim=256, num_pos_proxies=8, num_neg_proxies=16, tau=0.07, margin=0.15, gamma=16.0):
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
        # Ensure proxies are strictly unit normalized
        P_pos = F.normalize(self.pos_proxies, p=2, dim=-1) # [K, D]
        P_neg = F.normalize(self.neg_proxies, p=2, dim=-1) # [M, D]

        # Cosine similarities
        cos_pos = torch.matmul(fused_emb, P_pos.t()) # [B, K]
        cos_neg = torch.matmul(fused_emb, P_neg.t()) # [B, M]

        # Smooth-max aggregations
        # s+(x) = tau * ln( sum_k exp( (cos(x, p_k+) - m) / tau ) )
        s_pos = self.tau * torch.logsumexp((cos_pos - self.margin) / self.tau, dim=-1, keepdim=True)
        # s-(x) = tau * ln( sum_j exp( cos(x, p_j-) / tau ) )
        s_neg = self.tau * torch.logsumexp(cos_neg / self.tau, dim=-1, keepdim=True)

        z_proxy = self.gamma * (s_pos - s_neg) + self.b_0
        return z_proxy

    def ortho_loss(self):
        """Orthogonality penalty between positive proxies to prevent sub-center collapse."""
        P_pos = F.normalize(self.pos_proxies, p=2, dim=-1)
        cos_matrix = torch.matmul(P_pos, P_pos.t()) # [K, K]
        # Zero out diagonal
        mask = ~torch.eye(self.K, dtype=torch.bool, device=P_pos.device)
        off_diag = cos_matrix[mask]
        penalty = torch.clamp(off_diag - 0.25, min=0.0) ** 2
        return penalty.mean()

# ------------------------------------------------------------------------------
# 4. Top-25% Quantile Boundary Loss
# ------------------------------------------------------------------------------

class Top25BoundaryLoss(nn.Module):
    def __init__(self, w_pos=6.0, beta=4.0, m_pos=0.25, m_neg=0.15):
        super().__init__()
        self.w_pos = w_pos
        self.beta = beta
        self.m_pos = m_pos
        self.m_neg = m_neg

    def forward(self, logits, targets, tau_hat):
        # tau_hat is detached 78th percentile threshold
        diff_pos = logits - tau_hat - self.m_pos
        loss_pos = self.w_pos * targets * (1.0 / self.beta) * F.softplus(-self.beta * diff_pos)

        diff_neg = logits - tau_hat + self.m_neg
        loss_neg = (1.0 - targets) * (1.0 / self.beta) * F.softplus(self.beta * diff_neg)

        return (loss_pos + loss_neg).mean()

# ------------------------------------------------------------------------------
# 5. Context Feature Extraction Helper
# ------------------------------------------------------------------------------

def extract_context_features(store_paths, csv_path):
    df = pd.read_csv(csv_path)
    df['dt'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['dt'].dt.hour + df['dt'].dt.minute / 60.0
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Session tracking: gap >= 20 mins
    time_diff = df['dt'].diff().dt.total_seconds().fillna(0)
    is_new = (time_diff >= 20 * 60) | (time_diff < 0)
    df['session_id'] = is_new.cumsum()
    df['card_idx'] = df.groupby('session_id').cumcount() + 1
    df['log_card_idx'] = np.log1p(df['card_idx'])

    # Trailing 25 like rate (leak free: shift 1)
    df['binary_like'] = (df['rating'] >= 50).astype(float)
    df['trailing_25_like_rate'] = df['binary_like'].shift(1).rolling(25, min_periods=1).mean().fillna(0.08)

    # Build mapping from filename -> [sin_hour, cos_hour, log_card_idx, trailing_25_like_rate]
    feat_map = {}
    for idx, row in df.iterrows():
        fname = str(row['filename'])
        feat_map[fname] = [
            float(row['sin_hour']),
            float(row['cos_hour']),
            float(row['log_card_idx']),
            float(row['trailing_25_like_rate'])
        ]

    # Map for store_paths
    context_list = []
    default_feat = [0.0, 1.0, np.log1p(25), 0.08] # 2:00 PM baseline
    for p in store_paths:
        fname = Path(p).name
        context_list.append(feat_map.get(fname, default_feat))

    return np.array(context_list, dtype=np.float32)

# ------------------------------------------------------------------------------
# 6. Evaluation Function
# ------------------------------------------------------------------------------

def evaluate_holdout(preds, y_true):
    n_total = len(y_true)
    n_likes = int(np.sum(y_true))
    n_dislikes = int(n_total - n_likes)

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

    # Swipe Rate @ 95% Likes
    cum_likes = np.cumsum(sorted_y)
    cutoff_95 = np.where(cum_likes >= int(np.ceil(0.95 * n_likes)))[0][0]
    swipe_rate_95 = (cutoff_95 + 1) / n_total * 100

    return {
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc, 4),
        "Likes@20%Cap": f"{likes_20}/{n_likes} ({rec_20:.1f}%)",
        "HitRate@20%": f"{hit_rate_20:.1f}%",
        "Likes@25%Cap": f"{likes_25}/{n_likes} ({rec_25:.1f}%)",
        "HitRate@25%": f"{hit_rate_25:.1f}%",
        "SwipeRate@95%Likes": f"{swipe_rate_95:.1f}%"
    }

# ------------------------------------------------------------------------------
# 7. Main Training Pipeline
# ------------------------------------------------------------------------------

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    models_dir = root / "models" / "subcenter_phase1"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: MULTI-PROXY SUB-CENTER + TOP-25% BOUNDARY + CIRCADIAN DE-BIASING")
    print("=" * 80)

    # 1. Load Data
    print("Loading 20k pure training dataset...")
    train_face = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
    train_clip = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_pure.npz"))
    y_train = (train_face.ratings >= 50.0).astype(np.float32)

    X_train_clip = train_clip.embeddings
    X_train_face = train_face.embeddings
    print(f"Train samples: {len(y_train)} | Likes: {int(y_train.sum())} | Passes: {int((1-y_train).sum())}")

    print("Loading 20k-22k holdout benchmark...")
    val_face = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    val_clip = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_to_22k.npz"))
    y_val = (val_face.ratings >= 50.0).astype(int)
    X_val_clip = val_clip.embeddings
    X_val_face = val_face.embeddings
    print(f"Holdout samples: {len(y_val)} | Likes: {int(np.sum(y_val))} | Passes: {int(len(y_val)-np.sum(y_val))}")

    # Extract context features
    print("Extracting circadian & fatigue context features...")
    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"
    C_train = extract_context_features(train_face.paths, csv_path)
    # At inference/validation, context is clamped to 0 (Option A)
    C_val = np.zeros((len(y_val), 4), dtype=np.float32)

    # Convert to Tensors
    t_train_clip = torch.tensor(X_train_clip, dtype=torch.float32)
    t_train_face = torch.tensor(X_train_face, dtype=torch.float32)
    t_train_c = torch.tensor(C_train, dtype=torch.float32)
    t_train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    t_val_clip = torch.tensor(X_val_clip, dtype=torch.float32).to(device)
    t_val_face = torch.tensor(X_val_face, dtype=torch.float32).to(device)

    # Create DataLoader
    batch_size = 512
    train_dataset = TensorDataset(t_train_clip, t_train_face, t_train_c, t_train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # -------------------------------------------------------------
    # Step 1: Warm-start Gated Fusion for 2 Epochs
    # -------------------------------------------------------------
    print("\n[Step 1/3] Warm-starting Asymmetric Gated Fusion (2 Epochs)...")
    fusion_net = GatedResidualFusion(clip_dim=512, face_dim=512, proj_dim=256, mod_drop_prob=0.20).to(device)
    warm_head = nn.Linear(256, 1).to(device)
    warm_opt = optim.AdamW(list(fusion_net.parameters()) + list(warm_head.parameters()), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(2):
        fusion_net.train()
        warm_head.train()
        total_loss = 0.0
        for b_clip, b_face, b_c, b_y in train_loader:
            b_clip, b_face, b_y = b_clip.to(device), b_face.to(device), b_y.to(device)
            warm_opt.zero_grad()
            emb = fusion_net(b_clip, b_face, training=True)
            logits = warm_head(emb)
            loss = bce_loss_fn(logits, b_y)
            loss.backward()
            warm_opt.step()
            total_loss += loss.item()
        print(f"  Warmup Epoch {epoch+1}/2 Loss: {total_loss/len(train_loader):.4f}")

    # -------------------------------------------------------------
    # Step 2: Spherical K-Means Proxy Initialization (K=8, M=16)
    # -------------------------------------------------------------
    print("\n[Step 2/3] Extracting 256D embeddings and seeding Spherical K-Means Proxies...")
    fusion_net.eval()
    with torch.no_grad():
        all_train_embs = []
        eval_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
        for b_clip, b_face, _, _ in eval_loader:
            b_clip, b_face = b_clip.to(device), b_face.to(device)
            emb = fusion_net(b_clip, b_face, training=False)
            all_train_embs.append(emb.cpu().numpy())
    all_train_embs = np.concatenate(all_train_embs, axis=0)

    pos_mask = (y_train == 1.0)
    neg_mask = (y_train == 0.0)
    pos_embs = all_train_embs[pos_mask]
    neg_embs = all_train_embs[neg_mask]

    print(f"  Clustering {len(pos_embs)} Likes into K=8 positive archetypes...")
    kmeans_pos = KMeans(n_clusters=8, n_init=10, random_state=42).fit(pos_embs)
    pos_centroids = kmeans_pos.cluster_centers_
    pos_centroids = pos_centroids / np.linalg.norm(pos_centroids, axis=-1, keepdims=True)

    print(f"  Clustering {len(neg_embs)} Passes into M=16 negative failure clusters...")
    kmeans_neg = KMeans(n_clusters=16, n_init=10, random_state=42).fit(neg_embs)
    neg_centroids = kmeans_neg.cluster_centers_
    neg_centroids = neg_centroids / np.linalg.norm(neg_centroids, axis=-1, keepdims=True)

    # Initialize MultiProxy Classifier & Context Branch
    proxy_model = MultiProxyClassifier(
        in_dim=256,
        num_pos_proxies=8,
        num_neg_proxies=16,
        tau=0.07,
        margin=0.15,
        gamma=16.0
    ).to(device)
    proxy_model.init_proxies(pos_centroids, neg_centroids)

    context_net = CircadianContextBranch(in_features=4, hidden=16).to(device)
    boundary_loss_fn = Top25BoundaryLoss(w_pos=6.0, beta=4.0, m_pos=0.25, m_neg=0.15)

    # -------------------------------------------------------------
    # Step 3: Full End-to-End Training with Boundary & Context Loss
    # -------------------------------------------------------------
    print("\n[Step 3/3] Training Multi-Proxy Sub-Center Architecture (15 Epochs)...")
    optimizer = optim.AdamW([
        {'params': fusion_net.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4},
        {'params': proxy_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': context_net.parameters(), 'lr': 1e-3, 'weight_decay': 0.1} # Strongly regularized
    ])

    total_epochs = 15
    tau_hat = 0.0 # Initial EMA threshold
    tau_hat_ema_decay = 0.98

    best_likes_25 = 0
    best_stats = None

    for epoch in range(total_epochs):
        fusion_net.train()
        proxy_model.train()
        context_net.train()

        # Ramp boundary loss weight from 0.0 -> 0.40 over first 5 epochs
        w_boundary = min(0.40, 0.40 * (epoch / 5.0))
        w_bce = 0.55
        w_ortho = 0.05

        epoch_bce, epoch_bnd, epoch_ortho = 0.0, 0.0, 0.0

        for b_clip, b_face, b_c, b_y in train_loader:
            b_clip, b_face = b_clip.to(device), b_face.to(device)
            b_c, b_y = b_c.to(device), b_y.to(device)

            optimizer.zero_grad()

            # 1. Visual Feature & Proxy Logit
            emb = fusion_net(b_clip, b_face, training=True)
            z_proxy = proxy_model(emb)

            # 2. Context Logit Offset
            g_c = context_net(b_c)
            z_train = z_proxy + g_c

            # 3. Update EMA 78th-percentile cutline (with stop_gradient)
            with torch.no_grad():
                batch_q78 = torch.quantile(z_proxy.detach(), 0.78).item()
                if epoch == 0 and tau_hat == 0.0:
                    tau_hat = batch_q78
                else:
                    tau_hat = tau_hat_ema_decay * tau_hat + (1.0 - tau_hat_ema_decay) * batch_q78

            # 4. Losses
            # Label-smoothed BCE loss (epsilon = 0.03)
            y_smoothed = b_y * (1.0 - 0.03) + 0.5 * 0.03
            loss_bce = F.binary_cross_entropy_with_logits(z_train, y_smoothed)

            # Top-25% Quantile Boundary Loss
            loss_boundary = boundary_loss_fn(z_proxy, b_y, tau_hat)

            # Anti-Collapse Orthogonality Loss
            loss_ortho = proxy_model.ortho_loss()

            total_loss = w_bce * loss_bce + w_boundary * loss_boundary + w_ortho * loss_ortho
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(fusion_net.parameters()) + list(proxy_model.parameters()), 1.0)
            optimizer.step()

            epoch_bce += loss_bce.item()
            epoch_bnd += loss_boundary.item()
            epoch_ortho += loss_ortho.item()

        # Validation on Holdout (Clamping context g_c = 0)
        fusion_net.eval()
        proxy_model.eval()
        with torch.no_grad():
            val_emb = fusion_net(t_val_clip, t_val_face, training=False)
            val_preds = proxy_model(val_emb).cpu().numpy().flatten()

        stats = evaluate_holdout(val_preds, y_val)
        likes_25_count = int(stats['Likes@25%Cap'].split('/')[0])

        print(f"Epoch {epoch+1:2d}/{total_epochs} | Tau_78: {tau_hat:5.2f} | BCE: {epoch_bce/len(train_loader):.3f} | Bnd: {epoch_bnd/len(train_loader):.3f} | Likes@25%Cap: {stats['Likes@25%Cap']} | PR-AUC: {stats['PR-AUC']:.4f} | ROC: {stats['ROC-AUC']:.4f}")

        if likes_25_count > best_likes_25:
            best_likes_25 = likes_25_count
            best_stats = stats
            # Save best checkpoint
            torch.save({
                'fusion_net': fusion_net.state_dict(),
                'proxy_model': proxy_model.state_dict(),
                'context_net': context_net.state_dict(),
                'tau_hat': tau_hat,
                'stats': stats
            }, str(models_dir / "best_subcenter_phase1.pt"))

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE!")
    print(f"Previous Best Benchmark: 123 - 124 Likes (78.3% - 79.0%)")
    print(f"New Phase 1 Best:        {best_stats['Likes@25%Cap']} ({best_stats['HitRate@25%']} Hit Rate)")
    print(f"PR-AUC: {best_stats['PR-AUC']} | ROC-AUC: {best_stats['ROC-AUC']} | SwipeRate@95%Likes: {best_stats['SwipeRate@95%Likes']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
