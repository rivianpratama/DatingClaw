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
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Stratified Sampler (64 Likes : 448 Passes, B=512)
# ==============================================================================

class StratifiedSwipeSampler(Sampler):
    """Enforces exact 64 Likes : 448 Passes (B=512) per mini-batch."""
    def __init__(self, labels: np.ndarray, batch_size: int = 512, pos_per_batch: int = 64):
        self.pos_idx = np.where(labels == 1)[0]
        self.neg_idx = np.where(labels == 0)[0]
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = batch_size - pos_per_batch
        self.num_batches = len(self.neg_idx) // self.neg_per_batch

    def __iter__(self):
        pos_perm = np.random.permutation(self.pos_idx)
        neg_perm = np.random.permutation(self.neg_idx)
        for b in range(self.num_batches):
            neg_slice = neg_perm[b * self.neg_per_batch : (b + 1) * self.neg_per_batch]
            pos_indices = np.arange(b * self.pos_per_batch, (b + 1) * self.pos_per_batch) % len(pos_perm)
            pos_slice = pos_perm[pos_indices]
            batch = np.concatenate([pos_slice, neg_slice])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


# ==============================================================================
# 2. Shift-Invariant Top-Quartile Partial AUC Loss
# ==============================================================================

class ShiftInvariantTopQuartilePAUCLoss(nn.Module):
    def __init__(self, top_neg_frac: float = 0.30, beta: float = 3.0, margin: float = 0.30, smooth_eps: float = 0.03):
        super().__init__()
        self.top_neg_frac = top_neg_frac
        self.beta = beta
        self.margin = margin
        self.smooth_eps = smooth_eps

    def forward(self, z_proxy: torch.Tensor, z_train: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = targets.view(-1)
        z_proxy = z_proxy.view(-1)
        z_train = z_train.view(-1)

        # 1. Label-Smoothed BCE on context-adjusted logits (pos_weight = 1.0)
        y_smooth = targets * (1.0 - self.smooth_eps) + 0.5 * self.smooth_eps
        loss_bce = F.binary_cross_entropy_with_logits(z_train, y_smooth)

        # 2. Shift-Invariant Top-30% Hard-Negative Pairwise pAUC on pure visual logits
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        z_pos = z_proxy[pos_mask]
        z_neg = z_proxy[neg_mask]

        if z_pos.numel() == 0 or z_neg.numel() == 0:
            return loss_bce, torch.tensor(0.0, device=z_proxy.device), loss_bce

        k_hard = max(1, int(self.top_neg_frac * z_neg.numel()))
        z_neg_hard, _ = torch.topk(z_neg, k=k_hard, dim=0, sorted=False)

        # Pairwise matrix: [N_pos, 1] - [1, k_hard] -> [N_pos, k_hard]
        diff = z_pos.unsqueeze(1) - z_neg_hard.unsqueeze(0) - self.margin
        loss_pauc = (F.softplus(-self.beta * diff) / self.beta).mean()

        total_loss = 0.55 * loss_pauc + 0.40 * loss_bce
        return total_loss, loss_pauc, loss_bce


# ==============================================================================
# 3. Orthogonal Residual Bilinear Fusion Net (SigLIP 1152D + ArcFace 512D)
# ==============================================================================

class OrthogonalBilinearFusionNet(nn.Module):
    """
    Inputs:
      x_sig:      [B, 1152] (L2-normalized SigLIP-384)
      x_face:     [B, 512]  (L2-normalized InsightFace ArcFace)
      face_valid: [B, 1]    (1.0 if face detected, 0.0 otherwise)
    Output:
      h_fused:    [B, 256]  (Unit-norm embedding on S^255)
    """
    def __init__(self, sig_dim: int = 1152, face_dim: int = 512, latent_dim: int = 256, rank: int = 64):
        super().__init__()
        self.proj_sig = nn.Sequential(
            nn.Linear(sig_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.25)
        )
        self.proj_face = nn.Sequential(
            nn.Linear(face_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(0.25)
        )

        # Low-Rank Factorized Bilinear Interaction (Rank = 64)
        self.U_sig = nn.Linear(sig_dim, rank, bias=False)
        self.U_face = nn.Linear(face_dim, rank, bias=False)
        self.W_up = nn.Sequential(nn.Linear(rank, latent_dim, bias=False), nn.LayerNorm(latent_dim))
        nn.init.normal_(self.W_up[0].weight, std=0.01)

        # Dynamic Gates + Channel Scaling for Orthogonal Face & Bilinear Streams
        self.gate_ortho = nn.Linear(latent_dim * 2 + 1, latent_dim)
        self.gate_bilin = nn.Linear(latent_dim * 2 + 1, latent_dim)
        nn.init.constant_(self.gate_ortho.bias, -1.5)
        nn.init.constant_(self.gate_bilin.bias, -2.0)

        self.gamma_1 = nn.Parameter(torch.full((1, latent_dim), 0.20))
        self.gamma_2 = nn.Parameter(torch.full((1, latent_dim), 0.15))

    def forward(self, x_sig: torch.Tensor, x_face: torch.Tensor, face_valid: torch.Tensor) -> torch.Tensor:
        # 1. Primary aesthetic anchor on S^255
        h_siglip = F.normalize(self.proj_sig(x_sig), p=2, dim=-1)

        # 2. Gram-Schmidt Orthogonal Projection of ArcFace (keep raw sin(theta) magnitude!)
        h_face = F.normalize(self.proj_face(x_face), p=2, dim=-1) * face_valid
        proj_coeff = (h_face * h_siglip).sum(dim=-1, keepdim=True)
        h_face_ortho = h_face - proj_coeff * h_siglip  # Strictly orthogonal to h_siglip

        # 3. Low-Rank Factorized Bilinear Interaction
        h_bilinear = self.W_up(F.gelu(self.U_sig(x_sig)) * F.gelu(self.U_face(x_face))) * face_valid

        # 4. Context-Gated Orthogonal Residual Combination on S^255
        alpha_1 = self.gamma_1 * torch.sigmoid(self.gate_ortho(torch.cat([h_siglip, h_face_ortho, face_valid], dim=-1))) * face_valid
        alpha_2 = self.gamma_2 * torch.sigmoid(self.gate_bilin(torch.cat([h_siglip, h_bilinear, face_valid], dim=-1))) * face_valid

        h_fused = F.normalize(h_siglip + alpha_1 * h_face_ortho + alpha_2 * h_bilinear, p=2, dim=-1)
        return h_fused


# ==============================================================================
# 4. Circadian Context Branch
# ==============================================================================

class CircadianContextBranch(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


# ==============================================================================
# 5. Tangent-Space Operations & Multi-Proxy Regularization
# ==============================================================================

def project_proxy_gradients_to_tangent_space(proxies_pos: nn.Parameter, proxies_neg: nn.Parameter):
    """Call AFTER loss.backward() and BEFORE optimizer.step()."""
    with torch.no_grad():
        for p in (proxies_pos, proxies_neg):
            if p.grad is not None:
                # Remove radial component: g_tangent = g - <g, p> * p (assuming p is unit norm)
                radial = (p.grad * p).sum(dim=-1, keepdim=True)
                p.grad.sub_(radial * p)

def retract_proxies_to_hypersphere(proxies_pos: nn.Parameter, proxies_neg: nn.Parameter):
    """Call immediately AFTER optimizer.step()."""
    with torch.no_grad():
        proxies_pos.copy_(F.normalize(proxies_pos, p=2, dim=-1))
        proxies_neg.copy_(F.normalize(proxies_neg, p=2, dim=-1))

def compute_multi_proxy_logit_and_reg(
    h_fused: torch.Tensor,          # [B, 256], L2-normalized
    proxies_pos: torch.Tensor,      # [K=12, 256], L2-normalized
    proxies_neg: torch.Tensor,      # [M=20, 256], L2-normalized
    proxies_pos_init: torch.Tensor, # [K=12, 256], frozen K-Means init buffer
    tau: float,                     # Cosine annealed 0.15 -> 0.07
    margin: float = 0.15,
    gamma: float = 16.0,
    bias_b0: torch.Tensor = None,
    p_drop: float = 0.20,
    training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    p_pos = F.normalize(proxies_pos, p=2, dim=-1)
    p_neg = F.normalize(proxies_neg, p=2, dim=-1)

    cos_pos = torch.matmul(h_fused, p_pos.t())  # [B, K]
    cos_neg = torch.matmul(h_fused, p_neg.t())  # [B, M]

    if training and p_drop > 0.0:
        # Per-sample mask [B, K], protect the closest positive proxy (top-1 argmax)
        drop_mask_pos = torch.rand_like(cos_pos) < p_drop
        best_pos_idx = cos_pos.argmax(dim=-1, keepdim=True)
        drop_mask_pos.scatter_(1, best_pos_idx, False)
        cos_pos = cos_pos.masked_fill(drop_mask_pos, -1e4)

        # Per-sample mask [B, M], protect the closest negative proxy
        drop_mask_neg = torch.rand_like(cos_neg) < p_drop
        best_neg_idx = cos_neg.argmax(dim=-1, keepdim=True)
        drop_mask_neg.scatter_(1, best_neg_idx, False)
        cos_neg = cos_neg.masked_fill(drop_mask_neg, -1e4)

    s_pos = tau * torch.logsumexp((cos_pos - margin) / tau, dim=-1)
    s_neg = tau * torch.logsumexp(cos_neg / tau, dim=-1)
    z_proxy = gamma * (s_pos - s_neg) + (bias_b0 if bias_b0 is not None else 0.0)

    # Regularization 1: Elastic Tether to initial Spherical K-Means centroids
    loss_tether = 0.10 * (1.0 - (p_pos * proxies_pos_init).sum(dim=-1)).mean()

    # Regularization 2: SoftTriple Merge Tether on P+ (Singularity-Protected clamp)
    sim_pos_matrix = torch.matmul(p_pos, p_pos.t())  # [K, K]
    triu_mask = torch.triu(torch.ones_like(sim_pos_matrix, dtype=torch.bool), diagonal=1)
    dist_sq = torch.clamp(2.0 - 2.0 * sim_pos_matrix[triu_mask], min=1e-6)
    loss_merge = 0.02 * torch.sqrt(dist_sq).mean()

    # Regularization 3: Weak Repulsion on P- so negative clusters cover the Pass space
    sim_neg_matrix = torch.matmul(p_neg, p_neg.t())  # [M, M]
    triu_neg = torch.triu(torch.ones_like(sim_neg_matrix, dtype=torch.bool), diagonal=1)
    loss_neg_repel = 0.05 * F.relu(sim_neg_matrix[triu_neg] - 0.30).pow(2).mean()

    loss_reg = loss_tether + loss_merge + loss_neg_repel
    return z_proxy, loss_reg


# ==============================================================================
# 6. Feature Extraction & Dataset
# ==============================================================================

def extract_context_features(store_paths, csv_path):
    df = pd.read_csv(csv_path)
    df['dt'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['dt'].dt.hour + df['dt'].dt.minute / 60.0
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    time_diff = df['dt'].diff().dt.total_seconds().fillna(0)
    is_new = (time_diff >= 20 * 60) | (time_diff < 0)
    df['session_id'] = is_new.cumsum()
    df['card_idx'] = df.groupby('session_id').cumcount() + 1
    df['log_card_idx'] = np.log1p(df['card_idx'])

    df['binary_like'] = (df['rating'] >= 50).astype(float)
    df['trailing_25_like_rate'] = df['binary_like'].shift(1).rolling(25, min_periods=1).mean().fillna(0.08)

    feat_map = {}
    for _, row in df.iterrows():
        fname = Path(str(row['filename'])).name
        feat_map[fname] = [
            float(row['sin_hour']),
            float(row['cos_hour']),
            float(row['log_card_idx']),
            float(row['trailing_25_like_rate'])
        ]

    context_list = []
    default_feat = [0.0, 1.0, float(np.log1p(25)), 0.08]
    for p in store_paths:
        fname = Path(p).name
        context_list.append(feat_map.get(fname, default_feat))

    return np.array(context_list, dtype=np.float32)


class SwipeDataset(Dataset):
    def __init__(self, x_sig, x_face, face_valid, context, labels):
        self.x_sig = torch.tensor(x_sig, dtype=torch.float32)
        self.x_face = torch.tensor(x_face, dtype=torch.float32)
        self.face_valid = torch.tensor(face_valid, dtype=torch.float32)
        self.context = torch.tensor(context, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x_sig[idx], self.x_face[idx], self.face_valid[idx], self.context[idx], self.labels[idx]


# ==============================================================================
# 7. Evaluation Function
# ==============================================================================

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

    # Swipe Rate @ 90% and 95% Likes
    cum_likes = np.cumsum(sorted_y)
    cutoff_90 = np.where(cum_likes >= int(np.ceil(0.90 * n_likes)))[0][0]
    swipe_rate_90 = (cutoff_90 + 1) / n_total * 100

    cutoff_95 = np.where(cum_likes >= int(np.ceil(0.95 * n_likes)))[0][0]
    swipe_rate_95 = (cutoff_95 + 1) / n_total * 100

    return {
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Likes@20%Cap": f"{likes_20}/{n_likes} ({rec_20:.1f}%)",
        "HitRate@20%": f"{hit_rate_20:.1f}%",
        "Likes@25%Cap": f"{likes_25}/{n_likes} ({rec_25:.1f}%)",
        "HitRate@25%": f"{hit_rate_25:.1f}%",
        "Likes_25_int": likes_25,
        "SwipeRate@90%": f"{swipe_rate_90:.1f}%",
        "SwipeRate@95%": f"{swipe_rate_95:.1f}%",
    }


# ==============================================================================
# 8. Training Pipeline
# ==============================================================================

def train_regularized_subcenter(
    X_sig_tr, X_face_tr, y_tr, C_tr,
    X_sig_val, X_face_val, y_val, C_val,
    K=12, M=20, epochs=12, batch_size=512,
    save_path="models/subcenter_phase2/best_subcenter_pauc_regularized.pt"
):
    val_face_valid = np.ones((len(y_val), 1), dtype=np.float32)
    tr_face_valid = np.ones((len(y_tr), 1), dtype=np.float32)

    # Datasets
    train_dataset = SwipeDataset(X_sig_tr, X_face_tr, tr_face_valid, C_tr, y_tr)
    val_dataset = SwipeDataset(X_sig_val, X_face_val, val_face_valid, C_val, y_val)

    sampler = StratifiedSwipeSampler(y_tr, batch_size=batch_size, pos_per_batch=64)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # 1. Warm-start Orthogonal Fusion Net (2 epochs with BCE loss)
    print("--- Phase 1: Warm-starting Orthogonal Bilinear Fusion (2 Epochs) ---")
    fusion_net = OrthogonalBilinearFusionNet(sig_dim=1152, face_dim=512, latent_dim=256, rank=64).to(device)
    warm_head = nn.Linear(256, 1).to(device)
    warm_opt = optim.AdamW(list(fusion_net.parameters()) + list(warm_head.parameters()), lr=1e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(2):
        fusion_net.train()
        warm_head.train()
        for b_sig, b_face, b_fv, _, b_y in train_loader:
            b_sig, b_face, b_fv, b_y = b_sig.to(device), b_face.to(device), b_fv.to(device), b_y.to(device)
            warm_opt.zero_grad()
            emb = fusion_net(b_sig, b_face, b_fv)
            loss = bce_loss_fn(warm_head(emb).view(-1), b_y)
            loss.backward()
            warm_opt.step()

    # 2. Extract unit-norm embeddings and cluster proxies via Spherical K-Means
    print(f"\n--- Phase 2: Seeding P+ (K={K}) and P- (M={M}) via Spherical K-Means ---")
    fusion_net.eval()
    all_embs = []
    extract_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for b_sig, b_face, b_fv, _, _ in extract_loader:
            b_sig, b_face, b_fv = b_sig.to(device), b_face.to(device), b_fv.to(device)
            emb = fusion_net(b_sig, b_face, b_fv)
            all_embs.append(emb.cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)

    pos_embs = all_embs[y_tr == 1.0]
    neg_embs = all_embs[y_tr == 0.0]

    km_pos = KMeans(n_clusters=K, n_init=10, random_state=42).fit(pos_embs)
    pos_centroids = km_pos.cluster_centers_ / np.linalg.norm(km_pos.cluster_centers_, axis=-1, keepdims=True)

    km_neg = KMeans(n_clusters=M, n_init=10, random_state=42).fit(neg_embs)
    neg_centroids = km_neg.cluster_centers_ / np.linalg.norm(km_neg.cluster_centers_, axis=-1, keepdims=True)

    # Initialize Proxy Parameters
    proxies_pos = nn.Parameter(torch.tensor(pos_centroids, dtype=torch.float32, device=device))
    proxies_neg = nn.Parameter(torch.tensor(neg_centroids, dtype=torch.float32, device=device))
    proxies_pos_init = torch.tensor(pos_centroids, dtype=torch.float32, device=device) # Frozen anchor

    # Prior logit bias b0 = ln(p / (1 - p)) = ln(0.115 / 0.885) = -2.04
    bias_b0 = nn.Parameter(torch.tensor([-2.04], dtype=torch.float32, device=device))
    context_net = CircadianContextBranch(in_features=4, hidden=16).to(device)

    # 3. Parameter Groups with Separated LR and Zero Weight Decay on Proxies
    optimizer = optim.AdamW([
        {'params': [proxies_pos, proxies_neg], 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': list(fusion_net.parameters()) + list(context_net.parameters()) + [bias_b0], 'lr': 5e-4, 'weight_decay': 0.05}
    ])

    loss_fn = ShiftInvariantTopQuartilePAUCLoss(top_neg_frac=0.30, beta=3.0, margin=0.30, smooth_eps=0.03)

    print(f"\n--- Phase 3: Regularized Multi-Proxy Training with Tangent-Space Optimization ({epochs} Epochs) ---")
    best_val_score = -float('inf')
    best_metrics = {}
    patience = 4
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        fusion_net.train()
        context_net.train()

        # Cosine anneal temperature tau from 0.15 down to 0.07 over first 5 epochs
        if epoch <= 5:
            tau = 0.07 + 0.5 * (0.15 - 0.07) * (1.0 + np.cos(np.pi * (epoch - 1) / 5.0))
        else:
            tau = 0.07

        total_loss_sum = 0.0
        pauc_loss_sum = 0.0
        bce_loss_sum = 0.0
        reg_loss_sum = 0.0
        n_batches = 0

        for b_sig, b_face, b_fv, b_c, b_y in train_loader:
            b_sig, b_face, b_fv, b_c, b_y = (
                b_sig.to(device), b_face.to(device), b_fv.to(device), b_c.to(device), b_y.to(device)
            )

            optimizer.zero_grad()

            # Forward
            h_fused = fusion_net(b_sig, b_face, b_fv)
            z_proxy, loss_reg = compute_multi_proxy_logit_and_reg(
                h_fused, proxies_pos, proxies_neg, proxies_pos_init,
                tau=tau, margin=0.15, gamma=16.0, bias_b0=bias_b0,
                p_drop=0.20, training=True
            )
            g_context = context_net(b_c).squeeze(-1)
            z_train = z_proxy + g_context

            # Combined loss: 0.55 pAUC + 0.40 BCE + 0.05 Reg
            total_loss, l_pauc, l_bce = loss_fn(z_proxy, z_train, b_y)
            loss_with_reg = total_loss + 0.05 * loss_reg

            loss_with_reg.backward()

            # Tangent-Space Projection BEFORE optimizer.step()
            project_proxy_gradients_to_tangent_space(proxies_pos, proxies_neg)

            optimizer.step()

            # Retract Proxies to Hypersphere IMMEDIATELY AFTER optimizer.step()
            retract_proxies_to_hypersphere(proxies_pos, proxies_neg)

            total_loss_sum += loss_with_reg.item()
            pauc_loss_sum += l_pauc.item()
            bce_loss_sum += l_bce.item()
            reg_loss_sum += loss_reg.item()
            n_batches += 1

        # Validation Evaluation (Pure Visual Inference, zero context)
        fusion_net.eval()
        val_preds = []
        with torch.no_grad():
            for b_sig, b_face, b_fv, _, _ in val_loader:
                b_sig, b_face, b_fv = b_sig.to(device), b_face.to(device), b_fv.to(device)
                h_fused = fusion_net(b_sig, b_face, b_fv)
                z_proxy, _ = compute_multi_proxy_logit_and_reg(
                    h_fused, proxies_pos, proxies_neg, proxies_pos_init,
                    tau=0.07, margin=0.15, gamma=16.0, bias_b0=bias_b0,
                    p_drop=0.0, training=False
                )
                val_preds.append(z_proxy.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_m = evaluate_metrics(val_preds, y_val)

        # Smooth tie-breaking metric: val_likes_at_25pct + 10.0 * val_pr_auc
        val_score = val_m["Likes_25_int"] + 10.0 * val_m["PR-AUC"]

        print(
            f"Epoch {epoch:2d}/{epochs} (tau={tau:.3f}) | "
            f"Loss: {total_loss_sum/n_batches:.4f} (pAUC: {pauc_loss_sum/n_batches:.4f}, BCE: {bce_loss_sum/n_batches:.4f}, Reg: {reg_loss_sum/n_batches:.4f}) | "
            f"Val Likes@25%: {val_m['Likes@25%Cap']} | Val PR-AUC: {val_m['PR-AUC']:.4f} | ROC: {val_m['ROC-AUC']:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_metrics = val_m
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
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
                'K': K,
                'M': M,
                'tau': 0.07,
                'margin': 0.15,
                'gamma': 16.0
            }, save_path)
            print(f"  >>> Best checkpoint saved at epoch {epoch} (Val Likes@25%: {val_m['Likes@25%Cap']}, PR-AUC: {val_m['PR-AUC']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement for {patience} epochs. Best epoch was {best_epoch}.")
                break

    print(f"\nTraining Complete! Best Epoch: {best_epoch} | Best Val Score: {best_val_score:.2f}")
    return best_epoch, best_metrics


def predict_visual_logits(model_checkpoint_path, X_sig, X_face, face_valid=None, batch_size=1024):
    ckpt = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    fusion_net = OrthogonalBilinearFusionNet(sig_dim=1152, face_dim=512, latent_dim=256, rank=64).to(device)
    fusion_net.load_state_dict(ckpt['fusion_net'])
    fusion_net.eval()

    proxies_pos = ckpt['proxies_pos'].to(device)
    proxies_neg = ckpt['proxies_neg'].to(device)
    proxies_pos_init = ckpt['proxies_pos_init'].to(device)
    bias_b0 = ckpt['bias_b0'].to(device)

    if face_valid is None:
        face_valid = np.ones((len(X_sig), 1), dtype=np.float32)

    t_sig = torch.tensor(X_sig, dtype=torch.float32)
    t_face = torch.tensor(X_face, dtype=torch.float32)
    t_fv = torch.tensor(face_valid, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(t_sig, t_face, t_fv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for b_sig, b_face, b_fv in loader:
            b_sig, b_face, b_fv = b_sig.to(device), b_face.to(device), b_fv.to(device)
            h_fused = fusion_net(b_sig, b_face, b_fv)
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
    print("STEP 1: REGULARIZED 30K SUB-CENTER MULTI-PROXY WITH SHIFT-INVARIANT pAUC")
    print("=" * 80)

    # 1. Load 30k pure dataset (29,582 profiles)
    ref_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    sig_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))

    y_30k = (ref_30k.ratings >= 50.0).astype(int)
    X_face_30k = ref_30k.embeddings
    X_sig_30k = sig_30k['embeddings']
    paths_30k = ref_30k.paths

    print(f"Total 30k profiles: {len(y_30k)} | Likes: {int(y_30k.sum())} ({y_30k.sum()/len(y_30k)*100:.2f}%)")

    # 2. Extract Context Features for 30k
    print("Extracting circadian & session context features...")
    C_30k = extract_context_features(paths_30k, csv_path)

    # 3. Chronological Split: 0 to 26,500 Train (26.5k), 26,500 to 29,582 Val (~3.08k)
    train_cutoff = 26500
    X_sig_tr, X_face_tr, y_tr, C_tr = (
        X_sig_30k[:train_cutoff], X_face_30k[:train_cutoff], y_30k[:train_cutoff], C_30k[:train_cutoff]
    )
    X_sig_val, X_face_val, y_val, C_val = (
        X_sig_30k[train_cutoff:], X_face_30k[train_cutoff:], y_30k[train_cutoff:], C_30k[train_cutoff:]
    )

    print(f"Train Block: {len(y_tr)} profiles | Likes: {int(y_tr.sum())} ({y_tr.sum()/len(y_tr)*100:.2f}%)")
    print(f"Val Block (Early-Stop Monitor): {len(y_val)} profiles | Likes: {int(y_val.sum())} ({y_val.sum()/len(y_val)*100:.2f}%)")

    # 4. Train the 26.5k Model with Early Stopping
    save_path_26k = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt"
    best_epoch_26k, best_metrics_26k = train_regularized_subcenter(
        X_sig_tr, X_face_tr, y_tr, C_tr,
        X_sig_val, X_face_val, y_val, C_val,
        K=12, M=20, epochs=12, batch_size=512,
        save_path=str(save_path_26k)
    )

    # 5. Train the Companion Model on Full 30k for exactly best_epoch epochs
    print("\n" + "=" * 80)
    print(f"STEP 2: TRAINING COMPANION MODEL ON FULL 0–29.5k FOR EXACTLY E*={best_epoch_26k} EPOCHS")
    print("=" * 80)
    save_path_30k = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_30k_full.pt"
    train_regularized_subcenter(
        X_sig_30k, X_face_30k, y_30k, C_30k,
        X_sig_val, X_face_val, y_val, C_val,
        K=12, M=20, epochs=best_epoch_26k, batch_size=512,
        save_path=str(save_path_30k)
    )

    # 6. Benchmark Evaluation on Untouched Holdout 2 (30,055 to 33,629)
    print("\n" + "=" * 80)
    print("STEP 3: TOURNAMENT BENCHMARK ON UNTOUCHED HOLDOUT 2 (30k-33.6k)")
    print("=" * 80)

    ref_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    sig_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))

    y_test = (ref_holdout.ratings >= 50.0).astype(int)
    X_face_test = ref_holdout.embeddings
    X_sig_test = sig_holdout['embeddings']

    n_test = len(y_test)
    n_test_likes = int(y_test.sum())
    k_25_test = int(np.round(0.25 * n_test))
    print(f"Holdout 2: {n_test} profiles | Likes: {n_test_likes} ({n_test_likes/n_test*100:.2f}%) | 25% Budget: {k_25_test} swipes\n")

    # Predictions
    preds_26k = predict_visual_logits(str(save_path_26k), X_sig_test, X_face_test)
    preds_30k = predict_visual_logits(str(save_path_30k), X_sig_test, X_face_test)

    m_26k = evaluate_metrics(preds_26k, y_test)
    m_30k = evaluate_metrics(preds_30k, y_test)

    print(f"[Model 1] Regularized 26.5k pAUC Multi-Proxy: Likes@25% = {m_26k['Likes@25%Cap']}, PR-AUC = {m_26k['PR-AUC']}, ROC = {m_26k['ROC-AUC']}")
    print(f"[Model 2] Regularized 30k Full pAUC Multi-Proxy: Likes@25% = {m_30k['Likes@25%Cap']}, PR-AUC = {m_30k['PR-AUC']}, ROC = {m_30k['ROC-AUC']}")

    # Load Winning Multimodal MLP (30k Pure) for comparison and ensemble
    import joblib
    mlp_path = root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib"
    if mlp_path.exists():
        mlp_obj = joblib.load(mlp_path)
        mlp = mlp_obj['estimator'] if isinstance(mlp_obj, dict) else mlp_obj
        clip_holdout = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))
        X_clip_test = clip_holdout['embeddings']
        X_mlp_test = np.hstack([X_face_test, X_clip_test])
        preds_mlp = mlp.predict(X_mlp_test)
        m_mlp = evaluate_metrics(preds_mlp, y_test)
        print(f"[Baseline] Multimodal MLP (30k Pure): Likes@25% = {m_mlp['Likes@25%Cap']}, PR-AUC = {m_mlp['PR-AUC']}, ROC = {m_mlp['ROC-AUC']}")

        # Ensembles (Rank-Averaged)
        from scipy.stats import rankdata
        r_mlp = rankdata(preds_mlp) / len(preds_mlp)
        r_26k = rankdata(preds_26k) / len(preds_26k)
        r_30k = rankdata(preds_30k) / len(preds_30k)

        # Ensemble A: MLP (30k) + Regularized 26.5k
        ens_a = 0.50 * r_mlp + 0.50 * r_26k
        m_ens_a = evaluate_metrics(ens_a, y_test)
        print(f"[Ensemble A] Rank(MLP 30k + Reg 26.5k pAUC): Likes@25% = {m_ens_a['Likes@25%Cap']}, PR-AUC = {m_ens_a['PR-AUC']}, ROC = {m_ens_a['ROC-AUC']}")

        # Ensemble B: MLP (30k) + Regularized 30k Full
        ens_b = 0.50 * r_mlp + 0.50 * r_30k
        m_ens_b = evaluate_metrics(ens_b, y_test)
        print(f"[Ensemble B] Rank(MLP 30k + Reg 30k Full pAUC): Likes@25% = {m_ens_b['Likes@25%Cap']}, PR-AUC = {m_ens_b['PR-AUC']}, ROC = {m_ens_b['ROC-AUC']}")

        # Ensemble C: Triple Blend: MLP (30k) + Reg 26.5k + Reg 30k Full
        ens_c = 0.40 * r_mlp + 0.30 * r_26k + 0.30 * r_30k
        m_ens_c = evaluate_metrics(ens_c, y_test)
        print(f"[Ensemble C] Triple Blend (MLP + 26.5k + 30k): Likes@25% = {m_ens_c['Likes@25%Cap']}, PR-AUC = {m_ens_c['PR-AUC']}, ROC = {m_ens_c['ROC-AUC']}")


if __name__ == "__main__":
    main()
