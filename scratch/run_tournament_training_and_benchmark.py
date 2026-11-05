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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store
from face_similarity.regressor import load_regressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------------------
# 1. PyTorch Model Architectures (LayerNorm to prevent batch norm collapse)
# ------------------------------------------------------------------------------

class RobustScoringMLP(nn.Module):
    """Deep non-linear MLP with LayerNorm for stable ranking/classification."""
    def __init__(self, in_features=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# 2. Training Routines
# ------------------------------------------------------------------------------

def train_ranknet(X_train_s, y_train, epochs=15, batch_size=256, lr=1e-3):
    """Trains a RankNet model using pairwise logistic loss."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    num_pairs = min(len(pos_idx) * 40, 60000)
    
    model = RobustScoringMLP(in_features=X_train_s.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        p_sel = np.random.choice(pos_idx, size=num_pairs, replace=True)
        n_sel = np.random.choice(neg_idx, size=num_pairs, replace=True)
        
        for i in range(0, num_pairs, batch_size):
            pb = torch.tensor(X_train_s[p_sel[i:i+batch_size]], dtype=torch.float32).to(device)
            nb = torch.tensor(X_train_s[n_sel[i:i+batch_size]], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            sp = model(pb)
            sn = model(nb)
            diff = sp - sn
            targets = torch.ones_like(diff)
            loss = loss_fn(diff, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()

def train_focal_classifier(X_train_s, y_train, epochs=20, batch_size=256, lr=1e-3):
    """Trains a classifier with Focal Loss to heavily penalize false negatives."""
    model = RobustScoringMLP(in_features=X_train_s.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = FocalLoss(alpha=0.85, gamma=2.0)
    
    X_tensor = torch.tensor(X_train_s, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

def train_hard_negative_triplet(X_train_s, y_train, epochs=15, batch_size=256, lr=1e-3, margin=0.5):
    """Mines hard negatives and trains with Margin Triplet Loss."""
    X_pos = X_train_s[y_train == 1]
    X_neg = X_train_s[y_train == 0]
    
    # Compute centroid of positive likes
    pos_centroid = np.mean(X_pos, axis=0, keepdims=True)
    pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-8)
    
    # Cosine similarity to positive centroid
    X_neg_norm = X_neg / (np.linalg.norm(X_neg, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(X_neg_norm, pos_centroid.T).flatten()
    hard_neg_indices = np.argsort(sims)[::-1][:min(len(X_neg), len(X_pos) * 10)]
    X_hard_neg = X_neg[hard_neg_indices]
    
    num_triplets = min(len(X_pos) * 40, 60000)
    
    model = RobustScoringMLP(in_features=X_train_s.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    triplet_loss = nn.MarginRankingLoss(margin=margin)
    
    for epoch in range(epochs):
        model.train()
        p_sel = np.random.choice(len(X_pos), size=num_triplets, replace=True)
        n_sel = np.random.choice(len(X_hard_neg), size=num_triplets, replace=True)
        
        for i in range(0, num_triplets, batch_size):
            pb = torch.tensor(X_pos[p_sel[i:i+batch_size]], dtype=torch.float32).to(device)
            nb = torch.tensor(X_hard_neg[n_sel[i:i+batch_size]], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            sp = model(pb)
            sn = model(nb)
            target = torch.ones_like(sp)
            loss = triplet_loss(sp, sn, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model

def train_hist_gbt(X_train_s, y_train):
    """Trains a Gradient Boosted Decision Tree with balanced class weights."""
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_s, y_train)
    return clf

def predict_torch(model, X_np):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy().flatten()
    return preds

# ------------------------------------------------------------------------------
# 3. Evaluation Helper
# ------------------------------------------------------------------------------

def evaluate_model(preds, y_true):
    n_total = len(y_true)
    n_likes = int(np.sum(y_true))
    n_dislikes = int(n_total - n_likes)
    
    # 1. ROC-AUC and PR-AUC
    roc = roc_auc_score(y_true, preds)
    p_curve, r_curve, _ = precision_recall_curve(y_true, preds)
    pr_auc = auc(r_curve, p_curve)
    
    sorted_idx = np.argsort(preds)[::-1]
    sorted_y = y_true[sorted_idx]
    
    # 2. Performance at strict 20% Bot Like Cap
    k_20 = int(np.round(0.20 * n_total))
    likes_20 = int(np.sum(sorted_y[:k_20]))
    rec_20 = likes_20 / n_likes * 100
    dislikes_tossed_20 = (n_dislikes - (k_20 - likes_20)) / n_dislikes * 100
    hit_rate_20 = likes_20 / k_20 * 100
    
    # 3. Performance at strict 25% Bot Like Cap
    k_25 = int(np.round(0.25 * n_total))
    likes_25 = int(np.sum(sorted_y[:k_25]))
    rec_25 = likes_25 / n_likes * 100
    dislikes_tossed_25 = (n_dislikes - (k_25 - likes_25)) / n_dislikes * 100
    hit_rate_25 = likes_25 / k_25 * 100
    
    # 4. Bot Swipe Rate at 95% Like Retention
    cum_likes = np.cumsum(sorted_y)
    cutoff_95 = np.where(cum_likes >= int(np.ceil(0.95 * n_likes)))[0][0]
    bot_likes_95 = cutoff_95 + 1
    swipe_rate_95 = bot_likes_95 / n_total * 100
    dislikes_tossed_95 = (n_dislikes - (bot_likes_95 - int(cum_likes[cutoff_95]))) / n_dislikes * 100
    
    # 5. Bot Swipe Rate at 100% Like Retention
    cutoff_100 = np.where(cum_likes >= n_likes)[0][0]
    bot_likes_100 = cutoff_100 + 1
    swipe_rate_100 = bot_likes_100 / n_total * 100
    dislikes_tossed_100 = (n_dislikes - (bot_likes_100 - n_likes)) / n_dislikes * 100
    
    return {
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc, 4),
        "Likes@20%Cap": f"{likes_20}/{n_likes} ({rec_20:.1f}%)",
        "HitRate@20%": f"{hit_rate_20:.1f}%",
        "DislikesTossed@20%": f"{dislikes_tossed_20:.1f}%",
        "Likes@25%Cap": f"{likes_25}/{n_likes} ({rec_25:.1f}%)",
        "HitRate@25%": f"{hit_rate_25:.1f}%",
        "DislikesTossed@25%": f"{dislikes_tossed_25:.1f}%",
        "SwipeRate@95%Likes": f"{swipe_rate_95:.1f}%",
        "DislikesTossed@95%": f"{dislikes_tossed_95:.1f}%",
        "SwipeRate@100%Likes": f"{swipe_rate_100:.1f}%",
        "DislikesTossed@100%": f"{dislikes_tossed_100:.1f}%",
    }

# ------------------------------------------------------------------------------
# 4. Main Execution Loop
# ------------------------------------------------------------------------------

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    models_dir = root / "models" / "tournament"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TRAINING TOURNAMENT: 10K vs 20K MULTI-METHOD BENCHMARK ON 20K-22K HOLDOUT")
    print("=" * 80)
    
    print("Loading 20k-22k holdout dataset...")
    val_face = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    val_clip = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_to_22k.npz"))
    y_val = (val_face.ratings >= 50.0).astype(int)
    X_val_1024 = np.hstack([val_face.embeddings, val_clip.embeddings])
    print(f"Holdout: {len(y_val)} samples (Likes: {int(np.sum(y_val))}, Dislikes: {int(len(y_val)-np.sum(y_val))})")
    
    datasets = {
        "10k": {
            "face": str(root / "embeddings" / "reference_store_bumble_10k_pure.npz"),
            "clip": str(root / "embeddings" / "clip_store_bumble_10k_pure.npz")
        },
        "20k": {
            "face": str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"),
            "clip": str(root / "embeddings" / "clip_store_bumble_20k_pure.npz")
        }
    }
    
    tournament_records = []
    
    for split_name, paths in datasets.items():
        print("\n" + "#" * 60)
        print(f"TRAINING ON {split_name.upper()} DATASET...")
        print("#" * 60)
        
        train_face = load_store(paths["face"])
        train_clip = load_clip_store(paths["clip"])
        y_train = (train_face.ratings >= 50.0).astype(int)
        X_train_face = train_face.embeddings
        X_train_clip = train_clip.embeddings
        X_train_1024 = np.hstack([X_train_face, X_train_clip])
        print(f"Train samples: {len(y_train)} (Likes: {int(np.sum(y_train))}, Dislikes: {int(len(y_train)-np.sum(y_train))})")
        
        # Fit StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_1024)
        X_val_s = scaler.transform(X_val_1024)
        joblib.dump(scaler, str(models_dir / f"scaler_{split_name}.joblib"))
        
        # 1. Baseline Ridge Regressor (Face + Multimodal Blend)
        print(f"\n[{split_name}] 1. Evaluating Baseline Blend...")
        base_face = load_regressor(str(root / "models" / f"rating_regressor_bumble_{split_name}_pure.joblib"))
        base_multi = load_regressor(str(root / "models" / f"rating_regressor_multimodal_bumble_{split_name}_pure.joblib"))
        
        p_base_f = base_face.estimator.predict(val_face.embeddings)
        f_mode = base_multi.metadata.get("feature_mode", "face_only")
        if f_mode == "face_clip":
            p_base_m = base_multi.estimator.predict(X_val_1024)
        else:
            p_base_m = base_multi.estimator.predict(val_face.embeddings)
            
        p_baseline_blend = 0.50 * p_base_f + 0.50 * p_base_m
        res = evaluate_model(p_baseline_blend, y_val)
        res["Model"] = f"Baseline Blend ({split_name})"
        res["Architecture"] = "Ridge Face + MLP Blend"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")
        
        # 2. Pairwise RankNet (PyTorch/CUDA with LayerNorm)
        print(f"\n[{split_name}] 2. Training Pairwise RankNet (PyTorch/CUDA)...")
        ranknet_model = train_ranknet(X_train_s, y_train, epochs=15, batch_size=256, lr=1e-3)
        torch.save(ranknet_model.state_dict(), str(models_dir / f"ranknet_{split_name}.pt"))
        p_ranknet = predict_torch(ranknet_model, X_val_s)
        res = evaluate_model(p_ranknet, y_val)
        res["Model"] = f"RankNet ({split_name})"
        res["Architecture"] = "Pairwise Logistic Ranking MLP"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")
        
        # 3. Asymmetric Focal Loss Classifier (PyTorch/CUDA)
        print(f"\n[{split_name}] 3. Training Focal Loss Classifier (PyTorch/CUDA)...")
        focal_model = train_focal_classifier(X_train_s, y_train, epochs=20, batch_size=256, lr=1e-3)
        torch.save(focal_model.state_dict(), str(models_dir / f"focal_classifier_{split_name}.pt"))
        p_focal = predict_torch(focal_model, X_val_s)
        res = evaluate_model(p_focal, y_val)
        res["Model"] = f"Focal Loss Classifier ({split_name})"
        res["Architecture"] = "Cost-Sensitive Focal MLP (alpha=0.85, gamma=2)"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")
        
        # 4. Hard Negative Triplet Adapter (PyTorch/CUDA)
        print(f"\n[{split_name}] 4. Training Hard Negative Triplet Adapter (PyTorch/CUDA)...")
        triplet_model = train_hard_negative_triplet(X_train_s, y_train, epochs=15, batch_size=256, lr=1e-3, margin=0.5)
        torch.save(triplet_model.state_dict(), str(models_dir / f"hard_negative_triplet_{split_name}.pt"))
        p_triplet = predict_torch(triplet_model, X_val_s)
        res = evaluate_model(p_triplet, y_val)
        res["Model"] = f"Hard Negative Triplet ({split_name})"
        res["Architecture"] = "Mined Hard Negative Triplet MLP"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")
        
        # 5. Gradient Boosted Decision Trees (HistGradientBoostingClassifier)
        print(f"\n[{split_name}] 5. Training Gradient Boosted Trees (HistGBT)...")
        gbt_model = train_hist_gbt(X_train_s, y_train)
        joblib.dump(gbt_model, str(models_dir / f"hist_gbt_{split_name}.joblib"))
        p_gbt = gbt_model.predict_proba(X_val_s)[:, 1]
        res = evaluate_model(p_gbt, y_val)
        res["Model"] = f"Gradient Boosted Trees ({split_name})"
        res["Architecture"] = "HistGradientBoosting (Class-Balanced)"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")
        
        # 6. Hybrid Super-Ensemble (Baseline Blend + RankNet + Focal)
        print(f"\n[{split_name}] 6. Evaluating Hybrid Super-Ensemble...")
        p_norm_base = (p_baseline_blend - p_baseline_blend.min()) / (p_baseline_blend.max() - p_baseline_blend.min() + 1e-8)
        p_norm_rank = (p_ranknet - p_ranknet.min()) / (p_ranknet.max() - p_ranknet.min() + 1e-8)
        p_norm_focal = (p_focal - p_focal.min()) / (p_focal.max() - p_focal.min() + 1e-8)
        p_ensemble = 0.40 * p_norm_base + 0.30 * p_norm_rank + 0.30 * p_norm_focal
        res = evaluate_model(p_ensemble, y_val)
        res["Model"] = f"Hybrid Super-Ensemble ({split_name})"
        res["Architecture"] = "Weighted Blend (Baseline + RankNet + Focal)"
        res["Dataset"] = split_name
        tournament_records.append(res)
        print(f"  -> PR-AUC: {res['PR-AUC']} | Likes@25%Cap: {res['Likes@25%Cap']} | SwipeRate@95%Likes: {res['SwipeRate@95%Likes']}")

    # Save tournament results to CSV
    df = pd.DataFrame(tournament_records)
    cols = ["Model", "Dataset", "Architecture", "PR-AUC", "ROC-AUC", "Likes@20%Cap", "HitRate@20%", "Likes@25%Cap", "HitRate@25%", "SwipeRate@95%Likes", "SwipeRate@100%Likes"]
    df = df[cols].sort_values(by="PR-AUC", ascending=False)
    csv_path = results_dir / "tournament_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print(f"TOURNAMENT COMPLETE! Saved results to {csv_path}")
    print("=" * 80)
    print(df.to_string())

if __name__ == "__main__":
    main()
