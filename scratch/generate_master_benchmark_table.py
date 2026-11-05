import sys
from pathlib import Path

root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import torch
import joblib

from face_similarity.store import load_store
from face_similarity.clip_store import load_clip_store
from face_similarity.regressor import load_regressor
from scratch.run_tournament_training_and_benchmark import RobustScoringMLP, predict_torch, evaluate_model

models_dir = root / "models" / "tournament"

# Load holdout
val_face = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
val_clip = load_clip_store(str(root / "embeddings" / "clip_store_bumble_20k_to_22k.npz"))
y_val = (val_face.ratings >= 50.0).astype(int)
X_val_1024 = np.hstack([val_face.embeddings, val_clip.embeddings])

# Load scalers
scaler_10k = joblib.load(models_dir / "scaler_10k.joblib")
scaler_20k = joblib.load(models_dir / "scaler_20k.joblib")
X_val_s10k = scaler_10k.transform(X_val_1024)
X_val_s20k = scaler_20k.transform(X_val_1024)

# 1. Previous Legacy models
leg_f = load_regressor(str(root / "models" / "rating_regressor.joblib"))
leg_m = load_regressor(str(root / "models" / "rating_regressor_multimodal.joblib"))
p_leg_f = leg_f.estimator.predict(val_face.embeddings)
p_leg_m = leg_m.estimator.predict(X_val_1024)

# 2. Previous 10k Pure
p10k_f = load_regressor(str(root / "models" / "rating_regressor_bumble_10k_pure.joblib"))
p10k_m = load_regressor(str(root / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib"))
pred_10k_f = p10k_f.estimator.predict(val_face.embeddings)
pred_10k_m = p10k_m.estimator.predict(val_face.embeddings)

# 3. Previous 20k Pure
p20k_f = load_regressor(str(root / "models" / "rating_regressor_bumble_20k_pure.joblib"))
p20k_m = load_regressor(str(root / "models" / "rating_regressor_multimodal_bumble_20k_pure.joblib"))
pred_20k_f = p20k_f.estimator.predict(val_face.embeddings)
pred_20k_m = p20k_m.estimator.predict(X_val_1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pt_model(name):
    m = RobustScoringMLP().to(device)
    m.load_state_dict(torch.load(models_dir / f"{name}.pt"))
    return m

p_ranknet_20k = predict_torch(load_pt_model("ranknet_20k"), X_val_s20k)
p_triplet_20k = predict_torch(load_pt_model("hard_negative_triplet_20k"), X_val_s20k)
p_focal_20k = predict_torch(load_pt_model("focal_classifier_20k"), X_val_s20k)

p_ranknet_10k = predict_torch(load_pt_model("ranknet_10k"), X_val_s10k)
p_triplet_10k = predict_torch(load_pt_model("hard_negative_triplet_10k"), X_val_s10k)
p_focal_10k = predict_torch(load_pt_model("focal_classifier_10k"), X_val_s10k)

gbt_10k = joblib.load(models_dir / "hist_gbt_10k.joblib")
p_gbt_10k = gbt_10k.predict_proba(X_val_s10k)[:, 1]

gbt_20k = joblib.load(models_dir / "hist_gbt_20k.joblib")
p_gbt_20k = gbt_20k.predict_proba(X_val_s20k)[:, 1]

# Super ensembles
p_base_20k = 0.50 * pred_20k_f + 0.50 * pred_20k_m
p_norm_b20 = (p_base_20k - p_base_20k.min()) / (p_base_20k.max() - p_base_20k.min())
p_norm_rn20 = (p_ranknet_20k - p_ranknet_20k.min()) / (p_ranknet_20k.max() - p_ranknet_20k.min())
p_norm_fc20 = (p_focal_20k - p_focal_20k.min()) / (p_focal_20k.max() - p_focal_20k.min())
p_ensemble_20k = 0.40 * p_norm_b20 + 0.30 * p_norm_rn20 + 0.30 * p_norm_fc20

p_base_10k = 0.50 * pred_10k_f + 0.50 * pred_10k_m
p_norm_b10 = (p_base_10k - p_base_10k.min()) / (p_base_10k.max() - p_base_10k.min())
p_norm_rn10 = (p_ranknet_10k - p_ranknet_10k.min()) / (p_ranknet_10k.max() - p_ranknet_10k.min())
p_norm_fc10 = (p_focal_10k - p_focal_10k.min()) / (p_focal_10k.max() - p_focal_10k.min())
p_ensemble_10k = 0.40 * p_norm_b10 + 0.30 * p_norm_rn10 + 0.30 * p_norm_fc10

all_models = {
    "Hybrid Super-Ensemble (20k)": (p_ensemble_20k, "20k", "Weighted Blend (Baseline + RankNet + Focal)"),
    "Bumble 20k Pure (Multi Only)": (pred_20k_m, "20k", "CLIP ViT-B/32 Multimodal MLP"),
    "Bumble 20k Pure (50/50 Blend)": (0.50 * pred_20k_f + 0.50 * pred_20k_m, "20k", "Ridge Face + Multimodal MLP Blend"),
    "Bumble 20k Pure (60/40 Aligned)": (0.60 * pred_20k_f + 0.40 * pred_20k_m, "20k", "Ridge Face (60%) + Multimodal (40%)"),
    "Bumble 20k Pure (Face Only)": (pred_20k_f, "20k", "InsightFace 512D Ridge Regressor"),
    "Hybrid Super-Ensemble (10k)": (p_ensemble_10k, "10k", "Weighted Blend (Baseline + RankNet + Focal)"),
    "RankNet (20k)": (p_ranknet_20k, "20k", "Pairwise Logistic Ranking MLP"),
    "Hard Negative Triplet (20k)": (p_triplet_20k, "20k", "Mined Hard Negative Triplet MLP"),
    "Bumble 10k Pure": (0.60 * pred_10k_f + 0.40 * pred_10k_m, "10k", "Ridge Face + MLP Regressor"),
    "Focal Loss Classifier (10k)": (p_focal_10k, "10k", "Cost-Sensitive Focal MLP"),
    "Focal Loss Classifier (20k)": (p_focal_20k, "20k", "Cost-Sensitive Focal MLP"),
    "Hist Gradient Boosted Trees (10k)": (p_gbt_10k, "10k", "HistGradientBoosting (Class-Balanced)"),
    "Experimental3 (Production)": (0.50 * p_leg_f + 0.50 * p_leg_m, "~8k", "Ensemble Face + Clip Regressor"),
    "Legacy face_biased": (0.52 * p_leg_f + 0.48 * p_leg_m, "7k", "Legacy Stock Regressor"),
    "Hist Gradient Boosted Trees (20k)": (p_gbt_20k, "20k", "HistGradientBoosting (Class-Balanced)"),
    "RankNet (10k)": (p_ranknet_10k, "10k", "Pairwise Logistic Ranking MLP"),
    "Hard Negative Triplet (10k)": (p_triplet_10k, "10k", "Mined Hard Negative Triplet MLP"),
}

records = []
for name, (preds, dset, arch) in all_models.items():
    stats = evaluate_model(preds, y_val)
    stats["Model"] = name
    stats["Dataset"] = dset
    stats["Architecture"] = arch
    records.append(stats)

df = pd.DataFrame(records)
cols = ["Model", "Dataset", "Architecture", "PR-AUC", "ROC-AUC", "Likes@20%Cap", "HitRate@20%", "Likes@25%Cap", "HitRate@25%", "SwipeRate@95%Likes", "SwipeRate@100%Likes"]
df = df[cols].sort_values(by="PR-AUC", ascending=False)
out_csv = root / "results" / "master_comprehensive_benchmark_all_models.csv"
df.to_csv(out_csv, index=False)
print("=" * 80)
print(f"MASTER TABLE GENERATED ({len(df)} models total):")
print("=" * 80)
print(df.to_string())
