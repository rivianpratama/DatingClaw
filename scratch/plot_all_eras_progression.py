import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scratch"))

from scratch.train_dual_crop_subcenter_pauc import predict_dual_crop, predict_visual_logits

# 1. Load Common Untrained 40k-41k Slice (969 profiles, 82 likes)
ref_test = np.load(root / 'embeddings' / 'reference_store_bumble_40k_to_41k.npz')
clip_test = np.load(root / 'embeddings' / 'clip_store_bumble_40k_to_41k.npz')
sig_test = np.load(root / 'embeddings' / 'siglip_store_bumble_40k_to_41k.npz')
port_test = np.load(root / 'embeddings' / 'siglip_portrait_store_bumble_40k_to_41k.npz')

y = (ref_test['ratings'] >= 50.0).astype(int)
X_fa = ref_test['embeddings']
X_cl = clip_test['embeddings']
X_mlp = np.hstack([X_fa, X_cl])
X_fu = sig_test['embeddings']
X_po = port_test['embeddings']
FV = port_test['face_valid']
X_dc = np.hstack([X_fu, X_po, X_fa])

n = len(y)
k25 = int(np.round(0.25 * n)) # 242 swipes
k20 = int(np.round(0.20 * n)) # 194 swipes

def get_stats(p):
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    order = np.argsort(p)[::-1]
    rec25 = float(y[order[:k25]].sum()) / float(y.sum()) * 100.0
    rec20 = float(y[order[:k20]].sum()) / float(y.sum()) * 100.0
    hit25 = float(y[order[:k25]].sum()) / float(k25) * 100.0
    likes25 = int(y[order[:k25]].sum())
    likes20 = int(y[order[:k20]].sum())
    return {
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "Recall@20%": rec20,
        "Recall@25%": rec25,
        "HitRate@25%": hit25,
        "Likes@25%": likes25,
        "Likes@20%": likes20
    }

eras = []

# Era 1: Original (7k Ridge)
m_orig = joblib.load(root / 'models' / 'rating_regressor.joblib')
m_orig = m_orig['estimator'] if isinstance(m_orig, dict) else m_orig
p_orig = m_orig.predict(X_fa)
s_orig = get_stats(p_orig)
eras.append({
    "Era": "1. Original (7k)",
    "ShortName": "Original (7k)\nRidge Face",
    "DatasetSize": "7k",
    "ModelName": "Ridge Regression (ArcFace)",
    **s_orig
})

# Era 2: Multimodal (7k Initial)
m_mm7k = joblib.load(root / 'models' / 'rating_regressor_multimodal.joblib')
m_mm7k = m_mm7k['estimator'] if isinstance(m_mm7k, dict) else m_mm7k
p_mm7k = m_mm7k.predict(X_mlp)
s_mm7k = get_stats(p_mm7k)
eras.append({
    "Era": "2. Multimodal",
    "ShortName": "Multimodal (7k)\nLinear Combined",
    "DatasetSize": "7k",
    "ModelName": "Linear Multimodal (Face + CLIP)",
    **s_mm7k
})

# Era 3: ExperimentX (Combined Round 2)
m_r2 = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_combined_round2.joblib')
m_r2 = m_r2['estimator'] if isinstance(m_r2, dict) else m_r2
p_r2 = m_r2.predict(X_mlp)
s_r2 = get_stats(p_r2)
eras.append({
    "Era": "3. ExperimentX",
    "ShortName": "ExperimentX\nWeighted Regressor",
    "DatasetSize": "8.5k",
    "ModelName": "Multi-Component Weighted Regressor",
    **s_r2
})

# Era 4: 10k Pure
m_10k = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_10k_pure.joblib')
m_10k = m_10k['estimator'] if isinstance(m_10k, dict) else m_10k
p_10k = m_10k.predict(X_fa if m_10k.n_features_in_ == 512 else X_mlp)
s_10k = get_stats(p_10k)
eras.append({
    "Era": "4. 10k Era",
    "ShortName": "10k Pure\nMultimodal MLP",
    "DatasetSize": "10k",
    "ModelName": "Pure Bumble Multimodal MLP",
    **s_10k
})

# Era 5: 20k Pure
m_20k = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_20k_pure.joblib')
m_20k = m_20k['estimator'] if isinstance(m_20k, dict) else m_20k
p_20k = m_20k.predict(X_mlp)
s_20k = get_stats(p_20k)
eras.append({
    "Era": "5. 20k Era",
    "ShortName": "20k Pure\nMultimodal MLP",
    "DatasetSize": "20k",
    "ModelName": "Scaled Multimodal MLP",
    **s_20k
})

# Era 6: 30k Era (30k Grand Super-Ensemble)
m_30k = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_30k_pure.joblib')
m_30k = m_30k['estimator'] if isinstance(m_30k, dict) else m_30k
p_30k = m_30k.predict(X_mlp)
p_sc30 = predict_visual_logits(str(root / 'models' / 'subcenter_phase2' / 'best_subcenter_pauc_26k.pt'), X_fu, X_fa)
p_dc30 = predict_dual_crop(str(root / 'models' / 'subcenter_phase2' / 'best_dual_crop_pauc_30k_full.pt'), X_fu, X_po, X_fa, FV)
p_grand30 = 0.375 * (rankdata(p_30k)/n) + 0.25 * (rankdata(p_sc30)/n) + 0.375 * (rankdata(p_dc30)/n)
s_30k = get_stats(p_grand30)
eras.append({
    "Era": "6. 30k Era",
    "ShortName": "30k Era\nGrand Ensemble",
    "DatasetSize": "30k",
    "ModelName": "Grand Super-Ensemble (MLP + SigLIP)",
    **s_30k
})

# Era 7: 40k Era (40k Meta-Blended Grand Super-Ensemble)
m_m40 = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_40k.joblib')['estimator']
m_dc40 = joblib.load(root / 'models' / 'rating_regressor_dual_crop_mlp_40k.joblib')['estimator']
p_sc40 = predict_visual_logits(str(root / 'models' / 'subcenter_phase2' / 'best_subcenter_pauc_40k.pt'), X_fu, X_fa)
p_dc40 = predict_dual_crop(str(root / 'models' / 'subcenter_phase2' / 'best_dual_crop_pauc_40k.pt'), X_fu, X_po, X_fa, FV)

def to_rn(arr):
    r = rankdata(arr) / (len(arr) + 1.0)
    return norm.ppf(np.clip(r, 1e-4, 1.0 - 1e-4))

rn1 = to_rn(m_m40.predict(X_mlp))
rn2 = to_rn(m_dc40.predict(X_dc))
rn3 = to_rn(p_sc40)
rn4 = to_rn(p_dc40)
p_meta40 = 0.1404 * rn1 + 0.8868 * rn2 + 0.3079 * rn3 + 0.7871 * rn4
s_40k = get_stats(p_meta40)
eras.append({
    "Era": "7. 40k Era",
    "ShortName": "40k Era\nMeta-Blender",
    "DatasetSize": "40k",
    "ModelName": "4-Pillar Dual-Crop Meta-Blender",
    **s_40k
})

df = pd.DataFrame(eras)
print(df[["Era", "DatasetSize", "Recall@25%", "Recall@20%", "HitRate@25%", "ROC-AUC", "PR-AUC"]])
csv_out = root / "results" / "historical_progression_by_era_untrained_slice.csv"
df.to_csv(csv_out, index=False)
print(f"Saved era comparison to {csv_out}")

# ==============================================================================
# PLOTTING
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})

x_indices = np.arange(len(df))
short_names = df["ShortName"].tolist()

# Modern tech palette
c_rec25 = "#2563eb" # Royal Blue
c_rec20 = "#059669" # Emerald Green
c_roc = "#7c3aed"   # Deep Purple
c_pr = "#ea580c"    # Vivid Orange

# --- Subplot 1: Recall & Accuracy Curves ---
ax1.plot(x_indices, df["Recall@25%"], marker="o", markersize=11, linewidth=3.5, color=c_rec25, label="Recall @ 25% Cap (Strict ELO Ceiling: 242 Swipes)")
ax1.plot(x_indices, df["Recall@20%"], marker="s", markersize=10, linewidth=3.0, color=c_rec20, linestyle="--", label="Recall @ 20% Cap (Ultra-Safe Target: 194 Swipes)")

# Add shaded improvement band
ax1.fill_between(x_indices, df["Recall@20%"], df["Recall@25%"], color="#93c5fd", alpha=0.25, label="20%–25% Operating Zone")

for i, (r25, r20, l25, l20) in enumerate(zip(df["Recall@25%"], df["Recall@20%"], df["Likes@25%"], df["Likes@20%"])):
    ax1.annotate(f"{r25:.1f}%\n({l25}/82)", (i, r25), textcoords="offset points", xytext=(0, 14),
                 ha='center', fontsize=10.5, fontweight='bold', color=c_rec25)
    ax1.annotate(f"{r20:.1f}%\n({l20}/82)", (i, r20), textcoords="offset points", xytext=(0, -26),
                 ha='center', fontsize=9.5, fontweight='bold', color=c_rec20)

ax1.set_title("BumbleClaw Model Evolution Across Eras: Accuracy on Untrained 40k–41k Test Cards", fontsize=15, fontweight='bold', pad=15)
ax1.set_ylabel("User Likes Captured (%)", fontsize=12, fontweight='bold')
ax1.set_ylim(52, 98)
ax1.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)
ax1.grid(True, linestyle="--", alpha=0.6)

# --- Subplot 2: Discriminative Ranking Quality (ROC-AUC & PR-AUC) ---
ax2.plot(x_indices, df["ROC-AUC"], marker="^", markersize=10, linewidth=3.0, color=c_roc, label="ROC-AUC (Overall Queue Sorting Quality)")
ax2.plot(x_indices, df["PR-AUC"], marker="D", markersize=9, linewidth=2.8, color=c_pr, label="PR-AUC (Precision Focus on Needle-in-Haystack Likes)")

for i, (roc, pr) in enumerate(zip(df["ROC-AUC"], df["PR-AUC"])):
    ax2.annotate(f"{roc:.4f}", (i, roc), textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=10, fontweight='bold', color=c_roc)
    ax2.annotate(f"{pr:.4f}", (i, pr), textcoords="offset points", xytext=(0, -18),
                 ha='center', fontsize=10, fontweight='bold', color=c_pr)

ax2.set_ylabel("Metric Value (0.0 to 1.0)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Model Historical Generation / Era", fontsize=13, fontweight='bold', labelpad=10)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(short_names, fontsize=10.5, fontweight='bold')
ax2.set_ylim(0.20, 0.98)
ax2.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
png_out = root / "results" / "all_eras_accuracy_progression.png"
plt.savefig(png_out, dpi=200)
print(f"Saved clean chart to {png_out}")

# Copy to brain artifact directory
artifact_dir = Path(r"C:\Users\Rivian\.gemini\antigravity\brain\73824904-181e-4277-ac0d-85aa19274479")
if artifact_dir.exists():
    import shutil
    shutil.copy2(png_out, artifact_dir / "all_eras_accuracy_progression.png")
    print(f"Copied chart to {artifact_dir / 'all_eras_accuracy_progression.png'}")
