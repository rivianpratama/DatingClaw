import torch
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_similarity.store import load_store
from scratch.train_50k_grand_ensemble import (
    ConformalGMMGeometricVeto,
    MultiXSafeMetaBlender,
    Pillar1SingleCropNet as P1_v4,
    Pillar2DualCropNet as P2_v4,
    Pillar3SubcenterNet as P3_v4,
    Pillar4OrthogonalRankNet as P4_v4
)
import __main__
__main__.MultiXSafeMetaBlender = MultiXSafeMetaBlender
__main__.ConformalGMMGeometricVeto = ConformalGMMGeometricVeto

from face_similarity.v5_architecture import evaluate_holdout_v4_vs_v5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(__file__).resolve().parent.parent
MODELS_DIR = root / "models"

# 1. Load val_1582
r = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
y = (r.ratings >= 50.0).astype(int)
N = len(y)
K = int(np.floor(0.208 * N))

t_arc = torch.tensor(r.embeddings, device=device)
geom = np.load(str(root / "embeddings" / "geom_store_val_1582.npz"))["geom_16d"]
t_geom = torch.tensor(geom, device=device)
sp = np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))
fv = torch.tensor(sp["face_valid"], device=device)
t_sf = torch.tensor(np.load(str(root / "embeddings" / "siglip_store_val_1582.npz"))["embeddings"], device=device)
t_sp = torch.tensor(sp["embeddings"], device=device)
t_dino = torch.tensor(np.load(str(root / "embeddings" / "dinov2_portrait_store_val_1582.npz"))["embeddings"], device=device)

# Load Models
p1 = P1_v4().to(device); p1.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt")); p1.eval()
p2 = P2_v4().to(device); p2.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p2_50k.pt")); p2.eval()
p3 = P3_v4(K=16, M=48).to(device); p3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p3_50k.pt")); p3.eval()
p4 = P4_v4().to(device); p4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p4_50k.pt")); p4.eval()

with torch.no_grad():
    z1 = p1(t_arc, t_geom, fv).cpu().numpy()
    z2 = p2(t_arc, t_geom, t_sf, t_sp, t_dino, fv).cpu().numpy()
    z3 = p3.compute_logits(p3.embed(t_arc, t_geom, fv)).cpu().numpy()
    z4 = p4(t_arc, t_geom, t_sf, t_sp, t_dino, fv).cpu().numpy()
    Z_val = np.stack([z1, z2, z3, z4], axis=1)

df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False).drop_duplicates(subset=["screenshot"], keep="first")
smap = {
    row["screenshot"]: (
        float(row["score"]) if row.get("method") == "multimodalx_original" and pd.notna(row.get("score"))
        else (0.48 * float(row["ridge"]) + 0.48 * float(row["multimodal"]) + 4.0 if pd.notna(row.get("ridge")) and pd.notna(row.get("multimodal")) else 50.0)
    )
    for _, row in df_scores.iterrows() if pd.notna(row.get("screenshot"))
}
mx = np.array([smap.get(Path(p).name, 50.0) for p in r.paths], dtype=np.float32)

meta = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")
veto = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")

# v4 baseline score
s4 = meta.predict_score(Z_val, mx)
s4 = np.where(veto.veto(geom), -1e9, s4)

# Pure Visual Match Log-Odds L_match
L = meta._calibrated_log_odds(Z_val)
w_p = meta.meta_params[:4]
w_lse = meta.meta_params[4]
bias = meta.meta_params[7]
tau_lse = meta.tau_lse
m_max = np.max(L / tau_lse, axis=1, keepdims=True)
l_lse = tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L / tau_lse - m_max), axis=1)))
s_match = (L @ w_p) + w_lse * l_lse + bias
s_match = np.where(veto.veto(geom) | (mx < 42.5), -1e9, s_match)

ord_m = np.argsort(s_match)[::-1]
print("=== True Likes Count at Match Rank Cutoffs ===")
for n_top in [140, 160, 180, 200, 220, 250, 280, 300, 318]:
    hits = int(y[ord_m[:n_top]].sum())
    print(f"Top {n_top:3d} in s_match contains {hits:3d} / 140 True Likes ({hits/140*100:.2f}%)")

# Now let's test a Lexicographic 1-to-5 Graded Scoring
# Principle:
# 1. Any profile with s_match >= tau_like (e.g. Rank <= 260) gets Grade 5 range [80.0, 100.0]
# 2. Profiles below that are strictly non-matches, scored based on multimodalx_original:
#    - Cute Dislikes (mx >= 60): Grade 4 range [60.0, 79.9]
#    - Neutral Passes (50 <= mx < 60): Grade 3 range [45.0, 59.9]
#    - Low Passes (40 <= mx < 50): Grade 2 range [30.0, 44.9]
#    - Junk (mx < 40 or veto): Grade 1 range [-1e9]
print("\n=== Testing Different Match Locking Cutoffs for v5 Graded Scoring ===")
for lock_n in [200, 220, 240, 260, 280, 300]:
    tau_like = float(s_match[ord_m[lock_n - 1]])
    # Score 5: Liked face
    # Scores 1-4: Disliked face reranked strictly by MultiX
    score_v5 = np.where(
        s_match >= tau_like,
        100.0 + s_match,
        np.clip(mx, 0.0, 79.9)
    )
    score_v5 = np.where(veto.veto(geom) | (mx < 42.5), -1e9, score_v5)

    res = evaluate_holdout_v4_vs_v5(s4, score_v5, y, mx, setpoint=0.208, n_boot=1000)
    v5s = res["v5_summary"]
    v4s = res["v4_summary"]
    print(f"Lock Top {lock_n:3d} | Hits: {v5s['hits_at_k']} / 140 (Recall {v5s['recall_at_k']*100:.1f}%) | "
          f"Cute FP (G4): {v5s['cute_fp_share_g4']*100:.1f}% | FP MultiX Mean: {v5s['fp_multix_mean']:.2f} | "
          f"Catastrophic FP: {v5s['catastrophic_fp_g1_g2']}")
