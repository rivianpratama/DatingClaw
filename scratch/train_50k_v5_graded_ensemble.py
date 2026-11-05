import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_similarity.v4_architecture import ExpressionFiLMArcFace
from face_similarity.v5_architecture import (
    MultiHeadOrthogonalBilinearFusion,
    AsymmetricGradedLambdaCORALLossV51,
    DualHeadOutput,
    PyramidalConcentricSubcenterHead,
    MultiXSafeMetaBlenderV5,
    Pillar1V5SingleCropNet,
    evaluate_holdout_v4_vs_v5
)
from face_similarity.store import load_store

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASTER_STORE = Path(r"D:\master_50k_store.npz")
MODELS_DIR = root / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# V5 PILLARS WITH DUAL-HEAD OUTPUTS
# ==============================================================================
# Pillar 1 is imported directly from face_similarity.v5_architecture:
#   Decoupled Pure Facial Bone-Structure & Expression Specialist (ArcFace + Landmarks + Subcenters)
#   Zero CORAL ordinal projection -> completely immune to photographic manifold contortion.

class Pillar2V5DualCropNet(nn.Module):
    def __init__(self, mod_dim: int = 192, drop: float = 0.25):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=256, expr_dim=64)
        self.branch_arc  = nn.Sequential(nn.Linear(320, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_sigF = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_sigP = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_dino = nn.Sequential(nn.Linear(1536, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.mod_gates   = nn.Parameter(torch.ones(4, 1))

        self.backbone = nn.Sequential(
            nn.Linear(mod_dim * 4, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.dual_head = DualHeadOutput(dim=256, lambda_tie=0.15)

    def forward(self, x_arc, x_geom, x_sig_full, x_sig_port, x_dino, face_valid=None):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        if face_valid is not None:
            h_arc = h_arc * face_valid
            e_expr = e_expr * face_valid
        h_face = torch.cat([h_arc, e_expr], dim=-1)

        w = F.softmax(self.mod_gates, dim=0) * 4.0
        h0 = w[0] * self.branch_arc(h_face)
        h1 = w[1] * self.branch_sigF(F.normalize(x_sig_full, p=2, dim=-1))
        h2 = w[2] * self.branch_sigP(F.normalize(x_sig_port, p=2, dim=-1))
        h3 = w[3] * self.branch_dino(F.normalize(x_dino, p=2, dim=-1))

        fused = torch.cat([h0, h1, h2, h3], dim=-1)
        z = F.normalize(self.backbone(fused), p=2, dim=-1)
        return self.dual_head(z)


class Pillar3V5SubcenterNet(nn.Module):
    def __init__(self, latent_dim: int = 256, drop: float = 0.20):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=latent_dim, expr_dim=64)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + 64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(latent_dim, latent_dim)
        )
        self.subcenter_head = PyramidalConcentricSubcenterHead(dim=latent_dim, tau=0.08, gamma=16.0, lambda_tie=0.15)

    def embed(self, x_arc, x_geom, face_valid=None):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        if face_valid is not None:
            h_arc = h_arc * face_valid
            e_expr = e_expr * face_valid
        z = self.proj(torch.cat([h_arc, e_expr], dim=-1))
        return F.normalize(z, p=2, dim=-1)

    def forward(self, x_arc, x_geom, face_valid=None):
        z = self.embed(x_arc, x_geom, face_valid)
        return self.subcenter_head(z)


class Pillar4V5OrthogonalRankNet(nn.Module):
    def __init__(self, dim: int = 256, drop: float = 0.20):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=dim, expr_dim=64)
        self.vis_enc = nn.Sequential(
            nn.Linear(1152 + 1152 + 1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )
        self.fusion = MultiHeadOrthogonalBilinearFusion(dim=dim, num_heads=8, rank=32, expr_dim=64)
        self.dual_head = DualHeadOutput(dim=dim, lambda_tie=0.15)

    def forward(self, x_arc, x_geom, x_sig_full, x_sig_port, x_dino, face_valid):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        raw_vis = torch.cat([
            F.normalize(x_sig_full, p=2, dim=-1),
            F.normalize(x_sig_port, p=2, dim=-1),
            F.normalize(x_dino, p=2, dim=-1)
        ], dim=-1)
        h_vis = self.vis_enc(raw_vis)
        fused = self.fusion(h_vis, h_arc, e_expr, face_valid)
        z = F.normalize(fused, p=2, dim=-1)
        return self.dual_head(z)


# ==============================================================================
# STRATIFIED 5-GRADE MINI-BATCH SAMPLER
# ==============================================================================
class Stratified5GradeBatchSampler:
    """
    Fixed per-batch distribution (B=512):
      N5 = 64  (12.5% Grade 5 True Likes)
      N4 = 160 (31.25% Grade 4 Cute Dislikes)
      N3 = 128 (25.0% Grade 3 Neutral Passes)
      N2 = 96  (18.75% Grade 2 Low Passes)
      N1 = 64  (12.5% Grade 1 Rejects)
    """
    def __init__(self, grades: np.ndarray, sample_weights: np.ndarray, batch_size: int = 512, steps_per_epoch: int = 80):
        self.batch_size = batch_size
        self.steps = steps_per_epoch

        self.idx_by_grade = {}
        self.probs_by_grade = {}
        for g in range(1, 6):
            idx = np.where(grades == g)[0]
            self.idx_by_grade[g] = idx
            w = sample_weights[idx]
            self.probs_by_grade[g] = w / w.sum()

        self.counts = {5: 64, 4: 160, 3: 128, 2: 96, 1: 64}

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            for g in range(1, 6):
                sampled = np.random.choice(
                    self.idx_by_grade[g],
                    size=self.counts[g],
                    replace=True,
                    p=self.probs_by_grade[g]
                )
                batch.append(sampled)
            yield np.concatenate(batch)

    def __len__(self):
        return self.steps


# ==============================================================================
# MAIN TRAINING & VALIDATION PIPELINE
# ==============================================================================
def main():
    print("=" * 80)
    print("BUMBLECLAW v5: 5-GRADE GRADED ORDINAL ENSEMBLE TRAINING & BENCHMARK")
    print("=" * 80)

    # 1. Load Master Store
    print(f"Loading {MASTER_STORE}...")
    st = np.load(MASTER_STORE)
    X_arc = st["X_arc"]
    X_geom = st["X_geom"]
    X_sig_full = st["X_sig_full"]
    X_sig_port = st["X_sig_port"]
    X_dino = st["X_dino"]
    face_valid = st["face_valid"]
    y_raw = st["y"]
    s_multi_x = st["s_multi_x"]
    w_raw = st["weights"]
    N = len(y_raw)

    # 2. Build 5-Grade Target Vector
    grades = np.ones(N, dtype=np.int64)
    grades[(y_raw == 0) & (s_multi_x >= 40.0) & (s_multi_x < 50.0)] = 2
    grades[(y_raw == 0) & (s_multi_x >= 50.0) & (s_multi_x < 60.0)] = 3
    grades[(y_raw == 0) & (s_multi_x >= 60.0)] = 4
    grades[y_raw == 1] = 5

    print("\n--- 5-Grade Dataset Partition ---")
    for g in range(1, 6):
        cnt = int((grades == g).sum())
        print(f"Grade {g}: {cnt:6d} profiles ({cnt/N*100:5.2f}%)")

    # High-ELO Anchor Mask for Probability Calibration
    # Era 2 (21,121 to 35,821) has stationary 10.36% baseline
    high_elo_mask = (np.arange(N) >= 21121) & (np.arange(N) <= 35821)

    # Ensure X_arc is normalized to unit norm S^511
    norm_arc = np.linalg.norm(X_arc, axis=-1, keepdims=True)
    norm_arc[norm_arc < 1e-4] = 1.0
    X_arc = (X_arc / norm_arc).astype(np.float32)

    # 3. Connected-Component Identity Grouping
    print("\nComputing ArcFace Identity Clusters on GPU...", flush=True)
    row_list, col_list = [], []
    batch_chunk = 5000
    for i in range(0, N, batch_chunk):
        end_i = min(i + batch_chunk, N)
        chunk_a = torch.tensor(X_arc[i:end_i], device=device)
        for j in range(i, N, batch_chunk):
            end_j = min(j + batch_chunk, N)
            chunk_b = torch.tensor(X_arc[j:end_j], device=device)
            sim_block = torch.matmul(chunk_a, chunk_b.T)
            mask = (sim_block >= 0.80)
            if i == j:
                mask.fill_diagonal_(False)
            nz = torch.nonzero(mask).cpu().numpy()
            if len(nz) > 0:
                row_list.extend(nz[:, 0] + i)
                col_list.extend(nz[:, 1] + j)

    if len(row_list) > 0:
        adj = csr_matrix((np.ones(len(row_list)), (row_list, col_list)), shape=(N, N))
        n_clusters, cluster_ids = connected_components(adj, directed=False)
    else:
        n_clusters = N
        cluster_ids = np.arange(N)
    print(f"Identified {n_clusters} unique identity clusters across {N} profiles (Zero cross-fold leakage guaranteed).", flush=True)

    # 4. Era-Stratified 5-Fold Cross Validation
    eras = np.where(np.arange(N) <= 21120, 1, np.where(np.arange(N) <= 35821, 2, 3))
    era_group_labels = eras * 1000000 + cluster_ids
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Pre-allocate OOF arrays
    Z_oof = np.zeros((N, 4), dtype=np.float32)
    g_aes_oof = np.zeros((N, 4), dtype=np.float32)
    p_g_le2_oof = np.zeros(N, dtype=np.float32)

    # Move tensors to GPU
    t_arc_all = torch.tensor(X_arc, device=device)
    t_geom_all = torch.tensor(X_geom, device=device)
    t_sf_all = torch.tensor(X_sig_full, device=device)
    t_sp_all = torch.tensor(X_sig_port, device=device)
    t_dino_all = torch.tensor(X_dino, device=device)
    t_fv_all = torch.tensor(face_valid, device=device)
    t_grades_all = torch.tensor(grades, device=device)
    t_w_all = torch.tensor(w_raw, device=device)

    loss_fn = AsymmetricGradedLambdaCORALLossV51(setpoint_frac=0.208, skirt_frac=0.040, sigma=3.0).to(device)

    print("\n--- Training 5-Fold Graded Pillars ---")
    t0_train = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.arange(N), grades, era_group_labels)):
        print(f"\n>>> Fold {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)})")

        sampler = Stratified5GradeBatchSampler(grades[train_idx], w_raw[train_idx], batch_size=512, steps_per_epoch=60)

        # Initialize Pillars
        p1 = Pillar1V5SingleCropNet().to(device)
        p2 = Pillar2V5DualCropNet().to(device)
        p3 = Pillar3V5SubcenterNet().to(device)
        p4 = Pillar4V5OrthogonalRankNet().to(device)

        opt1 = optim.AdamW(p1.parameters(), lr=1e-3, weight_decay=1e-4)
        opt2 = optim.AdamW(p2.parameters(), lr=1e-3, weight_decay=1e-4)
        opt3 = optim.AdamW(p3.parameters(), lr=1e-3, weight_decay=1e-4)
        opt4 = optim.AdamW(p4.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(4):
            p1.train(); p2.train(); p3.train(); p4.train()
            for b_local in sampler:
                b_idx = train_idx[b_local]
                t_b_idx = torch.tensor(b_idx, device=device)

                arc_b   = t_arc_all[t_b_idx]
                geom_b  = t_geom_all[t_b_idx]
                sf_b    = t_sf_all[t_b_idx]
                sp_b    = t_sp_all[t_b_idx]
                dino_b  = t_dino_all[t_b_idx]
                fv_b    = t_fv_all[t_b_idx]
                g_b     = t_grades_all[t_b_idx]
                w_b     = t_w_all[t_b_idx]

                # Train P1 (Decoupled Identity Specialist: pure binary match logit, zero CORAL targets)
                opt1.zero_grad()
                z_m1, l_reg1 = p1(arc_b, geom_b, fv_b)
                l1 = loss_fn(z_m1, g_b, coral_logits=None, sample_weights=w_b) + l_reg1
                l1.backward()
                opt1.step()

                # Train P2
                opt2.zero_grad()
                z_m2, c_log2, _, _ = p2(arc_b, geom_b, sf_b, sp_b, dino_b, fv_b)
                l2 = loss_fn(z_m2, g_b, coral_logits=c_log2, sample_weights=w_b)
                l2.backward()
                opt2.step()

                # Train P3
                opt3.zero_grad()
                z_m3, c_log3, _, _, l_sub_reg = p3(arc_b, geom_b, fv_b)
                l3 = loss_fn(z_m3, g_b, coral_logits=c_log3, sample_weights=w_b) + l_sub_reg
                l3.backward()
                opt3.step()

                # Train P4
                opt4.zero_grad()
                z_m4, c_log4, _, _ = p4(arc_b, geom_b, sf_b, sp_b, dino_b, fv_b)
                l4 = loss_fn(z_m4, g_b, coral_logits=c_log4, sample_weights=w_b)
                l4.backward()
                opt4.step()

        # OOF Evaluation
        p1.eval(); p2.eval(); p3.eval(); p4.eval()
        with torch.no_grad():
            v_chunks = 2000
            for vi in range(0, len(val_idx), v_chunks):
                v_slice = val_idx[vi:vi + v_chunks]
                t_v_idx = torch.tensor(v_slice, device=device)

                arc_v  = t_arc_all[t_v_idx]
                geom_v = t_geom_all[t_v_idx]
                sf_v   = t_sf_all[t_v_idx]
                sp_v   = t_sp_all[t_v_idx]
                dino_v = t_dino_all[t_v_idx]
                fv_v   = t_fv_all[t_v_idx]

                z_m1, _ = p1(arc_v, geom_v, fv_v)
                z_m2, c_log2, g2, _ = p2(arc_v, geom_v, sf_v, sp_v, dino_v, fv_v)
                z_m3, c_log3, g3, _, _ = p3(arc_v, geom_v, fv_v)
                z_m4, c_log4, g4, _ = p4(arc_v, geom_v, sf_v, sp_v, dino_v, fv_v)

                Z_oof[v_slice, 0] = z_m1.cpu().numpy()
                Z_oof[v_slice, 1] = z_m2.cpu().numpy()
                Z_oof[v_slice, 2] = z_m3.cpu().numpy()
                Z_oof[v_slice, 3] = z_m4.cpu().numpy()

                g_aes_oof[v_slice, 0] = 3.0
                g_aes_oof[v_slice, 1] = g2.cpu().numpy()
                g_aes_oof[v_slice, 2] = g3.cpu().numpy()
                g_aes_oof[v_slice, 3] = g4.cpu().numpy()

                p_coral3 = torch.sigmoid(c_log3)
                p_g_le2 = 1.0 - p_coral3[:, 1]  # P(G <= 2) = 1 - P(G > 2)
                p_g_le2_oof[v_slice] = p_g_le2.cpu().numpy()

    print(f"5-Fold CV Training Completed in {(time.time() - t0_train)/60:.2f} minutes.")

    # 5. Fit MultiXSafeMetaBlenderV5
    print("\n--- Fitting MultiXSafeMetaBlenderV5 ---")
    meta_blender = MultiXSafeMetaBlenderV5(
        hard_floor=42.5,
        cor_g12_veto=0.55,
        tier1_lock_quantile=0.855,
        tier2_plausible_quantile=0.745,
        tau_lse=0.40
    )
    meta_blender.fit(Z_oof=Z_oof, multi_x_oof=s_multi_x, y_oof=y_raw, high_elo_mask=high_elo_mask)

    # 6. Save Checkpoints
    joblib.dump(meta_blender, MODELS_DIR / "bumbleclaw_v5_meta_blender.joblib")
    torch.save(p1.state_dict(), MODELS_DIR / "bumbleclaw_v5_p1.pt")
    torch.save(p2.state_dict(), MODELS_DIR / "bumbleclaw_v5_p2.pt")
    torch.save(p3.state_dict(), MODELS_DIR / "bumbleclaw_v5_p3.pt")
    torch.save(p4.state_dict(), MODELS_DIR / "bumbleclaw_v5_p4.pt")
    print(f"Saved all v5 checkpoints to {MODELS_DIR}/bumbleclaw_v5_*")

    # 7. Evaluate on Independent Holdout val_1582
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARK PROTOCOL: v4 BINARY vs v5 GRADED (val_1582 HOLDOUT)")
    print("=" * 80)

    # Load val_1582
    r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
    X_arc_val = r_val.embeddings
    y_val = (r_val.ratings >= 50.0).astype(int)
    n_val = len(y_val)

    s_val = np.load(str(root / "embeddings" / "siglip_store_val_1582.npz"))["embeddings"]
    sp_val = np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))
    X_sig_full_val = s_val
    X_sig_port_val = sp_val["embeddings"]
    fv_val = sp_val["face_valid"]

    geom_val = np.load(str(root / "embeddings" / "geom_store_val_1582.npz"))
    X_geom_val = geom_val["geom_16d"]
    dino_val = np.load(str(root / "embeddings" / "dinov2_portrait_store_val_1582.npz"))["embeddings"]

    # Match scores
    df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False).drop_duplicates(subset=["screenshot"], keep="first")
    score_map = {}
    for _, row in df_scores.iterrows():
        fn = row["screenshot"]
        if pd.isna(fn):
            continue
        sc = row.get("score")
        meth = row.get("method")
        r = row.get("ridge")
        m = row.get("multimodal")
        p = row.get("preference_probability")
        if meth == "multimodalx_original" and pd.notna(sc):
            val = float(sc)
        elif pd.notna(r) and pd.notna(m):
            p_val = float(p) if pd.notna(p) else 0.50
            val = float(0.48 * r + 0.48 * m + 0.04 * p_val * 100.0)
        elif pd.notna(sc):
            val = float(sc)
        else:
            val = 50.0
        score_map[fn] = val

    val_fnames = [Path(p).name for p in r_val.paths]
    s_multi_x_val = np.array([score_map.get(fn, 50.0) for fn in val_fnames], dtype=np.float32)

    # 1. Compute v4 Scores
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
    p1_v4 = P1_v4().to(device); p2_v4 = P2_v4().to(device); p3_v4 = P3_v4(K=16, M=48).to(device); p4_v4 = P4_v4().to(device)
    p1_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt"))
    p2_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p2_50k.pt"))
    p3_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p3_50k.pt"))
    p4_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p4_50k.pt"))
    p1_v4.eval(); p2_v4.eval(); p3_v4.eval(); p4_v4.eval()

    meta_v4 = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")
    veto_model = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")
    veto_flags = veto_model.veto(X_geom_val)

    with torch.no_grad():
        t_arc_v = torch.tensor(X_arc_val, device=device)
        t_geom_v = torch.tensor(X_geom_val, device=device)
        t_sf_v = torch.tensor(X_sig_full_val, device=device)
        t_sp_v = torch.tensor(X_sig_port_val, device=device)
        t_dino_v = torch.tensor(dino_val, device=device)
        t_fv_v = torch.tensor(fv_val, device=device)

        z1_v4 = p1_v4(t_arc_v, t_geom_v, t_fv_v).cpu().numpy()
        z2_v4 = p2_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()
        z3_emb = p3_v4.embed(t_arc_v, t_geom_v, t_fv_v)
        z3_v4 = p3_v4.compute_logits(z3_emb).cpu().numpy()
        z4_v4 = p4_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()

        Z_v4 = np.stack([z1_v4, z2_v4, z3_v4, z4_v4], axis=1)
        scores_v4 = meta_v4.predict_score(Z_v4, s_multi_x_val)
        scores_v4 = np.where(veto_flags, -1e9, scores_v4)

    # 2. Compute v5 Scores
    with torch.no_grad():
        z1_v5, _ = p1(t_arc_v, t_geom_v, t_fv_v)
        z2_v5, c_log2_v5, g2_v5, _ = p2(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v)
        z3_v5, c_log3_v5, g3_v5, _, _ = p3(t_arc_v, t_geom_v, t_fv_v)
        z4_v5, c_log4_v5, g4_v5, _ = p4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v)

        Z_v5 = np.stack([
            z1_v5.cpu().numpy(),
            z2_v5.cpu().numpy(),
            z3_v5.cpu().numpy(),
            z4_v5.cpu().numpy()
        ], axis=1)
        mean_g_v5 = np.mean([
            g2_v5.cpu().numpy(),
            g3_v5.cpu().numpy(),
            g4_v5.cpu().numpy()
        ], axis=0)

        p_coral3_v = torch.sigmoid(c_log3_v5)
        p_g_le2_v = (1.0 - p_coral3_v[:, 1]).cpu().numpy()

        scores_v5 = meta_blender.predict_score(
            Z=Z_v5,
            multi_x=s_multi_x_val,
            g_aes_hat=mean_g_v5,
            prob_g_le_2=p_g_le2_v,
            gmm_veto=veto_flags
        )

    # 3. Run Deliverable 5 Statistical Benchmark Protocol
    eval_res = evaluate_holdout_v4_vs_v5(
        scores_v4=scores_v4,
        scores_v5=scores_v5,
        y_true=y_val,
        multi_x=s_multi_x_val,
        setpoint=0.208,
        n_boot=10000
    )

    print("\n--- Deliverable 5 Benchmark Results ---")
    print(f"v4 Summary: {eval_res['v4_summary']}")
    print(f"v5 Summary: {eval_res['v5_summary']}")
    print(f"McNemar True Likes Test: {eval_res['mcnemar_true_likes']}")
    print(f"Brunner-Munzel FP Stochastic Dominance: {eval_res['brunner_munzel_fp_dominance']}")
    print(f"10,000-Iteration Bootstrap: {eval_res['paired_bootstrap_10k']}")


if __name__ == "__main__":
    main()
