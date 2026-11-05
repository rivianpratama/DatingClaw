import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

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

from scratch.extract_and_train_dinov3 import (
    Pillar1SingleCropNet as P1_d3,
    Pillar2DualCropNet as P2_d3,
    Pillar3SubcenterNet as P3_d3,
    Pillar4OrthogonalRankNet as P4_d3
)
from face_similarity.v5_architecture import evaluate_holdout_v4_vs_v5

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = root / "models"


def main():
    print("=" * 90)
    print("BENCHMARK PROTOCOL: v4 (DINOv2) vs v5.2 (DINOv3) ON val_1582 HOLDOUT")
    print("=" * 90)

    # 1. Load val_1582 Data
    r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
    y_val = (r_val.ratings >= 50.0).astype(int)
    N = len(y_val)
    K = int(np.floor(0.208 * N))

    geom_val = np.load(str(root / "embeddings" / "geom_store_val_1582.npz"))
    X_geom_v = geom_val["geom_16d"]

    veto_model = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")
    veto_flags = veto_model.veto(X_geom_v)

    df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False).drop_duplicates(subset=["screenshot"], keep="first")
    smap = {
        row["screenshot"]: (
            float(row["score"]) if row.get("method") == "multimodalx_original" and pd.notna(row.get("score"))
            else (0.48 * float(row["ridge"]) + 0.48 * float(row["multimodal"]) + 4.0 if pd.notna(row.get("ridge")) and pd.notna(row.get("multimodal")) else 50.0)
        )
        for _, row in df_scores.iterrows() if pd.notna(row.get("screenshot"))
    }
    s_multi_x_val = np.array([smap.get(Path(p).name, 50.0) for p in r_val.paths], dtype=np.float32)

    # Common Tensors
    t_arc_v = torch.tensor(r_val.embeddings, device=device)
    t_geom_v = torch.tensor(X_geom_v, device=device)
    t_sf_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_store_val_1582.npz"))["embeddings"], device=device)
    t_sp_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))["embeddings"], device=device)
    t_fv_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))["face_valid"], device=device)

    # 2. Run v4 Inference (DINOv2)
    print("\n[v4 Binary with DINOv2] Running Inference...")
    t_dino_v2 = torch.tensor(np.load(str(root / "embeddings" / "dinov2_portrait_store_val_1582.npz"))["embeddings"], device=device)

    p1_v4 = P1_v4().to(device); p1_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt")); p1_v4.eval()
    p2_v4 = P2_v4().to(device); p2_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p2_50k.pt")); p2_v4.eval()
    p3_v4 = P3_v4(K=16, M=48).to(device); p3_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p3_50k.pt")); p3_v4.eval()
    p4_v4 = P4_v4().to(device); p4_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p4_50k.pt")); p4_v4.eval()

    meta_v4 = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")

    with torch.no_grad():
        z1_v4 = p1_v4(t_arc_v, t_geom_v, t_fv_v).cpu().numpy()
        z2_v4 = p2_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v2, t_fv_v).cpu().numpy()
        z3_v4 = p3_v4.compute_logits(p3_v4.embed(t_arc_v, t_geom_v, t_fv_v)).cpu().numpy()
        z4_v4 = p4_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v2, t_fv_v).cpu().numpy()

        Z_val_v4 = np.stack([z1_v4, z2_v4, z3_v4, z4_v4], axis=1)
        s_v4 = meta_v4.predict_score(Z_val_v4, s_multi_x_val)
        s_v4 = np.where(veto_flags, -1e9, s_v4)

    # 3. Run v5.2 Inference (DINOv3)
    print("[v5.2 Super-Ensemble with DINOv3] Running Inference...")
    val_dino_d3 = np.load(str(root / "embeddings" / "dinov3_portrait_store_val_1582.npz"))["embeddings"]
    t_dino_d3 = torch.tensor(val_dino_d3, device=device)

    p1_d3 = P1_d3().to(device); p1_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p1_dinov3.pt")); p1_d3.eval()
    p2_d3 = P2_d3().to(device); p2_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p2_dinov3.pt")); p2_d3.eval()
    p3_d3 = P3_d3(K=16, M=48).to(device); p3_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p3_dinov3.pt")); p3_d3.eval()
    p4_d3 = P4_d3().to(device); p4_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p4_dinov3.pt")); p4_d3.eval()

    meta_d3 = joblib.load(MODELS_DIR / "bumbleclaw_v5_meta_blender_dinov3.joblib")

    with torch.no_grad():
        z1_d3 = p1_d3(t_arc_v, t_geom_v, t_fv_v).cpu().numpy()
        z2_d3 = p2_d3(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_d3, t_fv_v).cpu().numpy()
        z3_d3 = p3_d3.compute_logits(p3_d3.embed(t_arc_v, t_geom_v, t_fv_v)).cpu().numpy()
        z4_d3 = p4_d3(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_d3, t_fv_v).cpu().numpy()

        Z_val_d3 = np.stack([z1_d3, z2_d3, z3_d3, z4_d3], axis=1)
        s_d3 = meta_d3.predict_score(Z_val_d3, s_multi_x_val)
        s_d3 = np.where(veto_flags, -1e9, s_d3)

    # Check individual pillar hits for DINOv3
    print("\n--- Individual Pillar Hits @ 20.8% (val_1582) ---")
    for name, z_col in [("P1 (FiLM ArcFace)", z1_d3), ("P2 (Dual-Crop SigLIP + DINOv3)", z2_d3), ("P3 (Pyramidal Subcenters)", z3_d3), ("P4 (Orthogonal RankNet + DINOv3)", z4_d3)]:
        topK_p = np.argsort(z_col)[::-1][:K]
        hits_p = int(y_val[topK_p].sum())
        print(f"  {name:<38}: {hits_p:3d} / 140 ({hits_p/140*100:.2f}%)")

    # Apply v5.1 Conditional Low-MultiX Safety Gate to DINOv3
    L_d3 = meta_d3._calibrated_log_odds(Z_val_d3)
    w_p = meta_d3.meta_params[:4]
    w_lse = meta_d3.meta_params[4]
    bias = meta_d3.meta_params[7]
    tau_lse = meta_d3.tau_lse
    m_max = np.max(L_d3 / tau_lse, axis=1, keepdims=True)
    l_lse = tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L_d3 / tau_lse - m_max), axis=1)))
    s_match_d3 = (L_d3 @ w_p) + w_lse * l_lse + bias

    is_vetoed_d3 = veto_flags | (s_multi_x_val < 42.5) | ((s_multi_x_val < 49.0) & (s_match_d3 < 0.0))
    s_v52_dinov3 = np.where(is_vetoed_d3, -1e9, s_d3)

    # Hybrid Dual-Foundation Blend: DINOv2 + DINOv3
    s_hybrid = 0.50 * s_v4 + 0.50 * s_v52_dinov3

    # 4. Statistical Evaluation Protocol
    print("\nRunning Deliverable 5 Hypothesis Tests (10,000 bootstrap iterations)...")
    t0_eval = time.time()
    res_d3 = evaluate_holdout_v4_vs_v5(scores_v4=s_v4, scores_v5=s_v52_dinov3, y_true=y_val, multi_x=s_multi_x_val, setpoint=0.208, n_boot=10000)
    res_hyb = evaluate_holdout_v4_vs_v5(scores_v4=s_v4, scores_v5=s_hybrid, y_true=y_val, multi_x=s_multi_x_val, setpoint=0.208, n_boot=10000)
    print(f"Statistical Evaluation Completed in {time.time() - t0_eval:.2f}s")

    v4_s = res_d3["v4_summary"]
    d3_s = res_d3["v5_summary"]
    hyb_s = res_hyb["v5_summary"]

    print("\n" + "=" * 125)
    print(f"{'Metric':<36} | {'v4 (DINOv2)':<18} | {'v5.2 (DINOv3)':<18} | {'v5.2 Hybrid (D2+D3)':<20} | {'Delta vs v4':<12}")
    print("-" * 125)
    print(f"{'Hits @ 20.8% Setpoint (318 Swipes)':<36} | {v4_s['hits_at_k']:<18} | {d3_s['hits_at_k']:<18} | {hyb_s['hits_at_k']:<20} | {hyb_s['hits_at_k'] - v4_s['hits_at_k']:+d} likes")
    print(f"{'Recall @ 20.8% Setpoint':<36} | {v4_s['recall_at_k']*100:.2f}%{'':<11} | {d3_s['recall_at_k']*100:.2f}%{'':<11} | {hyb_s['recall_at_k']*100:.2f}%{'':<13} | {(hyb_s['recall_at_k'] - v4_s['recall_at_k'])*100:+.2f}%")
    print(f"{'Binary NDCG @ 20.8%':<36} | {v4_s['ndcg_binary_at_k']:<18.4f} | {d3_s['ndcg_binary_at_k']:<18.4f} | {hyb_s['ndcg_binary_at_k']:<20.4f} | {hyb_s['ndcg_binary_at_k'] - v4_s['ndcg_binary_at_k']:+.4f}")
    print(f"{'Graded NDCG @ 20.8%':<36} | {v4_s['ndcg_graded_at_k']:<18.4f} | {d3_s['ndcg_graded_at_k']:<18.4f} | {hyb_s['ndcg_graded_at_k']:<20.4f} | {hyb_s['ndcg_graded_at_k'] - v4_s['ndcg_graded_at_k']:+.4f}")
    print(f"{'Cute False-Positive Share (Grade 4)':<36} | {v4_s['cute_fp_share_g4']*100:.2f}%{'':<11} | {d3_s['cute_fp_share_g4']*100:.2f}%{'':<11} | {hyb_s['cute_fp_share_g4']*100:.2f}%{'':<13} | {(hyb_s['cute_fp_share_g4'] - v4_s['cute_fp_share_g4'])*100:+.2f}%")
    print(f"{'Catastrophic FPs (Grade 1 & 2)':<36} | {v4_s['catastrophic_fp_g1_g2']:<18} | {d3_s['catastrophic_fp_g1_g2']:<18} | {hyb_s['catastrophic_fp_g1_g2']:<20} | {hyb_s['catastrophic_fp_g1_g2'] - v4_s['catastrophic_fp_g1_g2']:+d}")
    print(f"{'Approved FP MultiX Mean':<36} | {v4_s['fp_multix_mean']:<18.2f} | {d3_s['fp_multix_mean']:<18.2f} | {hyb_s['fp_multix_mean']:<20.2f} | {hyb_s['fp_multix_mean'] - v4_s['fp_multix_mean']:+.2f} pts")
    print(f"{'Approved FP MultiX 5% CVaR':<36} | {v4_s['fp_multix_cvar5']:<18.2f} | {d3_s['fp_multix_cvar5']:<18.2f} | {hyb_s['fp_multix_cvar5']:<20.2f} | {hyb_s['fp_multix_cvar5'] - v4_s['fp_multix_cvar5']:+.2f} pts")
    print("-" * 125)

    print("\n--- Statistical Significance Testing (Hybrid vs v4) ---")
    print(f"1. Exact McNemar Mid-p on True Likes: p = {res_hyb['mcnemar_true_likes']['mid_p_value']:.4f} (Hybrid Unique: {res_hyb['mcnemar_true_likes']['v5_unique']}, v4 Unique: {res_hyb['mcnemar_true_likes']['v4_unique']})")
    print(f"2. Brunner-Munzel FP Stochastic Dominance: p = {res_hyb['brunner_munzel_fp_dominance']['p_value_one_sided']:.4e} (Statistic: {res_hyb['brunner_munzel_fp_dominance']['statistic']:.4f})")
    boot = res_hyb["paired_bootstrap_10k"]
    print(f"3. 10,000-Iteration Bootstrap Delta FP Mean: {boot['delta_fp_multix_mean']:+.3f} (99% CI: {boot['delta_fp_multix_99ci']}) | p = {boot['p_value_fp_superiority']:.4e}")
    print(f"4. Graded NDCG Superiority p-value: {boot['p_value_graded_ndcg_superiority']:.4e}")


if __name__ == "__main__":
    main()
