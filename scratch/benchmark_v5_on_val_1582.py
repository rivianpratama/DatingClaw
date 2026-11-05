import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch

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
from face_similarity.v5_architecture import Pillar1V5SingleCropNet, MultiXSafeMetaBlenderV5, evaluate_holdout_v4_vs_v5
from scratch.train_50k_v5_graded_ensemble import (
    Pillar2V5DualCropNet,
    Pillar3V5SubcenterNet,
    Pillar4V5OrthogonalRankNet
)

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = root / "models"


def main():
    print("=" * 80)
    print("DELIVERABLE 5 BENCHMARK: BUMBLECLAW v4 (BINARY) vs v5 (GRADED)")
    print("=" * 80)

    # 1. Load val_1582 Holdout Data
    r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
    X_arc_val = r_val.embeddings
    y_val = (r_val.ratings >= 50.0).astype(int)
    n_val = len(y_val)
    n_likes = int(y_val.sum())

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
    print(f"Loaded {n_val} holdout profiles | True Likes: {n_likes} ({n_likes/n_val*100:.2f}%)")

    # Load Veto Model
    veto_model = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")
    veto_flags = veto_model.veto(X_geom_val)

    # Move inputs to CUDA
    t_arc_v = torch.tensor(X_arc_val, device=device)
    t_geom_v = torch.tensor(X_geom_val, device=device)
    t_sf_v = torch.tensor(X_sig_full_val, device=device)
    t_sp_v = torch.tensor(X_sig_port_val, device=device)
    t_dino_v = torch.tensor(dino_val, device=device)
    t_fv_v = torch.tensor(fv_val, device=device)

    # 2. Run v4 Binary Model Inference
    print("\n[v4 Binary] Running Inference...")
    p1_v4 = P1_v4().to(device); p2_v4 = P2_v4().to(device); p3_v4 = P3_v4(K=16, M=48).to(device); p4_v4 = P4_v4().to(device)
    p1_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt"))
    p2_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p2_50k.pt"))
    p3_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p3_50k.pt"))
    p4_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p4_50k.pt"))
    p1_v4.eval(); p2_v4.eval(); p3_v4.eval(); p4_v4.eval()

    meta_v4 = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")
    with torch.no_grad():
        z1_v4 = p1_v4(t_arc_v, t_geom_v, t_fv_v).cpu().numpy()
        z2_v4 = p2_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()
        z3_emb = p3_v4.embed(t_arc_v, t_geom_v, t_fv_v)
        z3_v4 = p3_v4.compute_logits(z3_emb).cpu().numpy()
        z4_v4 = p4_v4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()

        Z_v4 = np.stack([z1_v4, z2_v4, z3_v4, z4_v4], axis=1)
        scores_v4 = meta_v4.predict_score(Z_v4, s_multi_x_val)
        scores_v4 = np.where(veto_flags, -1e9, scores_v4)

    # 3. Run v5 Graded Model Inference
    print("[v5 Graded] Running Inference...")
    p1_v5 = Pillar1V5SingleCropNet().to(device)
    p2_v5 = Pillar2V5DualCropNet().to(device)
    p3_v5 = Pillar3V5SubcenterNet().to(device)
    p4_v5 = Pillar4V5OrthogonalRankNet().to(device)

    p1_v5.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p1.pt"))
    p2_v5.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p2.pt"))
    p3_v5.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p3.pt"))
    p4_v5.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p4.pt"))
    p1_v5.eval(); p2_v5.eval(); p3_v5.eval(); p4_v5.eval()

    meta_v5 = joblib.load(MODELS_DIR / "bumbleclaw_v5_meta_blender.joblib")
    with torch.no_grad():
        z1_v5, _ = p1_v5(t_arc_v, t_geom_v, t_fv_v)
        z2_v5, c_log2_v5, g2_v5, _ = p2_v5(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v)
        z3_v5, c_log3_v5, g3_v5, _, _ = p3_v5(t_arc_v, t_geom_v, t_fv_v)
        z4_v5, c_log4_v5, g4_v5, _ = p4_v5(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v)

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

        scores_v5 = meta_v5.predict_score(
            Z=Z_v5,
            multi_x=s_multi_x_val,
            g_aes_hat=mean_g_v5,
            prob_g_le_2=p_g_le2_v,
            gmm_veto=veto_flags
        )

    # 4. Statistical Evaluation Protocol
    print("\nRunning Deliverable 5 Hypothesis Tests (10,000 bootstrap iterations)...")
    t0_eval = time.time()
    res = evaluate_holdout_v4_vs_v5(
        scores_v4=scores_v4,
        scores_v5=scores_v5,
        y_true=y_val,
        multi_x=s_multi_x_val,
        setpoint=0.208,
        n_boot=10000
    )
    print(f"Statistical Evaluation Completed in {time.time() - t0_eval:.2f}s")

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON: v4 BINARY vs v5 GRADED (val_1582)")
    print("=" * 80)
    v4_s = res["v4_summary"]
    v5_s = res["v5_summary"]

    print(f"{'Metric':<35} | {'v4 Binary':<15} | {'v5 Graded':<15} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Hits @ 20.8% Setpoint (319 Swipes)':<35} | {v4_s['hits_at_k']:<15} | {v5_s['hits_at_k']:<15} | {v5_s['hits_at_k'] - v4_s['hits_at_k']:+d}")
    print(f"{'Recall @ 20.8% Setpoint':<35} | {v4_s['recall_at_k']*100:.2f}%{'':<9} | {v5_s['recall_at_k']*100:.2f}%{'':<9} | {(v5_s['recall_at_k'] - v4_s['recall_at_k'])*100:+.2f}%")
    print(f"{'Binary NDCG @ 20.8%':<35} | {v4_s['ndcg_binary_at_k']:<15.4f} | {v5_s['ndcg_binary_at_k']:<15.4f} | {v5_s['ndcg_binary_at_k'] - v4_s['ndcg_binary_at_k']:+.4f}")
    print(f"{'Graded NDCG @ 20.8%':<35} | {v4_s['ndcg_graded_at_k']:<15.4f} | {v5_s['ndcg_graded_at_k']:<15.4f} | {v5_s['ndcg_graded_at_k'] - v4_s['ndcg_graded_at_k']:+.4f}")
    print(f"{'Cute False-Positive Share (Grade 4)':<35} | {v4_s['cute_fp_share_g4']*100:.2f}%{'':<9} | {v5_s['cute_fp_share_g4']*100:.2f}%{'':<9} | {(v5_s['cute_fp_share_g4'] - v4_s['cute_fp_share_g4'])*100:+.2f}%")
    print(f"{'Catastrophic FPs (Grade 1 & 2)':<35} | {v4_s['catastrophic_fp_g1_g2']:<15} | {v5_s['catastrophic_fp_g1_g2']:<15} | {v5_s['catastrophic_fp_g1_g2'] - v4_s['catastrophic_fp_g1_g2']:+d}")
    print(f"{'Approved FP MultiX Mean':<35} | {v4_s['fp_multix_mean']:<15.2f} | {v5_s['fp_multix_mean']:<15.2f} | {v5_s['fp_multix_mean'] - v4_s['fp_multix_mean']:+.2f}")
    print(f"{'Approved FP MultiX 5% CVaR':<35} | {v4_s['fp_multix_cvar5']:<15.2f} | {v5_s['fp_multix_cvar5']:<15.2f} | {v5_s['fp_multix_cvar5'] - v4_s['fp_multix_cvar5']:+.2f}")
    print("-" * 80)
    print("\n--- Statistical Significance Testing ---")
    print(f"1. Exact McNemar Mid-p on True Likes: p = {res['mcnemar_true_likes']['mid_p_value']:.4f} (v5 Unique: {res['mcnemar_true_likes']['v5_unique']}, v4 Unique: {res['mcnemar_true_likes']['v4_unique']})")
    print(f"2. Brunner-Munzel FP Stochastic Dominance: p = {res['brunner_munzel_fp_dominance']['p_value_one_sided']:.4e} (Statistic: {res['brunner_munzel_fp_dominance']['statistic']:.4f})")
    boot = res["paired_bootstrap_10k"]
    print(f"3. 10,000-Iteration Bootstrap Delta FP Mean: {boot['delta_fp_multix_mean']:+.3f} (99% CI: {boot['delta_fp_multix_99ci']}) | p = {boot['p_value_fp_superiority']:.4e}")
    print(f"4. Graded NDCG Superiority p-value: {boot['p_value_graded_ndcg_superiority']:.4e}")


if __name__ == "__main__":
    main()
