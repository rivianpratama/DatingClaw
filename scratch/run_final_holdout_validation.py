import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
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
HOLDOUT_NPZ = Path(r"D:\holdout_4540_store.npz")


def compute_metrics(scores, y_true, multi_x, K, name="Model"):
    N = len(y_true)
    N_pos = int(y_true.sum())
    top_k_indices = np.argsort(scores)[::-1][:K]
    top_k_mask = np.zeros(N, dtype=bool)
    top_k_mask[top_k_indices] = True

    hits = int(y_true[top_k_indices].sum())
    recall = float(hits / N_pos) if N_pos > 0 else 0.0

    # Binary NDCG @ K
    def dcg(rel, k):
        r = rel[:k]
        return np.sum(r / np.log2(np.arange(2, len(r) + 2)))

    ideal_binary = np.zeros(N, dtype=np.float32)
    ideal_binary[:N_pos] = 1.0
    idcg_bin = dcg(ideal_binary, K)
    actual_bin = y_true[top_k_indices].astype(np.float32)
    ndcg_bin = float(dcg(actual_bin, K) / idcg_bin) if idcg_bin > 0 else 0.0

    # Graded relevance
    # 5: True Like (32.0), 4: cute dislike (6.0), 3: avg pass (2.0), 2: low (0.5), 1: structural reject (0.0)
    g_arr = np.zeros(N, dtype=np.float32)
    for idx in range(N):
        if y_true[idx] == 1:
            g_arr[idx] = 32.0
        else:
            mx = multi_x[idx]
            if mx >= 60.0:
                g_arr[idx] = 6.0
            elif mx >= 50.0:
                g_arr[idx] = 2.0
            elif mx >= 40.0:
                g_arr[idx] = 0.5
            else:
                g_arr[idx] = 0.0

    ideal_graded = np.sort(g_arr)[::-1]
    idcg_grad = dcg(ideal_graded, K)
    actual_grad = g_arr[top_k_indices]
    ndcg_grad = float(dcg(actual_grad, K) / idcg_grad) if idcg_grad > 0 else 0.0

    # FP analysis
    fp_mask = top_k_mask & (y_true == 0)
    fp_multix = multi_x[fp_mask]
    n_fp = len(fp_multix)

    cute_fp_count = int(np.sum(fp_multix >= 60.0))
    cute_fp_share = float(cute_fp_count / n_fp) if n_fp > 0 else 0.0
    catastrophic_fp_count = int(np.sum(fp_multix < 50.0))

    fp_mean = float(np.mean(fp_multix)) if n_fp > 0 else 0.0
    fp_cvar5 = float(np.percentile(fp_multix, 5)) if n_fp > 0 else 0.0

    return {
        "name": name,
        "hits": hits,
        "n_pos": N_pos,
        "recall": recall,
        "ndcg_binary": ndcg_bin,
        "ndcg_graded": ndcg_grad,
        "cute_fp_share": cute_fp_share,
        "catastrophic_fp": catastrophic_fp_count,
        "fp_mean": fp_mean,
        "fp_cvar5": fp_cvar5,
        "scores": scores
    }


def main():
    print("=" * 110)
    print("FINAL HOLDOUT VALIDATION TOURNAMENT: CARDS 50,071 TO 54,610 (N = 4,540 UNSEEN)")
    print("=" * 110)

    if not HOLDOUT_NPZ.exists():
        print(f"[ERROR] {HOLDOUT_NPZ} does not exist.")
        return

    data = np.load(str(HOLDOUT_NPZ), allow_pickle=True)
    X_arc = data["X_arc"]
    X_geom = data["X_geom"]
    X_sf = data["X_sig_full"]
    X_sp = data["X_sig_port"]
    X_d2 = data["X_dino2"]
    X_d3 = data["X_dino3"]
    face_valid = data["face_valid"]
    hard_veto = data["hard_veto"]
    y_true = data["y"]
    multi_x = data["s_multi_x"]

    N = len(y_true)
    N_pos = int(y_true.sum())
    K = int(np.floor(0.208 * N))
    print(f"Holdout Dataset Size: N = {N} profiles | True Likes = {N_pos} ({N_pos/N*100:.2f}%) | Passes = {N - N_pos}")
    print(f"Right-Swipe Quota (20.8% Setpoint): K = {K} swipes out of {N}\n")

    # Tensors on device
    t_arc = torch.tensor(X_arc, device=device)
    t_geom = torch.tensor(X_geom, device=device)
    t_sf = torch.tensor(X_sf, device=device)
    t_sp = torch.tensor(X_sp, device=device)
    t_d2 = torch.tensor(X_d2, device=device)
    t_d3 = torch.tensor(X_d3, device=device)
    t_fv = torch.tensor(face_valid, device=device)

    # Conformal GMM Veto
    veto_model = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")
    gmm_veto_flags = veto_model.veto(X_geom)
    combined_veto = hard_veto | gmm_veto_flags

    results = []

    # 1. Baseline: MultimodalX Original
    print("1. Evaluating Baseline: MultimodalX Original...")
    s_multix = multi_x.copy()
    results.append(compute_metrics(s_multix, y_true, multi_x, K, name="MultimodalX Original"))

    # 2. Legacy: 10k Pure Ridge
    print("2. Evaluating Legacy Bumble 10k Pure Ridge...")
    reg_10k_path = MODELS_DIR / "rating_regressor_bumble_10k_pure.joblib"
    if reg_10k_path.exists():
        r10k = joblib.load(reg_10k_path)
        est10k = r10k["estimator"] if isinstance(r10k, dict) else (r10k.estimator if hasattr(r10k, "estimator") else r10k)
        p10k = est10k.predict(X_arc)
        results.append(compute_metrics(p10k, y_true, multi_x, K, name="Legacy 10k Pure Ridge"))

    # 3. Legacy: 20k Pure Ridge
    print("3. Evaluating Legacy Bumble 20k Pure Ridge...")
    reg_20k_path = MODELS_DIR / "rating_regressor_bumble_20k_pure.joblib"
    if reg_20k_path.exists():
        r20k = joblib.load(reg_20k_path)
        est20k = r20k["estimator"] if isinstance(r20k, dict) else (r20k.estimator if hasattr(r20k, "estimator") else r20k)
        p20k = est20k.predict(X_arc)
        results.append(compute_metrics(p20k, y_true, multi_x, K, name="Legacy 20k Pure Ridge"))

    # 4. Legacy: 30k Pure Ridge
    print("4. Evaluating Legacy Bumble 30k Pure Ridge...")
    reg_30k_path = MODELS_DIR / "rating_regressor_bumble_30k_pure.joblib"
    if reg_30k_path.exists():
        r30k = joblib.load(reg_30k_path)
        est30k = r30k["estimator"] if isinstance(r30k, dict) else (r30k.estimator if hasattr(r30k, "estimator") else r30k)
        p30k = est30k.predict(X_arc)
        results.append(compute_metrics(p30k, y_true, multi_x, K, name="Legacy 30k Pure Ridge"))

    # 5. v4 Binary Grand Super-Ensemble (50k DINOv2)
    print("5. Evaluating v4 Binary Grand Super-Ensemble (50k DINOv2)...")
    p1_v4 = P1_v4().to(device); p1_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt")); p1_v4.eval()
    p2_v4 = P2_v4().to(device); p2_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p2_50k.pt")); p2_v4.eval()
    p3_v4 = P3_v4(K=16, M=48).to(device); p3_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p3_50k.pt")); p3_v4.eval()
    p4_v4 = P4_v4().to(device); p4_v4.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v4_p4_50k.pt")); p4_v4.eval()
    meta_v4 = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")

    with torch.no_grad():
        z1_v4 = p1_v4(t_arc, t_geom, t_fv).cpu().numpy()
        z2_v4 = p2_v4(t_arc, t_geom, t_sf, t_sp, t_d2, t_fv).cpu().numpy()
        z3_v4 = p3_v4.compute_logits(p3_v4.embed(t_arc, t_geom, t_fv)).cpu().numpy()
        z4_v4 = p4_v4(t_arc, t_geom, t_sf, t_sp, t_d2, t_fv).cpu().numpy()

        Z_v4 = np.stack([z1_v4, z2_v4, z3_v4, z4_v4], axis=1)
        s_v4 = meta_v4.predict_score(Z_v4, multi_x)
        s_v4 = np.where(combined_veto, -1e9, s_v4)

    results.append(compute_metrics(s_v4, y_true, multi_x, K, name="v4 Binary (50k DINOv2)"))

    # 6. DINOv3 Individual Pillars
    print("6. Evaluating v5.2 DINOv3 Pillars and Meta-Blender...")
    p1_d3 = P1_d3().to(device); p1_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p1_dinov3.pt")); p1_d3.eval()
    p2_d3 = P2_d3().to(device); p2_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p2_dinov3.pt")); p2_d3.eval()
    p3_d3 = P3_d3(K=16, M=48).to(device); p3_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p3_dinov3.pt")); p3_d3.eval()
    p4_d3 = P4_d3().to(device); p4_d3.load_state_dict(torch.load(MODELS_DIR / "bumbleclaw_v5_p4_dinov3.pt")); p4_d3.eval()
    meta_d3 = joblib.load(MODELS_DIR / "bumbleclaw_v5_meta_blender_dinov3.joblib")

    with torch.no_grad():
        z1_d3 = p1_d3(t_arc, t_geom, t_fv).cpu().numpy()
        z2_d3 = p2_d3(t_arc, t_geom, t_sf, t_sp, t_d3, t_fv).cpu().numpy()
        z3_d3 = p3_d3.compute_logits(p3_d3.embed(t_arc, t_geom, t_fv)).cpu().numpy()
        z4_d3 = p4_d3(t_arc, t_geom, t_sf, t_sp, t_d3, t_fv).cpu().numpy()

        # Check pillar 4 standalone
        results.append(compute_metrics(np.where(combined_veto, -1e9, z4_d3), y_true, multi_x, K, name="  -> P4 Orthogonal Bilinear (DINOv3)"))

        Z_d3 = np.stack([z1_d3, z2_d3, z3_d3, z4_d3], axis=1)
        raw_s_d3 = meta_d3.predict_score(Z_d3, multi_x)

        L_d3 = meta_d3._calibrated_log_odds(Z_d3)
        w_p = meta_d3.meta_params[:4]
        w_lse = meta_d3.meta_params[4]
        bias = meta_d3.meta_params[7]
        tau_lse = meta_d3.tau_lse
        m_max = np.max(L_d3 / tau_lse, axis=1, keepdims=True)
        l_lse = tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L_d3 / tau_lse - m_max), axis=1)))
        s_match_d3 = (L_d3 @ w_p) + w_lse * l_lse + bias

        is_vetoed_d3 = combined_veto | (multi_x < 42.5) | ((multi_x < 49.0) & (s_match_d3 < 0.0))
        s_v52_dinov3 = np.where(is_vetoed_d3, -1e9, raw_s_d3)

    results.append(compute_metrics(s_v52_dinov3, y_true, multi_x, K, name="v5.2 Pure DINOv3 Super-Ensemble"))

    # 7. v5.2 Hybrid Super-Ensemble (Dual Foundation DINOv2 + DINOv3)
    print("7. Evaluating v5.2 Hybrid Super-Ensemble (Dual Foundation DINOv2 + DINOv3)...")
    s_hybrid = 0.50 * s_v4 + 0.50 * s_v52_dinov3
    results.append(compute_metrics(s_hybrid, y_true, multi_x, K, name="v5.2 Hybrid (D2 + D3)"))

    # Print Full Tournament Table
    print("\n" + "=" * 135)
    print(f"{'Model Architecture':<36} | {'Hits @ 20.8%':<14} | {'Recall':<9} | {'Bin NDCG':<10} | {'Grd NDCG':<10} | {'Cute FP %':<11} | {'Cat FP':<8} | {'FP MultiX':<10} | {'FP CVaR':<8}")
    print("-" * 135)
    for res in results:
        print(f"{res['name']:<36} | {res['hits']:3d} / {res['n_pos']}     | {res['recall']*100:6.2f}% | {res['ndcg_binary']:<10.4f} | {res['ndcg_graded']:<10.4f} | {res['cute_fp_share']*100:6.2f}%   | {res['catastrophic_fp']:<8d} | {res['fp_mean']:<10.2f} | {res['fp_cvar5']:<8.2f}")
    print("=" * 135)

    # Hypothesis Testing: Hybrid vs v4 Baseline
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING: v5.2 Hybrid vs. v4 Baseline (10,000 Bootstraps)")
    print("=" * 80)
    eval_stat = evaluate_holdout_v4_vs_v5(scores_v4=s_v4, scores_v5=s_hybrid, y_true=y_true, multi_x=multi_x, setpoint=0.208, n_boot=10000)
    mcn = eval_stat["mcnemar_true_likes"]
    bm = eval_stat["brunner_munzel_fp_dominance"]
    boot = eval_stat["paired_bootstrap_10k"]

    print(f"1. Exact McNemar Mid-p Test on True Likes: p = {mcn['mid_p_value']:.4f} (Hybrid Unique: {mcn['v5_unique']}, v4 Unique: {mcn['v4_unique']})")
    print(f"2. Brunner-Munzel Stochastic FP Dominance: statistic = {bm['statistic']:.4f}, p = {bm['p_value_one_sided']:.4e}")
    print(f"3. 10,000-Iteration Paired Bootstrap Delta FP MultiX Mean: {boot['delta_fp_multix_mean']:+.3f} (99% CI: {boot['delta_fp_multix_99ci']}) | p = {boot['p_value_fp_superiority']:.4e}")
    print(f"4. Graded NDCG Superiority p-value: {boot['p_value_graded_ndcg_superiority']:.4e}")

    # Latency verification
    print("\n--- Latency Benchmark on RTX 2060 (Batch Size = 1) ---")
    torch.cuda.synchronize()
    latencies = []
    for _ in range(50):
        t0 = time.time()
        _ = 0.50 * p4_v4(t_arc[:1], t_geom[:1], t_sf[:1], t_sp[:1], t_d2[:1], t_fv[:1]) + 0.50 * p4_d3(t_arc[:1], t_geom[:1], t_sf[:1], t_sp[:1], t_d3[:1], t_fv[:1])
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0)
    lat_p50 = np.median(latencies)
    lat_p95 = np.percentile(latencies, 95)
    print(f"  Pillar 4 Dual-Foundation Latency: p50 = {lat_p50:.2f} ms | p95 = {lat_p95:.2f} ms (< 25 ms target: {'PASS' if lat_p95 < 25.0 else 'FAIL'})")


if __name__ == "__main__":
    main()
