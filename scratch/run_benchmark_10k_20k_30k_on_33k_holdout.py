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
import joblib
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store
from scratch.train_subcenter_siglip_phase2 import SiglipGatedResidualFusion, MultiProxyClassifier
from scratch.train_subcenter_multiproxy_phase1 import GatedResidualFusion as ClipGatedResidualFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_holdout_metrics(preds, y_true):
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
        "SwipeRate@90%": f"{swipe_rate_90:.1f}%",
        "SwipeRate@95%": f"{swipe_rate_95:.1f}%",
    }

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCIENTIFIC BENCHMARK: 10K vs 20K vs 30K ON BRAND-NEW 30k-33.6k UNTRAINED LABELS")
    print("=" * 80)

    # 1. Load Untrained Holdout Store (30k-33.6k)
    ref_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    clip_holdout = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))
    siglip_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))

    y_val = (ref_holdout.ratings >= 50.0).astype(int)
    X_face = ref_holdout.embeddings
    X_clip = clip_holdout['embeddings']
    X_siglip = siglip_holdout['embeddings']

    n_total = len(y_val)
    n_likes = int(y_val.sum())
    print(f"Untrained Holdout: {n_total} profiles | Likes: {n_likes} ({n_likes/n_total*100:.2f}%) | Passes: {n_total - n_likes}")
    print(f"Strict <= 25% Swipe Budget: {int(np.round(0.25 * n_total))} right swipes\n")

    results = []

    # --------------------------------------------------------------------------
    # 1. Ridge Regressors (Face Only: 512D)
    # --------------------------------------------------------------------------
    for scale in ["10k", "20k", "30k"]:
        model_path = root / "models" / f"rating_regressor_bumble_{scale}_pure.joblib"
        if not model_path.exists():
            continue
        obj = joblib.load(str(model_path))
        estimator = obj['estimator']
        preds = estimator.predict(X_face)
        stats = evaluate_holdout_metrics(preds, y_val)
        stats['Model'] = f"Ridge Face ({scale} Pure)"
        results.append(stats)
        print(f"[Ridge {scale}] Likes@25%: {stats['Likes@25%Cap']} | HitRate: {stats['HitRate@25%']} | ROC: {stats['ROC-AUC']} | PR-AUC: {stats['PR-AUC']} | Swipe@95%: {stats['SwipeRate@95%']}")

    # --------------------------------------------------------------------------
    # 2. Multimodal Regressors (Face + CLIP ViT-B/32: 1024D)
    # --------------------------------------------------------------------------
    for scale in ["10k", "20k", "30k"]:
        model_path = root / "models" / f"rating_regressor_multimodal_bumble_{scale}_pure.joblib"
        if not model_path.exists():
            continue
        obj = joblib.load(str(model_path))
        estimator = obj['estimator']
        n_in = getattr(estimator, 'n_features_in_', 512)
        if n_in == 512:
            preds = estimator.predict(X_face)
        else:
            X_mm = np.hstack([X_face, X_clip])
            preds = estimator.predict(X_mm)
        stats = evaluate_holdout_metrics(preds, y_val)
        stats['Model'] = f"Multimodal MLP ({scale} Pure)"
        results.append(stats)
        print(f"[Multimodal {scale}] Likes@25%: {stats['Likes@25%Cap']} | HitRate: {stats['HitRate@25%']} | ROC: {stats['ROC-AUC']} | PR-AUC: {stats['PR-AUC']} | Swipe@95%: {stats['SwipeRate@95%']}")

    # --------------------------------------------------------------------------
    # 3. Phase 1 Multi-Proxy (ViT-B/32, 20k and 30k)
    # --------------------------------------------------------------------------
    for scale in ["20k", "30k"]:
        if scale == "20k":
            ckpt_p = root / "models" / "subcenter_phase1" / "best_subcenter_phase1.pt"
            k_val, m_val = 8, 16
        else:
            ckpt_p = root / "models" / "subcenter_30k" / "best_subcenter_30k_pure.pt"
            k_val, m_val = 12, 20

        if ckpt_p.exists():
            ckpt = torch.load(str(ckpt_p), map_location=device)
            f_net = ClipGatedResidualFusion(clip_dim=512, face_dim=512, proj_dim=256).to(device)
            p_net = MultiProxyClassifier(in_dim=256, num_pos_proxies=k_val, num_neg_proxies=m_val).to(device)
            f_net.load_state_dict(ckpt['fusion_net'])
            p_net.load_state_dict(ckpt['proxy_model'])
            f_net.eval()
            p_net.eval()
            with torch.no_grad():
                t_clip = torch.tensor(X_clip, dtype=torch.float32, device=device)
                t_face = torch.tensor(X_face, dtype=torch.float32, device=device)
                emb = f_net(t_clip, t_face, training=False)
                preds = p_net(emb).cpu().numpy().flatten()
            stats = evaluate_holdout_metrics(preds, y_val)
            stats['Model'] = f"Phase 1 Multi-Proxy (ViT-B/32, {scale}, K={k_val})"
            results.append(stats)
            print(f"[Phase 1 Multi-Proxy {scale}] Likes@25%: {stats['Likes@25%Cap']} | HitRate: {stats['HitRate@25%']} | ROC: {stats['ROC-AUC']} | PR-AUC: {stats['PR-AUC']} | Swipe@95%: {stats['SwipeRate@95%']}")

    # --------------------------------------------------------------------------
    # 4. Phase 2 SigLIP-384 Multi-Proxy (10k, 20k, 30k)
    # --------------------------------------------------------------------------
    for scale in ["10k", "20k", "30k"]:
        if scale == "10k":
            sig_p = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_10k.pt"
        elif scale == "20k":
            sig_p = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_20k.pt"
        else:
            sig_p = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_30k_pure.pt"

        if sig_p.exists():
            ckpt = torch.load(str(sig_p), map_location=device)
            num_pos = ckpt['proxy_model']['pos_proxies'].shape[0]
            num_neg = ckpt['proxy_model']['neg_proxies'].shape[0]
            f_net = SiglipGatedResidualFusion(siglip_dim=1152, face_dim=512, proj_dim=256).to(device)
            p_net = MultiProxyClassifier(in_dim=256, num_pos_proxies=num_pos, num_neg_proxies=num_neg).to(device)
            f_net.load_state_dict(ckpt['fusion_net'])
            p_net.load_state_dict(ckpt['proxy_model'])
            f_net.eval()
            p_net.eval()
            with torch.no_grad():
                t_sig = torch.tensor(X_siglip, dtype=torch.float32, device=device)
                t_face = torch.tensor(X_face, dtype=torch.float32, device=device)
                emb = f_net(t_sig, t_face, training=False)
                preds = p_net(emb).cpu().numpy().flatten()
            stats = evaluate_holdout_metrics(preds, y_val)
            stats['Model'] = f"Phase 2 SigLIP-384 ({scale}, K={num_pos})"
            results.append(stats)
            print(f"[Phase 2 SigLIP {scale}] Likes@25%: {stats['Likes@25%Cap']} | HitRate: {stats['HitRate@25%']} | ROC: {stats['ROC-AUC']} | PR-AUC: {stats['PR-AUC']} | Swipe@95%: {stats['SwipeRate@95%']}")

    # Save DataFrame
    cols = ['Model', 'Likes@25%Cap', 'HitRate@25%', 'Likes@20%Cap', 'HitRate@20%', 'SwipeRate@90%', 'SwipeRate@95%', 'PR-AUC', 'ROC-AUC']
    df_res = pd.DataFrame(results)[cols]
    csv_out = results_dir / "benchmark_10k_vs_20k_vs_30k_on_33k_holdout.csv"
    df_res.to_csv(str(csv_out), index=False)

    print("\n" + "=" * 90)
    print("MASTER TOURNAMENT TABLE (30k-33.6k UNTRAINED HOLDOUT)")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print(f"\nSaved tournament results to {csv_out}")

if __name__ == "__main__":
    main()
