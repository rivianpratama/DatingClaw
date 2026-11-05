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
from scipy.stats import rankdata

from face_similarity.store import load_store
from scratch.train_subcenter_pauc_regularized import (
    train_regularized_subcenter,
    predict_visual_logits,
    evaluate_metrics,
    extract_context_features
)

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"

    print("=" * 80)
    print("EMPIRICAL SCALING EXPERIMENT: REGULARIZED pAUC ON 10K, 20K, AND 30K DATASETS")
    print("=" * 80)

    # 1. Load Stores
    ref_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
    sig_30k = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_pure.npz"))

    y_all = (ref_30k.ratings >= 50.0).astype(int)
    X_face_all = ref_30k.embeddings
    X_sig_all = sig_30k['embeddings']
    paths_all = ref_30k.paths

    print(f"Total reference samples available: {len(y_all)}")
    print("Extracting context features...")
    C_all = extract_context_features(paths_all, csv_path)

    # ==========================================================================
    # Train 10k Regularized pAUC Model
    # ==========================================================================
    ckpt_10k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_10k.pt"
    if not ckpt_10k_path.exists():
        print("\n" + "=" * 60)
        print("TRAINING 10K REGULARIZED pAUC SUB-CENTER MODEL (Profiles 0 to 9,990)")
        print("=" * 60)
        cut_10k_tr = 8500
        cut_10k_end = 9990
        train_regularized_subcenter(
            X_sig_all[:cut_10k_tr], X_face_all[:cut_10k_tr], y_all[:cut_10k_tr], C_all[:cut_10k_tr],
            X_sig_all[cut_10k_tr:cut_10k_end], X_face_all[cut_10k_tr:cut_10k_end], y_all[cut_10k_tr:cut_10k_end], C_all[cut_10k_tr:cut_10k_end],
            K=12, M=20, epochs=10, batch_size=512,
            save_path=str(ckpt_10k_path)
        )
    else:
        print(f"[FOUND] {ckpt_10k_path}")

    # ==========================================================================
    # Train 20k Regularized pAUC Model
    # ==========================================================================
    ckpt_20k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_20k.pt"
    if not ckpt_20k_path.exists():
        print("\n" + "=" * 60)
        print("TRAINING 20K REGULARIZED pAUC SUB-CENTER MODEL (Profiles 0 to 19,720)")
        print("=" * 60)
        cut_20k_tr = 17500
        cut_20k_end = 19720
        train_regularized_subcenter(
            X_sig_all[:cut_20k_tr], X_face_all[:cut_20k_tr], y_all[:cut_20k_tr], C_all[:cut_20k_tr],
            X_sig_all[cut_20k_tr:cut_20k_end], X_face_all[cut_20k_tr:cut_20k_end], y_all[cut_20k_tr:cut_20k_end], C_all[cut_20k_tr:cut_20k_end],
            K=12, M=20, epochs=10, batch_size=512,
            save_path=str(ckpt_20k_path)
        )
    else:
        print(f"[FOUND] {ckpt_20k_path}")

    # ==========================================================================
    # Benchmark on Untouched Holdout 2 (30k to 33.6k: 3,481 profiles, 443 likes)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TOURNAMENT BENCHMARK ON UNTOUCHED HOLDOUT 2 (30k to 33.6k)")
    print("=" * 80)

    ref_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    sig_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))
    clip_holdout = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))

    y_test = (ref_holdout.ratings >= 50.0).astype(int)
    X_face_test = ref_holdout.embeddings
    X_sig_test = sig_holdout['embeddings']
    X_clip_test = clip_holdout['embeddings']
    X_mlp_test = np.hstack([X_face_test, X_clip_test])
    n_test = len(y_test)

    # Predictions
    ckpt_30k_path = root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt"

    preds_pauc_10k = predict_visual_logits(str(ckpt_10k_path), X_sig_test, X_face_test)
    preds_pauc_20k = predict_visual_logits(str(ckpt_20k_path), X_sig_test, X_face_test)
    preds_pauc_30k = predict_visual_logits(str(ckpt_30k_path), X_sig_test, X_face_test)

    # Load MLPs
    mlp_10k_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_10k_pure.joblib")
    mlp_10k = mlp_10k_obj['estimator'] if isinstance(mlp_10k_obj, dict) else mlp_10k_obj
    n_in_10 = mlp_10k.n_features_in_ if hasattr(mlp_10k, 'n_features_in_') else 512
    preds_mlp_10k = mlp_10k.predict(X_face_test if n_in_10 == 512 else X_mlp_test)

    mlp_20k_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_20k_pure.joblib")
    mlp_20k = mlp_20k_obj['estimator'] if isinstance(mlp_20k_obj, dict) else mlp_20k_obj
    n_in_20 = mlp_20k.n_features_in_ if hasattr(mlp_20k, 'n_features_in_') else 1024
    preds_mlp_20k = mlp_20k.predict(X_face_test if n_in_20 == 512 else X_mlp_test)

    mlp_30k_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    mlp_30k = mlp_30k_obj['estimator'] if isinstance(mlp_30k_obj, dict) else mlp_30k_obj
    n_in_30 = mlp_30k.n_features_in_ if hasattr(mlp_30k, 'n_features_in_') else 1024
    preds_mlp_30k = mlp_30k.predict(X_face_test if n_in_30 == 512 else X_mlp_test)

    # Load Earlier Unregularized Subcenter Models for direct comparison
    old_10k_path = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_10k.pt"
    old_20k_path = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_20k.pt"
    old_30k_path = root / "models" / "subcenter_phase2" / "best_subcenter_siglip_30k_pure.pt"

    from scratch.train_subcenter_siglip_phase2 import SiglipGatedResidualFusion, MultiProxyClassifier
    def predict_old(path):
        ckpt = torch.load(path, map_location=torch.device('cuda'), weights_only=False)
        num_pos = ckpt['proxy_model']['pos_proxies'].shape[0]
        num_neg = ckpt['proxy_model']['neg_proxies'].shape[0]
        fn = SiglipGatedResidualFusion(siglip_dim=1152, face_dim=512, proj_dim=256).cuda()
        fn.load_state_dict(ckpt['fusion_net'])
        fn.eval()
        head = MultiProxyClassifier(in_dim=256, num_pos_proxies=num_pos, num_neg_proxies=num_neg).cuda()
        head.load_state_dict(ckpt['proxy_model'])
        head.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_sig_test, dtype=torch.float32), torch.tensor(X_face_test, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        preds = []
        with torch.no_grad():
            for b_s, b_f in loader:
                b_s, b_f = b_s.cuda(), b_f.cuda()
                emb = fn(b_s, b_f, training=False)
                z = head(emb)
                preds.append(z.cpu().numpy().flatten())
        return np.concatenate(preds, axis=0)

    preds_old_10k = predict_old(old_10k_path)
    preds_old_20k = predict_old(old_20k_path)
    preds_old_30k = predict_old(old_30k_path)

    # Evaluate all
    models_to_eval = {
        "Multimodal MLP (10k Pure)": preds_mlp_10k,
        "Multimodal MLP (20k Pure)": preds_mlp_20k,
        "Multimodal MLP (30k Pure)": preds_mlp_30k,
        "Old Subcenter SigLIP (10k)": preds_old_10k,
        "Old Subcenter SigLIP (20k)": preds_old_20k,
        "Old Subcenter SigLIP (30k)": preds_old_30k,
        "NEW Regularized pAUC Subcenter (10k)": preds_pauc_10k,
        "NEW Regularized pAUC Subcenter (20k)": preds_pauc_20k,
        "NEW Regularized pAUC Subcenter (30k)": preds_pauc_30k,
    }

    # Rank data for ensembles
    r_mlp_10k = rankdata(preds_mlp_10k) / n_test
    r_pauc_10k = rankdata(preds_pauc_10k) / n_test

    r_mlp_20k = rankdata(preds_mlp_20k) / n_test
    r_pauc_20k = rankdata(preds_pauc_20k) / n_test

    r_mlp_30k = rankdata(preds_mlp_30k) / n_test
    r_pauc_30k = rankdata(preds_pauc_30k) / n_test

    ensembles = {
        "Hybrid Ensemble (10k Pure: 50% MLP + 50% pAUC)": 0.50 * r_mlp_10k + 0.50 * r_pauc_10k,
        "Hybrid Ensemble (20k Pure: 50% MLP + 50% pAUC)": 0.50 * r_mlp_20k + 0.50 * r_pauc_20k,
        "Hybrid Ensemble (30k Pure: 50% MLP + 50% pAUC)": 0.50 * r_mlp_30k + 0.50 * r_pauc_30k,
    }

    all_eval = {**models_to_eval, **ensembles}
    results = []
    print("\n--- Leaderboard on Untouched Holdout 2 (3,481 profiles, 443 likes) ---")
    for name, p in all_eval.items():
        m = evaluate_metrics(p, y_test)
        results.append({"Model": name, **m})
        print(f"{name:<50} | Likes@25%: {m['Likes@25%Cap']:<18} | PR-AUC: {m['PR-AUC']:.4f} | ROC: {m['ROC-AUC']:.4f}")

    df_res = pd.DataFrame(results)
    out_csv = root / "results" / "scaling_benchmark_10k_vs_20k_vs_30k_regularized_pauc.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved scaling benchmark table to {out_csv}")


if __name__ == "__main__":
    main()
