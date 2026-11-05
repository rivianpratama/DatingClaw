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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from face_similarity.store import load_store
from scratch.train_subcenter_pauc_regularized import predict_visual_logits, evaluate_metrics

def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")
    
    print("=" * 80)
    print("STEP 1 BENCHMARK & ENSEMBLE EVALUATION ON UNTOUCHED HOLDOUT 2 (30k-33.6k)")
    print("=" * 80)

    # 1. Load Untouched Holdout 2
    ref_holdout = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
    sig_holdout = np.load(str(root / "embeddings" / "siglip_store_bumble_30k_to_33k.npz"))
    clip_holdout = np.load(str(root / "embeddings" / "clip_store_bumble_30k_to_33k.npz"))

    y_test = (ref_holdout.ratings >= 50.0).astype(int)
    X_face = ref_holdout.embeddings
    X_sig = sig_holdout['embeddings']
    X_clip = clip_holdout['embeddings']
    X_mlp = np.hstack([X_face, X_clip])

    n_test = len(y_test)
    n_likes = int(y_test.sum())
    k_25 = int(np.round(0.25 * n_test))
    print(f"Holdout 2: {n_test} profiles | Likes: {n_likes} ({n_likes/n_test*100:.2f}%) | 25% Budget: {k_25} swipes\n")

    # 2. Score Individual Models
    ckpt_26k = str(root / "models" / "subcenter_phase2" / "best_subcenter_pauc_26k.pt")
    ckpt_30k = str(root / "models" / "subcenter_phase2" / "best_subcenter_pauc_30k_full.pt")

    preds_26k = predict_visual_logits(ckpt_26k, X_sig, X_face)
    preds_30k = predict_visual_logits(ckpt_30k, X_sig, X_face)

    # Multimodal MLP (30k Pure)
    mlp_obj = joblib.load(root / "models" / "rating_regressor_multimodal_bumble_30k_pure.joblib")
    mlp = mlp_obj['estimator'] if isinstance(mlp_obj, dict) else mlp_obj
    preds_mlp = mlp.predict(X_mlp)

    # Ridge Face (30k Pure)
    ridge_obj = joblib.load(root / "models" / "rating_regressor_bumble_30k_pure.joblib")
    ridge = ridge_obj['estimator'] if isinstance(ridge_obj, dict) else ridge_obj
    preds_ridge = ridge.predict(X_face)

    # Individual Metrics
    models = {
        "Ridge Face (30k Pure)": preds_ridge,
        "Multimodal MLP (30k Pure)": preds_mlp,
        "Regularized 26.5k pAUC Multi-Proxy": preds_26k,
        "Regularized 30k Full pAUC Multi-Proxy": preds_30k,
    }

    results = []
    for name, p in models.items():
        m = evaluate_metrics(p, y_test)
        results.append({"Model": name, **m})
        print(f"[{name}] Likes@25% = {m['Likes@25%Cap']} | PR-AUC = {m['PR-AUC']} | ROC = {m['ROC-AUC']}")

    print("\n" + "=" * 80)
    print("HYBRID ENSEMBLE BENCHMARKS (RANK-AVERAGING)")
    print("=" * 80)

    # Convert to percentiles (rank scores)
    r_mlp = rankdata(preds_mlp) / n_test
    r_26k = rankdata(preds_26k) / n_test
    r_30k = rankdata(preds_30k) / n_test
    r_ridge = rankdata(preds_ridge) / n_test

    ensembles = {
        "Ensemble 1: MLP (30k) + Reg 26.5k pAUC (50/50)": 0.50 * r_mlp + 0.50 * r_26k,
        "Ensemble 2: MLP (30k) + Reg 30k Full pAUC (50/50)": 0.50 * r_mlp + 0.50 * r_30k,
        "Ensemble 3: MLP (30k) + Reg 26.5k + Reg 30k (40/30/30)": 0.40 * r_mlp + 0.30 * r_26k + 0.30 * r_30k,
        "Ensemble 4: MLP (30k) + Reg 26.5k + Reg 30k + Ridge (35/30/25/10)": 0.35 * r_mlp + 0.30 * r_26k + 0.25 * r_30k + 0.10 * r_ridge,
        "Ensemble 5: MLP (60) + Reg 26.5k (40)": 0.60 * r_mlp + 0.40 * r_26k,
        "Ensemble 6: MLP (40) + Reg 26.5k (60)": 0.40 * r_mlp + 0.60 * r_26k,
    }

    for name, ens_p in ensembles.items():
        m = evaluate_metrics(ens_p, y_test)
        results.append({"Model": name, **m})
        print(f"[{name}]")
        print(f"  -> Likes@25%: {m['Likes@25%Cap']} | HitRate@25%: {m['HitRate@25%']} | PR-AUC: {m['PR-AUC']} | ROC: {m['ROC-AUC']}")

    # Save to CSV
    df_res = pd.DataFrame(results)
    out_csv = root / "results" / "benchmark_regularized_pauc_and_ensembles_holdout2.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nFull results saved to {out_csv}")

if __name__ == "__main__":
    main()
