import sys
from pathlib import Path
root = Path(r'C:\Users\Rivian\Documents\GitHub\BumbleClaw')
sys.path.insert(0, str(root))

import numpy as np
import torch
from face_similarity.store import load_store
from scratch.train_subcenter_pauc_regularized import predict_visual_logits, evaluate_metrics
from scipy.stats import rankdata
import joblib

root = Path(r'C:\Users\Rivian\Documents\GitHub\BumbleClaw')
ref_h1 = load_store(str(root / 'embeddings' / 'reference_store_bumble_20k_to_22k.npz'))
sig_h1 = np.load(str(root / 'embeddings' / 'siglip_store_bumble_20k_to_22k.npz'))
clip_h1 = np.load(str(root / 'embeddings' / 'clip_store_bumble_20k_to_22k.npz'))

y_h1 = (ref_h1.ratings >= 50.0).astype(int)
X_face = ref_h1.embeddings
X_sig = sig_h1['embeddings']
X_clip = clip_h1['embeddings']

print(f"Holdout 1: {len(y_h1)} profiles | Likes: {int(y_h1.sum())} | 25% budget: {int(np.round(0.25*len(y_h1)))}")

# Score 10k Regularized pAUC
ckpt_10k = str(root / 'models' / 'subcenter_phase2' / 'best_subcenter_pauc_10k.pt')
p_10k = predict_visual_logits(ckpt_10k, X_sig, X_face)
m_10k = evaluate_metrics(p_10k, y_h1)
print(f"[10k Reg pAUC] Likes@25%: {m_10k['Likes@25%Cap']} | PR-AUC: {m_10k['PR-AUC']} | ROC: {m_10k['ROC-AUC']}")

# Score 10k MLP
mlp10_obj = joblib.load(root / 'models' / 'rating_regressor_multimodal_bumble_10k_pure.joblib')
mlp10 = mlp10_obj['estimator'] if isinstance(mlp10_obj, dict) else mlp10_obj
p_mlp10 = mlp10.predict(X_face)
m_mlp10 = evaluate_metrics(p_mlp10, y_h1)
print(f"[10k MLP]      Likes@25%: {m_mlp10['Likes@25%Cap']} | PR-AUC: {m_mlp10['PR-AUC']} | ROC: {m_mlp10['ROC-AUC']}")

# 10k Ensemble
r_10k = rankdata(p_10k) / len(y_h1)
r_mlp10 = rankdata(p_mlp10) / len(y_h1)
ens_10k = 0.5 * r_10k + 0.5 * r_mlp10
m_ens10 = evaluate_metrics(ens_10k, y_h1)
print(f"[10k Ensemble] Likes@25%: {m_ens10['Likes@25%Cap']} | PR-AUC: {m_ens10['PR-AUC']} | ROC: {m_ens10['ROC-AUC']}")
