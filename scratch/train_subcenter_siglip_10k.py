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
import torch

from face_similarity.store import load_store
from scratch.train_subcenter_siglip_phase2 import train_siglip_subcenter
from scratch.train_subcenter_multiproxy_phase1 import extract_context_features

def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models" / "subcenter_phase2"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = models_dir / "best_subcenter_siglip_10k.pt"

    if out_ckpt.exists():
        print(f"[EXISTS] {out_ckpt}")
        return

    print("Training Phase 2 SigLIP-384 on 10k Pure dataset...")
    s_20k = load_store(str(root / "embeddings" / "reference_store_bumble_20k_pure.npz"))
    sig_20k = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_pure.npz"))

    # First 9,990 samples correspond to 10k pure
    n_10k = 9990
    X_face_10k = s_20k.embeddings[:n_10k]
    X_sig_10k = sig_20k['embeddings'][:n_10k]
    y_10k = (s_20k.ratings[:n_10k] >= 50.0).astype(np.float32)
    paths_10k = s_20k.paths[:n_10k]

    csv_path = r"D:\BumbleLog\binary_swipe_labels.csv"
    C_10k = extract_context_features(paths_10k, csv_path)

    # Validation on 20k-22k holdout
    s_val = load_store(str(root / "embeddings" / "reference_store_bumble_20k_to_22k.npz"))
    sig_val = np.load(str(root / "embeddings" / "siglip_store_bumble_20k_to_22k.npz"))
    y_val = (s_val.ratings >= 50.0).astype(int)

    stats_10k, _, f_10k, p_10k, c_10k, tau_10k = train_siglip_subcenter(
        X_sig_10k, X_face_10k, y_10k, C_10k,
        sig_val['embeddings'], s_val.embeddings, y_val,
        K=8, M=16, epochs=12, batch_size=512, lr=3e-4, exp_name="SigLIP-384 10k Pure (K=8)"
    )

    torch.save({
        'fusion_net': f_10k.state_dict(),
        'proxy_model': p_10k.state_dict(),
        'context_net': c_10k.state_dict(),
        'tau_hat': tau_10k,
        'stats': stats_10k,
        'train_samples': n_10k
    }, str(out_ckpt))
    print(f"Saved 10k SigLIP checkpoint to {out_ckpt}")

if __name__ == "__main__":
    main()
