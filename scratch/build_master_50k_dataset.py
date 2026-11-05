import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

root = Path(__file__).resolve().parent.parent
MASTER_OUT = Path(r"D:\master_50k_store.npz")
MISSING_NPZ = Path(r"D:\missing_slice_features.npz")


def main():
    if not MISSING_NPZ.exists():
        print(f"[ERROR] {MISSING_NPZ} does not exist.")
        return

    print("=" * 80, flush=True)
    print("STEP 2: ASSEMBLING MASTER 50,003 PROFILE TENSOR STORE ON D: (OPTIMIZED)", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()

    # 1. Load Ground Truth Labels
    df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv").drop_duplicates(subset=["filename"], keep="first").reset_index(drop=True)
    n_total = len(df_labels)
    fname_to_idx = {fn: i for i, fn in enumerate(df_labels["filename"])}
    print(f"Total Unique Labeled Profiles: {n_total}", flush=True)

    # Preallocate Destination Tensors
    X_arc      = np.zeros((n_total, 512), dtype=np.float32)
    X_geom     = np.zeros((n_total, 16), dtype=np.float32)
    X_sig_full = np.zeros((n_total, 1152), dtype=np.float32)
    X_sig_port = np.zeros((n_total, 1152), dtype=np.float32)
    X_dino     = np.zeros((n_total, 1536), dtype=np.float32)
    face_valid = np.zeros((n_total, 1), dtype=np.float32)
    bboxes     = np.full((n_total, 4), -1.0, dtype=np.float32)
    hard_veto  = np.zeros(n_total, dtype=bool)

    # 2. Map InsightFace Stores
    print("\nMapping ArcFace 512D Embeddings...", flush=True)
    ref_stores = [
        "embeddings/reference_store_bumble_30k_pure.npz",
        "embeddings/reference_store_bumble_30k_to_33k.npz",
        "embeddings/reference_store_bumble_33k_to_40k.npz",
        "embeddings/reference_store_val_1582.npz"
    ]
    for s_path in ref_stores:
        with np.load(str(root / s_path)) as st:
            embs = st["embeddings"][:]
            paths = st["paths"][:]
            for i, p in enumerate(paths):
                idx = fname_to_idx.get(Path(str(p)).name)
                if idx is not None:
                    X_arc[idx] = embs[i]
            print(f"  Loaded {len(paths)} from {s_path}", flush=True)

    # 3. Map SigLIP Full & Portrait Stores
    print("\nMapping SigLIP Full 1152D Embeddings...", flush=True)
    sig_full_stores = [
        "embeddings/siglip_store_bumble_30k_pure.npz",
        "embeddings/siglip_store_bumble_30k_to_33k.npz",
        "embeddings/siglip_store_bumble_33k_to_40k.npz",
        "embeddings/siglip_store_val_1582.npz"
    ]
    for s_path in sig_full_stores:
        with np.load(str(root / s_path)) as st:
            embs = st["embeddings"][:]
            paths = st["paths"][:]
            for i, p in enumerate(paths):
                idx = fname_to_idx.get(Path(str(p)).name)
                if idx is not None:
                    X_sig_full[idx] = embs[i]
            print(f"  Loaded {len(paths)} from {s_path}", flush=True)

    print("\nMapping SigLIP Portrait 1152D Embeddings...", flush=True)
    sig_port_stores = [
        "embeddings/siglip_portrait_store_bumble_30k_pure.npz",
        "embeddings/siglip_portrait_store_bumble_30k_to_33k.npz",
        "embeddings/siglip_portrait_store_bumble_33k_to_40k.npz",
        "embeddings/siglip_portrait_store_val_1582.npz"
    ]
    for s_path in sig_port_stores:
        with np.load(str(root / s_path)) as st:
            embs = st["embeddings"][:]
            fv = st["face_valid"][:]
            paths = st["paths"][:]
            for i, p in enumerate(paths):
                idx = fname_to_idx.get(Path(str(p)).name)
                if idx is not None:
                    X_sig_port[idx] = embs[i]
                    face_valid[idx] = fv[i]
            print(f"  Loaded {len(paths)} from {s_path}", flush=True)

    # 4. Map Geom Stores
    print("\nMapping 16D Geometry & Vetoes...", flush=True)
    geom_stores = [
        "embeddings/geom_store_train_40k.npz",
        "embeddings/geom_store_val_1582.npz"
    ]
    for s_path in geom_stores:
        with np.load(str(root / s_path)) as st:
            g16 = st["geom_16d"][:]
            bb  = st["bboxes"][:]
            hv  = st["hard_veto"][:]
            paths = st["paths"][:]
            for i, p in enumerate(paths):
                idx = fname_to_idx.get(Path(str(p)).name)
                if idx is not None:
                    X_geom[idx] = g16[i]
                    bboxes[idx] = bb[i]
                    hard_veto[idx] = hv[i]
            print(f"  Loaded {len(paths)} from {s_path}", flush=True)

    # 5. Map DINOv2 Stores
    print("\nMapping DINOv2-reg 1536D Embeddings...", flush=True)
    dino_stores = [
        r"D:\dinov2_portrait_store_train_40k.npz",
        str(root / "embeddings" / "dinov2_portrait_store_val_1582.npz")
    ]
    for s_path in dino_stores:
        with np.load(s_path) as st:
            embs = st["embeddings"][:]
            paths = st["paths"][:]
            for i, p in enumerate(paths):
                idx = fname_to_idx.get(Path(str(p)).name)
                if idx is not None:
                    X_dino[idx] = embs[i]
            print(f"  Loaded {len(paths)} from {s_path}", flush=True)

    # 6. Map Missing Slice
    print(f"\nMapping Missing Slice Features from {MISSING_NPZ}...", flush=True)
    with np.load(str(MISSING_NPZ)) as mis:
        mis_fnames = mis["filenames"][:]
        mis_arc    = mis["arc_embs"][:]
        mis_geom   = mis["geom_16d"][:]
        mis_fv     = mis["face_valids"][:]
        mis_bb     = mis["bboxes"][:]
        mis_hv     = mis["hard_vetos"][:]
        mis_sf     = mis["sig_full_embs"][:]
        mis_sp     = mis["sig_port_embs"][:]
        mis_dino   = mis["dino_embs"][:]

        count_mapped = 0
        for i, fn in enumerate(mis_fnames):
            idx = fname_to_idx.get(fn)
            if idx is not None:
                X_arc[idx]      = mis_arc[i]
                X_geom[idx]     = mis_geom[i]
                face_valid[idx] = mis_fv[i]
                bboxes[idx]     = mis_bb[i]
                hard_veto[idx]  = mis_hv[i]
                X_sig_full[idx] = mis_sf[i]
                X_sig_port[idx] = mis_sp[i]
                X_dino[idx]     = mis_dino[i]
                count_mapped += 1
        print(f"  Mapped {count_mapped} / {len(mis_fnames)} missing profiles!", flush=True)

    # Coverage verification
    populated_arc = (np.linalg.norm(X_arc, axis=-1) > 0.1).sum()
    populated_sf  = (np.linalg.norm(X_sig_full, axis=-1) > 0.1).sum()
    populated_dino = (np.linalg.norm(X_dino, axis=-1) > 0.1).sum()
    print(f"\nCoverage check across {n_total} rows:")
    print(f"  ArcFace populated:  {populated_arc} / {n_total} ({populated_arc/n_total*100:.2f}%)")
    print(f"  SigLIP populated:   {populated_sf} / {n_total} ({populated_sf/n_total*100:.2f}%)")
    print(f"  DINOv2 populated:   {populated_dino} / {n_total} ({populated_dino/n_total*100:.2f}%)")

    # 7. Labels & S_MultiX
    y_labels = (df_labels["rating"] >= 50.0).astype(int).values

    print("\nMerging MultimodalX_original baseline scores from scores.csv...", flush=True)
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

    S_multi_x = np.array([score_map.get(fn, 50.0) for fn in df_labels["filename"]], dtype=np.float32)
    print(f"  S_MultiX: mean={S_multi_x.mean():.2f}, median={np.median(S_multi_x):.2f}, min={S_multi_x.min():.2f}, max={S_multi_x.max():.2f}")

    # 8. Question 1: Class-Conditional Odds-Ratio Weights
    print("\nComputing Question 1 class-conditional importance weights...", flush=True)
    s_y = pd.Series(y_labels)
    p_deck = s_y.rolling(window=1500, min_periods=100, center=True).mean().bfill().ffill().values
    p_deck = np.clip(p_deck, 0.025, 0.140)

    p_target = 0.112
    odds_ratio = (p_deck / (1.0 - p_deck)) * ((1.0 - p_target) / p_target)
    r_odds = np.minimum(1.0, odds_ratio)

    hardness = expit((S_multi_x - 48.0) / 6.0)
    w_neg = r_odds * (0.20 + 0.80 * hardness)

    w_pos = np.zeros(n_total, dtype=np.float32)
    w_pos[:21121] = 0.90
    w_pos[21121:35822] = 1.00
    w_pos[35822:] = 1.15

    weights = np.where(y_labels == 1, w_pos, w_neg).astype(np.float32)

    logit_target = logit(0.0980)
    delta_mu = (logit(p_deck) - logit_target).astype(np.float32)

    print(f"  True Likes count: {y_labels.sum()} (base rate: {y_labels.mean()*100:.2f}%)")
    print(f"  Weights: pos_mean={weights[y_labels==1].mean():.3f}, neg_mean={weights[y_labels==0].mean():.3f}")
    print(f"  Era 3 pass mean weight: {weights[(y_labels==0) & (np.arange(n_total) >= 35822)].mean():.3f}")

    # 9. Save Consolidated Master Store
    print(f"\nSaving consolidated master tensors to {MASTER_OUT}...", flush=True)
    np.savez_compressed(
        MASTER_OUT,
        X_arc=X_arc,
        X_geom=X_geom,
        X_sig_full=X_sig_full,
        X_sig_port=X_sig_port,
        X_dino=X_dino,
        face_valid=face_valid,
        bboxes=bboxes,
        hard_veto=hard_veto,
        y=y_labels,
        s_multi_x=S_multi_x,
        weights=weights,
        delta_mu=delta_mu,
        paths=df_labels["path"].values,
        filenames=df_labels["filename"].values
    )
    sz_mb = MASTER_OUT.stat().st_size / 1024 / 1024
    print(f"[SUCCESS] Master store compiled in {time.time()-t_start:.1f}s -> {MASTER_OUT} ({sz_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
