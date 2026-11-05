import os
import sys
import subprocess
from pathlib import Path

# Lock caches to D: drive
os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image

root = Path(__file__).resolve().parent.parent
python_clip = r"D:\BumbleClawClipVenv\Scripts\python.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_csv = Path(r"D:\BumbleLog\binary_swipe_labels_val_1582.csv")
ref_store_path = root / "embeddings" / "reference_store_val_1582.npz"
clip_store_path = root / "embeddings" / "clip_store_val_1582.npz"
siglip_store_path = root / "embeddings" / "siglip_store_val_1582.npz"
portrait_store_path = root / "embeddings" / "siglip_portrait_store_val_1582.npz"

def main():
    print("=" * 80)
    print("PREPARING 1,582 VALIDATION HOLDOUT DATASET & EXTRACTING EMBEDDINGS")
    print("=" * 80)

    # 1. Create CSV
    df = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv")
    df_val = df.iloc[40000:].copy()
    df_val.to_csv(val_csv, index=False)
    n_likes = (df_val['rating'] >= 50.0).sum()
    n_passes = (df_val['rating'] < 50.0).sum()
    print(f"Validation CSV: {val_csv} | Rows: {len(df_val)} | Likes: {n_likes} ({n_likes/len(df_val)*100:.2f}%) | Passes: {n_passes}")

    # 2. Extract ArcFace Reference Store
    if not ref_store_path.exists():
        print(f"\n[1/4] Extracting InsightFace embeddings -> {ref_store_path}...")
        cmd_ref = [
            python_clip, "build_references.py",
            "--labels", str(val_csv),
            "--output", str(ref_store_path),
            "--provider", "cuda"
        ]
        subprocess.run(cmd_ref, check=True)
        print("[OK] ArcFace store saved.")
    else:
        print(f"[EXISTS] {ref_store_path}")

    # 3. Extract CLIP Store
    if not clip_store_path.exists():
        print(f"\n[2/4] Extracting CLIP ViT-B/32 -> {clip_store_path}...")
        cmd_clip = [
            python_clip, "build_clip_store.py",
            "--labels", str(val_csv),
            "--store", str(ref_store_path),
            "--output", str(clip_store_path),
            "--batch-size", "64",
            "--provider", "cuda"
        ]
        subprocess.run(cmd_clip, check=True)
        print("[OK] CLIP store saved.")
    else:
        print(f"[EXISTS] {clip_store_path}")

    # 4. Extract SigLIP Full Store
    if not siglip_store_path.exists():
        print(f"\n[3/4] Extracting SigLIP-384 Full Scene -> {siglip_store_path}...")
        cmd_siglip = [
            python_clip, "build_siglip_store.py",
            "--store", str(ref_store_path),
            "--output", str(siglip_store_path),
            "--batch-size", "32",
            "--workers", "8"
        ]
        subprocess.run(cmd_siglip, check=True)
        print("[OK] SigLIP Full store saved.")
    else:
        print(f"[EXISTS] {siglip_store_path}")

    # 5. Extract SigLIP Portrait Store
    if not portrait_store_path.exists():
        print(f"\n[4/4] Extracting SigLIP-384 Portrait Crops -> {portrait_store_path}...")
        from transformers import SiglipVisionModel
        from insightface.app import FaceAnalysis
        from scratch.build_siglip_portrait_crop_store import extract_portrait_crop_384
        from face_similarity.store import load_store

        ref_store = load_store(str(ref_store_path))
        paths = ref_store.paths

        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        siglip = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            cache_dir=r"D:\BumbleClawHFCache",
            torch_dtype=torch.float16
        ).to(device)
        siglip.eval()

        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

        portrait_embeddings = []
        face_valid_flags = []
        batch_tensors = []

        print(f"Processing {len(paths)} portrait crops on GPU...")
        for i, raw_p in enumerate(paths):
            p = Path(raw_p)
            if not p.exists():
                p_alt = Path(r"D:\BumbleLog") / p.name
                if p_alt.exists():
                    p = p_alt
                else:
                    p_alt2 = Path(r"D:\BumbleLog\cropped") / (p.stem + ".webp")
                    if p_alt2.exists():
                        p = p_alt2

            img_bgr = cv2.imread(str(p))
            if img_bgr is not None:
                crop_rgb, has_face = extract_portrait_crop_384(img_bgr, app, expand_ratio=1.8)
            else:
                crop_rgb = np.zeros((384, 384, 3), dtype=np.uint8)
                has_face = False

            face_valid_flags.append(1.0 if has_face else 0.0)

            t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            batch_tensors.append(t)

            if len(batch_tensors) == 32 or i == len(paths) - 1:
                b_t = torch.stack(batch_tensors).to(device)
                b_t = (b_t - mean) / std
                with torch.no_grad():
                    out = siglip(pixel_values=b_t.half())
                    emb = out.pooler_output
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().float().numpy()
                portrait_embeddings.append(emb)
                batch_tensors = []

        port_all = np.concatenate(portrait_embeddings, axis=0)
        fv_all = np.array(face_valid_flags, dtype=np.float32)

        np.savez_compressed(
            str(portrait_store_path),
            embeddings=port_all,
            face_valid=fv_all,
            paths=np.array(paths)
        )
        print(f"[OK] Saved portrait store to {portrait_store_path}: shape {port_all.shape}")
    else:
        print(f"[EXISTS] {portrait_store_path}")

    print("\n[SUCCESS] All 4 embedding stores for the 1,582 validation set are ready!")

if __name__ == "__main__":
    main()
