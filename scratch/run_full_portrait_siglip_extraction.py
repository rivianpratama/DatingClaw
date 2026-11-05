import os
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from transformers import SiglipVisionModel
from insightface.app import FaceAnalysis
from face_similarity.store import load_store
from scratch.build_siglip_portrait_crop_store import extract_portrait_crop_384

def extract_store(app, siglip, store_path, out_path, batch_size=32):
    ref_store = load_store(store_path)
    paths = ref_store.paths
    n_total = len(paths)
    print(f"\n=======================================================")
    print(f"Extracting {n_total} portrait crops from: {store_path}")
    print(f"Target: {out_path}")
    print(f"=======================================================")

    all_embs = []
    all_face_valids = []
    t_start = time.time()
    device = torch.device("cuda")

    for i in range(0, n_total, batch_size):
        batch_paths = paths[i : min(i + batch_size, n_total)]
        batch_crops = []
        batch_valids = []

        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                dummy = np.zeros((3, 384, 384), dtype=np.float32)
                batch_crops.append(dummy)
                batch_valids.append(0.0)
                continue

            try:
                bboxes, _ = app.det_model.detect(img, max_num=1)
                if bboxes is not None and len(bboxes) > 0:
                    bbox = bboxes[0][:4].tolist()
                else:
                    bbox = None
            except Exception:
                bbox = None

            crop_arr, f_valid = extract_portrait_crop_384(img, bbox)
            batch_crops.append(crop_arr)
            batch_valids.append(f_valid)

        batch_np = np.stack(batch_crops, axis=0)
        with torch.no_grad():
            pixel_values = torch.tensor(batch_np, dtype=torch.float16, device=device)
            out = siglip(pixel_values=pixel_values).pooler_output
            out = out / out.norm(dim=-1, keepdim=True)
            batch_emb = out.cpu().to(torch.float32).numpy()

        all_embs.append(batch_emb)
        all_face_valids.extend(batch_valids)

        elapsed = time.time() - t_start
        done = min(i + batch_size, n_total)
        speed = done / elapsed
        eta = (n_total - done) / speed if speed > 0 else 0
        if done % (batch_size * 10) == 0 or done == n_total:
            print(f"[{done:5d}/{n_total:5d}] ({done/n_total*100:5.1f}%) | Speed: {speed:4.1f} img/s | ETA: {eta/60:4.1f} min | Valid Faces: {sum(all_face_valids):.0f}/{done}")

    all_embs = np.concatenate(all_embs, axis=0)
    all_face_valids = np.array(all_face_valids, dtype=np.float32).reshape(-1, 1)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_p,
        embeddings=all_embs,
        face_valid=all_face_valids,
        paths=np.array(paths),
        model_name="google/siglip-so400m-patch14-384",
        resolution=384,
        patch_size=14
    )
    print(f"[DONE] Saved {len(all_embs)} portrait SigLIP embeddings to {out_p}")


def main():
    root = Path(r"C:\Users\Rivian\Documents\GitHub\BumbleClaw")

    print("Initializing InsightFace detector on CUDA...")
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    device = torch.device("cuda")
    print(f"Initializing SiglipVisionModel on {device}...")
    model_id = "google/siglip-so400m-patch14-384"
    siglip = SiglipVisionModel.from_pretrained(
        model_id,
        cache_dir=r"D:\BumbleClawHFCache",
        torch_dtype=torch.float16
    ).to(device)
    siglip.eval()

    # 1. First extract Holdout 2 (3,481 profiles)
    h2_in = root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"
    h2_out = root / "embeddings" / "siglip_portrait_store_bumble_30k_to_33k.npz"
    extract_store(app, siglip, str(h2_in), str(h2_out), batch_size=32)

    # 2. Next extract 30k pure (29,582 profiles)
    p30_in = root / "embeddings" / "reference_store_bumble_30k_pure.npz"
    p30_out = root / "embeddings" / "siglip_portrait_store_bumble_30k_pure.npz"
    extract_store(app, siglip, str(p30_in), str(p30_out), batch_size=32)

    print("\nALL DUAL-CROP PORTRAIT SIGLIP EXTRACTIONS COMPLETE!")


if __name__ == "__main__":
    main()
