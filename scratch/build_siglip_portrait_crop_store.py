import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

def extract_portrait_crop_384(img_bgr: np.ndarray, bbox: list | None) -> tuple[np.ndarray, float]:
    """
    img_bgr: [H, W, 3] numpy array in BGR format
    bbox: [x1, y1, x2, y2] from InsightFace or None if no face detected
    Returns: (crop_384 [3, 384, 384] float32 normalized, face_valid_flag {1.0 or 0.0})
    """
    H, W, _ = img_bgr.shape
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) - 0.05 * h  # Shift up 5% to capture hair/crown
        side = int(round(1.8 * max(w, h)))
        face_valid = 1.0
    else:
        # Option A Fallback: Upper-center portrait zone
        cx = 0.5 * W
        cy = 0.35 * H
        side = int(round(0.65 * min(W, H)))
        face_valid = 0.0

    side = max(16, side)
    half = side // 2

    # Shift box inside image bounds if possible before resorting to border padding
    if side <= W:
        cx = np.clip(cx, half, W - (side - half))
    if side <= H:
        cy = np.clip(cy, half, H - (side - half))

    x_start, y_start = int(round(cx)) - half, int(round(cy)) - half
    x_end, y_end = x_start + side, y_start + side

    pad_left = max(0, -x_start)
    pad_top = max(0, -y_start)
    pad_right = max(0, x_end - W)
    pad_bottom = max(0, y_end - H)

    crop = img_bgr[max(0, y_start):min(H, y_end), max(0, x_start):min(W, x_end)]
    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    if crop.size == 0:
        crop = cv2.resize(img_bgr, (384, 384), interpolation=cv2.INTER_CUBIC)
    else:
        crop = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)

    # Convert BGR to RGB, normalize to [-1, 1] for SigLIP
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop_norm = (crop_rgb - 0.5) / 0.5
    crop_tensor = crop_norm.transpose(2, 0, 1)  # [3, 384, 384]

    return crop_tensor, face_valid


def main():
    parser = argparse.ArgumentParser(description="Extract Dual-Crop Portrait SigLIP-384 Embeddings")
    parser.add_argument("--store", required=True, help="Path to reference store .npz")
    parser.add_argument("--output", required=True, help="Path to output portrait siglip .npz")
    parser.add_argument("--batch-size", type=int, default=32, help="SigLIP inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    ref_store = load_store(args.store)
    paths = ref_store.paths
    if args.limit is not None:
        paths = paths[:args.limit]

    n_total = len(paths)
    print(f"Total paths to process: {n_total}")
    print("Initializing InsightFace detector on CUDA...")
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing SiglipVisionModel on {device}...")
    model_id = "google/siglip-so400m-patch14-384"
    siglip = SiglipVisionModel.from_pretrained(
        model_id,
        cache_dir=r"D:\BumbleClawHFCache",
        torch_dtype=torch.float16
    ).to(device)
    siglip.eval()

    all_embs = []
    all_face_valids = []
    t_start = time.time()

    # Process in batches
    for i in range(0, n_total, args.batch_size):
        batch_paths = paths[i : min(i + args.batch_size, n_total)]
        batch_crops = []
        batch_valids = []

        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                # Fallback dummy crop
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

        # SigLIP forward pass
        batch_np = np.stack(batch_crops, axis=0)
        with torch.no_grad():
            pixel_values = torch.tensor(batch_np, dtype=torch.float16, device=device)
            out = siglip(pixel_values=pixel_values).pooler_output
            out = out / out.norm(dim=-1, keepdim=True)
            batch_emb = out.cpu().to(torch.float32).numpy()

        all_embs.append(batch_emb)
        all_face_valids.extend(batch_valids)

        elapsed = time.time() - t_start
        done = min(i + args.batch_size, n_total)
        speed = done / elapsed
        eta = (n_total - done) / speed if speed > 0 else 0
        if done % (args.batch_size * 5) == 0 or done == n_total:
            print(f"[{done}/{n_total}] ({done/n_total*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {eta/60:.1f} mins | Valid Faces: {sum(all_face_valids):.0f}/{done}")

    all_embs = np.concatenate(all_embs, axis=0)
    all_face_valids = np.array(all_face_valids, dtype=np.float32).reshape(-1, 1)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_p,
        embeddings=all_embs,
        face_valid=all_face_valids,
        paths=np.array(paths),
        model_name=model_id,
        resolution=384,
        patch_size=14
    )
    print(f"\n[DONE] Saved {len(all_embs)} portrait SigLIP embeddings to {out_p}")
    print(f"Embeddings shape: {all_embs.shape} | Face valid shape: {all_face_valids.shape}")


if __name__ == "__main__":
    main()
