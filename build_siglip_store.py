import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import SiglipVisionModel

from face_similarity.store import load_store

def letterbox_image(img: Image.Image, target_size=384) -> Image.Image:
    """Aspect-ratio-preserving resize with zero padding to target_size x target_size."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = img.resize((nw, nh), Image.Resampling.BICUBIC)

    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - nw) // 2
    paste_y = (target_size - nh) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img

def load_and_preprocess_single(path_str: str, target_size=384):
    try:
        img = Image.open(path_str).convert("RGB")
        lb = letterbox_image(img, target_size=target_size)
        arr = np.array(lb, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5 # Standard SigLIP normalization
        return path_str, arr.transpose(2, 0, 1) # (3, 384, 384)
    except Exception as e:
        print(f"SKIP {path_str}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Extract SigLIP-So400m-384 high-resolution embeddings.")
    parser.add_argument("--store", required=True, help="Face reference store .npz to align paths with")
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of thread workers for image preprocessing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SiglipVisionModel (384x384) on {device}...")
    model_id = "google/siglip-so400m-patch14-384"
    model = SiglipVisionModel.from_pretrained(
        model_id,
        cache_dir=r"D:\BumbleClawHFCache",
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    face_store = load_store(args.store)
    paths = [str(p) for p in face_store.paths]
    print(f"Found {len(paths)} paths in store: {args.store}")

    embeddings = []
    kept_paths = []

    progress = tqdm(range(0, len(paths), args.batch_size), desc="Extracting SigLIP embeddings")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in progress:
            batch_paths = paths[start : start + args.batch_size]
            results = list(pool.map(lambda p: load_and_preprocess_single(p, 384), batch_paths))

            valid_paths = [r[0] for r in results if r[0] is not None]
            valid_tensors = [r[1] for r in results if r[1] is not None]

            if not valid_tensors:
                continue

            batch_np = np.stack(valid_tensors, axis=0)
            pixel_values = torch.tensor(batch_np, dtype=torch.float16, device=device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                feats = outputs.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                batch_emb = feats.cpu().to(torch.float32).numpy()

            embeddings.append(batch_emb)
            kept_paths.extend(valid_paths)

    all_embeddings = np.concatenate(embeddings, axis=0)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=all_embeddings,
        paths=np.array(kept_paths),
        model_name=model_id,
        device=str(device),
        resolution=384,
        patch_size=14
    )
    print(f"\nSaved {len(all_embeddings)} SigLIP embeddings to {out_path} (Shape: {all_embeddings.shape})")

if __name__ == "__main__":
    main()
    main()
