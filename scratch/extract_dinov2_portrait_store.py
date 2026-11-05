import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache"
os.environ["TORCH_HOME"] = r"D:\hf_cache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel
from face_similarity.store import load_store

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_portrait_crop_378(img_bgr: np.ndarray, bbox: list | None) -> tuple[np.ndarray, float]:
    """
    Crops 1.8x head box and resizes to 378x378 (27x27 grid of 14x14 patches).
    Returns: (crop_378 [378, 378, 3] uint8, face_valid_flag)
    """
    H, W, _ = img_bgr.shape
    if bbox is not None and bbox[0] >= 0:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) - 0.05 * h  # Shift up 5% to capture crown
        side = int(round(1.8 * max(w, h)))
        face_valid = 1.0
    else:
        cx = 0.5 * W
        cy = 0.35 * H
        side = int(round(0.65 * min(W, H)))
        face_valid = 0.0

    side = max(16, side)
    half = side // 2

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
        crop = cv2.resize(img_bgr, (378, 378), interpolation=cv2.INTER_CUBIC)
    else:
        crop = cv2.resize(crop, (378, 378), interpolation=cv2.INTER_CUBIC)

    return crop, face_valid


def _load_and_crop(item):
    p, bb = item
    img = cv2.imread(p)
    if img is None:
        return np.zeros((378, 378, 3), dtype=np.uint8), 0.0
    bbox_list = bb.tolist() if (bb is not None and bb[0] >= 0) else None
    return extract_portrait_crop_378(img, bbox_list)


class DINOv2PortraitExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-base",
            cache_dir=r"D:\hf_cache",
            torch_dtype=torch.float16
        ).to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        B = len(crops_bgr)
        batch_np = np.stack(crops_bgr, axis=0)
        rgb = batch_np[..., ::-1].copy()
        x = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(0, 3, 1, 2).half() / 255.0
        x = (x - self.mean) / self.std

        hidden = self.model.embeddings(x)
        for layer in self.model.encoder.layer[:-1]:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out

        last_layer = self.model.encoder.layer[-1]
        normed_in  = last_layer.norm1(hidden)
        _, N, C    = normed_in.shape
        num_heads  = last_layer.attention.attention.num_attention_heads
        head_dim   = C // num_heads

        q_cls = last_layer.attention.attention.query(normed_in[:, :1, :])
        k_all = last_layer.attention.attention.key(normed_in)

        q_cls = q_cls.view(B, 1, num_heads, head_dim).transpose(1, 2)
        k_all = k_all.view(B, N, num_heads, head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q_cls, k_all.transpose(-1, -2)) / (head_dim ** 0.5)
        attn_probs  = F.softmax(attn_logits.float(), dim=-1).squeeze(2)

        # Slice spatial patches (5:734), average over 12 heads, Bayes L1 re-normalization
        a_spatial = attn_probs[:, :, 5:].mean(dim=1)
        w_patch   = a_spatial / (a_spatial.sum(dim=-1, keepdim=True) + 1e-8)

        last_out    = last_layer(hidden)
        last_hidden = last_out[0] if isinstance(last_out, tuple) else last_out
        out_hidden  = self.model.layernorm(last_hidden).float()
        
        x_cls      = F.normalize(out_hidden[:, 0, :], p=2, dim=-1)
        p_patches  = out_hidden[:, 5:, :]
        p_fg       = torch.bmm(w_patch.unsqueeze(1), p_patches).squeeze(1)
        p_fg       = F.normalize(p_fg, p=2, dim=-1)
        
        x_dino     = F.normalize(torch.cat([x_cls, p_fg], dim=-1), p=2, dim=-1)
        return x_dino.cpu().numpy().astype(np.float32)


def extract_dinov2_store(extractor, paths, bboxes, out_path, batch_size=48, desc="Extracting"):
    n_total = len(paths)
    print(f"\n[{desc}] Processing {n_total} images -> {out_path}...", flush=True)
    
    all_embs = []
    all_face_valids = []
    t0 = time.time()

    executor = ThreadPoolExecutor(max_workers=8)

    for i in range(0, n_total, batch_size):
        end_idx = min(i + batch_size, n_total)
        batch_paths = paths[i:end_idx]
        batch_bboxes = bboxes[i:end_idx] if bboxes is not None else [None] * len(batch_paths)
        
        items = list(zip(batch_paths, batch_bboxes))
        results = list(executor.map(_load_and_crop, items))
        
        batch_crops = [r[0] for r in results]
        batch_valids = [r[1] for r in results]

        batch_embs = extractor.extract_batch(batch_crops)
        all_embs.append(batch_embs)
        all_face_valids.extend(batch_valids)

        if end_idx % 480 == 0 or end_idx == n_total:
            dt = time.time() - t0
            speed = end_idx / dt
            rem = (n_total - end_idx) / speed if speed > 0 else 0
            print(f"  {end_idx}/{n_total} ({end_idx/n_total*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m | Face Valids: {sum(all_face_valids):.0f}/{end_idx}", flush=True)

    executor.shutdown()
    all_embs = np.concatenate(all_embs, axis=0)
    all_face_valids = np.array(all_face_valids, dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(
        out_path,
        embeddings=all_embs,
        face_valid=all_face_valids,
        paths=np.array(paths),
        model_name="facebook/dinov2-with-registers-base"
    )
    print(f"[OK] Saved DINOv2 store ({all_embs.shape}) to: {out_path}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("PHASE 1: 1536D DINOv2-WITH-REGISTERS PORTRAIT CROP FEATURE EXTRACTION", flush=True)
    print("=" * 80, flush=True)

    extractor = DINOv2PortraitExtractor(device="cuda")

    # 1. Validation Store
    val_out_path = root / "embeddings" / "dinov2_portrait_store_val_1582.npz"
    if not val_out_path.exists():
        r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
        geom_val = np.load(str(root / "embeddings" / "geom_store_val_1582.npz")) if (root / "embeddings" / "geom_store_val_1582.npz").exists() else None
        bboxes_val = geom_val["bboxes"] if geom_val is not None else None
        extract_dinov2_store(extractor, r_val.paths, bboxes_val, val_out_path, batch_size=48, desc="Validation 1582")
    else:
        print(f"[EXISTS] {val_out_path}", flush=True)

    # 2. Training Store (saved to D: drive)
    train_out_path = Path(r"D:\dinov2_portrait_store_train_40k.npz")
    if not train_out_path.exists():
        r_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
        r_h2  = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
        r_new = load_store(str(root / "embeddings" / "reference_store_bumble_33k_to_40k.npz"))
        train_paths = np.concatenate([r_30k.paths, r_h2.paths, r_new.paths])
        
        geom_train = np.load(str(root / "embeddings" / "geom_store_train_40k.npz")) if (root / "embeddings" / "geom_store_train_40k.npz").exists() else None
        bboxes_train = geom_train["bboxes"] if geom_train is not None else None
        extract_dinov2_store(extractor, train_paths, bboxes_train, train_out_path, batch_size=48, desc="Full 40k Training")
    else:
        print(f"[EXISTS] {train_out_path}", flush=True)

    print("\n[ALL DINOv2 EXTRACTIONS COMPLETE!]", flush=True)

if __name__ == "__main__":
    main()
