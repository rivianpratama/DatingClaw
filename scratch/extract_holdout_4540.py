import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel, SiglipVisionModel
from insightface.app import FaceAnalysis
from insightface.utils import face_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NPZ = Path(r"D:\holdout_4540_store.npz")
MASTER_STORE = Path(r"D:\master_50k_dinov3_store.npz")
DINOV3_LOCAL_PATH = r"C:\Users\Rivian\dinov3-vitb16-pretrain-lvd1689m"


def extract_16d_geometry_and_veto(face, img_bgr):
    img_h, img_w = img_bgr.shape[:2]
    pts3d = face.landmark_3d_68.astype(np.float32)
    pts2d = pts3d[:, :2]
    eps = 1e-6

    p_left_eye_3d = pts3d[36:42].mean(axis=0)
    p_right_eye_3d = pts3d[42:48].mean(axis=0)
    d_iod = float(np.linalg.norm(p_left_eye_3d - p_right_eye_3d)) + eps

    pitch_deg = float(face.pose[0])
    yaw_deg = float(face.pose[1])
    roll_deg = float(face.pose[2])

    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)
    roll_rad = np.deg2rad(roll_deg)

    f1_sin_yaw = np.sin(yaw_rad)
    f2_sin_pitch = np.sin(pitch_rad)
    f3_sin_roll = np.sin(roll_rad)
    f4_cos_yaw = np.cos(yaw_rad)

    y_stomion = 0.5 * (pts2d[62, 1] + pts2d[66, 1])
    y_corners = 0.5 * (pts2d[48, 1] + pts2d[54, 1])
    f5_smile_curve = (y_stomion - y_corners) / d_iod

    mouth_width = float(np.linalg.norm(pts2d[48] - pts2d[54])) + eps
    f6_smile_width = mouth_width / d_iod

    mouth_open = float(np.linalg.norm(pts2d[62] - pts2d[66]))
    f7_mar = mouth_open / mouth_width

    ear_L_num = np.linalg.norm(pts2d[37] - pts2d[41]) + np.linalg.norm(pts2d[38] - pts2d[40])
    ear_L_den = 2.0 * np.linalg.norm(pts2d[36] - pts2d[39]) + eps
    f8_ear_L = float(ear_L_num / ear_L_den)

    ear_R_num = np.linalg.norm(pts2d[43] - pts2d[47]) + np.linalg.norm(pts2d[44] - pts2d[46])
    ear_R_den = 2.0 * np.linalg.norm(pts2d[42] - pts2d[45]) + eps
    f9_ear_R = float(ear_R_num / ear_R_den)

    f10_eye_sym = float(np.exp(-5.0 * abs(f8_ear_L - f9_ear_R)))

    brow_L = pts2d[18:21].mean(axis=0)
    brow_R = pts2d[23:26].mean(axis=0)
    eye_L_2d = pts2d[36:42].mean(axis=0)
    eye_R_2d = pts2d[42:48].mean(axis=0)
    f11_brow_open = 0.5 * (np.linalg.norm(brow_L - eye_L_2d) + np.linalg.norm(brow_R - eye_R_2d)) / d_iod

    f12_lower_face = float(np.linalg.norm(pts3d[33] - pts3d[8])) / d_iod

    d_cheek_L = float(np.linalg.norm(pts2d[30] - pts2d[2]))
    d_cheek_R = float(np.linalg.norm(pts2d[30] - pts2d[14]))
    f13_mid_sym = min(d_cheek_L, d_cheek_R) / (max(d_cheek_L, d_cheek_R) + eps)

    x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
    w_box, h_box = max(1.0, x2 - x1), max(1.0, y2 - y1)
    f14_frame_ratio = np.sqrt((w_box * h_box) / float(img_w * img_h))
    f15_vert_thirds = 0.5 * (y1 + y2) / float(img_h)
    f16_det_score = float(face.det_score)

    geom_16d = np.array([
        f1_sin_yaw, f2_sin_pitch, f3_sin_roll, f4_cos_yaw,
        f5_smile_curve, f6_smile_width, f7_mar,
        f8_ear_L, f9_ear_R, f10_eye_sym, f11_brow_open,
        f12_lower_face, f13_mid_sym,
        f14_frame_ratio, f15_vert_thirds, f16_det_score
    ], dtype=np.float32)

    aimg = face_align.norm_crop(img_bgr, face.kps, 112)
    gray = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2) / (np.std(gray)**2 + 1e-4))

    f_fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f_fft))
    cy, cx = 56, 56
    y_g, x_g = np.ogrid[:112, :112]
    r_dist = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
    r_fft = float(np.sum(mag[r_dist > 14]) / (np.sum(mag) + eps))

    lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB)
    L_chan = lab[:, :, 0].astype(np.float32)
    eye_patch_L = L_chan[45:57, 29:47]
    eye_patch_R = L_chan[45:57, 65:83]
    cheek_patch_L = L_chan[66:84, 29:47]
    cheek_patch_R = L_chan[66:84, 65:83]

    eyes_L_vals = np.concatenate([eye_patch_L.flatten(), eye_patch_R.flatten()])
    cheeks_L_vals = np.concatenate([cheek_patch_L.flatten(), cheek_patch_R.flatten()])

    med_eyes = float(np.median(eyes_L_vals))
    med_cheeks = float(np.median(cheeks_L_vals))
    p98_eyes = float(np.percentile(eyes_L_vals, 98))

    r_sunglass = float(med_eyes / (med_cheeks + 1.0))
    c_spec = float((p98_eyes - med_eyes) / (med_cheeks + 1.0))

    pose_veto = (abs(yaw_deg) > 58.0) or (pitch_deg < -38.0) or (pitch_deg > 35.0)
    blur_veto = (tenengrad < 0.018) and (r_fft < 0.12)
    sunglass_veto = (r_sunglass < 0.35) and (f14_frame_ratio < 0.03)

    return geom_16d, {
        "tenengrad": tenengrad,
        "r_fft": r_fft,
        "r_sunglass": r_sunglass,
        "c_spec": c_spec,
        "pose_veto": pose_veto,
        "blur_veto": blur_veto,
        "sunglass_veto": sunglass_veto,
        "bbox": [x1, y1, x2, y2]
    }


def letterbox_siglip_full(img_bgr, target_size=384):
    h, w = img_bgr.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    px = (target_size - nw) // 2
    py = (target_size - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - 0.5) / 0.5
    return norm.transpose(2, 0, 1)


def crop_portrait(img_bgr, bbox, target_size=384):
    H, W, _ = img_bgr.shape
    if bbox is not None and bbox[0] >= 0:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) - 0.05 * h
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
        crop = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    else:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    return crop, face_valid


def main():
    print("=" * 80, flush=True)
    print("STEP 1: EXTRACTING COMPLETE FEATURE SUITE FOR UNSEEN HOLDOUT (4,540 PROFILES)", flush=True)
    print("=" * 80, flush=True)

    master = np.load(str(MASTER_STORE), allow_pickle=True)
    master_fnames = set(master["filenames"])
    print(f"Master training set profiles: {len(master_fnames)}", flush=True)

    df_labels = pd.read_csv(r"D:\BumbleLog\binary_swipe_labels.csv").drop_duplicates(subset=["filename"], keep="first").reset_index(drop=True)
    holdout_df = df_labels[~df_labels["filename"].isin(master_fnames)].copy().reset_index(drop=True)
    n = len(holdout_df)
    paths = holdout_df["path"].tolist()
    fnames = holdout_df["filename"].tolist()
    y_labels = (holdout_df["rating"] >= 50.0).astype(int).values

    print(f"Holdout Set Size: {n} profiles (Likes: {y_labels.sum()}, Passes: {n - y_labels.sum()})", flush=True)

    # 1. InsightFace
    print("\n--- STAGE 1: InsightFace buffalo_l (ArcFace 512D + 16D Geometry + Vetoes) ---", flush=True)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    arc_embs = np.zeros((n, 512), dtype=np.float32)
    geom_16d = np.zeros((n, 16), dtype=np.float32)
    face_valids = np.zeros((n, 1), dtype=np.float32)
    bboxes = np.full((n, 4), -1.0, dtype=np.float32)
    hard_vetos = np.zeros(n, dtype=bool)

    t0 = time.time()
    for idx, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            hard_vetos[idx] = True
            continue

        faces = app.get(img)
        m_faces = len(faces)

        if m_faces == 0:
            hard_vetos[idx] = True
            continue

        faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        primary = faces_sorted[0]
        a1 = (primary.bbox[2] - primary.bbox[0]) * (primary.bbox[3] - primary.bbox[1])

        rho = 0.0
        if m_faces >= 2:
            a2 = (faces_sorted[1].bbox[2] - faces_sorted[1].bbox[0]) * (faces_sorted[1].bbox[3] - faces_sorted[1].bbox[1])
            rho = float(a2 / (a1 + 1e-6))

        group_veto = (rho > 0.55) or (m_faces >= 4 and rho > 0.35)

        g16, v_dict = extract_16d_geometry_and_veto(primary, img)
        emb = primary.embedding
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        arc_embs[idx] = emb
        geom_16d[idx] = g16
        face_valids[idx, 0] = 1.0
        bboxes[idx] = v_dict["bbox"]
        hard_vetos[idx] = group_veto or v_dict["pose_veto"] or v_dict["blur_veto"] or v_dict["sunglass_veto"]

        if (idx + 1) % 1000 == 0 or (idx + 1) == n:
            dt = time.time() - t0
            speed = (idx + 1) / dt
            rem = (n - idx - 1) / speed if speed > 0 else 0
            n_v = int(np.sum(hard_vetos[:idx+1]))
            print(f"  InsightFace: {idx+1}/{n} ({idx+1/n*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m | Vetoed: {n_v}", flush=True)

    del app
    torch.cuda.empty_cache()

    executor = ThreadPoolExecutor(max_workers=8)

    # 2. SigLIP SO400M Full & Portrait Crops
    print("\n--- STAGE 2: SigLIP SO400M (Full 1.0x & Portrait 1.8x) ---", flush=True)
    sig_model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        cache_dir=r"D:\BumbleClawHFCache",
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device).eval()

    sig_full_embs = np.zeros((n, 1152), dtype=np.float32)
    sig_port_embs = np.zeros((n, 1152), dtype=np.float32)

    def load_siglip_pair(item):
        p, bb = item
        img = cv2.imread(p)
        if img is None:
            return np.zeros((3, 384, 384), dtype=np.float32), np.zeros((3, 384, 384), dtype=np.float32)
        f_norm = letterbox_siglip_full(img, 384)
        c_bgr, _ = crop_portrait(img, bb, 384)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        p_norm = ((c_rgb - 0.5) / 0.5).transpose(2, 0, 1)
        return f_norm, p_norm

    batch_size_sig = 64
    t0 = time.time()
    for i in range(0, n, batch_size_sig):
        end_idx = min(i + batch_size_sig, n)
        b_paths = paths[i:end_idx]
        b_bboxes = bboxes[i:end_idx]
        items = list(zip(b_paths, b_bboxes))
        results = list(executor.map(load_siglip_pair, items))

        b_full = torch.tensor(np.stack([r[0] for r in results]), dtype=torch.float16, device=device)
        b_port = torch.tensor(np.stack([r[1] for r in results]), dtype=torch.float16, device=device)

        with torch.no_grad():
            out_full = sig_model(pixel_values=b_full).pooler_output
            out_full = F.normalize(out_full, p=2, dim=-1)
            sig_full_embs[i:end_idx] = out_full.cpu().float().numpy()

            out_port = sig_model(pixel_values=b_port).pooler_output
            out_port = F.normalize(out_port, p=2, dim=-1)
            sig_port_embs[i:end_idx] = out_port.cpu().float().numpy()

        if (end_idx % 1000 == 0) or end_idx == n:
            dt = time.time() - t0
            speed = end_idx / dt
            rem = (n - end_idx) / speed if speed > 0 else 0
            print(f"  SigLIP: {end_idx}/{n} ({end_idx/n*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m", flush=True)

    del sig_model
    torch.cuda.empty_cache()

    # 3. DINOv2 Portrait Crop (1536D Attentive Saliency Pooling)
    print("\n--- STAGE 3: DINOv2-reg (Portrait 1.8x Attentive Saliency Pooling 1536D) ---", flush=True)
    dino2_model = AutoModel.from_pretrained(
        "facebook/dinov2-with-registers-base",
        cache_dir=r"D:\hf_cache",
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device).eval()

    dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    dino_std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    dino2_embs = np.zeros((n, 1536), dtype=np.float32)

    def load_dino2_crop(item):
        p, bb = item
        img = cv2.imread(p)
        if img is None:
            return np.zeros((378, 378, 3), dtype=np.uint8)
        c_bgr, _ = crop_portrait(img, bb, 378)
        return c_bgr

    batch_size_d2 = 48
    t0 = time.time()
    for i in range(0, n, batch_size_d2):
        end_idx = min(i + batch_size_d2, n)
        b_paths = paths[i:end_idx]
        b_bboxes = bboxes[i:end_idx]
        items = list(zip(b_paths, b_bboxes))
        results = list(executor.map(load_dino2_crop, items))

        B = len(results)
        batch_np = np.stack(results, axis=0)
        rgb = batch_np[..., ::-1].copy()
        x = torch.from_numpy(rgb).to(device, non_blocking=True).permute(0, 3, 1, 2).half() / 255.0
        x = (x - dino_mean) / dino_std

        with torch.no_grad():
            hidden = dino2_model.embeddings(x)
            for layer in dino2_model.encoder.layer[:-1]:
                out = layer(hidden)
                hidden = out[0] if isinstance(out, tuple) else out

            last_layer = dino2_model.encoder.layer[-1]
            normed_in = last_layer.norm1(hidden)
            _, N, C = normed_in.shape
            num_heads = last_layer.attention.attention.num_attention_heads
            head_dim = C // num_heads

            q_cls = last_layer.attention.attention.query(normed_in[:, :1, :])
            k_all = last_layer.attention.attention.key(normed_in)

            q_cls = q_cls.view(B, 1, num_heads, head_dim).transpose(1, 2)
            k_all = k_all.view(B, N, num_heads, head_dim).transpose(1, 2)

            attn_logits = torch.matmul(q_cls, k_all.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_probs  = F.softmax(attn_logits.float(), dim=-1).squeeze(2)

            a_spatial = attn_probs[:, :, 5:].mean(dim=1)
            w_patch   = a_spatial / (a_spatial.sum(dim=-1, keepdim=True) + 1e-8)

            last_out = last_layer(hidden)
            last_hidden = last_out[0] if isinstance(last_out, tuple) else last_out
            out_hidden = dino2_model.layernorm(last_hidden).float()

            x_cls = F.normalize(out_hidden[:, 0, :], p=2, dim=-1)
            p_patches = out_hidden[:, 5:, :]
            p_fg = torch.bmm(w_patch.unsqueeze(1), p_patches).squeeze(1)
            p_fg = F.normalize(p_fg, p=2, dim=-1)

            x_dino = F.normalize(torch.cat([x_cls, p_fg], dim=-1), p=2, dim=-1)
            dino2_embs[i:end_idx] = x_dino.cpu().numpy()

        if (end_idx % 1000 == 0) or end_idx == n:
            dt = time.time() - t0
            speed = end_idx / dt
            rem = (n - end_idx) / speed if speed > 0 else 0
            print(f"  DINOv2: {end_idx}/{n} ({end_idx/n*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m", flush=True)

    del dino2_model
    torch.cuda.empty_cache()

    # 4. DINOv3 Portrait Crop (1536D Gram-anchored Saliency Pooling)
    print("\n--- STAGE 4: DINOv3 ViT-B/16 (Portrait 1.8x Gram Saliency Pooling 1536D) ---", flush=True)
    dino3_model = AutoModel.from_pretrained(DINOV3_LOCAL_PATH).to(device).eval()
    dino3_embs = np.zeros((n, 1536), dtype=np.float32)

    def load_dino3_crop(item):
        p, bb = item
        img = cv2.imread(p)
        if img is None:
            return np.zeros((384, 384, 3), dtype=np.uint8)
        c_bgr, _ = crop_portrait(img, bb, 384)
        return c_bgr

    batch_size_d3 = 48
    t0 = time.time()
    for i in range(0, n, batch_size_d3):
        end_idx = min(i + batch_size_d3, n)
        b_paths = paths[i:end_idx]
        b_bboxes = bboxes[i:end_idx]
        items = list(zip(b_paths, b_bboxes))
        results = list(executor.map(load_dino3_crop, items))

        batch_np = np.stack(results, axis=0)
        rgb = batch_np[..., ::-1].copy()
        x = torch.from_numpy(rgb).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x = (x - dino_mean.float()) / dino_std.float()

        with torch.no_grad():
            out = dino3_model(x).last_hidden_state
            cls_tok = F.normalize(out[:, 0, :], p=2, dim=-1)
            patches = out[:, 5:, :]
            p_norm = F.normalize(patches, p=2, dim=-1)

            gram_sim = torch.bmm(p_norm, cls_tok.unsqueeze(-1)).squeeze(-1) / 0.15
            w_patch  = F.softmax(gram_sim, dim=-1).unsqueeze(1)
            p_fg     = F.normalize(torch.bmm(w_patch, p_norm).squeeze(1), p=2, dim=-1)

            x_dino1536 = F.normalize(torch.cat([cls_tok, p_fg], dim=-1), p=2, dim=-1)
            dino3_embs[i:end_idx] = x_dino1536.cpu().numpy().astype(np.float32)

        if (end_idx % 1000 == 0) or end_idx == n:
            dt = time.time() - t0
            speed = end_idx / dt
            rem = (n - end_idx) / speed if speed > 0 else 0
            print(f"  DINOv3: {end_idx}/{n} ({end_idx/n*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m", flush=True)

    del dino3_model
    torch.cuda.empty_cache()

    # 5. MultimodalX Original Baseline Scores
    print("\n--- STAGE 5: Mapping S_MultiX from scores.csv ---", flush=True)
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

    S_multi_x = np.array([score_map.get(fn, 50.0) for fn in fnames], dtype=np.float32)
    print(f"  S_MultiX: mean={S_multi_x.mean():.2f}, min={S_multi_x.min():.2f}, max={S_multi_x.max():.2f}", flush=True)

    # 6. Save Complete Holdout Store
    print(f"\nSaving consolidated holdout tensors to {OUT_NPZ}...", flush=True)
    np.savez_compressed(
        OUT_NPZ,
        X_arc=arc_embs,
        X_geom=geom_16d,
        X_sig_full=sig_full_embs,
        X_sig_port=sig_port_embs,
        X_dino2=dino2_embs,
        X_dino3=dino3_embs,
        face_valid=face_valids,
        bboxes=bboxes,
        hard_veto=hard_vetos,
        y=y_labels,
        s_multi_x=S_multi_x,
        paths=np.array(paths),
        filenames=np.array(fnames)
    )
    print(f"[OK] Successfully saved {n} holdout profiles to {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1024 / 1024:.1f} MB)!", flush=True)


if __name__ == "__main__":
    main()
