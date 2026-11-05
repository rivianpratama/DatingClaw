import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache"
os.environ["TORCH_HOME"] = r"D:\hf_cache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from face_similarity.store import load_store

root = Path(__file__).resolve().parent.parent

def extract_16d_geometry_and_veto(face, img_bgr):
    """
    face: InsightFace Face object
    img_bgr: Full image [H, W, 3]
    Returns: (geom_16d, veto_dict)
    """
    img_h, img_w = img_bgr.shape[:2]
    pts3d = face.landmark_3d_68.astype(np.float32)  # [68, 3]
    pts2d = pts3d[:, :2]                            # [68, 2] image-plane
    eps = 1e-6

    # 3D eye centers for pose-invariant Inter-Ocular Distance (d_iod)
    p_left_eye_3d  = pts3d[36:42].mean(axis=0)
    p_right_eye_3d = pts3d[42:48].mean(axis=0)
    d_iod = float(np.linalg.norm(p_left_eye_3d - p_right_eye_3d)) + eps

    # 1-4: 3D Head Pose (InsightFace ordering: [pitch, yaw, roll])
    pitch_deg = float(face.pose[0])
    yaw_deg   = float(face.pose[1])
    roll_deg  = float(face.pose[2])
    
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad   = np.deg2rad(yaw_deg)
    roll_rad  = np.deg2rad(roll_deg)
    
    f1_sin_yaw   = np.sin(yaw_rad)
    f2_sin_pitch = np.sin(pitch_rad)
    f3_sin_roll  = np.sin(roll_rad)
    f4_cos_yaw   = np.cos(yaw_rad)

    # 5: Mouth Corner Curvature (y increases downward; stomion > corners => smile)
    y_stomion = 0.5 * (pts2d[62, 1] + pts2d[66, 1])
    y_corners = 0.5 * (pts2d[48, 1] + pts2d[54, 1])
    f5_smile_curve = (y_stomion - y_corners) / d_iod

    # 6: Smile Width Ratio
    mouth_width = float(np.linalg.norm(pts2d[48] - pts2d[54])) + eps
    f6_smile_width = mouth_width / d_iod

    # 7: Mouth Aspect Ratio (MAR)
    mouth_open = float(np.linalg.norm(pts2d[62] - pts2d[66]))
    f7_mar = mouth_open / mouth_width

    # 8-9: Left & Right Eye Aspect Ratio (EAR)
    ear_L_num = np.linalg.norm(pts2d[37] - pts2d[41]) + np.linalg.norm(pts2d[38] - pts2d[40])
    ear_L_den = 2.0 * np.linalg.norm(pts2d[36] - pts2d[39]) + eps
    f8_ear_L  = float(ear_L_num / ear_L_den)

    ear_R_num = np.linalg.norm(pts2d[43] - pts2d[47]) + np.linalg.norm(pts2d[44] - pts2d[46])
    ear_R_den = 2.0 * np.linalg.norm(pts2d[42] - pts2d[45]) + eps
    f9_ear_R  = float(ear_R_num / ear_R_den)

    # 10: Bilateral Eye Symmetry
    f10_eye_sym = float(np.exp(-5.0 * abs(f8_ear_L - f9_ear_R)))

    # 11: Brow-to-Eye Openness
    brow_L = pts2d[18:21].mean(axis=0)
    brow_R = pts2d[23:26].mean(axis=0)
    eye_L_2d = pts2d[36:42].mean(axis=0)
    eye_R_2d = pts2d[42:48].mean(axis=0)
    f11_brow_open = 0.5 * (np.linalg.norm(brow_L - eye_L_2d) + np.linalg.norm(brow_R - eye_R_2d)) / d_iod

    # 12: Lower-Face Neoteny Ratio (subnasale 33 to menton 8)
    f12_lower_face = float(np.linalg.norm(pts3d[33] - pts3d[8])) / d_iod

    # 13: Bilateral Mid-Face Symmetry (nose tip 30 to outer cheeks 2 and 14)
    d_cheek_L = float(np.linalg.norm(pts2d[30] - pts2d[2]))
    d_cheek_R = float(np.linalg.norm(pts2d[30] - pts2d[14]))
    f13_mid_sym = min(d_cheek_L, d_cheek_R) / (max(d_cheek_L, d_cheek_R) + eps)

    # 14-16: Framing & Detection Confidence
    x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
    w_box, h_box = max(1.0, x2 - x1), max(1.0, y2 - y1)
    f14_frame_ratio = np.sqrt((w_box * h_box) / float(img_w * img_h))
    f15_vert_thirds = 0.5 * (y1 + y2) / float(img_h)
    f16_det_score   = float(face.det_score)

    geom_16d = np.array([
        f1_sin_yaw, f2_sin_pitch, f3_sin_roll, f4_cos_yaw,
        f5_smile_curve, f6_smile_width, f7_mar,
        f8_ear_L, f9_ear_R, f10_eye_sym, f11_brow_open,
        f12_lower_face, f13_mid_sym,
        f14_frame_ratio, f15_vert_thirds, f16_det_score
    ], dtype=np.float32)

    # Pre-ViT Veto Metrics:
    # A) Tenengrad Sharpness & FFT on 112x112 aligned face crop
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

    # B) Sunglasses Contrast Ratio (LAB space)
    lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB)
    L_chan = lab[:, :, 0].astype(np.float32)
    # Left eye ~(38, 51), right eye ~(74, 51)
    # Cheeks below eyes ~(38, 75), ~(74, 75)
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

    # Pre-ViT Veto Boolean Decisions:
    # Pose veto: |yaw| > 58 deg or pitch < -38 deg or pitch > +35 deg
    pose_veto = (abs(yaw_deg) > 58.0) or (pitch_deg < -38.0) or (pitch_deg > 35.0)
    # Blur veto: tenengrad < 0.018 and r_fft < 0.12
    blur_veto = (tenengrad < 0.018) and (r_fft < 0.12)
    # Sunglasses veto: r_sunglass < 0.35 and frame_ratio < 0.03
    sunglass_veto = (r_sunglass < 0.35) and (f14_frame_ratio < 0.03)

    return geom_16d, {
        "tenengrad": tenengrad,
        "r_fft": r_fft,
        "r_sunglass": r_sunglass,
        "c_spec": c_spec,
        "pose_veto": pose_veto,
        "blur_veto": blur_veto,
        "sunglass_veto": sunglass_veto,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "roll_deg": roll_deg,
        "bbox": [x1, y1, x2, y2],
        "det_score": f16_det_score
    }


def process_paths(app, paths, desc="Processing"):
    n = len(paths)
    print(f"\n[{desc}] Extracting 16D Geometry & Vetoes for {n} images...")
    
    all_geom = np.zeros((n, 16), dtype=np.float32)
    all_face_valids = np.zeros((n, 1), dtype=np.float32)
    all_bboxes = np.full((n, 4), -1.0, dtype=np.float32)
    all_m_faces = np.zeros(n, dtype=np.int32)
    all_rho_area = np.zeros(n, dtype=np.float32)
    all_tenengrad = np.zeros(n, dtype=np.float32)
    all_r_fft = np.zeros(n, dtype=np.float32)
    all_r_sunglass = np.zeros(n, dtype=np.float32)
    all_c_spec = np.zeros(n, dtype=np.float32)
    all_hard_veto = np.zeros(n, dtype=bool)

    t0 = time.time()
    for idx, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            all_hard_veto[idx] = True
            continue

        faces = app.get(img)
        m_faces = len(faces)
        all_m_faces[idx] = m_faces

        if m_faces == 0:
            all_hard_veto[idx] = True
            continue

        # Sort faces by area descending
        faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        primary = faces_sorted[0]
        a1 = (primary.bbox[2] - primary.bbox[0]) * (primary.bbox[3] - primary.bbox[1])
        
        rho = 0.0
        if m_faces >= 2:
            a2 = (faces_sorted[1].bbox[2] - faces_sorted[1].bbox[0]) * (faces_sorted[1].bbox[3] - faces_sorted[1].bbox[1])
            rho = float(a2 / (a1 + 1e-6))
        all_rho_area[idx] = rho

        # Check group photo veto
        group_veto = (rho > 0.55) or (m_faces >= 4 and rho > 0.35)

        geom_16d, v_dict = extract_16d_geometry_and_veto(primary, img)
        
        all_geom[idx] = geom_16d
        all_face_valids[idx, 0] = 1.0
        all_bboxes[idx] = v_dict["bbox"]
        all_tenengrad[idx] = v_dict["tenengrad"]
        all_r_fft[idx] = v_dict["r_fft"]
        all_r_sunglass[idx] = v_dict["r_sunglass"]
        all_c_spec[idx] = v_dict["c_spec"]
        
        is_veto = group_veto or v_dict["pose_veto"] or v_dict["blur_veto"] or v_dict["sunglass_veto"]
        all_hard_veto[idx] = is_veto

        if (idx + 1) % 1000 == 0 or (idx + 1) == n:
            dt = time.time() - t0
            speed = (idx + 1) / dt
            rem = (n - idx - 1) / speed if speed > 0 else 0
            n_veto = int(np.sum(all_hard_veto[:idx+1]))
            print(f"  {idx+1}/{n} ({idx+1/n*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m | Vetoed: {n_veto}/{idx+1} ({n_veto/(idx+1)*100:.1f}%)")

    return {
        "geom_16d": all_geom,
        "face_valid": all_face_valids,
        "bboxes": all_bboxes,
        "m_faces": all_m_faces,
        "rho_area": all_rho_area,
        "tenengrad": all_tenengrad,
        "r_fft": all_r_fft,
        "r_sunglass": all_r_sunglass,
        "c_spec": all_c_spec,
        "hard_veto": all_hard_veto,
        "paths": np.array(paths)
    }


def main():
    print("=" * 80)
    print("PHASE 1: 16D GEOMETRY & PRE-VIT HEURISTIC VETO FEATURE EXTRACTION")
    print("=" * 80)

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 1. Validation Set (1,533 profiles)
    val_out_path = root / "embeddings" / "geom_store_val_1582.npz"
    if not val_out_path.exists():
        r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
        val_data = process_paths(app, r_val.paths, desc="Holdout Validation 1582")
        np.savez_compressed(val_out_path, **val_data)
        print(f"[OK] Saved Validation Geom Store: {val_out_path}")
    else:
        print(f"[EXISTS] {val_out_path}")

    # 2. Consolidated 40k Training Set (39,346 profiles)
    train_out_path = root / "embeddings" / "geom_store_train_40k.npz"
    if not train_out_path.exists():
        r_30k = load_store(str(root / "embeddings" / "reference_store_bumble_30k_pure.npz"))
        r_h2  = load_store(str(root / "embeddings" / "reference_store_bumble_30k_to_33k.npz"))
        r_new = load_store(str(root / "embeddings" / "reference_store_bumble_33k_to_40k.npz"))
        train_paths = np.concatenate([r_30k.paths, r_h2.paths, r_new.paths])
        
        train_data = process_paths(app, train_paths, desc="Full 40k Training Corpus")
        np.savez_compressed(train_out_path, **train_data)
        print(f"[OK] Saved Training Geom Store: {train_out_path}")
    else:
        print(f"[EXISTS] {train_out_path}")

    print("\n[ALL GEOMETRY EXTRACTIONS COMPLETE!]")

if __name__ == "__main__":
    main()
