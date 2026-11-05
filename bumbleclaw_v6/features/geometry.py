"""16-D landmark geometry and legacy quality indicators.

This is a pure function over an InsightFace face object; it does not load models.
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_16d_geometry_and_veto(face, img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    img_h, img_w = img_bgr.shape[:2]
    pts3d = face.landmark_3d_68.astype(np.float32)
    pts2d = pts3d[:, :2]
    eps = 1e-6
    p_left_eye_3d = pts3d[36:42].mean(axis=0)
    p_right_eye_3d = pts3d[42:48].mean(axis=0)
    d_iod = float(np.linalg.norm(p_left_eye_3d - p_right_eye_3d)) + eps
    pitch_deg, yaw_deg, roll_deg = (float(v) for v in face.pose)
    pitch_rad, yaw_rad, roll_rad = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    y_stomion = 0.5 * (pts2d[62, 1] + pts2d[66, 1])
    y_corners = 0.5 * (pts2d[48, 1] + pts2d[54, 1])
    mouth_width = float(np.linalg.norm(pts2d[48] - pts2d[54])) + eps
    mouth_open = float(np.linalg.norm(pts2d[62] - pts2d[66]))
    ear_l_num = np.linalg.norm(pts2d[37] - pts2d[41]) + np.linalg.norm(pts2d[38] - pts2d[40])
    ear_l_den = 2.0 * np.linalg.norm(pts2d[36] - pts2d[39]) + eps
    ear_r_num = np.linalg.norm(pts2d[43] - pts2d[47]) + np.linalg.norm(pts2d[44] - pts2d[46])
    ear_r_den = 2.0 * np.linalg.norm(pts2d[42] - pts2d[45]) + eps
    ear_l, ear_r = float(ear_l_num / ear_l_den), float(ear_r_num / ear_r_den)
    brow_l = pts2d[18:21].mean(axis=0)
    brow_r = pts2d[23:26].mean(axis=0)
    eye_l = pts2d[36:42].mean(axis=0)
    eye_r = pts2d[42:48].mean(axis=0)
    d_cheek_l = float(np.linalg.norm(pts2d[30] - pts2d[2]))
    d_cheek_r = float(np.linalg.norm(pts2d[30] - pts2d[14]))
    x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
    w_box, h_box = max(1.0, x2 - x1), max(1.0, y2 - y1)
    geom = np.array([
        np.sin(yaw_rad), np.sin(pitch_rad), np.sin(roll_rad), np.cos(yaw_rad),
        (y_stomion - y_corners) / d_iod, mouth_width / d_iod, mouth_open / mouth_width,
        ear_l, ear_r, float(np.exp(-5.0 * abs(ear_l - ear_r))),
        0.5 * (np.linalg.norm(brow_l - eye_l) + np.linalg.norm(brow_r - eye_r)) / d_iod,
        float(np.linalg.norm(pts3d[33] - pts3d[8])) / d_iod,
        min(d_cheek_l, d_cheek_r) / (max(d_cheek_l, d_cheek_r) + eps),
        np.sqrt((w_box * h_box) / float(img_w * img_h)),
        0.5 * (y1 + y2) / float(img_h),
        float(face.det_score),
    ], dtype=np.float32)
    aimg = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2) / (np.std(gray) ** 2 + 1e-4))
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    y_g, x_g = np.ogrid[:112, :112]
    r_dist = np.sqrt((x_g - 56) ** 2 + (y_g - 56) ** 2)
    r_fft = float(np.sum(mag[r_dist > 14]) / (np.sum(mag) + eps))
    lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    eyes = np.concatenate([l_chan[45:57, 29:47].ravel(), l_chan[45:57, 65:83].ravel()])
    cheeks = np.concatenate([l_chan[66:84, 29:47].ravel(), l_chan[66:84, 65:83].ravel()])
    med_eyes, med_cheeks = float(np.median(eyes)), float(np.median(cheeks))
    r_sunglass = med_eyes / (med_cheeks + 1.0)
    return geom, {
        "tenengrad": tenengrad,
        "r_fft": r_fft,
        "r_sunglass": r_sunglass,
        "pose_veto": (abs(yaw_deg) > 58.0) or (pitch_deg < -38.0) or (pitch_deg > 35.0),
        "blur_veto": (tenengrad < 0.018) and (r_fft < 0.12),
        "sunglass_veto": (r_sunglass < 0.35) and (geom[13] < 0.03),
        "bbox": [x1, y1, x2, y2],
    }
