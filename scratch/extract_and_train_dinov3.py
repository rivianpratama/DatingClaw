import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import joblib

os.environ["HF_HOME"] = r"D:\BumbleClawHFCache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\BumbleClawHFCache"
os.environ["TORCH_HOME"] = r"D:\BumbleClawTorchCache"
os.environ["TEMP"] = r"D:\BumbleTemp"
os.environ["TMP"] = r"D:\BumbleTemp"
os.environ["TMPDIR"] = r"D:\BumbleTemp"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel
from face_similarity.store import load_store
from face_similarity.v4_architecture import ExpressionFiLMArcFace
from face_similarity.v5_architecture import (
    MultiHeadOrthogonalBilinearFusion,
    PyramidalConcentricSubcenterHead,
    evaluate_holdout_v4_vs_v5
)
from scratch.train_50k_grand_ensemble import (
    ConformalGMMGeometricVeto,
    MultiXSafeMetaBlender,
    FrontierLambdaLoss208,
    Master50kDataset
)
import __main__
__main__.MultiXSafeMetaBlender = MultiXSafeMetaBlender
__main__.ConformalGMMGeometricVeto = ConformalGMMGeometricVeto

root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV3_LOCAL_PATH = r"C:\Users\Rivian\dinov3-vitb16-pretrain-lvd1689m"
MASTER_STORE_IN = Path(r"D:\master_50k_store.npz")
MASTER_STORE_DINOV3 = Path(r"D:\master_50k_dinov3_store.npz")
MODELS_DIR = root / "models"


# ==============================================================================
# 1. DINOv3 1536D FEATURE EXTRACTOR
# ==============================================================================
def extract_portrait_crop_384(img_bgr: np.ndarray, bbox: list | None) -> tuple[np.ndarray, float]:
    """
    Crops 1.8x head box and resizes to 384x384 (24x24 grid of 16x16 patches).
    Returns: (crop_384 [384, 384, 3] uint8, face_valid_flag)
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
        crop = cv2.resize(img_bgr, (384, 384), interpolation=cv2.INTER_CUBIC)
    else:
        crop = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)

    return crop, face_valid


def _load_and_crop(item):
    p, bb = item
    img = cv2.imread(str(p))
    if img is None:
        return np.zeros((384, 384, 3), dtype=np.uint8), 0.0
    bbox_list = bb.tolist() if (bb is not None and bb[0] >= 0) else None
    return extract_portrait_crop_384(img, bbox_list)


class DINOv3PortraitExtractor:
    def __init__(self, model_path: str = DINOV3_LOCAL_PATH, device: str = "cuda"):
        self.device = torch.device(device)
        print(f"Loading local DINOv3 ViT-B/16 from {model_path}...", flush=True)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        batch_np = np.stack(crops_bgr, axis=0)
        rgb = batch_np[..., ::-1].copy()
        x = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std

        out = self.model(x).last_hidden_state
        cls_tok = F.normalize(out[:, 0, :], p=2, dim=-1)
        patches = out[:, 5:, :]
        p_norm = F.normalize(patches, p=2, dim=-1)

        # Gram-anchored saliency pooling
        gram_sim = torch.bmm(p_norm, cls_tok.unsqueeze(-1)).squeeze(-1) / 0.15
        w_patch  = F.softmax(gram_sim, dim=-1).unsqueeze(1)
        p_fg     = F.normalize(torch.bmm(w_patch, p_norm).squeeze(1), p=2, dim=-1)

        x_dino1536 = F.normalize(torch.cat([cls_tok, p_fg], dim=-1), p=2, dim=-1)
        return x_dino1536.cpu().numpy().astype(np.float32)


def run_extraction(extractor, paths, bboxes, out_path, batch_size=48, desc="DINOv3"):
    n_total = len(paths)
    print(f"\n[{desc}] Processing {n_total} images -> {out_path}...", flush=True)
    all_embs = []
    all_valids = []
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
        all_valids.extend(batch_valids)

        if end_idx % 480 == 0 or end_idx == n_total:
            dt = time.time() - t0
            speed = end_idx / dt
            rem = (n_total - end_idx) / speed if speed > 0 else 0
            print(f"  {end_idx}/{n_total} ({end_idx/n_total*100:.1f}%) | Speed: {speed:.1f} img/s | ETA: {rem/60:.1f}m | Valids: {sum(all_valids):.0f}/{end_idx}", flush=True)

    executor.shutdown()
    all_embs = np.concatenate(all_embs, axis=0)
    all_valids = np.array(all_valids, dtype=np.float32).reshape(-1, 1)

    np.savez_compressed(
        out_path,
        embeddings=all_embs,
        face_valid=all_valids,
        paths=np.array(paths),
        model_name="DINOv3-ViT-B/16"
    )
    print(f"[OK] Saved {out_path} ({all_embs.shape})", flush=True)
    return all_embs


# ==============================================================================
# 2. PILLAR ARCHITECTURES WITH DINOv3 SUPPORT
# ==============================================================================
class Pillar1SingleCropNet(nn.Module):
    def __init__(self, drop: float = 0.25):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=256, expr_dim=64)
        self.head = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )

    def forward(self, x_arc, x_geom, face_valid=None):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        if face_valid is not None:
            h_arc = h_arc * face_valid
            e_expr = e_expr * face_valid
        feat = torch.cat([h_arc, e_expr], dim=-1)
        return self.head(feat).squeeze(-1)


class Pillar2DualCropNet(nn.Module):
    def __init__(self, mod_dim: int = 192, drop: float = 0.25):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=256, expr_dim=64)
        self.branch_arc  = nn.Sequential(nn.Linear(320, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_sigF = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_sigP = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_dino = nn.Sequential(nn.Linear(1536, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.mod_gates   = nn.Parameter(torch.ones(4, 1))

        self.head = nn.Sequential(
            nn.Linear(mod_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )

    def forward(self, x_arc, x_geom, x_sig_full, x_sig_port, x_dino, face_valid=None):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        if face_valid is not None:
            h_arc = h_arc * face_valid
            e_expr = e_expr * face_valid
        h_face = torch.cat([h_arc, e_expr], dim=-1)

        w = F.softmax(self.mod_gates, dim=0) * 4.0
        h0 = w[0] * self.branch_arc(h_face)
        h1 = w[1] * self.branch_sigF(F.normalize(x_sig_full, p=2, dim=-1))
        h2 = w[2] * self.branch_sigP(F.normalize(x_sig_port, p=2, dim=-1))
        h3 = w[3] * self.branch_dino(F.normalize(x_dino, p=2, dim=-1))

        fused = torch.cat([h0, h1, h2, h3], dim=-1)
        return self.head(fused).squeeze(-1)


class Pillar3SubcenterNet(nn.Module):
    def __init__(self, latent_dim: int = 256, K: int = 16, M: int = 48, tau: float = 0.08, gamma: float = 16.0, drop: float = 0.20):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=latent_dim, expr_dim=64)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + 64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(latent_dim, latent_dim)
        )
        self.K = K
        self.M = M
        self.tau = tau
        self.gamma = gamma
        self.proxies_pos = nn.Parameter(torch.randn(K, latent_dim))
        self.proxies_neg = nn.Parameter(torch.randn(M, latent_dim))
        self.bias_b0 = nn.Parameter(torch.tensor(-2.10))

    def embed(self, x_arc, x_geom, face_valid=None):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        if face_valid is not None:
            h_arc = h_arc * face_valid
            e_expr = e_expr * face_valid
        z = self.proj(torch.cat([h_arc, e_expr], dim=-1))
        return F.normalize(z, p=2, dim=-1)

    def compute_logits(self, z):
        p_pos = F.normalize(self.proxies_pos, p=2, dim=-1)
        p_neg = F.normalize(self.proxies_neg, p=2, dim=-1)
        cos_pos = torch.matmul(z, p_pos.t())
        cos_neg = torch.matmul(z, p_neg.t())
        s_pos = self.tau * torch.logsumexp((cos_pos - 0.15) / self.tau, dim=-1)
        s_neg = self.tau * torch.logsumexp(cos_neg / self.tau, dim=-1)
        return self.gamma * (s_pos - s_neg) + self.bias_b0

    def repulsion_loss(self):
        p_neg = F.normalize(self.proxies_neg, p=2, dim=-1)
        p_pos = F.normalize(self.proxies_pos, p=2, dim=-1)
        gram_neg = torch.matmul(p_neg, p_neg.t())
        triu_mask = torch.triu(torch.ones_like(gram_neg, dtype=torch.bool), diagonal=1)
        loss_neg_repel = F.relu(gram_neg[triu_mask] - 0.25).pow(2).mean()
        cross_sim = torch.matmul(p_neg, p_pos.t())
        loss_cross_repel = F.relu(cross_sim - 0.20).pow(2).mean()
        return loss_neg_repel + loss_cross_repel


class Pillar4OrthogonalRankNet(nn.Module):
    def __init__(self, dim: int = 256, drop: float = 0.20):
        super().__init__()
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=dim, expr_dim=64)
        self.vis_enc = nn.Sequential(
            nn.Linear(1152 + 1152 + 1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )
        self.fusion = MultiHeadOrthogonalBilinearFusion(dim=dim, num_heads=8, rank=32, expr_dim=64)
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )

    def forward(self, x_arc, x_geom, x_sig_full, x_sig_port, x_dino, face_valid):
        h_arc, e_expr = self.film_arc(x_arc, x_geom)
        raw_vis = torch.cat([
            F.normalize(x_sig_full, p=2, dim=-1),
            F.normalize(x_sig_port, p=2, dim=-1),
            F.normalize(x_dino, p=2, dim=-1)
        ], dim=-1)
        h_vis = self.vis_enc(raw_vis)
        fused = self.fusion(h_vis, h_arc, e_expr, face_valid)
        return self.head(fused).squeeze(-1)


# ==============================================================================
# 3. MAIN WORKFLOW
# ==============================================================================
def main():
    print("=" * 80, flush=True)
    print("BUMBLECLAW v5.2: DINOv3 EXTRACTION & FULL SUPER-ENSEMBLE RETRAINING", flush=True)
    print(f"Local DINOv3 Source: {DINOV3_LOCAL_PATH}", flush=True)
    print("=" * 80, flush=True)

    # 1. Extract DINOv3 for Holdout val_1582
    val_dinov3_path = root / "embeddings" / "dinov3_portrait_store_val_1582.npz"
    r_val = load_store(str(root / "embeddings" / "reference_store_val_1582.npz"))
    geom_val = np.load(str(root / "embeddings" / "geom_store_val_1582.npz"))
    bboxes_val = geom_val["bboxes"]

    extractor = DINOv3PortraitExtractor(model_path=DINOV3_LOCAL_PATH, device="cuda")

    if not val_dinov3_path.exists():
        print("\n--- Phase 1: Extracting DINOv3 for val_1582 Holdout ---", flush=True)
        val_dino_embs = run_extraction(extractor, r_val.paths, bboxes_val, val_dinov3_path, batch_size=48, desc="val_1582 DINOv3")
    else:
        print(f"\n[EXISTS] {val_dinov3_path} already extracted.", flush=True)
        val_dino_embs = np.load(str(val_dinov3_path))["embeddings"]

    # 2. Extract DINOv3 for 50k Master Store
    dinov3_50k_path = Path(r"D:\dinov3_portrait_store_50k.npz")
    print("\nLoading master 50k image paths...", flush=True)
    st_50k = np.load(str(MASTER_STORE_IN), allow_pickle=True)
    paths_50k = st_50k["paths"]
    bboxes_50k = st_50k["bboxes"]
    N_50k = len(paths_50k)

    if not dinov3_50k_path.exists():
        print(f"\n--- Phase 2: Extracting DINOv3 across {N_50k} Profiles ---", flush=True)
        dino_50k_embs = run_extraction(extractor, paths_50k, bboxes_50k, dinov3_50k_path, batch_size=48, desc="50k DINOv3")
    else:
        print(f"\n[EXISTS] {dinov3_50k_path} already extracted.", flush=True)
        dino_50k_embs = np.load(str(dinov3_50k_path))["embeddings"]

    # 3. Assemble or Verify Master DINOv3 Store
    if not MASTER_STORE_DINOV3.exists():
        print(f"\n--- Phase 3: Building {MASTER_STORE_DINOV3} ---", flush=True)
        np.savez_compressed(
            MASTER_STORE_DINOV3,
            X_arc=st_50k["X_arc"],
            X_geom=st_50k["X_geom"],
            X_sig_full=st_50k["X_sig_full"],
            X_sig_port=st_50k["X_sig_port"],
            X_dino=dino_50k_embs,
            face_valid=st_50k["face_valid"],
            bboxes=st_50k["bboxes"],
            hard_veto=st_50k["hard_veto"],
            y=st_50k["y"],
            s_multi_x=st_50k["s_multi_x"],
            weights=st_50k["weights"],
            delta_mu=st_50k["delta_mu"],
            paths=st_50k["paths"],
            filenames=st_50k["filenames"]
        )
        print(f"[OK] Saved DINOv3 Master Store to {MASTER_STORE_DINOV3}", flush=True)
    else:
        print(f"\n[EXISTS] {MASTER_STORE_DINOV3} already ready.", flush=True)

    # Free extractor from VRAM
    del extractor
    torch.cuda.empty_cache()

    # 4. Phase 4: Retrain 5-Fold Super-Ensemble with DINOv3
    print("\n" + "=" * 80, flush=True)
    print("PHASE 4: 5-FOLD CV SUPER-ENSEMBLE RETRAINING WITH DINOv3", flush=True)
    print("=" * 80, flush=True)

    st_d3 = np.load(str(MASTER_STORE_DINOV3), allow_pickle=True)
    X_arc = st_d3["X_arc"]
    X_geom = st_d3["X_geom"]
    X_sig_full = st_d3["X_sig_full"]
    X_sig_port = st_d3["X_sig_port"]
    X_dino = st_d3["X_dino"]
    face_valid = st_d3["face_valid"]
    y = st_d3["y"]
    weights = st_d3["weights"]
    s_multi_x = st_d3["s_multi_x"]
    N = len(y)

    norm_arc = np.linalg.norm(X_arc, axis=-1, keepdims=True)
    norm_arc[norm_arc < 1e-4] = 1.0
    X_arc = (X_arc / norm_arc).astype(np.float32)

    era_ids = np.zeros(N, dtype=int)
    era_ids[21121:35822] = 1
    era_ids[35822:] = 2
    strata_key = 2 * era_ids + y

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X_arc, strata_key, groups=np.arange(N)))

    Z_oof = np.zeros((N, 4), dtype=np.float32)
    batch_size = 512
    epochs = 12

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n>>> [Fold {fold + 1}/5] Train: {len(train_idx)} | Val: {len(val_idx)}", flush=True)

        train_ds = Master50kDataset(
            X_arc[train_idx], X_geom[train_idx], X_sig_full[train_idx], X_sig_port[train_idx],
            X_dino[train_idx], face_valid[train_idx], y[train_idx], weights[train_idx], s_multi_x[train_idx]
        )
        val_ds = Master50kDataset(
            X_arc[val_idx], X_geom[val_idx], X_sig_full[val_idx], X_sig_port[val_idx],
            X_dino[val_idx], face_valid[val_idx], y[val_idx], weights[val_idx], s_multi_x[val_idx]
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        p1 = Pillar1SingleCropNet().to(device)
        p2 = Pillar2DualCropNet().to(device)
        p3 = Pillar3SubcenterNet(K=16, M=48).to(device)
        p4 = Pillar4OrthogonalRankNet().to(device)

        lambda_crit = FrontierLambdaLoss208(setpoint_frac=0.208, skirt_frac=0.040, sigma=3.0, margin=0.15)
        opt1 = optim.AdamW(p1.parameters(), lr=1e-3, weight_decay=1e-4)
        opt2 = optim.AdamW(p2.parameters(), lr=1e-3, weight_decay=1e-4)
        opt3 = optim.AdamW(p3.parameters(), lr=1e-3, weight_decay=1e-4)
        opt4 = optim.AdamW(p4.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            p1.train(); p2.train(); p3.train(); p4.train()
            for b_arc, b_geom, b_sf, b_sp, b_dino, b_fv, b_y, b_w, _ in train_loader:
                b_arc, b_geom, b_sf, b_sp, b_dino, b_fv, b_y, b_w = [
                    t.to(device) for t in [b_arc, b_geom, b_sf, b_sp, b_dino, b_fv, b_y, b_w]
                ]

                # Train P1
                opt1.zero_grad()
                out1 = p1(b_arc, b_geom, b_fv)
                loss1 = F.binary_cross_entropy_with_logits(out1, b_y, weight=b_w)
                loss1.backward()
                opt1.step()

                # Train P2 (Dual-Crop SigLIP + DINOv3)
                opt2.zero_grad()
                out2 = p2(b_arc, b_geom, b_sf, b_sp, b_dino, b_fv)
                loss2 = F.binary_cross_entropy_with_logits(out2, b_y, weight=b_w)
                loss2.backward()
                opt2.step()

                # Train P3
                opt3.zero_grad()
                z3 = p3.embed(b_arc, b_geom, b_fv)
                out3 = p3.compute_logits(z3)
                loss3 = F.binary_cross_entropy_with_logits(out3, b_y, weight=b_w) + 0.10 * p3.repulsion_loss()
                loss3.backward()
                opt3.step()

                # Train P4 (Orthogonal RankNet with DINOv3)
                opt4.zero_grad()
                out4 = p4(b_arc, b_geom, b_sf, b_sp, b_dino, b_fv)
                loss4 = lambda_crit(out4, b_y, b_w)
                loss4.backward()
                opt4.step()

        p1.eval(); p2.eval(); p3.eval(); p4.eval()
        with torch.no_grad():
            preds1, preds2, preds3, preds4 = [], [], [], []
            for b_arc, b_geom, b_sf, b_sp, b_dino, b_fv, b_y, b_w, _ in val_loader:
                b_arc, b_geom, b_sf, b_sp, b_dino, b_fv = [t.to(device) for t in [b_arc, b_geom, b_sf, b_sp, b_dino, b_fv]]
                preds1.append(p1(b_arc, b_geom, b_fv).cpu().numpy())
                preds2.append(p2(b_arc, b_geom, b_sf, b_sp, b_dino, b_fv).cpu().numpy())
                z3_v = p3.embed(b_arc, b_geom, b_fv)
                preds3.append(p3.compute_logits(z3_v).cpu().numpy())
                preds4.append(p4(b_arc, b_geom, b_sf, b_sp, b_dino, b_fv).cpu().numpy())

            Z_oof[val_idx, 0] = np.concatenate(preds1)
            Z_oof[val_idx, 1] = np.concatenate(preds2)
            Z_oof[val_idx, 2] = np.concatenate(preds3)
            Z_oof[val_idx, 3] = np.concatenate(preds4)

    # 5. Fit Meta-Blender
    print("\n--- Fitting MultiXSafeMetaBlender with DINOv3 OOFs ---", flush=True)
    high_elo_mask = (np.arange(N) >= 21121) & (np.arange(N) <= 35821)
    meta_blender = MultiXSafeMetaBlender(hard_floor=42.5, soft_barrier_knee=51.0, barrier_temp=2.0)
    meta_blender.fit_calibrators(Z_oof, y, high_elo_mask)
    meta_blender.fit_meta_weights(Z_oof, s_multi_x, y, weights)

    # Save Models
    joblib.dump(meta_blender, MODELS_DIR / "bumbleclaw_v5_meta_blender_dinov3.joblib")
    torch.save(p1.state_dict(), MODELS_DIR / "bumbleclaw_v5_p1_dinov3.pt")
    torch.save(p2.state_dict(), MODELS_DIR / "bumbleclaw_v5_p2_dinov3.pt")
    torch.save(p3.state_dict(), MODELS_DIR / "bumbleclaw_v5_p3_dinov3.pt")
    torch.save(p4.state_dict(), MODELS_DIR / "bumbleclaw_v5_p4_dinov3.pt")
    print(f"Checkpoints saved to {MODELS_DIR}/bumbleclaw_v5_*_dinov3.*", flush=True)

    # 6. Evaluate on val_1582
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARKING DINOv3 SUPER-ENSEMBLE ON val_1582 HOLDOUT", flush=True)
    print("=" * 80, flush=True)

    veto_model = joblib.load(MODELS_DIR / "bumbleclaw_conformal_gmm_veto.joblib")
    veto_flags = veto_model.veto(geom_val["geom_16d"])

    df_scores = pd.read_csv(r"D:\BumbleLog\scores.csv", low_memory=False).drop_duplicates(subset=["screenshot"], keep="first")
    smap = {
        row["screenshot"]: (
            float(row["score"]) if row.get("method") == "multimodalx_original" and pd.notna(row.get("score"))
            else (0.48 * float(row["ridge"]) + 0.48 * float(row["multimodal"]) + 4.0 if pd.notna(row.get("ridge")) and pd.notna(row.get("multimodal")) else 50.0)
        )
        for _, row in df_scores.iterrows() if pd.notna(row.get("screenshot"))
    }
    s_multi_x_val = np.array([smap.get(Path(p).name, 50.0) for p in r_val.paths], dtype=np.float32)

    with torch.no_grad():
        t_arc_v = torch.tensor(r_val.embeddings, device=device)
        t_geom_v = torch.tensor(geom_val["geom_16d"], device=device)
        t_sf_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_store_val_1582.npz"))["embeddings"], device=device)
        t_sp_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))["embeddings"], device=device)
        t_dino_v = torch.tensor(val_dino_embs, device=device)
        t_fv_v = torch.tensor(np.load(str(root / "embeddings" / "siglip_portrait_store_val_1582.npz"))["face_valid"], device=device)

        z1 = p1(t_arc_v, t_geom_v, t_fv_v).cpu().numpy()
        z2 = p2(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()
        z3 = p3.compute_logits(p3.embed(t_arc_v, t_geom_v, t_fv_v)).cpu().numpy()
        z4 = p4(t_arc_v, t_geom_v, t_sf_v, t_sp_v, t_dino_v, t_fv_v).cpu().numpy()

        Z_val_d3 = np.stack([z1, z2, z3, z4], axis=1)
        s_d3 = meta_blender.predict_score(Z_val_d3, s_multi_x_val)
        s_d3 = np.where(veto_flags, -1e9, s_d3)

    # v4 baseline for statistical test
    meta_v4 = joblib.load(MODELS_DIR / "bumbleclaw_v4_meta_blender_50k.joblib")
    z1_v4 = torch.load(MODELS_DIR / "bumbleclaw_v4_p1_50k.pt")
    # Load v4 Z from scratch
    from scratch.test_lexicographic_ranking import s4 as s_v4_baseline
    y_val = (r_val.ratings >= 50.0).astype(int)

    # v5.1 Clean Scoring with DINOv3:
    L_d3 = meta_blender._calibrated_log_odds(Z_val_d3)
    w_p = meta_blender.meta_params[:4]
    w_lse = meta_blender.meta_params[4]
    bias = meta_blender.meta_params[7]
    tau_lse = meta_blender.tau_lse
    m_max = np.max(L_d3 / tau_lse, axis=1, keepdims=True)
    l_lse = tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L_d3 / tau_lse - m_max), axis=1)))
    s_match_d3 = (L_d3 @ w_p) + w_lse * l_lse + bias

    is_vetoed_d3 = veto_flags | (s_multi_x_val < 42.5) | ((s_multi_x_val < 49.0) & (s_match_d3 < 0.0))
    s_v51_d3 = np.where(is_vetoed_d3, -1e9, s_d3)

    res = evaluate_holdout_v4_vs_v5(s_v4_baseline, s_v51_d3, y_val, s_multi_x_val, setpoint=0.208, n_boot=2000)

    print("\n" + "=" * 90, flush=True)
    print("FINAL HEAD-TO-HEAD COMPARISON: v4 (DINOv2) vs v5.2 (DINOv3)", flush=True)
    print("=" * 90, flush=True)
    v4_s = res["v4_summary"]
    d3_s = res["v5_summary"]

    for k in ["hits_at_k", "recall_at_k", "catastrophic_fp_g1_g2", "fp_multix_cvar5", "fp_multix_mean", "cute_fp_share_g4", "ndcg_graded_at_k", "ndcg_binary_at_k"]:
        print(f"{k:<30} | v4: {v4_s[k]} | v5.2 (DINOv3): {d3_s[k]}", flush=True)
    print(f"McNemar Test: {res['mcnemar_true_likes']}", flush=True)
    print(f"Brunner-Munzel Dominance: {res['brunner_munzel_fp_dominance']}", flush=True)


if __name__ == "__main__":
    main()
