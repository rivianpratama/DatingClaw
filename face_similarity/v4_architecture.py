import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from scipy.optimize import minimize

# ==============================================================================
# 1. EXPRESSION-FILM ARCFACE
# ==============================================================================
class ExpressionFiLMArcFace(nn.Module):
    """
    Decouples expression & pose from facial identity bone structure.
    Projects ArcFace (512D) to 256D, modulates via 16D landmark geometry (gamma, beta),
    and outputs 256D modulated identity + 64D pure landmark expression token.
    """
    def __init__(self, arc_dim: int = 512, geom_dim: int = 16, out_dim: int = 256, expr_dim: int = 64):
        super().__init__()
        self.arc_proj   = nn.Sequential(nn.Linear(arc_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.geom_mlp   = nn.Sequential(nn.Linear(geom_dim, 64), nn.LayerNorm(64), nn.GELU())

        self.film_gamma = nn.Linear(64, out_dim)
        self.film_beta  = nn.Linear(64, out_dim)
        self.expr_proj  = nn.Linear(64, expr_dim)
        self.arc_norm   = nn.LayerNorm(out_dim)

        # Zero-initialize FiLM generators so epoch 0 starts as pure uncorrupted ArcFace
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x_arc: torch.Tensor, x_geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_arc = self.arc_proj(F.normalize(x_arc, p=2, dim=-1))
        g = self.geom_mlp(x_geom)
        gamma = torch.tanh(self.film_gamma(g))  # Bounded in (-1, +1)
        beta  = self.film_beta(g)
        h_modulated = F.normalize(self.arc_norm((1.0 + gamma) * h_arc + beta), p=2, dim=-1)
        e_expr = F.gelu(self.expr_proj(g))
        return h_modulated, e_expr


# ==============================================================================
# 2. UNIFIED PILLAR 4: DUAL-CROP ORTHOGONAL METRIC NET
# ==============================================================================
class DualCropOrthogonalMetricNet(nn.Module):
    """
    Unified Pillar 4 Backbone:
      - Base Anchor: Balanced SigLIP-Full (1152D) + SigLIP-Portrait (1152D) + DINOv2-Portrait (1536D) -> 256D
      - Modulated Identity: ExpressionFiLMArcFace -> h_arc_mod (256D) + e_expr (64D)
      - Gram-Schmidt Orthogonal Residual in 256D -> Unit Hypersphere S^255
    """
    def __init__(self, mod_dim: int = 192, latent_dim: int = 256, expr_dim: int = 64, drop: float = 0.25):
        super().__init__()
        # 1. Visual Base Anchor Branches
        self.branch_sigF = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_sigP = nn.Sequential(nn.Linear(1152, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.branch_dino = nn.Sequential(nn.Linear(1536, mod_dim), nn.LayerNorm(mod_dim), nn.GELU(), nn.Dropout(drop))
        self.base_logits = nn.Parameter(torch.zeros(3, 1))

        self.base_head = nn.Sequential(
            nn.Linear(mod_dim * 3, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(384, latent_dim)
        )

        # 2. Expression-FiLM ArcFace
        self.film_arc = ExpressionFiLMArcFace(arc_dim=512, geom_dim=16, out_dim=latent_dim, expr_dim=expr_dim)

        # 3. Orthogonal Gating Controller
        self.ortho_gate = nn.Linear(latent_dim * 2 + expr_dim + 1, latent_dim)
        nn.init.constant_(self.ortho_gate.bias, -1.5)  # Conservative start (~0.18)
        self.gamma_scale = nn.Parameter(torch.full((1, latent_dim), 0.25))

    def forward(
        self,
        x_arc: torch.Tensor,
        x_geom: torch.Tensor,
        x_sig_full: torch.Tensor,
        x_sig_port: torch.Tensor,
        x_dino: torch.Tensor,
        face_valid: torch.Tensor = None
    ) -> torch.Tensor:
        if face_valid is None:
            face_valid = torch.ones((x_arc.size(0), 1), device=x_arc.device, dtype=x_arc.dtype)

        # Step 1: Base Visual Anchor h_base on S^255
        w_base = F.softmax(self.base_logits, dim=0) * 3.0
        h_sF = w_base[0] * self.branch_sigF(F.normalize(x_sig_full, p=2, dim=-1))
        h_sP = w_base[1] * self.branch_sigP(F.normalize(x_sig_port, p=2, dim=-1))
        h_dP = w_base[2] * self.branch_dino(F.normalize(x_dino, p=2, dim=-1))

        h_base_raw = self.base_head(torch.cat([h_sF, h_sP, h_dP], dim=-1))
        h_base = F.normalize(h_base_raw, p=2, dim=-1)  # [B, 256]

        # Step 2: FiLM-Modulated Identity & Expression Token
        h_arc_mod, e_expr = self.film_arc(x_arc, x_geom)
        h_arc_mod = h_arc_mod * face_valid
        e_expr = e_expr * face_valid

        # Step 3: Gram-Schmidt Orthogonal Projection in 256D
        cos_proj = (h_arc_mod * h_base).sum(dim=-1, keepdim=True)
        h_arc_ortho = h_arc_mod - cos_proj * h_base

        # Step 4: Expression-Conditioned Orthogonal Gating onto S^255
        gate_in = torch.cat([h_base, h_arc_ortho, e_expr, face_valid], dim=-1)
        alpha = self.gamma_scale * torch.sigmoid(self.ortho_gate(gate_in)) * face_valid

        z_sphere = F.normalize(h_base + alpha * h_arc_ortho, p=2, dim=-1)
        return z_sphere


# ==============================================================================
# 3. TRUNCATED LISTWISE LAMBDA LOSS @ K WITH CROSS-ERA MARGIN RELAXATION
# ==============================================================================
class TruncatedLambdaLossAtK(nn.Module):
    """
    Directly optimizes Precision/Recall & NDCG strictly inside the Top-20% to Top-25% quota.
    Incorporates asymmetric cross-era margin relaxation m_ij and era prior-shifted BCE anchor.
    """
    def __init__(
        self,
        quota_frac: float = 0.225,
        sigma: float = 2.5,
        margin: float = 0.20,
        bce_weight: float = 0.25
    ):
        super().__init__()
        self.quota_frac = quota_frac
        self.sigma = sigma
        self.margin = margin
        self.bce_weight = bce_weight

    def forward(
        self,
        logits: torch.Tensor,                # Raw visual logits f_theta(x)
        targets: torch.Tensor,               # Binary labels {0, 1}
        delta_mu: torch.Tensor = None,       # Era prior shift [B]
        sample_weights: torch.Tensor = None  # Curation sample weights [B]
    ) -> torch.Tensor:
        B = logits.size(0)
        K_cut = int(round(self.quota_frac * B))
        K_skirt = int(round(1.5 * K_cut))

        # 1. Descending batch ranks r_i in {1, ..., B} sorted strictly by raw visual logits
        with torch.no_grad():
            sorted_idx = torch.argsort(logits, descending=True)
            ranks = torch.empty_like(sorted_idx)
            ranks[sorted_idx] = torch.arange(1, B + 1, device=logits.device)
            ranks_f = ranks.float()

            inv_log = 1.0 / torch.log2(1.0 + ranks_f)
            discount = torch.where(
                ranks <= K_cut, inv_log,
                torch.where(ranks <= K_skirt, 0.15 * inv_log, 0.02 * inv_log)
            )

        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return F.binary_cross_entropy_with_logits(logits, targets)

        z_pos, d_pos = logits[pos_mask], discount[pos_mask]
        z_neg, d_neg = logits[neg_mask], discount[neg_mask]

        delta_ij = torch.abs(d_pos.unsqueeze(1) - d_neg.unsqueeze(0))

        # Asymmetric cross-era margin relaxation:
        if delta_mu is not None:
            mu_pos = delta_mu[pos_mask].unsqueeze(1)
            mu_neg = delta_mu[neg_mask].unsqueeze(0)
            cross_era_diff = torch.clamp(mu_pos - mu_neg, min=0.0)
            m_ij = torch.clamp(self.margin - 0.35 * cross_era_diff, min=0.0)
        else:
            m_ij = self.margin

        diff_ij = z_pos.unsqueeze(1) - z_neg.unsqueeze(0) - m_ij
        pairwise_loss = delta_ij * (F.softplus(-self.sigma * diff_ij) / self.sigma)

        if sample_weights is not None:
            w_pos = sample_weights[pos_mask].unsqueeze(1)
            w_neg = sample_weights[neg_mask].unsqueeze(0)
            pairwise_loss = pairwise_loss * (w_pos * w_neg)

        loss_lambda = pairwise_loss.sum(dim=1).mean()

        # 2. Calibrated Label-Smoothed BCE Anchor with Full Era Offset
        y_smooth = targets * 0.97 + 0.015
        logits_bce = logits + delta_mu if delta_mu is not None else logits
        loss_bce = F.binary_cross_entropy_with_logits(logits_bce, y_smooth, weight=sample_weights)

        return (1.0 - self.bce_weight) * loss_lambda + self.bce_weight * loss_bce


# ==============================================================================
# 4. SPECIALIST META-BLENDER (LOG-SUM-EXP SPECIALIST RESCUE)
# ==============================================================================
class SpecialistMetaBlender:
    """
    Combines 4 Pillar scores + Subcenter Peak + Face Framing Routing
    using L2-regularized Logistic Regression with non-negativity constraints on base pillars.
    """
    def __init__(self, tau_meta: float = 0.35, l2_reg: float = 1.0):
        self.tau_meta = tau_meta
        self.l2_reg = l2_reg
        self.ref_pillars = None
        self.weights = None
        self.bias = 0.0

    def _to_gaussian_quantiles(self, Z: np.ndarray) -> np.ndarray:
        N_ref = self.ref_pillars.shape[0]
        Q = np.zeros_like(Z, dtype=np.float64)
        for m in range(Z.shape[1]):
            ranks = np.searchsorted(self.ref_pillars[:, m], Z[:, m], side="right")
            u = np.clip((ranks + 0.5) / (N_ref + 1.0), 1e-4, 1.0 - 1e-4)
            Q[:, m] = norm.ppf(u)
        return Q

    def _build_meta_features(self, Z: np.ndarray, max_subcenter_cos: np.ndarray, frame_ratio_std: np.ndarray) -> np.ndarray:
        Q = self._to_gaussian_quantiles(Z)  # [N, 4]
        # Smooth-Max Specialist Rescue (LogSumExp with tau_meta = 0.35)
        m_max = np.max(Q / self.tau_meta, axis=1, keepdims=True)
        q_soft_max = self.tau_meta * (m_max.squeeze(1) + np.log(np.sum(np.exp(Q / self.tau_meta - m_max), axis=1)))

        # Framing-conditioned routing between Dual-Crop (P2, P4) and Single-Crop (P1, P3)
        dual_vs_single = 0.5 * (Q[:, 1] + Q[:, 3]) - 0.5 * (Q[:, 0] + Q[:, 2])
        route_interact = frame_ratio_std * dual_vs_single

        return np.column_stack([Q, q_soft_max, max_subcenter_cos, route_interact])

    def fit(self, Z_oof: np.ndarray, max_cos_oof: np.ndarray, frame_std_oof: np.ndarray, y_oof: np.ndarray):
        self.ref_pillars = np.sort(Z_oof, axis=0)
        X = self._build_meta_features(Z_oof, max_cos_oof, frame_std_oof)

        def obj(params):
            w, b = params[:-1], params[-1]
            logits = X @ w + b
            loss = np.mean(np.logaddexp(0.0, logits) - y_oof * logits)
            return loss + 0.5 * self.l2_reg * np.sum(w ** 2) / len(y_oof)

        init_p = np.array([0.15, 0.30, 0.15, 0.25, 0.35, 0.50, 0.10, -2.20])
        bounds = [(0.02, None)] * 6 + [(-1.0, 1.0), (None, None)]
        res = minimize(obj, init_p, method="L-BFGS-B", bounds=bounds)
        self.weights, self.bias = res.x[:-1], float(res.x[-1])
        return self

    def predict_logit(self, Z: np.ndarray, max_cos: np.ndarray, frame_std: np.ndarray) -> np.ndarray:
        X = self._build_meta_features(Z, max_cos, frame_std)
        return (X @ self.weights + self.bias).astype(np.float32)

    def predict_proba(self, Z: np.ndarray, max_cos: np.ndarray, frame_std: np.ndarray) -> np.ndarray:
        logits = self.predict_logit(Z, max_cos, frame_std)
        return 1.0 / (1.0 + np.exp(-logits))
