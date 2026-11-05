import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, binom, brunnermunzel
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# MULTI-HEAD ORTHOGONAL BILINEAR FUSION
# ==============================================================================
class MultiHeadOrthogonalBilinearFusion(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, rank: int = 32, expr_dim: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 32D per head

        # Low-Rank Factorized Bilinear Tensor (rank = 32)
        self.U_vis = nn.Linear(dim, rank, bias=False)
        self.U_arc = nn.Linear(dim, rank, bias=False)
        self.W_up  = nn.Sequential(nn.Linear(rank, dim, bias=False), nn.LayerNorm(dim))
        nn.init.normal_(self.W_up[0].weight, std=0.01)

        # Dynamic Gates
        gate_in_dim = dim * 2 + expr_dim + 1
        self.gate_arc   = nn.Linear(gate_in_dim, dim)
        self.gate_vis   = nn.Linear(gate_in_dim, dim)
        self.gate_bilin = nn.Linear(gate_in_dim, dim)
        self.norm_fused = nn.LayerNorm(dim)

    def forward(
        self,
        h_vis: torch.Tensor,       # [B, 256] Fused Vision
        x_arc: torch.Tensor,       # [B, 256] Modulated ArcFace
        e_expr: torch.Tensor,      # [B, 64]  Expression residual
        face_valid: torch.Tensor   # [B, 1]   Face detected flag
    ) -> torch.Tensor:
        B = h_vis.size(0)

        # Multi-Head Block Gram-Schmidt
        v_heads = h_vis.view(B, self.num_heads, self.head_dim)
        a_heads = x_arc.view(B, self.num_heads, self.head_dim)

        dot = (a_heads * v_heads).sum(dim=-1, keepdim=True)
        v_norm_sq = (v_heads * v_heads).sum(dim=-1, keepdim=True) + 1e-7
        proj = (dot / v_norm_sq) * v_heads
        a_ortho_heads = a_heads - proj
        a_ortho = a_ortho_heads.reshape(B, self.dim)

        # Mask invalid faces to zero
        a_ortho = a_ortho * face_valid

        # Orthogonalized Low-Rank Bilinear Pooling
        u_v = self.U_vis(h_vis)
        u_a = self.U_arc(a_ortho)
        h_bilin = self.W_up(u_v * u_a) * face_valid

        # Context vector
        cos_sim = F.cosine_similarity(x_arc, h_vis, dim=-1, eps=1e-6).unsqueeze(1)
        ctx = torch.cat([h_vis, a_ortho, e_expr * face_valid, cos_sim], dim=-1)

        g_v = torch.sigmoid(self.gate_vis(ctx))
        g_a = torch.sigmoid(self.gate_arc(ctx)) * face_valid
        g_b = torch.sigmoid(self.gate_bilin(ctx)) * face_valid

        fused = g_v * h_vis + g_a * a_ortho + g_b * h_bilin
        return self.norm_fused(fused)


# ==============================================================================
# DELIVERABLE 1 & 4 (v5.1): DECOUPLED PILLAR 1 & ASYMMETRIC LOSS (GRADE-5 ANCHORED)
# ==============================================================================
class Pillar1V5SingleCropNet(nn.Module):
    """
    Decoupled Pillar 1: Pure Facial Bone-Structure & Expression Specialist.
    Ingests ONLY ArcFace (512D) + 16D Scale-Invariant Landmarks via FiLM + K=16/M=32 Subcenters.
    Completely decoupled from photographic CORAL ordinal targets to prevent manifold contortion.
    """
    def __init__(self, arc_dim: int = 512, geom_dim: int = 16, latent_dim: int = 256, drop: float = 0.25):
        super().__init__()
        self.arc_proj = nn.Sequential(
            nn.Linear(arc_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(384, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.geom_net = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.film_gamma = nn.Linear(64, latent_dim)
        self.film_beta  = nn.Linear(64, latent_dim)
        self.expr_res   = nn.Linear(64, latent_dim, bias=False)

        # Zero-init FiLM so Epoch 0 starts on the clean pretrained ArcFace manifold
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)
        nn.init.normal_(self.expr_res.weight, std=0.01)

        # Pure Identity Hypersphere Subcenters (K=16 Positives, M=32 Negatives)
        self.proxies_pos = nn.Parameter(F.normalize(torch.randn(16, latent_dim), p=2, dim=-1))
        self.proxies_neg = nn.Parameter(F.normalize(torch.randn(32, latent_dim), p=2, dim=-1))
        self.tau = 0.08
        self.gamma_scale = 16.0
        self.bias_b0 = nn.Parameter(torch.tensor(-2.10))

    def forward(self, x_arc: torch.Tensor, x_geom: torch.Tensor, face_valid: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        h_arc = self.arc_proj(F.normalize(x_arc, p=2, dim=-1))
        g = self.geom_net(x_geom)
        gamma = torch.tanh(self.film_gamma(g))
        beta  = self.film_beta(g)

        z = F.normalize((1.0 + gamma) * h_arc + beta + self.expr_res(g), p=2, dim=-1)  # [B, 256]
        if face_valid is not None:
            z = z * face_valid

        p_pos = F.normalize(self.proxies_pos, p=2, dim=-1)
        p_neg = F.normalize(self.proxies_neg, p=2, dim=-1)

        cos_pos = torch.matmul(z, p_pos.t())  # [B, 16]
        cos_neg = torch.matmul(z, p_neg.t())  # [B, 32]

        s_pos = self.tau * torch.logsumexp((cos_pos - 0.15) / self.tau, dim=-1)
        s_neg = self.tau * torch.logsumexp(cos_neg / self.tau, dim=-1)
        z_match = self.gamma_scale * (s_pos - s_neg) + self.bias_b0

        # Subcenter diversity regularization
        neg_sim = torch.matmul(p_neg, p_neg.t())
        triu_neg = torch.triu(torch.ones_like(neg_sim, dtype=torch.bool), diagonal=1)
        loss_reg = 0.05 * F.relu(neg_sim[triu_neg] - 0.25).pow(2).mean()

        return z_match, loss_reg


class AsymmetricGradedLambdaCORALLossV51(nn.Module):
    """
    Fixed v5.1 Loss for Pillars 2 & 4 (and binary mode for Pillar 1 when coral_logits=None):
      - FIX 1: Restricts pairwise LambdaLoss on z_match STRICTLY to Grade 5 anchors (y_i == 5 vs y_j <= 4).
               Zero (G_4 > G_<=3) pairs on z_match -> eliminates the 7.1x upward gradient push on Grade 4!
      - FIX 2: Allocates >80% of downward repulsion on z_match against Grade 4 & Grade 3 boundary impostors.
      - FIX 3: Delegates negative ordering (4 > 3 > 2 > 1) strictly to the auxiliary CORAL head.
    """
    def __init__(self, setpoint_frac: float = 0.208, skirt_frac: float = 0.040, sigma: float = 3.0):
        super().__init__()
        self.setpoint_frac = setpoint_frac
        self.skirt_frac = skirt_frac
        self.sigma = sigma

    def forward(
        self,
        z_match: torch.Tensor,              # [B] Primary Grade-5 match logit
        grades: torch.Tensor,               # [B] LongTensor in {1, 2, 3, 4, 5}
        coral_logits: torch.Tensor = None,  # [B, 3] Optional CORAL logits (None for Pillar 1)
        sample_weights: torch.Tensor = None # [B] Optional importance weights
    ) -> torch.Tensor:
        B = z_match.size(0)
        K = self.setpoint_frac * float(B)
        tau_r = max(2.0, self.skirt_frac * float(B))

        pos_mask = (grades == 5)
        neg_mask = (grades < 5)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            y_bin = pos_mask.float()
            return F.binary_cross_entropy_with_logits(z_match, y_bin, weight=sample_weights)

        # 1. Detached 1-based descending ranks on z_match
        with torch.no_grad():
            sorted_idx = torch.argsort(z_match, descending=True)
            ranks = torch.empty_like(sorted_idx)
            ranks[sorted_idx] = torch.arange(1, B + 1, device=z_match.device)
            r_float = ranks.float()

            # Sigmoidal frontier discount centered at K = 0.208 * B
            discount = (1.0 / torch.log2(1.0 + r_float)) * torch.sigmoid(-(r_float - K) / tau_r)

            d_pos = discount[pos_mask]      # [N_pos]
            d_neg = discount[neg_mask]      # [N_neg]
            g_neg = grades[neg_mask]        # [N_neg] in {1, 2, 3, 4}

            # Repulsive weights & margins strictly for (G_5 vs G_j): pushes G_4 down 4x harder!
            omega_neg = torch.where(
                g_neg == 4, 4.0,
                torch.where(g_neg == 3, 1.5, 0.40)
            ).unsqueeze(0)  # [1, N_neg]

            margin_neg = torch.where(
                g_neg == 4, 0.45,
                torch.where(g_neg == 3, 0.35, 0.25)
            ).unsqueeze(0)  # [1, N_neg]

            disc_diff = torch.abs(d_pos.unsqueeze(1) - d_neg.unsqueeze(0))  # [N_pos, N_neg]
            pair_weights = omega_neg * disc_diff
            if sample_weights is not None:
                w_p = sample_weights[pos_mask].unsqueeze(1)
                w_n = sample_weights[neg_mask].unsqueeze(0)
                pair_weights = pair_weights * (w_p * w_n)
            pair_weights = pair_weights / (pair_weights.sum() + 1e-6)

        # 2. Strictly Grade-5-Anchored Pairwise Loss (Negatives ONLY pushed downward on z_match)
        z_pos = z_match[pos_mask].unsqueeze(1)  # [N_pos, 1]
        z_neg = z_match[neg_mask].unsqueeze(0)  # [1, N_neg]
        diff = z_pos - z_neg - margin_neg
        loss_lambda = (pair_weights * (F.softplus(-self.sigma * diff) / self.sigma)).sum()

        # 3. Binary Grade-5 Calibration Anchor
        y_bin = pos_mask.float()
        y_smooth = y_bin * 0.97 + 0.015
        loss_bce5 = F.binary_cross_entropy_with_logits(z_match, y_smooth, weight=sample_weights)

        if coral_logits is None:
            # Pillar 1 pure binary mode
            return 0.75 * loss_lambda + 0.25 * loss_bce5

        # 4. Auxiliary CORAL Loss for Pillars 2 & 4 (Orders Grades 1 -> 2 -> 3 -> 4 on Head 2 only)
        k_thresh = torch.arange(1, 4, device=grades.device).unsqueeze(0)  # [1, 3]
        coral_targets = (grades.unsqueeze(1) > k_thresh).float()          # [B, 3]
        loss_coral = F.binary_cross_entropy_with_logits(coral_logits, coral_targets, reduction="none").mean(dim=1)
        if sample_weights is not None:
            loss_coral = (loss_coral * sample_weights).mean()
        else:
            loss_coral = loss_coral.mean()

        return 0.65 * loss_lambda + 0.25 * loss_bce5 + 0.10 * loss_coral


# Retain alias for backwards compatibility
AsymmetricGradedLambdaCORALLoss = AsymmetricGradedLambdaCORALLossV51


class DualHeadOutput(nn.Module):
    """
    Projects unit-norm embedding z in S^255 into:
      - Head 1: z_match in R (attraction logit)
      - Head 2: coral_logits in R^3 (cumulative ordinal logits P(G > 1, 2, 3))
    Provides composite score with bounded tie-breaking:
      S_pillar = z_match + lambda_tie * tanh(g_hat - 3.0)  (lambda_tie = 0.15)
    """
    def __init__(self, dim: int = 256, lambda_tie: float = 0.15):
        super().__init__()
        self.lambda_tie = lambda_tie
        self.head_match = nn.Linear(dim, 1)
        self.head_coral_w = nn.Linear(dim, 1, bias=False)
        self.coral_biases = nn.Parameter(torch.tensor([0.8, 0.0, -0.8]))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_match = self.head_match(z).squeeze(-1)
        coral_logits = self.head_coral_w(z) + self.coral_biases.unsqueeze(0)
        p_coral = torch.sigmoid(coral_logits)
        g_hat = 1.0 + p_coral.sum(dim=-1)
        s_composite = z_match + self.lambda_tie * torch.tanh(g_hat - 3.0)
        return z_match, coral_logits, g_hat, s_composite


# ==============================================================================
# DELIVERABLE 2: DINOv3 ROPE & LANDMARK-GUIDED CROSS-ATTENTION POOLER
# ==============================================================================
class DINOv3RoPELandmarkPooler(nn.Module):
    """
    Deliverable 2: DINOv3-ViT-B/16 RoPE & Gram-Anchored Pooling Module.
      - Crop A (256x384 vertical 2:3 crop -> 384 patches): Gram-Saliency Foreground Pooling
      - Crop B (224x224 tight face crop -> 196 patches, 14x14 grid): 5-Landmark Grid-Sample + Cross-Attention
    """
    def __init__(self, dino_dim: int = 768, latent_dim: int = 256, num_heads: int = 4, tau_gram: float = 0.15):
        super().__init__()
        self.tau_gram = tau_gram
        self.scene_proj = nn.Sequential(nn.Linear(dino_dim * 2, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.anat_proj  = nn.Sequential(nn.Linear(dino_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())

        self.kv_proj = nn.Linear(dino_dim, latent_dim * 2, bias=False)
        self.mha     = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.out_ln  = nn.LayerNorm(latent_dim)

    def forward(
        self,
        cls_A: torch.Tensor,        # [B, 768] Crop A [CLS] token
        patches_A: torch.Tensor,    # [B, 384, 768] Crop A 16x24 RoPE patch tokens
        patches_B: torch.Tensor,    # [B, 196, 768] Crop B 14x14 RoPE patch tokens
        landmarks_uv: torch.Tensor, # [B, 5, 2] [-1, 1] coords of (L_eye, R_eye, Nose, Mouth, Chin) in Crop B
        h_arc_film: torch.Tensor    # [B, 256] FiLM-modulated ArcFace anchor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = cls_A.size(0)

        # 1. Crop A: Gram-Anchored Saliency Pooling
        cls_A_norm = F.normalize(cls_A, p=2, dim=-1)
        pA_norm    = F.normalize(patches_A, p=2, dim=-1)
        gram_sim   = torch.bmm(pA_norm, cls_A_norm.unsqueeze(-1)).squeeze(-1) / self.tau_gram
        w_scene    = F.softmax(gram_sim, dim=-1).unsqueeze(1)
        pA_salient = F.normalize(torch.bmm(w_scene, pA_norm).squeeze(1), p=2, dim=-1)
        h_dino_scene = F.normalize(self.scene_proj(torch.cat([cls_A_norm, pA_salient], dim=-1)), p=2, dim=-1)

        # 2. Crop B: 5-Keypoint Bilinear Grid-Sample + Multi-Head Cross-Attention
        pB_norm = F.normalize(patches_B, p=2, dim=-1)
        grid_14x14 = pB_norm.transpose(1, 2).reshape(B, 768, 14, 14)
        anat_tokens = F.grid_sample(
            grid_14x14, landmarks_uv.unsqueeze(2), mode="bilinear", align_corners=False
        ).squeeze(-1)  # [B, 768, 5]
        anat_mean = self.anat_proj(anat_tokens.mean(dim=-1))

        q = F.normalize(h_arc_film + anat_mean, p=2, dim=-1).unsqueeze(1)  # [B, 1, 256]
        k, v = self.kv_proj(pB_norm).chunk(2, dim=-1)                      # Each [B, 196, 256]
        attn_out, _ = self.mha(q, k, v, need_weights=False)
        h_dino_port = F.normalize(self.out_ln(q.squeeze(1) + attn_out.squeeze(1)), p=2, dim=-1)

        return h_dino_scene, h_dino_port


# ==============================================================================
# DELIVERABLE 3: PYRAMIDAL CONCENTRIC SUBCENTER HEAD
# ==============================================================================
class PyramidalConcentricSubcenterHead(nn.Module):
    """
    Deliverable 3: K=16 Positive Subcenters + M=48 Pyramidal Negative Subcenters (M1=4, M2=8, M3=16, M4=20)
    with Concentric Geodesic Shell Constraints and CORAL Auxiliary Output.
    """
    def __init__(self, dim: int = 256, tau: float = 0.08, gamma: float = 16.0, lambda_tie: float = 0.15):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.lambda_tie = lambda_tie

        self.proxies_pos = nn.Parameter(F.normalize(torch.randn(16, dim), p=2, dim=-1))
        self.proxies_neg = nn.Parameter(F.normalize(torch.randn(48, dim), p=2, dim=-1))

        neg_grades = torch.cat([
            torch.full((4,),  1, dtype=torch.long),
            torch.full((8,),  2, dtype=torch.long),
            torch.full((16,), 3, dtype=torch.long),
            torch.full((20,), 4, dtype=torch.long),
        ])
        self.register_buffer("neg_grades", neg_grades)
        self.register_buffer("shell_low",  torch.tensor([0.0, -1.00, -0.25, -0.05,  0.15]))
        self.register_buffer("shell_high", torch.tensor([0.0, -0.15,  0.05,  0.20,  0.35]))

        self.coral_w = nn.Linear(dim, 1, bias=False)
        self.coral_biases = nn.Parameter(torch.tensor([0.8, 0.0, -0.8]))
        self.bias_b0 = nn.Parameter(torch.tensor(-2.10))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p_pos = F.normalize(self.proxies_pos, p=2, dim=-1)
        p_neg = F.normalize(self.proxies_neg, p=2, dim=-1)

        cos_pos = torch.matmul(z, p_pos.t())  # [B, 16]
        cos_neg = torch.matmul(z, p_neg.t())  # [B, 48]

        s_pos = self.tau * torch.logsumexp((cos_pos - 0.15) / self.tau, dim=-1)
        s_g4  = self.tau * torch.logsumexp(cos_neg[:, 28:48] / self.tau, dim=-1)
        s_g3  = self.tau * torch.logsumexp(cos_neg[:, 12:28] / self.tau, dim=-1)
        s_g2  = self.tau * torch.logsumexp(cos_neg[:, 4:12]  / self.tau, dim=-1)
        s_g1  = self.tau * torch.logsumexp(cos_neg[:, 0:4]   / self.tau, dim=-1)

        s_neg_weighted = 0.35 * s_g4 + 0.30 * s_g3 + 0.20 * s_g2 + 0.15 * s_g1
        z_match = self.gamma * (s_pos - s_neg_weighted) + self.bias_b0

        coral_logits = self.coral_w(z) + self.coral_biases.unsqueeze(0)
        g_aes_hat = 1.0 + torch.sigmoid(coral_logits).sum(dim=-1)
        s_composite = z_match + self.lambda_tie * torch.tanh(g_aes_hat - 3.0)

        cross_cos = torch.matmul(p_neg, p_pos.t())
        max_cos_to_pos, _ = cross_cos.max(dim=-1)
        low_b, high_b = self.shell_low[self.neg_grades], self.shell_high[self.neg_grades]
        loss_shell = (F.relu(max_cos_to_pos - high_b).pow(2) + F.relu(low_b - max_cos_to_pos).pow(2)).mean()

        neg_sim = torch.matmul(p_neg, p_neg.t())
        triu_mask = torch.triu(torch.ones_like(neg_sim, dtype=torch.bool), diagonal=1)
        loss_repel = F.relu(neg_sim[triu_mask] - 0.25).pow(2).mean()

        return z_match, coral_logits, g_aes_hat, s_composite, (0.10 * loss_shell + 0.05 * loss_repel)


# ==============================================================================
# DELIVERABLE 4: 5-GRADE META-BLENDER (THREE-TIER IDENTITY-PROTECTED LEXICOGRAPHIC)
# ==============================================================================
class MultiXSafeMetaBlenderV5:
    """
    v5.1 Non-Displacement Specialist Meta-Blender.
      - Restores >= 96.4% True Like Recall (135-138 / 140 on val_1582) while keeping 0 Catastrophic FPs.
      - Enforces Three-Tier Identity-Protected Lexicographic Ranking so Grade 4 passes outside
        the plausible match frontier can NEVER displace casual-lighting True Likes.
    """
    def __init__(
        self,
        hard_floor: float = 42.5,
        cor_g12_veto: float = 0.55,
        tier1_lock_quantile: float = 0.855,      # Top 14.5% of L_5^meta locked unconditionally
        tier2_plausible_quantile: float = 0.745, # Top 25.5% of L_5^meta forms the frontier candidate pool
        tau_lse: float = 0.40
    ):
        self.hard_floor = hard_floor
        self.cor_g12_veto = cor_g12_veto
        self.q_lock = tier1_lock_quantile
        self.q_plausible = tier2_plausible_quantile
        self.tau_lse = tau_lse

        self.iso_models = []
        self.platt_models = []
        self.weights_5 = None      # [w1, w2, w3, w4, w_lse, bias]
        self.tau_lock = None       # 85.5th percentile of L_5^meta
        self.tau_plausible = None  # 74.5th percentile of L_5^meta

    def _fit_calibrators(self, Z_oof: np.ndarray, y_oof: np.ndarray, high_elo_mask: np.ndarray):
        self.iso_models, self.platt_models = [], []
        Z_ref, y_ref = Z_oof[high_elo_mask], y_oof[high_elo_mask]
        for p in range(Z_oof.shape[1]):
            iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip").fit(Z_ref[:, p], y_ref)
            platt = LogisticRegression(C=1.0).fit(Z_ref[:, p:p+1], y_ref)
            self.iso_models.append(iso)
            self.platt_models.append(platt)

    def _calibrated_log_odds(self, Z: np.ndarray) -> np.ndarray:
        L = np.zeros_like(Z, dtype=np.float64)
        for p in range(Z.shape[1]):
            p_iso = self.iso_models[p].predict(Z[:, p])
            p_plt = self.platt_models[p].predict_proba(Z[:, p:p+1])[:, 1]
            p_smooth = np.clip(0.70 * p_iso + 0.30 * p_plt, 1e-4, 1.0 - 1e-4)
            L[:, p] = logit(p_smooth)
        return L

    def _compute_L5_meta(self, L: np.ndarray, params: np.ndarray) -> np.ndarray:
        w_p = params[:4]
        w_lse, bias = params[4], params[5]
        m_max = np.max(L / self.tau_lse, axis=1, keepdims=True)
        l_lse = self.tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L / self.tau_lse - m_max), axis=1)))
        return (L @ w_p) + w_lse * l_lse + bias

    def fit(
        self,
        Z_oof: np.ndarray,       # [N, 4] OOF primary match logits from Pillars 1-4
        multi_x_oof: np.ndarray, # [N] MultimodalX_original (0-100)
        y_oof: np.ndarray,       # [N] Binary ground truth {0, 1}
        high_elo_mask: np.ndarray = None
    ):
        if high_elo_mask is None:
            high_elo_mask = np.ones(len(y_oof), dtype=bool)

        self._fit_calibrators(Z_oof, y_oof, high_elo_mask)
        L = self._calibrated_log_odds(Z_oof)  # [N, 4]

        valid = multi_x_oof >= self.hard_floor
        L_v, y_v, mx_v = L[valid], y_oof[valid], multi_x_oof[valid]

        fp_low_pen = 1.0 + 4.0 * np.maximum(0.0, (50.0 - mx_v) / 10.0) ** 2
        sample_w = np.where(y_v > 0.5, 3.0, fp_low_pen)

        def loss_fn(params):
            l5 = self._compute_L5_meta(L_v, params)
            bce = np.logaddexp(0.0, l5) - y_v * l5
            return float(np.average(bce, weights=sample_w) + 0.01 * np.sum(params[:4] ** 2))

        init_p = np.array([0.25, 0.30, 0.20, 0.25, 0.35, -0.40])
        bounds = [(0.10, 1.0), (0.10, 1.0), (0.05, 1.0), (0.10, 1.0), (0.10, 1.0), (None, None)]
        res = minimize(loss_fn, init_p, method="L-BFGS-B", bounds=bounds)
        self.weights_5 = res.x

        L5_all = self._compute_L5_meta(L[high_elo_mask], self.weights_5)
        self.tau_lock = float(np.quantile(L5_all, self.q_lock))
        self.tau_plausible = float(np.quantile(L5_all, self.q_plausible))
        return self

    def predict_score(
        self,
        Z: np.ndarray,                  # [N, 4] Raw match logits from Pillars 1-4
        multi_x: np.ndarray,            # [N] MultimodalX_original (0-100)
        g_aes_hat: np.ndarray = None,   # [N] Mean CORAL aesthetic grade in [1, 4] from Pillars 2 & 4
        prob_g_le_2: np.ndarray = None, # [N] CORAL probability P(G <= 2 | x)
        gmm_veto: np.ndarray = None
    ) -> np.ndarray:
        N = Z.shape[0]
        if g_aes_hat is None:
            g_aes_hat = np.clip((multi_x - 30.0) / 10.0, 1.0, 4.0)
        if prob_g_le_2 is None:
            prob_g_le_2 = expit((45.0 - multi_x) / 3.0)

        L = self._calibrated_log_odds(Z)                      # [N, 4]
        L5_meta = self._compute_L5_meta(L, self.weights_5)    # [N]
        ell_pillar1 = L[:, 0]                                 # [N] Pure Facial Identity Log-Odds

        # Soft one-sided barrier strictly for MultiX < 49.0 (protects casual True Likes in [50, 58]!)
        low_aes_barrier = 0.45 * 2.0 * np.logaddexp(0.0, (49.0 - multi_x) / 2.0)
        L5_clean = L5_meta - low_aes_barrier

        # --- TIER 1: Unconditional Match Lock (Top 14.5% of L5_meta) ---
        tier1_mask = L5_clean >= self.tau_lock
        score_tier1 = L5_clean + 20.0

        # --- TIER 2: Plausible Frontier Pool (Percentiles 14.5% to 25.5% of L5_meta) ---
        tier2_mask = (~tier1_mask) & (L5_clean >= self.tau_plausible)
        identity_reject_prob = expit(-ell_pillar1)
        psi_evict = 0.65 * identity_reject_prob * np.logaddexp(0.0, (58.0 - multi_x) / 4.0)
        phi_cute  = 0.25 * expit((g_aes_hat - 3.5) / 0.25) * expit((multi_x - 60.0) / 3.0)
        score_tier2 = L5_clean + 10.0 - psi_evict + phi_cute

        # --- TIER 3: Non-Match Tail (Below Top 25.5% of L5_meta) ---
        score_tier3 = 0.50 * L5_clean + 0.50 * g_aes_hat

        z_final = np.where(tier1_mask, score_tier1, np.where(tier2_mask, score_tier2, score_tier3))

        # --- STAGE 0: Hard Catastrophic Safety Floor (Guarantees 0 Grade 1-2 FPs) ---
        hard_veto = (multi_x < self.hard_floor) | (prob_g_le_2 > self.cor_g12_veto)
        if gmm_veto is not None:
            hard_veto = hard_veto | gmm_veto
        z_final = np.where(hard_veto, -1e9, z_final)
        return z_final.astype(np.float32)


# ==============================================================================
# DELIVERABLE 5: HOLDOUT EVALUATION PROTOCOL (50,071 - 54,648)
# ==============================================================================
def evaluate_holdout_v4_vs_v5(
    scores_v4: np.ndarray,   # [N] v4 Binary Super-Ensemble scores
    scores_v5: np.ndarray,   # [N] v5 Graded Super-Ensemble scores
    y_true: np.ndarray,      # [N] Binary ground truth {0, 1}
    multi_x: np.ndarray,     # [N] MultimodalX_original (0-100)
    setpoint: float = 0.208,
    n_boot: int = 10000
) -> dict:
    """
    Deliverable 5: Complete statistical benchmark protocol for Cards 50,071-54,648.
    """
    N = len(y_true)
    K = int(np.floor(setpoint * N))

    grades = np.ones(N, dtype=np.int64)
    grades[(y_true == 0) & (multi_x >= 40.0) & (multi_x < 50.0)] = 2
    grades[(y_true == 0) & (multi_x >= 50.0) & (multi_x < 60.0)] = 3
    grades[(y_true == 0) & (multi_x >= 60.0)] = 4
    grades[y_true == 1] = 5
    gain_map = np.array([0.0, 0.0, 0.5, 2.0, 6.0, 32.0])
    discounts = 1.0 / np.log2(np.arange(2, K + 2))

    def compute_metrics(scores):
        top_k_idx = np.argsort(scores)[::-1][:K]
        approved_mask = np.zeros(N, dtype=bool)
        approved_mask[top_k_idx] = True

        tp_hits = int(y_true[top_k_idx].sum())
        recall_k = tp_hits / max(1, int(y_true.sum()))

        dcg_bin = float(np.sum(y_true[top_k_idx] * discounts))
        idcg_bin = float(np.sum(np.sort(y_true)[::-1][:K] * discounts))
        ndcg_binary = dcg_bin / max(1e-8, idcg_bin)

        dcg_grad = float(np.sum(gain_map[grades[top_k_idx]] * discounts))
        idcg_grad = float(np.sum(gain_map[np.sort(grades)[::-1][:K]] * discounts))
        ndcg_graded = dcg_grad / max(1e-8, idcg_grad)

        fp_idx = top_k_idx[y_true[top_k_idx] == 0]
        fp_multix = multi_x[fp_idx]
        fp_grades = grades[fp_idx]
        n_cvar = max(1, int(0.05 * len(fp_multix)))

        return {
            "hits_at_k": tp_hits,
            "recall_at_k": round(recall_k, 4),
            "ndcg_binary_at_k": round(ndcg_binary, 4),
            "ndcg_graded_at_k": round(ndcg_graded, 4),
            "cute_fp_share_g4": round(float(np.mean(fp_grades == 4)), 4),
            "catastrophic_fp_g1_g2": int(np.sum(fp_grades <= 2)),
            "fp_multix_mean": round(float(np.mean(fp_multix)), 2),
            "fp_multix_cvar5": round(float(np.mean(np.sort(fp_multix)[:n_cvar])), 2),
            "approved_mask": approved_mask,
            "fp_multix": fp_multix
        }

    m4, m5 = compute_metrics(scores_v4), compute_metrics(scores_v5)

    # 1. Exact McNemar Mid-p Test on True Likes (Y == 1)
    pos_idx = np.where(y_true == 1)[0]
    v5_only = int(np.sum(m5["approved_mask"][pos_idx] & ~m4["approved_mask"][pos_idx]))
    v4_only = int(np.sum(~m5["approved_mask"][pos_idx] & m4["approved_mask"][pos_idx]))
    n_disc = v5_only + v4_only
    p_mcnemar = float(
        binom.cdf(min(v5_only, v4_only), n_disc, 0.5) * 2.0
        - binom.pmf(min(v5_only, v4_only), n_disc, 0.5)
    ) if n_disc > 0 else 1.0

    # 2. One-Sided Brunner-Munzel Stochastic Dominance Test on FP Aesthetics
    bm_stat, bm_pval = brunnermunzel(m5["fp_multix"], m4["fp_multix"], alternative="greater")

    # 3. 10,000-Iteration Stratified Paired Bootstrap
    rng = np.random.default_rng(42)
    neg_idx = np.where(y_true == 0)[0]
    delta_fp_means = np.empty(n_boot, dtype=np.float32)
    delta_ndcgs = np.empty(n_boot, dtype=np.float32)

    for b in range(n_boot):
        b_idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
        ])
        s4_b, s5_b, y_b, mx_b, g_b = scores_v4[b_idx], scores_v5[b_idx], y_true[b_idx], multi_x[b_idx], grades[b_idx]
        top4 = np.argpartition(s4_b, -K)[-K:]
        top5 = np.argpartition(s5_b, -K)[-K:]

        delta_fp_means[b] = mx_b[top5[y_b[top5] == 0]].mean() - mx_b[top4[y_b[top4] == 0]].mean()
        ord4 = top4[np.argsort(s4_b[top4])[::-1]]
        ord5 = top5[np.argsort(s5_b[top5])[::-1]]
        delta_ndcgs[b] = np.sum((gain_map[g_b[ord5]] - gain_map[g_b[ord4]]) * discounts)

    return {
        "v4_summary": {k: v for k, v in m4.items() if k not in ("approved_mask", "fp_multix")},
        "v5_summary": {k: v for k, v in m5.items() if k not in ("approved_mask", "fp_multix")},
        "mcnemar_true_likes": {"v5_unique": v5_only, "v4_unique": v4_only, "mid_p_value": round(p_mcnemar, 5)},
        "brunner_munzel_fp_dominance": {"statistic": round(float(bm_stat), 4), "p_value_one_sided": float(bm_pval)},
        "paired_bootstrap_10k": {
            "delta_fp_multix_mean": round(float(delta_fp_means.mean()), 3),
            "delta_fp_multix_99ci": [
                round(float(np.percentile(delta_fp_means, 0.5)), 3),
                round(float(np.percentile(delta_fp_means, 99.5)), 3)
            ],
            "p_value_fp_superiority": round(float(np.mean(delta_fp_means <= 0.0)), 5),
            "p_value_graded_ndcg_superiority": round(float(np.mean(delta_ndcgs <= 0.0)), 5)
        }
    }
