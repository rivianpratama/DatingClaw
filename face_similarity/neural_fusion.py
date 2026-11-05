from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContextGatedBilinearNet(nn.Module):
    """
    Context-Gated Low-Rank Bilinear Fusion Network for multimodal personal preference.
    Supports dynamic visual embedding dimensions (512 for CLIP ViT-B/32, 1152 for SigLIP-384).
    """
    def __init__(
        self,
        face_dim: int = 512,
        vis_dim: int = 512,
        meta_dim: int = 4,
        memory_dim: int = 4,
        latent_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. Modality Bottleneck Projections
        self.proj_face = nn.Sequential(
            nn.Linear(face_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_vis = nn.Sequential(
            nn.Linear(vis_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Dynamic Context Gating (conditioned on face, scene, and quality metadata)
        gate_input_dim = latent_dim * 2 + meta_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        # 3. Final Classification Head with Factorized Bilinear + Gated Branches + Memory Stats
        # Fused representation: [g * h_face || (1-g) * h_vis || h_face * h_vis || memory_stats]
        classifier_input_dim = latent_dim * 3 + memory_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_face: torch.Tensor,
        x_vis: torch.Tensor,
        x_meta: torch.Tensor,
        x_memory: torch.Tensor,
    ) -> torch.Tensor:
        # Project modalities into shared latent space
        h_face = self.proj_face(x_face)
        h_vis = self.proj_vis(x_vis)

        # Dynamic sample-specific gate g in (0, 1)^d
        gate_in = torch.cat([h_face, h_vis, x_meta], dim=-1)
        g = self.gate_net(gate_in)

        # Factorized bilinear interaction (Hadamard product)
        h_bilinear = h_face * h_vis

        # Gated fusion
        h_gated_face = g * h_face
        h_gated_vis = (1.0 - g) * h_vis

        # Concatenate fused visual representations with non-parametric k-NN memory features
        fused = torch.cat([h_gated_face, h_gated_vis, h_bilinear, x_memory], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits


class SessionPairwiseSmoothedLoss(nn.Module):
    """
    Composite objective:
    L_total = alpha * L_BCE_smooth + (1 - alpha) * L_Pairwise_BPR
    with sample weights w_i and pairwise weights w_i * w_j for noise down-weighting.
    """
    def __init__(self, alpha: float = 0.5, label_smoothing: float = 0.025):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
        session_pairs: list[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        # 1. Label-Smoothed Binary Cross Entropy with sample weights
        smooth_labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_raw = F.binary_cross_entropy_with_logits(logits, smooth_labels, reduction="none")
        weighted_bce = (bce_raw * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)

        if not session_pairs:
            return weighted_bce

        # 2. Session-Local Pairwise Ranking Loss (Bradley-Terry / BPR)
        # Pair (i, j) where i is a Like (1) and j is a Pass (0) within the same session
        idx_pos = torch.tensor([p[0] for p in session_pairs], dtype=torch.long, device=logits.device)
        idx_neg = torch.tensor([p[1] for p in session_pairs], dtype=torch.long, device=logits.device)

        logits_pos = logits[idx_pos]
        logits_neg = logits[idx_neg]
        pair_weights = sample_weights[idx_pos] * sample_weights[idx_neg]

        # Log-loss: ln(1 + exp(-(z_i - z_j))) = softplus(-(z_i - z_j))
        diff = logits_pos - logits_neg
        bpr_raw = F.softplus(-diff)
        weighted_bpr = (bpr_raw * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)

        return self.alpha * weighted_bce + (1.0 - self.alpha) * weighted_bpr
