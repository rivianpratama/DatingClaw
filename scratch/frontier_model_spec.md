<USER_REQUEST>
Execute

| RFC Question | Selected Architectural Strategy | Key Mathematical / Operational Payoff |
| --- | --- | --- |
| **Q1: Non-Stationary Drift (Era 3 ELO Drop)** | **Class-Conditional Odds-Ratio + `MultiX` Hardness Importance Weighting + Two-Stage High-ELO Calibration** | Separates *human selectivity drift* (Era 1 $\to$ 2) from *platform covariate shift* (Era 3 `5.23%` ELO penalty); preserves 100% of the 4,899 True Likes while preventing 13.4k low-ELO Era 3 passes from warping the prior. |
| **Q2: Negative Subcenter Scaling ($M$)** | **Option 2B: Dual-Hierarchical Stratified Spherical $K$-Means ($K = 16$ Pos, $M = 48$ Neg: $28\text{ Hard} + 20\text{ Easy}$)** | Prevents unconstrained $K$-Means from wasting negative centroids on Era 3 junk; concentrates 28 negative proxies directly along the $S_{\text{MultiX}} \ge 50$ boundary with Log-Det cosine repulsion. |
| **Q3: Truncated `LambdaLoss@20.8%**` | **Hybrid Sigmoidal-Attenuated Top-$K$ DCG ($K = \lfloor 0.208 B \rfloor$, $\tau_r = 0.04 B$, $\sigma = 3.0$)** | Concentrates peak gradient steepness $-\frac{dG}{dr}$ directly across the 17%–24% rank frontier while eliminating gradient starvation for positives sitting just outside rank $K$. |
| **Q4: Orthogonal Feature Gating** | **Option 4B + 4C Unified: Multi-Head Block Gram-Schmidt ($H = 8 \times 32\text{D}$) + Orthogonalized Low-Rank Bilinear Pooling** | Removes 8 independent semantic dimensions of identity-aesthetic collinearity per sample with zero extra projection parameters, plus a rank-32 multiplicative synergy branch. |
| **Q5: Meta-Blender Validation & Calibration** | **Connected-Component Identity-Group ($\cos \ge 0.80$) + Era-Stratified 5-Fold CV with Smooth Platt-Isotonic Calibration** | Eliminates duplicate-face leakage across folds and removes step-function ties at the `20.8%` cutline while anchoring calibration to the High-ELO (Era 1–2) prior. |
| **Q6: Pre-ViT Geometric Veto** | **Yeo-Johnson 4-Component Positive GMM Mahalanobis Envelope + Split-Conformal 99.95% Bound** | Captures multimodal valid poses (frontal, $\pm 3/4$ turn, open laugh vs. closed smile) in `0.18 ms`, guaranteeing $\le 0.05\%$ False Negative rate while rejecting ~14% of junk cards before ViT inference. |
| **Q7: `MultimodalX_original` Asymmetric Floor** | **Unified Hard Floor ($\theta_{\text{hard}} = 42.5$) + Logit-Space Softplus Barrier ($\theta_{\text{soft}} = 50.0$) + Utility-Weighted Meta-Loss** | Guarantees **zero** catastrophic FPs ($S_{\text{MultiX}} < 42.5$), crushes $[42.5, 50.0]$ passes with an exponential logit penalty, rescues the 2.1% high-consensus true likes in $[43, 50]$, and preserves the exact `20.8%` ELO setpoint. |

---

## Question 1: Handling Severe Non-Stationary Chronological Drift (The Era 3 ELO Drop)

### 1.1 Mathematical Diagnosis: Platform Covariate Shift vs. Human Concept Drift

The transition from **Era 1 (`12.45%` likes)** and **Era 2 (`10.36%` likes)** into **Era 3 (`5.23%` likes, dropping to `2.0%–3.0%` in bins `45k–49k`)** is not a change in your internal aesthetic preference function $P(Y=1 \mid \mathbf{x})$. Rather, because the bot briefly swiped at `28.73%`, Bumble's queue algorithm demoted the account's ELO and shifted the **marginal input distribution $P_{\text{Era3}}(\mathbf{x})$** heavily toward low-attractiveness profiles (**Covariate Shift**).

This creates two traps in standard re-weighting options:

1. **Why Pure Option 1A ($w_i = \bar{p}_{\text{target}} / p_{\text{deck}}(t_i)$ on all samples) Backfires:** Multiplying both likes and passes in Era 3 (`45k–49k`) by $0.115 / 0.025 = 4.6\times$ amplifies the gradient of **13,452 bottom-tier ELO passes** by up to $4.6\times$, swamping the decision boundary with easy negatives.
2. **Why Option 1C (Exponential Recency Discounting) Fails:** Down-weighting Era 1 (`0–21k`) discards **2,629 high-ELO True Likes** (`53.7%` of all positive training examples in your entire 50k corpus).

### 1.2 The Solution: Class-Conditional Odds-Ratio Balancing + `MultiX` Hardness Weighting

To recover the true stationary preference distribution $P_{\text{High-ELO}}(Y \mid \mathbf{x})$ while extracting full value from all **4,899 True Likes**, apply **Class-Conditional Importance Weighting** coupled with **Two-Stage High-ELO Calibration (Option 1B)**:

1. **Positive Samples ($y_i = 1$, $N = 4,899$): Retain Full Weight ($w_i^+ \ge 1.0$).**
Every True Like found during the Era 3 drought ($N = 729$) is a genuine attractive profile that passed your human visual bar. Assign positive weights:

$$w_i^+ = \begin{cases} 0.90, & t_i \in [0, 21120] \text{ (Era 1: slightly generous)} \\ 1.00, & t_i \in [21121, 35821] \text{ (Era 2: target selectivity)} \\ 1.15, & t_i \in [35822, 50003] \text{ (Era 3: high-selectivity survivors)} \end{cases}$$


2. **Negative Samples ($y_i = 0$, $N = 45,104$): Deck-Odds Prior Correction $\times$ `MultimodalX` Hardness Importance.**
Let $p_{\text{deck}}(t_i)$ be the 1,500-card rolling human like rate at card index $t_i$ (clipped to $[0.025, 0.140]$), and let $\bar{p}_{\text{target}} = 0.112$ be the healthy high-ELO base rate. To correct the negative-to-positive odds ratio in Era 3 without losing hard boundary negatives, weight each pass $i$ by:

$$r_{\text{odds}}(t_i) = \min\left(1.0,\; \frac{p_{\text{deck}}(t_i)}{1 - p_{\text{deck}}(t_i)} \cdot \frac{1 - \bar{p}_{\text{target}}}{\bar{p}_{\text{target}}}\right)$$


$$w_i^-(t_i, \mathbf{x}_i) = r_{\text{odds}}(t_i) \cdot \left(0.20 + 0.80 \cdot \sigma\left(\frac{S_{\text{MultiX}}(\mathbf{x}_i) - 48.0}{6.0}\right)\right)$$


* **How this works in Era 3 ($p_{\text{deck}} = 0.028$):** $r_{\text{odds}} = 0.228$. If an Era 3 pass is an obvious low-ELO reject ($S_{\text{MultiX}} = 32$), its weight drops to $0.228 \times 0.205 = \mathbf{0.047}$, preventing the flood of degraded ELO cards from distorting the network. Conversely, if an Era 3 pass is a **true near-miss hard negative** ($S_{\text{MultiX}} = 62$), its hardness gate is $0.93$, preserving an effective weight of $0.21$ so boundary discrimination stays razor-sharp.


3. **Two-Stage High-ELO Calibration (Option 1B Integration):**
* **Stage 1 (Representation & Subcenter Training):** Train Pillars 1–4 across all 50,003 profiles using the importance weights $w_i$ above. In `StratifiedSwipeSampler`, sample negative passes with probability proportional to $w_i^-(t_i, \mathbf{x}_i)$ so every mini-batch of $B = 512$ contains high-utility boundary negatives at a stationary $1:7$ positive-to-negative ratio.
* **Stage 2 (Prior Anchor):** Fit the final probability calibration and PID threshold reference `val_scores` strictly on out-of-fold predictions from **Era 2 (`21,121–35,821`, $10.36\%$ base rate) + Era 3 cards resampled to an $11.0\%$ positive rate**, guaranteeing that the deployed model is calibrated to a healthy high-ELO queue rather than the `5.23%` Era 3 trough.



---

## Question 2: Scaling Negative Subcenter Proxies ($M$) for 45,104 Negative Samples

### 2.1 Why Option 2B ($K = 16$ Positives, $M = 48$ Stratified Negatives) Is Optimal

If you run unconstrained Spherical $K$-Means with $M = 48$ across all 45,104 passes (Option 2A), Lloyd's objective minimizes within-cluster cosine variance over the densest regions of the pass distribution. Because `49.2%` of passes have $S_{\text{MultiX}} < 50.0$ (and Era 3 added 13,452 low-ELO passes), unconstrained $K$-Means allocates ~34 of the 48 negative proxies to deep junk/blur/low-ELO clusters far from the decision boundary, leaving only ~14 proxies to guard the critical 20.8% attraction frontier.

**Recommended Configuration (Option 2B Enhanced):**

* **Positive Subcenters ($K = 16$):** Scale from $K = 12 \to \mathbf{K = 16}$ ($\sim 306$ positive likes per archetype across 4,899 Likes).
* **Two-Tier Stratified Negative Subcenters ($M = 48 = 28_{\text{hard}} + 20_{\text{easy}}$):**
Partition the 45,104 negative embeddings in the warm-started `256D` hypersphere $\mathbb{S}^{255}$ into two disjoint strata before running Spherical $K$-Means:
1. **Stratum 1 — Boundary / Near-Miss Negatives ($\mathcal{N}_{\text{hard}}: y_i = 0 \land S_{\text{MultiX}}(\mathbf{x}_i) \ge 50.0$, ~22,913 samples):** Seed **$M_{\text{hard}} = 28$ proxies** ($\mathbf{C}_{\text{hard}}^- \in \mathbb{R}^{28 \times 256}$). These represent clean, well-lit portraits of women whose facial geometry or aesthetic vibe does not match your preference.
2. **Stratum 2 — Structural / Technical Failure Modes ($\mathcal{N}_{\text{easy}}: y_i = 0 \land S_{\text{MultiX}}(\mathbf{x}_i) < 50.0$, ~22,191 samples):** Seed **$M_{\text{easy}} = 20$ proxies** ($\mathbf{C}_{\text{easy}}^- \in \mathbb{R}^{20 \times 256}$). These capture dark indoor shots, group photos, bad angles, heavy filters, and Era 3 low-ELO cards.



### 2.2 Anti-Collapse Regularization & Dead-Proxy Revival for $M = 48$

To prevent $M = 48$ negative proxies from collapsing into redundant directions on $\mathbb{S}^{255}$ during Riemannian AdamW updates:

1. **Truncated Cosine Repulsion on $\mathbf{C}^-$:** Penalize any pair of negative subcenters whose angular separation falls below $\arccos(0.25) \approx 75.5^\circ$:

$$\mathcal{L}_{\text{neg\_repel}}(\mathbf{C}^-) = \frac{2}{M(M-1)} \sum_{1 \le j < l \le 48} \max\big(0,\, \cos(\mathbf{c}_j^-, \mathbf{c}_l^-) - 0.25\big)^2$$


2. **SoftTriple Adaptive Merging on $\mathbf{C}^+$:** Keep the singularity-clamped attraction tether on the $K = 16$ positive proxies so over-allocated positive subcenters merge smoothly:

$$\mathcal{L}_{\text{pos\_merge}}(\mathbf{C}^+) = \frac{2}{K(K-1)} \sum_{1 \le k < l \le 16} \sqrt{\max\big(2.0 - 2.0\cos(\mathbf{c}_k^+, \mathbf{c}_l^+),\, 10^{-6}\big)}$$


3. **Epoch-Boundary Dead-Proxy Revival (Simplified Option 2C):** Track an exponential moving average of the top-1 assignment frequency $f_j^-$ for each negative proxy $j \in \{1..48\}$. At the end of Epochs 2 and 4, if any proxy has $f_j^- < \frac{0.15}{48}$, replace $\mathbf{c}_j^-$ with a perturbed copy of the most heavily populated hard-negative proxy $\mathbf{c}_{j^*}^-$ ($\text{L2Norm}(\mathbf{c}_{j^*}^- + \mathcal{N}(0, 0.05\mathbf{I}))$).

---

## Question 3: Truncated LambdaLoss@K Optimization at the 20.8% Production Operating Point

### 3.1 Resolving the Hard-Truncation Gradient Trap

In pure **Option 3A (Hard Truncation $\mathbb{I}[\min(\pi(i), \pi(j)) \le K]$)**, any positive profile $i$ that starts an epoch ranked at $\pi(i) = 65$ in a batch of $B = 256$ ($K = 53$) receives **zero gradient** from swapping with negative profiles ranked at $\pi(j) \in \{54, \dots, 64\}$. It cannot climb through the immediate buffer zone below rank $K$ via local swaps. Conversely, pure **Option 3B** (replacing the logarithmic DCG discount entirely with a flat sigmoid) treats Rank #1 and Rank #50 identically.

### 3.2 Exact Formulation: Frontier-Steepened Sigmoidal DCG Discount $G_{20.8\%}(r)$

Combine logarithmic DCG discount with a **Sigmoidal Frontier Gate** centered exactly at the $20.8\%$ cutoff rank $K = 0.208 B$ with transition width $\tau_r = 0.04 B$:

$$G_{20.8\%}(r) = \frac{1}{\log_2(1 + r)} \cdot \underbrace{\frac{1}{1 + \exp\left(\frac{r - 0.208 B}{0.04 B}\right)}}_{w_{\text{frontier}}(r)}, \qquad r \in \{1, 2, \dots, B\}$$

Look at how the pairwise weight $\vert{}\Delta G_{ij}\vert{} = \vert{}G_{20.8\%}(\pi(i)) - G_{20.8\%}(\pi(j))\vert{}$ behaves across a batch of $B = 512$ ($K = 106.5$, $\tau_r = 20.5$):

* **Inside the Top 15% ($r \le 75$):** $w_{\text{frontier}}(r) \in [0.82, 0.99]$, preserving standard logarithmic precision-at-the-top prioritization.
* **Across the 20.8% Decision Frontier ($r \in [85, 130]$, i.e., $16.6\% \to 25.4\%$):** Because $w_{\text{frontier}}(r)$ drops steeply from $0.74 \to 0.24$ across these 45 ranks, **pairs $(i, j)$ that straddle the 20.8% cutoff receive the largest $\vert{}\Delta G_{ij}\vert{}$ gradient multiplier in the entire batch**.
* **Deep Pass Tail ($r > 175$, below the top 34%):** $w_{\text{frontier}}(r) < 0.03$, silencing gradients among irrelevant passes in the bottom two-thirds of the deck.

### 3.3 Temperature $\sigma$ & Complete PyTorch Implementation

For unit-hypersphere subcenter logits scaled to $[-4, +4]$, set **$\sigma = 3.0$** with pairwise margin **$m = 0.15$**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrontierLambdaLoss208(nn.Module):
    """
    Optimizes NDCG & Recall strictly around the 20.8% Bumble ELO operating point.
    Combines logarithmic DCG discount with a sigmoidal frontier gate centered at K = 0.208 * B.
    """
    def __init__(
        self,
        setpoint_frac: float = 0.208,
        skirt_frac: float = 0.040,
        sigma: float = 3.0,
        margin: float = 0.15,
        bce_weight: float = 0.20
    ):
        super().__init__()
        self.setpoint_frac = setpoint_frac
        self.skirt_frac = skirt_frac
        self.sigma = sigma
        self.margin = margin
        self.bce_weight = bce_weight

    def forward(
        self,
        logits: torch.Tensor,          # [B] pure visual logits
        targets: torch.Tensor,         # [B] binary {0, 1}
        sample_weights: torch.Tensor = None  # [B] importance weights from Question 1
    ) -> torch.Tensor:
        B = logits.size(0)
        K = self.setpoint_frac * float(B)      # 106.5 for B=512 (53.2 for B=256)
        tau_r = max(2.0, self.skirt_frac * float(B))  # 20.5 for B=512

        # 1. Compute detached 1-based descending ranks r in {1, ..., B}
        with torch.no_grad():
            sorted_idx = torch.argsort(logits, descending=True)
            ranks = torch.empty_like(sorted_idx)
            ranks[sorted_idx] = torch.arange(1, B + 1, device=logits.device)
            r_float = ranks.float()

            # Sigmoidal-Attenuated DCG Gain G_20.8%(r)
            dcg_discount = 1.0 / torch.log2(1.0 + r_float)
            frontier_gate = torch.sigmoid(-(r_float - K) / tau_r)
            G_r = dcg_discount * frontier_gate  # [B]

            # Ideal normalization factor IDCG@K for stability across batch sizes
            num_pos = int((targets > 0.5).sum().item())
            if num_pos > 0:
                ideal_r = torch.arange(1, num_pos + 1, device=logits.device, dtype=torch.float32)
                idcg = (1.0 / torch.log2(1.0 + ideal_r) * torch.sigmoid(-(ideal_r - K) / tau_r)).sum().clamp(min=1e-5)
            else:
                idcg = torch.tensor(1.0, device=logits.device)

        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return F.binary_cross_entropy_with_logits(logits, targets, weight=sample_weights)

        s_pos, G_pos = logits[pos_mask], G_r[pos_mask]  # [N_pos]
        s_neg, G_neg = logits[neg_mask], G_r[neg_mask]  # [N_neg]

        # 2. |Delta NDCG_ij| = |G(pi(i)) - G(pi(j))| / IDCG  -> shape [N_pos, N_neg]
        delta_ndcg = torch.abs(G_pos.unsqueeze(1) - G_neg.unsqueeze(0)) / idcg

        # 3. Pairwise logistic margin loss: (1 / sigma) * ln(1 + exp(-sigma * (s_i - s_j - m)))
        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0) - self.margin
        pair_loss = delta_ndcg * (F.softplus(-self.sigma * diff) / self.sigma)

        if sample_weights is not None:
            w_pos = sample_weights[pos_mask].unsqueeze(1)
            w_neg = sample_weights[neg_mask].unsqueeze(0)
            pair_loss = pair_loss * (w_pos * w_neg)

        loss_lambda = pair_loss.sum() / pos_mask.sum().float()

        # 4. Calibrated Label-Smoothed BCE Anchor (20% weight)
        y_smooth = targets * 0.97 + 0.015
        loss_bce = F.binary_cross_entropy_with_logits(logits, y_smooth, weight=sample_weights)

        return (1.0 - self.bce_weight) * loss_lambda + self.bce_weight * loss_bce

```

---

## Question 4: Gram-Schmidt Orthogonal Feature Gating Formulation at 50k Scale

### 4.1 Why Multi-Head Block Orthogonalization (Option 4B) + Orthogonalized Low-Rank Bilinear (Option 4C) Beats 1D Projection

In `256D` space, a single scalar Gram-Schmidt projection removes only **1 out of 256 degrees of freedom** between $\mathbf{h}_{\text{arc}}$ and $\mathbf{h}_{\text{vision}}$, allowing multi-dimensional confounders (e.g., head angle correlating with selfie framing, or skin contrast correlating with indoor lighting) to leak identity bias into the visual anchor. Meanwhile, **Option 4A** ($\mathbf{I} - \mathbf{U}_k \mathbf{U}_k^\top$) projects away from a *static* global subspace rather than the specific visual features present in candidate image $\mathbf{x}_i$.

### 4.2 Exact Mathematical Formulation: $H = 8$ Block Gram-Schmidt + Orthogonalized Rank-$32$ Bilinear Pooling

1. **Multi-Head Subspace Partition ($H = 8$ Heads $\times 32\text{D} = 256\text{D}$):**
Split the unit-normalized visual anchor $\hat{\mathbf{h}}_{\text{vis}} \in \mathbb{S}^{255}$ and FiLM-modulated identity vector $\mathbf{h}_{\text{arc}} \in \mathbb{S}^{255}$ into $H = 8$ semantic blocks $\mathbf{v}^{(h)}, \mathbf{a}^{(h)} \in \mathbb{R}^{32}$ ($h = 1 \dots 8$). Orthogonalize each block independently:

$$\mathbf{a}^{(h)\perp} = \mathbf{a}^{(h)} - \frac{\langle \mathbf{a}^{(h)}, \mathbf{v}^{(h)} \rangle}{\Vert{}\mathbf{v}^{(h)}\Vert{}_2^2 + 10^{-6}} \mathbf{v}^{(h)}, \qquad \mathbf{h}_{\text{arc}}^{\perp} = \Big[\mathbf{a}^{(1)\perp} \;\big\Vert{}\; \dots \;\big\Vert{}\; \mathbf{a}^{(8)\perp}\Big] \in \mathbb{R}^{256}$$



This eliminates **8 sample-specific dimensions of collinearity** with **zero extra parameters**.
2. **Low-Rank Bilinear Interaction (Rank $r = 32$) Projected Orthogonal to $\hat{\mathbf{h}}_{\text{vis}}$:**
Capture second-order conditional preference interactions (e.g., specific jawline/eye spacing in $\mathbf{h}_{\text{arc}}$ paired with specific hairstyle/aesthetic vibe in $\hat{\mathbf{h}}_{\text{vis}}$):

$$\mathbf{b}_{\text{raw}} = \mathbf{W}_{\text{up}} \Big( \text{GELU}(\mathbf{U}_v \hat{\mathbf{h}}_{\text{vis}}) \odot \text{GELU}(\mathbf{U}_a \mathbf{h}_{\text{arc}}) \Big), \qquad \mathbf{h}_{\text{bilin}}^{\perp} = \mathbf{b}_{\text{raw}} - \langle \mathbf{b}_{\text{raw}}, \hat{\mathbf{h}}_{\text{vis}} \rangle \hat{\mathbf{h}}_{\text{vis}}$$


3. **Pythagorean Norm Guarantee:**

$$\mathbf{z}_{\text{fused}} = \text{L2Norm}\Big( \hat{\mathbf{h}}_{\text{vis}} + \boldsymbol{\alpha}_1 \odot \mathbf{h}_{\text{arc}}^{\perp} + \boldsymbol{\alpha}_2 \odot \mathbf{h}_{\text{bilin}}^{\perp} \Big) \in \mathbb{S}^{255}$$



Because both residual branches are strictly orthogonal to $\hat{\mathbf{h}}_{\text{vis}}$, $\Vert{}\hat{\mathbf{h}}_{\text{vis}} + \boldsymbol{\delta}^\perp\Vert{}_2^2 = 1 + \Vert{}\boldsymbol{\delta}^\perp\Vert{}_2^2 \ge 1$. **It is mathematically impossible for ArcFace or the bilinear branch to cancel or reduce the primary visual aesthetic signal.**

```python
class MultiHeadOrthogonalBilinearFusion(nn.Module):
    """
    Combines H=8 Block Gram-Schmidt Orthogonalization (Option 4B)
    with Orthogonalized Low-Rank Bilinear Pooling (Option 4C, r=32).
    """
    def __init__(self, dim: int = 256, num_heads: int = 8, rank: int = 32, expr_dim: int = 64):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 32D per head

        # Option 4C: Low-Rank Factorized Bilinear Tensor (rank = 32)
        self.U_vis = nn.Linear(dim, rank, bias=False)
        self.U_arc = nn.Linear(dim, rank, bias=False)
        self.W_up  = nn.Sequential(nn.Linear(rank, dim, bias=False), nn.LayerNorm(dim))
        nn.init.normal_(self.W_up[0].weight, std=0.01)

        # Dynamic Gates conditioned on vision, orthogonal identity, and 64D expression
        gate_in_dim = dim * 2 + expr_dim + 1
        self.gate_arc   = nn.Linear(gate_in_dim, dim)
        self.gate_bilin = nn.Linear(gate_in_dim, dim)
        nn.init.constant_(self.gate_arc.bias, -1.5)
        nn.init.constant_(self.gate_bilin.bias, -2.0)

        self.scale_arc   = nn.Parameter(torch.full((1, dim), 0.25))
        self.scale_bilin = nn.Parameter(torch.full((1, dim), 0.15))

    def forward(
        self,
        h_vis: torch.Tensor,     # [B, 256] unit-normalized visual base anchor
        h_arc: torch.Tensor,     # [B, 256] unit-normalized FiLM-modulated ArcFace
        e_expr: torch.Tensor,    # [B, 64] landmark expression embedding
        face_valid: torch.Tensor # [B, 1] {1.0 or 0.0}
    ) -> torch.Tensor:
        B = h_vis.size(0)
        h_vis = F.normalize(h_vis, p=2, dim=-1)
        h_arc = F.normalize(h_arc, p=2, dim=-1) * face_valid

        # 1. Option 4B: H=8 Multi-Head Block Gram-Schmidt Orthogonalization
        v_heads = h_vis.view(B, self.num_heads, self.head_dim)  # [B, 8, 32]
        a_heads = h_arc.view(B, self.num_heads, self.head_dim)  # [B, 8, 32]

        dot_hv = (a_heads * v_heads).sum(dim=-1, keepdim=True)  # [B, 8, 1]
        norm_v = (v_heads * v_heads).sum(dim=-1, keepdim=True) + 1e-6
        a_ortho_heads = a_heads - (dot_hv / norm_v) * v_heads
        h_arc_ortho = a_ortho_heads.reshape(B, self.dim) * face_valid  # [B, 256]

        # 2. Option 4C: Low-Rank Bilinear Interaction projected orthogonal to h_vis
        b_raw = self.W_up(F.gelu(self.U_vis(h_vis)) * F.gelu(self.U_arc(h_arc))) * face_valid
        b_dot = (b_raw * h_vis).sum(dim=-1, keepdim=True)
        h_bilin_ortho = (b_raw - b_dot * h_vis) * face_valid  # [B, 256]

        # 3. Gated Orthogonal Residual Fusion on S^255
        gate_ctx = torch.cat([h_vis, h_arc_ortho, e_expr, face_valid], dim=-1)
        alpha_1 = self.scale_arc * torch.sigmoid(self.gate_arc(gate_ctx)) * face_valid
        alpha_2 = self.scale_bilin * torch.sigmoid(self.gate_bilin(gate_ctx)) * face_valid

        return F.normalize(h_vis + alpha_1 * h_arc_ortho + alpha_2 * h_bilin_ortho, p=2, dim=-1)

```

---

## Question 5: Meta-Blender Cross-Validation & Out-of-Fold Calibration

### 5.1 Connected-Component Identity-Group + Era-Stratified 5-Fold CV (Option 5A + 5B Hybrid)

Pure Chronological Walk-Forward (Option 5B: testing on `42k–50k`) evaluates your ensemble inside the **Era 3 ELO trough (`2%–5%` likes)**, whereas ungrouped K-Fold leaks duplicate profiles across folds. Combine **Option 5A** with **Era Stratification**:

1. **Identity + Temporal Session Connected Components:**
Run an exact cosine similarity search over all 50,003 `ArcFace` embeddings $\mathbf{X}_{\text{arc}} \in \mathbb{R}^{50003 \times 512}$. Form an undirected graph where an edge $(i, j)$ exists if:
* $\cos(\mathbf{x}_{\text{arc}, i}, \mathbf{x}_{\text{arc}, j}) \ge 0.80$ (catches repeat profiles of the same woman across different outfits/lighting), **OR**
* $\vert{}t_i - t_j\vert{} \le 15\text{ minutes}$ (keeps contiguous swiping sessions together so session-level threshold drift never leaks across folds).


2. **Stratified Group Assignment:**
Extract connected components (`scipy.sparse.csgraph.connected_components`) to assign `cluster_id`. Propagate the maximum label per cluster and assign clusters to 5 folds via `StratifiedGroupKFold(n_splits=5)` stratified jointly on `strata_key = 2 * era_id + label`.

### 5.2 Strict-Monotone Platt-Isotonic Hybrid Calibration (Zero Ties at the 20.8% Cutline)

* **Why Pure Isotonic Regression Fails at a Quantile Setpoint:** Isotonic regression fits a piecewise-constant step function. Around the 79.2nd percentile (`20.8%` swipe setpoint), dozens of distinct profiles get mapped to the exact same step value $\hat{P}_p(\mathbf{x}) = c$, creating arbitrary tie-breaking!
* **Why Pure Platt / Temperature Scaling Fails:** A 2-parameter sigmoid cannot correct asymmetric tail skew in subcenter metric logits.
* **The Solution (Smooth Monotone Log-Odds Calibrator):** For each pillar $p \in \{1..4\}$, fit both **Platt Scaling** $\sigma(a_p z + b_p)$ and **Isotonic Regression** $\text{Iso}_p(z)$ on the OOF predictions of the **High-ELO Reference Strata (Eras 1 & 2)**, blend their probabilities $70/30$ to guarantee strict monotonicity ($z_1 > z_2 \implies \hat{P}_p(z_1) > \hat{P}_p(z_2)$ with zero flat ties), and feed **calibrated log-odds** $\hat{\ell}_p(\mathbf{x}) = \text{logit}\big(0.70\,\text{Iso}_p(z) + 0.30\,\sigma(a_p z + b_p)\big)$ into the LogSumExp Meta-Blender.

---

## Question 6: Pre-ViT Geometric Vetoes: Mahalanobis Ellipsoid vs. Box Cutoffs

### 6.1 Why a Single Unimodal Mahalanobis Ellipsoid (Option 6A) Fails on 16D Landmarks

Several of the 16 geometric features are **multimodal or heavily skewed** among attractive photos:

* **Head Yaw $\sin(\psi_{\text{yaw}})$** has three distinct aesthetic modes: frontal portrait ($0^\circ$), left $3/4$ over-the-shoulder pose ($-30^\circ$), and right $3/4$ pose ($+30^\circ$).
* **Mouth Aspect Ratio ($\text{MAR}$)** is bimodal: closed-lip warm smile ($\text{MAR} \approx 0.03$) vs. open-teeth Duchenne laugh ($\text{MAR} \approx 0.32$).
Fitting a single unimodal covariance matrix $\boldsymbol{\Sigma}_+$ across bimodal modes places high density in the empty valley between modes while inflating variance and cutting off valid tail poses. Meanwhile, **Option 6C (Linear SVM)** only learns a flat hyperplane and cannot enclose a bounded manifold.

### 6.2 The Solution: Yeo-Johnson 4-Mode Positive GMM Mahalanobis Envelope + Split-Conformal Zero-FN Bound (<0.2ms)

1. **Coordinate Gaussianization:** Fit a monotonic **Yeo-Johnson Power Transform** $\tilde{\mathbf{g}} = \Psi(\mathbf{g}) \in \mathbb{R}^{16}$ on the 4,899 True Likes so skewed features ($\text{MAR}$, framing ratio, $s_{\text{det}}$) become symmetric and bell-shaped.
2. **Multi-Mode Positive Manifold ($K_g = 4$ Gaussians):** Fit a 4-component Gaussian Mixture Model $\{\boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m\}_{m=1}^4$ (with shrinkage regularization $\boldsymbol{\Sigma}_m \leftarrow \boldsymbol{\Sigma}_m + 0.02\mathbf{I}_{16}$) strictly on the 4,400 in-fold True Likes. Define the **Minimum Component Mahalanobis Distance**:

$$D_{\text{GMM}}^2(\mathbf{g}) = \min_{m \in \{1..4\}} \big(\Psi(\mathbf{g}) - \boldsymbol{\mu}_m\big)^\top \boldsymbol{\Sigma}_m^{-1} \big(\Psi(\mathbf{g}) - \boldsymbol{\mu}_m\big)$$


3. **Conformal Zero-FN Threshold Calibration:** Instead of relying on the theoretical $\chi^2_{16, 0.999} = 39.25$ table value, evaluate $D_{\text{GMM}}^2(\mathbf{g})$ on the out-of-fold True Likes ($N = 499$) and set the conformal cutoff at the **99.95th empirical quantile** multiplied by a $1.05$ safety buffer:

$$\tau_{\text{conf}} = 1.05 \cdot \text{Quantile}_{0.9995}\Big(\big\{D_{\text{GMM}}^2(\mathbf{g}_i) : y_i = 1\big\}\Big)$$



```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import GaussianMixture

class ConformalGMMGeometricVeto:
    """
    Executes in <0.2ms per candidate on CPU.
    Guarantees <= 0.05% False Negative rate on True Likes while rejecting ~13-15% of obvious passes.
    """
    def __init__(self, n_components: int = 4, reg_covar: float = 0.02, safety_mult: float = 1.05):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.safety_mult = safety_mult
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)
        self.means = None       # [4, 16]
        self.precisions = None  # [4, 16, 16] pre-inverted Sigma_m^{-1}
        self.tau_conf = None

    def fit(self, G_pos_train: np.ndarray, G_pos_cal: np.ndarray):
        # G_pos_train: [N_pos_train, 16], G_pos_cal: [N_pos_cal, 16] (True Likes only)
        G_tr_norm = self.pt.fit_transform(G_pos_train)
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.reg_covar,
            n_init=5,
            random_state=42
        ).fit(G_tr_norm)

        self.means = gmm.means_.astype(np.float32)            # [4, 16]
        self.precisions = gmm.precisions_.astype(np.float32)  # [4, 16, 16]

        # Conformal calibration on held-out positive likes
        cal_scores = self.mahalanobis_min_sq(G_pos_cal)
        self.tau_conf = float(np.quantile(cal_scores, 0.9995) * self.safety_mult)
        return self

    def mahalanobis_min_sq(self, G: np.ndarray) -> np.ndarray:
        single = (G.ndim == 1)
        if single:
            G = G[None, :]
        G_norm = self.pt.transform(G).astype(np.float32)      # [N, 16]
        diff = G_norm[:, None, :] - self.means[None, :, :]    # [N, 4, 16]
        # Vectorized quadratic form: (g - mu_m)^T Sigma_m^{-1} (g - mu_m) in 0.15ms
        d_sq = np.einsum("nki,kij,nkj->nk", diff, self.precisions, diff)  # [N, 4]
        min_d_sq = np.min(d_sq, axis=1)
        return float(min_d_sq[0]) if single else min_d_sq

    def should_veto(self, g_16d: np.ndarray) -> tuple[bool, float]:
        # Hard physical bounds + Conformal 4-GMM Mahalanobis Ellipsoid
        det_score = float(g_16d[15])
        cos_yaw   = float(g_16d[3])  # cos(58 deg) = 0.5299
        d_sq = self.mahalanobis_min_sq(g_16d)
        veto = (det_score < 0.35) or (cos_yaw < 0.50) or (d_sq > self.tau_conf)
        return veto, d_sq

```

---

## Question 7: `MultimodalX_original` as an Asymmetric Negative Floor & Fail-Safe Reranker

### 7.1 Mathematical Synthesis: Why Unifying Options 7A + 7B (in Log-Odds Space) + 7C Is Strictly Optimal

Look at the empirical boundary distribution of $S_{\text{MultiX}} \in [0, 100]$:

* **Zone 1 ($S_{\text{MultiX}} < 42.5$):** Contains **$0.0\%$ of True Likes** and **$\sim 31\%$ of Passes**.
* **Zone 2 ($42.5 \le S_{\text{MultiX}} < 50.0$):** Contains **$2.1\%$ of True Likes** (~3 likes out of 140 on `val_1582`) and **$\sim 18.2\%$ of Passes**.
* **Zone 3 ($S_{\text{MultiX}} \ge 50.0$):** Contains **$97.9\%$ of True Likes** and **$50.8\%$ of Passes**.

If you use a flat **Hard Floor at $\theta_{\text{floor}} = 50.0$ (Option 7A alone)**, you permanently lose the `2.1%` of True Likes in Zone 2 (catching at most `97.9%` even with an oracle ranker). Conversely, if you use **Option 7B** as a raw multiplication $S_{\text{meta}} \cdot \sigma(\dots)$ when $S_{\text{meta}} \in (-\infty, +\infty)$ is a signed logit, multiplying a negative logit by $\sigma \to 0$ pushes it *upward toward zero*!

In log-odds/logit space, a multiplicative probability gate $P_{\text{final}} = P_{\text{meta}} \cdot \sigma\big(\frac{S_{\text{MultiX}} - \theta_{\text{soft}}}{T}\big)$ corresponds to an **Additive Softplus Barrier Penalty**:

$$Z_{\text{final}}(\mathbf{x}) = \begin{cases} -\infty \quad (\text{100\% Hard Veto}), & \text{if } S_{\text{MultiX}}(\mathbf{x}) < 42.5 \\ S_{\text{meta}}(\mathbf{x}) - \lambda_{\text{barrier}} \cdot T \ln\left(1 + \exp\left(\frac{51.0 - S_{\text{MultiX}}(\mathbf{x})}{T}\right)\right) + \beta_{\text{anchor}} \left(\frac{S_{\text{MultiX}}(\mathbf{x}) - 65.0}{15.0}\right), & \text{if } S_{\text{MultiX}}(\mathbf{x}) \ge 42.5 \end{cases}$$

### 7.2 How This 3-Zone Equation Behaves Across Every Regime

With barrier stiffness $\lambda_{\text{barrier}} = 0.35$, temperature $T = 2.0$, and tie-break anchor $\beta_{\text{anchor}} = 0.20$:

1. **Zone 1 ($S_{\text{MultiX}} < 42.5$ — Hard Veto Floor, Option 7A):** Immediately returns $-\infty$ (Hard Pass). Guarantees **zero** bottom-tier profiles ever enter the swipe queue with **$0.0\%$ True Like loss**.
2. **Zone 2 ($S_{\text{MultiX}} \in [42.5, 51.0]$ — Exponential Softplus Barrier, Option 7B in Logit Space):**
* At $S_{\text{MultiX}} = 45.0$, the barrier subtracts **$-2.13$ logits** from $S_{\text{meta}}(\mathbf{x})$.
* Any mediocre false positive that one neural pillar accidentally scored moderately high ($S_{\text{meta}} \approx +1.2$) is crushed down to $-0.93$ and completely rejected from the top 20.8% bucket.
* However, if a candidate is one of the rare **`2.1%` True Likes** in $[43, 50]$ where the high-res `DINOv2 + SigLIP` portrait specialists see an overwhelmingly attractive photo ($S_{\text{meta}} \ge +3.4$, top $1\%$ neural confidence), she survives the $-2.13$ barrier ($Z_{\text{final}} \ge +1.27$) and is rescued!


3. **Zone 3 ($S_{\text{MultiX}} \ge 55.0$ — Clean Approval Slice Reranking, Option 7D):**
* For $S_{\text{MultiX}} \ge 58.0$, the softplus barrier vanishes ($\approx 0.00$), and the mild linear anchor $+\beta_{\text{anchor}}\left(\frac{S_{\text{MultiX}} - 65}{15}\right)$ acts as a lexicographic tie-breaker among borderline candidates near the 20.8% cutline—automatically preferring a candidate with $S_{\text{MultiX}} = 72$ over a borderline candidate with $S_{\text{MultiX}} = 53$.



### 7.3 Training the Specialist Meta-Blender with Asymmetric FP Utility Cost (Option 7C)

To align the Meta-Blender's learned pillar weights $\mathbf{w}$ directly with your human utility function, weight each negative sample $i \in \mathcal{N}$ during Out-of-Fold meta-training by an asymmetric cost multiplier that penalizes low-`MultiX` false positives up to **$6\times$ harder**:

$$w_{\text{meta}, i} = \begin{cases} 1.0, & \text{if } y_i = 1 \text{ (True Like)} \\ 1.0 + 5.0 \cdot \max\left(0,\, \frac{55.0 - S_{\text{MultiX}}(\mathbf{x}_i)}{15.0}\right)^2, & \text{if } y_i = 0 \text{ (True Pass)} \end{cases}$$

### 7.4 Drop-In Production `MultiXSafeMetaBlender` & 20.8% ELO PID Integration

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class MultiXSafeMetaBlender:
    """
    50k Milestone Specialist Meta-Blender with:
      1. Smooth Platt-Isotonic Hybrid OOF Calibration (Zero step-function ties)
      2. LogSumExp Specialist Max-Voting across Pillars 1-4
      3. Asymmetric Cost-Sensitive Meta-Loss (Option 7C: 6x penalty on low-MultiX FPs)
      4. Hard Floor @ MultiX < 42.5 (Option 7A) + Softplus Logit Barrier @ MultiX < 51.0 (Option 7B)
    """
    def __init__(
        self,
        hard_floor: float = 42.5,
        soft_barrier_knee: float = 51.0,
        barrier_temp: float = 2.0,
        tau_lse: float = 0.40,
        fp_cost_gamma: float = 5.0
    ):
        self.hard_floor = hard_floor
        self.soft_knee = soft_barrier_knee
        self.barrier_temp = barrier_temp
        self.tau_lse = tau_lse
        self.fp_cost_gamma = fp_cost_gamma

        self.iso_models = []
        self.platt_models = []
        self.meta_params = None  # [w1, w2, w3, w4, w_lse, lambda_barrier, beta_anchor, bias]

    def _fit_calibrators(self, Z_oof: np.ndarray, y_oof: np.ndarray, high_elo_mask: np.ndarray):
        """Fit smooth 70% Isotonic + 30% Platt calibrators anchored on High-ELO strata (Eras 1 & 2)."""
        self.iso_models, self.platt_models = [], []
        Z_ref, y_ref = Z_oof[high_elo_mask], y_oof[high_elo_mask]

        for p in range(Z_oof.shape[1]):
            iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
            iso.fit(Z_ref[:, p], y_ref)
            platt = LogisticRegression(C=1.0)
            platt.fit(Z_ref[:, p:p+1], y_ref)
            self.iso_models.append(iso)
            self.platt_models.append(platt)

    def _calibrated_log_odds(self, Z: np.ndarray) -> np.ndarray:
        L = np.zeros_like(Z, dtype=np.float64)
        for p in range(Z.shape[1]):
            p_iso = self.iso_models[p].predict(Z[:, p])
            p_plt = self.platt_models[p].predict_proba(Z[:, p:p+1])[:, 1]
            # 70/30 blend guarantees strict monotonicity with zero flat ties at 20.8% cutoff
            p_smooth = np.clip(0.70 * p_iso + 0.30 * p_plt, 1e-4, 1.0 - 1e-4)
            L[:, p] = logit(p_smooth)
        return L

    def _compute_softplus_barrier(self, multi_x: np.ndarray) -> np.ndarray:
        # T * ln(1 + exp((51.0 - S_MultiX) / T))
        u = (self.soft_knee - multi_x) / self.barrier_temp
        return self.barrier_temp * np.logaddexp(0.0, u)

    def fit(
        self,
        Z_oof: np.ndarray,          # [N, 4] OOF raw scores from Pillars 1-4
        multi_x_oof: np.ndarray,    # [N] MultimodalX_original (0-100 scale)
        y_oof: np.ndarray,          # [N] binary {0, 1}
        high_elo_mask: np.ndarray = None
    ):
        if high_elo_mask is None:
            high_elo_mask = np.ones(len(y_oof), dtype=bool)

        self._fit_calibrators(Z_oof, y_oof, high_elo_mask)
        L = self._calibrated_log_odds(Z_oof)  # [N, 4]

        # LogSumExp Specialist Max-Voting across the 4 calibrated log-odds
        m_max = np.max(L / self.tau_lse, axis=1, keepdims=True)
        l_lse = self.tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L / self.tau_lse - m_max), axis=1)))

        barrier_feat = self._compute_softplus_barrier(multi_x_oof)
        anchor_feat  = (multi_x_oof - 65.0) / 15.0

        # Option 7C: Asymmetric FP Cost Weighting (penalizes FPs with MultiX < 55 up to 6x)
        fp_penalty = 1.0 + self.fp_cost_gamma * np.maximum(0.0, (55.0 - multi_x_oof) / 15.0) ** 2
        sample_w = np.where(y_oof > 0.5, 2.5, fp_penalty)

        # Optimize only on non-hard-vetoed samples (MultiX >= 42.5)
        valid = multi_x_oof >= self.hard_floor
        L_v, lse_v, bar_v, anc_v = L[valid], l_lse[valid], barrier_feat[valid], anchor_feat[valid]
        y_v, w_v = y_oof[valid], sample_w[valid]

        def loss_fn(params):
            w_p = params[:4]
            w_lse, lam_bar, beta_anc, bias = params[4], params[5], params[6], params[7]
            s_meta = L_v @ w_p + w_lse * lse_v
            z_final = s_meta - lam_bar * bar_v + beta_anc * anc_v + bias
            bce = np.logaddexp(0.0, z_final) - y_v * z_final
            return float(np.average(bce, weights=w_v) + 0.01 * np.sum(w_p ** 2))

        init_params = np.array([0.15, 0.35, 0.15, 0.35, 0.30, 0.35, 0.20, -0.50])
        bounds = [
            (0.02, 1.0), (0.05, 1.0), (0.02, 1.0), (0.05, 1.0), # Non-negative pillar weights
            (0.05, 1.0),                                         # w_lse specialist rescue
            (0.20, 1.50),                                        # lambda_barrier (enforces MultiX < 51 suppression)
            (0.05, 0.60),                                        # beta_anchor (MultiX tie-breaker)
            (None, None)                                         # bias
        ]
        res = minimize(loss_fn, init_params, method="L-BFGS-B", bounds=bounds)
        self.meta_params = res.x
        return self

    def predict_score(self, Z: np.ndarray, multi_x: np.ndarray) -> np.ndarray:
        """
        Returns final continuous ranking score Z_final.
        Profiles with MultiX < 42.5 receive -1e9 (guaranteed Hard Pass).
        """
        L = self._calibrated_log_odds(Z)
        m_max = np.max(L / self.tau_lse, axis=1, keepdims=True)
        l_lse = self.tau_lse * (m_max.squeeze(1) + np.log(np.sum(np.exp(L / self.tau_lse - m_max), axis=1)))

        barrier_feat = self._compute_softplus_barrier(multi_x)
        anchor_feat  = (multi_x - 65.0) / 15.0

        w_p = self.meta_params[:4]
        w_lse, lam_bar, beta_anc, bias = self.meta_params[4:]
        z_final = (L @ w_p) + w_lse * l_lse - lam_bar * barrier_feat + beta_anc * anchor_feat + bias

        # Option 7A: Hard Cascade Veto Floor at MultiX < 42.5 (0.0% FN rate on True Likes)
        z_final = np.where(multi_x < self.hard_floor, -1e9, z_final)
        return z_final.astype(np.float32)

```

### 7.5 How the `MultiX` Floor Interacts with the 20.8% PID Controller in Production

In `BumbleELOPIDController` (with setpoint `target_rate = 0.208`, `target_q = 0.792`, and rolling 100-card ceiling `< 21`):

1. Whenever a card fails the **Pre-ViT Conformal GMM Veto** *or* fails the **Hard Floor ($S_{\text{MultiX}} < 42.5$)**, immediately append `0` to `decision_window` (crediting your 20.8% Bumble ELO budget with a left swipe), **and do NOT append `-1e9` to `score_window**`.
2. For all surviving candidates ($S_{\text{MultiX}} \ge 42.5$), append $Z_{\text{final}}(\mathbf{x})$ to `score_window` and adjust the PID quantile target over the non-vetoed pool:

$$q_t = \text{clip}\left(1.0 - \frac{0.208}{\hat{p}_{\text{survive}}} + K_p\big(\hat{r}_{200} - 0.208\big),\; 0.700,\; 0.830\right)$$



where $\hat{p}_{\text{survive}}$ is the rolling fraction of cards passing the hard floors (~65%–70%). This guarantees your live bot maintains a rock-solid **20.8% overall right-swipe rate** while 100% of approved swipes come from the clean $S_{\text{MultiX}} \ge 50+$ tier.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-02T16:18:26+07:00.
</ADDITIONAL_METADATA>