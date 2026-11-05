# BumbleClaw: Pure-Vision Preference Ranking Architecture (RFC Dossier)
**Document Purpose:** System specification, empirical benchmarks, and technical Request for Comments (RFC) for consulting Advanced LLM / AI Research Scientists.  
**Constraint Mandate:** **100% PURE COMPUTER VISION (Picture Only)**. Zero text, zero OCR, zero metadata, single photo (Photo 1) only, sub-200ms GPU latency budget.

---

## Instructions for the Reviewing LLM

You are acting as a **Senior Staff Machine Learning & Computer Vision Research Scientist** specializing in:
1. Deep metric learning and subcenter representation learning.
2. Fine-grained visual preference and aesthetic recommendation systems.
3. Ranking under extreme class imbalance ($9\% - 11\%$ positive base rate).
4. Real-time edge inference latency optimization ($\le 200\text{ms}$ per candidate on modern GPU).

Please review the complete system architecture, empirical progression tables, mathematical formulations, and failure modes presented below. Then provide concrete, mathematically rigorous, and actionable recommendations for the **6 Research Questions** in Section 6.

---

## 1. Problem Formulation & Operational Constraints

### 1.1 The Objective
Given an uncurated stream of single smartphone screenshots of dating profiles (Photo 1 only), learn a visual scoring function $f_\theta(I) \in \mathbb{R}$ that ranks candidates according to a single human user's visual attraction preferences, such that swiping right strictly on the top $K = 20\%$ or $25\%$ of profiles maximizes positive recall while maintaining high precision.

### 1.2 Non-Negotiable Operational Constraints
1. **100% Single-Photo Evaluation (Photo 1 Only):**
   - The system **strictly cannot scroll** to view subsequent photos or read full bios due to ADB mobile automation speed requirements. All decisions must be made from the primary profile photo.
2. **Pure Computer Vision Only (Zero Text / Zero Metadata):**
   - The user has strictly mandated that candidate bios, occupations, alma maters, names, and ages **must NOT be used** as scoring features. The model must learn exclusively from visual pixels: facial geometry, physical aesthetics, photographic composition, styling, and expressions.
3. **Strict Latency Budget (<200ms End-to-End):**
   - Inference per card must execute in $\le 200\text{ms}$ on an NVIDIA GPU (CUDA) to enable seamless real-time mobile automation without UI lag.
4. **Platform ELO Quota Restriction ($\le 20\% - 25\%$ Swipe Rate):**
   - Dating platform recommendation algorithms heavily penalize accounts with high right-swipe rates by demoting their profile ELO. The bot must operate under a strict PID-regulated right-swipe ceiling ($\le 20\% - 25\%$).
5. **Corpus Integrity (100% Human Ground Truth, Zero Pseudo-Labeling):**
   - The dataset consists of **41,582 sequentially swiped real profiles** manually reviewed and labeled by the exact same human user over 6 months:
     - **Full Training Corpus (Rows 0 to 40,000):** 40,000 profiles (**39,346 with detected faces | 4,400 True Likes**, $11.18\%$ like rate).
     - **Untouched Holdout Validation Corpus (Rows 40,000 to 41,582):** 1,582 profiles (**1,533 with detected faces | 140 True Likes**, $9.13\%$ like rate).
   - Strictly zero synthetic or pseudo-labels are permitted.

---

## 2. Current Model Architecture: The 40k Grand Super-Ensemble

The current state-of-the-art model is a **4-Pillar Hybrid Ensemble** combining deep facial metric learning with dual-crop vision-language foundation representations, blended via an out-of-fold Rank-Normal Meta-Regressor.

```
                         Raw Screenshot I (Photo 1)
                                     │
           ┌─────────────────────────┴────────────────────────┐
           ▼                                                  ▼
   InsightFace Alignment                           Dual-Crop Generation
 (RetinaFace + ResNet-100)                     ┌──────────────┴──────────────┐
           │                                   ▼                             ▼
   Face Crop (112x112)                 Full Crop (384x384)        Portrait Crop (384x384)
           │                                   │                     (1.8x Head Bbox)
           ▼                                   ▼                             │
  ArcFace ResNet-100                    SigLIP SO400M                        ▼
   512D Normalized                      1152D Normalized               SigLIP SO400M
           │                                   │                      1152D Normalized
           │                                   │                             │
  ┌────────┴──────────────┐             ┌──────┴──────────────┐              │
  ▼                       ▼             ▼                     ▼              ▼
Pillar 1:              Pillar 3:     Pillar 2:             Pillar 4:
Single-Crop MLP        Single pAUC   Dual-Crop MLP         Dual-Crop pAUC
(ArcFace + CLIP)       Subcenters    (ArcFace + SigLIP-F   (SoftTriple + Gated
(1024D)                (12 Proxies)   + SigLIP-P) (2816D)   Portrait Attention)
  │                       │             │                     │
  └───────────────┬───────┴─────────────┴─────────────────────┘
                  ▼
         Rank-Normal Meta-Blender (L2 Logistic Regression / Gaussian Quantiles)
                  │
                  ▼
          Final Preference Probability p_like ∈ [0, 1]
```

### 2.1 Feature Representation Extraction

| Feature Vector | Dimensionality | Backbone Architecture | Input Resolution | Crop Target | What It Captures |
| :--- | :---: | :--- | :---: | :--- | :--- |
| $\mathbf{x}_{\text{arc}}$ | 512D | ArcFace (ResNet-100) | $112 \times 112$ | 5-point aligned face | Facial bone structure, eye/nose/jaw geometry, identity sphere |
| $\mathbf{x}_{\text{sig\_full}}$ | 1152D | SigLIP (SO400M / 14) | $384 \times 384$ | Full screen frame | Overall scene, clothing, environment, framing, background |
| $\mathbf{x}_{\text{sig\_port}}$ | 1152D | SigLIP (SO400M / 14) | $384 \times 384$ | $1.8\times$ Face Bbox | Upper torso, shoulders, hairstyle, makeup, neck/decolletage |
| $\mathbf{x}_{\text{clip}}$ | 512D | OpenAI CLIP (ViT-B/32) | $224 \times 224$ | Full screen frame | Global coarse semantics (legacy anchor) |

### 2.2 The Four Model Pillars

#### Pillar 1: Single-Crop Multimodal MLP (1024D)
- **Input:** Concatenation $\mathbf{z}_1 = [\mathbf{x}_{\text{arc}} \,\|\, \mathbf{x}_{\text{clip}}] \in \mathbb{R}^{1024}$.
- **Architecture:** `Linear(1024, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 64) -> ReLU -> Linear(64, 1)`.
- **Loss:** Binary Cross-Entropy (BCE) with $L_2$ weight decay ($\lambda = 10^{-4}$).

#### Pillar 2: Dual-Crop SigLIP Multimodal MLP (2816D)
- **Input:** Concatenation $\mathbf{z}_2 = [\mathbf{x}_{\text{arc}} \,\|\, \mathbf{x}_{\text{sig\_full}} \,\|\, \mathbf{x}_{\text{sig\_port}}] \in \mathbb{R}^{2816}$.
- **Architecture:** `Linear(2816, 512) -> LayerNorm -> GELU -> Dropout(0.35) -> Linear(512, 128) -> GELU -> Linear(128, 1)`.
- **Loss:** BCE + Cosine Regularization.

#### Pillar 3: Single-Crop Regularized pAUC Subcenter (512D)
- **Mathematical Basis:** SoftTriple Subcenter Metric Learning with $K = 12$ positive proxy centroids $\mathbf{c}_k^+ \in \mathbb{R}^{512}$ and 1 negative centroid $\mathbf{c}^-$.
- The similarity to the positive class is the smooth-maximum over the 12 archetypes:
  $$S^+(\mathbf{x}) = \frac{1}{\lambda} \ln \sum_{k=1}^{K} \exp\left(\lambda \cos(\mathbf{x}, \mathbf{c}_k^+)\right)$$
- **Loss:** Partial AUC (pAUC) margin ranking loss focusing exclusively on the top false-positive quartile ($[\text{FPR}_0, \text{FPR}_1] = [0.0, 0.25]$).

#### Pillar 4: Dual-Crop Regularized pAUC Subcenter (1664D)
- **Input:** Gated fusion of ArcFace (512D), SigLIP-Full (1152D), and SigLIP-Portrait (1152D).
- **Architecture:** Learned self-attention gating $\beta \in [0, 1]$ between facial geometry and portrait styling:
  $$\mathbf{h} = \mathbf{W}_a \mathbf{x}_{\text{arc}} + \sigma(\mathbf{W}_g [\mathbf{h}_{\text{full}} \,\|\, \mathbf{h}_{\text{port}} \,\|\, \mathbf{x}_{\text{valid}}]) \odot \mathbf{W}_p \mathbf{x}_{\text{sig\_port}}$$
- Features are mapped into a 256D metric hypersphere and classified against 12 regularized subcenters with hard-margin penalty.

### 2.3 Meta-Blender Formulation
On out-of-fold validation distributions, raw scores from all 4 pillars are transformed into Gaussian Rank-Normal quantiles:
$$q^{(m)}_i = \Phi^{-1}\left(\frac{\text{rank}(z^{(m)}_i)}{N + 1}\right)$$
The ensemble combines these quantiles with weights calibrated to favor dual-crop representations ($w = [0.15, 0.40, 0.15, 0.30]$), generating the final calibrated preference score.

---

## 3. Empirical Progression Across All Historical Eras

### Tournament A: Historical Scaling Benchmark across 7 Eras
Evaluated on a strictly locked holdout slice of **969 profiles (Slice 40k–41k)** containing **82 true likes (8.46% base rate)** and **887 true passes**:

| Model Era | Training Corpus Size | Input Modality | Precision @ Top-20% Cap (194 Swipes) | Recall @ Top-20% Cap (194 Swipes) | Recall @ Top-25% Cap (242 Swipes) | PR-AUC | ROC-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original 7k (Chandra)** | 7,126 | ArcFace 512D Ridge | 24.7% (48 / 194) | 58.5% (48 / 82) | 65.9% (54 / 82) | 0.2212 | 0.7788 |
| **Multimodal 7k** | 7,126 | ArcFace + CLIP ViT-B | 25.3% (49 / 194) | 59.8% (49 / 82) | 69.5% (57 / 82) | 0.2541 | 0.8142 |
| **ExperimentX (Rounds 1–3)** | 9,886 | Re-weighted Ridge | 28.4% (55 / 194) | 67.1% (55 / 82) | 74.4% (61 / 82) | 0.3108 | 0.8491 |
| **10k Pure Bumble** | 9,990 | ArcFace + CLIP MLP | 28.9% (56 / 194) | 68.3% (56 / 82) | 75.6% (62 / 82) | 0.2974 | 0.8529 |
| **20k Pure Bumble** | 19,768 | 1024D MLP + Cosine | 33.0% (64 / 194) | 78.1% (64 / 82) | 82.9% (68 / 82) | 0.3524 | 0.8876 |
| **30k Grand Ensemble** | 29,582 | 3-Pillar Single+Dual | 32.5% (63 / 194) | 76.8% (63 / 82) | 85.4% (70 / 82) | 0.4851 | 0.9064 |
| **40k Grand Super-Ensemble (33.5k Split)** | 33,063 | 4-Pillar Dual-Crop Meta | 34.5% (67 / 194) | 81.7% (67 / 82) | 87.8% (72 / 82) | 0.4909 | 0.9126 |

---

### Tournament B: Full 40,000 Master Model Head-to-Head Tournament
Evaluated on the newly labeled, untouched **1,582 Validation Holdout (`40,000–41,582`)** containing **140 true likes (9.13% base rate)** and **1,393 passes** (1,533 with faces):

| Model | Training Size | Hits @ 20% Cap (307 Swipes) | Recall @ 20% Cap | Precision @ 20% Cap | Hits @ 25% Cap (383 Swipes) | Recall @ 25% Cap | Precision @ 25% Cap | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30k Multimodal MLP (Pure)** | 29,582 | 85 | 60.7% | 27.7% | 92 | 65.7% | 24.0% | 0.2742 | 0.8258 |
| **33.5k Pillar 1 (Single MLP)** | 33,063 | 76 | 54.3% | 24.8% | 81 | 57.9% | 21.1% | 0.2221 | 0.7181 |
| **33.5k Pillar 2 (Dual-Crop MLP)** | 33,063 | 36 | 25.7% | 11.7% | 45 | 32.1% | 11.7% | 0.1099 | 0.5233 |
| **33.5k Pillar 4 (Dual pAUC)** | 33,063 | 94 | 67.1% | 30.6% | 107 | 76.4% | 27.9% | 0.2999 | 0.8545 |
| **TRUE 40k Pillar 1 (Single MLP Full)** | **39,346** | 102 | 72.9% | 33.2% | 113 | 80.7% | 29.5% | 0.4435 | 0.8833 |
| **TRUE 40k Pillar 2 (Dual-Crop MLP Full)** | **39,346** | 100 | 71.4% | 32.6% | 111 | 79.3% | 29.0% | 0.4779 | 0.8873 |
| **TRUE 40k Pillar 4 (Dual pAUC Full)** | **39,346** | 93 | 66.4% | 30.3% | 105 | 75.0% | 27.4% | 0.2990 | 0.8515 |
| **TRUE 40k Grand Super-Ensemble (FULL)** | **39,346** | **109** | **77.9%** | **35.5%** | **120** | **85.7%** | **31.3%** | **0.5493** | **0.9100** |

### Key Diagnostic Observations:
1. **The +6,500 Full Corpus Absorption Effect:**
   - Training on the complete 39,346 dataset jumped Pillar 1 recall from **54.3% $\to$ 72.9%** (+18.6 percentage points) and PR-AUC from **0.2221 $\to$ 0.4435** (+99.7% relative gain!).
   - Pillar 2 (Dual-Crop MLP, 2,816D) experienced a massive jump in PR-AUC from **0.1099 $\to$ 0.4779** (+334% relative gain), confirming that high-dimensional dual-crop representations require at least 35k–40k samples to properly calibrate their weights.
2. **Theoretical Precision Ceiling:** At a 9.13% natural like rate, swiping 307 cards (20% cap) can yield at most 140 hits ($140 / 307 = 45.6\%$). The True 40k model's **35.5% precision** operates at **77.9% of the theoretical mathematical maximum**.
3. **The "20 Missed Likes" Bottleneck:** Under the 25% quota (383 cards), the True 40k model captures 120 of 140 likes (85.7%). Exactly 20 likes are missed.

---

## 4. Known Failure Modes & Pure-Vision Challenges

### Failure Mode 1: Over-Reliance on Contrastive Language Supervision
Both CLIP and SigLIP are trained via image-text contrastive learning on web-scraped alt-text. While they excel at categorical semantic concepts ("woman smiling", "beach dress"), they suffer from:
- **Textual Compression Bottleneck:** They discard subtle, non-verbal visual properties that are never described in alt-text: exact body proportions, skin luster, fine hair texture, photographic depth of field, fabric quality, and subtle aesthetic posture.
- **Semantic Distraction:** A visually unappealing photo with a luxury car or scenic background can receive an inflated SigLIP aesthetic embedding because the text decoder recognizes prestige items.

### Failure Mode 2: ArcFace Identity Invariance vs. Aesthetic Variance
ArcFace was specifically trained on Glint360k to achieve **expression and pose invariance** for biometric identification. It maps the same woman smiling, frowning, wearing heavy makeup, or looking tired to nearly the exact same 512D identity sphere vector.  
However, human romantic attraction is **inherently expression-variant**: a candid warm smile, playful head tilt, and direct eye contact are core attraction drivers that ArcFace actively suppresses.

### Failure Mode 3: Pointwise Binary Cross-Entropy Calibration Mismatch
Our MLPs output pointwise probabilities $P(y=1 \mid \mathbf{x})$. However, the user does not evaluate candidates in isolation; candidates are ranked dynamically against each other. When an MLP is trained with BCE under a 10:1 class imbalance, gradients are dominated by easy negative suppression, leading to suboptimal rank separation among top-quartile borderline profiles.

### Failure Mode 4: Selectivity Drift Across the 41.5k Timeline
The user's human right-swipe rate has drifted monotonically over 6 months of labeling:
- Profiles 0–10k: **14.50% Like Rate** (early exploratory phase, generous)
- Profiles 10k–20k: **11.20% Like Rate** (stabilizing)
- Profiles 20k–30k: **10.15% Like Rate** (mature criteria)
- Profiles 30k–41.5k: **9.13% Like Rate** (highly selective target regime)

Treating all 40k samples equally forces the model to learn early "generous" boundary thresholds, leading to false-positive overprediction on newer, stricter distributions.

---

## 5. Architectural Upgrade Candidates Under Evaluation

We are currently considering the following pure-vision architectural enhancements:

1. **Self-Supervised Foundation Models (DINOv2):** Adding Meta's `dinov2_vitb14` (768D) to bypass contrastive language bottlenecks and capture dense visual geometry and aesthetic textures.
2. **Facial Expression & Landmark Geometry Tokens (16D):** Extracting normalized 3D head pose (yaw, pitch, roll), mouth corner curvature (Duchenne smile intensity), eye aspect ratio, and facial symmetry directly from InsightFace's 68/106 landmarks.
3. **Pairwise & Listwise Ranking Loss Formulation:** Replacing or augmenting BCE with pairwise ranking (RankNet, MarginRankingLoss) or listwise ranking (LambdaLoss / ApproxNDCG).
4. **Online Hard Negative Mining in SoftTriple Subcenters:** Dynamically sampling passed profiles that have the highest cosine similarity to positive proxies.
5. **Continuous Exponential Recency Weighting:** Applying $w_i = \exp(-\lambda(t_{\text{current}} - t_i))$ to training samples.

---

## 6. The Six RFC Questions for External AI Research Advisors

We request detailed, mathematically rigorous, and code-actionable advice on the following six problems:

### Question 1: Self-Supervised Vision Integration (DINOv2 vs. SigLIP)
- Our empirical tests show that SigLIP Dual-Crop (2816D) achieves **0.4779 PR-AUC** on its own. How should DINOv2 (ViT-B/14, 768D) be integrated alongside SigLIP?
- Should we use the `[CLS]` token, mean-pooled patch tokens, or train a lightweight Multi-Head Cross-Attention pooling module over the $27 \times 27$ patch grid?
- Does DINOv2's feature representation require $L_2$ normalization before concatenation, and how can we prevent DINOv2 from dominating the lower-dimensional (512D) ArcFace representation without ballooning total dimensionality beyond 3,500D?

### Question 2: Decoupling Expression/Pose from ArcFace Identity
- What is the most effective mathematical formulation to combine ArcFace's 512D identity vector with 16D geometric/pose landmark tokens?
- Should this be a residual projection, a bilinear pooling layer ($\mathbf{x}_{\text{arc}}^T \mathbf{W} \mathbf{x}_{\text{geom}}$), or a gated feature modulation (FiLM: Feature-wise Linear Modulation) where geometry scales and shifts the visual embedding?

### Question 3: Pointwise vs. Pairwise vs. Listwise Loss for Top-20% Truncated Quota
- Given that our inference policy swiping strictly enforces a top-20% or top-25% cutoff, what objective function guarantees maximum precision at $K$?
- Specifically, how does **Pairwise Margin Ranking Loss** compare against **LambdaLoss** (which directly optimizes NDCG / truncated recall) on tabular embedding vectors? What is the recommended batch formation and negative-to-positive pairing ratio?

### Question 4: Stabilizing SoftTriple Subcenter Learning with Hard Negative Mining
- In our 12-archetype subcenter model, random negative sampling causes gradient starvation because easy negatives yield zero loss.
- However, aggressive hard negative mining can cause cluster collapse (proxies drifting toward the negative centroid). What curriculum or semi-hard mining boundary (e.g., sampling negatives within $[\cos(\mathbf{x}, \mathbf{c}_k) - \alpha, \cos(\mathbf{x}, \mathbf{c}_k)]$) maintains proxy diversity while sharpening boundaries?

### Question 5: Continuous Continual Learning Under Human Selectivity Drift
- The user's like rate shifted from 14.5% to 9.13% across 41,582 sequential swipes.
- If we apply exponential sample weights $w_i = \exp(-\lambda \Delta t)$, how should $\lambda$ be calibrated to prevent catastrophic forgetting of rare, valid aesthetic archetypes that appeared only in the first 10,000 cards?
- Is sample re-weighting superior to training separate temporal adapters or a continual replay buffer?

### Question 6: Sub-200ms Single-Photo Heuristic Guardrails
- Before running deep transformer backbones (SigLIP / DINOv2), what fast (<15ms) OpenCV / InsightFace filters should we enforce on Photo 1 to veto cards that have an empirical $<3\%$ like probability?
- What exact mathematical thresholds do you recommend for:
  - Head yaw angle ($|\theta_{\text{yaw}}|$).
  - Eye landmark confidence (sunglasses/occlusion detection).
  - Multi-face ambiguity resolution.
  - Image sharpness (Laplacian variance vs. frequency-domain FFT energy).
