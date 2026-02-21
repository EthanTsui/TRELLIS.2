# Comprehensive Quality Taxonomy for Image-to-3D Generation

**Date**: 2026-02-19
**Author**: Research Optimizer Agent
**Scope**: Definitions of "quality" in single-image-to-3D generation, grounded in academic literature
**Applicability**: TRELLIS.2 pipeline evaluation and optimization

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Quality Taxonomy Overview](#2-quality-taxonomy-overview)
3. [Dimension 1: Geometric Fidelity](#3-dimension-1-geometric-fidelity)
4. [Dimension 2: Appearance/Texture Quality](#4-dimension-2-appearancetexture-quality)
5. [Dimension 3: Multi-View Consistency](#5-dimension-3-multi-view-consistency)
6. [Dimension 4: Perceptual Quality / Human Preference](#6-dimension-4-perceptual-quality--human-preference)
7. [Dimension 5: Physical Plausibility](#7-dimension-5-physical-plausibility)
8. [Dimension 6: Prompt/Input Alignment](#8-dimension-6-promptinput-alignment)
9. [Dimension 7: Material Quality (PBR)](#9-dimension-7-material-quality-pbr)
10. [Dimension 8: Mesh Production Quality](#10-dimension-8-mesh-production-quality)
11. [Benchmarks and Datasets](#11-benchmarks-and-datasets)
12. [Metrics Most Used in SOTA Papers](#12-metrics-most-used-in-sota-papers)
13. [Relevance to TRELLIS.2](#13-relevance-to-trellis2)
14. [Comparison with Our Current Evaluator](#14-comparison-with-our-current-evaluator)
15. [References](#15-references)

---

## 1. Executive Summary

"Quality" in image-to-3D generation is **not a single number**. The academic community has converged on a multi-dimensional definition spanning geometry, appearance, consistency, human preference, physical plausibility, input alignment, material properties, and production readiness. Key findings:

- **No single metric captures quality.** The field has moved from individual metrics (FID, CD) to multi-dimensional evaluation frameworks (GPTEval3D, 3DGen-Bench, MATE-3D, T3Bench).
- **Human preference is the gold standard.** Automated metrics consistently show weak-to-moderate correlation with human judgments. The best automated evaluators (3DGen-Score, HyperScore, Rank2Score) are trained on human preference data.
- **Geometry and appearance are separable concerns.** Normal maps evaluate geometry; rendered RGB evaluates appearance. Best practices use both.
- **No-reference metrics are emerging.** SRAM, objectness-based detection metrics, and LLM-based evaluators can assess quality without ground-truth 3D.
- **For TRELLIS.2 optimization**, the most actionable metrics are: Silhouette IoU (shape), LPIPS (perceptual detail), CLIP-similarity (input alignment), normal consistency (surface smoothness), and LLM-based multi-view quality assessment.

---

## 2. Quality Taxonomy Overview

```
                         3D Generation Quality
                                |
         +----------+----------+----------+----------+
         |          |          |          |          |
    Geometric   Appearance  Consistency  Perceptual  Physical
    Fidelity    Quality     (Multi-View) Quality     Plausibility
         |          |          |          |          |
    +----+----+ +---+---+ +---+--+ +---+---+ +---+---+
    |CD  |F1  | |LPIPS  | |MEt3R | |GPTEval| |Manifold|
    |IoU |NC  | |SSIM   | |PPLC  | |ELO    | |Watertight
    |HD  |NL2 | |FID    | |FVD   | |DreamSim |Self-int|
    +----+----+ +---+---+ +------+ +-------+ +-------+

         +----------+----------+----------+
         |          |          |          |
    Prompt/Input  Material   Production
    Alignment     (PBR)      Readiness
         |          |          |
    +----+----+ +---+---+ +---+---+
    |CLIP-S   | |Albedo | |Topology|
    |ObjectDet| |Rough  | |UV qual |
    |LLM eval | |Metal  | |Poly cnt|
    +----+----+ +---+---+ +-------+
```

### Summary Table: The Eight Dimensions

| # | Dimension | What It Measures | Key Metrics | Weight in Field |
|---|-----------|------------------|-------------|-----------------|
| 1 | Geometric Fidelity | Shape accuracy vs ground truth | CD, F-score, V-IoU, HD, NC | High |
| 2 | Appearance/Texture Quality | Visual quality of surfaces | LPIPS, SSIM, PSNR, FID, KID | High |
| 3 | Multi-View Consistency | Coherence across viewpoints | MEt3R, PPLC, FVD | Medium-High |
| 4 | Perceptual Quality | Human-aligned overall quality | GPTEval3D, 3DGen-Score, DreamSim, ELO | Very High |
| 5 | Physical Plausibility | Realistic, believable shapes | Manifold checks, self-intersection ratio | Medium |
| 6 | Prompt/Input Alignment | Fidelity to input image/text | CLIP-S, ULIP-2, Uni3D, object detection | High |
| 7 | Material Quality (PBR) | PBR material accuracy | RMSE on albedo/roughness/metallic, relighting | Medium |
| 8 | Production Quality | Usability in downstream apps | Topology, UV quality, texel density | Medium |

---

## 3. Dimension 1: Geometric Fidelity

Geometric fidelity measures how accurately the generated 3D shape matches the intended geometry.

### 3.1 Chamfer Distance (CD)

**Definition**: The average nearest-neighbor distance between two point clouds.

**Formula** (Bidirectional L2):
```
CD(P, Q) = (1/|P|) * sum_{p in P} min_{q in Q} ||p - q||^2
          + (1/|Q|) * sum_{q in Q} min_{p in P} ||q - p||^2
```

**Properties**:
- Scale-dependent (normalize mesh bounding boxes before computing)
- Symmetric (P-to-Q + Q-to-P)
- L1 variant is more robust to outliers; L2 variant (squared) penalizes large errors more

**Usage**: The most commonly reported geometric metric in 3D generation papers. Used by TRELLIS, InstantMesh, TripoSR, AssetGen, and virtually all reconstruction papers.

**Limitations**: Sensitive to point sampling density. Does not capture topology (a sphere with a hole vs. a complete sphere may have similar CD).

**Relevance to TRELLIS.2**: HIGH. TRELLIS.2 paper reports CD on Toys4K and Sketchfab datasets.

---

### 3.2 F-Score (F1 at threshold tau)

**Definition**: The harmonic mean of precision and recall, where precision is the fraction of predicted points within distance tau of a ground-truth point, and recall is the fraction of ground-truth points within distance tau of a predicted point.

**Formula**:
```
Precision(tau) = |{p in P : min_{q in Q} ||p-q|| < tau}| / |P|
Recall(tau)    = |{q in Q : min_{p in P} ||q-p|| < tau}| / |Q|
F-score(tau)   = 2 * Precision * Recall / (Precision + Recall)
```

Common thresholds: tau = 0.01 (1% of bounding box diagonal), tau = 0.05

**Properties**: More interpretable than CD. Captures surface completeness (recall) and accuracy (precision) separately.

**Usage**: Reported alongside CD in most papers. TRELLIS.2 reports F-score.

**Relevance to TRELLIS.2**: HIGH.

---

### 3.3 Volume IoU (V-IoU)

**Definition**: The intersection-over-union of voxelized occupied volumes.

**Formula**:
```
V-IoU = |V_pred intersection V_gt| / |V_pred union V_gt|
```

where V_pred and V_gt are voxelized at a fixed resolution (typically 128^3 or 256^3).

**Properties**: Scale-invariant if both shapes are voxelized at the same resolution. Captures interior volume, not just surface.

**Usage**: Common in older papers and for sparse structure evaluation. Less common for high-res mesh evaluation.

**Relevance to TRELLIS.2**: MEDIUM. Directly applicable to Stage 1 (sparse structure) evaluation, which produces a 32^3 or 64^3 occupancy grid.

---

### 3.4 Hausdorff Distance (HD)

**Definition**: The maximum nearest-neighbor distance between two point sets.

**Formula**:
```
HD(P, Q) = max(max_{p in P} min_{q in Q} ||p-q||,
               max_{q in Q} min_{p in P} ||q-p||)
```

**Properties**: Captures worst-case deviation. Very sensitive to outliers.

**Usage**: Common in CAD/manufacturing quality assessment. Less common in generative 3D due to outlier sensitivity. Often 90th-percentile HD is used instead.

**Relevance to TRELLIS.2**: LOW for generation evaluation, but useful for detecting gross geometric errors.

---

### 3.5 Earth Mover's Distance (EMD)

**Definition**: The minimum total work needed to transform one point distribution into another.

**Formula**:
```
EMD(P, Q) = min_{phi: P->Q bijection} sum_i ||p_i - phi(p_i)||
```

**Properties**: Requires equal point counts (or relaxation). O(n^3 log n) computational complexity. More principled than CD but much more expensive.

**Usage**: Occasionally reported as a supplementary metric. Less common due to computational cost.

**Relevance to TRELLIS.2**: LOW.

---

### 3.6 Normal Consistency (NC) / Normal Correctness

**Definition**: The average angular alignment between predicted and ground-truth surface normals.

**Formula**:
```
NC = (1/|P|) sum_{p in P} |n_p . n_{NN(p,Q)}|
```

where n_p is the normal at point p and NN(p,Q) is p's nearest neighbor in Q.

**Properties**: Captures surface orientation quality independent of position. Correlates with visual quality in rendered normal maps.

**Usage**: TRELLIS.2 reports PSNR-N and LPIPS-N (on rendered normal maps) as proxies for NC. MeshFormer, AssetGen, and other mesh-focused papers report NC directly.

**Relevance to TRELLIS.2**: HIGH. Normal quality is critical for PBR rendering (lighting depends on normals). Our current evaluator has a mesh_smoothness_score that measures face-vertex normal consistency, which is a related concept.

---

### 3.7 Silhouette IoU (Mask IoU)

**Definition**: IoU between binary silhouette masks rendered from the same viewpoint.

**Formula**:
```
Sil-IoU(v) = |M_pred(v) intersection M_gt(v)| / |M_pred(v) union M_gt(v)|
```

averaged over multiple views v.

**Properties**: Captures outline shape without requiring dense 3D ground truth. Can be computed from rendered images alone.

**Limitations**: Only captures external boundary; ignores interior geometry. A sphere and a solid cube with the same silhouette would have identical Sil-IoU from certain views.

**Usage**: Very common for image-to-3D evaluation where 3D ground truth is unavailable. Used in AssetGen and many others.

**Relevance to TRELLIS.2**: HIGH. Our current evaluator uses this as the primary shape metric (20 points, highest weight).

---

### 3.8 Depth Error

**Definition**: Per-pixel L1 or L2 error between rendered depth maps.

**Formula**:
```
DepthErr = (1/|M|) sum_{(u,v) in M} |D_pred(u,v) - D_gt(u,v)|
```

where M is the intersection of valid depth pixels.

**Usage**: AssetGen reports L1 depth error. Common in reconstruction papers.

**Relevance to TRELLIS.2**: MEDIUM. Useful for evaluating geometric detail beyond silhouette.

---

## 4. Dimension 2: Appearance/Texture Quality

Appearance quality measures the visual fidelity of rendered surfaces.

### 4.1 LPIPS (Learned Perceptual Image Patch Similarity)

**Definition**: Distance between deep feature representations (AlexNet or VGG) of two images, trained on human perceptual judgments.

**Formula**:
```
LPIPS(x, y) = sum_l w_l * ||phi_l(x) - phi_l(y)||^2
```

where phi_l extracts features from layer l of a pretrained network, and w_l are learned weights.

**Properties**: Better aligned with human perception than PSNR/SSIM. Robust to small geometric perturbations. Lower is better.

**Usage**: The de facto standard perceptual metric. Used by TRELLIS, TRELLIS.2, AssetGen, InstantMesh, and virtually all papers evaluating rendered appearance.

**Relevance to TRELLIS.2**: VERY HIGH. Our evaluator uses LPIPS as 50% of the detail score (10pt dimension).

---

### 4.2 SSIM (Structural Similarity Index)

**Definition**: Measures structural similarity between images based on luminance, contrast, and structure.

**Formula**:
```
SSIM(x, y) = [l(x,y)]^alpha * [c(x,y)]^beta * [s(x,y)]^gamma

where:
  l(x,y) = (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
  c(x,y) = (2*sigma_x*sigma_y + C2) / (sigma_x^2 + sigma_y^2 + C2)
  s(x,y) = (sigma_xy + C3) / (sigma_x*sigma_y + C3)
```

**Properties**: 0 to 1 (higher is better). Computed in sliding windows (typically 11x11 Gaussian). More perceptually meaningful than PSNR but still pixel-aligned.

**Usage**: Second most common appearance metric after LPIPS. Reported in most NVS and 3D generation papers.

**Relevance to TRELLIS.2**: HIGH. Our evaluator includes windowed SSIM.

---

### 4.3 PSNR (Peak Signal-to-Noise Ratio)

**Definition**: Logarithmic ratio of peak signal power to noise power.

**Formula**:
```
PSNR = 10 * log10(MAX^2 / MSE)
     = 20 * log10(MAX / sqrt(MSE))

where MSE = (1/N) sum_i (x_i - y_i)^2
```

**Properties**: In dB. Higher is better. Purely pixel-level; does not capture perceptual quality well. Two images can have identical PSNR but very different perceptual quality.

**Usage**: Still widely reported due to convention. TRELLIS.2 reports PSNR on both RGB renders and normal maps.

**Relevance to TRELLIS.2**: MEDIUM. Less informative than LPIPS for optimization decisions.

---

### 4.4 FID (Frechet Inception Distance)

**Definition**: Statistical distance between feature distributions of real and generated image sets, using Inception-V3 features.

**Formula**:
```
FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^(1/2))
```

where (mu_r, Sigma_r) and (mu_g, Sigma_g) are the mean and covariance of Inception features for real and generated sets.

**Properties**: Measures both quality and diversity. Requires large sample sizes (>10K) for stability. Lower is better. Not meaningful for single-sample evaluation.

**Usage**: Common for evaluating distributional quality of generative models (not individual outputs). TRELLIS reports FD (Frechet Distance) with multiple feature extractors (Inception-v3, DINOv2, PointNet++).

**Relevance to TRELLIS.2**: LOW for per-sample optimization, MEDIUM for model-level evaluation.

---

### 4.5 KID (Kernel Inception Distance)

**Definition**: Maximum Mean Discrepancy (MMD) using a polynomial kernel on Inception features.

**Formula**:
```
KID = MMD^2(phi(X_r), phi(X_g))
    = E[k(phi(x), phi(x'))] + E[k(phi(y), phi(y'))] - 2*E[k(phi(x), phi(y))]
```

where k is a polynomial kernel and phi extracts Inception features.

**Properties**: Unbiased (unlike FID). Works better with smaller sample sizes. Lower is better.

**Usage**: Reported alongside FID in TRELLIS and some other papers. Same distributional caveats apply.

**Relevance to TRELLIS.2**: LOW for per-sample optimization.

---

### 4.6 DreamSim

**Definition**: A perceptual similarity metric bridging low-level (LPIPS) and high-level (CLIP) similarity, trained on human judgments of synthetic image triplets.

**Architecture**: Concatenates CLIP, OpenCLIP, and DINO embeddings, finetuned on ~20K human-judged triplets generated by diffusion models.

**Properties**: Better alignment with human similarity judgments than LPIPS, SSIM, PSNR, or CLIP alone. Focuses on foreground objects and semantic content while remaining sensitive to color and layout.

**Venue**: NeurIPS 2023 Spotlight, with follow-up at NeurIPS 2024.

**Usage**: Increasingly used in novel view synthesis evaluation. Available as a pip package.

**Relevance to TRELLIS.2**: HIGH. Could replace or supplement LPIPS in our evaluator for better human-aligned scoring.

---

## 5. Dimension 3: Multi-View Consistency

Multi-view consistency measures whether a 3D asset looks coherent when viewed from different angles.

### 5.1 MEt3R (Measuring Multi-View Consistency)

**Definition**: A pose-free, content-independent metric that measures 3D consistency between image pairs using DUSt3R-based reconstruction and DINO/FeatUp feature comparison.

**Method**:
1. Given an image pair, use DUSt3R to obtain dense 3D reconstructions
2. Warp image contents from one view to the other using the 3D reconstruction
3. Compare feature maps (DINO + FeatUp) of warped and target images
4. Aggregate similarity scores

**Properties**: Does not require camera poses or image quality assessment. Invariant to view-dependent lighting effects. Measures purely geometric/structural consistency.

**Venue**: CVPR 2025.

**Relevance to TRELLIS.2**: HIGH. Directly measures whether TRELLIS.2's multi-view outputs are 3D-consistent, which is relevant for texture baking quality.

---

### 5.2 PPLC (Patch-wise Perceptual Likelihood Consistency)

**Definition**: Measures perceptual consistency across generated views by computing patch-wise LPIPS between corresponding regions.

**Usage**: Used in Free3D (CVPR 2024) for evaluating NVS consistency.

**Relevance to TRELLIS.2**: MEDIUM.

---

### 5.3 FVD (Frechet Video Distance)

**Definition**: Extension of FID to video (sequences of frames), using I3D features. Applied to 3D generation by treating multi-view renders as a video sequence.

**Usage**: Used in SV4D (ICLR 2025) for evaluating 4D generation consistency.

**Relevance to TRELLIS.2**: LOW (designed for video, not static 3D).

---

## 6. Dimension 4: Perceptual Quality / Human Preference

Human preference is the ultimate quality measure. Several frameworks formalize this.

### 6.1 GPTEval3D (CVPR 2024)

**Definition**: Uses GPT-4V(ision) to evaluate text-to-3D generation across five criteria, producing ELO ratings via pairwise comparisons.

**Five Evaluation Criteria**:
1. **Text-Asset Alignment**: How well the 3D asset mirrors the input text description
2. **3D Plausibility**: Whether the asset is plausible (no Janus problem, no floaters, no distortion)
3. **Texture Details**: Whether textures are realistic, high-resolution, appropriately saturated
4. **Geometry Details**: Whether geometry is sensible with appropriate detail level
5. **Texture-Geometry Coherency**: Whether geometry and textures agree with each other

**Performance**: Kendall's tau = 0.710 average correlation with human judgments. Pairwise agreement probability = 0.789.

**ELO System**:
```
Pr(i beats j) = 1 / (1 + 10^((sigma_j - sigma_i) / 400))
```

Initial scores set at 1000, optimized via maximum likelihood with Adam optimizer.

**Relevance to TRELLIS.2**: VERY HIGH. The five criteria map well to our optimization goals. GPTEval3D could be integrated as a quality oracle, though it requires API calls and is expensive.

---

### 6.2 3DGen-Bench / 3DGen-Arena (2025)

**Definition**: The first comprehensive multi-dimensional human preference benchmark for 3D generation.

**Five Evaluation Dimensions**:
1. **Geometry Plausibility**: Shape realism, avoidance of distortions
2. **Geometry Details**: Fineness of surface features
3. **Texture Quality**: Aesthetic realism, material quality, coloring
4. **Geometry-Texture Coherence**: Alignment between geometric and textural characteristics
5. **Prompt-Asset Alignment**: Fidelity to input prompt

**Scale**: 68,000+ expert votes, 56,000+ absolute scores on 11,220 models from 1,020 prompts.

**Automated Models**:
- **3DGen-Score**: CLIP-based multi-view encoding with dimension-specific scoring
- **3DGen-Eval**: Multimodal LLM-based evaluator with chain-of-thought reasoning

Both show superior correlation with human ranks compared to CLIP similarity and aesthetic scoring alone.

**Relevance to TRELLIS.2**: HIGH. The evaluation dimensions and automated models could be adopted for our optimization loop.

---

### 6.3 MATE-3D / HyperScore (ICCV 2025)

**Definition**: Multi-dimensional quality evaluator trained on 107,520 human annotations across 1,280 textured meshes.

**Four Evaluation Dimensions**:
1. **Alignment**: Semantic consistency with input prompt
2. **Geometry**: Shape fidelity and structural completeness
3. **Texture**: Color, material, vibrancy, resolution
4. **Overall**: Holistic integration of all perspectives

**HyperScore Architecture**:
- CLIP feature extraction from multi-view renders
- Conditional Feature Fusion (CFF): dimension-specific attention weights
- Adaptive Quality Mapping (AQM): hypernetwork-generated dimension-specific scoring functions

**Evaluation Protocol**: ITU-T P.910 standard, 11-level impairment scale (0-10), 21 subjects per sample.

**Relevance to TRELLIS.2**: HIGH. HyperScore could potentially replace our hand-crafted evaluator.

---

### 6.4 3D Arena (2025)

**Definition**: An open platform for pairwise human preference collection across 19 SOTA models.

**Scale**: 123,243 votes from 8,096 users since June 2024.

**Key Findings**:
- Gaussian splat outputs achieve 16.6 ELO advantage over meshes
- Textured models receive 144.1 ELO advantage over untextured models
- Automated metrics show weak correlation with human preferences

**Relevance to TRELLIS.2**: MEDIUM. The ELO rankings contextualize TRELLIS.2's standing among competitors.

---

### 6.5 DreamSim (NeurIPS 2023 Spotlight)

See Section 4.6 above.

---

### 6.6 LAION Aesthetic Score

**Definition**: A linear regressor on top of CLIP ViT-L/14 embeddings, trained on human aesthetic ratings (1-10 scale).

**Training Data**: Simulacra Aesthetic Captions (176K synthetic images), LAION-Logos (15K), AVA dataset (250K photos).

**Properties**: Quick, cheap. Measures overall visual appeal. Not specific to 3D.

**Usage**: Common as an auxiliary quality signal in text-to-3D and text-to-image evaluation. Scores above 5 represent aesthetically good images.

**Relevance to TRELLIS.2**: MEDIUM. Useful as a cheap quality proxy when applied to rendered views.

---

### 6.7 T3Bench (2024)

**Definition**: First comprehensive text-to-3D benchmark with 300 prompts at three complexity levels.

**Two Metrics**:
- **Quality**: Multi-view text-image scoring + regional convolution for inconsistency detection
- **Alignment**: Multi-view captioning + LLM evaluation for text-3D consistency

Both achieve >0.75 Spearman correlation with human 1-5 scale ratings.

**Relevance to TRELLIS.2**: MEDIUM. Useful for benchmarking against other methods.

---

### 6.8 Rank2Score / T23D-CompBench (2025)

**Definition**: Fine-grained text-to-3D quality assessment via two-stage rank learning.

**Method**: Pairwise ranking + supervised contrastive regression + curriculum learning on 129,600 human ratings across 3,600 textured meshes from 10 SOTA models.

**Relevance to TRELLIS.2**: MEDIUM. Represents the most recent and fine-grained quality assessment framework.

---

## 7. Dimension 5: Physical Plausibility

Physical plausibility measures whether generated 3D assets are geometrically valid and could exist in the real world.

### 7.1 Manifoldness

**Definition**: A mesh is manifold if every edge bounds exactly one or two triangles (edge-manifold) and every vertex's star is edge-manifold and edge-connected (vertex-manifold).

**Metric**: Binary (manifold or not), or percentage of non-manifold edges/vertices.

**Relevance to TRELLIS.2**: MEDIUM. TRELLIS.2's FDG mesh extraction generally produces manifold meshes, but simplification can introduce non-manifold features.

---

### 7.2 Watertightness

**Definition**: A mesh is watertight if it is edge-manifold, vertex-manifold, and has no boundary edges.

**Metric**: Binary, or percentage of boundary edges.

**Relevance to TRELLIS.2**: MEDIUM. Important for downstream applications (3D printing, physics simulation).

---

### 7.3 Self-Intersection Ratio

**Definition**: The fraction of triangles that intersect other triangles in the mesh.

**Formula**:
```
SIR = |{f in F : exists f' in F, f != f', f intersects f'}| / |F|
```

**Relevance to TRELLIS.2**: LOW-MEDIUM. Self-intersections are uncommon in TRELLIS.2 outputs.

---

### 7.4 Janus Detection

**Definition**: The "Janus problem" refers to multi-face artifacts where a 3D object has duplicated features (e.g., two faces) visible from different viewpoints.

**Detection**: Typically done visually or via face detection from multiple rendered views. GPTEval3D's "3D Plausibility" criterion explicitly includes this.

**Relevance to TRELLIS.2**: LOW. TRELLIS.2's native 3D representation (sparse voxels) is less susceptible to the Janus problem compared to SDS-based methods.

---

### 7.5 SRAM (Shape-Realism Alignment Metric)

**Definition**: A no-reference metric using an LLM bridge to evaluate shape realism without ground truth.

**Architecture**: Point-BERT mesh encoder -> PointLLM reasoning bridge -> MLP realism decoder

**Dataset**: RealismGrading -- 16 categories, 16 algorithms, 319 annotators, 5,223 ratings.

**Performance**: PLCC = 0.689, SROCC = 0.696, KROCC = 0.566.

**Relevance to TRELLIS.2**: MEDIUM-HIGH. As a no-reference metric, SRAM could evaluate generated meshes without needing ground truth, which is exactly our optimization scenario.

---

## 8. Dimension 6: Prompt/Input Alignment

Alignment measures how faithfully the generated 3D asset represents the input image or text.

### 8.1 CLIP Similarity Score (CLIP-S)

**Definition**: Cosine similarity between CLIP embeddings of the input (image or text) and rendered views of the generated 3D asset.

**Formula**:
```
CLIP-S = (1/K) sum_{k=1}^{K} cos(CLIP(input), CLIP(render_k))
```

where K is the number of rendered viewpoints.

**Properties**: Captures semantic similarity. Not sensitive to low-level visual details.

**Usage**: The most commonly reported alignment metric. Used by TRELLIS, TRELLIS.2, AssetGen, and most text/image-to-3D papers.

**Limitations**: CLIP was trained on text-image pairs, so image-to-image similarity is not its primary strength. May not capture fine-grained geometric detail.

**Relevance to TRELLIS.2**: HIGH. The primary alignment metric for image-to-3D evaluation.

---

### 8.2 ULIP-2 / Uni3D Similarity

**Definition**: Multimodal embeddings that align 3D shapes with images and text in a joint space.

**Usage**: TRELLIS.2 uses ULIP-2 and Uni3D embeddings for evaluating visual and geometric consistency.

**Relevance to TRELLIS.2**: MEDIUM-HIGH. Already used in TRELLIS.2's evaluation.

---

### 8.3 Object Detection-Based Alignment (SIGGRAPH Asia 2024)

**Definition**: A no-reference metric that compares object detection results between the input image and novel-view renders of the generated 3D object.

**Method**: Run an object detector on both input image and rendered views. Compare detection confidence, bounding box overlap, and semantic class consistency.

**Advantage**: Works with just the input image (no 3D ground truth or multi-view references needed).

**Relevance to TRELLIS.2**: MEDIUM. Novel approach that could augment our evaluator's silhouette matching.

---

## 9. Dimension 7: Material Quality (PBR)

Material quality measures the accuracy of physically-based rendering attributes.

### 9.1 PBR Attribute PSNR/LPIPS

**Definition**: PSNR and LPIPS computed directly on PBR material maps (albedo, roughness, metallic) rather than final rendered images.

**Usage**: TRELLIS.2 reports 38.89 dB PSNR / 0.033 LPIPS on PBR attributes. AssetGen and PBR_Boost also report PBR-specific metrics.

**Relevance to TRELLIS.2**: HIGH. Direct measure of material generation quality.

---

### 9.2 Relighting Consistency

**Definition**: Visual quality of the 3D asset when rendered under different lighting conditions, testing whether PBR materials enable realistic relighting.

**Method**: Render under 3+ diverse HDRI environments and evaluate visual plausibility.

**Usage**: AssetGen, PBR_Boost, DreamMat, MCMat all demonstrate relighting as a quality indicator.

**Relevance to TRELLIS.2**: MEDIUM. TRELLIS.2 outputs PBR materials, so relighting quality is a natural evaluation dimension.

---

### 9.3 Albedo Accuracy

**Definition**: Whether the base color texture represents intrinsic surface color without baked-in lighting/shadows.

**Detection**: Compare rendered albedo under different lighting; albedo should remain constant. High variance indicates baked lighting.

**Relevance to TRELLIS.2**: HIGH. A common issue with feed-forward 3D generation is "baked lighting" in the albedo map.

---

## 10. Dimension 8: Mesh Production Quality

Production quality measures whether the mesh is suitable for use in games, films, or other applications.

### 10.1 Topology Quality

**Metrics**:
- Edge-flow quality (quad-dominant meshes preferred for animation)
- Pole density (vertices with != 4 edges in quad meshes)
- Triangle count vs. detail level
- Triangle aspect ratio distribution (equilateral preferred)

**Relevance to TRELLIS.2**: MEDIUM. TRELLIS.2 generates ~50K triangles, primarily quads after FDG extraction.

---

### 10.2 UV Mapping Quality

**Metrics**:
- **Conformal energy**: Average distortion across all UV triangles (lower is better)
- **UV utilization**: Percentage of UV space covered by islands
- **Texel density**: Variance in texels/unit-area across the mesh (lower variance = more uniform)
- **Seam length**: Total edge length of UV seam cuts
- **Seam visibility**: How visible seams are in rendered output

**Relevance to TRELLIS.2**: HIGH. UV quality directly affects texture rendering. Our postprocess.py handles UV unwrapping via xatlas.

---

### 10.3 Texture Resolution and Mapping (TGE)

**TGE (Textured Geometry Evaluation)**: A perceptual metric that jointly evaluates geometry and texture quality using PointNet++-style feature extraction with Latent-Geometry Set Abstraction blocks.

**Properties**: Does not rely on rendering (operates directly on 3D meshes with texture). Aligns with human perceptual judgments.

**Relevance to TRELLIS.2**: MEDIUM-HIGH. Could provide a render-independent quality signal.

---

### 10.4 Textured Mesh Quality Assessment (TMQA)

**Definition**: Large-scale deep learning quality metric trained on 343K distorted stimuli from 55 source models, with 148,929 quality judgments from 4,500+ participants.

**Distortion Types**: Compression-based distortions on geometry, texture mapping, and texture image.

**Venue**: ACM Transactions on Graphics, 2023.

**Relevance to TRELLIS.2**: MEDIUM. The metric targets distortion detection rather than generation quality, but the framework and dataset are valuable references.

---

## 11. Benchmarks and Datasets

### Summary of Major 3D Quality Benchmarks

| Benchmark | Year | Venue | Scale | Dimensions | Input Types |
|-----------|------|-------|-------|------------|-------------|
| T3Bench | 2024 | arXiv | 300 prompts, 10 methods | Quality + Alignment | Text |
| GPTEval3D | 2024 | CVPR | 5 criteria, ELO ratings | 5 (see 6.1) | Text |
| 3DGen-Bench | 2025 | arXiv | 1,020 prompts, 11,220 models | 5 (see 6.2) | Text + Image |
| 3D Arena | 2025 | arXiv | 100 prompts, 123K votes, 19 models | Overall preference | Image |
| MATE-3D | 2025 | ICCV | 1,280 meshes, 107K annotations | 4 (see 6.3) | Text |
| T23D-CompBench | 2025 | arXiv | 3,600 meshes, 129K ratings | Compositional quality | Text |
| GT23D-Bench | 2024 | arXiv | 400K curated from Objaverse | 10 metrics | Text |
| TMQA | 2023 | ACM TOG | 343K stimuli, 148K judgments | Distortion quality | N/A |
| RealismGrading | 2025 | arXiv | 16 categories, 5K ratings | Shape realism | N/A |

### Commonly Used Evaluation Datasets

| Dataset | Objects | Use |
|---------|---------|-----|
| Objaverse / Objaverse-XL | 800K / 10M+ | Training + evaluation |
| Objaverse++ | Curated subset with quality annotations | Quality-filtered evaluation |
| Google Scanned Objects (GSO) | 1,030 | High-quality reconstruction evaluation |
| Toys4K | ~4K | TRELLIS evaluation set |
| ShapeNet | 51K | Classic shape evaluation |
| Sketchfab Staff Picks | Curated professional | TRELLIS.2 evaluation set |

---

## 12. Metrics Most Used in SOTA Papers

Based on our survey of 20+ papers from CVPR 2024-2025, NeurIPS 2024-2025, ICCV 2025, and SIGGRAPH 2024-2025:

### Tier 1: Nearly Universal (reported by >80% of papers)
| Metric | Type | Direction | Papers Using It |
|--------|------|-----------|-----------------|
| CLIP-S | Alignment | Higher=Better | TRELLIS, AssetGen, InstantMesh, TripoSR, ... |
| LPIPS | Appearance | Lower=Better | TRELLIS, AssetGen, 3DEnhancer, SV3D, ... |
| User Study (pairwise) | Perceptual | Preference % | TRELLIS, TRELLIS.2, AssetGen, ... |
| Chamfer Distance | Geometry | Lower=Better | TRELLIS, AssetGen, InstantMesh, ... |

### Tier 2: Very Common (reported by 40-80% of papers)
| Metric | Type | Direction |
|--------|------|-----------|
| PSNR | Appearance | Higher=Better |
| SSIM | Appearance | Higher=Better |
| F-Score | Geometry | Higher=Better |
| FID/FD | Distributional | Lower=Better |
| Normal Maps (PSNR-N, LPIPS-N) | Geometry+Appearance | Various |

### Tier 3: Emerging / Specialized (reported by <40% but growing)
| Metric | Type | Direction |
|--------|------|-----------|
| DreamSim | Perceptual | Lower=Better |
| MEt3R | Consistency | Higher=Better |
| GPTEval3D / VLM-based | Perceptual | ELO/Score |
| LAION Aesthetic | Aesthetic | Higher=Better |
| ULIP-2 / Uni3D | Alignment | Higher=Better |
| SRAM | Shape Realism | Higher=Better |
| HyperScore | Multi-dimensional | Higher=Better |

---

## 13. Relevance to TRELLIS.2

### TRELLIS.2's Own Evaluation (from paper)
- **Geometry**: Mesh Distance (bidirectional point-to-mesh), F-score, CD, PSNR-N, LPIPS-N
- **Appearance**: PSNR, LPIPS on rendered views
- **PBR**: PSNR and LPIPS on PBR attribute maps (38.89 dB / 0.033)
- **Alignment**: CLIP score, ULIP-2, Uni3D
- **User Study**: ~40 participants, 100 AI-generated image prompts, 66.5% overall preference, 69% normal quality preference

### Most Relevant Metrics for Our Optimization

Given that we are optimizing TRELLIS.2 **without retraining** (parameter tuning + postprocessing only) and **without ground-truth 3D** (only input images as reference):

| Priority | Metric | Why | Implementation Difficulty |
|----------|--------|-----|--------------------------|
| 1 | Silhouette IoU | Primary shape match; cheap; no GT 3D needed | Already implemented |
| 2 | LPIPS (views) | Best single perceptual quality metric | Already implemented |
| 3 | CLIP-S (input alignment) | Measures how well output matches input | Easy to add |
| 4 | DreamSim | Better human alignment than LPIPS | Easy to add (pip install) |
| 5 | Normal consistency | Surface quality independent of texture | Partially implemented |
| 6 | LAION Aesthetic | Cheap aesthetic quality proxy | Easy to add |
| 7 | VLM-based (GPTEval3D-style) | Best human alignment | Medium (requires API) |
| 8 | MEt3R | Multi-view consistency of renders | Medium (requires DUSt3R) |
| 9 | SRAM | No-reference shape realism | Hard (requires PointLLM) |
| 10 | TGE | Joint geometry+texture perceptual quality | Hard (requires custom model) |

### Gaps in Our Current Evaluator

Our current `evaluate.py` (7-dimension, 100-point system) covers:

| Current Dimension | Academic Equivalent | Coverage |
|-------------------|---------------------|----------|
| Silhouette (20pts) | Silhouette IoU | Good |
| Contour (15pts) | Edge F1 (custom) | Reasonable |
| Color (15pts) | Histogram correlation + MAE | Weak (should use CLIP or color space metrics) |
| Detail (10pts) | LPIPS + SSIM | Good |
| Artifacts (15pts) | Custom dark/speckle/hole detection | Unique (no academic equivalent) |
| Smoothness (15pts) | Laplacian + normal consistency | Reasonable |
| Coherence (10pts) | Custom crack/gradient detection | Unique |

**Missing from our evaluator**:
1. **Input alignment** (CLIP-S) -- how well does the 3D match the input image?
2. **Aesthetic quality** (LAION Aesthetic) -- is it visually pleasing?
3. **Multi-view consistency** -- do different views look coherent?
4. **PBR material quality** -- are albedo/roughness/metallic physically plausible?
5. **UV quality** -- seam visibility, texel density uniformity
6. **Depth fidelity** -- does the depth map look correct?
7. **No-reference realism** -- does it look like a real object?

### Recommended Evaluator Improvements

**Phase 1 (Quick, high-impact)**:
1. Add CLIP-S for input alignment (replace or augment Color dimension)
2. Add DreamSim as alternative to LPIPS
3. Add LAION Aesthetic score as bonus dimension

**Phase 2 (Medium effort)**:
4. Add MEt3R-style multi-view consistency check
5. Add UV seam visibility metric (render with checker texture)
6. Add PBR plausibility checks (metallic < 0.3 for non-metals, roughness distribution)

**Phase 3 (Research-grade)**:
7. Integrate HyperScore or 3DGen-Score as learned quality predictor
8. Add VLM-based evaluation (GPTEval3D-style)
9. Add SRAM for no-reference shape realism

---

## 14. Comparison with Our Current Evaluator

### Strengths of Our Current System
- **Fast**: Pure OpenCV/PyTorch, no external model downloads beyond LPIPS
- **Artifact-specific**: Custom dark spot, speckle, crack detection tuned for TRELLIS.2 failure modes
- **Actionable**: Scoring dimensions map directly to parameter optimization targets
- **No GT required**: Works with only input image as reference

### Weaknesses
- **Not human-aligned**: Hand-crafted heuristics, not trained on human judgments
- **Missing semantic understanding**: Does not assess whether the object "makes sense"
- **Single-view evaluation**: Evaluates each view independently, no cross-view consistency
- **No learned features**: Relies on pixel-level analysis (except LPIPS for detail)
- **Color metric is weak**: Histogram correlation is a poor proxy for perceptual color matching

### Academic Grounding Assessment
Our 7 dimensions partially overlap with academic frameworks:

```
Current Evaluator:       GPTEval3D:              3DGen-Bench:
  Silhouette (20)  --->  3D Plausibility    --->  Geometry Plausibility
  Contour (15)     --->  Geometry Details    --->  Geometry Details
  Color (15)       --->  Texture Details     --->  Texture Quality
  Detail (10)      --->  Texture Details     --->  Texture Quality
  Artifacts (15)   --->  3D Plausibility     --->  Geometry Plausibility
  Smoothness (15)  --->  Geometry Details    --->  Geometry Details
  Coherence (10)   --->  Tex-Geom Coherency  --> Geom-Tex Coherence
  (missing)        --->  Text-Asset Alignment --> Prompt-Asset Alignment
```

---

## 15. References

### Benchmarks and Evaluation Frameworks
- [GPTEval3D](https://arxiv.org/abs/2401.04092) -- Wu et al., CVPR 2024. GPT-4V as human-aligned evaluator.
- [3DGen-Bench](https://arxiv.org/abs/2503.21745) -- 3DTopia, 2025. Comprehensive human preference benchmark.
- [3D Arena](https://arxiv.org/abs/2506.18787) -- 2025. Open platform for pairwise 3D evaluation.
- [T3Bench](https://arxiv.org/abs/2310.02977) -- THU, 2024. Text-to-3D benchmark with 300 prompts.
- [MATE-3D / HyperScore](https://arxiv.org/abs/2412.11170) -- ICCV 2025. Multi-dimensional quality evaluator.
- [Rank2Score / T23D-CompBench](https://arxiv.org/abs/2509.23841) -- 2025. Fine-grained rank-learning metric.
- [GT23D-Bench](https://arxiv.org/abs/2412.09997) -- 2024. Curated Objaverse evaluation with 10 metrics.
- [Objaverse++](https://github.com/TCXX/ObjaversePlusPlus) -- ICCV 2025 Workshop. Quality annotations for Objaverse.
- [NerfBaselines](https://arxiv.org/abs/2406.17345) -- 2024. Consistent NVS evaluation framework.

### Perceptual and Learned Metrics
- [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) -- Zhang et al., CVPR 2018. Learned perceptual similarity.
- [DreamSim](https://arxiv.org/abs/2306.09344) -- Fu et al., NeurIPS 2023 Spotlight. Human-aligned similarity.
- [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor) -- LAION. CLIP-based aesthetic scoring.
- [MEt3R](https://arxiv.org/abs/2501.06336) -- Asim et al., CVPR 2025. Multi-view consistency metric.
- [SRAM](https://arxiv.org/abs/2512.01373) -- 2025. No-reference shape realism via LLM bridge.
- [TGE](https://arxiv.org/abs/2512.01380) -- 2025. Perceptual 3D textured shape metric.
- [TMQA](https://dl.acm.org/doi/10.1145/3592786) -- Nehme et al., ACM TOG 2023. Textured mesh quality assessment.
- [Rethinking FID](https://arxiv.org/abs/2401.09603) -- Jayasumana et al., CVPR 2024.

### 3D Generation Methods (with evaluation details)
- [TRELLIS](https://arxiv.org/abs/2412.01506) -- Yang et al., CVPR 2025 Spotlight. CD, F-score, CLIP-S, FD, KD, user study.
- [TRELLIS.2](https://arxiv.org/abs/2512.14692) -- Dec 2025. MD, F-score, PSNR, LPIPS, PBR attributes, user study.
- [Meta 3D AssetGen](https://proceedings.neurips.cc/paper_files/paper/2024/file/123cfe7d8b7702ac97aaf4468fc05fa5-Paper-Conference.pdf) -- NeurIPS 2024. PSNR, LPIPS, Depth L1, Sil-IoU, CD, NC, user study.
- [InstantMesh](https://arxiv.org/abs/2404.07191) -- 2024. CD, F-score, CLIP-S.
- [PBR_Boost](https://arxiv.org/abs/2411.16080) -- SIGGRAPH Asia 2024. PBR relighting quality.
- [SV3D](https://sv3d.github.io/) -- ECCV 2024. LPIPS per-frame consistency.

### Quality Metrics (Geometry)
- [Learnable Chamfer Distance](https://arxiv.org/abs/2312.16582) -- Huang et al., 2024.
- [Point Cloud Utils](https://fwilliams.info/point-cloud-utils/sections/shape_metrics/) -- CD, HD, EMD implementations.
- [MTGNet](https://link.springer.com/article/10.1007/s00366-024-02006-x) -- Multi-label mesh quality (orthogonality, smoothness, distribution).
- [Mesh Quality Survey](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14779) -- CGF 2023. Comprehensive survey of mesh quality indicators.

### Quality Metrics (Texture and Material)
- [Sloyd Texture Quality Metrics](https://www.sloyd.ai/blog/texture-quality-metrics-for-3d-models) -- Practical guide.
- [DreamMat](https://dl.acm.org/doi/10.1145/3658170) -- ACM TOG. PBR material generation evaluation.
- [MCMat](https://arxiv.org/abs/2412.14148) -- Multi-view consistent PBR materials.
- [SuperMat](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_SuperMat_Physically_Consistent_PBR_Material_Estimation_at_Interactive_Rates_ICCV_2025_paper.pdf) -- ICCV 2025. PBR material estimation.
- [TexGaussian](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiong_TexGaussian_Generating_High-quality_PBR_Material_via_Octree-based_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) -- CVPR 2025. PBR via 3DGS.
- [CHORD / Generative Base Material](https://www.ubisoft.com/en-us/studio/laforge/news/1i3YOvQX2iArLlScBPqBZs) -- Ubisoft, SIGGRAPH Asia 2025.

### Perceptual Metrics Comparisons
- [Benchmarking Image Similarity for NVS](https://arxiv.org/abs/2506.12563) -- 2025.
- [Task-Aware NVS Evaluation](https://arxiv.org/abs/2511.12675) -- 2025.
- [Perceptual Metrics Face-Off](https://eureka.patsnap.com/article/perceptual-metrics-face-off-lpips-vs-ssim-vs-psnr) -- LPIPS vs SSIM vs PSNR comparison.
