# Survey: State-of-the-Art 3D Generation Quality Evaluation Metrics (2024-2026)

**Date**: 2026-02-21
**Author**: Research Optimizer Agent
**Scope**: Comprehensive survey of quality metrics, evaluation frameworks, human-preference alignment, and self-improvement methods for 3D mesh generation
**Target**: Identifying actionable metrics for TRELLIS.2's automated optimization pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Perceptual Quality Metrics for 3D Meshes](#2-perceptual-quality-metrics-for-3d-meshes)
3. [Multi-View Consistency Metrics](#3-multi-view-consistency-metrics)
4. [Geometric Quality Metrics](#4-geometric-quality-metrics)
5. [Texture and PBR Material Quality Metrics](#5-texture-and-pbr-material-quality-metrics)
6. [Automated Evaluation Pipelines and Verifier Models](#6-automated-evaluation-pipelines-and-verifier-models)
7. [Human Preference Alignment](#7-human-preference-alignment)
8. [Self-Improvement and Quality-Aware Generation](#8-self-improvement-and-quality-aware-generation)
9. [What Top Papers Actually Use](#9-what-top-papers-actually-use)
10. [Applicability Assessment for TRELLIS.2](#10-applicability-assessment-for-trellis2)
11. [Recommended Evaluator Upgrade Plan](#11-recommended-evaluator-upgrade-plan)
12. [References](#12-references)

---

## 1. Executive Summary

The 3D generation evaluation landscape has undergone a paradigm shift between 2024 and 2026. Key developments:

**1. From single metrics to multi-dimensional frameworks.** The field has moved decisively from reporting isolated metrics (CD, FID) to structured multi-dimensional evaluation. GPTEval3D (CVPR 2024), MATE-3D/HyperScore (ICCV 2025), Hi3DEval (NeurIPS 2025), 3DGen-Bench (2025), and T23D-CompBench (2025) all define 4-5 evaluation dimensions with human-calibrated scoring.

**2. Learned metrics outperform hand-crafted ones.** HyperScore, 3DGen-Score, Rank2Score, and GSOQA all demonstrate significantly higher correlation with human preference than traditional metrics (CLIP-S, LPIPS, SSIM, PSNR). The gap is substantial: Hi3DEval achieves 0.774 Kendall's tau vs GPTEval3D's 0.690 on geometry plausibility.

**3. DreamSim supersedes LPIPS for perceptual quality.** Multiple studies (NeurIPS 2023/2024, NVS benchmarking 2025) confirm DreamSim is more robust and better aligned with human judgment than LPIPS, especially for 3D-rendered content where minor geometric perturbations cause large LPIPS deviations.

**4. VLM-based evaluation is becoming practical.** GPTEval3D, Hi3DEval's M2AP pipeline, VLM3D, and 3DGen-Eval demonstrate that vision-language models (GPT-4V, Qwen2.5-VL, InternVL) can serve as reliable quality judges. Cost is decreasing with open-source VLMs.

**5. Self-improvement via quality metrics is emerging.** DreamDPO (2025), DSO (ICCV 2025), and VLM3D (2025) demonstrate that quality metrics can be used as differentiable reward signals to improve generation quality through DPO/DRO, with massive improvements (+24-48 points on GPTEval3D dimensions).

**6. No-reference metrics are the frontier.** SRAM, GSOQA, and Hi3DEval's 3D-based scorer can assess quality without ground-truth references -- exactly what our optimization scenario requires.

### For TRELLIS.2 Specifically

Our current evaluator (7-dimension, 100-point, heuristic-based) has significant gaps compared to the state of the art:

| Gap | Impact | Fix Difficulty |
|-----|--------|---------------|
| No input alignment metric (CLIP-S/VQAScore) | Missing the most universally reported metric | Easy |
| LPIPS instead of DreamSim | Suboptimal human alignment | Easy |
| No learned quality predictor | Low correlation with human preference | Medium |
| No multi-view consistency metric | Cannot detect Janus/inconsistency artifacts | Medium |
| Heuristic artifact detection instead of learned model | Fragile thresholds | Medium-Hard |
| No PBR material plausibility checks | Missing a whole quality dimension | Easy |
| No aesthetic quality proxy | Missing overall "looks good" signal | Easy |

---

## 2. Perceptual Quality Metrics for 3D Meshes

### 2.1 DreamSim (NeurIPS 2023 Spotlight, updated 2024)

**What it is**: A perceptual similarity metric trained on ~20K human-judged synthetic image triplets, combining CLIP, OpenCLIP, and DINO embeddings with learned fusion weights.

**Architecture**: Concatenates features from CLIP ViT-B/32, OpenCLIP ViT-B/32, and DINO ViT-B/16. Three variants: (1) MLP-tuned on concatenated features (93.4% accuracy), (2) LoRA-tuned on individual models then concatenated (96.2% accuracy), (3) single-model versions.

**Why it matters for 3D evaluation**: DreamSim bridges the gap between low-level metrics (LPIPS/SSIM focus on pixel-level differences) and high-level metrics (CLIP ignores low-level visual quality). For 3D-rendered content specifically, a 2025 NVS benchmarking study showed that DreamSim's similarity scores degrade gracefully with increasing corruption (blur, noise), while LPIPS/SSIM/PSNR show a cliff-edge drop at minimal corruption levels and then plateau -- making them unable to distinguish severity of degradation.

**Key advantage over LPIPS**: DreamSim is more robust to minor geometric perturbations that cause large LPIPS changes but are imperceptible to humans. When a 3D mesh is viewed from a slightly different angle, DreamSim correctly identifies the views as similar, while LPIPS penalizes pixel misalignment.

**Quantitative comparison**: 2-opt human judgment agreement: DreamSim 96.2% vs LPIPS-VGG 80.2% vs SSIM 76.1% vs PSNR 62.8%.

**Implementation**:
```python
pip install dreamsim
from dreamsim import dreamsim
model, preprocess = dreamsim(pretrained=True)
distance = model(img1_tensor, img2_tensor)  # lower = more similar
```

**Applicability to TRELLIS.2**: VERY HIGH. Drop-in replacement for LPIPS in our evaluator. Same input format (two RGB images), same interpretation (lower = better). Requires ~200MB model download.

**Compute cost**: ~15ms per pair on GPU (comparable to LPIPS).

**Correlation with human judgment**: Tau = 0.77 (vs LPIPS 0.58, CLIP 0.52 on NIGHTS dataset).

---

### 2.2 LPIPS (Learned Perceptual Image Patch Similarity)

**What it is**: Deep feature distance between images using pretrained AlexNet or VGG features with learned linear weights.

**Status in 2025-2026**: Still the most widely reported perceptual metric due to inertia. Every major paper (TRELLIS.2, Hunyuan3D, InstantMesh, AssetGen) reports LPIPS. However, its limitations for 3D are well-documented:
- Pixel-aligned: Penalizes geometric imprecision even when perceptually acceptable
- No semantic understanding: Cannot distinguish meaningful texture detail from noise
- Single-scale: Does not capture multi-scale perceptual quality

**When to prefer LPIPS over DreamSim**: When comparing images that are pixel-aligned (same viewpoint, same geometry, different textures only). This is the case for our texture-only optimization experiments.

**Implementation**: Already in our evaluator (`lpips.LPIPS(net='alex')`).

---

### 2.3 SSIM / MS-SSIM

**Status in 2025-2026**: Still reported by convention but increasingly recognized as inadequate for 3D evaluation. The NVS benchmarking study (2025) shows SSIM cannot discriminate between corruption levels.

**MS-SSIM**: Multi-scale variant with slightly better perceptual alignment. Not enough improvement to justify adoption if DreamSim is available.

**Applicability to TRELLIS.2**: LOW priority for changes. Keep in evaluator for baseline comparison but do not weight heavily.

---

### 2.4 PSNR

**Status in 2025-2026**: Reported universally due to convention. Widely acknowledged as poorly correlated with human perception. TRELLIS.2 reports 28.28 dB PSNR on RGB renders and 38.89 dB on PBR attributes.

**Applicability to TRELLIS.2**: LOW. Keep for paper comparisons only.

---

### 2.5 FID / KID

**Status in 2025-2026**: Still used for distributional quality assessment (comparing model outputs to a reference dataset). Not useful for per-sample evaluation. TRELLIS.2 reports FD (Frechet Distance) using Inception-V3, DINOv2, and PointNet++ feature extractors.

**Rethinking FID (CVPR 2024)**: Jayasumana et al. demonstrate FID has systematic biases and propose CMMD (CLIP Maximum Mean Discrepancy) as a more reliable distributional metric.

**Applicability to TRELLIS.2**: NOT APPLICABLE. Our optimization is per-sample, not distributional.

---

### 2.6 LAION Aesthetic Score

**What it is**: A linear MLP on top of CLIP ViT-L/14 embeddings, trained on SAC (176K synthetic), LAION-Logos (15K), and AVA (250K photos) with human aesthetic ratings (1-10 scale).

**Score interpretation**: 1-4 = poor, 5 = acceptable, 6+ = aesthetically good, 8+ = excellent.

**Implementation**:
```python
# Uses CLIP ViT-L/14 features + small linear head
import torch
from transformers import CLIPModel, CLIPProcessor
# Load aesthetic predictor weights from LAION-AI/aesthetic-predictor
# Score = linear_head(clip_image_features).item()
```

**Applicability to TRELLIS.2**: MEDIUM-HIGH. Cheap auxiliary quality signal (~5ms per image). Applied to multi-view renders, provides an "overall visual appeal" score. Does not measure 3D-specific quality but captures whether the output "looks good" to humans. Useful as a tie-breaker or secondary objective.

**Limitation**: Trained on 2D photos, not 3D renders. May undervalue stylized or non-photorealistic content. Scores for typical 3D renders cluster in the 4-6 range.

**Compute cost**: Negligible once CLIP features are computed.

---

### 2.7 VQAScore (ECCV 2024)

**What it is**: Uses a VQA model to compute P("Yes" | image, "Does this show {text}?") as an alignment/quality score. Instantiated with CLIP-FlanT5.

**Key advantage**: Significantly outperforms CLIPScore and PickScore on compositional prompts. Adopted by Google DeepMind (Imagen3/4), ByteDance Seed, and NVIDIA.

**Implementation**: `pip install t2v-metrics` -- one-line evaluation.

**Applicability to TRELLIS.2**: MEDIUM for image-to-3D (requires text description of input). HIGH if we generate captions from input images and use them as quality anchors.

**Compute cost**: ~100ms per evaluation (requires VQA model inference).

---

## 3. Multi-View Consistency Metrics

### 3.1 MEt3R (CVPR 2025)

**What it is**: The first dedicated multi-view 3D consistency metric. Uses DUSt3R for dense 3D reconstruction from image pairs, warps views using the reconstruction, and compares DINO+FeatUp feature maps of warped vs. target images.

**Key innovation**: Does not require camera poses, ground truth geometry, or pixel-level image quality. Measures purely whether multiple views of the same object are geometrically consistent -- i.e., whether they could have been rendered from a single coherent 3D object.

**Architecture**:
1. Input: Two rendered views of the 3D object
2. DUSt3R reconstruction: Dense 3D points from each view
3. View warping: Project view A into view B's coordinate frame
4. Feature comparison: DINO ViT features + FeatUp upsampling
5. Output: Consistency score (higher = more consistent)

**Why it matters**: TRELLIS.2's texture SLAT generates texture in 3D voxel space, but UV baking introduces view-dependent artifacts. MEt3R can detect whether these artifacts create multi-view inconsistencies (e.g., different colors visible from different angles at the same surface point).

**Implementation**: Code available at https://github.com/mohammadasim98/met3r

**Applicability to TRELLIS.2**: HIGH. Can be applied post-generation to rendered views. Detects texture baking artifacts that our current evaluator misses entirely.

**Compute cost**: ~2-3 seconds per pair (DUSt3R inference dominates). For 6 view pairs: ~15 seconds.

**Limitation**: Requires DUSt3R model (~1.5GB). Memory-intensive for high-resolution images.

---

### 3.2 PPLC (Patch-wise Perceptual Likelihood Consistency)

**What it is**: Measures perceptual consistency across views by computing patch-wise LPIPS between corresponding regions warped via known camera parameters.

**Used by**: Free3D (CVPR 2024).

**Applicability to TRELLIS.2**: MEDIUM. Simpler than MEt3R but requires known camera parameters (which we have for our evaluation renders). Less robust to view-dependent effects.

---

### 3.3 Cross-View CLIP Consistency

**What it is**: Compute CLIP embeddings of renders from N viewpoints, then measure pairwise cosine similarity. High mean + low variance = consistent.

**Formula**:
```
CLIP-Consistency = mean(cos(CLIP(view_i), CLIP(view_j))) for all i,j pairs
CLIP-Variance = var(cos(CLIP(view_i), CLIP(view_j)))
```

**Applicability to TRELLIS.2**: MEDIUM. Very cheap to compute (~5ms per view) but captures only semantic-level consistency, not fine-grained geometric consistency.

---

### 3.4 FVD (Frechet Video Distance)

**What it is**: Extension of FID to video sequences using I3D features. Applied to 3D by treating multi-view renders as a video.

**Used by**: SV4D (ICLR 2025) for 4D generation evaluation.

**Applicability to TRELLIS.2**: LOW. Designed for temporal/sequential consistency, not static multi-view. Overkill for our use case.

---

### 3.5 SED / TSED (Stereo Error Distance)

**What it is**: Measures stereo consistency by computing disparity map errors between view pairs.

**Limitation**: Requires camera poses, which MEt3R avoids.

**Applicability to TRELLIS.2**: LOW. MEt3R is strictly superior.

---

## 4. Geometric Quality Metrics

### 4.1 Chamfer Distance (CD)

**Definition**: Average nearest-neighbor distance between two point clouds (bidirectional).

**Variants**:
- **L1-CD**: `(1/|P|) * sum min ||p-q|| + (1/|Q|) * sum min ||q-p||` -- more robust to outliers
- **L2-CD**: Uses squared distances -- penalizes large errors more heavily
- **NCD** (Normal-guided CD, 2025): Weights point distances by normal alignment, improving reconstruction of thin features and sharp edges

**Implementation libraries**:
- PyTorch3D: `pytorch3d.loss.chamfer_distance` (CUDA-accelerated, batched)
- ChamferDistancePytorch: Standalone implementation with F-score
- NVIDIA Kaolin: `kaolin.metrics.pointcloud.chamfer_distance`
- Point Cloud Utils: `point_cloud_utils` pip package

**Applicability to TRELLIS.2**: HIGH for ground-truth comparison, NOT APPLICABLE for our no-reference optimization scenario. We do not have ground-truth 3D meshes for our test examples.

**Compute cost**: ~50ms for 10K points on GPU.

---

### 4.2 F-Score (F1 at threshold tau)

**Definition**: Harmonic mean of precision (predicted points near GT) and recall (GT points near predicted), at distance threshold tau.

**Common thresholds**: tau = 0.01 (1% of bounding box diagonal), tau = 0.05.

**Advantage over CD**: More interpretable -- captures both surface completeness and accuracy. CD can be low even if large portions of the surface are missing, as long as the present portions are accurate.

**Implementation**: Available in PyTorch3D and ChamferDistancePytorch.

**Applicability to TRELLIS.2**: Same as CD -- requires ground truth.

---

### 4.3 Normal Consistency (NC)

**Definition**: Average angular alignment between predicted and ground-truth surface normals at nearest-neighbor points.

**Formula**: `NC = (1/|P|) * sum |n_p . n_NN(p,Q)|`

**Self-contained variant (no GT)**: Measure consistency between adjacent face normals within the same mesh. This is what our `_mesh_smoothness_score` already computes. High NC means smooth surfaces; low NC means bumpy/noisy geometry.

**Applicability to TRELLIS.2**: HIGH. The self-contained variant is already in our evaluator. Could be enhanced by comparing normal maps of rendered views against normal maps predicted by monocular depth models (Depth Anything v2, Marigold).

---

### 4.4 Edge Chamfer Distance

**Definition**: CD computed only on mesh edge points (silhouette edges from specific viewpoints), capturing contour accuracy.

**Applicability to TRELLIS.2**: MEDIUM. Our contour score already captures edge alignment via Canny edge matching, which is a 2D approximation of Edge CD.

---

### 4.5 Hausdorff Distance (HD)

**Definition**: Maximum nearest-neighbor distance -- captures worst-case deviation.

**90th-percentile variant**: More robust to outliers, commonly used in practice.

**Applicability to TRELLIS.2**: LOW. Too sensitive to outliers for optimization guidance.

---

### 4.6 Silhouette IoU (Mask IoU)

**Definition**: IoU of binary masks rendered from multiple viewpoints.

**Status**: Already our primary shape metric (20 points). Well-established in the field.

**Applicability to TRELLIS.2**: HIGH (already implemented, well-calibrated).

---

### 4.7 Depth Error

**Definition**: Per-pixel L1/L2 error between rendered depth maps and reference depth.

**Reference sources for no-GT evaluation**: Monocular depth estimation (Depth Anything v2, Marigold, ZoeDepth) applied to input image provides a pseudo-GT depth map.

**Applicability to TRELLIS.2**: MEDIUM-HIGH. Could compare rendered depth maps against monocular depth predictions from the input image. Captures concavity/convexity that silhouette IoU misses.

**Implementation complexity**: Medium (requires monocular depth model).

---

### 4.8 Manifoldness and Watertightness

**Definition**: Whether the mesh is a valid 2-manifold (every edge shared by exactly 1 or 2 faces) and watertight (no boundary edges).

**Implementation**: `trimesh.is_watertight`, `pymeshlab` manifoldness checks.

**Applicability to TRELLIS.2**: LOW-MEDIUM. TRELLIS.2's FDG extraction generally produces manifold meshes, but decimation can introduce issues. Worth checking as a binary pass/fail.

---

## 5. Texture and PBR Material Quality Metrics

### 5.1 PBR Attribute PSNR/LPIPS

**What it is**: PSNR and LPIPS computed directly on PBR material maps (albedo, roughness, metallic) rather than final shaded renders.

**TRELLIS.2 baselines**: 38.89 dB PSNR / 0.033 LPIPS on PBR attributes.

**Applicability to TRELLIS.2**: HIGH with ground truth, NOT APPLICABLE without it. For no-reference evaluation, see PBR plausibility checks below.

---

### 5.2 PBR Plausibility Checks (No-Reference)

These heuristic checks can be applied without ground truth:

**Albedo plausibility**:
- Albedo should be lighting-independent. Render under different HDRI environments; if albedo map "changes" (has baked shadows), it fails.
- Albedo luminance should be in [30, 240] range for non-emissive materials.
- Albedo should not contain directional shadows or specular highlights.

**Metallic plausibility**:
- Metallic should be approximately binary: 0.0 (dielectric) or 1.0 (metal).
- Values in (0.1, 0.9) are physically implausible for most materials.
- Our postprocess.py already clamps `max_metallic=0.05` for non-metal objects.

**Roughness plausibility**:
- Distribution should match expected material type.
- Very low roughness (<0.1) creates mirror-like surfaces -- rare in nature.
- Our postprocess.py sets `min_roughness=0.2`.

**Implementation**: Simple histogram analysis of PBR maps. ~1ms compute.

**Applicability to TRELLIS.2**: HIGH. Easy to implement, catches physically implausible PBR values.

---

### 5.3 Relighting Consistency

**What it is**: Render the mesh under 3+ diverse HDRI environments and evaluate whether the appearance changes plausibly (PBR materials should respond correctly to different lighting).

**Detection of baked lighting**: If the rendered appearance looks nearly identical under vastly different lighting (because lighting is baked into albedo), the PBR decomposition has failed.

**Metric**: Variance of rendered appearance across environments, normalized by expected variance from PBR model.

**Applicability to TRELLIS.2**: MEDIUM-HIGH. TRELLIS.2's outputs sometimes exhibit baked lighting. We already have envmap rendering infrastructure in `evaluate.py`.

---

### 5.4 UV Quality Metrics

**Texel density variance**: Measure texels per world-space unit across the mesh. Lower variance = more uniform quality.

**UV utilization**: Percentage of UV space covered by actual mesh islands. Higher = better use of texture resolution.

**Seam visibility**: Render with a checker texture; visible seam discontinuities indicate UV quality problems.

**Conformal energy**: Average UV triangle distortion (stretching/compression). Lower = better.

**Implementation**: Accessible via `xatlas` (already used in TRELLIS.2) and `pymeshlab`.

**Applicability to TRELLIS.2**: MEDIUM. UV quality directly impacts final texture appearance. The checker-texture test is particularly cheap and informative.

---

### 5.5 Texture Spatial Frequency (Laplacian Energy)

**What it is**: Measure of texture detail richness via Laplacian energy of the base color map.

**Already implemented**: Our evaluator uses this for the "detail" dimension via `cv2.Laplacian`.

**Interpretation**: Higher energy = more texture detail. But must be normalized by object type (a smooth metallic object should have low energy).

---

### 5.6 FMQM (Field Mesh Quality Metric, 2025)

**What it is**: Novel metric using signed distance fields and color fields to extract four perception-related features: geometry similarity, geometry gradient similarity, space color distribution similarity, and space color gradient similarity.

**Applicability to TRELLIS.2**: MEDIUM. Requires implementation from scratch. More principled than our current heuristics but similar computational complexity.

---

### 5.7 TMQA (Textured Mesh Quality Assessment)

**What it is**: Large-scale deep learning quality metric trained on 343K distorted textured meshes from 55 source models, with 148,929 quality judgments from 4,500+ participants. Published in ACM TOG 2023.

**Distortion types covered**: Texture compression, geometry quantization, UV map quantization, LOD, texture sub-sampling.

**Applicability to TRELLIS.2**: MEDIUM. The model was trained on compression/distortion artifacts, not generation artifacts. May not generalize well to TRELLIS.2's failure modes (dark spots, UV fragmentation, noisy normals).

---

## 6. Automated Evaluation Pipelines and Verifier Models

### 6.1 Hi3DEval (NeurIPS 2025 Datasets and Benchmarks)

**What it is**: The most comprehensive automated 3D quality evaluation framework as of early 2026. Hierarchical evaluation at object-level, part-level, and material-subject level.

**Evaluation dimensions**:

*Object-Level (5 dimensions)*:
1. Geometry Plausibility -- structural integrity, absence of distortions
2. Geometry Details -- fine-scale surface features
3. Texture Quality -- visual realism, resolution, consistency
4. Geometry-Texture Coherency -- shape-appearance alignment
5. Prompt Alignment -- semantic consistency with input

*Part-Level (2 dimensions)*:
1. Geometry Plausibility per part
2. Geometry Details per part

*Material-Subject (4 dimensions)*:
1. Details and Complexity -- texture richness, visual harmony
2. Colorfulness and Saturation -- color diversity and clarity
3. Consistency and Artifacts -- realism under varying lighting, seam detection
4. Material Plausibility -- realistic diffuse/specular effects

**Automated scoring architecture**:
- **Video-based model**: InternVideo2.5 encoder with contrastive learning alignment + quality prediction head. Used for object-level and material evaluations.
- **3D-based model**: PartField geometric embeddings with cross-attention and self-attention. Used for part-level perception.
- **Annotation pipeline (M2AP)**: Multi-agent MLLM pipeline with reflection mechanism (multiple GPT-4-class models cross-verify each other). Achieves L1 loss 0.257 vs 0.702 for single-agent GPT-4.1.

**Performance vs baselines** (Kendall's tau on pairwise agreement):

| Dimension | CLIP | Aesthetic | GPTEval3D | Hi3DEval |
|-----------|------|-----------|-----------|----------|
| Geometry Plausibility | 0.556 | 0.657 | 0.690 | **0.774** |
| Geometry Details | 0.580 | 0.634 | 0.689 | **0.725** |
| Texture Quality | 0.606 | 0.607 | 0.677 | **0.755** |

**Dataset**: Hi3DBench -- 15,300 3D assets with multi-agent annotations.

**Applicability to TRELLIS.2**: VERY HIGH. This is the current state-of-the-art automated evaluator. The video-based scoring model could replace our entire hand-crafted scoring system. However, running InternVideo2.5 + PartField requires significant GPU memory.

**Implementation complexity**: HIGH. Requires InternVideo2.5 and PartField models. Not yet available as a pip package.

**Compute cost**: ~5-10 seconds per asset (model inference).

---

### 6.2 HyperScore / MATE-3D (ICCV 2025)

**What it is**: Multi-dimensional quality evaluator trained on 107,520 human annotations across 1,280 textured meshes from the MATE-3D benchmark.

**Four evaluation dimensions**:
1. Alignment -- semantic consistency with input prompt
2. Geometry -- shape fidelity and structural completeness
3. Texture -- color, material, vibrancy, resolution
4. Overall -- holistic integration

**Architecture**:
- CLIP feature extraction from multi-view renders (6 views)
- Conditional Feature Fusion (CFF): Dimension-specific attention weights
- Adaptive Quality Mapping (AQM): Hypernetwork generates dimension-specific scoring functions

**Key advantage**: Provides absolute scores (not just pairwise rankings), enabling comparison across different inputs. GPTEval3D can only rank outputs for the same prompt.

**Performance**: Competitive with GPTEval3D on pairwise agreement while providing absolute scores. HyperScore leverages CLIP ViT-L/14 features, so it inherits CLIP's limitations for fine-grained texture assessment.

**Applicability to TRELLIS.2**: HIGH. CLIP-based pipeline is lightweight. Multi-view rendering is already in our evaluation workflow. Could be implemented by adding a small learned head on top of CLIP features.

**Implementation complexity**: MEDIUM. Requires training/fine-tuning the hypernetwork, or using pre-trained weights if released.

**Compute cost**: ~100ms per asset (CLIP inference + small MLP).

---

### 6.3 3DGen-Score / 3DGen-Eval (3DGen-Bench, 2025)

**What it is**: Dual evaluation system -- 3DGen-Score is a CLIP-based scoring model, and 3DGen-Eval is a multimodal LLM evaluator with chain-of-thought reasoning.

**Evaluation dimensions** (5, shared with 3DGen-Bench):
1. Geometry Plausibility
2. Geometry Details
3. Texture Quality
4. Geometry-Texture Coherence
5. Prompt-Asset Alignment

**3DGen-Score architecture**: Multi-view CLIP encoding with dimension-specific scoring heads. Trained on 68,000+ expert votes from 56,000+ absolute scores.

**3DGen-Eval**: Uses multimodal LLM (GPT-4V or open-source equivalent) with structured prompts and chain-of-thought reasoning for each dimension.

**Key contribution**: Unifies text-to-3D and image-to-3D evaluation under one framework, with both a fast scoring model (3DGen-Score) and a deliberative evaluator (3DGen-Eval).

**Applicability to TRELLIS.2**: HIGH. The CLIP-based 3DGen-Score is fast enough for our optimization loop. The LLM-based 3DGen-Eval could be used for periodic deep evaluation.

**Compute cost**: 3DGen-Score ~100ms, 3DGen-Eval ~2-5 seconds (LLM inference).

---

### 6.4 GPTEval3D (CVPR 2024)

**What it is**: The first VLM-based 3D evaluation framework, using GPT-4V(ision) for pairwise comparison across 5 criteria.

**Five criteria**:
1. Text-Asset Alignment
2. 3D Plausibility (no Janus, no floaters)
3. Texture Details
4. Geometry Details
5. Texture-Geometry Coherency

**Performance**: Kendall's tau = 0.710 average with human judgments. Pairwise agreement = 0.789.

**ELO system**: Pairwise comparisons aggregated via ELO rating (initial 1000, optimized via MLE).

**Limitation**: Pairwise only (cannot score individual assets). Requires GPT-4V API calls (~$0.01-0.03 per comparison). Superseded by Hi3DEval and 3DGen-Score in accuracy.

**Applicability to TRELLIS.2**: MEDIUM. Useful for pairwise A/B testing of configurations but too expensive and slow for optimization loop use.

---

### 6.5 Rank2Score / T23D-CompBench (2025)

**What it is**: Two-stage rank-learning metric for fine-grained text-to-3D quality assessment.

**Stage 1**: Pairwise ranking via supervised contrastive regression with curriculum learning (easy pairs first, hard pairs later).

**Stage 2**: Mean opinion score (MOS) regression for absolute scoring.

**Dataset**: 3,600 textured meshes from 10 SOTA models, 129,600 human ratings across multiple quality dimensions.

**Key advantage**: Can serve as both an evaluator AND a reward function for generative model optimization.

**Applicability to TRELLIS.2**: MEDIUM-HIGH. The dual ranking+scoring approach is elegant. If weights are released, could be used as our optimization objective.

---

### 6.6 SRAM (Shape-Realism Alignment Metric)

**What it is**: No-reference shape realism metric using an LLM bridge between mesh encoding and realism prediction.

**Architecture**: Point-BERT mesh encoder -> PointLLM reasoning bridge -> MLP realism decoder.

**Dataset**: RealismGrading -- 16 categories, 16 algorithms, 319 annotators, 5,223 ratings.

**Performance**: PLCC = 0.689, SROCC = 0.696, KROCC = 0.566.

**Key advantage**: No-reference (does not need ground truth or input image). Evaluates shape realism purely from the mesh.

**Applicability to TRELLIS.2**: MEDIUM. Requires PointLLM model. Would need to be adapted for our mesh format. Most useful for evaluating geometry quality independent of texture.

**Compute cost**: ~1-2 seconds per mesh (Point-BERT encoding + LLM inference).

---

### 6.7 TGE (Textured Geometry Evaluation)

**What it is**: Perceptual metric for textured 3D shapes using PointNet++ with Latent-Geometry Set Abstraction (LG-SA) blocks.

**Architecture**: Input colored mesh pair -> hierarchical point sampling -> LG-SA feature extraction (joint geometry+color) -> latent space comparison -> fidelity score.

**Key advantage**: Operates directly on 3D meshes with texture -- no rendering required. Jointly evaluates geometry and texture quality.

**Limitation**: Requires a reference mesh (full-reference metric). Not directly applicable to our no-reference scenario.

**Applicability to TRELLIS.2**: LOW for optimization (requires reference). MEDIUM for benchmarking against other methods.

---

### 6.8 GSOQA (3DGS Quality Assessment)

**What it is**: No-reference quality prediction model that operates directly on 3D Gaussian primitives (not rendered images).

**Dataset**: 3DGS-QA -- 225 degraded reconstructions across 15 object types with subjective quality scores.

**Key advantage**: No rendering required, no reference required. Extracts spatial and photometric features directly from the 3D representation.

**Applicability to TRELLIS.2**: LOW directly (designed for 3DGS, not meshes). But the concept of extracting quality features directly from the 3D representation (without rendering) is applicable to TRELLIS.2's sparse voxel representation.

---

### 6.9 Production Pipeline Practices (Meshy, Tripo, Rodin)

Based on industry analysis:

**Common QC dimensions**:
1. Geometry fidelity (silhouette match, topology cleanliness)
2. Topology / hole detection
3. Texture quality (resolution, seam visibility, color accuracy)
4. Editability (clean mesh topology for downstream use)
5. Failure rate / generation success probability
6. Iteration speed

**Evaluation approach**: Hybrid of automated metrics + human artist review. No fully automated production system has been disclosed.

**Rodin (Hyper3D)**: Produces AI Quad Mesh topology with auto PBR textures. Evaluated at 8.5-9.5/10 by professional reviewers.

**Tripo**: Best workflow/editability. Geometry captures form and occlusions well.

**Meshy**: Focus on rapid iteration, lower individual quality.

**Key insight**: Production systems do NOT rely on single metrics. They use a dashboard of 5-8 metrics with human QA oversight. Fully automated scoring is used for filtering/triaging, not final quality determination.

---

### 6.10 3D Arena (2025)

**What it is**: Open platform for pairwise human preference collection. 123,243 votes, 8,096 users, 19 models.

**Key findings**:
- Gaussian splat > mesh by 16.6 ELO
- Textured > untextured by 144.1 ELO
- TRELLIS-3DGS: 1384 ELO, 80.1% preference
- Hunyuan3D-2: 1298 ELO, 65.5% preference
- Automated metrics show "weak correlation" with human preferences

**Applicability to TRELLIS.2**: Contextual (benchmarking position). The weak correlation finding reinforces the need for better automated metrics.

---

## 7. Human Preference Alignment

### 7.1 Metric-Human Correlation Summary

Based on multiple studies (GPTEval3D, MATE-3D, Hi3DEval, 3D Arena, NVS Benchmarking 2025):

| Metric | Human Correlation (Kendall tau) | Type | Notes |
|--------|-------------------------------|------|-------|
| Hi3DEval (video) | 0.725-0.774 | Learned, multi-dim | Best automated |
| GPTEval3D | 0.690-0.710 | VLM pairwise | Requires GPT-4V |
| HyperScore | ~0.65-0.70 | Learned, CLIP-based | Absolute scores |
| DreamSim | ~0.77 | Learned perceptual | 2D similarity only |
| VQAScore | ~0.65 | VQA-based alignment | Best for compositional |
| CLIP-S | 0.52-0.60 | Feature cosine sim | Semantic only |
| LAION Aesthetic | ~0.55 | Learned aesthetic | 2D photos only |
| LPIPS | ~0.58 | Learned perceptual | Pixel-sensitive |
| SSIM | ~0.45-0.50 | Hand-crafted structural | Poor for 3D |
| PSNR | ~0.30-0.40 | Pixel-level | Worst correlation |
| Chamfer Distance | ~0.40-0.50 | Geometric only | Requires GT |

**Key findings**:
1. Learned multi-dimensional metrics (Hi3DEval, HyperScore) dominate
2. Single perceptual metrics (DreamSim > LPIPS > SSIM > PSNR) have clear ordering
3. VLM-based evaluators achieve strong alignment but are expensive
4. Hand-crafted heuristics (like our current evaluator) are not represented in these studies, but likely fall in the 0.3-0.5 range given they are simpler than even CLIP-S

### 7.2 What Humans Actually Care About

From 3D Arena (123K votes) and 3DGen-Bench (68K expert votes):

1. **Texture quality** dominates human preference (144.1 ELO advantage for textured vs untextured)
2. **Overall coherence** matters more than individual dimensions
3. **Geometric plausibility** is a binary threshold: objects must "make sense" or they fail completely
4. **Detail level** has diminishing returns -- moderate detail with clean execution beats high detail with artifacts
5. **Material realism** is increasingly noticed by professional evaluators (Hi3DEval's material dimensions)

---

## 8. Self-Improvement and Quality-Aware Generation

### 8.1 DreamDPO: DPO for Text-to-3D (2025)

**What it is**: Direct Preference Optimization applied to text-to-3D generation via SDS. Constructs pairwise preferences using reward models or VLMs, then optimizes 3D representation with a contrastive loss.

**Reward models used**: HPSv2 (default), ImageReward, QwenVL (for LMM-based ranking).

**Pairwise construction**: At each diffusion timestep, add two different Gaussian noise vectors to the rendered view, denoise both, rank the denoised outputs using the reward model. The higher-scoring output becomes the "preferred" example.

**Quantitative improvements** (vs MVDream baseline on GPTEval3D):

| Dimension | Improvement |
|-----------|------------|
| Text-Asset Alignment | +28.4 |
| 3D Plausibility | +24.4 |
| Texture Details | **+48.4** |
| Geometry Details | **+41.4** |
| Overall | +24.4 |

**Key innovation**: Adaptive threshold mechanism -- when preference score gap is small (< tau=0.001), switches to SDS-style pull-only loss to avoid chaotic gradients from near-equal pairs.

**Applicability to TRELLIS.2**: MEDIUM. DreamDPO targets SDS-based optimization methods, not feed-forward generators. However, the concept of using reward models to construct pairwise preferences is directly applicable to our GA optimization: instead of our hand-crafted score, use HPSv2 or VQAScore as the fitness function.

---

### 8.2 DSO: Direct Simulation Optimization (ICCV 2025)

**What it is**: Aligns 3D generators with physical soundness using simulation feedback. Fine-tunes the 3D generator using stability scores from a physics simulator as the alignment metric.

**Two alignment objectives**:
- **DPO**: Construct pairs of (stable, unstable) outputs, optimize the generator to prefer stable ones
- **DRO** (Direct Reward Optimization): Novel objective that directly uses the stability reward, bypassing pairwise construction. Achieves faster convergence than DPO.

**Key insight**: DRO avoids the need for pairwise construction entirely -- the reward signal is used directly to update the generator weights. This is more sample-efficient than DPO.

**Applicability to TRELLIS.2**: MEDIUM for the specific use case (physical stability), HIGH for the methodology. The DRO objective could be adapted to use quality metrics (HyperScore, LAION Aesthetic, DreamSim) as reward signals for fine-tuning TRELLIS.2's models -- but this requires retraining, which is outside our current scope.

**For our parameter optimization**: The DRO concept translates naturally -- our GA fitness function IS a reward signal, and we're already doing direct optimization on it.

---

### 8.3 VLM3D: VLM as Differentiable Reward (2025)

**What it is**: Integrates Qwen2.5-VL as a differentiable semantic and spatial reward in the SDS pipeline. The VLM provides language-grounded supervision for fine-grained prompt alignment and spatial correctness.

**Key advantage over CLIP-based rewards**: VLMs understand compositional prompts, spatial relationships, and fine-grained semantic content. CLIP's text encoder cannot parse complex prompts.

**Results**: Significantly outperforms prior SDS methods on GPTEval3D benchmark across all dimensions.

**Extension (Nov 2025)**: Applied across both optimization-based (SDS) and feed-forward generation paradigms.

**Applicability to TRELLIS.2**: MEDIUM-HIGH for quality evaluation (Qwen2.5-VL is open-source and can score multi-view renders). LOW for direct reward optimization (requires differentiable path through the VLM into the generator).

**Practical use**: Could replace our evaluator's color/detail dimensions with VLM-based scoring: render 6 views, prompt VLM with "Rate the texture quality, color fidelity, and detail level of this 3D object on a scale of 1-10."

---

### 8.4 DGPO: Direct Group Preference Optimization (2025)

**What it is**: Extension of DPO from pairwise to group preferences. Instead of comparing pairs, compares groups of outputs and optimizes based on fine-grained relative rankings within the group.

**Key advantage**: 20x faster training than Flow-GRPO while achieving superior performance.

**Applicability to TRELLIS.2**: LOW directly (requires retraining). Conceptually relevant: our GA already operates on "groups" (populations) with relative rankings.

---

### 8.5 Rank2Score as Reward Function

**From the T23D-CompBench paper**: Rank2Score can serve as a reward function to optimize generative models, not just evaluate them. The two-stage training (ranking then scoring) produces a metric that is both discriminative (for comparison) and calibrated (for absolute quality estimation).

**Applicability to TRELLIS.2**: HIGH if weights are released. Could replace our hand-crafted fitness function with a learned, human-aligned reward signal.

---

## 9. What Top Papers Actually Use

### 9.1 TRELLIS.2 (CVPR 2025 / Dec 2025)

| Category | Metrics |
|----------|---------|
| Geometry | Mesh Distance (bidirectional), F-score, CD, PSNR-N, LPIPS-N |
| Appearance | PSNR, LPIPS on rendered views |
| PBR | PSNR and LPIPS on PBR attribute maps |
| Alignment | CLIP score, ULIP-2, Uni3D |
| User study | 40 participants, 100 prompts, pairwise preference |

### 9.2 Hunyuan3D 2.0/2.5 (2025)

| Category | Metrics |
|----------|---------|
| Geometry | PSNR-N, LPIPS-N |
| Appearance | PSNR, SSIM, LPIPS |
| Alignment | CLIP score (0.821 for 2.5) |
| User study | Pairwise preference |

### 9.3 InstantMesh (2024)

| Category | Metrics |
|----------|---------|
| Geometry | CD, F-score |
| Alignment | CLIP-S |

### 9.4 TripoSR/SparseFlex (2025)

| Category | Metrics |
|----------|---------|
| Geometry | CD (82% reduction), F-score (88% increase), IoU |
| Appearance | PSNR, LPIPS |

### 9.5 AssetGen (NeurIPS 2024)

| Category | Metrics |
|----------|---------|
| Geometry | PSNR, LPIPS, Depth L1, Sil-IoU, CD, NC |
| User study | Pairwise preference |

### 9.6 Common Pattern

All top papers report:
1. **LPIPS** (universal appearance metric)
2. **CLIP-S** (universal alignment metric)
3. **CD or F-score** (when GT geometry is available)
4. **User study** (pairwise preference, typically N=30-50)

Missing from most: Multi-view consistency, PBR quality, aesthetic quality, production readiness.

---

## 10. Applicability Assessment for TRELLIS.2

### 10.1 Constraints

Our optimization scenario has specific constraints:
1. **No ground-truth 3D**: We evaluate against input images only
2. **Per-sample evaluation**: Not distributional (FID/KID unusable)
3. **Must run in optimization loop**: <10 seconds per evaluation preferred
4. **GPU memory limited**: DGX Spark has ample VRAM but models stack
5. **Must produce actionable signal**: Metrics must differentiate good from bad configurations

### 10.2 Metric Applicability Matrix

| Metric | No-GT? | Per-Sample? | <10s? | Actionable? | Overall |
|--------|--------|-------------|-------|-------------|---------|
| **DreamSim** | Yes | Yes | Yes (15ms) | Yes | **A+** |
| **CLIP-S** | Yes | Yes | Yes (5ms) | Yes | **A+** |
| **LAION Aesthetic** | Yes | Yes | Yes (5ms) | Medium | **A** |
| **VQAScore** | Yes | Yes | Yes (100ms) | Yes | **A** |
| **Silhouette IoU** | Yes | Yes | Yes (1ms) | Yes | **A** |
| **MEt3R** | Yes | Yes | Marginal (15s) | Yes | **B+** |
| **HyperScore** | Yes | Yes | Yes (100ms) | Yes | **B+** |
| **PBR plausibility** | Yes | Yes | Yes (1ms) | Yes | **A** |
| **Normal consistency** | Yes | Yes | Yes (10ms) | Medium | **B+** |
| **Depth error (mono)** | Yes | Yes | Yes (200ms) | Medium | **B** |
| **SRAM** | Yes | Yes | Marginal (2s) | Medium | **B** |
| **Hi3DEval** | Yes | Yes | Marginal (10s) | Yes | **B** |
| **VLM scoring** | Yes | Yes | No (5s+) | Yes | **B-** |
| **GPTEval3D** | Yes | Pairwise | No (API) | Yes | **C+** |
| **CD/F-score** | No | Yes | Yes | Yes | **N/A** |
| **FID/KID** | No | No | N/A | No | **N/A** |
| **TGE** | No | Yes | Yes | Medium | **N/A** |

---

## 11. Recommended Evaluator Upgrade Plan

### Phase 1: Quick Wins (1-2 days implementation, immediate impact)

**Priority 1: Add CLIP-S (Input Alignment)**
- Compute CLIP cosine similarity between input image and 6 rendered views
- Average across views for CLIP-S score
- Weight: 15 points (replace or augment current Color dimension)
- Rationale: The single most commonly reported metric, missing from our evaluator
- Implementation: `pip install open-clip-torch`, ~10 lines of code

**Priority 2: Replace LPIPS with DreamSim**
- Drop-in replacement in `_detail_score`
- `pip install dreamsim`
- Better human alignment at same compute cost
- Keep LPIPS as secondary metric for backward compatibility

**Priority 3: Add LAION Aesthetic Score**
- Apply to 6 rendered views, average
- Weight: 5 bonus points (above 100-point base)
- Implementation: CLIP features + small linear head

**Priority 4: Add PBR Plausibility Checks**
- Metallic histogram analysis (should be ~0.0)
- Roughness distribution check (no extreme values)
- Albedo luminance range check
- Weight: 5 points (within current Artifacts dimension or new dimension)

### Phase 2: Medium Effort (1 week, significant improvement)

**Priority 5: Add VQAScore/VLM Scoring**
- Generate caption from input image (e.g., BLIP-2)
- Use VQAScore to evaluate alignment of rendered views with caption
- Alternative: Direct VLM scoring via Qwen2.5-VL-7B
- Weight: Replace Color dimension (15 points) or add as new dimension

**Priority 6: Add MEt3R Multi-View Consistency**
- Evaluate 3 view pairs (front-left, front-right, left-right)
- Requires DUSt3R model download (~1.5GB)
- Weight: 10 points (new dimension)

**Priority 7: Add Depth Consistency Check**
- Run monocular depth estimation (Depth Anything v2) on input image
- Compare predicted depth with rendered depth map from same viewpoint
- Weight: 5 points (augment Silhouette dimension)

### Phase 3: Learned Evaluator (2-4 weeks, paradigm shift)

**Priority 8: Implement HyperScore-style Learned Evaluator**
- Extract CLIP ViT-L/14 features from 6 rendered views
- Train small MLP per quality dimension on human preference data
- Use MATE-3D, Hi3DBench, or 3DGen-Bench annotations
- Replaces all hand-crafted dimensions

**Priority 9: Add Hi3DEval Video-Based Scoring**
- Render 360-degree turntable video
- Run InternVideo2.5 encoder + quality head
- Highest human alignment of any automated metric

**Priority 10: Reward Function for GA**
- Replace hand-crafted fitness with Rank2Score or HyperScore
- GA optimizes directly for human-aligned quality
- Expected: Much better convergence toward human-preferred configurations

### Revised Scoring System (Phase 1+2 Complete)

| Dimension | Points | Metric(s) | Status |
|-----------|--------|-----------|--------|
| Shape (Silhouette IoU) | 15 | Mask IoU + Depth consistency | Enhanced |
| Contour (Edge alignment) | 10 | Edge F1 | Existing |
| Input Alignment | 15 | CLIP-S + VQAScore | **NEW** |
| Perceptual Quality | 15 | DreamSim + LAION Aesthetic | **Upgraded** |
| Artifacts | 10 | Dark/speckle/hole detection | Existing |
| Surface Smoothness | 10 | Normal consistency + Laplacian | Existing |
| Texture Coherence | 10 | Crack/seam detection | Existing |
| Multi-View Consistency | 10 | MEt3R | **NEW** |
| PBR Plausibility | 5 | Material checks | **NEW** |
| **Total** | **100** | | |

---

## 12. References

### Evaluation Frameworks and Benchmarks

1. **Hi3DEval** (NeurIPS 2025) -- Zhang et al. "Advancing 3D Generation Evaluation with Hierarchical Validity." https://arxiv.org/abs/2508.05609 | https://github.com/3DTopia/Hi3DEval/
2. **GPTEval3D** (CVPR 2024) -- Wu et al. "GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation." https://arxiv.org/abs/2401.04092 | https://github.com/3DTopia/GPTEval3D
3. **MATE-3D / HyperScore** (ICCV 2025) -- Zhang et al. "Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation." https://arxiv.org/abs/2412.11170
4. **3DGen-Bench** (2025) -- 3DTopia. "Comprehensive Benchmark Suite for 3D Generative Models." https://arxiv.org/abs/2503.21745
5. **3D Arena** (2025) -- "An Open Platform for Generative 3D Evaluation." https://arxiv.org/abs/2506.18787
6. **T3Bench** (2024) -- He et al. "Benchmarking Current Progress in Text-to-3D Generation." https://arxiv.org/abs/2310.02977
7. **Rank2Score / T23D-CompBench** (2025) -- "Towards Fine-Grained Text-to-3D Quality Assessment." https://arxiv.org/abs/2509.23841
8. **GT23D-Bench** (2024) -- "Comprehensive General Text-to-3D Generation Benchmark." https://arxiv.org/abs/2412.09997

### Perceptual and Learned Metrics

9. **DreamSim** (NeurIPS 2023 Spotlight) -- Fu et al. "Learning New Dimensions of Human Visual Similarity." https://arxiv.org/abs/2306.09344 | `pip install dreamsim`
10. **LPIPS** (CVPR 2018) -- Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." https://github.com/richzhang/PerceptualSimilarity
11. **VQAScore** (ECCV 2024) -- Lin et al. "Evaluating Text-to-Visual Generation with Image-to-Text Generation." https://github.com/linzhiqiu/t2v_metrics
12. **LAION Aesthetic Predictor** -- https://github.com/LAION-AI/aesthetic-predictor
13. **MEt3R** (CVPR 2025) -- Asim et al. "Measuring Multi-View Consistency in Generated Images." https://github.com/mohammadasim98/met3r
14. **SRAM** (2025) -- "Shape-Realism Alignment Metric for No-Reference 3D Shape Evaluation." https://arxiv.org/abs/2512.01373
15. **TGE** (2025) -- "Textured Geometry Evaluation: Perceptual 3D Textured Shape Metric." https://arxiv.org/abs/2512.01380
16. **TMQA** (ACM TOG 2023) -- Nehme et al. "Textured Mesh Quality Assessment." https://dl.acm.org/doi/10.1145/3592786
17. **GSOQA / 3DGS-QA** (2025) -- "Perceptual Quality Assessment of 3D Gaussian Splatting." https://arxiv.org/abs/2511.08032 | https://github.com/diaoyn/3DGSQA
18. **FMQM** (2025) -- "Textured Mesh Quality Assessment Using Geometry and Color Field Similarity." https://arxiv.org/abs/2505.10824
19. **NVS Metric Benchmarking** (2025) -- "Benchmarking Image Similarity Metrics for Novel View Synthesis Applications." https://arxiv.org/abs/2506.12563
20. **Rethinking FID** (CVPR 2024) -- Jayasumana et al. https://arxiv.org/abs/2401.09603

### Self-Improvement and Quality-Aware Generation

21. **DreamDPO** (2025) -- "Aligning Text-to-3D Generation with Human Preferences via DPO." https://arxiv.org/abs/2502.04370
22. **DSO** (ICCV 2025) -- Li et al. "Aligning 3D Generators with Simulation Feedback for Physical Soundness." https://arxiv.org/abs/2503.22677 | https://github.com/RuiningLi/dso
23. **VLM3D** (2025) -- "Vision-Language Models as Differentiable Semantic and Spatial Rewards for Text-to-3D Generation." https://arxiv.org/abs/2509.15772
24. **DGPO** (2025) -- "Reinforcing Diffusion Models by Direct Group Preference Optimization." https://arxiv.org/abs/2510.08425
25. **VLM Geometry Critic** (2025) -- "Let Language Constrain Geometry." https://arxiv.org/abs/2511.14271

### Geometric Metrics and Libraries

26. **PyTorch3D** -- `pytorch3d.loss.chamfer_distance`, `mesh_normal_consistency` -- https://pytorch3d.org
27. **NVIDIA Kaolin** -- `kaolin.metrics.pointcloud` -- https://kaolin.readthedocs.io
28. **ChamferDistancePytorch** -- https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
29. **NCD** (CGF 2025) -- Li et al. "Normal-Guided Chamfer Distance Loss for Watertight Mesh Reconstruction." https://onlinelibrary.wiley.com/doi/10.1111/cgf.70088

### Texture and PBR Evaluation

30. **Sloyd Texture Quality Metrics** -- "How to Evaluate 3D Model Quality: 7 Key Metrics." https://www.sloyd.ai/blog/top-7-metrics-for-evaluating-3d-model-quality
31. **SeamCrafter** (2025) -- "Enhancing Mesh Seam Generation via Reinforcement Learning." https://arxiv.org/abs/2509.20725
32. **PBR3DGen** (2025) -- "VLM-guided Mesh Generation with High-quality PBR Texture." https://arxiv.org/abs/2503.11368
33. **MeshGen** (CVPR 2025) -- Chen et al. "Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder." CVPR 2025.

### 3D Generation Methods (Evaluation Details)

34. **TRELLIS** (CVPR 2025 Spotlight) -- Yang et al. https://arxiv.org/abs/2412.01506
35. **TRELLIS.2** (Dec 2025) -- https://arxiv.org/abs/2512.14692
36. **Hunyuan3D 2.5** (June 2025) -- https://arxiv.org/abs/2506.16504
37. **AssetGen** (NeurIPS 2024) -- https://proceedings.neurips.cc/paper_files/paper/2024/file/123cfe7d8b7702ac97aaf4468fc05fa5-Paper-Conference.pdf
38. **SparseFlex/TripoSF** (ICCV 2025) -- https://arxiv.org/abs/2503.21732

### Industry and Production

39. **SimInsights** (2025) -- "Is AI Ready for High-Quality 3D Assets?" https://www.siminsights.com/ai-3d-generators-2025-production-readiness/
40. **Scenario** -- "Comparing Generative 3D Models." https://help.scenario.com/en/articles/comparing-generative-3d-models/

---

## Appendix A: Quick Reference -- Metric Implementation Difficulty

| Metric | pip install | Lines of Code | Model Size | GPU Required? |
|--------|-------------|---------------|------------|---------------|
| CLIP-S | `open-clip-torch` | ~15 | 400MB | Yes |
| DreamSim | `dreamsim` | ~10 | 200MB | Yes |
| LAION Aesthetic | Custom | ~20 | 5MB (head only) | With CLIP |
| VQAScore | `t2v-metrics` | ~5 | 3GB | Yes |
| MEt3R | Custom (DUSt3R) | ~50 | 1.5GB | Yes |
| PBR Checks | None | ~30 | 0 | No |
| Depth Consistency | `transformers` | ~30 | 500MB | Yes |
| HyperScore | Custom | ~100+ training | 400MB | Yes |
| Hi3DEval | Custom | ~200+ | 5GB+ | Yes |
| SRAM | Custom | ~100+ | 2GB+ | Yes |

## Appendix B: Metric Correlation Chain

```
Human Preference (Gold Standard)
    |
    |--- Hi3DEval (tau ~0.75)          [Best automated, heavy]
    |--- GPTEval3D (tau ~0.71)         [VLM-based, expensive]
    |--- HyperScore (tau ~0.68)        [Learned, fast]
    |--- DreamSim (tau ~0.77)          [Perceptual, very fast]
    |--- 3DGen-Score (tau ~0.65)       [CLIP-based, fast]
    |--- VQAScore (tau ~0.65)          [VQA-based, medium]
    |--- LAION Aesthetic (tau ~0.55)   [Cheap aesthetic proxy]
    |--- CLIP-S (tau ~0.55)            [Semantic alignment]
    |--- LPIPS (tau ~0.58)             [Pixel-level perceptual]
    |--- SSIM (tau ~0.48)              [Structural, outdated]
    |--- PSNR (tau ~0.35)              [Pixel MSE, baseline]
    |
    |--- Our Current Evaluator (tau ~0.3-0.5?)  [Hand-crafted heuristics]
```

## Appendix C: Priority Action Items

1. **TODAY**: Add CLIP-S and DreamSim to evaluator (2 hours)
2. **THIS WEEK**: Add LAION Aesthetic and PBR plausibility (1 day)
3. **NEXT WEEK**: Add VQAScore or VLM-based scoring (2 days)
4. **THIS MONTH**: Add MEt3R and implement HyperScore-style learned evaluator
5. **ONGOING**: Collect pairwise human preference data on our outputs to calibrate metrics
