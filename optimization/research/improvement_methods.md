# Comprehensive Research: Methods for Improving Image-to-3D Generation Quality

**Date**: 2026-02-19
**Scope**: State-of-the-art methods (2023-2026) for improving TRELLIS.2 3D generation quality
**Author**: Research Optimizer Agent
**Status**: Deep research survey -- no experiments run

---

## Executive Summary

This document surveys 50+ papers and open-source implementations across five key areas for improving image-to-3D generation quality. Each method is evaluated for applicability to TRELLIS.2's architecture (sparse voxel + flow matching transformer + FDG mesh extraction), implementation feasibility, and expected quality impact.

### Architecture Context: TRELLIS.2 Pipeline

```
Input Image --> DINOv3 Features --> Stage 1: Sparse Structure (32^3/64^3 occupancy)
    --> Stage 2: Shape SLAT (geometry features on voxels)
    --> Stage 3: Texture SLAT (PBR attributes on voxels)
    --> FDG Mesh Extraction --> Mesh Simplification --> UV Unwrap --> Texture Bake --> GLB
```

### Key Constraints

- Stage 1 voxel coordinates are immutable in Stages 2-3 (shape/texture cannot add/remove voxels)
- DINOv3 features are view-agnostic (no camera/viewpoint encoding)
- Flow matching with Euler ODE solver (not diffusion DDPM/DDIM)
- CuMesh remesh broken on SM 12.1 (Blackwell GPUs) -- must use remesh=False
- Current best score: ~37.8/100 (GA v2 champion)
- Theoretical ceiling without retraining: ~70/100

### Top-Priority Recommendations (TL;DR)

| Priority | Method | Area | Feasibility | Impact | Section |
|----------|--------|------|-------------|--------|---------|
| 1 | Render-and-Compare Texture Optimization | Texture | Medium | Very High | 2.6 |
| 2 | CFG-MP (Manifold Projection) | Guidance | Easy | High | 3.3 |
| 3 | PBR-SR Texture Super-Resolution | Texture | Medium | High | 2.2 |
| 4 | DetailGen3D Geometry Enhancement | Geometry | Medium | High | 1.3 |
| 5 | UV Seam Smoothing (Laplacian blending) | Post-proc | Easy | Medium | 5.2 |
| 6 | Logit-Normal Noise Schedule | Sampling | Easy | Medium | 3.5 |
| 7 | 3DEnhancer Multi-View Refinement | Texture | Hard | Very High | 2.4 |
| 8 | MeshAnything V2 Retopology | Geometry | Medium | Medium | 1.5 |
| 9 | Material Anything PBR Refinement | Texture | Medium | High | 2.1 |
| 10 | MS3D Multi-Scale Sparse Voxel | Architecture | Hard | Very High | 4.3 |

---

## 1. Geometry Quality Improvements

### 1.1 Unique3D: Multi-View Normal Diffusion + ISOMER Mesh Reconstruction

**Paper**: "Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image"
**Venue**: NeurIPS 2024
**ArXiv**: https://arxiv.org/abs/2405.20343
**Code**: https://github.com/AiuniAI/Unique3D

**Key Technique**: Generates multi-view images AND corresponding normal maps simultaneously via a dual diffusion model. The ISOMER reconstruction algorithm directly deforms meshes using the generated normals as supervision, producing meshes with millions of faces and fine geometric detail.

**Relevance to TRELLIS.2**: TRELLIS.2 generates geometry via sparse voxel latents decoded through FDG mesh extraction. The core insight -- using explicit normal map supervision during mesh extraction -- could be adapted as a post-extraction refinement step. Instead of generating normals from scratch, one could render normal maps from the initial FDG mesh, then use a normal-diffusion model to enhance them, and deform the mesh to match.

**Feasibility**: Medium (requires normal diffusion model or adaptation of existing normals)
**Expected Impact**: High (+3-5 points on silhouette/contour)

### 1.2 MeshFormer: SDF + Surface Rendering with Sparse Voxels

**Paper**: "MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model"
**Venue**: NeurIPS 2024
**ArXiv**: https://arxiv.org/abs/2408.10198
**Code**: https://meshformer3d.github.io/

**Key Technique**: Stores features in 3D sparse voxels (like TRELLIS.2) but combines SDF supervision with surface rendering for direct mesh learning. The key innovation is using projective bias from 3D convolutions alongside transformers, rather than pure attention.

**Relevance to TRELLIS.2**: MeshFormer's use of 3D convolutions for local feature aggregation within sparse voxels is directly applicable. TRELLIS.2's sparse transformer could benefit from hybrid conv-attention layers that better capture local geometric structure. The SDF supervision approach is also relevant -- TRELLIS.2 uses FDG (FlexiCubes/dual graph) mesh extraction, which could potentially be enhanced with SDF-guided vertex positioning.

**Feasibility**: Hard (requires architectural changes to the sparse transformer)
**Expected Impact**: High (+4-6 points on geometry scores)

### 1.3 DetailGen3D: Generative 3D Geometry Enhancement via Data-Dependent Flow

**Paper**: "DetailGen3D: Generative 3D Geometry Enhancement via Data-Dependent Flow"
**Venue**: 2025 (arXiv 2411.16820)
**Code**: https://github.com/VAST-AI-Research/DetailGen3D

**Key Technique**: Models the coarse-to-fine geometric transformation directly through data-dependent flows in latent space. Uses a token matching strategy for accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure. Inference takes only a few seconds.

**Relevance to TRELLIS.2**: This is a direct post-processing geometry enhancer. TRELLIS.2's output meshes tend to be smooth due to the sparse voxel resolution and mesh simplification. DetailGen3D could add fine geometric details (wrinkles, creases, surface patterns) that are lost during the voxel-to-mesh conversion. The DINO feature conditioning aligns well since TRELLIS.2 already extracts DINOv3 features.

**Feasibility**: Medium (pre-trained model available, needs integration into post-processing pipeline)
**Expected Impact**: High (+3-5 points on detail and contour scores)

### 1.4 SuperCarver: Normal Map Diffusion + Noise-Resistant Distance Field Deformation

**Paper**: "SuperCarver: Texture-Consistent 3D Geometry Super-Resolution"
**Venue**: 2025 (arXiv 2503.09439)
**Code**: Not yet released

**Key Technique**: Constructs a deterministic prior-guided normal diffusion model fine-tuned on paired detail-lacking/detail-rich normal map renderings. Uses a noise-resistant inverse rendering scheme through deformable distance field to update mesh surfaces from potentially imperfect normal predictions.

**Relevance to TRELLIS.2**: Directly applicable as a geometry post-processor. Render normal maps from TRELLIS.2 output mesh, enhance them with the normal diffusion model, then deform the mesh to match enhanced normals. The noise-resistant distance field is particularly relevant since generated normals will have some inaccuracies.

**Feasibility**: Medium (requires pre-trained normal diffusion model)
**Expected Impact**: Medium-High (+2-4 points on detail/contour)

### 1.5 MeshAnything V2: Artist-Quality Mesh Topology

**Paper**: "MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization"
**Venue**: ICCV 2025
**ArXiv**: https://arxiv.org/abs/2408.02555
**Code**: https://github.com/buaacyw/MeshAnythingV2

**Key Technique**: Autoregressive transformer that generates clean, artist-quality mesh topology aligned to given shapes. Uses Adjacent Mesh Tokenization (AMT) requiring ~half the token sequence vs. previous methods, supporting up to 1600 faces with clean quad/triangle topology.

**Relevance to TRELLIS.2**: TRELLIS.2's FDG mesh extraction produces triangulated meshes with irregular topology. Running MeshAnything V2 as a post-processor could convert these to clean meshes with proper edge flow. This matters for downstream use (rigging, animation) more than visual quality scores, but better topology can also improve UV unwrapping and texture quality.

**Feasibility**: Medium (pre-trained model available, but 1600-face limit may be too low for detailed objects)
**Expected Impact**: Medium (+1-3 points on quality, significant for production pipelines)

### 1.6 CraftsMan3D: Interactive Normal-Based Geometry Refinement

**Paper**: "CraftsMan3D: High-fidelity Mesh Generation with 3D Native Diffusion and Interactive Geometry Refiner"
**Venue**: CVPR 2025
**ArXiv**: https://arxiv.org/abs/2405.14979
**Code**: https://github.com/wyysf-98/CraftsMan3D

**Key Technique**: Two-stage approach: (1) 3D-native DiT generates coarse geometry from latent space, (2) normal-based geometry refiner enhances surface details using multi-view normals. The refiner works by predicting per-vertex displacements guided by rendered normal maps.

**Relevance to TRELLIS.2**: The normal-based geometry refiner could be applied as a post-processor to TRELLIS.2 meshes. Render normals from multiple views, predict enhanced normals via a diffusion model, then compute vertex displacements to match. This is complementary to the texture refinement pipeline.

**Feasibility**: Medium (geometry refiner code available)
**Expected Impact**: Medium (+2-3 points on silhouette/contour)

### 1.7 Elevate3D: HFS-SDEdit for Texture-Geometry Co-Refinement

**Paper**: "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model"
**Venue**: SIGGRAPH 2025
**ArXiv**: https://arxiv.org/abs/2507.11465
**Project**: https://elevate3d.pages.dev/

**Key Technique**: HFS-SDEdit allows diffusion to freely generate low-frequency content while constraining high-frequency detail from the input image. Uses monocular geometry predictors on the enhanced images to derive geometric cues for mesh refinement. Iterative texture-geometry co-refinement loop.

**Relevance to TRELLIS.2**: This addresses both geometry and texture simultaneously. The core idea of using SDEdit with frequency separation is powerful -- it can refine TRELLIS.2's coarse textures by injecting high-frequency detail from the input image while allowing the diffusion model to clean up low-frequency artifacts.

**Feasibility**: Medium-Hard (requires frequency-domain SDEdit implementation)
**Expected Impact**: High (+3-6 points on detail/color, +1-2 on geometry)

### 1.8 Summary Table: Geometry Methods

| Method | Year | Venue | Technique | Feasibility | Impact | TRELLIS.2 Applicability |
|--------|------|-------|-----------|-------------|--------|------------------------|
| Unique3D | 2024 | NeurIPS | Normal diffusion + ISOMER | Medium | High | Post-processing normal refinement |
| MeshFormer | 2024 | NeurIPS | SDF + sparse voxel conv | Hard | High | Architectural inspiration |
| DetailGen3D | 2025 | arXiv | Data-dependent flow refinement | Medium | High | Direct post-processing enhancer |
| SuperCarver | 2025 | arXiv | Normal diffusion + deformable SDF | Medium | Med-High | Geometry super-resolution |
| MeshAnything V2 | 2025 | ICCV | Autoregressive mesh tokenization | Medium | Medium | Topology improvement |
| CraftsMan3D | 2025 | CVPR | Normal-based vertex displacement | Medium | Medium | Geometry refiner module |
| Elevate3D | 2025 | SIGGRAPH | HFS-SDEdit co-refinement | Med-Hard | High | Texture+geometry enhancement |

---

## 2. Texture Quality Improvements

### 2.1 Material Anything: Feed-Forward PBR Generation

**Paper**: "Material Anything: Generating Materials for Any 3D Object via Diffusion"
**Venue**: CVPR 2025 (Highlight)
**ArXiv**: https://arxiv.org/abs/2411.15138
**Code**: https://github.com/3DTopia/MaterialAnything

**Key Technique**: Feed-forward PBR material generation using a pre-trained image diffusion model with a triple-head architecture (albedo, roughness, metallic) and rendering loss. Uses confidence masks as dynamic switchers for handling textured vs. texture-less objects. Includes a UV-space material refiner for consistent outputs.

**Relevance to TRELLIS.2**: Material Anything could serve as a post-processing step to refine TRELLIS.2's PBR outputs. TRELLIS.2 generates albedo, roughness, and metallic maps in the texture SLAT stage, but these can have artifacts from the voxel-to-UV baking process. Material Anything's confidence-mask approach is particularly useful for handling partially textured regions (e.g., UV seam boundaries).

**Feasibility**: Medium (pre-trained model available, needs render-reproject-refine loop)
**Expected Impact**: High (+3-5 points on color/detail)

### 2.2 PBR-SR: Zero-Shot PBR Texture Super-Resolution

**Paper**: "PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors"
**Venue**: NeurIPS 2025
**ArXiv**: https://arxiv.org/abs/2506.02846

**Key Technique**: Zero-shot PBR texture super-resolution using pre-trained 2D image priors (DiffBIR). Applies 2D prior constraints across multi-view renderings to mitigate view inconsistencies. Incorporates identity constraints in PBR texture domain to ensure fidelity to input. No training or additional data required.

**Relevance to TRELLIS.2**: Direct post-processing enhancement for TRELLIS.2's texture output. Current texture baking produces textures at the pipeline resolution (typically 1024 or 2048). PBR-SR could upscale these to 4096+ with perceptual quality enhancement. The zero-shot nature means no additional training is needed.

**Feasibility**: Medium (requires DiffBIR integration + multi-view rendering loop)
**Expected Impact**: High (+3-5 points on detail score, +1-2 on color)

### 2.3 MVPaint: Synchronized Multi-View Texture Diffusion

**Paper**: "MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D"
**Venue**: CVPR 2025
**ArXiv**: https://arxiv.org/abs/2411.02336
**Code**: https://github.com/3DTopia/MVPaint

**Key Technique**: Three components: (1) Synchronized Multi-view Generation (SMG) for coarse texturing, (2) S3I method for painting unobserved areas, (3) UV Refinement (UVR) module with UV-space super-resolution and spatial-aware seam-smoothing. Produces high-fidelity textures with minimal Janus issues.

**Relevance to TRELLIS.2**: MVPaint's UVR module is directly applicable as a post-processing step. The UV-space super-resolution and seam-smoothing are independent of the generation method. Additionally, MVPaint's synchronized multi-view generation approach could inspire improvements to TRELLIS.2's multi-view fusion strategy.

**Feasibility**: Medium (UVR module can be extracted; full pipeline requires multi-view diffusion)
**Expected Impact**: High (+3-5 points on texture coherence and seams)

### 2.4 3DEnhancer: Consistent Multi-View Diffusion Enhancement

**Paper**: "3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement"
**Venue**: CVPR 2025
**ArXiv**: https://arxiv.org/abs/2412.18565
**Code**: https://huggingface.co/Luo-Yihang/3DEnhancer

**Key Technique**: DiT-based multi-view enhancement model with row attention and epipolar aggregation modules. Supports texture-level editing via text prompts and adjustable noise levels. Compatible with renders from coarse 3D representations.

**Relevance to TRELLIS.2**: Render multi-view images from TRELLIS.2's output mesh, enhance them with 3DEnhancer, then re-project onto the mesh. The adjustable noise level is useful -- lower noise for preserving TRELLIS.2's good outputs, higher noise for fixing problematic areas. The text prompt support enables targeted corrections.

**Feasibility**: Hard (requires 3DEnhancer model + render-enhance-reproject pipeline)
**Expected Impact**: Very High (+5-8 points on texture quality across all dimensions)

### 2.5 Hunyuan3D 2.1: 3D-Aware RoPE for UV Seam Reduction

**Paper**: "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material"
**Venue**: 2025
**ArXiv**: https://arxiv.org/abs/2506.15442
**Code**: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

**Key Technique**: 3D-Aware RoPE embeddings injected into the multiview attention block for enhanced cross-view coherence. Physics-grounded PBR material simulation with illumination-invariant training. Dynamic UV unwrapping with texture refinement.

**Relevance to TRELLIS.2**: Two applicable ideas: (1) The 3D-Aware RoPE concept could improve TRELLIS.2's cross-attention between views by encoding spatial relationships (requires retraining). (2) The UV seam reduction technique (post-processing) is directly applicable -- smooth texture across UV seam boundaries using position-aware blending.

**Feasibility**: Easy (UV seam smoothing) / Hard (3D-Aware RoPE requires retraining)
**Expected Impact**: Medium (UV seam smoothing: +1-2 points) / High (RoPE: +3-5 points, but requires retraining)

### 2.6 Render-and-Compare Texture Optimization (nvdiffrast)

**Paper**: Multiple works (Meta 3D TextureGen, DiffTex, Paint3D)
**Infrastructure**: nvdiffrast already in TRELLIS.2 Dockerfile

**Key Technique**: Differentiable rendering (nvdiffrast) renders the textured mesh from multiple viewpoints, compares against reference images (input image + generated multi-view images), and backpropagates gradients directly to the texture map. Iterative optimization over 50-200 steps typically converges.

**Loss Functions** (in order of importance):
- L1/L2 pixel loss between rendered and reference images
- Perceptual loss (LPIPS/VGG features)
- Style loss (Gram matrix matching)
- Regularization: texture smoothness, normal consistency

**Relevance to TRELLIS.2**: This is the single highest-impact improvement available without retraining. TRELLIS.2's baked textures often have blurriness and color inaccuracies from the voxel-to-UV conversion. Direct optimization of the texture using the input image as reference can dramatically improve fidelity. nvdiffrast is already installed in the Docker container.

**Feasibility**: Medium (nvdiffrast available; need optimization loop with camera poses)
**Expected Impact**: Very High (+5-10 points on color/detail)

### 2.7 TexPainter: Optimization-Based Multi-View Texture Fusion

**Paper**: "TexPainter: Generative Mesh Texturing with Multi-view Consistency"
**Venue**: SIGGRAPH 2024
**ArXiv**: https://arxiv.org/abs/2406.18539
**Code**: https://github.com/Quantuman134/TexPainter

**Key Technique**: Uses DDIM-based optimization to enforce multi-view consistency. Blends textures from different views in color space using visibility- and orientation-weighted averaging. Gradient back-propagation through the latent diffusion process to indirectly modify latent codes.

**Relevance to TRELLIS.2**: The visibility/orientation-weighted blending is applicable to TRELLIS.2's multi-view texture projection. Currently, TRELLIS.2 uses TAPA or concat modes for multi-view fusion during generation. A post-generation TexPainter-style optimization could improve texture consistency across views.

**Feasibility**: Medium-Hard (requires adapting DDIM optimization to TRELLIS.2's flow matching)
**Expected Impact**: Medium-High (+2-4 points on coherence/color)

### 2.8 DiffBIR: Blind Image Restoration as Texture Enhancer

**Paper**: "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior"
**Venue**: ECCV 2024
**ArXiv**: https://arxiv.org/abs/2308.15070
**Code**: https://github.com/XPixelGroup/DiffBIR

**Key Technique**: Two-stage pipeline: (1) restoration module for degradation removal, (2) latent diffusion model for realistic texture generation. Effective at generating natural textures and preserving semantic regions.

**Relevance to TRELLIS.2**: DiffBIR could be applied to rendered views of TRELLIS.2 output to enhance texture quality before reprojection. The two-stage approach is useful: first remove baking artifacts (stage 1), then enhance texture detail (stage 2). This is the same 2D prior used by PBR-SR.

**Feasibility**: Easy (pre-trained model available, straightforward render-enhance-reproject)
**Expected Impact**: Medium (+2-3 points on detail/color)

### 2.9 3DTopia-XL: PrimX Compact PBR Representation

**Paper**: "3DTopia-XL: Scaling High-quality 3D PBR Asset Generation via Primitive Diffusion"
**Venue**: CVPR 2025 (Highlight)
**ArXiv**: https://arxiv.org/abs/2409.12957
**Code**: https://github.com/3DTopia/3DTopia-XL

**Key Technique**: PrimX representation encodes shape, albedo, and material into compact tensorial format. DiT-based generation with Primitive Patch Compression. Generates high-quality PBR assets in 5 seconds.

**Relevance to TRELLIS.2**: The PrimX representation is an alternative to TRELLIS.2's sparse voxel latents. While not directly applicable as a post-processor, the tensorial PBR encoding approach could inspire improvements to how TRELLIS.2's texture SLAT stores and decodes PBR attributes.

**Feasibility**: Hard (requires architectural changes)
**Expected Impact**: High (but requires fundamental representation change)

### 2.10 Summary Table: Texture Methods

| Method | Year | Venue | Technique | Feasibility | Impact | TRELLIS.2 Applicability |
|--------|------|-------|-----------|-------------|--------|------------------------|
| Material Anything | 2025 | CVPR | Triple-head PBR diffusion | Medium | High | PBR refinement post-processor |
| PBR-SR | 2025 | NeurIPS | Zero-shot PBR upscaling | Medium | High | Texture super-resolution |
| MVPaint | 2025 | CVPR | Synchronized MV diffusion | Medium | High | UV refinement module |
| 3DEnhancer | 2025 | CVPR | DiT multi-view enhancement | Hard | Very High | Render-enhance-reproject |
| Hunyuan3D 2.1 | 2025 | - | 3D-Aware RoPE, UV seams | Easy/Hard | Med/High | UV seam smoothing |
| Render-and-Compare | Various | - | nvdiffrast optimization | Medium | Very High | Direct texture optimization |
| TexPainter | 2024 | SIGGRAPH | DDIM optimization fusion | Med-Hard | Med-High | MV texture consistency |
| DiffBIR | 2024 | ECCV | Blind image restoration | Easy | Medium | 2D texture enhancement |
| 3DTopia-XL | 2025 | CVPR | PrimX PBR encoding | Hard | High | Architectural inspiration |

---

## 3. Guidance and Sampling Improvements

### 3.1 CFG-Zero*: Adaptive Scale + Zero-Init for Flow Matching

**Paper**: "CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models"
**ArXiv**: https://arxiv.org/abs/2503.18886
**Code**: https://github.com/WeichenFan/CFG-Zero-star

**Key Technique**: Two modifications to standard CFG:
1. **Optimized Scale (s*)**: s* = (v_cond^T * v_uncond) / ||v_uncond||^2. Replaces fixed scale with adaptive projection.
2. **Zero-Init**: Zero out first K=1-2 ODE solver steps where velocity estimates are unreliable.

**TRELLIS.2 Status**: Already implemented in `cfg_utils.py` and integrated into all 4 CFG sites. GA v1 experiments showed negligible impact, possibly because the implementation is applied at the wrong level or the flow matching dynamics differ from standard diffusion.

**Assessment**: Low priority for further experimentation. The GA v2 experiments confirmed that CFG-Zero* and APG have minimal impact on TRELLIS.2's specific flow matching architecture.

**Feasibility**: Already implemented
**Expected Impact**: Low (0-1 points, confirmed by experiments)

### 3.2 APG: Adaptive Projected Guidance

**Paper**: "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"
**Venue**: ICLR 2025
**ArXiv**: https://arxiv.org/abs/2410.02416

**Key Technique**: Decomposes CFG update into parallel (oversaturation) and orthogonal (quality enhancement) components. Down-weights parallel component.

**TRELLIS.2 Status**: Already implemented. GA experiments showed minimal impact on scores.

**Assessment**: The lack of improvement may be because TRELLIS.2's flow matching dynamics differ from standard diffusion. The oversaturation issue that APG addresses may not be the primary quality bottleneck in TRELLIS.2.

**Feasibility**: Already implemented
**Expected Impact**: Low (0-1 points, confirmed by experiments)

### 3.3 CFG-MP: Manifold Projection for Flow Matching

**Paper**: "Improving Classifier-Free Guidance of Flow Matching via Manifold Projection"
**ArXiv**: https://arxiv.org/abs/2601.21892 (January 2025)

**Key Technique**: Reformulates CFG as homotopy optimization with manifold constraint. Implements manifold projection step via incremental gradient descent during sampling. Uses Anderson Acceleration for efficiency without additional model evaluations. Validated on DiT-XL-2-256, Flux, and Stable Diffusion 3.5.

**Relevance to TRELLIS.2**: Unlike CFG-Zero* and APG which modify the guidance direction/scale, CFG-MP constrains the sampling trajectory to stay on the learned data manifold. This addresses a different failure mode -- off-manifold drift -- which may be more relevant to TRELLIS.2's 3D generation where the data manifold is more complex.

**Key Advantage Over CFG-Zero*/APG**: CFG-Zero* optimizes scale, APG decomposes direction, but CFG-MP constrains the entire trajectory. This is fundamentally more principled for flow matching models.

**Implementation**:
- After each Euler step, project the sample back onto the data manifold
- Manifold approximated by the conditional model's denoised prediction
- Anderson Acceleration avoids extra NFE (neural function evaluations)

**Feasibility**: Easy-Medium (5-15 lines in the Euler sampling loop, no extra model calls)
**Expected Impact**: Medium-High (+2-4 points -- addresses a different mechanism than CFG-Zero*/APG)

### 3.4 Rectified-CFG++: Predictor-Corrector Guidance for Flow Models

**Paper**: "Rectified-CFG++ for Flow Based Models"
**Venue**: CVPR Workshop on Generative Vision 2025
**Project**: https://shreshthsaini.github.io/Rectified-CFGpp/

**Key Technique**: Adaptive predictor-corrector guidance that couples rectified flow's deterministic efficiency with geometry-aware conditioning. Each step: (1) conditional RF update anchoring sample near learned transport path, (2) weighted conditional correction interpolating between conditional and unconditional velocity fields. Trajectories remain within bounded tubular neighbourhood of data manifold.

**Relevance to TRELLIS.2**: Directly applicable to TRELLIS.2's FlowEuler sampler. The predictor-corrector formulation is more robust than single-step Euler with CFG, especially at high guidance scales where TRELLIS.2 performs best (tex_guidance=12.0). The bounded trajectory property could reduce artifacts at high guidance.

**Feasibility**: Medium (requires modifying FlowEuler sampler to add corrector step)
**Expected Impact**: Medium (+1-3 points, especially at high guidance scales)

### 3.5 Logit-Normal Noise Schedule Optimization

**Paper**: "Improved Noise Schedule for Diffusion Training" (ICCV 2025); also used in SD3 and Movie Gen
**ArXiv**: https://arxiv.org/abs/2407.03297

**Key Technique**: Reframe noise schedule design as probability distribution design. Logit-normal sampling concentrates probability mass around critical regions of the schedule (typically log SNR near 0). Importance sampling of the log Signal-to-Noise ratio improves training efficiency and generation quality.

**Relevance to TRELLIS.2**: TRELLIS.2 uses a linear flow matching schedule. Replacing the uniform timestep sampling during inference with a logit-normal distribution would spend more ODE steps in the critical transition region where most structural decisions are made. This is training-free at inference time.

**Implementation**: In the Euler sampler, instead of uniform spacing t in [0, 1], use logit-normal spacing that concentrates steps around t=0.5 (the critical transition region).

**Feasibility**: Easy (3-5 lines change in timestep scheduling)
**Expected Impact**: Medium (+1-3 points from better ODE trajectory)

### 3.6 Consistent Flow Distillation (CFD): 3D-Consistent Noise

**Paper**: "Consistent Flow Distillation for Text-to-3D Generation"
**Venue**: ICLR 2025
**OpenReview**: https://openreview.net/forum?id=A51NEXIq1J

**Key Technique**: Multi-view consistent Gaussian noise rendered from various viewpoints to compute flow gradients. The consistency of 2D image flows across viewpoints is critical for high-quality 3D generation.

**Relevance to TRELLIS.2**: While CFD is designed for SDS-style optimization, the concept of 3D-consistent noise is applicable to TRELLIS.2's texture stage. Instead of independent noise per voxel, use a 3D noise field that is consistent when projected to any view. This could reduce texture seam artifacts.

**Feasibility**: Medium (requires modifying noise initialization in texture SLAT)
**Expected Impact**: Medium (+1-3 points on texture coherence)

### 3.7 Target-Balanced Score Distillation (TBSD)

**Paper**: "Target-Balanced Score Distillation"
**ArXiv**: https://arxiv.org/abs/2511.11710 (November 2025)

**Key Technique**: Formulates SDS as multi-objective optimization with adaptive strategy. Addresses over-saturation and over-smoothing. Yields 3D assets with high-fidelity textures and geometrically accurate shapes.

**Relevance to TRELLIS.2**: Applicable if implementing a post-generation SDS-based texture refinement loop. The adaptive balancing between shape and texture objectives is relevant to the render-and-compare optimization approach.

**Feasibility**: Medium (requires SDS optimization loop)
**Expected Impact**: Medium (+2-3 points in SDS refinement context)

### 3.8 Summary Table: Guidance/Sampling Methods

| Method | Year | Venue | Technique | Feasibility | Impact | Status |
|--------|------|-------|-----------|-------------|--------|--------|
| CFG-Zero* | 2025 | arXiv | Adaptive scale + zero-init | Implemented | Low | Tested, minimal effect |
| APG | 2025 | ICLR | Orthogonal decomposition | Implemented | Low | Tested, minimal effect |
| CFG-MP | 2025 | arXiv | Manifold projection | Easy-Med | Med-High | **Not yet tested** |
| Rect-CFG++ | 2025 | CVPR-W | Predictor-corrector | Medium | Medium | Not yet tested |
| Logit-Normal | 2025 | ICCV | Non-uniform timesteps | Easy | Medium | **Not yet tested** |
| CFD | 2025 | ICLR | 3D-consistent noise | Medium | Medium | Not yet tested |
| TBSD | 2025 | arXiv | Adaptive SDS balancing | Medium | Medium | Not yet tested |

---

## 4. Architecture Improvements

### 4.1 FlashAttention-3: Asynchronous, Low-Precision Attention

**Paper**: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
**ArXiv**: https://arxiv.org/abs/2407.08608

**Key Technique**: Exploits Hopper/Blackwell GPU asynchrony with (1) warp-specialization for compute/data overlap, (2) interleaved block-wise matmul and softmax, (3) FP8 block quantization. Achieves 840 TFLOPs/s BF16 (85% utilization), 1.3 PFLOPs/s FP8 on H100.

**Relevance to TRELLIS.2**: TRELLIS.2 currently uses flash-attn v2.7.4. Upgrading to FlashAttention-3 could provide 1.5-2x speedup, enabling more sampling steps (from 16 to 24+) within the same time budget, or supporting higher resolution (1536+ instead of 1024).

**Caveat**: FlashAttention-3 targets Hopper (SM 9.0). DGX Spark uses Blackwell (SM 12.1). Need to verify FA3 compatibility with SM 12.1.

**Feasibility**: Easy (pip install upgrade, if SM 12.1 supported)
**Expected Impact**: Indirect -- enables more steps/higher resolution within same time budget

### 4.2 Dynamic Sparse Voxel Attention (DSVA)

**Paper**: "Dynamic Sparse Voxel Attention for Efficient Transformers" (Stanford CS231n 2025)
**Also**: DSVT (CVPR 2023)

**Key Technique**: Learns a sparsity predictor from low-rank query/key projections to mask out non-salient attention pairs. Achieves up to 95% runtime sparsity with minimal accuracy loss. Dynamic partitioning of local regions based on sparsity.

**Relevance to TRELLIS.2**: TRELLIS.2's sparse transformer operates on ~9.6K voxel tokens. Most voxels only need to attend to their spatial neighbors for geometry/texture, not all other voxels. Dynamic sparse attention could allow the model to process more voxels (higher resolution) within the same compute budget.

**Feasibility**: Hard (requires modifying the sparse transformer architecture)
**Expected Impact**: Indirect -- enables higher resolution, potentially +2-4 points on detail

### 4.3 MS3D: Multi-Scale Sparse Voxel Generation

**Paper**: "MS3D: High-Quality 3D Generation via Multi-Scale Representation Modeling"
**Venue**: ICCV 2025

**Key Technique**: Decomposes geometric reconstruction into progressive multi-scale modeling -- from low-resolution full latents to high-resolution sparse structured latents. Aggregates hierarchical structured latents from all levels to define SDF in continuous space.

**Relevance to TRELLIS.2**: This is the most architecturally relevant improvement for TRELLIS.2. The current single-scale sparse voxel approach (32^3 or 64^3) limits geometric detail. MS3D's progressive refinement -- generating 16^3 first, then refining to 32^3, then 64^3 -- could dramatically improve quality by allowing the model to allocate resolution where needed.

**Feasibility**: Hard (requires architectural redesign and retraining)
**Expected Impact**: Very High (+5-10 points on all geometry dimensions)

### 4.4 FlexAttention: Custom Sparse Attention Patterns

**Paper**: PyTorch FlexAttention (2024)

**Key Technique**: Compiles custom score_mod operators into fused attention kernels. Supports arbitrary sparse patterns (block masks, causal, sliding window, etc.) with FlashAttention-level performance.

**Relevance to TRELLIS.2**: TRELLIS.2's sparse transformer could use FlexAttention to implement spatially-aware attention patterns where voxels primarily attend to their 3D neighbors with a small global attention budget. This could improve quality by focusing attention on local geometric features.

**Feasibility**: Medium (requires replacing attention implementation)
**Expected Impact**: Medium (+1-3 points from better attention patterns)

### 4.5 Hunyuan3D 2.1 3D-Aware RoPE

**Paper**: Hunyuan3D 2.1 (see Section 2.5)

**Key Technique**: Injects 3D spatial information into rotary position embeddings for cross-view coherence in the multiview attention block.

**Relevance to TRELLIS.2**: TRELLIS.2 already uses position encoding in its sparse transformer. Replacing standard position encoding with 3D-aware RoPE that encodes actual voxel coordinates could improve spatial coherence. However, this requires retraining.

**Feasibility**: Hard (requires retraining)
**Expected Impact**: High (+3-5 points on coherence/smoothness)

### 4.6 GaussianAnything: Cascaded Latent Diffusion with Shape-Texture Disentanglement

**Paper**: "GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation"
**Venue**: ICLR 2025
**Code**: https://github.com/NIRVANALAN/GaussianAnything

**Key Technique**: Cascaded pipeline with explicit shape-texture disentanglement. Multi-modal conditioning (point cloud, caption, single/multi-view images). Latent space preserves 3D shape information via RGB-D-Normal renderings as VAE input.

**Relevance to TRELLIS.2**: TRELLIS.2 already uses a cascaded approach (sparse structure -> shape -> texture). GaussianAnything's explicit shape-texture disentanglement in the latent space could improve TRELLIS.2's texture quality by reducing shape-texture entanglement in the SLAT representation.

**Feasibility**: Hard (requires VAE redesign)
**Expected Impact**: High (but requires fundamental changes)

### 4.7 Summary Table: Architecture Methods

| Method | Year | Venue | Technique | Feasibility | Impact | TRELLIS.2 Applicability |
|--------|------|-------|-----------|-------------|--------|------------------------|
| FlashAttention-3 | 2024 | arXiv | Async + FP8 attention | Easy | Indirect | Speed -> more steps |
| DSVA | 2025 | Stanford | Dynamic sparse attention | Hard | Indirect | Higher resolution |
| MS3D | 2025 | ICCV | Multi-scale sparse voxels | Hard | Very High | Progressive refinement |
| FlexAttention | 2024 | PyTorch | Custom attention patterns | Medium | Medium | Spatial attention |
| 3D-Aware RoPE | 2025 | Hunyuan | 3D position encoding | Hard | High | Requires retraining |
| GaussianAnything | 2025 | ICLR | Cascaded disentanglement | Hard | High | Latent design inspiration |

---

## 5. Post-Processing Improvements

### 5.1 Render-and-Compare Texture Optimization (Detail)

Building on Section 2.6, here is a detailed implementation plan for the highest-priority post-processing improvement.

**Components Required**:
1. **Camera pose estimation**: TRELLIS.2 already knows the input view direction. For multi-view, use the same poses as multi-view generation.
2. **Differentiable renderer**: nvdiffrast (already installed in Docker container)
3. **Loss functions**: L1, perceptual (VGG/LPIPS), and optionally style loss
4. **Optimizer**: Adam on the texture map, 100-200 iterations
5. **Multi-view consistency**: Optimize from multiple views simultaneously

**Pseudo-code**:
```python
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()
texture = mesh.texture_map.clone().requires_grad_(True)
optimizer = torch.optim.Adam([texture], lr=0.01)

for i in range(200):
    # Render from input viewpoint
    rendered = render_mesh(mesh, texture, camera_pose, glctx)

    # Compare to input image
    loss_l1 = F.l1_loss(rendered, input_image)
    loss_perceptual = lpips_fn(rendered, input_image)
    loss_smooth = texture_smoothness(texture)

    loss = loss_l1 + 0.1 * loss_perceptual + 0.01 * loss_smooth
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

mesh.texture_map = texture.detach()
```

**Feasibility**: Medium (need to set up rendering pipeline with correct camera matrices)
**Expected Impact**: Very High (+5-10 points on color/detail from front view, +2-5 on other views)

### 5.2 UV Seam Smoothing via Laplacian Blending

**Technique**: Identify UV seam edges, dilate the texture past UV island boundaries, and blend at seams using Laplacian smoothing or Poisson blending.

**Implementation**:
1. Build UV adjacency map (which texels are across seam boundaries)
2. For each seam boundary texel, average with its across-seam neighbor
3. Apply Gaussian blur in a narrow band around seams (2-4 texels wide)
4. Leave interior texels untouched

**Relevance to TRELLIS.2**: UV seams are a consistent source of artifacts in TRELLIS.2's output, especially visible on smooth surfaces where color discontinuities are jarring. This is one of the simplest improvements with guaranteed benefit.

**Feasibility**: Easy (pure image processing on UV texture map, ~50 lines of code)
**Expected Impact**: Medium (+1-2 points on smoothness/coherence)

### 5.3 Improved UV Unwrapping: PartUV / ArtUV

**Papers**:
- "PartUV: Part-Based UV Unwrapping" (2025)
- "ArtUV: Artist-style UV Unwrapping" (2025)

**Key Technique**: PartUV uses learned part priors with geometric cues to generate part-aligned charts, reducing fragmentation and color bleeding. ArtUV produces semantically meaningful UV islands outperforming xatlas.

**Relevance to TRELLIS.2**: TRELLIS.2 currently uses xatlas for UV unwrapping, which produces overly fragmented segments. Replacing xatlas with PartUV or ArtUV would reduce the number of UV islands, minimize padding between charts, and reduce color bleeding artifacts at lower texture resolutions.

**Feasibility**: Medium (requires integrating new UV library, may have compatibility issues)
**Expected Impact**: Medium (+1-3 points on texture quality due to fewer seams)

### 5.4 Normal Map Enhancement via Bilateral Filtering

**Technique**: Apply bilateral filtering to baked normal maps to denoise high-frequency noise while preserving geometric edges. Then optionally attenuate normal map strength for smoother shading.

**Current Status**: Already partially implemented in TRELLIS.2's postprocess.py with:
- Normal strength attenuation (normal_strength=0.7)
- Normal map Gaussian blur (normal_blur_sigma=1.5)

**Potential Improvement**: Replace Gaussian blur with bilateral filtering (preserves edges better). Use surface curvature to adaptively vary filter strength -- smooth on flat regions, preserve on edges.

**Feasibility**: Easy (replace cv2.GaussianBlur with cv2.bilateralFilter)
**Expected Impact**: Low (+0.5-1 point improvement over current approach)

### 5.5 Ambient Occlusion Quality Improvement

**Current Status**: Already implemented using cuBVH ray tracing (16 rays, radius=0.03).

**Potential Improvements**:
1. **More rays**: Increase from 16 to 64 or 128 for smoother AO
2. **Screen-space AO baking**: Render AO from multiple views and blend in UV space
3. **Differential AO algorithm**: Use normal map gradient information for AO estimation (no ray tracing needed, faster)
4. **Bent normals**: In addition to AO scalar, compute bent normal direction for better indirect lighting approximation

**Feasibility**: Easy (increase ray count) / Medium (differential AO)
**Expected Impact**: Low (+0.5-1 point, AO is already working)

### 5.6 PBR Material Clamping and Correction

**Current Status**: Already implemented with max_metallic=0.05 and min_roughness=0.2.

**Potential Improvements**:
1. **Energy conservation check**: Ensure albedo * (1 - metallic) + F0 * metallic <= 1.0
2. **Roughness from normal map**: Derive roughness variations from normal map frequency content (high-frequency normals -> higher roughness for micro-surface scattering)
3. **Metallic edge cleanup**: At mesh silhouette edges, force metallic to 0 to avoid Fresnel artifacts

**Feasibility**: Easy (pure post-processing math)
**Expected Impact**: Low (+0.5-1 point on material quality)

### 5.7 Mesh Decimation with Feature Preservation

**Current Status**: TRELLIS.2 decimates to a target face count (default varies).

**Potential Improvements**:
1. **Error-weighted decimation**: Weight edge collapse error by texture gradient magnitude -- preserve edges where texture changes rapidly
2. **Curvature-adaptive decimation**: Allocate more triangles to high-curvature regions
3. **Remeshing post-decimation**: After decimation, apply isotropic remeshing to regularize triangle shape/size distribution

**Feasibility**: Easy-Medium (most algorithms available in trimesh/PyMeshLab)
**Expected Impact**: Low-Medium (+1-2 points on geometry quality)

### 5.8 Summary Table: Post-Processing Methods

| Method | Technique | Feasibility | Impact | Status |
|--------|-----------|-------------|--------|--------|
| Render-and-Compare | nvdiffrast optimization | Medium | Very High | Not yet implemented |
| UV Seam Smoothing | Laplacian blending | Easy | Medium | Not yet implemented |
| PartUV/ArtUV | Better UV unwrapping | Medium | Medium | Not yet implemented |
| Normal Bilateral Filter | Edge-preserving smoothing | Easy | Low | Partially implemented |
| AO Improvement | More rays / differential | Easy | Low | Partially implemented |
| PBR Correction | Energy conservation | Easy | Low | Partially implemented |
| Mesh Decimation | Feature-aware collapse | Easy-Med | Low-Med | Not yet implemented |

---

## 6. Prioritized Roadmap for TRELLIS.2

Based on feasibility, expected impact, and dependencies, here is the recommended order of implementation:

### Phase A: Quick Wins (1-2 days each, no retraining)

**A1. UV Seam Smoothing** (Section 5.2)
- Effort: ~50 lines of Python
- Expected: +1-2 points on smoothness/coherence
- Dependencies: None
- Risk: Very low

**A2. Logit-Normal Timestep Scheduling** (Section 3.5)
- Effort: ~5 lines change in Euler sampler
- Expected: +1-3 points from better ODE trajectory
- Dependencies: None
- Risk: Low (can revert if no improvement)

**A3. CFG-MP Manifold Projection** (Section 3.3)
- Effort: ~15 lines in Euler sampling loop
- Expected: +2-4 points from trajectory correction
- Dependencies: None
- Risk: Low (different mechanism from already-tested CFG-Zero*/APG)

**A4. Normal Map Bilateral Filtering** (Section 5.4)
- Effort: ~5 lines change (replace GaussianBlur with bilateralFilter)
- Expected: +0.5-1 point
- Dependencies: None
- Risk: Very low

### Phase B: Medium Effort (3-7 days each, no retraining)

**B1. Render-and-Compare Texture Optimization** (Section 5.1 / 2.6)
- Effort: ~200-300 lines of Python, camera pose setup
- Expected: +5-10 points on color/detail
- Dependencies: nvdiffrast (already installed), LPIPS
- Risk: Medium (camera pose accuracy critical)

**B2. PBR-SR Texture Super-Resolution** (Section 2.2)
- Effort: Integrate DiffBIR + multi-view rendering loop
- Expected: +3-5 points on detail
- Dependencies: DiffBIR model download (~2GB)
- Risk: Low (zero-shot, no training needed)

**B3. DetailGen3D Geometry Enhancement** (Section 1.3)
- Effort: Integrate pre-trained model into pipeline
- Expected: +3-5 points on detail/contour
- Dependencies: DetailGen3D model download
- Risk: Medium (may not generalize to all object types)

**B4. Improved UV Unwrapping** (Section 5.3)
- Effort: Replace xatlas with PartUV/ArtUV
- Expected: +1-3 points on texture quality
- Dependencies: PartUV library
- Risk: Medium (compatibility with FDG mesh output)

### Phase C: High Effort (1-2 weeks each, may require retraining)

**C1. 3DEnhancer Multi-View Enhancement** (Section 2.4)
- Effort: Full render-enhance-reproject pipeline
- Expected: +5-8 points on all texture dimensions
- Dependencies: 3DEnhancer model, multi-view rendering
- Risk: Medium (multi-view consistency)

**C2. Material Anything PBR Refinement** (Section 2.1)
- Effort: Integration of Material Anything pipeline
- Expected: +3-5 points on color/detail
- Dependencies: Material Anything model
- Risk: Low (pre-trained, works on any mesh)

**C3. CraftsMan3D Geometry Refiner** (Section 1.6)
- Effort: Integrate normal-based geometry refiner
- Expected: +2-3 points on geometry
- Dependencies: CraftsMan3D refiner model
- Risk: Medium (mesh topology compatibility)

### Phase D: Architectural Changes (weeks-months, requires retraining)

**D1. MS3D Multi-Scale Sparse Voxels** (Section 4.3)
- Effort: Architectural redesign + retraining
- Expected: +5-10 points on all dimensions
- Risk: High (requires significant engineering + training compute)

**D2. 3D-Aware RoPE** (Section 4.5)
- Effort: Position encoding replacement + retraining
- Expected: +3-5 points on coherence
- Risk: Medium (proven technique, but retraining cost)

**D3. FlexAttention Spatial Patterns** (Section 4.4)
- Effort: Attention implementation replacement + fine-tuning
- Expected: +1-3 points from better attention
- Risk: Medium

---

## 7. Cross-Cutting Themes and Insights

### 7.1 The Render-Refine-Reproject Paradigm

Multiple top-performing methods (3DEnhancer, MVPaint, Material Anything, PBR-SR, TexPainter) share a common pattern:
1. Render the 3D model from multiple viewpoints
2. Refine the 2D renders using powerful 2D priors (diffusion models, super-resolution)
3. Reproject the enhanced 2D images back onto the 3D mesh

This paradigm leverages the fact that 2D generative models are far more mature than 3D ones. For TRELLIS.2, implementing this pattern once (render-enhance-reproject infrastructure) enables plugging in multiple enhancement methods.

### 7.2 Normal Maps as the Bridge Between Geometry and Texture

Normal map diffusion (SuperCarver, CraftsMan3D, Unique3D) has emerged as the dominant approach for geometry enhancement. Normal maps are:
- Easy to render from any viewpoint
- Can be enhanced by 2D diffusion models trained on normal map datasets
- Can be used to deform meshes via inverse rendering
- Bridge the geometry-texture gap (carry geometric information in image format)

### 7.3 UV Space is a Bottleneck

Multiple papers (MVPaint, Hunyuan3D 2.1, PartUV) identify UV unwrapping as a quality bottleneck:
- Fragmented UV charts (xatlas) cause color bleeding at low resolutions
- UV seams are visible on smooth surfaces
- UV space inefficiency wastes texture resolution

The solution is either better UV unwrapping (PartUV, ArtUV) or UV-aware post-processing (seam smoothing, UV-space super-resolution).

### 7.4 Flow Matching Guidance is Fundamentally Different from Diffusion

The relatively low impact of CFG-Zero* and APG on TRELLIS.2 (compared to their success on standard image diffusion models) suggests that flow matching guidance operates differently. Key differences:
- Flow matching uses velocity prediction, not noise/score prediction
- The ODE trajectory is more deterministic (less noise injection)
- CFG's effect on flow matching may be more about trajectory correction than noise reduction

This motivates exploring CFG-MP (manifold projection) and Rectified-CFG++ (predictor-corrector), which are specifically designed for flow matching.

### 7.5 Shape Quality is Bottlenecked by Stage 1

GA experiments confirmed that shape SLAT guidance has negligible impact (r<0.1 correlation). This means geometry quality is primarily determined by:
1. Stage 1 (sparse structure) -- determines which voxels exist
2. Mesh extraction (FDG) -- converts voxels to mesh
3. Post-processing -- decimation, smoothing, normal baking

Improving shape requires either: better Stage 1 (higher resolution, more steps, better guidance) or post-generation geometry refinement (DetailGen3D, SuperCarver, CraftsMan3D).

---

## 8. Complete Paper Reference List

### Geometry (Section 1)
1. Unique3D (NeurIPS 2024) - https://arxiv.org/abs/2405.20343
2. MeshFormer (NeurIPS 2024) - https://arxiv.org/abs/2408.10198
3. DetailGen3D (2025) - https://arxiv.org/abs/2411.16820
4. SuperCarver (2025) - https://arxiv.org/abs/2503.09439
5. MeshAnything V2 (ICCV 2025) - https://arxiv.org/abs/2408.02555
6. CraftsMan3D (CVPR 2025) - https://arxiv.org/abs/2405.14979
7. Elevate3D (SIGGRAPH 2025) - https://arxiv.org/abs/2507.11465

### Texture (Section 2)
8. Material Anything (CVPR 2025) - https://arxiv.org/abs/2411.15138
9. PBR-SR (NeurIPS 2025) - https://arxiv.org/abs/2506.02846
10. MVPaint (CVPR 2025) - https://arxiv.org/abs/2411.02336
11. 3DEnhancer (CVPR 2025) - https://arxiv.org/abs/2412.18565
12. Hunyuan3D 2.1 (2025) - https://arxiv.org/abs/2506.15442
13. TexPainter (SIGGRAPH 2024) - https://arxiv.org/abs/2406.18539
14. DiffBIR (ECCV 2024) - https://arxiv.org/abs/2308.15070
15. 3DTopia-XL (CVPR 2025) - https://arxiv.org/abs/2409.12957
16. Paint3D (CVPR 2024) - https://github.com/OpenTexture/Paint3D

### Guidance & Sampling (Section 3)
17. CFG-Zero* (2025) - https://arxiv.org/abs/2503.18886
18. APG (ICLR 2025) - https://arxiv.org/abs/2410.02416
19. CFG-MP (2025) - https://arxiv.org/abs/2601.21892
20. Rectified-CFG++ (CVPR-W 2025) - https://shreshthsaini.github.io/Rectified-CFGpp/
21. Logit-Normal Schedule (ICCV 2025) - https://arxiv.org/abs/2407.03297
22. Consistent Flow Distillation (ICLR 2025) - https://openreview.net/forum?id=A51NEXIq1J
23. TBSD (2025) - https://arxiv.org/abs/2511.11710
24. GSD: Geometry-Aware Score Distillation (2024) - https://arxiv.org/abs/2406.16695

### Architecture (Section 4)
25. FlashAttention-3 (2024) - https://arxiv.org/abs/2407.08608
26. DSVT (CVPR 2023) - https://arxiv.org/abs/2301.06051
27. MS3D (ICCV 2025) - ICCV 2025 proceedings
28. FlexAttention (PyTorch 2024) - PyTorch blog
29. GaussianAnything (ICLR 2025) - https://arxiv.org/abs/2411.08033
30. VoxFormer (CVPR 2023) - https://arxiv.org/abs/2302.12251
31. CGFormer (NeurIPS 2024) - https://github.com/pkqbajng/CGFormer

### Post-Processing (Section 5)
32. PartUV (2025) - https://liner.com/review/partuv-partbased-uv-unwrapping-3d-meshes
33. ArtUV (2025) - https://openreview.net/pdf/5553f526a0ae17b6ff3174882e9913bf8a5bda1c.pdf
34. Bilateral Mesh Denoising - https://dl.acm.org/doi/10.1145/882262.882368
35. MeshGen (CVPR 2025) - https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_MeshGen_Generating_PBR_Textured_Mesh_with_Render-Enhanced_Auto-Encoder_and_Generative_CVPR_2025_paper.pdf
36. WonderZoom (2025) - https://arxiv.org/abs/2512.09164

### Additional References
37. DreamMesh (2024) - https://arxiv.org/abs/2409.07454
38. Direct3D (NeurIPS 2024) - https://arxiv.org/abs/2405.14832
39. InstantMesh (2024) - https://arxiv.org/abs/2404.07191
40. Meta 3D AssetGen (NeurIPS 2024) - https://proceedings.neurips.cc/paper_files/paper/2024/file/123cfe7d8b7702ac97aaf4468fc05fa5-Paper-Conference.pdf
41. Hunyuan3D 2.0 (2025) - https://arxiv.org/abs/2501.12202
42. Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN
43. MaterialMVP (ICCV 2025) - https://openaccess.thecvf.com/content/ICCV2025/papers/He_MaterialMVP_Illumination-Invariant_Material_Generation_via_Multi-view_PBR_Diffusion_ICCV_2025_paper.pdf
44. GTR (2024) - https://arxiv.org/abs/2406.05649

---

## 9. Methodology Notes

### Search Strategy
- Web searches conducted across arXiv, CVPR/ICCV/NeurIPS/ICLR/SIGGRAPH 2024-2025 proceedings, and GitHub
- Focused on methods with open-source implementations or clear algorithmic descriptions
- Prioritized papers with quantitative evaluations on standard benchmarks

### Evaluation Criteria
- **Feasibility**: How much effort to integrate with TRELLIS.2 (Easy < 1 day, Medium 3-7 days, Hard > 1 week)
- **Impact**: Expected quality score improvement based on paper's reported gains adjusted for TRELLIS.2 context
- **Risk**: Probability that the method fails or causes regressions
- **Dependencies**: Additional models, data, or infrastructure required

### Limitations
- Expected impact estimates are extrapolated from paper results on different benchmarks
- Feasibility estimates assume familiarity with TRELLIS.2 codebase
- Some methods require GPU memory that may exceed DGX Spark's capacity when combined
- Many papers report results on text-to-3D or multi-view reconstruction, not single-image-to-3D
