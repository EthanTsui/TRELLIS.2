# Survey: 2D Image to 3D Model Quality Improvement (2024-2026)

**Date**: 2026-02-16
**Author**: Deep Learning Researcher Agent
**Scope**: Practical, implementable techniques for improving TRELLIS.2 3D generation quality
**Focus**: Post-processing, guidance, texture, shape, and multi-view consistency

---

## Executive Summary

This survey covers 25+ papers from 2024-2026 across five quality improvement dimensions for
image-to-3D generation. The top 10 findings are ranked by expected impact on TRELLIS.2
(sparse voxel + flow matching + FlexiCubes/FDG mesh) and implementation feasibility.

**Key takeaway**: The highest-impact improvements come from (1) improved CFG strategies for
flow matching that are trivial to implement, (2) post-generation texture super-resolution via
differentiable rendering, and (3) normal-guided geometry refinement. Architectural changes
requiring retraining (cross-view attention, 3D-aware RoPE) offer larger gains but are
significantly harder to deploy.

---

## Top 10 Findings

### 1. CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching

- **Paper**: "CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models"
- **Authors**: Weichen Fan, Amber Yijia Zheng, Raymond A. Yeh, Ziwei Liu
- **Venue/Year**: arXiv 2503.18886, March 2025
- **Code**: https://github.com/WeichenFan/CFG-Zero-star

**Key Technique**:
Two modifications to standard CFG in flow matching models:

1. **Optimized Scale (s*)**: Replace the fixed guidance scale with an adaptive scalar that
   projects the conditional velocity onto the unconditional velocity:
   ```
   s* = (v_cond^T * v_uncond) / ||v_uncond||^2
   ```
   Then the guided velocity becomes:
   ```
   v_guided = (1 - w) * s* * v_uncond + w * v_cond
   ```

2. **Zero-Init**: Zero out the first K ODE solver steps (K=1-2 by default). Early in
   the flow trajectory, velocity estimates are inaccurate and CFG pushes samples onto
   wrong trajectories. Skipping these steps (x_{t+1} = x_t) avoids this.

**Quantitative**: FID 2.23 -> 2.10 on ImageNet-256 (SiT-XL); Aesthetic Score 6.96 -> 7.10
on SD3.5.

**Application to TRELLIS.2**:
TRELLIS.2 uses `FlowEulerGuidanceIntervalSampler` with velocity prediction and Euler
stepping. The CFG is applied in `ClassifierFreeGuidanceSamplerMixin._inference_model()`.
Both modifications can be implemented directly:

- **s***: Compute the projection scalar from `pred_pos` and `pred_neg` before combining.
  Requires ~3 lines of code change in `classifier_free_guidance_mixin.py`.
- **Zero-Init**: Skip the first 1-2 steps in the Euler loop in `flow_euler.py` by
  setting `pred_x_prev = x_t` when `step_index < K`.

**Relevant files**:
- `/workspace/TRELLIS.2/trellis2/pipelines/samplers/classifier_free_guidance_mixin.py`
- `/workspace/TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py`

**Implementation Difficulty**: **Easy** (5-10 lines of code, no retraining, no new deps)

---

### 2. Adaptive Projected Guidance (APG)

- **Paper**: "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"
- **Authors**: Suttisak Wizadwongsa, Worameth Chinchuthakun (Disney Research)
- **Venue/Year**: ICLR 2025 (arXiv 2410.02416, October 2024)
- **Code**: https://github.com/huggingface/diffusers/pull/9626 (diffusers integration)

**Key Technique**:
Decompose the CFG update delta = v_guided - v_cond into parallel and orthogonal components
relative to v_cond:

```
delta_parallel = (delta . v_cond / ||v_cond||^2) * v_cond
delta_orthogonal = delta - delta_parallel
```

The parallel component causes oversaturation; the orthogonal component enhances quality.
APG down-weights the parallel component:

```
v_apg = v_cond + alpha * delta_parallel + delta_orthogonal
```

where alpha < 1.0 (typically 0.0-0.5). Additionally introduces EMA momentum on the
orthogonal component for smoother trajectories.

**Application to TRELLIS.2**:
TRELLIS.2's current CFG uses `pred = guidance_strength * pred_pos + (1-guidance_strength) * pred_neg`,
which is standard linear extrapolation. APG can replace this with the decomposed version.
This is especially valuable for texture generation where high guidance (5.0-7.0) causes
saturation artifacts and dark spots.

The `guidance_rescale` mechanism already in the code addresses a related problem (std
normalization), but APG is more principled -- it operates on the direction, not just
the magnitude.

**Relevant files**:
- `/workspace/TRELLIS.2/trellis2/pipelines/samplers/classifier_free_guidance_mixin.py`

**Implementation Difficulty**: **Easy** (10-15 lines, drop-in replacement for CFG computation)

---

### 3. PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors

- **Paper**: "PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors"
- **Authors**: Yujin Chen, Yinyu Nie, Benjamin Ummenhofer, Reiner Birkl, et al.
- **Venue/Year**: NeurIPS 2025 (arXiv 2506.02846, June 2025)
- **Project**: https://terencecyj.github.io/projects/PBR-SR/

**Key Technique**:
Zero-shot 4x PBR texture super-resolution using:

1. Render the mesh from multiple strategic viewpoints using differentiable rendering
2. Apply DiffBIR (pretrained 2D SR model) to generate high-res pseudo-GT renders
3. Back-project enhanced renders into UV texture space via differentiable rasterization
4. Optimize with multi-view consistency: `L_total = L_robust + lambda_pbr * L_pbr + lambda_tv * L_tv`
5. Identity constraints in PBR domain preserve material fidelity

Operates on albedo, roughness, metallic, and normal maps simultaneously.

**Runtime**: ~30 min for 2K-to-8K, ~8 min for 1K-to-2K on NVIDIA A6000.
**No training required** -- uses frozen pretrained models.

**Application to TRELLIS.2**:
TRELLIS.2 already outputs meshes with PBR textures (base_color, metallic, roughness)
via o-voxel's FlexiCubes extraction. The output texture resolution is typically 1K-2K.
PBR-SR can be applied as a post-processing step on the final GLB:

1. Load the generated mesh + PBR textures
2. Render with nvdiffrast (already a TRELLIS.2 dependency) from 8-16 viewpoints
3. Run DiffBIR on each render
4. Optimize UV textures via gradient descent

This directly addresses the "low texture detail" and "blurry textures" issues observed
in current TRELLIS.2 outputs.

**Relevant files**:
- Post-process: `/workspace/TRELLIS.2/o-voxel/o_voxel/postprocess.py` (to_glb function)
- Renderer: nvdiffrast (already installed)

**Implementation Difficulty**: **Medium** (requires DiffBIR model, render loop, UV optimization)

---

### 4. Elevate3D: High-Quality Texture and Geometry Refinement

- **Paper**: "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model"
- **Authors**: Nuri Ryu et al.
- **Venue/Year**: SIGGRAPH 2025 (arXiv 2507.11465, July 2025)
- **Code**: https://github.com/ryunuri/Elevate3D (released 2025-07-23)

**Key Technique**:
HFS-SDEdit (High-Frequency-Swapping SDEdit): A texture refinement method that decouples
low-frequency and high-frequency components in the diffusion latent space:

- **Low-frequency**: Let the diffusion model generate freely (captures domain-appropriate
  appearance)
- **High-frequency**: Constrain to match the input (preserves structural identity)

This solves the classic SDEdit quality-fidelity tradeoff. Combined with:

1. Monocular normal prediction from refined texture views
2. Regularized normal integration (energy minimization balancing normal consistency and
   depth regularization with lambda=0.008)
3. Poisson surface reconstruction to merge refined regions

Operates view-by-view, alternating texture and geometry passes.

**Runtime**: ~25 min on A6000 (unoptimized).
**Requires**: FLUX (texture gen), SAM (segmentation), Marigold (depth), 48GB+ VRAM.

**Application to TRELLIS.2**:
Can be applied as a final post-processing stage on any TRELLIS.2 output mesh.
Particularly valuable for:
- Enhancing texture quality on dark or low-contrast regions
- Adding geometric surface detail (wrinkles, surface patterns) via normal integration
- Correcting texture artifacts in occluded regions

The DGX Spark has sufficient VRAM (96GB unified) for the full pipeline.

**Implementation Difficulty**: **Medium** (conda env setup, model downloads, integration script)

---

### 5. 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement

- **Paper**: "3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement"
- **Authors**: Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, Chen Change Loy
- **Venue/Year**: CVPR 2025 (arXiv 2412.18565, December 2024)
- **Code**: https://github.com/Luo-Yihang/3DEnhancer
- **Weights**: https://huggingface.co/Luo-Yihang/3DEnhancer

**Key Technique**:
A dedicated multi-view enhancement model (DiT-based, PixArt-Sigma backbone) that takes
coarse multi-view renders and produces enhanced versions with:

1. **Pose-Aware Encoder**: Concatenates Plucker ray coordinates with RGB input, injected
   via ControlNet-style copies into the DiT backbone
2. **Epipolar Aggregation**: Cross-view attention constrained by epipolar geometry for
   multi-view consistency -- for each feature location, correspondence maps are computed
   with near-view images using the fundamental matrix
3. **Controllable Noise Augmentation**: Low noise = restoration, high noise = regeneration

PSNR 27.53 / SSIM 0.9265 on Objaverse; MUSIQ 73.32 on in-the-wild.

**Application to TRELLIS.2**:
Can be used as a multi-view enhancement step before or after mesh extraction:

- **Before**: Render the generated 3D from multiple views, enhance, then re-project
  enhanced textures onto the mesh
- **After**: Enhance the multi-view renders used for texture baking

The pose-aware conditioning makes it compatible with TRELLIS.2's camera setup.
NTU S-Lab License 1.0 may limit commercial use.

**Implementation Difficulty**: **Medium** (model download, inference pipeline, texture reprojection)

---

### 6. Generative Detail Enhancement for Physically Based Materials

- **Paper**: "Generative Detail Enhancement for Physically Based Materials"
- **Authors**: Saeed Hadadan, Benedikt Bitterli, et al. (NVIDIA)
- **Venue/Year**: SIGGRAPH 2025 (arXiv 2502.13994, February 2025)
- **Code**: https://github.com/saeedhd96/generative-detail

**Key Technique**:
Adds physically-plausible detail (wear, aging, weathering) to basic PBR materials using:

1. **UV-consistent noise seeding**: Noise for the diffusion process is generated in
   UV space rather than image space, ensuring multi-view consistency
2. **Projective attention bias**: Attention mechanism is biased so that pixels attend
   strongly to their corresponding pixel locations in other views (geometric consistency)
3. **Differentiable back-projection**: Enhanced details are back-propagated from
   rendered images to PBR material parameters via inverse rendering

Text-prompted: "worn leather", "weathered metal", etc.

**Application to TRELLIS.2**:
Directly applicable to TRELLIS.2's PBR outputs. The UV-consistent noise seeding idea
is particularly relevant -- TRELLIS.2 generates textures that sometimes look too clean
or uniform. This method could add realistic material detail as a post-process.

The UV-space noise seeding technique could also be adapted for the generation stage itself
(using noise consistent across the sparse voxel structure rather than random per-voxel).

**Implementation Difficulty**: **Medium** (requires diffusion model, UV rendering pipeline)

---

### 7. Entropy Rectifying Guidance (ERG)

- **Paper**: "Entropy Rectifying Guidance for Diffusion and Flow Models"
- **Authors**: Abdelrahman Ifriqi, Romero-Soriano et al.
- **Venue/Year**: NeurIPS 2025 (arXiv 2504.13987, April 2025)

**Key Technique**:
Uses attention entropy as a guidance signal within the DiT architecture itself:

1. For each attention layer, compute a "strong" prediction (low temperature) and
   "weak" prediction (high temperature) using the Hopfield energy formulation
2. The guidance signal comes from the difference between strong and weak attention
   outputs, without requiring a separate unconditional forward pass
3. Works for both conditional and unconditional generation

Key advantages over CFG:
- No unconditional training needed
- Can be combined with CFG, APG, or CADS for additive improvements
- Works on flow matching models (tested on DiT)

**Application to TRELLIS.2**:
TRELLIS.2 uses DiT-style transformer blocks (`ModulatedTransformerCrossBlock` and
`ModulatedSparseTransformerCrossBlock`). ERG operates on the attention mechanism
within these blocks by modifying the softmax temperature.

Could provide texture quality improvements without requiring the separate unconditional
forward pass, effectively halving the compute for guided generation. However, the
sparse attention implementation adds complexity.

**Relevant files**:
- `/workspace/TRELLIS.2/trellis2/modules/sparse/transformer/`
- `/workspace/TRELLIS.2/trellis2/modules/transformer/`

**Implementation Difficulty**: **Medium** (attention layer modification, but architecture-specific)

---

### 8. CFG Manifold Projection for Flow Matching

- **Paper**: "Improving Classifier-Free Guidance of Flow Matching via Manifold Projection"
- **Authors**: Jian-Feng Cai, Haixia Liu, Zhengyi Su, Chao Wang
- **Venue/Year**: arXiv 2601.21892, January 2026

**Key Technique**:
Reformulates CFG sampling as homotopy optimization with a manifold constraint:

1. The velocity field in flow matching = gradient of smoothed distance functions
   guiding latents toward the target image set
2. Standard CFG uses heuristic linear extrapolation, which is sensitive to guidance
   scale because it can push samples off the data manifold
3. **Manifold projection**: After each CFG step, project back onto the data manifold
   via incremental gradient descent
4. **Anderson Acceleration**: Speeds up the projection without additional model
   evaluations

Training-free, tested on DiT-XL-2-256, Flux, and SD3.5 with significant improvements
in fidelity, prompt alignment, and robustness to guidance scale.

**Application to TRELLIS.2**:
Directly applicable to the FlowEuler sampler. After computing the CFG-guided velocity,
add a manifold projection step. The key challenge is that "the data manifold" for
sparse voxels is less well-defined than for images, but the gradient descent projection
can still be applied to the velocity field.

This is the most theoretically principled fix for the guidance sensitivity issues observed
in TRELLIS.2 texture generation (where tex_guidance=5.0 vs 7.0 makes a large difference).

**Relevant files**:
- `/workspace/TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py`

**Implementation Difficulty**: **Medium** (gradient descent inner loop, Anderson acceleration)

---

### 9. SuperCarver: Texture-Consistent Geometry Super-Resolution

- **Paper**: "SuperCarver: Texture-Consistent 3D Geometry Super-Resolution for High-Fidelity
  Surface Detail Generation"
- **Authors**: Qijian Zhang et al.
- **Venue/Year**: arXiv 2503.09439, March 2025

**Key Technique**:
Two-stage framework for adding surface detail to coarse meshes:

**Stage 1 -- Normal Diffusion**:
- Render the coarse mesh into multi-view normal maps
- Fine-tune a normal diffusion model on paired low-poly/high-poly normal renderings
- Generate detail-enhanced normal maps (deterministic, prior-guided)

**Stage 2 -- Distance Field Deformation**:
- Convert the mesh to a signed distance field (SDF)
- Optimize the SDF to match the predicted normals via differentiable rendering
- **Noise-resistant**: Uses a carefully designed distance field deformation that is
  robust to imperfect normal predictions (unlike direct vertex optimization)
- Extract the refined mesh from the optimized SDF

**Application to TRELLIS.2**:
TRELLIS.2 generates meshes via FlexiCubes/FDG extraction from the sparse voxel
structure. These meshes often lack fine geometric detail (surface patterns, edge
sharpness). SuperCarver can be applied as post-processing to add:

- Surface texture detail guided by the generated albedo texture
- Sharp edges and creases
- Fine-grained geometric features visible in the input image

The normal diffusion model would need to be trained or use the authors' pretrained
weights. The SDF optimization uses nvdiffrast (already available).

**Implementation Difficulty**: **Hard** (requires pretrained normal diffusion model, SDF pipeline)

---

### 10. GPU-Friendly Laplacian Texture Blending + UV Seam Smoothing

- **Paper**: "GPU-Friendly Laplacian Texture Blending"
- **Authors**: Published in JCGT (arXiv 2502.13945, February 2025)
- **Related**: Hunyuan3D 2.1 Spatial-aware Seam-Smoothing (arXiv 2506.15442, June 2025)

**Key Technique**:

**Laplacian Pyramid Blending**:
- Different Laplacian pyramid levels are blended with different mask sharpness
  proportional to feature size
- Low-frequency blending uses soft masks (smooth transitions)
- High-frequency blending uses sharp masks (preserve details)
- Implemented using standard texture mipmaps -- no additional data structures

**Hunyuan3D 2.1 UV Seam Smoothing**:
- 3D-Aware RoPE in multi-view attention blocks for cross-view coherence
- UV-space super-resolution followed by spatial-aware seam smoothing
- Seam detection + Gaussian blur along seam edges in UV space
- Illumination-invariant PBR training to separate lighting from materials

**Application to TRELLIS.2**:
TRELLIS.2's o-voxel postprocessing (`postprocess.py`) already includes some texture
enhancement (bilateral sharpening, dark texel recovery, etc.) but lacks dedicated UV
seam handling. Adding seam detection + Laplacian blending would address visible seam
artifacts, particularly on thin structures.

The Laplacian blending approach works entirely in UV space on the baked texture and
requires only numpy/OpenCV operations.

**Relevant files**:
- `/workspace/TRELLIS.2/o-voxel/o_voxel/postprocess.py`

**Implementation Difficulty**: **Easy** (pure image processing in UV space, no models needed)

---

## Additional Notable Findings

### Guidance Strategies

| Method | Key Idea | Venue | Difficulty |
|--------|----------|-------|------------|
| beta-CFG | Beta-distribution timestep schedule for guidance weight | arXiv 2502.10574 | Easy |
| DTC123 | Large timesteps = geometry, small timesteps = texture detail | CVPR 2024 | Easy |
| Dynamic Negative Guidance | State/time-dependent guidance scale from first principles | ICLR 2025 | Easy |
| Navigating with Annealing | Guidance scale annealing schedule | SIGGRAPH Asia 2025 | Easy |

**Common insight**: All agree that fixed-scale CFG is suboptimal. Early timesteps should
have lower (or zero) guidance to establish geometry; late timesteps should have higher
guidance to refine texture detail. TRELLIS.2 already has `guidance_interval` support but
uses a binary on/off rather than a smooth schedule.

**Recommended TRELLIS.2 integration**: Implement a beta-shaped guidance schedule:
```python
# In guidance_interval_mixin.py, replace binary on/off with smooth schedule
import scipy.stats
beta_a, beta_b = 2.0, 5.0  # peaks early, tapers late
t_norm = (t - guidance_interval[0]) / (guidance_interval[1] - guidance_interval[0])
scale = scipy.stats.beta.pdf(t_norm, beta_a, beta_b)
scale = scale / scale.max()  # normalize to [0, 1]
effective_guidance = 1.0 + (guidance_strength - 1.0) * scale
```

### Texture Quality

| Method | Key Idea | Venue | Difficulty |
|--------|----------|-------|------------|
| Material Anything | Unified PBR material estimation via confidence-masked diffusion | CVPR 2025 | Medium |
| MatLat | Fine-tuned VAE for material latent space, locality-preserving | arXiv 2512.17302 | Hard |
| PacTure | Packed multi-view PBR generation via visual autoregressive model | arXiv 2505.22394 | Hard |
| Meta 3D TextureGen | Visibility-weighted back-projection + UV inpainting | arXiv 2407.02430 | Medium |
| MVPaint | Synchronized multi-view diffusion + UV-space SR + seam smoothing | CVPR 2025 | Medium |

### Shape Accuracy

| Method | Key Idea | Venue | Difficulty |
|--------|----------|-------|------------|
| DetailGen3D | Data-dependent rectified flow for geometry refinement | arXiv 2411.16820 | Hard |
| CraftsMan3D | Normal-based geometry refiner with interactive magic brush | CVPR 2025 | Medium |
| GTR | Differentiable mesh extraction from NeRF + fine-tuning | ICLR 2025 | Medium |
| FlexiDreamer | FlexiCubes + orientation-aware texture mapping | arXiv 2404.00987 | Medium |

### Multi-View Consistency

| Method | Key Idea | Venue | Difficulty |
|--------|----------|-------|------------|
| CFD | 3D-consistent Gaussian noise via noise transport equation | ICLR 2025 | Medium |
| Im2SurfTex | Neural back-projection with geodesic positional encoding | CGF 2025 | Hard |
| Generative Detail | UV-consistent noise + projective attention bias | SIGGRAPH 2025 | Medium |

---

## Priority Implementation Roadmap for TRELLIS.2

### Phase 1: Quick Wins (1-2 days each, no retraining)

1. **CFG-Zero* zero-init** -- Skip first 1-2 steps of the ODE solver
   - Expected impact: +1-2% quality across the board
   - Risk: Minimal (can be disabled via parameter)

2. **APG guidance decomposition** -- Replace linear CFG with projected guidance
   - Expected impact: +2-5% on texture quality, especially at high guidance scales
   - Risk: Low (same compute cost, easy to A/B test)

3. **Beta-shaped guidance schedule** -- Replace binary guidance interval with smooth curve
   - Expected impact: +1-3% on shape+texture balance
   - Risk: Low (parametric, easy to tune)

4. **UV seam Laplacian blending** -- Post-process UV textures with seam-aware blending
   - Expected impact: +1-2% on seam-visible objects
   - Risk: None (pure post-processing)

### Phase 2: Medium-Effort Improvements (1-2 weeks each)

5. **PBR-SR texture super-resolution** -- 4x upscale PBR textures via differentiable rendering
   - Expected impact: +5-10% on texture detail
   - Requires: DiffBIR model, render-and-compare loop

6. **3DEnhancer multi-view enhancement** -- Enhance renders before texture baking
   - Expected impact: +3-5% on overall quality
   - Requires: PixArt-Sigma weights, Plucker encoding

7. **Render-and-compare texture optimization** -- Iterative texture refinement via nvdiffrast
   - Expected impact: +3-5% on texture fidelity to input image
   - Requires: Render pipeline, loss function design

### Phase 3: Deeper Integrations (weeks-months)

8. **Elevate3D full pipeline** -- HFS-SDEdit + normal-guided geometry refinement
   - Expected impact: +5-10% on both texture and geometry
   - Requires: FLUX, SAM, Marigold, 48GB+ VRAM

9. **SuperCarver geometry detail** -- Normal diffusion + SDF deformation
   - Expected impact: +3-5% on geometric detail
   - Requires: Normal diffusion model (training or pretrained weights)

10. **Generative material detail** -- UV-consistent noise + projective attention
    - Expected impact: +3-5% on material realism
    - Requires: Diffusion model modification, UV rendering pipeline

---

## Implementation Notes for TRELLIS.2

### Existing Infrastructure That Helps

1. **nvdiffrast**: Already installed and used for mesh rendering. Enables render-and-compare
   and PBR-SR integration.

2. **FlexiCubes/FDG**: Already used for mesh extraction. Geometry refinement methods can
   operate on the extracted mesh.

3. **Flow matching sampler**: Clean, modular code in `flow_euler.py` with mixin architecture.
   New guidance strategies can be added as new mixins without modifying base classes.

4. **PBR texture pipeline**: `postprocess.py` already has a sophisticated texture processing
   pipeline (bilateral sharpening, dark texel recovery, AO baking, etc.). New post-processing
   steps can be inserted into this pipeline.

5. **Multi-view infrastructure**: `inject_sampler_multi_image` already handles multi-view
   conditioning. Camera parameters and visibility weighting are implemented.

### Critical Constraints

1. **No retraining**: The TRELLIS.2-4B model weights are frozen. All improvements must be
   training-free or use separately trained auxiliary models.

2. **Memory budget**: DGX Spark has 96GB unified memory, but the TRELLIS.2 pipeline already
   uses significant memory during cascade generation. Post-processing steps must manage
   memory carefully (load/unload models).

3. **Latency budget**: Current generation takes ~2-3 minutes. Post-processing should add
   minutes, not hours. PBR-SR's 8-30 minute runtime may be acceptable for quality-critical
   applications but not for interactive use.

4. **PBR format**: TRELLIS.2 outputs base_color[0:3], metallic[3:4], roughness[4:5],
   alpha[5:6]. Any texture processing must handle all channels, not just RGB.

---

## References

### Guidance Strategies
- [CFG-Zero*](https://arxiv.org/abs/2503.18886) - Fan et al., 2025
- [APG](https://arxiv.org/abs/2410.02416) - Wizadwongsa & Chinchuthakun, Disney Research, ICLR 2025
- [Manifold Projection CFG](https://arxiv.org/abs/2601.21892) - Cai et al., January 2026
- [ERG](https://arxiv.org/abs/2504.13987) - Ifriqi et al., NeurIPS 2025
- [beta-CFG](https://arxiv.org/abs/2502.10574) - 2025
- [DTC123](https://arxiv.org/abs/2404.04562) - Yi et al., CVPR 2024
- [Dynamic Negative Guidance](https://openreview.net/forum?id=5647763d4245b) - ICLR 2025

### Texture Quality
- [PBR-SR](https://arxiv.org/abs/2506.02846) - Chen et al., NeurIPS 2025
- [Material Anything](https://arxiv.org/abs/2411.15138) - Huang et al., CVPR 2025
- [MatLat](https://arxiv.org/abs/2512.17302) - Yeo et al., December 2025
- [PacTure](https://arxiv.org/abs/2505.22394) - May 2025
- [Generative Detail Enhancement](https://arxiv.org/abs/2502.13994) - Hadadan et al., SIGGRAPH 2025
- [Hunyuan3D 2.1](https://arxiv.org/abs/2506.15442) - Tencent, June 2025
- [Meta 3D TextureGen](https://arxiv.org/abs/2407.02430) - Meta, 2024
- [MVPaint](https://arxiv.org/abs/2411.02336) - CVPR 2025
- [Im2SurfTex](https://arxiv.org/abs/2502.14006) - CGF 2025
- [TexVerse](https://arxiv.org/abs/2508.10868) - August 2025 (dataset)

### Shape & Geometry
- [Elevate3D](https://arxiv.org/abs/2507.11465) - Ryu et al., SIGGRAPH 2025
- [3DEnhancer](https://arxiv.org/abs/2412.18565) - Luo et al., CVPR 2025
- [SuperCarver](https://arxiv.org/abs/2503.09439) - Zhang et al., March 2025
- [DetailGen3D](https://arxiv.org/abs/2411.16820) - Deng et al., 2024
- [CraftsMan3D](https://arxiv.org/abs/2405.14979) - Li et al., CVPR 2025
- [GTR](https://arxiv.org/abs/2406.05649) - Zhuang et al., ICLR 2025

### Multi-View Consistency
- [CFD](https://arxiv.org/abs/2501.05445) - Yan et al., ICLR 2025
- [Laplacian Texture Blending](https://arxiv.org/abs/2502.13945) - JCGT 2025

### Differentiable Rendering
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) - NVIDIA, maintained through 2025
- [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/) - NVIDIA, SIGGRAPH 2023
