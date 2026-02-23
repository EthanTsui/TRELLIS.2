# Survey v7: Breaking Through the V4.1 ~93/100 Quality Ceiling

**Date**: 2026-02-23
**Author**: Research Optimizer Agent
**Scope**: Techniques for >1 point improvement on A1 (silhouette), A2 (color distribution), or C3 (detail richness) given frozen TRELLIS.2 weights
**Status**: Literature survey + analysis, no experiments run

---

## Executive Summary

The V4.1 quality ceiling of ~93/100 is bounded by three dimensions:
- **A1 Silhouette (Crown at 74.7)**: Limited by 32^3 sparse structure discretization
- **A2 Color Distribution (Crown at 70-75)**: Limited by single-view color ambiguity
- **C3 Detail Richness (75-88)**: Limited by voxel-to-mesh resolution and texture bake

After surveying 25+ papers from 2024-2026 and analyzing existing implementations, I identify **8 actionable techniques** organized into three tiers based on expected impact and implementation effort.

### Tier 1: Already Implemented, Untested (Test First)

| # | Method | Target | Expected | Effort |
|---|--------|--------|----------|--------|
| 1 | 64^3 native cascade (ss_native_64) | A1 +3-5 | High | 0 (test) |
| 2 | Staged Best-of-N (N=4 shape) | A1 +3-5 | High | 0 (test) |
| 3 | Silhouette Corrector (w/ Dice loss upgrade) | A1 +2-6 | High | 2-4h tune |
| 4 | FDG (Frequency-Decoupled Guidance) | C3 +2-4 | Medium-High | 0 (test) |

### Tier 2: New Techniques from Literature (Build)

| # | Method | Target | Expected | Effort |
|---|--------|--------|----------|--------|
| 5 | Flow-Model Inference-Time Scaling (SMC/RBF) | A1, A2, C3 all +2-5 | High | 1-2 days |
| 6 | Elevate3D / HFS-SDEdit Texture Refinement | A2 +3-8, C3 +5-10 | Very High | 2-3 days |
| 7 | Input-View Color Transfer Post-Processing | A2 +3-6 | Medium-High | 4-8h |
| 8 | Multi-Resolution Silhouette Deformation | A1 +3-6 | High | 4-8h |

### Tier 3: Architecture-Aware Insights (Strategic)

| # | Insight | Impact |
|---|---------|--------|
| A | A1 theoretical ceiling at 32^3 is ~93% for compact, ~80% for complex shapes |
| B | A2 is fundamentally limited by single-view color ambiguity on back/hidden surfaces |
| C | C3 is bottlenecked by UV bake resolution, not voxel resolution |

---

## 1. Tier 1 Analysis: Already Implemented, Needs Testing

### 1.1 64^3 Native Cascade (ss_native_64) -- A1 Target

**What it does**: Skips the `max_pool3d(decoded, 2, 2, 0) > 0.5` operation that downsamples the Stage 1 decoder's native 64^3 occupancy output to 32^3. Feeds the full 64^3 coordinates into the cascade pipeline.

**Why it matters for A1**: The A1 ceiling analysis (a1_ceiling_analysis.md) shows:
- At 32^3: each voxel projects to ~2.25 pixels on 256x256 eval grid
- At 64^3: each voxel projects to ~1.125 pixels
- Theoretical Dice ceiling: 32^3 = 93-96% compact / 80-90% complex; 64^3 = 95-98% / 90-95%
- The Crown object (A1=74.7) likely has thin features/filigree that are completely lost at 32^3

**Implementation status**: Already coded in `trellis2_image_to_3d.py` lines 1114-1116. Set `ss_native_64=True` in `sparse_structure_sampler_params`.

**Risk**: 4-8x more SLAT tokens. With max_num_tokens=65536, may overflow for dense objects. Need to increase to 130K (feasible on 128GB GPU) or accept slower generation (~2x).

**Expected impact**: +3-5 on A1, especially for complex/filigree objects. This is the single highest-ROI test available.

**Paper grounding**: XCube (NVIDIA, NeurIPS 2024) demonstrates that hierarchical coarse-to-fine sparse voxel generation preserves fine detail that is lost at lower resolutions.

### 1.2 Staged Best-of-N (N=4 Shape Selection) -- A1 Target

**What it does**: Generates N shape candidates with different seeds, evaluates silhouette Dice for each (fast: no texture needed), then textures only the best shape.

**Why it matters for A1**: The Crown object's A1=74.7 may represent a bad random draw. With N=4 candidates, the probability of getting a better shape is high. Research on Best-of-N (arXiv:2501.09732, Ma et al. CVPR 2025) shows quality scales monotonically with N, with sqrt(N) diminishing returns.

**Implementation status**: Best-of-N is already wired in `app.py` line 534. However, it runs the FULL pipeline for each candidate. The STAGED variant (shape-only evaluation before texturing) is described in silhouette_accuracy_research_2026_02.md but needs testing.

**Cost analysis**:
- Baseline: 1 shape + 1 texture = 1.0x
- N=4 shape + 1 texture: ~1.8x (shape is ~20% of total time)
- N=4 shape + 2 texture (top-2): ~2.6x

**Expected impact**: +3-5 on A1 (variance reduction). For the Crown outlier at 74.7, could be +5-10 if the low score is due to bad shape draw.

**Paper grounding**: "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps" (arXiv:2501.09732) establishes that Best-of-N with a quality verifier provides consistent improvement across model sizes and tasks.

### 1.3 Silhouette Corrector with Dice Loss Upgrade -- A1 Target

**What it does**: Post-generation mesh vertex deformation via differentiable rendering (nvdiffrast) to match the input image's silhouette.

**Current state**: Implemented at `trellis2/postprocessing/silhouette_corrector.py`, already wired in `app.py` line 691-697. Uses BCE + distance transform loss.

**Why it underperforms**: The silhouette_accuracy_research_2026_02.md analysis identified 6 weaknesses:
1. Single-view only (yaw=0, pitch=0.25)
2. High Laplacian weight (w=50, too conservative; optimal is 5-15)
3. Max displacement too conservative (0.02; needs 0.05-0.08)
4. Uses BCE loss instead of Soft Dice (the actual metric)
5. No multi-resolution pyramid (misses both coarse errors and fine boundary)
6. No SoftRas-style smooth gradients

**Recommended upgrade**:
```python
# Replace BCE with Soft Dice (directly optimizes evaluation metric)
def soft_dice_loss(rendered, target):
    p, t = rendered.flatten(), target.flatten()
    return 1.0 - (2.0 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)

# Multi-resolution pyramid: coarse-to-fine
stages = [
    (256, 30, 2e-3, 0.06),  # coarse: large movements
    (512, 50, 5e-4, 0.03),  # fine: precision
]
```

**Expected impact**: +2-6 on A1. Ceiling ~88-90 (limited by topology -- can move vertices but not add/remove them).

**Paper grounding**: Palfinger (2022) "Continuous remeshing for inverse rendering" shows cotangent Laplacian with w=5-15 outperforms uniform Laplacian at w=50; Liu et al. (ICCV 2019) SoftRas demonstrates that soft rasterization enables long-range vertex movement.

### 1.4 Frequency-Decoupled Guidance (FDG) -- C3 Target

**What it does**: Decomposes the CFG direction into low-frequency (global structure) and high-frequency (detail/fidelity) components, applies separate guidance weights to each.

**Implementation status**: Already coded in `cfg_utils.py` line 179-202. Uses Gaussian blur for frequency decomposition on sparse voxel features. Parameters: `fdg_sigma`, `fdg_lambda_low`, `fdg_lambda_high`.

**Why it matters for C3**: The paper "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales" (arXiv:2506.19713) demonstrates that high-frequency guidance mainly enhances visual fidelity while low-frequency guidance governs structure. By increasing `fdg_lambda_high` relative to `fdg_lambda_low`, we can boost texture detail without over-constraining global shape.

**Recommended test configuration**:
```python
tex_slat_sampler_params = {
    "cfg_mode": "fdg",
    "fdg_sigma": 2.0,       # Gaussian blur sigma for decomposition
    "fdg_lambda_low": 0.8,  # Slightly reduce structure guidance
    "fdg_lambda_high": 1.5, # Boost detail guidance
    "guidance_strength": 12.0,
    "guidance_rescale": 1.0,
}
```

**Risk**: FDG on 3D sparse voxels is novel -- the paper validates on 2D latents. The `_sparse_gaussian_blur_3d` implementation needs to handle the irregular grid correctly. May have similar issues to APG/CFG-Zero* which failed on flow matching.

**Expected impact**: +2-4 on C3 (detail richness), +0-1 on A2 (color fidelity at high freq).

**Paper grounding**: Sadat et al. (arXiv:2506.19713) "Guidance in the Frequency Domain"; DeCo (arXiv:2511.19365) validates frequency-aware flow matching loss; ZeResFDG (ResearchGate 2025) combines FDG with zero-projection and energy rescaling.

---

## 2. Tier 2 Analysis: New Techniques from Literature

### 2.1 Flow-Model Inference-Time Scaling via Stochastic Sampling (SMC/RBF)

**Paper**: "Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing" (Kim et al., NeurIPS 2025, arXiv:2503.19385)

**Core idea**: Convert TRELLIS.2's deterministic ODE sampling to an SDE with injected stochasticity, enabling particle-based quality search. The method has three components:

1. **ODE-to-SDE conversion**: Add a diffusion term g_t to the flow ODE:
   ```
   dx_t = u_t(x_t) dt  -->  dx_t = [u_t(x_t) - (g_t^2/2) * score(x_t)] dt + g_t dw
   ```
   where score is computed from the velocity field via:
   ```
   score(x_t) = (1/sigma_t) * [alpha_t * u_t(x_t) - alpha_dot_t * x_t] / [alpha_dot_t * sigma_t - alpha_t * sigma_dot_t]
   ```

2. **VP interpolant conversion**: Replace linear interpolant (alpha_t=1-t, sigma_t=t) with variance-preserving interpolant for increased sample diversity at each step.

3. **Rollover Budget Forcing (RBF)**: Allocate N function evaluations per timestep. If a particle achieves higher reward than the current best, immediately proceed (roll over unused budget). This adaptively concentrates compute where most needed.

**Applicability to TRELLIS.2**: TRELLIS.2 uses rectified flow with Euler ODE -- this is exactly the setting the paper targets. The SDE conversion requires only the velocity field (which we have) and a score estimate (derivable from velocity). Each of the 3 stages (SS, shape SLAT, texture SLAT) could independently benefit.

**Reward functions for TRELLIS.2**:
- Stage 1 (SS): Silhouette Dice as reward (fast: render occupancy grid, compare to input alpha)
- Stage 2 (Shape): Silhouette Dice on quick mesh extraction
- Stage 3 (Texture): CLIP-S or color histogram similarity as reward

**Compute cost**: 500 NFEs total across 10-16 steps = ~30-50x per stage. However, only applying to Stage 1 (which is fast at 16^3) would be ~30x on the cheapest stage. With SS taking ~5 seconds, 30x = ~2.5 minutes -- acceptable for quality-critical applications.

**Expected impact**: +2-5 on A1 (silhouette reward), +1-3 on A2 (color reward), +1-2 on C3 (aesthetic reward)

**Implementation effort**: 1-2 days. Need to implement:
- Score computation from velocity field (Eq. 8 in paper)
- SDE integration step (Euler-Maruyama)
- Particle resampling with RBF logic
- Reward function integration

**Code reference**: https://github.com/KAIST-Visual-AI-Group/Flow-Inference-Time-Scaling

### 2.2 Elevate3D / HFS-SDEdit Texture Refinement

**Paper**: "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model" (Ryunuri et al., SIGGRAPH 2025, arXiv:2507.11465)

**Core idea**: View-by-view alternating texture+geometry refinement using a novel diffusion-based enhancement:

1. **HFS-SDEdit**: High-Frequency-Swapping SDEdit. During diffusion denoising, the model freely generates low-frequency features (color, global appearance) while injecting the reference image's high-frequency components (edges, details) during early steps:
   ```
   z'_t = (I - G_sigma) * z~_t + G_sigma * z^_t
   ```
   where G_sigma is a Gaussian filter. This preserves structural detail while allowing quality enhancement.

2. **View-by-view refinement**: Process each camera viewpoint iteratively -- render, identify unrefined regions, apply HFS-SDEdit with masking, update texture, then refine geometry via normal integration.

**Quantitative results**:
- MUSIQ: 66.5 (vs DreamGaussian 61.7, MagicBoost 51.6)
- TOPIQ: 0.530 (vs 0.469, 0.392)
- Q-Align: 3.22 (vs 2.74, 2.50)
- ~25 minutes per model (unoptimized)

**Applicability**: Directly applicable as post-processing on TRELLIS.2's GLB output. Requires a 2D diffusion model (FLUX, recommended) and a monocular normal predictor. The infrastructure is similar to the existing TextureRefiner but uses SDEdit instead of direct optimization.

**Why this could break through A2**: The Crown's A2=70-75 is limited by color ambiguity on occluded surfaces. HFS-SDEdit can generate plausible colors for unseen regions while preserving the input view's appearance.

**Expected impact**: A2 +3-8 (color coherence across views), C3 +5-10 (texture detail from diffusion model)

**Implementation effort**: 2-3 days. Requires FLUX model download (~12GB), implementing the HFS-SDEdit loop, and camera pose alignment.

**Code reference**: https://github.com/ryunuri/Elevate3D

### 2.3 Input-View Color Transfer Post-Processing -- A2 Target

**Core idea**: After generating the 3D model, render it from the input viewpoint and perform color histogram matching/transfer between the rendered view and the input image. Apply the same color transformation to the texture map.

**Method**: This is a classical technique (Reinhard et al. 2001) with modern variants:
1. Render 3D model from input camera pose
2. Compute per-channel (Lab or RGB) statistics: mean and std of rendered vs. input
3. Transform texture pixels: `tex_new = (tex - mean_render) * (std_input / std_render) + mean_input`
4. Optionally use optimal transport (Pitie et al. 2007) for better distribution matching

**Why this could help A2**: The A2 metric computes histogram correlation between the input image's color distribution and the rendered model's color distribution. A color transfer that explicitly matches these distributions would directly optimize the metric.

**Nuance**: Must only apply to regions visible from the input view (to avoid color-shifting backside textures incorrectly). Use the rasterized UV mapping from nvdiffrast to identify which texture pixels are visible.

**Expected impact**: A2 +3-6 (directly optimizes the metric)

**Implementation effort**: 4-8h. Use existing nvdiffrast for visibility, OpenCV for histogram analysis, numpy for color transfer.

**Paper grounding**: "A Sequential Color Correction Approach for Texture Mapping of 3D Meshes" (MDPI Sensors 2023); "Color Transfer with Modulated Flows" (arXiv 2025) for optimal transport approach.

### 2.4 Multi-Resolution Silhouette Deformation -- A1 Target

**Core idea**: Upgrade the existing SilhouetteCorrector with a coarse-to-fine pyramid and Soft Dice loss, as detailed in silhouette_accuracy_research_2026_02.md.

**Specific changes**:
1. Replace BCE with Soft Dice loss (directly optimizes the eval metric)
2. Add coarse-to-fine pyramid: 256px/30 steps/large displacement -> 512px/50 steps/fine
3. Reduce Laplacian weight from 50 to 10-15 (allow more deformation)
4. Increase max displacement from 0.02 to 0.05
5. Add edge-length regularization (prevent triangle collapse)
6. Optional: multi-view regularization (4 views, penalize extreme backside deformation)

**Expected impact**: +3-6 on A1 beyond current corrector. Ceiling ~88-90.

**Implementation effort**: 4-8h (modifying existing code).

**Paper grounding**: DMesh++ (ICCV 2025) shows differentiable mesh optimization with proper regularization converges to high-quality geometry; Palfinger (2022) demonstrates cotangent Laplacian at w=5-15 optimal for silhouette fitting.

---

## 3. Tier 3: Strategic Insights

### 3.1 A1 Theoretical Ceiling Analysis

From the detailed analysis in `a1_ceiling_analysis.md`:

**At 32^3 resolution (current)**:
- Each voxel projects to ~2.25 pixels on the 256x256 eval grid
- Compact objects: Dice ceiling ~93-96%
- Complex objects (crown filigree, antlers): Dice ceiling ~80-90%
- Thin features (<1 voxel wide) are completely lost

**At 64^3 resolution (ss_native_64)**:
- Each voxel projects to ~1.125 pixels
- Compact objects: Dice ceiling ~95-98%
- Complex objects: Dice ceiling ~90-95%
- Fine features are 2x better resolved

**Implication**: To get Crown above 90 on A1, we MUST move to 64^3 or use post-generation deformation. Parameter tuning alone cannot overcome the discretization limit.

### 3.2 A2 Color Distribution Fundamental Limits

**Problem structure**: A2 compares the color histogram of an input image (2D, one viewpoint) against renders of a 3D model (potentially from different viewpoints). The model must hallucinate colors for surfaces not visible in the input.

**Fundamental ambiguity**: For a crown with gold front and unknown back, the model could produce gold back, dark back, or any other color. The A2 metric penalizes all of these equally if the overall histogram doesn't match.

**Actionable insight**: The best strategy is to make the visible-side colors as close to the input as possible (color transfer, Sec 2.3) and ensure the hallucinated backside doesn't introduce colors not present in the input (e.g., suppress grey patches).

### 3.3 C3 Detail Richness Bottleneck

**C3 measures**: Laplacian energy on texture (high-frequency detail) + dihedral angle std on mesh (geometric detail).

**Current bottleneck**: The texture bake at 2048x2048 with xatlas UV unwrapping introduces fragmentation (5000+ body count). Fine geometric detail from the voxel representation is lost in the mesh simplification (800K target faces) and UV rasterization.

**Insight**: Texture resolution increase to 4096 had no effect (already tested). The bottleneck is not pixel count but UV island fragmentation and the smoothing effect of baking from discrete voxels.

**Best path**: FDG guidance (high-freq weight boost) to generate more detail at the voxel level, combined with Elevate3D's HFS-SDEdit to enhance the baked texture.

---

## 4. Recommended Test Priority

### Priority 1: Test 64^3 Cascade (0 effort, test only)

```python
# In app.py or test script:
sparse_structure_sampler_params = {
    "ss_native_64": True,
    "steps": 12,
    "guidance_strength": 10.0,
    "guidance_rescale": 0.8,
}
# Ensure max_num_tokens >= 130000
```

**Rationale**: Already implemented. Expected to provide the largest single improvement to A1 for complex objects. Zero implementation effort.

### Priority 2: Test FDG Guidance (0 effort, test only)

```python
tex_slat_sampler_params = {
    "cfg_mode": "fdg",
    "fdg_sigma": 2.0,
    "fdg_lambda_low": 0.8,
    "fdg_lambda_high": 1.5,
    "steps": 16,
    "guidance_strength": 12.0,
    "guidance_rescale": 1.0,
    "rescale_t": 4.0,
}
```

**Rationale**: Already implemented. Could improve C3 detail richness. Zero implementation effort.

### Priority 3: Test Staged Best-of-N (0 effort if full pipeline BoN used)

```python
# Use existing best_of_n=4 with quality_verifier
# The verifier already computes silhouette Dice
```

**Rationale**: Already wired. Expected +3-5 on A1 from variance reduction.

### Priority 4: Upgrade Silhouette Corrector (4-8h)

Replace BCE with Soft Dice, add coarse-to-fine pyramid, relax constraints. See Section 1.3 for code.

### Priority 5: Input-View Color Transfer (4-8h)

Post-processing color histogram matching using nvdiffrast visibility and Lab-space statistics transfer. See Section 2.3.

### Priority 6: Flow Inference-Time Scaling with RBF (1-2 days)

Implement SDE conversion for Stage 1 with silhouette Dice reward. See Section 2.1 for equations.

### Priority 7: Elevate3D / HFS-SDEdit (2-3 days)

Full texture+geometry refinement pipeline. Highest potential impact on A2 and C3 but requires FLUX model. See Section 2.2.

---

## 5. Combined Ceiling Estimation

If all Priority 1-5 techniques work as expected:

| Dimension | Current | +64^3 | +BoN-4 | +SilCorr | +FDG | +ColorXfer | Combined |
|-----------|---------|-------|--------|----------|------|------------|----------|
| A1 (Crown) | 74.7 | 80-84 | 83-87 | 86-90 | 86-90 | 86-90 | **86-90** |
| A2 (Crown) | 70-75 | 70-75 | 71-76 | 71-76 | 72-77 | 76-82 | **76-82** |
| C3 | 75-88 | 76-89 | 76-89 | 76-89 | 79-92 | 79-92 | **79-92** |

**Overall V4.1 score projection**: ~93 -> ~95-97 (optimistic) or ~94-95 (conservative).

Breaking through 97/100 would require Tier 2 methods (Elevate3D, RBF) or architectural changes (native 3D texture, higher-resolution SLAT model).

---

## 6. New Papers Surveyed

| Paper | Venue | Key Technique | Relevance |
|-------|-------|---------------|-----------|
| Flow Inference-Time Scaling (Kim et al.) | NeurIPS 2025 | SDE conversion + RBF for flow models | 9/10 |
| Elevate3D (Ryunuri et al.) | SIGGRAPH 2025 | HFS-SDEdit texture+geometry refinement | 8/10 |
| Noise Trajectory Search (arXiv:2506.03164) | arXiv 2025 | Epsilon-greedy MDP noise search | 7/10 |
| FDG Guidance (Sadat et al.) | arXiv 2025 | Frequency-decoupled CFG | 8/10 |
| FKC Correctors (Skreta et al.) | ICML 2025 | Feynman-Kac SMC for diffusion | 6/10 |
| Performance Plateaus (arXiv:2506.12633) | arXiv 2025 | Best-of-N scaling limits | 7/10 |
| DMesh++ (Son) | ICCV 2025 | Differentiable mesh optimization | 6/10 |
| Inference-Time Scaling (Ma et al.) | CVPR 2025 | Search vs. denoising steps tradeoff | 7/10 |
| General FM Guidance (OpenReview) | 2025 | First framework for flow matching guidance | 7/10 |
| DeCo (Ma et al.) | arXiv 2025 | Frequency-decoupled pixel diffusion | 6/10 |
| ZeResFDG (ResearchGate) | arXiv 2025 | FDG + zero-projection + energy rescaling | 7/10 |
| Color Transfer Modulated Flows | arXiv 2025 | Optimal transport color matching | 5/10 |
| SpArSe-Up | arXiv 2025 | Surface-anchored sparse upsampling | 7/10 |
| GTR (Zhuang et al.) | ICLR 2025 | 4-sec per-instance texture refinement | 8/10 |

---

## 7. References

1. Kim, J., Yoon, T., Hwang, J., Sung, M. "Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing." NeurIPS 2025. arXiv:2503.19385.
2. Ryunuri et al. "Elevating 3D Models: High-Quality Texture and Geometry Refinement from a Low-Quality Model." SIGGRAPH 2025. arXiv:2507.11465.
3. Sadat et al. "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales." arXiv:2506.19713.
4. Ma et al. "Scaling Inference Time Compute for Diffusion Models." CVPR 2025.
5. Tang et al. "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps." arXiv:2501.09732.
6. Skreta et al. "Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts." ICML 2025.
7. Wu et al. "A General Framework for Inference-time Scaling and Steering of Diffusion Models." arXiv:2501.06848.
8. Zhuang, P. et al. "GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement." ICLR 2025.
9. Ren et al. "XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies." NeurIPS 2024.
10. Palfinger. "Continuous remeshing for inverse rendering." 2022.
11. Son, S. "DMesh++: An Efficient Differentiable Mesh for Complex Shapes." ICCV 2025.
12. Ma, Z. "DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation." arXiv:2511.19365.
13. SpArSe-Up. "Learnable Sparse Upsampling for 3D Generation with High-Fidelity Textures." arXiv:2509.23646.
14. Performance Plateaus. "Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models." arXiv:2506.12633.
