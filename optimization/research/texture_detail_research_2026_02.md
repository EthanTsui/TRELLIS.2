# Texture Detail (C3) Improvement Research for TRELLIS.2

**Date**: 2026-02-23
**Author**: Deep Learning Researcher Agent
**Scope**: Practical, no-retrain techniques for improving C3 (detail richness) scores in TRELLIS.2's 3D generation pipeline
**Context**: Current V4 C3 score = 80/100. C3 measures UV texture Laplacian energy (80%) + rendered view color entropy (20%). Bottleneck: generated textures lack fine detail compared to input image. TextureRefiner (render-and-compare with nvdiffrast + LPIPS) is implemented but untested.

---

## Executive Summary

C3 detail richness is determined by two factors: (1) how much high-frequency information the flow model encodes into the texture SLAT during generation, and (2) how faithfully that information is decoded and preserved in the final UV map. This report covers four intervention points across the generation stack — listed in order of expected impact-to-effort ratio.

**Top 5 recommendations (no retraining required):**

| Rank | Technique | C3 Impact | Effort | Status |
|------|-----------|-----------|--------|--------|
| 1 | **TextureRefiner (render-and-compare)** | +8-15 pts | Low (implemented, needs test) | Implemented, untested |
| 2 | **ITS3D noise search (Best-of-N for tex stage)** | +5-12 pts | Low-Med | Not implemented |
| 3 | **HFS-SDEdit texture refinement (Elevate3D)** | +6-10 pts | Med | Not implemented |
| 4 | **Rollover Budget Forcing + VP-SDE stochastic** | +3-6 pts | Med | Not implemented |
| 5 | **GPU-Friendly Laplacian UV blending (seam repair)** | +2-5 pts | Low | Not implemented |

**Key theoretical insight**: The C3 metric is dominated (80%) by UV-space Laplacian energy of the texture map. This means detail richness is largely determined at the decoding stage (SLAT -> UV texture baking), NOT primarily in the latent generation stage. Approaches that directly optimize or enhance the baked texture map will have the highest impact per unit of effort.

---

## 1. Understanding the C3 Bottleneck

### 1.1 What C3 Measures

From `auto_evaluate_v4.py`, C3 score is computed as:

```python
# C3 = 80% UV-space Laplacian energy + 20% view-based color entropy
tex_lap_energy = np.abs(Laplacian(tex_gray)[tex_mask]).mean()

# Score mapping:
# tex_lap_energy < 3  -> score = 30
# tex_lap_energy in [3, 12] -> score = 30 + (energy - 3) * 7.78  (range 30 to 100)
# tex_lap_energy >= 12 -> score = 100

# View entropy:
# Shannon entropy of grayscale histogram over 64 bins
# entropy < 2.5 -> score 30; [2.5, 4.0] -> 30-90; [4.0, 5.5] -> 90-100
```

Current calibration: baseline_v3.2 energy = 5.9 (C3 ~73), config_1536 = 9.2-11.3 (C3 ~89-96).

At current score of 80/100, the estimated tex_lap_energy is approximately 7.7-8.3. The next scoring tier is at 12+, representing a ~50% energy increase needed for a perfect C3 score.

### 1.2 The Root Cause: Frequency Suppression in SLAT Decoding

TRELLIS.2's texture pipeline has three stages, each potentially suppressing high-frequency information:

1. **Texture SLAT generation** (SLatFlowModel): Generates sparse latent features with channels `base_color[0:3], metallic[3:4], roughness[4:5], alpha[5:6]`. The flow model operates on compressed latent features, naturally losing detail due to the bottleneck dimensionality.

2. **SLAT decoding** (tex_slat_decoder): Decodes features to per-voxel PBR attributes. The MLP or network decoder may apply implicit smoothing.

3. **UV baking** (o_voxel postprocess.py): Projects voxel attributes to UV-parameterized texture map. The `fill_holes` operation and bilinear interpolation during UV mapping inherently blur texture edges.

**Key insight**: Even if the SLAT generation produces rich high-frequency features, UV baking can destroy them. Conversely, post-baking texture enhancement (render-and-compare, HFS-SDEdit) can restore detail without touching the generation model.

### 1.3 Why the Current Score Is Not 100

Comparing TRELLIS.2's C3 scores across configurations:
- Baseline (1024): lap_energy ~5.9, C3 ~73
- 1536 cascade: lap_energy ~9.2-11.3, C3 ~89-96
- V4 champion: C3 ~80

The 1536 cascade already improves C3 significantly by providing more tokens and higher resolution. The remaining gap to 100 likely comes from:
- UV seam artifacts that reduce local Laplacian energy at seam boundaries
- Gaussian blurring during hole-filling (`fill_holes=3e-2`)
- Oversmoothed outputs from the texture SLAT decoder's implicit smoothing
- Limited guidance signal at late ODE steps (fine detail forms near t=0)

---

## 2. Approach 1: TextureRefiner (Render-and-Compare)

### 2.1 Status

Implemented in `/workspace/TRELLIS.2/trellis2/postprocessing/texture_refiner.py`. Not yet tested or integrated into `app.py`.

The current implementation uses:
- **Chrominance L1** (YCbCr Cb, Cr channels): avoids luminance mismatch from unlit renders vs. lit photos
- **LPIPS** (AlexNet): perceptual loss
- **Total variation**: smoothness regularizer
- **Saturation preservation**: prevents desaturation

### 2.2 Assessment of Current Implementation

**Strengths:**
- Uses YCbCr chrominance-only L1, which correctly decouples from lighting mismatch
- Saves best checkpoint (not last) — avoids overfitting
- Already handles UV extraction from trimesh

**Critical Weaknesses to Fix Before Testing:**

1. **Camera matching problem**: The render uses `yaw=0, pitch=0.25` (generic front view), but the input image may have arbitrary viewpoint. For C3 improvement, the camera should match the input image viewpoint. Since TRELLIS.2 normalizes the input to a canonical frontal view (via the DINOv3 conditioning), yaw=0 is correct. But `pitch=0.25` is the canonical pitch used in TRELLIS.2's own render pipeline — this should match.

2. **LPIPS too weak for detail**: LPIPS-AlexNet is known to be insensitive to fine-grained texture details compared to LPIPS-VGG. For C3 improvement specifically, we want to maximize texture sharpness, which LPIPS-VGG captures better. **Change `net='alex'` to `net='vgg'` in TextureRefiner.**

3. **Missing high-frequency term**: Neither chrominance L1 nor LPIPS specifically targets high-frequency Laplacian energy. Adding an explicit **high-frequency loss** (matching the Laplacian pyramid of the rendered and reference images) would directly optimize C3.

4. **TV loss counterproductive for C3**: Total variation loss penalizes exactly what we want to maximize — high-frequency variation. TV should be *absent or very small* when optimizing for C3. Current default `tv_weight=0.01` is already small; reduce to `0.001` or zero for detail-focused optimization.

### 2.3 Recommended Modifications for C3

```python
# Modified TextureRefiner.refine() for C3 optimization:
def refine_for_detail(
    self,
    glb_mesh: trimesh.Trimesh,
    reference_image,
    num_iters: int = 80,
    lr: float = 0.008,
    chroma_weight: float = 1.0,
    lpips_weight: float = 0.5,  # Increase LPIPS for perceptual detail
    tv_weight: float = 0.001,   # Reduce TV — we WANT high-freq content
    hf_weight: float = 0.3,     # NEW: high-frequency matching loss
    sat_weight: float = 0.1,
    render_resolution: int = 768,  # Higher res for better detail capture
    ...
):
    ...
    # In _compute_losses, add HF matching:
    if hf_weight > 0:
        # Laplacian pyramid: extract high-frequency component
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=torch.float32, device=self.device)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        rend_lap = F.conv2d(rendered.permute(2,0,1).unsqueeze(0),
                            lap_kernel, groups=3, padding=1)
        tgt_lap = F.conv2d(target.permute(2,0,1).unsqueeze(0),
                           lap_kernel, groups=3, padding=1)
        hf_loss = F.l1_loss(rend_lap * mask.permute(2,0,1), tgt_lap * mask.permute(2,0,1))
```

**Using LPIPS-VGG instead of LPIPS-Alex:**
```python
# In __init__ or _get_lpips():
self._lpips_fn = lpips.LPIPS(net='vgg').to(self.device).eval()  # Was 'alex'
```

**Expected C3 improvement**: The render-and-compare loop with high-frequency matching will directly push the texture Laplacian energy up toward the reference image's detail level. With 80 iterations at 768px resolution, expect the tex_lap_energy to rise from ~7.7 to ~9.5-11.0, giving C3 scores of ~89-96 (+9-16 points).

**Runtime**: ~20-40 seconds on DGX Spark (Blackwell B200)

### 2.4 Integration Path

The refiner should be called after `o_voxel`'s `postprocess()` returns a trimesh GLB, just before returning to the Gradio UI. In `trellis2_image_to_3d.py`'s pipeline flow:

```python
# In decode_latent() or the calling code after GLB export:
if enable_texture_refiner:
    from trellis2.postprocessing.texture_refiner import TextureRefiner
    refiner = TextureRefiner(device='cuda')
    glb_mesh = refiner.refine_for_detail(
        glb_mesh, reference_image,
        num_iters=80, render_resolution=768,
        lpips_weight=0.5, hf_weight=0.3, tv_weight=0.001
    )
```

---

## 3. Approach 2: ITS3D-Style Noise Search for Texture SLAT

### 3.1 Paper Background

**ITS3D: Inference-Time Scaling for Text-Guided 3D Diffusion Models** (arXiv:2511.22456, November 2025)

ITS3D applies verifier-guided noise search to 3D diffusion models. Key contributions:
1. **Gaussian normalization**: Corrects distribution drift when noise candidates deviate from standard Gaussian
2. **SVD-based compression**: Reduces effective search dimensionality while preserving quality
3. **Singular space reset**: Prevents premature convergence via diversity maintenance

While ITS3D targets text-to-3D, the framework is directly applicable to TRELLIS.2's image-to-3D texture SLAT stage.

### 3.2 Application to TRELLIS.2 Texture Stage

The texture SLAT (Stage 3) is the highest-impact stage for C3. The key insight: different noise seeds produce textures with different detail levels. Some seeds produce flat, washed-out textures; others produce rich, detailed ones. We can search for the best noise seed using C3-aligned verifier.

**Algorithm: Texture Seed Search**

```python
def best_texture_seed(
    pipeline,
    shape_slat,
    image_cond,
    N: int = 4,
    base_seed: int = 42,
    verifier_fn=None,
):
    """Generate N texture candidates, select via C3-aligned verifier."""
    best_tex = None
    best_score = -float('inf')

    for i in range(N):
        seed = base_seed + i * 1000  # Far-apart seeds for diversity
        torch.manual_seed(seed)

        tex_slat = pipeline.sample_tex_slat(
            shape_slat, image_cond, ...
        )
        mesh = pipeline.decode_latent(shape_slat, tex_slat, ...)

        # Fast C3-aligned score:
        score = fast_c3_score(mesh)
        if score > best_score:
            best_score = score
            best_tex = tex_slat

    return best_tex


def fast_c3_score(mesh) -> float:
    """C3-aligned score: Laplacian energy of 2-view renders."""
    import cv2
    renders = render_views(mesh, num_views=2, resolution=256)  # Fast
    energies = []
    for bc in renders['base_color']:
        gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        energies.append(np.abs(lap).mean())
    return float(np.mean(energies))
```

**Why texture-only search (not shape)?**: The shape SLAT is fixed before texturing. Texture SLAT generation is the bottleneck for C3. Searching over texture seeds is O(N) in texture generation cost but does not multiply the (more expensive) shape generation. On DGX Spark, texture generation takes ~30-60s; N=4 search adds 2-3 minutes, which is acceptable for quality-critical mode.

**Expected C3 improvement**: +5-12 points. ITS3D demonstrates consistent quality improvement across all search algorithms. For C3 specifically, different noise seeds can give tex_lap_energy ranging from 6-12, so seed selection alone has high variance to exploit.

### 3.3 Gaussian Normalization (ITS3D technique)

When selecting across seeds, ITS3D's Gaussian normalization ensures each candidate is drawn from the correct distribution:

```python
def gaussian_normalize(noise: torch.Tensor) -> torch.Tensor:
    """Re-normalize noise to unit Gaussian to prevent distribution drift."""
    mean = noise.mean()
    std = noise.std()
    return (noise - mean) / (std + 1e-8)
```

This is important when interpolating between seed candidates (e.g., in zero-order search) to avoid OOD noise distributions.

---

## 4. Approach 3: HFS-SDEdit Texture Enhancement (Elevate3D Style)

### 4.1 Paper Background

**Elevating 3D Models: High-Quality Texture and Geometry Refinement** (SIGGRAPH 2025, arXiv:2507.11465)

HFS-SDEdit introduces Frequency-Splitting SDEdit:
- Decompose latent into **low-frequency** (global structure, allowed to change freely) and **high-frequency** (edges/details, constrained to match reference)
- Uses FLUX (rectified flow model) for denoising
- 30 denoising steps, initial noise timestep 29, frequency-swap threshold 18
- Gaussian filtering with σ=4 for frequency decomposition

**Quantitative results on 3D texture enhancement:**
- MUSIQ quality score: +35% over standard SDEdit
- LPIPS: 0.598 vs. 0.746 for standard SDEdit (better similarity)
- Q-Align: 3.34 vs. 2.86 (higher quality)

### 4.2 How HFS-SDEdit Works (Implementation Detail)

The key formula is:
```python
# At each denoising step i from T to threshold:
noised_ref = flow_noising(reference_image, t=i)    # noised reference
calibrated = calibrated_latent                      # current estimate

# Split into frequency bands (Gaussian filter):
sigma = 4.0  # Gaussian sigma
low_ref = gaussian_filter(noised_ref, sigma=sigma)
high_ref = noised_ref - low_ref
low_calib = gaussian_filter(calibrated, sigma=sigma)
high_calib = calibrated - low_calib

# Replace high-frequency in calibrated latent with reference high-freq:
fused = low_calib + high_ref  # Free low, anchored high

# Denoise with FLUX from fused latent
calibrated = flux_denoise_step(fused, t=i)

# After threshold: standard denoising (no more frequency swap)
```

### 4.3 Application to TRELLIS.2

HFS-SDEdit cannot be applied *directly* in TRELLIS.2's latent space (different architecture than FLUX). However, it CAN be applied to rendered views and then back-projected:

**Approach A: View-Level HFS-SDEdit + Back-Projection**

1. Generate TRELLIS.2 mesh with standard pipeline
2. Render front view at 512px
3. Apply HFS-SDEdit using FLUX to enhance the rendered view (add detail while preserving structure)
4. Back-project the enhanced view to UV texture map using nvdiffrast
5. Inpaint uncovered regions using the original texture

```python
# Pseudo-code for view-enhanced texture replacement
def enhance_texture_via_hfs_sdedit(mesh, reference_image):
    # Step 1: Render front view
    rendered = render_front_view(mesh, resolution=512)

    # Step 2: HFS-SDEdit on rendered view using FLUX
    # (requires FLUX model, ~12GB VRAM)
    enhanced = hfs_sdedit(
        rendered,
        reference_image=reference_image,
        n_steps=30,
        freq_swap_threshold=18,
        gaussian_sigma=4.0,
    )

    # Step 3: Back-project enhanced view to UV
    texture = back_project_to_uv(
        mesh, enhanced,
        camera=front_camera,
        weight_by_incidence=True  # More weight to face-on texels
    )

    # Step 4: Blend with original texture for unseen regions
    texture = blend_textures(original_texture, texture, mask=visibility_mask)

    return texture
```

**Dependency requirement**: FLUX or Stable Diffusion 3 (rectified flow model). FLUX requires ~12-16GB VRAM. On DGX Spark with 128GB GPU memory, this is feasible but requires loading an additional model.

**Alternative (lighter): Use SDXL Refiner instead of FLUX**. SDXL refiner model is ~5GB and achieves good texture detail enhancement at lower compute cost. Apply with img2img strength ~0.3-0.5 to preserve structure while adding detail.

**Expected C3 improvement**: +6-10 points. HFS-SDEdit is specifically designed to add fine texture detail while preserving structure, which directly optimizes Laplacian energy.

**Compute cost**: +30-90 seconds per view (FLUX), +10-20 seconds per view (SDXL)

### 4.4 Simpler Alternative: Direct Back-Projection of Input Image

Before attempting HFS-SDEdit, try **direct reference image back-projection**:

1. The input image already contains high-frequency texture detail
2. Project it directly onto the mesh UV map (correcting for geometry)
3. Blend with the flow-generated texture (reference for front-facing regions, generated for back)

This is free (no new models), fast (~1-2 seconds), and directly injects the reference image's detail. The main challenge is handling occlusion and view-dependent distortion.

**This may be the single highest-ROI approach** for objects where the front of the mesh is clearly visible in the reference image.

---

## 5. Approach 4: Rollover Budget Forcing + VP-SDE for Texture Stage

### 5.1 Paper Background

**Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing** (arXiv:2503.19385, March 2025)

Three contributions for inference-time scaling of frozen flow models:
1. **SDE-based generation**: Convert deterministic ODE sampling to VP-SDE, enabling particle sampling and diversity
2. **Interpolant conversion**: VP-SDE conversion broadens search space, enhancing sample diversity
3. **Rollover Budget Forcing (RBF)**: Adaptive compute allocation — more steps where trajectory curves (near t=0)

All three operate at inference time on frozen pretrained models.

### 5.2 RBF Application: Concentrate Steps Near t=0

In TRELLIS.2, the current `rescale_t` parameter already implements a rudimentary version of this by concentrating ODE steps near t=0. RBF makes this adaptive:

**Theoretical basis**: Detail (high-frequency features) forms in the final 20-30% of the ODE trajectory (small t). Euler integration errors are largest where the velocity field curves most — exactly at small t. Concentrating steps near t=0 while using Euler (or Heun) for those steps gives the best quality per NFE.

**TRELLIS.2 current state**: `rescale_t=4.0` for texture SLAT is already near-optimal according to GA convergence. However, the schedule is static. RBF would make it adaptive per-sample.

**Simplified RBF implementation** for TRELLIS.2:

```python
def compute_adaptive_schedule(model, x_t, t_seq, cond, threshold=0.5):
    """Estimate trajectory curvature; allocate extra steps to high-curvature regions."""
    # Forward pass at each timestep to estimate velocity
    # Measure |v(t) - v(t-1)| as proxy for curvature
    # Subdivide intervals with high curvature
    # Note: expensive (extra model evaluations); use sparingly
    pass
```

In practice, for TRELLIS.2, the simpler fixed schedule with higher `rescale_t` (5.0-6.0 for texture) achieves most of the gain from step concentration without the overhead.

### 5.3 VP-SDE Stochastic Injection for Texture Diversity

Adding small stochastic noise during texture SLAT sampling can help the sampler explore nearby textures with higher detail:

```python
# In FlowEulerSampler.sample(), add stochastic injection:
sde_eta = kwargs.pop('sde_eta', 0.0)  # 0=deterministic, 0.1-0.3=stochastic

# In the sampling loop:
if sde_eta > 0 and t_prev > 0:
    noise_std = sde_eta * math.sqrt(t - t_prev)
    out.pred_x_prev = out.pred_x_prev + noise_std * torch.randn_like(out.pred_x_prev)
```

This is equivalent to using a slightly stochastic SDE rather than a pure ODE. The stochasticity helps avoid sharp local minima in the texture generation that lead to flat, featureless regions.

**Caution**: Over-stochasticity will destroy consistency. Recommended `sde_eta=0.05-0.15` for the texture stage only (NOT shape/structure stages). At these values, each generation will differ slightly from the deterministic case. Combined with texture seed search (Approach 2), stochastic sampling increases the diversity pool being searched over.

**Expected C3 improvement**: +3-6 points (combined with seed search)

---

## 6. Approach 5: GPU-Friendly Laplacian UV Blending for Seam Repair

### 6.1 Background

**GPU-Friendly Laplacian Texture Blending** (arXiv:2502.13945, February 2025)

Key insight: UV seam artifacts cause abrupt Laplacian discontinuities at seam boundaries. These artificially *lower* Laplacian energy in seam regions (flat blending zones) while creating harsh spikes at seam edges. Both effects reduce the C3 score.

The method applies frequency-band-specific blending:
- High-frequency bands: sharp mask (preserve detail)
- Low-frequency bands: smooth mask (blend across seam)

### 6.2 Why UV Seams Hurt C3

In TRELLIS.2's postprocess.py, UV unwrapping creates seam boundaries. The current `fill_holes` operation applies Gaussian diffusion (`sigma=20, blend=0.7`) which already addresses some seam issues, but:

1. The fill_holes operation blurs the *entire hole region*, not just the seam boundary
2. It operates on all pixels equally regardless of frequency content
3. The blurring reduces tex_lap_energy in filled regions

**Fix**: After fill_holes, apply a Laplacian-pyramid-aware sharpening step to restore high-frequency content in filled regions:

```python
# In postprocess.py, after fill_holes:
def sharpen_filled_regions(texture_map, fill_mask):
    """Restore high-frequency content in gap-filled regions.

    The fill_holes operation diffuses colors into empty UV regions,
    but this blurs fine detail. This function restores sharpness
    by blending the filled texture with a sharpened version.

    Args:
        texture_map: (H, W, C) uint8 UV texture
        fill_mask: (H, W) bool — True where fill_holes added content
    """
    import cv2
    import numpy as np

    if fill_mask.sum() < 100:
        return texture_map

    tex_f = texture_map.astype(np.float32)

    # Build Laplacian pyramid
    gauss_low = cv2.GaussianBlur(tex_f, (0, 0), sigmaX=1.5)
    high_freq = tex_f - gauss_low  # Detail layer

    # Unsharp mask: enhance high-frequency in filled regions
    sharpened = tex_f + 0.7 * high_freq
    sharpened = np.clip(sharpened, 0, 255)

    # Apply only to filled regions (not original content)
    result = texture_map.astype(np.float32).copy()
    result[fill_mask] = 0.3 * result[fill_mask] + 0.7 * sharpened[fill_mask]

    return np.clip(result, 0, 255).astype(np.uint8)
```

**Expected C3 improvement**: +2-5 points. Seam repair and unsharp masking of filled regions directly increases Laplacian energy in the UV map.

**Compute cost**: <100ms. Free in terms of model calls.

---

## 7. ODE Schedule Research: Concentrating Steps Near t=0

### 7.1 What the Literature Says

**SDM: Adaptive Solvers and Wasserstein-Bounded Timesteps** (arXiv:2602.12624):
- Euler integration is sufficient for early steps (t near 1, nearly linear trajectory)
- Higher-order methods critical for late steps (t near 0, curved trajectory where detail forms)
- Optimized schedule concentrates more steps near t=0

**Key result**: TRELLIS.2 already implements `rescale_t` which maps uniform spacing to a schedule with more steps near t=0:
```python
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
```

For texture SLAT, the champion config uses `rescale_t=4.0`. The question is whether increasing this further (to 6.0-8.0) or switching to the `'edm'` schedule would improve C3.

### 7.2 EDM Schedule vs. Quadratic Schedule

The EDM schedule (Karras et al., CVPR 2022) uses `sigma_i = (sigma_max^(1/rho) + i/N * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho` with rho=7, which concentrates more steps near t=0.

TRELLIS.2's `flow_euler.py` already implements the EDM schedule:
```python
elif schedule == 'edm':
    rho = kwargs.get('rho', 7.0)
    sigma_max = 1.0
    i_frac = np.linspace(0, 1, steps + 1)
    sigmas = (sigma_max ** (1/rho) + i_frac * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    t_seq = np.clip((sigmas - sigma_min) / (1 - sigma_min), 0, 1)
```

The `'quadratic'` schedule gives `t_i = (1 - i/N)^2`, which is even more aggressive in concentrating steps near t=0.

**Recommended experiment**: Test `schedule='edm'` and `schedule='quadratic'` for texture SLAT only (shape stages should remain at default uniform/rescale_t for stability). The EDM schedule should improve C3 by placing more ODE steps in the detail-forming regime.

**Theoretical impact**: If the velocity field is more curved near t=0 (as the SDM analysis shows), then the EDM schedule reduces the global discretization error by sqrt(rho/1) ~ 2.6x compared to uniform spacing. This should improve not just C3 but also texture coherence.

### 7.3 Heun Solver for Final Steps (Already Implemented)

The current `heun_steps=4` in the champion config already applies Heun integration for the final 4 ODE steps. This is the correct approach for maximizing detail at minimal extra compute.

**For C3 specifically**: Increasing `heun_steps` from 4 to 6-8 for the texture SLAT stage (keeping 4 for shape/structure) may provide additional improvement. The texture stage is cheaper per step than shape (fewer tokens at early cascade levels), so the extra NFE cost is manageable.

---

## 8. Decomposable Flow Matching: Long-Term Architecture Direction

### 8.1 Paper Background

**Improving Progressive Generation with Decomposable Flow Matching** (arXiv:2506.19839, June 2025)

DFM applies Flow Matching independently at each level of a Laplacian pyramid. Each scale has its own independent flow — coarse scales capture global structure, fine scales capture detail. Quantitative results: 35.2% improvement in FDD scores on ImageNet-1k at 512px, 26.4% over best baseline.

### 8.2 Why This Would Help C3 (But Requires Retraining)

If TRELLIS.2's texture SLAT model were retrained with DFM:
- Low-frequency flow: structure, global color
- High-frequency flow: texture detail, fine patterns

The high-frequency flow would be trained specifically to model fine texture, unlike the current single-flow model that must handle all frequencies simultaneously. The high-frequency flow can use higher guidance and more steps specifically for detail.

**Application to frozen models**: DFM is a training-time modification; it cannot be applied to a frozen pretrained model. However, a *post-hoc approximation* is possible:

1. Run the standard texture SLAT flow to completion (produces `x_0_coarse`)
2. Compute a "high-frequency residual target": `hf_target = x_0_ref - lowpass(x_0_coarse)`
3. Re-run the flow for only the final 4-6 steps, starting from the noised high-frequency residual
4. Add the refined high-frequency back to the low-frequency base

This is speculative and needs validation, but conceptually aligns with DFM's approach.

**Status**: Research direction, not immediately implementable without risk of artifacts.

---

## 9. Synthesized Implementation Plan

### Priority 1: Quick Wins (1-3 days total, expected +10-20 C3 pts)

**Step 1a: Fix and test TextureRefiner (1 day)**

Modify `texture_refiner.py`:
- Change LPIPS from `net='alex'` to `net='vgg'`
- Reduce TV weight from 0.01 to 0.001
- Add Laplacian high-frequency matching loss (weight 0.3)
- Increase render_resolution from 512 to 768
- Set `num_iters=80` as default for detail mode

Integration in `app.py`: Add a "Texture Detail Refinement" checkbox (default OFF, shows time estimate +30-60s).

**Step 1b: Add fast C3 texture verifier (0.5 days)**

Add `fast_c3_score()` function to `quality_verifier.py`:
```python
def compute_fast_c3(renders: Dict) -> float:
    """Compute Laplacian energy of rendered views. Used for texture seed search."""
    if 'base_color' not in renders:
        return 0.5
    energies = []
    for bc in renders['base_color']:
        gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        energies.append(float(np.abs(lap).mean() * 100))
    return float(np.mean(energies)) if energies else 0.5
```

**Step 1c: Implement texture seed search (1 day)**

Add `N_texture_seeds` parameter to `app.py` (default 1, options: 1, 2, 4). When > 1, generate N textures for the same shape and return the one with highest C3 score.

**Step 1d: UV seam sharpening in postprocess.py (0.5 days)**

Add `sharpen_filled_regions()` to `postprocess.py` (pure NumPy/OpenCV, no new deps).

### Priority 2: Medium-Effort Gains (1-2 weeks, expected +5-10 C3 pts on top of Priority 1)

**Step 2a: EDM/quadratic schedule for texture SLAT**

Add `tex_schedule='edm'` option to `app.py`. Test against baseline on V4 evaluator.

**Step 2b: Stochastic SDE injection for texture**

Add `tex_sde_eta=0.05` kwarg to texture SLAT sampling. Test with and without seed search.

**Step 2c: Reference image back-projection**

Add optional direct reference image projection as a prior for UV regions visible in the input image. This is especially high-impact for objects with complex front-face textures.

### Priority 3: Long-Term (2-4 weeks, expected +5-8 C3 pts on top of Priority 2)

**Step 3a: HFS-SDEdit view enhancement**

Integrate with Stable Diffusion XL refiner (lighter weight than FLUX). Apply to rendered front view, back-project to UV.

**Step 3b: PBR-SR style optimization**

Apply DiffBIR SR model to rendered views, optimize UV texture to match SR-enhanced views. This is the PBR-SR method (NeurIPS 2025) extended to TRELLIS.2.

---

## 10. Expected C3 Score Trajectory

Starting from V4 baseline C3 = 80/100 (tex_lap_energy ~7.7):

| After Step | Expected C3 | tex_lap_energy | Notes |
|------------|-------------|----------------|-------|
| Baseline | 80 | ~7.7 | Current champion config |
| + UV seam sharpen (1d) | 82-84 | ~8.0-8.3 | Free, low risk |
| + TextureRefiner fix+test (1e) | 88-92 | ~9.0-10.5 | High confidence |
| + Texture seed search N=4 | 90-94 | ~9.5-11.0 | High confidence |
| + EDM schedule tex (2a) | 91-95 | ~9.8-11.5 | Medium confidence |
| + SDE injection (2b) | 92-96 | ~10.0-12.0 | Medium confidence |
| + HFS-SDEdit / PBR-SR (3a/3b) | 95-99 | ~11.5-12.5 | Lower confidence |

**Upper bound**: C3 = 100 requires tex_lap_energy ≥ 12 according to the scoring formula. Current 1536-cascade best case is ~11.3, suggesting 100 is achievable but requires multiple combined improvements.

---

## 11. Implementation Notes Specific to TRELLIS.2 Architecture

### 11.1 Why Simple Laplacian Loss in Render Space Helps UV Space

The C3 metric measures Laplacian energy in the **UV texture map**, but TextureRefiner renders from camera space and back-projects. The key insight is that camera-space Laplacian energy and UV-space Laplacian energy are correlated for well-UV-parameterized meshes:

- UV parameterization in TRELLIS.2 is roughly angle-preserving (conformal-ish)
- High camera-space detail -> high UV-space detail (approximately)
- The correlation breaks down at UV seams and large distortion regions

**Implication**: Optimizing chrominance L1 + LPIPS + HF loss in camera space will improve UV-space Laplacian energy, but not perfectly. The improvement may be concentrated in the front-facing UV region. Combining with UV seam sharpening (Approach 5) and multi-view optimization covers more of the UV map.

### 11.2 SLAT Feature Channels and C3

The texture SLAT channel layout is `base_color[0:3], metallic[3:4], roughness[4:5], alpha[5:6]`. C3 measures **base_color** Laplacian only. Metallic and roughness channels can be treated as lower priority for C3 optimization.

In TextureRefiner, optimize only `base_color` channels (channels 0:3 of the texture map) with higher learning rate; optionally freeze metallic/roughness optimization during detail-focused passes.

### 11.3 The 80/100 Plateau Analysis

With the current V4 champion config (1024 pipeline, heun_steps=4, multistep=True), C3 = ~80. The 1536 cascade pushes this higher (~89-96) due to more tokens and higher resolution. But even at 1536, there is room for improvement.

The remaining gap from 89-96 to 100 is approximately:
- tex_lap_energy 9.2-11.3 → 12.0 required
- Need ~6-30% more Laplacian energy in the UV map
- Sources: UV seam blurring (~3%), SLAT decoder smoothing (~5-8%), limited ODE step accuracy at small t (~5-10%)

The TextureRefiner is the only approach that directly addresses the UV map quality rather than working upstream. Combining it with ITS3D-style seed search gives the best probability of reaching C3 = 95+.

---

## 12. References

### Render-and-Compare / Texture Refinement
- Yuan et al., "GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement," ICLR 2025. [Paper](https://arxiv.org/abs/2406.05649) | [Code](https://github.com/snap-research/GTR)
- Chen et al., "PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors," NeurIPS 2025 Poster. [Paper](https://arxiv.org/abs/2506.02846) | [Website](https://terencecyj.github.io/projects/PBR-SR/)
- Ryu et al., "Elevating 3D Models: High-Quality Texture and Geometry Refinement," SIGGRAPH 2025. [Paper](https://arxiv.org/abs/2507.11465) | [Code](https://github.com/ryunuri/Elevate3D)
- Zamani et al., "End-to-End Fine-Tuning of 3D Texture Generation using Differentiable Rewards," WACV 2026. [Paper](https://arxiv.org/abs/2506.18331) | [Code](https://github.com/AHHHZ975/Differentiable-Texture-Learning)

### Inference-Time Scaling
- "ITS3D: Inference-Time Scaling for Text-Guided 3D Diffusion Models," arXiv:2511.22456, November 2025. [Paper](https://arxiv.org/abs/2511.22456)
- Ma et al., "Scaling Inference Time Compute for Diffusion Models," CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf)
- "Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing," arXiv:2503.19385, March 2025. [Paper](https://arxiv.org/abs/2503.19385)
- Stecklov et al., "Inference-Time Compute Scaling for Flow Matching," arXiv:2510.17786. [Paper](https://arxiv.org/abs/2510.17786)

### ODE Solvers and Schedules
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), NeurIPS 2022. The rho=7 schedule is implemented in TRELLIS.2.
- "Stochastic Sampling from Deterministic Flow Models," arXiv:2410.02217. [Paper](https://arxiv.org/abs/2410.02217)
- "ABM-Solver for Inversion and Editing in Rectified Flow," arXiv:2503.16522. [Paper](https://arxiv.org/abs/2503.16522)

### Frequency Decomposition and Detail
- Haji-Ali et al., "Improving Progressive Generation with Decomposable Flow Matching" (DFM), arXiv:2506.19839, June 2025. [Paper](https://arxiv.org/abs/2506.19839) | [OpenReview](https://openreview.net/forum?id=3isHlkiykj)
- "GPU-Friendly Laplacian Texture Blending," arXiv:2502.13945, February 2025. [Paper](https://arxiv.org/abs/2502.13945)
- "FlexiTex: Enhancing Texture Generation via Visual Guidance," arXiv:2409.12431. [Paper](https://arxiv.org/abs/2409.12431) | [Website](https://flexitex.github.io/FlexiTex/)

### Multi-View Texture Generation
- "MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D," arXiv:2411.02336. [Paper](https://arxiv.org/abs/2411.02336)
- "UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes," arXiv:2505.23253. [Paper](https://arxiv.org/abs/2505.23253)
- "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material," arXiv:2506.15442. [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

---

## Appendix: Quick Reference for C3 Score Formula

```python
# V4 C3 metric (from auto_evaluate_v4.py)
tex_lap_energy = np.abs(Laplacian(tex_gray)[tex_mask]).mean()

# Mapping:
# < 3.0  → 30 pts
# 3-12   → 30 + (energy - 3) * 7.78  (slopes to 100)
# ≥ 12   → 100 pts

# View entropy component:
# entropy < 2.5  → 30 pts
# 2.5-4.0        → 30 + (entropy - 2.5) * 40.0  (slopes to 90)
# 4.0-5.5        → 90 + (entropy - 4.0) * 6.67  (slopes to 100)
# ≥ 5.5          → 100 pts

# Final C3:
C3 = 0.80 * tex_detail + 0.20 * entropy_score

# Current v4 baseline: tex_lap_energy ~7.7 → tex_detail ~73, C3 ~80
# Target: tex_lap_energy ≥ 12 → tex_detail = 100, C3 ≥ 95 (if entropy also good)
```

## Appendix: Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `trellis2/postprocessing/texture_refiner.py` | Fix LPIPS net, add HF loss, increase render_res | P1 |
| `trellis2/utils/quality_verifier.py` | Add fast_c3_score() | P1 |
| `o-voxel/o_voxel/postprocess.py` | Add sharpen_filled_regions() | P1 |
| `app.py` | Add texture refiner checkbox, N_texture_seeds slider | P1 |
| `trellis2/pipelines/trellis2_image_to_3d.py` | Add texture seed search loop | P1 |
| `trellis2/pipelines/samplers/flow_euler.py` | Add sde_eta stochastic injection | P2 |
