# Survey: Practical Methods to Improve Texture Coherence (C1) and Color Vitality (C2)

**Date**: 2026-02-22
**Author**: Research Optimizer Agent
**Scope**: Actionable improvements for TRELLIS.2 texture quality in postprocess.py and texture_refiner.py
**Constraint**: Implementable in 1-2 days with existing infrastructure (nvdiffrast, PyTorch, OpenCV, trimesh)
**Focus metrics**: C1 Texture Coherence (15pts), C2 Color Vitality (10pts), C3 Detail Richness (10pts)

---

## Table of Contents

1. [Root Cause Analysis](#1-root-cause-analysis)
2. [Ranked Improvement Methods](#2-ranked-improvement-methods)
3. [Detailed Implementation Plans](#3-detailed-implementation-plans)
4. [Expected Impact Summary](#4-expected-impact-summary)
5. [References](#5-references)

---

## 1. Root Cause Analysis

### 1.1 Where Grey/Desaturated Texels Come From

TRELLIS.2's texture pipeline has THREE stages where color information degrades:

**Stage A: Sparse Voxel Trilinear Interpolation (grid_sample_3d)**
The texture SLAT produces PBR attributes as features on sparse voxels. When `grid_sample_3d` with mode='trilinear' samples a position where some of the 8 neighboring voxels are empty (zero), it interpolates toward zero -- producing dark, desaturated values. This is the PRIMARY source of grey patches.

Evidence: BVH reprojection test showed 41.5% grey without vs 13.8% with, confirming that position accuracy at the voxel grid matters enormously.

The current `grid_sample_3d` from flex_gemm treats empty voxels as zero-valued. For a surface point near the edge of a voxel cluster, 3-4 of the 8 trilinear neighbors may be empty, diluting the color by 37-50%.

**Stage B: UV Baking Discretization**
The continuous voxel field is sampled at discrete texel centers. UV charts from xatlas fragment the surface into many small islands (body_count ~5000+). At island boundaries, the rasterization in UV space may sample positions slightly off-surface, hitting even more empty voxels. The inpainting step (cv2.INPAINT_TELEA, radius=3) fills gaps but introduces its own grey/muddy colors by averaging nearby valid texels.

**Stage C: Seam Discontinuities**
Adjacent UV islands map to distant positions in texture space. When the mesh is rendered with mipmapping, texels from different islands bleed into each other. Even without mipmapping, bilinear texture filtering at rendering time pulls in texels from neighboring (potentially empty or differently-colored) islands.

### 1.2 Why Current Grey Recovery is Insufficient

The existing grey recovery (`grey_recovery_rounds=2, sigma=25`) uses Gaussian-weighted diffusion from colorful texels to fill grey areas. This has two weaknesses:

1. **Threshold too strict for partial desaturation**: `chroma < 18` only catches severely grey texels. Texels with chroma 18-35 are still visibly washed out but escape detection.
2. **Gaussian blur crosses UV island boundaries**: The sigma=25 blur diffuses color across UV islands, potentially blending unrelated colors from different surface regions.
3. **Only 2 rounds**: Large grey patches (e.g., entire back of an object) may not be fully covered in 2 rounds with sigma=25. Sigma increases to 35 in round 2 but may still not reach the center of large grey regions.

### 1.3 Why Current Seam Smoothing is Insufficient

The `_laplacian_seam_smooth` function builds a Gaussian pyramid and blends smoothed versions at seam pixels. Issues:

1. **Seam detection by erosion is imprecise**: It detects ALL mask boundaries, not just UV chart boundaries. Actual UV seams are a subset of mask boundaries.
2. **Blend strength too low**: Alpha ranges from 0.3-0.7, leaving visible discontinuities.
3. **Only applied to base_color**: Metallic and roughness maps have the same seam issue.
4. **Applied AFTER inpainting**: The inpainting step may have already baked in grey/muddy colors at seam regions.

---

## 2. Ranked Improvement Methods

### Rank 1: Push-Pull Texture Padding (Replace cv2.inpaint)
- **Impact**: C1 +3-5pts, C2 +1-2pts
- **Effort**: 4-6 hours
- **Risk**: Very low

### Rank 2: Chroma-Aware Sampling Mode in grid_sample_3d
- **Impact**: C2 +3-5pts, C1 +1-2pts
- **Effort**: 4-8 hours
- **Risk**: Low-medium

### Rank 3: UV-Aware Seam Smoothing with Rasterized Face ID
- **Impact**: C1 +2-4pts
- **Effort**: 3-5 hours
- **Risk**: Low

### Rank 4: Saturation Boost in HSV Space for Low-Chroma Texels
- **Impact**: C2 +2-3pts
- **Effort**: 1-2 hours
- **Risk**: Low

### Rank 5: Enhanced Multi-View Texture Refinement (Fix TextureRefiner)
- **Impact**: C1 +3-5pts, C2 +2-4pts, C3 +2-3pts
- **Effort**: 6-10 hours
- **Risk**: Medium

### Rank 6: Increased Inpainting Dilation Before Baking
- **Impact**: C1 +1-3pts
- **Effort**: 2-3 hours
- **Risk**: Low

### Rank 7: Texture-Space Bilateral Denoising
- **Impact**: C1 +1-2pts, C3 -1pt (slight detail loss)
- **Effort**: 2-3 hours
- **Risk**: Low

### Rank 8: UV Island Stitching via Shared Vertex Colors
- **Impact**: C1 +2-3pts
- **Effort**: 8-12 hours
- **Risk**: Medium

---

## 3. Detailed Implementation Plans

### 3.1 Push-Pull Texture Padding (RANK 1)

**Problem**: `cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)` uses Telea's method which averages nearby valid pixels. At UV island borders, this pulls in colors from unrelated islands, creating muddy blends. The radius of 3 pixels is also too small for mipmapping -- when the texture is downsampled 2x, the 3-pixel padding becomes 1.5 pixels, which is insufficient.

**Solution**: Replace cv2.inpaint with a push-pull algorithm that:
1. **Push** (downscale): Build a Gaussian pyramid of the texture, at each level replacing invalid pixels with the average of valid neighbors
2. **Pull** (upscale): Reconstruct from coarse to fine, using the coarser level to fill holes in the finer level

This produces padding that naturally extends the color of each UV island outward, without mixing colors across islands.

**Implementation**:
```python
def push_pull_inpaint(image, mask, max_levels=10):
    """
    Push-pull inpainting for UV texture padding.
    Fills invalid texels with colors that naturally extend from nearby valid texels.
    Superior to cv2.inpaint for UV maps because it respects island color gradients.

    Args:
        image: [H, W, C] float32 texture
        mask: [H, W] bool, True = valid texel
        max_levels: maximum pyramid levels

    Returns:
        [H, W, C] float32 with all texels filled
    """
    h, w = image.shape[:2]
    levels = min(max_levels, int(np.log2(min(h, w))))

    # Push phase: build pyramid
    pyramid_img = [image.copy()]
    pyramid_mask = [mask.astype(np.float32)]

    for level in range(1, levels):
        prev_img = pyramid_img[-1]
        prev_mask = pyramid_mask[-1]

        # Weight image by mask for correct averaging
        weighted = prev_img * prev_mask[..., None]

        # Downsample both
        down_weighted = cv2.resize(weighted, (w >> level, h >> level), interpolation=cv2.INTER_AREA)
        down_mask = cv2.resize(prev_mask, (w >> level, h >> level), interpolation=cv2.INTER_AREA)

        # Normalize
        safe_mask = np.maximum(down_mask, 1e-8)
        down_img = down_weighted / safe_mask[..., None]

        pyramid_img.append(down_img)
        pyramid_mask.append(down_mask)

    # Pull phase: reconstruct from coarse to fine
    result = pyramid_img[-1].copy()
    result_mask = pyramid_mask[-1].copy()

    for level in range(levels - 2, -1, -1):
        target_h, target_w = pyramid_img[level].shape[:2]

        # Upsample coarse result
        upsampled = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Use fine-level valid pixels where available, coarse fill elsewhere
        fine_mask = pyramid_mask[level]
        result = np.where(fine_mask[..., None] > 0.5,
                          pyramid_img[level],
                          upsampled)
        result_mask = np.maximum(fine_mask, cv2.resize(result_mask, (target_w, target_h),
                                                        interpolation=cv2.INTER_LINEAR))

    return np.clip(result, 0, 1)
```

**Where to apply**: In `postprocess.py`, replace the `cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)` call with push_pull_inpaint. Keep cv2.inpaint as fallback for metallic/roughness (less critical channels).

**Increase padding radius**: Change from 3 to at least 8 pixels for base_color. Even with push-pull, the traditional inpaint can serve as a secondary fill for any remaining gaps. Industry standard is 4-8 pixel padding (Adobe Substance uses 8px default, TurboSquid requires minimum 4px).

**Expected C1 improvement**: +3-5 points. Push-pull naturally extends island colors outward without cross-island contamination, reducing visible seam artifacts in renders.

**Expected C2 improvement**: +1-2 points. Eliminates muddy grey padding at island borders that gets sampled during bilinear texture filtering.


### 3.2 Chroma-Aware Sampling with Boundary Detection (RANK 2)

**Problem**: `grid_sample_3d` with `mode='trilinear'` interpolates toward zero at sparse voxel boundaries. For a surface point where 3 of 8 neighbors are empty, the sampled color is diluted to ~62% of its true value. This is the root cause of grey patches.

**Solution A (easy, in postprocess.py)**: After sampling, detect low-chroma texels that are near high-chroma texels and boost their saturation toward the local average.

```python
def boundary_chroma_recovery(base_color, mask, attrs_raw=None):
    """
    Recover desaturated texels caused by trilinear interpolation at voxel boundaries.

    Instead of simple grey recovery (chroma < 18), detect texels whose chroma is
    significantly lower than their local neighborhood and boost toward the local median.

    Args:
        base_color: [H, W, 3] uint8
        mask: [H, W] bool
    Returns:
        [H, W, 3] uint8 with recovered colors
    """
    bc_float = base_color.astype(np.float32)

    # Convert to HSV for chroma manipulation
    hsv = cv2.cvtColor(base_color, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Local median saturation (31x31 window for context)
    sat_channel = hsv[..., 1]
    sat_masked = np.where(mask, sat_channel, 0).astype(np.float32)
    mask_float = mask.astype(np.float32)

    # Compute local average saturation using box filter
    local_sat_sum = cv2.blur(sat_masked, (31, 31))
    local_mask_sum = cv2.blur(mask_float, (31, 31))
    local_avg_sat = local_sat_sum / np.maximum(local_mask_sum, 1e-6)

    # Desaturated texels: saturation significantly below local average
    sat_deficit = local_avg_sat - sat_channel
    desaturated = mask & (sat_deficit > 30) & (sat_channel < 100) & (hsv[..., 2] > 40)

    # Boost saturation of desaturated texels toward local average
    boost_factor = 0.7  # blend 70% toward local average
    new_sat = sat_channel.copy()
    new_sat[desaturated] = (sat_channel[desaturated] * (1 - boost_factor) +
                            local_avg_sat[desaturated] * boost_factor)
    new_sat = np.clip(new_sat, 0, 255)

    hsv[..., 1] = new_sat
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Only modify desaturated pixels
    output = base_color.copy()
    output[desaturated] = result[desaturated]
    return output
```

**Solution B (harder, in grid_sample_3d wrapper)**: Detect empty neighbors before sampling and use only valid neighbors. This requires wrapping the grid_sample_3d call:

```python
def smart_grid_sample(attr_volume, sparse_coords, shape, grid_pts, mode='trilinear'):
    """
    Enhanced grid sampling that handles sparse voxel boundaries.

    For each query point, checks if all 8 trilinear neighbors have valid voxels.
    If not, falls back to nearest-valid-neighbor interpolation.
    """
    # First: standard trilinear sample
    sampled = grid_sample_3d(attr_volume, sparse_coords, shape=shape, grid=grid_pts, mode=mode)

    # Detect low-confidence samples by checking occupancy at neighbor positions
    # Create binary occupancy grid
    occ_volume = torch.zeros(1, 1, *shape[2:], device=attr_volume.device)
    occ_coords = sparse_coords.clone()
    occ_volume_flat = torch.ones(sparse_coords.shape[0], 1, device=attr_volume.device)

    # Sample occupancy at same positions (should be ~1.0 if all 8 neighbors exist)
    occ_sampled = grid_sample_3d(occ_volume_flat, occ_coords, shape=torch.Size([1, 1, *shape[2:]]),
                                  grid=grid_pts, mode=mode)

    # Where occupancy < 0.9 (some neighbors missing), boost the sample
    # by dividing by occupancy (inverse compositing)
    low_occ = occ_sampled.squeeze(-1) < 0.9
    if low_occ.any():
        safe_occ = torch.clamp(occ_sampled, min=0.1)
        # Rescale: if only 5/8 neighbors had values, multiply by 8/5
        sampled[low_occ] = sampled[low_occ] / safe_occ[low_occ]

    return sampled
```

**Note on Solution B**: This "occupancy normalization" approach is directly inspired by the Interpolation-Aware Padding paper (ICCV 2021) which found that padding empty voxels for sparse CNNs dramatically improves feature quality. For TRELLIS.2, we cannot pad the voxels (they come from the model), but we CAN normalize the interpolation result by the occupancy fraction.

**Where to apply**: Solution A goes in postprocess.py after base_color extraction (line 567), before grey recovery. Solution B wraps the grid_sample_3d call at line 407.

**Expected C2 improvement**: +3-5 points. Solution B is theoretically cleaner (fixes the source) while Solution A is safer (post-hoc correction).


### 3.3 UV-Aware Seam Smoothing with Rasterized Face ID (RANK 3)

**Problem**: Current `_detect_uv_seams` uses mask erosion which detects ALL mask boundaries, not just actual UV chart boundaries. This over-smooths legitimate texture boundaries and under-smooths actual seam discontinuities.

**Solution**: Use the rasterized face ID (`rast[0, ..., 3]`) to detect actual UV seams. Two adjacent texels whose rasterized face IDs correspond to faces on different UV charts are on a seam. Additionally, use the actual UV discontinuity (large UV gradient jump) to detect seams.

```python
def _detect_uv_seams_precise(rast, out_faces, out_uvs, texture_size, mask, width=3):
    """
    Precise UV seam detection using rasterized face IDs.

    A UV seam occurs where adjacent texels in the texture map correspond to
    faces that are adjacent in 3D but on different UV charts (their UVs are
    not continuous).

    Args:
        rast: [1, H, W, 4] rasterization output (face_id in channel 3)
        out_faces: [F, 3] face indices
        out_uvs: [V, 2] UV coordinates
        texture_size: int
        mask: [H, W] bool
        width: seam band width

    Returns:
        [H, W] bool seam mask
    """
    face_ids = rast[0, :, :, 3].cpu().numpy().astype(np.int32) - 1  # 0-indexed
    h, w = face_ids.shape

    seam_mask = np.zeros((h, w), dtype=bool)

    # For each texel, check if its 4-connected neighbors have the same face ID
    # or a face that shares a UV edge. Large face ID discontinuities = UV seam.
    for dy, dx in [(0, 1), (1, 0)]:
        # Shifted face IDs
        shifted = np.roll(face_ids, -dy, axis=0) if dy else np.roll(face_ids, -dx, axis=1)
        shifted_mask = np.roll(mask, -dy, axis=0) if dy else np.roll(mask, -dx, axis=1)

        # Both texels valid but face IDs differ significantly
        # Adjacent faces in 3D typically have close face IDs, but UV chart boundaries
        # map spatially-distant faces to adjacent texels
        both_valid = mask & shifted_mask
        face_diff = np.abs(face_ids - shifted) * both_valid
        seam_mask |= (face_diff > 2) & both_valid

    # Also detect UV discontinuities: large UV gradient in texture space
    if out_uvs is not None:
        uvs_np = out_uvs.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        # Build per-texel UV coordinate map
        # (already available from rasterization interpolation)
        pass  # UV gradient check is more complex; face ID is sufficient for most cases

    # Dilate to create a band
    if width > 1:
        kernel = np.ones((width, width), np.uint8)
        seam_mask = cv2.dilate(seam_mask.astype(np.uint8), kernel).astype(bool)

    return seam_mask & mask
```

**Improved smoothing**: Once we have precise seam locations, use a cross-bilateral filter that blends ONLY with texels on the SAME side of the seam, preventing cross-island color contamination:

```python
def seam_cross_bilateral_smooth(base_color, seam_mask, mask, sigma_space=5, sigma_color=30):
    """
    Cross-bilateral smoothing at UV seams.
    Only blends with similarly-colored neighbors on the same UV island side.
    """
    bc_float = base_color.astype(np.float32)
    smoothed = cv2.bilateralFilter(bc_float, d=11, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Apply only at seam pixels
    result = bc_float.copy()
    seam_3d = seam_mask[..., None].astype(np.float32)
    result = result * (1 - seam_3d) + smoothed * seam_3d
    return np.clip(result, 0, 255).astype(np.uint8)
```

**Where to apply**: Replace `_detect_uv_seams` and `_laplacian_seam_smooth` in postprocess.py.

**Expected C1 improvement**: +2-4 points. Precise seam detection avoids over-smoothing interior texels while properly handling actual seam discontinuities.


### 3.4 Saturation Boost for Low-Chroma Texels (RANK 4)

**Problem**: Even after grey recovery, many texels have chroma 18-40 (technically not "grey" by the current threshold) but are visibly washed out compared to the input image. The input images typically have vibrant colors that lose saturation through the voxel pipeline.

**Solution**: Adaptive saturation boost in HSV space, calibrated against the input image's color statistics.

```python
def adaptive_saturation_boost(base_color, mask, reference_image=None, boost_factor=1.3):
    """
    Boost saturation of desaturated texels toward the expected color vibrancy.

    If a reference image is provided, match the saturation histogram.
    Otherwise, apply a gentle multiplicative boost clamped to prevent oversaturation.

    Args:
        base_color: [H, W, 3] uint8
        mask: [H, W] bool
        reference_image: optional [H, W, 3] uint8 reference for calibration
        boost_factor: multiplicative saturation boost (1.0 = no change)
    Returns:
        [H, W, 3] uint8
    """
    hsv = cv2.cvtColor(base_color, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    if reference_image is not None:
        # Match saturation statistics to reference
        ref_hsv = cv2.cvtColor(reference_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        ref_sat = ref_hsv[..., 1]
        ref_mask = ref_hsv[..., 2] > 30  # exclude very dark reference pixels

        ref_mean_sat = ref_sat[ref_mask].mean() if ref_mask.any() else 100
        cur_mean_sat = sat[mask & (val > 30)].mean() if (mask & (val > 30)).any() else 80

        # Compute boost to match reference mean saturation
        if cur_mean_sat > 10:
            computed_boost = min(ref_mean_sat / cur_mean_sat, 1.8)  # cap at 1.8x
            boost_factor = max(boost_factor, computed_boost)

    # Apply boost only to mid-saturation texels (avoid boosting already-vivid colors)
    # Target: texels with saturation 15-120 (out of 255)
    boost_eligible = mask & (sat > 15) & (sat < 120) & (val > 30)

    new_sat = sat.copy()
    new_sat[boost_eligible] = np.clip(sat[boost_eligible] * boost_factor, 0, 255)

    hsv[..., 1] = new_sat
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    output = base_color.copy()
    output[mask] = result[mask]
    return output
```

**Where to apply**: In postprocess.py, after grey recovery and before inpainting. Pass the reference image through to to_glb (requires adding a parameter).

**Expected C2 improvement**: +2-3 points. Restores color vibrancy lost during voxel interpolation.


### 3.5 Enhanced Multi-View Texture Refinement (RANK 5)

**Problem**: The existing `TextureRefiner` has several issues:
1. **Single-view only in practice**: The multi-view path requires mv_images and mv_cam_params which are not always available
2. **Camera coordinate mismatch**: The postprocess pipeline applies Y/Z swap and Y flip (`vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]`), but the refiner camera setup doesn't account for this
3. **No mask-aware loss**: The L1 loss compares rendered foreground against reference foreground, but the reference background is black, which biases color toward darker values at object edges
4. **LPIPS at 256x256**: Downsampling to 256x256 loses fine detail that the perceptual loss should be capturing
5. **No style/color histogram loss**: L1 + LPIPS do not penalize global color distribution mismatch (e.g., overall desaturation)

**Improvements**:

```python
# 1. Add color histogram matching loss
def color_histogram_loss(rendered, target, mask, n_bins=64):
    """Differentiable histogram matching loss."""
    loss = torch.tensor(0.0, device=rendered.device)
    for c in range(3):
        r_vals = rendered[mask.squeeze(-1) > 0.5, c]
        t_vals = target[mask.squeeze(-1) > 0.5, c]
        if r_vals.numel() < 10:
            continue
        # Compute soft histograms using sigmoid binning
        bins = torch.linspace(0, 1, n_bins, device=rendered.device)
        sigma = 1.0 / n_bins
        r_hist = torch.sigmoid((r_vals.unsqueeze(1) - bins.unsqueeze(0)) / sigma).mean(0)
        t_hist = torch.sigmoid((t_vals.unsqueeze(1) - bins.unsqueeze(0)) / sigma).mean(0)
        loss = loss + F.l1_loss(r_hist, t_hist)
    return loss / 3.0

# 2. Add saturation preservation loss
def saturation_loss(rendered, target, mask):
    """Penalize rendered saturation being lower than target."""
    r_max = rendered.max(dim=-1).values
    r_min = rendered.min(dim=-1).values
    r_sat = (r_max - r_min) / (r_max + 1e-6)

    t_max = target.max(dim=-1).values
    t_min = target.min(dim=-1).values
    t_sat = (t_max - t_min) / (t_max + 1e-6)

    mask_1d = mask.squeeze(-1) > 0.5
    # Asymmetric: only penalize desaturation, not oversaturation
    deficit = F.relu(t_sat[mask_1d] - r_sat[mask_1d])
    return deficit.mean()

# 3. Run LPIPS at 512x512 instead of 256x256
# In _compute_lpips, change:
#   t1 = F.interpolate(t1, size=(512, 512), ...)

# 4. Add DreamSim loss (already installed in container)
# from dreamsim import dreamsim
# dreamsim_model, preprocess = dreamsim(pretrained=True, device='cuda')
# dreamsim_dist = dreamsim_model(rendered_preprocessed, target_preprocessed)

# 5. Multi-view with estimated camera params from generation
# The pipeline already has yaw/pitch used for multi-view generation.
# Pass these through to the refiner.
```

**Where to apply**: Modify `texture_refiner.py`. Add histogram_loss and saturation_loss to the optimization loop. Increase LPIPS resolution. Wire multi-view camera params from pipeline to refiner.

**Expected impact**: C1 +3-5pts (multi-view consistency), C2 +2-4pts (saturation preservation), C3 +2-3pts (higher-res perceptual loss).


### 3.6 Increased Inpainting Dilation (RANK 6)

**Problem**: The current inpainting radius of 3 pixels for base_color is the minimum viable. Industry standard (TurboSquid, Substance Painter) is 4-8 pixels. When textures are downsampled for mipmapping, 3 pixels of padding shrinks to 1.5 pixels, which allows empty-space colors to bleed into visible areas.

**Solution**:
1. Increase inpainting radius from 3 to 8 for base_color
2. Add explicit dilation BEFORE inpainting: iteratively dilate valid texels outward using morphological operations, which extends colors without averaging across islands
3. Apply push-pull (Rank 1) as primary fill, then cv2.inpaint with larger radius as secondary fill for any remaining gaps

```python
def dilate_texture(texture, mask, iterations=8):
    """
    Morphological dilation of texture into empty space.
    Extends border texels outward without averaging across islands.
    """
    result = texture.copy()
    current_mask = mask.copy()

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        # Find pixels to fill: invalid but adjacent to valid
        dilated_mask = cv2.dilate(current_mask.astype(np.uint8), kernel) > 0
        new_pixels = dilated_mask & ~current_mask

        if not new_pixels.any():
            break

        # For each new pixel, average its valid neighbors
        for c in range(texture.shape[2]):
            channel = result[..., c].astype(np.float32)
            valid_weighted = channel * current_mask.astype(np.float32)
            neighbor_sum = cv2.blur(valid_weighted, (3, 3)) * 9
            neighbor_count = cv2.blur(current_mask.astype(np.float32), (3, 3)) * 9
            safe_count = np.maximum(neighbor_count, 1)
            result[new_pixels, c] = (neighbor_sum[new_pixels] / safe_count[new_pixels]).astype(np.uint8)

        current_mask = dilated_mask

    return result
```

**Where to apply**: In postprocess.py, before the cv2.inpaint calls. Apply dilation first, then inpaint any remaining gaps.

**Expected C1 improvement**: +1-3 points. Prevents mipmap bleeding and reduces visible seam artifacts in real-time renderers.


### 3.7 Texture-Space Bilateral Denoising (RANK 7)

**Problem**: The trilinear interpolation from the sparse voxel grid introduces high-frequency noise in the texture, especially at voxel cell boundaries (where the interpolation "switches" between different voxel neighborhoods). This noise manifests as subtle speckle patterns in the rendered output.

**Solution**: Apply a bilateral filter to the base_color texture in UV space. The bilateral filter smooths spatially-close texels with similar colors while preserving sharp color edges (e.g., between a red shoe and white sole).

```python
def texture_bilateral_denoise(base_color, mask, d=7, sigma_color=25, sigma_space=5):
    """
    Bilateral denoising of base_color texture.
    Preserves edges while removing voxel-grid-induced noise.
    """
    # Only denoise valid texels
    result = base_color.copy()
    denoised = cv2.bilateralFilter(base_color, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    result[mask] = denoised[mask]
    return result
```

**Caution**: sigma_color must be tuned carefully. Too high (>40) blurs legitimate texture detail. Too low (<15) has no visible effect. Start with sigma_color=25.

**Where to apply**: In postprocess.py, after grey recovery and saturation boost, before seam smoothing.

**Expected C1 improvement**: +1-2 points. Reduces speckle/noise that the C1 crack detector and gradient harshness metrics penalize.
**C3 risk**: -1 point possible if sigma_color is too high (detail loss).


### 3.8 UV Island Stitching via Shared Vertex Colors (RANK 8)

**Problem**: UV seams are visible because the same 3D vertex can have different UV coordinates (and thus different sampled colors) on different UV islands. Even with seam smoothing, the colors may differ.

**Solution**: For each vertex shared by multiple UV islands, compute its color as the average across all islands. Then, during texture baking, use this averaged color at seam vertices to enforce consistency.

This requires modifying the baking pass to identify seam vertices (vertices that appear multiple times in the UV-unwrapped mesh with different UV coordinates) and force their colors to match.

**Implementation complexity**: Medium-high. Requires access to the vmaps (vertex maps) from UV unwrapping to identify which vertices are duplicated. The vmaps are already computed (`out_vmaps`).

**Where to apply**: In postprocess.py, after texture baking and before the final mesh construction.

**Expected C1 improvement**: +2-3 points. Eliminates the root cause of seam color discontinuities.


---

## 4. Expected Impact Summary

### Cumulative Impact (if all methods are applied)

| Method | C1 Coherence | C2 Vitality | C3 Detail | Risk | Time |
|--------|-------------|-------------|-----------|------|------|
| 1. Push-Pull Padding | +3-5 | +1-2 | 0 | Very Low | 4-6h |
| 2. Chroma-Aware Sampling | +1-2 | +3-5 | 0 | Low-Med | 4-8h |
| 3. UV-Aware Seam Smooth | +2-4 | 0 | 0 | Low | 3-5h |
| 4. Saturation Boost | 0 | +2-3 | 0 | Low | 1-2h |
| 5. Enhanced Refiner | +3-5 | +2-4 | +2-3 | Medium | 6-10h |
| 6. Dilation Padding | +1-3 | 0 | 0 | Low | 2-3h |
| 7. Bilateral Denoise | +1-2 | 0 | -1 | Low | 2-3h |
| 8. UV Stitching | +2-3 | 0 | 0 | Medium | 8-12h |
| **Total (optimistic)** | **+13-24** | **+8-14** | **+1-2** | | **30-49h** |
| **Total (realistic, w/ diminishing returns)** | **+6-10** | **+5-8** | **+1-2** | | **20-30h** |

### Recommended Phase 1 (Day 1, highest ROI)

1. **Saturation Boost** (Rank 4): 1-2 hours, immediate C2 improvement, zero risk
2. **Push-Pull Padding** (Rank 1): 4-6 hours, major C1 improvement, very low risk
3. **Chroma-Aware Sampling** (Rank 2, Solution A only): 2-3 hours, major C2 improvement

**Day 1 expected total**: C1 +4-7pts, C2 +6-10pts

### Recommended Phase 2 (Day 2, medium ROI)

4. **UV-Aware Seam Smoothing** (Rank 3): 3-5 hours
5. **Increased Dilation** (Rank 6): 2-3 hours
6. **Bilateral Denoise** (Rank 7): 2-3 hours

**Day 2 expected total**: C1 +4-9pts cumulative

### Future Phase (requires more testing)

7. **Enhanced TextureRefiner** (Rank 5): 6-10 hours, high impact but needs camera calibration work
8. **UV Stitching** (Rank 8): 8-12 hours, cleanest solution but most complex

---

## 5. References

### Papers Cited

1. **Interpolation-Aware Padding for 3D Sparse CNNs** (ICCV 2021) -- Yang et al. Empty voxel padding for correct trilinear interpolation in sparse convolution networks. Directly applicable to TRELLIS.2's grid_sample_3d boundary behavior.
   https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Interpolation-Aware_Padding_for_3D_Sparse_Convolutional_Neural_Networks_ICCV_2021_paper.pdf

2. **GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement** (ICLR 2025) -- Agrawal et al. Per-instance texture refinement via differentiable rendering in 4 seconds. PSNR +1.12 dB. Architecture-agnostic.
   https://arxiv.org/abs/2406.05649

3. **SeamCrafter: Enhancing Mesh Seam Generation via Reinforcement Learning** (2025) -- Li et al. GPT-style seam generation conditioned on point clouds, optimized with DPO for low-distortion UV unwrapping.
   https://arxiv.org/abs/2509.20725

4. **Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models** (CVPR 2024) -- UV inpainting + UVHD diffusion for illumination-free texture completion.
   https://github.com/opentexture/paint3d

5. **Texture Inpainting for Photogrammetric Models** (CGF 2023) -- Maggiordomo et al. Specialized texture inpainting for 3D scanned models, addressing UV seam and hole artifacts.
   https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14735

6. **GeoScaler: Geometry and Rendering-Aware Downsampling of 3D Mesh Textures** (2023) -- Geometry-aware texture filtering that accounts for UV distortion during downsampling.
   https://arxiv.org/html/2311.16581v2

7. **DiffTex: Differentiable Texturing for Architectural Proxy Models** (2025) -- Multi-view differentiable texture optimization with SSIM + LPIPS evaluation.
   https://arxiv.org/html/2509.23336

8. **Push-Pull Texture Padding** -- Standard computer graphics technique for UV island padding. Substance 3D Painter uses 8-pixel default padding. See:
   https://helpx.adobe.com/substance-3d-painter/technical-support/workflow-issues/export-issues/texture-dilation-or-padding.html

9. **Image Margin (UV Dilation)** -- Open-source implementation of UV island dilation/padding.
   https://github.com/KirilStrezikozin/image_margin

### Industry Standards

- **Adobe Substance Painter**: Default 8-pixel texture dilation/padding
- **TurboSquid CheckMate Pro v2**: Minimum 4-pixel UV padding, 2048+ texture resolution, no visible seam artifacts
- **Game Industry Standard**: UV padding must survive 2x mipmap downsampling (minimum 4 pixels at highest resolution)

### Existing Infrastructure

- **nvdiffrast**: Already installed and used in postprocess.py and texture_refiner.py
- **OpenCV (cv2)**: Already used for bilateral filter, blur, inpaint, morphological ops
- **trimesh**: Already used for mesh construction and material handling
- **LPIPS**: Already installed, used in TextureRefiner
- **DreamSim**: Already installed in container (`pip install dreamsim`)
- **pyiqa**: Already installed, provides MUSIQ, TOPIQ-NR, CLIPIQA+ metrics
