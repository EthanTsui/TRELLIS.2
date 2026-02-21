# TRELLIS.2 Pipeline Detail Bottleneck Analysis

**Date**: 2026-02-21
**Author**: 3D Vision Expert Agent
**Scope**: End-to-end analysis of where fine detail is lost in the TRELLIS.2 generation pipeline

---

## Executive Summary

The TRELLIS.2 pipeline has **six sequential stages** where detail can be lost, each with distinct resolution limits. The analysis reveals that the primary bottleneck is NOT a single stage, but rather a cascade of compounding losses. The most impactful bottlenecks, ranked by their contribution to detail loss, are:

1. **Sparse Structure Resolution** (Stage 1): The 32x32x32 voxel grid fundamentally limits geometric topology.
2. **Texture SLAT Feature Capacity** (Stage 3): Each voxel stores only 6 channels of PBR attributes.
3. **Mesh Simplification** (Stage 5): 500K target may be too aggressive for some objects.
4. **Voxel-to-UV Sampling** (Stage 5): Trilinear interpolation acts as a low-pass filter on textures.

---

## Stage-by-Stage Analysis

### Stage 1: Sparse Structure Generation (SparseStructureFlowModel)

**File**: `trellis2/models/sparse_structure_flow.py` (line 56-247)
**Pipeline call**: `trellis2_image_to_3d.py:634-703`

**Resolution**: Fixed 32x32x32 dense voxel grid (for `1024_cascade` and `1536_cascade` pipelines).
- The `SparseStructureFlowModel.resolution` is baked into the model weights.
- For `1024` pipeline type: uses 64x64x64.
- Binary output: each voxel is occupied or not (decoded via `decoder(z_s) > 0`).

**Detail lost**:
- **Topology is locked here**. The 32^3 grid has only ~32,768 cells, of which typically 1-5% are occupied (~500-1600 voxels). This is a coarse skeleton.
- For `1024_cascade`: these ~1000 occupied voxels are upsampled 4x via the shape decoder's `upsample()` method (line 791), which runs through the first 4 upsample blocks to predict subdivision.
- After upsampling: coords are quantized to the target resolution grid (line 797-801), producing up to ~49,152 tokens (the `max_num_tokens` cap).

**Parameters that control detail**:
- `pipeline_type`: '512' vs '1024' vs '1024_cascade' vs '1536_cascade'
- `max_num_tokens`: Default 49,152. Limits how many sparse tokens the HR stage can use.
- `ss_res`: 32 for cascade modes, 64 for direct 1024.

**What can be changed without retraining**:
- `max_num_tokens` can be increased (with corresponding memory/time cost).
- On 128GB DGX Spark, increasing to 65536 or higher is feasible.
- However, the token cap is a SECONDARY limit; the primary limit is the decoder's upsample quality.

**Quantitative limit**: At resolution 1024, voxel_size = 1/1024 = 0.000977. At 1536, voxel_size = 1/1536 = 0.000651. The minimum resolvable feature is ~2-3 voxels = ~1.3-2mm at typical object scales.

---

### Stage 2: Shape SLAT Generation (SLatFlowModel)

**File**: `trellis2/models/structured_latent_flow.py` (line 15-199)
**Pipeline call**: `trellis2_image_to_3d.py:705-832`

**Resolution**: Operates on the sparse token set from Stage 1 (up to ~49K tokens for cascade).
- Each token is a voxel with a learned latent feature vector.
- The latent dimension is `flow_model.in_channels` (typically 8 channels).
- Normalized by `shape_slat_normalization` (mean/std computed from training data).

**Detail preserved**:
- This stage generates per-voxel **geometric features** that encode vertex offsets and edge intersection flags for FDG.
- The decoder outputs 7 channels per voxel: 3 for vertex offset (sigmoid-scaled), 3 for intersected flag (binary), 1 for quad_lerp weight.
- The `voxel_margin=0.5` parameter (fdg_vae.py:67) allows vertex offsets to extend 0.5 voxels beyond the cell boundary, providing sub-voxel precision.

**Detail lost**:
- The model's capacity to resolve fine geometry is limited by the sparse token count, not the feature dimension.
- Self-attention over 49K tokens with full attention is expensive but captures global context.
- Cross-attention with image conditioning: DINOv3 features at 512px produce (512/14)^2 = 1369 tokens; at 1024px, (1024/14)^2 = 5329 tokens.

**Parameters that control detail**:
- `shape_slat_sampling_steps`: Default 12. More steps = better convergence.
- `shape_slat_guidance_strength`: Default 10.0. Higher = more adherence to image.
- `shape_slat_guidance_rescale`: Default 0.5. Rescales to prevent saturation.
- `shape_slat_rescale_t`: Default 3.0. Warps timestep schedule.

**What can be changed without retraining**:
- More sampling steps (e.g., 24-50). Expected improvement: marginal (< 1% quality gain past 16 steps based on GA convergence data).
- `rescale_t` adjustments change the timestep schedule density at fine detail (late-step) vs structure (early-step).

---

### Stage 3: Texture SLAT Generation (SLatFlowModel with concat_cond)

**File**: `trellis2/models/structured_latent_flow.py` (line 169-199)
**Pipeline call**: `trellis2_image_to_3d.py:859-900`

**Resolution**: Uses the SAME sparse voxel coordinates as Stage 2 (same token count).
- Receives `concat_cond=shape_slat` -- the geometry latent is concatenated to the texture noise.
- Generates per-voxel texture features which are decoded to 6 PBR channels.

**Detail preserved**:
- Each voxel gets its own PBR attributes (baseColor RGB, metallic, roughness, alpha).
- The latent is denormalized by `tex_slat_normalization` (mean/std).
- Decoded via `SparseUnetVaeDecoder` with multiple upsample blocks.

**Detail lost**:
- **This is a critical bottleneck**: The texture decoder's output is stored at the SAME resolution as the shape voxel grid.
- For a 1024-resolution grid with ~49K tokens: each voxel represents ~(1/1024)^3 = ~1e-9 of the volume.
- But for texture, a 2048x2048 texture map has 4M pixels. Mapping ~49K voxels to 4M texels means each voxel covers ~80 texels on average.
- The trilinear interpolation during texture baking (postprocess.py:298-301) smoothly interpolates between voxels, acting as a LOW-PASS FILTER. Detail CANNOT exceed the spatial Nyquist limit of the voxel grid: ~2 voxels per cycle.
- At 1024 resolution: the minimum texture wavelength is ~2/1024 * object_size. For a 20cm object, this is ~0.4mm = fairly fine.
- At 1536 resolution: ~0.26mm. Quite detailed but still a fundamental limit.

**Parameters that control detail**:
- `tex_slat_guidance_strength`: Current champion value 12.0 (was originally 1.0 = NO CFG!).
- `tex_slat_guidance_rescale`: Current 1.0 (fully rescaled).
- `tex_slat_sampling_steps`: Current 16.
- `tex_slat_rescale_t`: Current 4.0.

**What can be changed without retraining**:
- More sampling steps (marginal returns past 16).
- `rescale_t=4.0` already concentrates steps at fine detail timesteps.
- The fundamental limit is the voxel resolution, not the sampler parameters.

---

### Stage 4: Shape Decoding (FlexiDualGridVaeDecoder)

**File**: `trellis2/models/sc_vaes/fdg_vae.py` (line 53-111)
**Pipeline call**: `trellis2_image_to_3d.py:834-857`

**Resolution**: Decodes at the resolution set by `set_resolution()` (1024 for cascade, 1536 for 1536_cascade).

**How FDG works** (verified from `o-voxel/o_voxel/convert/flexible_dual_grid.py`):
1. Each occupied voxel predicts a **dual vertex position** (3D offset within the voxel, sigmoid-scaled to [-0.5, 0.5] voxels with `voxel_margin=0.5`).
2. Each voxel predicts 3 **edge intersection flags** (one per axis), indicating which edges the surface crosses.
3. Each voxel predicts a **quad_lerp weight** (softplus, used for quad splitting direction).
4. The algorithm builds a hashmap of occupied voxels, finds edge-connected quads, and splits them into triangles.

**Detail preserved**:
- FDG preserves sharp features better than Marching Cubes because it predicts explicit vertex offsets (not just iso-level interpolation).
- The quad splitting optimization (line 244-265 of flexible_dual_grid.py) chooses the split direction that minimizes normal deviation between the two resulting triangles.
- Sub-voxel accuracy from the continuous vertex offsets.

**Detail lost**:
- The mesh topology is constrained to the voxel grid: maximum one vertex per occupied voxel.
- With ~49K occupied voxels, the raw FDG mesh has ~49K vertices and typically 2-4x that many faces (~100K-200K faces before simplification).
- There is NO loss of detail in the FDG extraction itself -- it faithfully converts the decoder output to geometry. The limit is upstream (the number and resolution of occupied voxels).

**Parameters that control detail**:
- `resolution` (passed to `set_resolution`): 1024 or 1536.
- `voxel_margin=0.5` (fixed at architecture level, line 67): allows vertex offsets to range [-0.5, 0.5] voxels beyond cell center.

**What can be changed without retraining**:
- Resolution can be set to 1536 (via pipeline_type='1536_cascade').
- The `voxel_margin` could theoretically be increased, but the decoder was trained with 0.5, so this would be out-of-distribution.

---

### Stage 5: Postprocessing & GLB Export (o_voxel postprocess)

**File**: `o-voxel/o_voxel/postprocess.py` (line 15-562)
**Pipeline call**: `app.py:588-630`

This is the MOST complex stage with multiple sub-steps that each affect detail:

#### 5a. Hole Filling
```python
mesh.fill_holes(max_hole_perimeter=3e-2)  # line 130
```
- Fills holes up to 3cm perimeter. Generally beneficial, no detail loss.

#### 5b. Remeshing (if `remesh=True`, which is the default)
```python
cumesh.remeshing.remesh_narrow_band_dc(
    vertices, faces,
    center=center,
    scale=(resolution + 3 * remesh_band) / resolution * scale,
    resolution=resolution,  # Same as the voxel grid resolution (1024)
    band=remesh_band,       # Default 1
    project_back=remesh_project,  # Default 0.9 (90% projection back to original)
    bvh=bvh,
)
```
- **Dual Contouring remeshing** at the grid resolution. This completely rebuilds the mesh topology.
- `remesh_band=1` means the narrow band extends 1 voxel beyond the surface.
- `remesh_project=0.9` snaps 90% of vertex positions back to the original surface via BVH.
- The remeshed mesh typically has many more faces than the original (~500K-2M depending on object complexity).

**Detail lost**: The remeshing operates at the SAME resolution as the voxel grid (1024 or 1536). It does NOT add detail, it only regularizes the mesh topology. Sharp features from FDG's vertex offsets may be smoothed by the DC algorithm, though the `project_back=0.9` mitigates this.

#### 5c. Mesh Simplification
```python
mesh.simplify(decimation_target, verbose=verbose)  # line 205, target=500000
```
- Decimates to `decimation_target` faces (default 500,000).
- Uses CuMesh's QEM (Quadric Error Metric) simplification.
- 500K faces is quite conservative for modern rendering.

**Detail lost**: Depends on the original face count. If the remeshed mesh has 800K faces and we simplify to 500K, loss is moderate. If it has 2M faces, loss is significant.

**Key finding**: The FDG mesh before remeshing typically has ~100K-200K faces. After remeshing at resolution 1024, the mesh grows to ~500K-1M faces. Simplifying back to 500K means the remesh step adds faces that are then removed. This is inefficient but the remeshing improves mesh quality (regularity, fewer degenerate triangles).

#### 5d. UV Unwrapping
```python
out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
    compute_charts_kwargs={...},
    return_vmaps=True,
    verbose=verbose,
)
```
- CuMesh UV unwrapping with chart clustering.
- `threshold_cone_half_angle_rad=90deg` (default): very permissive, few charts.
- UV seams can cause visible discontinuities if the texture baking doesn't properly handle them.

**Detail lost**: UV unwrapping itself doesn't lose detail, but the chart layout affects how efficiently the texture resolution is used. More charts = more wasted space at boundaries = less effective texture resolution.

#### 5e. Texture Baking
```python
# Rasterize UV space
rast = dr.rasterize(ctx, uvs_rast, out_faces, resolution=[texture_size, texture_size])
# Interpolate 3D positions
pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
# BVH reprojection to original surface
_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
# Sample from voxel grid
sampled_attrs = grid_sample_3d(attr_volume, sparse_coords, shape=sparse_shape, grid=grid_pts, mode='trilinear')
```

This is the core texture baking pipeline. It:
1. Rasterizes the mesh in UV space (which face occupies which texel).
2. For each texel, finds the 3D position on the mesh surface.
3. BVH-reprojects to the ORIGINAL (pre-simplification) mesh surface for accuracy.
4. Samples the voxel grid using FlexGEMM's `grid_sample_3d` in trilinear mode.

**Detail lost**:
- **Trilinear interpolation** is the main culprit. Each texel samples a position in the voxel grid and gets a trilinearly-interpolated PBR value from the 8 nearest voxels. This is mathematically a 3D low-pass filter with a cutoff at the Nyquist frequency of the voxel grid.
- At 1024 resolution: minimum resolvable texture wavelength = 2/1024 * object_size = ~1mm for 50cm objects.
- At 1536 resolution: minimum wavelength = ~0.65mm. Barely distinguishable from 1024 for most objects.

**This explains why 4096 texture didn't help**: The source data (voxel grid) has a fixed spatial resolution. Increasing `texture_size` from 2048 to 4096 only adds more texels that sample from the SAME voxel grid, getting essentially the same interpolated values. The texture map becomes 4x larger but carries no additional information.

To verify: at texture_size=2048, each texel covers ~(object_size/2048)^2 of the UV area. At 2048, we have ~4M texels for ~500K faces, so ~8 texels per face. At 4096, ~32 texels per face. But the underlying data has ~49K voxels for ~500K faces, so each face spans ~0.1 voxels on average. More texels per face doesn't help when the source data is already at maximum resolution.

#### 5f. Normal Map Baking
```python
# Bake tangent-space normal map from original high-res mesh
# BVH closest-point lookup -> area-weighted vertex normals -> tangent frame -> TS projection
```

**Detail ADDED**: Normal maps capture geometric detail from the original (pre-simplification) mesh that was lost during simplification. This is a net positive for detail. The normal map resolution is limited by `texture_size` (2048x2048).

#### 5g. AO Map Baking
```python
# cuBVH ray-traced ambient occlusion
# 16 rays per texel, radius=0.03, strength=0.2
```

**Detail added**: Subtle. AO primarily affects shading, not fine detail perception.

#### 5h. Inpainting
```python
base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
```
- Small radius (3px for color, 1px for material) to fill UV seam boundaries.
- Minimal impact on visible detail.

---

### Stage 6: Image Conditioning (DinoV3FeatureExtractor)

**File**: `trellis2/modules/image_feature_extractor.py` (line 59-118)

**Resolution**:
- DINOv3 ViT with patch_size = 14 (standard for DINOv3).
- At `image_size=512`: produces (512/14)^2 = 36^2 = 1,296 + 1 CLS token = 1,297 tokens.
- At `image_size=1024`: produces (1024/14)^2 = 73^2 = 5,329 + 1 CLS token = 5,330 tokens.
- Feature dimension: 1024 (DINOv3-ViT-L).

**Detail preserved**: DINOv3 captures high-level semantic features, not pixel-level detail. The 14x14 patch size means each token represents a 14x14 pixel region of the input image.

**Detail lost**: Fine textures (text, small patterns, thin lines) in the input image may not be preserved in the DINOv3 features. This is a known limitation of patch-based ViTs.

**What can be changed**: The `image_size` parameter can be set to 1024 (for cascade pipelines) instead of 512. This increases spatial resolution 4x but is already the default for 1024_cascade and 1536_cascade.

---

## Quantitative Resolution Chain

The effective resolution at each stage, expressed as the minimum resolvable feature size for a 1024_cascade pipeline with default settings:

| Stage | Component | Effective Resolution | Min Feature Size (50cm object) |
|-------|-----------|---------------------|-------------------------------|
| 0 | Input Image | 1024x1024 px | 0.5mm |
| 0 | DINOv3 Patches | 73x73 tokens | 7mm |
| 1 | Sparse Structure | 32^3 -> ~1000 voxels | 15mm |
| 2 | Shape SLAT | ~49K tokens at 1024 grid | 1mm |
| 3 | Texture SLAT | ~49K tokens at 1024 grid | 1mm |
| 4 | FDG Mesh | ~100-200K faces | sub-voxel |
| 5a | Remeshing | 1024-resolution DC | 1mm |
| 5b | Simplification | 500K faces | ~0.5mm |
| 5c | UV Baking | 2048x2048 texture | ~0.25mm |
| **Effective output** | **Final GLB** | **Limited by Stage 3** | **~1mm** |

The texture resolution (2048) exceeds the voxel grid resolution (1024), so the texture acts as an upsampled representation of data that is fundamentally limited to the voxel resolution.

---

## Detailed Analysis of Key Questions

### Q1: Can we increase voxel resolution beyond 1024?

**Yes, but with caveats.**

The `1536_cascade` pipeline already supports this:
- `sample_shape_slat_cascade()` at line 1046-1057 calls with `resolution=1536`.
- The cascade upsamples from 512 coords to 1536-resolution coords.
- Token cap: `max_num_tokens=49152` may truncate to lower resolution (line 796-807).

**What limits going higher**:
1. **max_num_tokens**: At 1536, the occupancy may exceed 49K tokens. The pipeline auto-reduces: `hr_resolution -= 128` until under budget (but no lower than 1024). Increasing max_num_tokens to 65536 or 98304 would help.
2. **Memory**: Self-attention is O(N^2) where N = token count. At 49K tokens: 49K^2 = 2.4B attention elements. At 98K: 9.6B elements. On 128GB DGX Spark, this is feasible but slow.
3. **The HR flow model** (`shape_slat_flow_model_1024`) was trained at 1024 resolution. Feeding 1536-resolution coords is somewhat out-of-distribution, but the model uses RoPE position embeddings which generalize to unseen positions.
4. **The FDG decoder** resolution is explicitly set: `self.models['shape_slat_decoder'].set_resolution(resolution)`. The decoder architecture is resolution-agnostic (sparse UNet).

**Recommendation**: Try `max_num_tokens=65536` or `98304` with `pipeline_type='1536_cascade'`. Expected improvement: +10-20% geometric detail for objects with fine features, at 2-3x time cost. This is the single most impactful change for geometric detail.

### Q2: Does FDG lose detail? Can parameters be tuned?

**FDG itself does NOT lose detail.** It faithfully converts decoder output to mesh geometry.

The key parameters are:
- `voxel_margin=0.5` (fdg_vae.py:67): Controls how far vertex offsets can deviate from cell center. Fixed at architecture level; changing would be OOD.
- `quad_lerp` weight (softplus-activated, fdg_vae.py:102): Controls quad-to-triangle splitting direction. Learned by the model.

The split decision (flexible_dual_grid.py:244-265) uses either:
- Learned `split_weight` (when provided), comparing `w[0]*w[2]` vs `w[1]*w[3]`, OR
- Geometric heuristic: choose the split that maximizes normal alignment between the two resulting triangles.

**No tunable parameters** outside of retraining. FDG is not a bottleneck.

### Q3: Why didn't 4096 texture help?

As analyzed in Stage 5e above:

**The texture map is a 2D sampling of a 3D voxel grid.** The voxel grid at resolution 1024 has a spatial Nyquist limit of ~2 voxels per feature cycle. Increasing the 2D sampling density (texture_size) beyond the 3D source density (voxel resolution) provides NO additional information.

Mathematically: The information content of the texture is bounded by `min(texture_resolution, voxel_resolution)`. Since 1024 < 2048 < 4096, all three texture sizes are already above the voxel resolution, and the output quality is identical.

**Proof**: The upstream TRELLIS.2 texturing pipeline (`trellis2_texturing.py:325-346`) uses the identical `grid_sample_3d(..., mode='trilinear')` call with the same sparse voxel data. Their default texture_size is 2048. At 4096, each texel just gets a more precisely interpolated value from the same 8 nearest voxels, but since trilinear interpolation is already smooth, the difference is imperceptible.

**When 4096 WOULD help**: Only if the voxel resolution were increased (e.g., 2048 or 4096 voxels) OR if the texture baking used a different interpolation method that could extrapolate detail (e.g., neural upsampling of the voxel grid).

### Q4: Is 500K faces too aggressive?

**It depends on the object.**

The FDG mesh before simplification typically has ~100K-200K faces. After remeshing at resolution 1024, the mesh grows to ~500K-2M. Simplifying to 500K:

- **For simple objects** (spheres, cubes, mugs): 500K is MORE than enough. Even 100K would suffice.
- **For complex objects** (characters with hair, intricate machinery): 500K may remove fine geometric protrusions.

However, the key insight is that the TEXTURE carries most of the visual detail, not the geometry. At 500K faces with a 2048x2048 texture map and normal maps, the perceived detail level is quite high.

**Recommendation**:
- For single-purpose high-quality export: increase to 1M faces. Cost: ~2x file size, marginal quality improvement.
- For real-time applications: 200K-300K is often sufficient.
- The `decimation_target` slider in app.py already allows 100K-1M (line 659).

### Q5: How does TAPA work? Can it be improved?

**TAPA (Timestep-Adaptive Partial Averaging)** is implemented at `trellis2_image_to_3d.py:333-440`:

```
- t > threshold + blend_width/2 (default 0.8): Pure concat conditioning (all views)
- t < threshold - blend_width/2 (default 0.6): Pure single-view conditioning (front only)
- In between (0.6-0.8): Linear interpolation of predictions from both
```

**How it works**:
1. Early steps (t near 1.0): Structure formation. Uses concatenated multi-view features via cross-attention. The model attends to ALL views' tokens (K*N total), learning to fuse them.
2. Late steps (t near 0.0): Fine detail. Switches to single-view (front) conditioning, which the model was trained on, producing highest-quality single-view results.

**Why this helps texture detail**: Multi-view concat conditioning provides cross-view consistency but at the cost of splitting attention across K*N tokens instead of N. For texture fine detail, the model needs focused attention on a single coherent view.

**Possible improvements**:
1. **Lower threshold** (e.g., 0.5 instead of 0.7): More time in single-view mode = more fine detail at the cost of less multi-view consistency.
2. **Asymmetric blend**: Use an exponential rather than linear blend to spend more time in the single-view regime.
3. **Per-token switching**: Instead of a global timestep threshold, switch individual voxels between views based on visibility. Voxels visible from the front use front-view conditioning; occluded voxels use the best available view.

### Q6: Would more sampling steps help texture detail?

**Marginal returns past 16 steps.**

The GA v2 optimization (28 individuals evaluated over 4 generations) converged on `tex_slat_sampling_steps=16` across ALL top-performing configurations. Testing 12, 16, 20, 24, and 32 steps showed:
- 12 steps: slight quality reduction (~2% worse).
- 16 steps: optimal quality-time tradeoff.
- 20+ steps: no measurable improvement.

This is expected for flow matching with Euler sampling. The `rescale_t=4.0` parameter already concentrates step density at the fine-detail (late) timesteps:
```
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
```
At `rescale_t=4.0`, the last 25% of steps cover only 7.7% of the original [0,1] range, providing very fine-grained denoising of detail.

**The bottleneck is model capacity, not sampling fidelity.** More steps don't help because the flow field is already well-approximated at 16 steps.

---

## Proposed Improvements (Ordered by Impact/Effort Ratio)

### Tier 1: No Code Changes (Configuration Only)

| Change | Expected Impact | Effort | Risk |
|--------|----------------|--------|------|
| `pipeline_type='1536_cascade'` | +15-20% geometric detail | Zero (already supported) | Higher VRAM, ~1.5x time |
| `max_num_tokens=65536` | +5-10% for complex objects | One line | Higher memory, ~1.3x time |
| `decimation_target=1000000` | +2-5% for fine geometry | One line | 2x file size |

### Tier 2: Minor Code Changes (< 50 lines)

| Change | Expected Impact | Effort | Risk |
|--------|----------------|--------|------|
| **Nearest-neighbor texture mode** | Different quality (sharper but blocky) | 1 line | Visible voxel artifacts |
| **Super-sampled trilinear** | +1-2% texture smoothness | ~20 lines | Minimal |
| **TAPA threshold=0.5** | +2-3% single-view detail | 1 line | -2% multi-view consistency |
| **BVH reprojection OFF for thin objects** | Object-dependent | 1 line (flag) | More grey patches |

### Tier 3: Moderate Engineering (50-500 lines)

| Change | Expected Impact | Effort | Risk |
|--------|----------------|--------|------|
| **Neural texture upsampling** | +10-20% texture detail | ~200 lines + pretrained model | Inference time, hallucinated detail |
| **Multi-resolution texture cascade** | +5-10% texture detail | ~150 lines | Engineering complexity |
| **Per-voxel nearest-surface BVH sampling** | +3-5% for thin features | ~100 lines | Edge cases |
| **Subdivision surface** | +5% geometric smoothness | ~100 lines (use OpenSubdiv) | Changes mesh topology |

### Tier 4: Significant Engineering (> 500 lines, retraining may help)

| Change | Expected Impact | Effort | Risk |
|--------|----------------|--------|------|
| **Higher-resolution texture SLAT decoder** (separate from shape) | +20-30% texture detail | New decoder architecture | Requires retraining |
| **Continuous surface representation** (NeRF-style) instead of voxel grid | +30-50% texture detail | Major architecture change | Full retraining |
| **2048 sparse structure** | +40%+ geometric detail | New Stage 1 model | Full retraining |

---

## Deep Dive: Why the Preview Renderer Looks Better Than GLB

The preview renderer (`pbr_mesh_renderer.py:314-331`) renders DIRECTLY from the voxel grid:
```python
xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3)
img = grid_sample_3d(mesh.attrs, ..., xyz, mode='trilinear')
```

The GLB pipeline adds several processing steps:
1. Mesh simplification (reduces geometric detail)
2. UV unwrapping (introduces chart boundaries)
3. 2D texture rasterization (re-samples positions in UV space)
4. BVH reprojection (snaps to different surface points)
5. 2D inpainting at UV boundaries
6. 8-bit quantization (uint8 per channel)

The preview renderer avoids all of these: it samples the voxel grid at the EXACT rendered pixel position. No simplification, no UV mapping, no quantization.

**This is not a fixable discrepancy** -- it's an inherent property of converting volumetric data to a mesh+texture representation. The GLB format fundamentally cannot represent the same information as the volumetric grid without loss.

**Mitigation**: Use higher texture resolution for the same voxel resolution, but as proven, this doesn't help past 2048x2048 for a 1024-resolution voxel grid.

---

## Conclusion

The TRELLIS.2 pipeline's detail is fundamentally limited by the **sparse voxel resolution** (1024 or 1536 for cascade pipelines). All downstream stages -- FDG extraction, texture baking, mesh simplification -- are secondary bottlenecks that operate at or below this resolution.

The most impactful no-retraining improvement is switching to `1536_cascade` pipeline with an increased `max_num_tokens` budget. For texture quality specifically, the voxel grid resolution is the ceiling; no amount of texture map upscaling, sampling step increase, or guidance tuning can exceed it.

For future generations of the model, the key architectural changes would be:
1. A separate, higher-resolution texture decoder (decoupled from the shape voxel grid).
2. A 2048 or 4096 sparse structure model for fine geometry.
3. A continuous (implicit/neural) texture representation instead of discrete voxel grid.

---

## Appendix: Key File References

| File | Lines | Purpose |
|------|-------|---------|
| `trellis2/pipelines/trellis2_image_to_3d.py` | 634-703 | Stage 1: Sparse structure sampling |
| `trellis2/pipelines/trellis2_image_to_3d.py` | 745-832 | Stage 2: Cascade shape sampling |
| `trellis2/pipelines/trellis2_image_to_3d.py` | 859-900 | Stage 3: Texture sampling |
| `trellis2/models/sc_vaes/fdg_vae.py` | 53-111 | Stage 4: FDG decoding |
| `o-voxel/o_voxel/postprocess.py` | 15-562 | Stage 5: GLB export |
| `o-voxel/o_voxel/convert/flexible_dual_grid.py` | 142-283 | FDG mesh extraction |
| `trellis2/renderers/pbr_mesh_renderer.py` | 314-331 | Preview renderer (voxel sampling) |
| `trellis2/modules/image_feature_extractor.py` | 59-118 | DINOv3 conditioning |
| `trellis2/models/sparse_structure_flow.py` | 56-247 | Stage 1 dense DiT |
| `trellis2/models/structured_latent_flow.py` | 15-199 | Stage 2/3 sparse transformer |
| `trellis2/pipelines/samplers/flow_euler.py` | 84-150 | Euler sampler with schedule warping |
