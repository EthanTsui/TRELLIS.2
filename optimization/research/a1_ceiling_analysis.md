# A1 Silhouette Ceiling Analysis: Breaking Through 83.8

## Date: 2026-02-23
## Author: 3D Vision Expert Agent

---

## 1. Problem Statement

A1_silhouette measures scale-invariant Dice coefficient between the rendered 3D model's
silhouette (from the best-matching view among 8 rendered views) and the input image's
alpha mask. Both masks are cropped to their bounding boxes and resized to 256x256 before
comparison, removing scale/framing differences.

**Current scores (10-image 1536_cascade test):**

| Image | A1 | Notes |
|-------|------|-------|
| T.png | 87.9 | Simple shape, clean silhouette |
| f351... | 89.0 | Compact object |
| 5a6c... | 88.3 | Compact object |
| cd3c... | 88.2 | Compact object |
| ee8e... | 86.7 | Moderate complexity |
| 7d58... | 84.5 | Moderate complexity |
| d39c... | 84.3 | Moderate complexity |
| 454e... | 82.7 | Complex shape |
| e4d6... | 78.7 | Complex shape, concavities |
| 0a34... | 67.5 | **Outlier** -- complex topology |
| **Mean** | **83.8** | **std = 6.2** |

Key observation: the distribution is bimodal. Compact/simple objects score 86-89, while
complex objects with thin features, concavities, or unusual topology score 67-84. The
outlier at 67.5 drags the mean significantly.

---

## 2. Architecture Constraints Analysis

### 2.1 Stage 1: Sparse Structure Resolution

**Flow Model Architecture** (`sparse_structure_flow.py`):
- `SparseStructureFlowModel` is a dense 3D DiT operating on a fixed-resolution cube
- `self.resolution` is baked into the pretrained weights (RoPE position embeddings, layer shapes)
- **Verified from HuggingFace config** (`ss_flow_img_dit_1_3B_64_bf16.json`):
  - `resolution = 16` (16^3 = 4096 tokens in latent space)
  - `in_channels = 8`, `out_channels = 8`
  - `model_channels = 1536`, `num_blocks = 30`, `num_heads = 12`
  - 1.3B parameters, bfloat16
- Full self-attention over 4096 tokens (manageable; 16384 tokens at R=32 would be ~4x cost)

**Sparse Structure Decoder** (`sparse_structure_vae.py` lines 210-307):
- **Verified from HuggingFace config** (`ss_dec_conv3d_16l8_fp16.json`):
  - `channels = [512, 128, 32]` (3 channel groups)
  - `latent_channels = 8`, `out_channels = 1`
  - `num_res_blocks = 2`, `num_res_blocks_middle = 2`
- Number of upsample blocks: `len(channels) - 1 = 2` (each 2x spatial)
- Input: 16^3 latent -> after 2 upsamples: 16*4 = **64^3 output**
- Output is binary occupancy: `decoded = decoder(z_s) > threshold`

**Resolution selection** (`trellis2_image_to_3d.py` lines 1114, 1253, 1508):
```python
ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
```

**Critical finding**: The flow model generates at 16^3 latent resolution, and the decoder
upsamples to 64^3 occupancy. For cascade pipelines (`1024_cascade`, `1536_cascade`), this
64^3 output is downsampled to 32^3 via `max_pool3d(decoded, 2, 2, 0) > 0.5` (line 706).
This is a deliberate design choice: each occupied 32^3 voxel expands to many tokens in the
512/1024/1536 SLAT grid, so using 64^3 directly would produce ~4-8x more SLAT tokens.

The max_pool is a LOSSY operation: it takes the logical OR of 2^3=8 decoder voxels per
output voxel. This means thin features at the 64^3 scale that occupy only 1-3 of the 8
sub-voxels are PRESERVED (any one triggers the output voxel). But it also means that
voids/concavities smaller than 2 voxels (at 64^3 scale) are FILLED IN by the OR operation.

**Can we increase Stage 1 resolution beyond 32^3?**

**Answer: YES, trivially -- use `pipeline_type='1024'` which sets `ss_res=64`.**

But there is a catch: with `pipeline_type='1024'`, the SLAT flow model operates at 1024
resolution directly (no cascade). This means:
- More tokens (up to 4x more occupied voxels at 64^3 vs 32^3)
- No cascade upsampling benefit (1536 resolution impossible)
- The 1024 model handles more tokens but at lower SLAT resolution

The real question is: can we get 64^3 sparse structure WITH 1536 cascade? This would
require modifying the cascade pipeline to accept 64^3 coords instead of 32^3.

### 2.2 Theoretical Maximum A1 at 32^3

A 32^3 voxel grid discretizes the [-0.5, 0.5]^3 canonical space into voxels of size
1/32 = 0.03125 per side. When projected to 2D at the canonical camera (r=2.0, fov=40deg):

- Object diameter: ~1.0 (normalized to [-0.5, 0.5])
- At r=2.0, fov=40deg, the object subtends ~28% of the image width
- On a 256x256 silhouette comparison grid: object is ~72 pixels across
- Each voxel projects to ~2.25 pixels (72/32)
- Boundary error: ~1 voxel = ~2.25 pixels on each edge

**Silhouette Dice impact of boundary discretization:**

For a sphere of radius r=0.4 in the canonical space:
- True area (projected): pi * (0.4 * 72/1.0)^2 = pi * 28.8^2 ~ 2607 pixels
- Boundary perimeter: 2*pi*28.8 ~ 181 pixels
- Each boundary pixel has ~50% chance of being wrong (1 voxel jitter)
- False positives + false negatives: ~181 pixels
- Dice = 2*intersection / (pred + gt) ~ 2*(2607 - 90) / (2*2607) ~ 96.5%

For more complex shapes with 2-3x longer boundary (concavities, protrusions):
- Boundary: ~360-540 pixels
- Dice = 2*(2607 - 180) / (2*2607) ~ 93.1%

For shapes with thin features (crown filigree, antlers):
- Thin features <1 voxel wide are completely lost
- Could drop Dice to 80-85% depending on feature prominence

**Theoretical ceiling at 32^3: ~93-96% for compact objects, ~80-90% for complex ones.**

The data matches: compact objects score 86-89, which is ~5-8 pts below the theoretical
maximum (accounting for model imperfection, not just discretization).

### 2.3 Theoretical Maximum A1 at 64^3

At 64^3, each voxel projects to ~1.125 pixels. The boundary error halves:
- Compact shapes: Dice ~ 98%
- Complex shapes: Dice ~ 92-95%
- Thin features: still limited (1 voxel = 1.125 px), but 2x better than 32^3

**Theoretical ceiling at 64^3: ~95-98% for compact, ~90-95% for complex.**

### 2.4 Why 32^3 is Used for Cascade

Looking at the cascade code (`sample_shape_slat_cascade`, line 815-831):

```python
hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
# ...
quant_coords = ... ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int()
coords = quant_coords.unique(dim=0)
num_tokens = coords.shape[0]
if num_tokens < max_num_tokens or hr_resolution <= 1024:
    break
hr_resolution -= 128
```

The cascade first generates shape at 512 (LR) resolution, then upsamples the decoded
shape's coordinates to the HR resolution (1024 or 1536). Each occupied 32^3 voxel expands
into many HR tokens via the decoder's upsample path.

Starting from 64^3 instead of 32^3 would produce ~4x more tokens (since 2^3=8x more
voxels, but occupancy rate is similar so ~4x more occupied voxels). With max_num_tokens=65536,
this could overflow.

**However**: on 128GB GPU, we can increase max_num_tokens to ~130K. This makes 64^3
cascading feasible if we accept ~2x longer generation time.

---

## 3. Per-Image Failure Mode Analysis

### 3.1 Outlier: 0a34... (A1 = 67.5)

This is the worst performer by a large margin (16 pts below mean). Without seeing the
image, based on the score pattern:
- A2_color_dist = 47.1 (also very low) -- color/appearance is also wrong
- C2_color_vitality = 100 (colors are vivid, just wrong)
- This suggests a **fundamental shape misinterpretation**: the model generated a
  plausible but incorrect 3D shape for this input

**Root cause**: Stage 1 (sparse structure) generated wrong topology. Stages 2-3 cannot
fix this. The single front-view conditioning is inherently ambiguous for this image.

### 3.2 Low-mid performers: e4d6... (78.7), d39c... (84.3)

These score 5-15 pts below the top cluster. The likely cause is:
- Concavities or recesses that are visible in the silhouette but lost in 32^3 voxelization
- Back-side geometry that differs significantly from what the model hallucinated
- The best-matching view among 8 rendered views may not perfectly align with the input

---

## 4. Concrete Interventions

### Intervention A: 64^3 Cascade Pipeline (HIGH IMPACT, MODERATE EFFORT)

**Concept**: Skip the max_pool that downsamples the decoder's 64^3 output to 32^3, and
feed the full 64^3 coordinates into the cascade pipeline. This preserves fine boundary
detail from Stage 1 that is currently lost by the OR-based max_pool.

**Architecture recap** (verified from HuggingFace configs):
- Flow model: 16^3 latent (4096 tokens, `resolution=16`)
- Decoder: 16 -> 32 -> 64 (two learned 2x upsample blocks)
- Decoder output: [B, 1, 64, 64, 64] occupancy logits
- Current cascade: `max_pool3d(decoded, 2) > 0.5` -> 32^3

**Implementation**:

File: `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`

The simplest change: set `ss_res=64` for `1536_cascade`:

```python
# Line 1253/1508: Change cascade ss_res from 32 to 64
ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 64}[pipeline_type]
```

BUT this requires changes in `sample_shape_slat_cascade()` (line 769) because the LR
shape flow model (512) expects coordinates at a certain density. With 64^3 occupancy,
the number of occupied voxels is ~4-8x higher (more voxels at higher resolution).

**Approach: Two-resolution coordinate path**:
```python
# Stage 1: Decode at 64^3 (skip max_pool for cascade)
decoded_64 = decoder(z_s) > threshold  # [B, 1, 64, 64, 64]
coords_64 = argwhere(decoded_64)  # Full-resolution coords

# For LR shape model: downsample coords to 32^3 scale
coords_32 = (coords_64[:, 1:] // 2)  # Integer downsample
coords_32_unique = torch.unique(torch.cat([coords_64[:, :1], coords_32], dim=1), dim=0)
# Run LR model on 32^3 coords (same as current pipeline)
shape_slat_lr = sample_shape_slat(cond_512, model_512, coords_32_unique, ...)

# Upsample LR shape to HR coords
hr_coords = decoder.upsample(shape_slat_lr, upsample_times=4)
# Merge with original 64^3 coords for denser HR grid
quant_coords = quantize_to_hr_grid(hr_coords, coords_64, lr_res=512, hr_res=1536)
coords_hr = unique(quant_coords)

# Run HR model on merged coords
shape_slat_hr = sample_shape_slat(cond_1024, model_1024, coords_hr, ...)
```

**Why this helps A1**: The max_pool OR operation currently fills in small concavities
and smooths boundaries. At 64^3, each boundary voxel is 1/64 of the unit cube (0.016),
projecting to ~1.1 pixels on the 256x256 evaluation grid. At 32^3, each is 0.031,
projecting to ~2.2 pixels. The boundary quantization error halves.

**Expected impact**: +2-4 pts A1 on average, +3-6 pts on complex shapes with concavities
**Risk**: The LR model (512) was trained on 32^3-scale coords. Using downsampled 64^3
coords should produce identical coordinate distributions after quantization.
The HR model (1024) is a sparse transformer, agnostic to token count, so more tokens
from 64^3 are handled naturally (subject to max_num_tokens).
**Difficulty**: Medium (30-50 lines of code change, careful coordinate math)
**Time cost**: ~1.5-2x baseline (50-100% more tokens in LR stage, ~25% more in HR)
**Memory**: ~80-100 GB peak at max_num_tokens=98304 (feasible on 128GB DGX Spark)

### Intervention B: Occupancy Threshold Sweep per Image (LOW EFFORT, LOW-MODERATE IMPACT)

**Concept**: The occupancy threshold controls how aggressively voxels are activated.
`decoded = decoder(z_s) > occupancy_threshold`. Higher threshold = tighter silhouette,
fewer false-positive voxels, but misses thin features.

The current default is 0.0 (any positive decoder output is "occupied"). This is quite
permissive. For images where the model over-generates (fat silhouette), raising the
threshold can help.

**Implementation**: Already exposed via `ss_occupancy_threshold` parameter.

The issue is that the OPTIMAL threshold varies per image. A simple sweep:
```python
for thresh in [0.0, 0.05, 0.1, 0.15, 0.2]:
    coords = sample_sparse_structure(cond, ..., occupancy_threshold=thresh)
    quick_mesh = decode_shape_only(coords)
    dice = silhouette_dice(quick_mesh, ref_alpha)
    # keep best
```

This is essentially Best-of-N over thresholds, much cheaper than Best-of-N over seeds
since we only re-threshold the SAME decoder output (no re-running the flow model).

**Expected impact**: +1-3 pts A1 (reduces fat silhouette problem)
**Risk**: Very low (single parameter)
**Difficulty**: Easy (10-15 lines)
**Time cost**: +5-10 seconds (5 quick mesh decodes)

### Intervention C: Post-Generation Silhouette Deformation with Soft Dice Loss (MODERATE EFFORT, HIGH IMPACT)

**Concept**: The existing `SilhouetteCorrector` was tested but DEGRADED C3 (detail_richness)
from 81 to 69.9. Analysis shows this is because:

1. The corrector moves vertices to match silhouette, distorting the texture mapping
2. High Laplacian regularization (w=50) prevents large enough movements to fix real errors
3. Single-resolution (512) misses both coarse and fine corrections

**The core tension**: moving vertices improves A1 but hurts C3 because vertex positions
define the UV mapping and texture sampling coordinates.

**Critical insight**: The texture degradation can be avoided if we:
1. Perform silhouette correction BEFORE texture sampling (in the voxel pipeline)
2. Or re-sample texture after correction using the voxel grid (which is position-invariant)

**Implementation strategy**:

Instead of correcting the final GLB mesh, correct the mesh BEFORE texture baking:

```python
# In decode_latent() or postprocess():
# 1. Extract shape mesh from SLAT
mesh = fdg_decode(shape_slat)
# 2. Correct silhouette (deform vertices)
corrected_mesh = silhouette_corrector.correct(mesh, ref_alpha)
# 3. THEN bake texture from voxel grid onto corrected mesh
# Since voxel grid is a volumetric representation, re-sampling from
# it at the new vertex positions gives correct colors
textured_mesh = bake_texture(corrected_mesh, tex_voxels)
```

This avoids the C3 degradation because texture is sampled AFTER geometric correction.

**Key changes to SilhouetteCorrector**:

1. **Soft Dice loss** (directly optimizes evaluation metric):
   ```python
   def soft_dice(pred, target):
       intersection = (pred * target).sum()
       return 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
   ```

2. **Multi-resolution pyramid** (coarse-to-fine):
   - Stage 1: 256px, 30 steps, lr=2e-3, max_disp=0.06 (large corrections)
   - Stage 2: 512px, 50 steps, lr=5e-4, max_disp=0.03 (fine tuning)

3. **Reduced Laplacian weight**: w=10 (from 50), with cotangent weighting

4. **Edge-length regularization**: prevent triangle collapse without preventing deformation

**Expected impact**: +4-8 pts A1, +0 pts C3 (if texture is re-baked)
**Risk**: Medium (need to verify re-baking works cleanly)
**Difficulty**: Medium-Hard (modify pipeline order, tune corrector)
**Time cost**: +5-15 seconds
**Ceiling**: ~92-95 A1 (topology-limited -- cannot add/remove vertices or faces)

---

## 5. Answers to Specific Questions

### Q1: Can we increase Stage 1 resolution beyond 32^3?

**Yes, the decoder already outputs 64^3.** The 32^3 for cascade is a deliberate downsampling.

1. **Use `pipeline_type='1024'`** which uses `ss_res=64` (already supported, zero code
   change). This gives 64^3 native occupancy but loses 1536 cascade capability.

2. **Modify cascade to use 64^3**: The decoder already produces 64^3 output. Just skip
   the `max_pool3d` downsampling in `sample_sparse_structure()`. The LR shape model needs
   32^3-scale coords (downsample 64->32 for LR, then upsample for HR). With 128GB GPU
   and max_num_tokens=98304, the additional tokens are feasible.

3. **Increasing beyond 64^3** requires retraining:
   - Flow model: currently 16^3 (4096 tokens). Going to 32^3 (32768 tokens) with full
     self-attention would be ~16x slower and ~32GB for attention alone.
   - Decoder: would need 3 upsample blocks (16->128) instead of 2 (16->64).
   - **NOT feasible** without model retraining and significant architecture changes.

4. **Refinement approaches at 64^3**: The decoder's learned upsampling from 16^3 to 64^3
   is where most boundary detail is created. The quality of this upsampling limits how
   much information is in the 64^3 output beyond what's already in 32^3. Empirically,
   the max_pool OR operation rarely loses information for COMPACT objects (voxels near
   surfaces tend to be locally dense), but it does lose concavity information.

### Q2: Can we do multi-resolution Stage 1 (coarse-to-fine)?

**The architecture already IS multi-resolution internally.** The flow model operates at
16^3, and the decoder upsamples to 64^3 via learned convolutional upsample blocks. This
is a form of coarse-to-fine: the flow model determines coarse structure at 16^3, and the
decoder refines boundaries at 64^3.

A further cascade at the sparse structure level would require:
1. Generate at 64^3, identify boundary voxels (decoder-level refinement already does this)
2. Subdivide boundary voxels to 128^3 (octree-style)
3. Use a local refinement model to predict fine occupancy

This is conceptually similar to SparseFlex (arXiv 2503.21732) which uses sparse octree
refinement. However, the TRELLIS.2 decoder has a fixed 16->64 upsample path with no
mechanism for further local refinement. **Not feasible without decoder modifications.**

A cheaper alternative: at 64^3, apply **signed distance field smoothing** to the binary
occupancy before thresholding. This can recover sub-voxel boundary detail:
```python
# Blur the decoder output before thresholding
decoded_smooth = F.avg_pool3d(decoder(z_s), kernel_size=3, stride=1, padding=1)
decoded = decoded_smooth > threshold  # smoother boundaries
```
This is trivially implementable and may recover 0.5-1 pts A1 from quantization artifacts.

### Q3: Post-generation shape refinement that preserves texture?

**Yes -- this is Intervention C above.** The key is to perform vertex deformation BEFORE
texture baking (not after). Since TRELLIS.2's texture representation is volumetric (voxels),
re-sampling at deformed vertex positions gives correct colors. The existing `SilhouetteCorrector`
operates AFTER the full pipeline, which is why it degrades C3.

File: `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/postprocessing/silhouette_corrector.py`

The corrector already has the right structure (nvdiffrast rendering, gradient descent,
Laplacian regularization). It needs:
1. Integration point moved earlier in the pipeline (before `o_voxel` postprocess)
2. Soft Dice loss instead of BCE
3. Multi-resolution pyramid
4. Re-baking texture from voxels onto deformed mesh

### Q4: Theoretical A1 maximum given 32^3 -> Dual Contouring -> mesh?

The FDG (Flexible Dual Grid) decoder does NOT lose significant silhouette accuracy
relative to the voxel grid. It has sub-voxel precision via vertex offsets and edge
intersection flags. The main accuracy loss is in the VOXEL GRID, not the mesh extraction.

**Theoretical maxima:**
- 32^3 voxels: ~93-96% A1 for compact objects, ~80-90% for complex
- 64^3 voxels: ~95-98% A1 for compact, ~90-95% for complex
- Post-correction: +3-8 pts on top (limited by topology, not resolution)

**Practical ceiling with all interventions**:
- Compact objects: 94-98 A1 (currently 86-89, so +5-9 pts achievable)
- Complex objects: 85-93 A1 (currently 67-84, so +5-18 pts achievable)
- Mean: 90-95 A1 (from current 83.8)

### Q5: Can render-time optimization (deformable mesh) improve silhouette matching?

**Yes**, this is exactly what the `SilhouetteCorrector` does. The question is WHERE in the
pipeline to apply it. The results show that applying it AFTER texture baking degrades C3.
Applying it BEFORE texture baking (Intervention C) should avoid this.

An alternative is **non-rigid ICP** between the rendered silhouette contour and the reference
contour, which would produce a displacement field without requiring differentiable rendering.
This would be faster (~1 second) but less accurate than gradient-based optimization.

---

## 6. Ranked Implementation Plan

### Priority 1: Threshold Sweep + Best-of-N Integration (QUICK WIN)

**Expected gain**: +1-3 pts A1
**Effort**: 2-4 hours
**Risk**: Very low

Sweep 5 occupancy thresholds on the SAME decoder output. Pick the one with best silhouette
Dice. Zero additional flow model evaluations needed.

Files to modify:
- `trellis2_image_to_3d.py`: Add threshold sweep in `sample_sparse_structure()`

### Priority 2: 64^3 Sparse Structure for Cascade (HIGH IMPACT)

**Expected gain**: +3-5 pts A1 average, +8-12 pts on complex shapes
**Effort**: 8-16 hours
**Risk**: Medium (token count management, SLAT model behavior with different coord density)

Files to modify:
- `trellis2_image_to_3d.py`: Add `1536_cascade_64` pipeline type
- `trellis2_image_to_3d.py`: Modify `sample_shape_slat_cascade()` to handle 64^3 input

This is the single highest-impact change because it addresses the ROOT CAUSE: 32^3
quantization losing thin features and boundary detail.

### Priority 3: Pre-Texture Silhouette Correction (HIGH IMPACT, MODERATE EFFORT)

**Expected gain**: +4-8 pts A1, neutral C3 (if texture re-baked)
**Effort**: 16-24 hours
**Risk**: Medium (pipeline integration, re-baking correctness)

Files to modify:
- `silhouette_corrector.py`: Add Soft Dice loss, multi-resolution, reduced regularization
- `trellis2_image_to_3d.py`: Move correction before `o_voxel` texture baking
- `postprocess.py` (o_voxel): Ensure texture baking works on deformed mesh

### Combined Trajectory

| Phase | Intervention | Expected A1 | Cumulative |
|-------|-------------|-------------|------------|
| Baseline | - | 83.8 | 83.8 |
| Phase 1 | Threshold sweep | +1.5 | 85.3 |
| Phase 2 | 64^3 cascade | +4.0 | 89.3 |
| Phase 3 | Pre-texture silcorr | +4.0 | 93.0 |

**Conservative estimate**: 88-90 after Phase 1+2
**Optimistic estimate**: 92-95 after all phases

---

## 7. Why the Silhouette Corrector Degraded C3

The test result `eval_v4_v4_baseline_silcorr_20260223_055012.json` shows:
- A1: 81.5 (no improvement over baseline 81.4!)
- C3: 69.9 (DOWN from 80.4 baseline)

**Root cause analysis**:

1. **A1 not improving**: The corrector's max_displacement=0.02 is too conservative for
   meaningful silhouette correction. At unit-cube scale, 0.02 = 2% of the bounding box.
   Many silhouette errors require 5-8% displacement.

2. **C3 degrading**: The corrector moves vertices AFTER texture has been baked. This
   means:
   - UV coordinates become distorted (stretched where vertices moved)
   - Texture sampling from the original voxel grid uses the ORIGINAL positions
   - The mismatch between vertex position (moved) and texture coordinate (original)
     causes smearing, stretching, and detail loss
   - The evaluation renders the deformed mesh with its baked texture, showing artifacts

3. **Wrong loss function**: BCE on raw silhouette pixels is a weak signal. Most pixels
   are far from the boundary and provide zero gradient. Soft Dice concentrates the signal
   on the boundary region.

**Fix**: Move correction before texture baking, increase max_displacement, use Soft Dice.

---

## 8. Analysis of the Quadratic Schedule Result

The `s_quad_ha` config scored A1=88.1 on T.png (vs 85.8 baseline). This is a +2.3 pt
improvement from just changing the timestep schedule to quadratic. The quadratic schedule
allocates more steps near t=0 where fine detail is resolved.

**Why this helps A1**: Near t=0, the flow model makes its final corrections to the shape.
With uniform spacing, the last step covers t=0.083->0 (for 12 steps). With quadratic,
the last step covers t=0.007->0, giving 12x finer granularity at the critical final moment.
This allows the model to make sub-voxel adjustments via the SDF features that improve
boundary precision.

**However**: This result is from a single image (T.png) and may not generalize. The
quadratic schedule with Heun also scored slightly lower on C3 (74.4 vs 79.3 baseline).
This suggests the quadratic schedule may over-commit to geometry at the expense of texture.

**Recommendation**: Test quadratic schedule on the full 10-image set to confirm generalization.
If it generalizes, combine with Interventions A-C for compound improvement.

---

## 9. Sparse Structure Stage: What the Model Actually Learns

Looking at `SparseStructureFlowModel.forward()` (line 224-247):

```python
# Input: x of shape [B, C_in, R, R, R] (dense 3D volume)
h = x.view(*x.shape[:2], -1).permute(0, 2, 1)  # [B, R^3, C_in]
h = self.input_layer(h)  # [B, R^3, D]
# + positional embedding
# N transformer blocks with full self-attention over R^3 tokens
# + cross-attention to image features (DINOv3 tokens)
h = self.out_layer(h)  # [B, R^3, C_out]
h = h.reshape(B, C_out, R, R, R)
```

The model uses FULL self-attention (not windowed) over all R^3 tokens. At R=16 (verified
from config), this is 4096 tokens -- manageable but already non-trivial.

The decoder upsamples from 16^3 to 64^3 via 2 learned `UpsampleBlock3d` layers. Each
uses `Conv3d(in_ch, out_ch*8, 3, padding=1)` followed by `pixel_shuffle_3d(x, 2)`,
providing a learned 2x spatial upsampling. The quality of the 64^3 output depends on:
1. How much information the 16^3 latent captures (4096 tokens, 8 channels = 32768 floats)
2. How well the decoder reconstructs fine structure from coarse latents

**Increasing R_latent from 16 to 32** would give 8x more spatial resolution in the
latent space (32768 tokens), but self-attention cost would grow to O(32K^2) = 16x current
cost. This requires retraining the flow model and VAE -- not feasible for inference-time
optimization. Additionally, the full self-attention at 32^3 (32768 tokens) would require
~32GB for attention alone, making it impractical even on 128GB GPU during training.

**The 16^3 latent is the fundamental information bottleneck.** 4096 floats per channel
(8 channels = 32768 total) must encode the full 3D occupancy of an arbitrary shape.
This is ~2 bits per 64^3 output voxel, which is sufficient for binary occupancy but
leaves little room for fine boundary detail.

---

## 10. The 67.5 Outlier: A Special Case

The 0a34... image at A1=67.5 deserves special attention because it also has A2=47.1
(extremely low color match). This suggests the model FUNDAMENTALLY misunderstood this
image, generating a shape with wrong topology or orientation.

For this class of failure:
- Threshold sweep: minimal help (wrong topology, not boundary precision)
- 64^3 cascade: moderate help (better boundary resolution, but topology still wrong)
- Silhouette correction: moderate help (can fix boundary, not topology)
- **Best-of-N seed search**: HIGH help (different seed -> different topology interpretation)
- **Multi-view generation**: HIGHEST help (additional views constrain interpretation)

For the 67.5 outlier specifically, the best strategy is **Staged Best-of-N** (already
implemented as `_run_staged_bon`). Running N=8 shape candidates with different seeds
would likely find at least one with correct topology, pushing this case from 67 to 80+.

---

## 11. Key References

### Sparse Voxel Representations
- [SparseFlex](https://arxiv.org/abs/2503.21732): High-resolution sparse isosurface modeling
- [Sparc3D](https://arxiv.org/abs/2505.14521): Sparse representation for high-res 3D shapes

### Differentiable Rendering for Shape Fitting
- [SoftRas](https://ar5iv.labs.arxiv.org/html/1901.05567) (ICCV 2019): Soft rasterizer
  with smooth gradients for silhouette fitting
- [Geometry in Style](https://openaccess.thecvf.com/content/CVPR2025/papers/Dinh_Geometry_in_Style_3D_Stylization_via_Surface_Normal_Deformation_CVPR_2025_paper.pdf) (CVPR 2025):
  dARAP mesh deformation with differentiable rendering

### Flow Matching Guidance
- [FlowDPS](https://arxiv.org/abs/2503.08136) (ICCV 2025): Posterior sampling for inverse
  problems via flow Tweedie formula
- [Physics-Constrained Flow Matching](https://arxiv.org/abs/2506.04171): Hard constraints
  in flow matching sampling

### Test-Time Optimization for 3D
- [GTR](https://arxiv.org/abs/2406.05649) (ICLR 2025): Render-and-compare refinement
- [Adventures with Differentiable Mesh Rendering](https://andrewkchan.dev/posts/diff-render.html):
  Practical guide to mesh fitting via differentiable rendering

---

## 12. Summary

The A1 ceiling at 83.8 is caused by THREE compounding factors:

1. **32^3 voxel quantization** (root cause, ~5-8 pts loss): Thin features and boundary
   detail are lost. Fix: use 64^3 sparse structure.

2. **Conservative post-generation correction** (preventable loss, ~3-5 pts): The existing
   corrector is too conservative and applies in the wrong pipeline stage. Fix: more
   aggressive correction BEFORE texture baking.

3. **Topology errors on hard images** (model limitation, ~5-15 pts for worst cases): The
   model sometimes misinterprets complex images. Fix: Best-of-N seed selection.

**Combined theoretical ceiling**: 92-95 A1 (from 83.8), with most of the improvement
coming from the 64^3 cascade and pre-texture correction.

**Effort-adjusted recommendation**: Start with the threshold sweep (2 hours, +1.5 pts),
then implement 64^3 cascade (16 hours, +4 pts), then pre-texture correction (24 hours,
+4 pts). Total: ~42 hours of implementation for an expected A1 of ~93.
