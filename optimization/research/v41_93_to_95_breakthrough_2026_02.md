# Research Report: Breaking Through 93/100 to 95+ on V4.1

**Date**: 2026-02-23
**Author**: Deep Learning Researcher Agent
**Scope**: Training-free techniques to recover 7.8 theoretical points from the V4.1 ceiling
**Status**: Literature survey complete, 15+ papers surveyed, ranked recommendations

---

## 1. Executive Summary

TRELLIS.2 currently scores ~93/100 on the V4.1 evaluator, with 5 of 9 dimensions at or near ceiling. The remaining 7.8 points are distributed across four non-ceiling dimensions:

| Dim | Name | Weight | Current | Gap (pts) | Weighted Gap |
|-----|------|--------|---------|-----------|-------------|
| A1 | Silhouette | 15% | ~81.4 | 2.79 | **LARGEST** |
| C3 | Detail Richness | 10% | ~79.0 | 2.10 | |
| A2 | Color Distribution | 10% | ~80.9 | 1.91 | |
| C1 | Tex Coherence | 15% | ~94.7 | 0.80 | |
| | **Total** | | | **7.60** | |

After surveying 15+ papers from 2024-2026 and analyzing the V4.1 scoring formulas in detail, I recommend **6 techniques** that can collectively push the score to 95-96/100 without retraining TRELLIS.2. The key insight is that the three gap dimensions require fundamentally different interventions: A1 needs geometric correction, C3 needs texture-space enhancement, and A2 needs color-space alignment.

### Top 6 Ranked Techniques

| Rank | Technique | Target | V4.1 pts | Effort | Risk |
|------|-----------|--------|----------|--------|------|
| 1 | **Render-and-Compare Texture Refinement** | C3 +1.5-2.1 | Medium | 1-2 days | Low |
| 2 | **Sliced OT Color Transfer** | A2 +1.2-1.9 | Medium | 4-8 hours | Low |
| 3 | **Multi-Resolution Silhouette Corrector** | A1 +1.5-2.5 | High | 1-2 days | Medium |
| 4 | **64^3 Sparse Structure Cascade** | A1 +0.5-1.5 | Medium | 8-16 hours | Medium |
| 5 | **Staged Best-of-N (Shape Selection)** | A1 +0.5-1.5 | Low | 2-4 hours | Very Low |
| 6 | **UV Seam Laplacian Sharpening** | C3 +0.3-0.8 | Low | 2-4 hours | Very Low |

**Conservative projection**: +3.5-5.0 pts (93 -> 96.5-98.0)
**Optimistic projection**: +5.5-7.8 pts (93 -> 98.5-100.8, capped at ~99)

---

## 2. Deep Analysis of Each Scoring Gap

### 2.1 A1 Silhouette (Gap: 2.79 weighted pts)

**What V4.1 measures**: Scale-invariant Dice coefficient between the best-matching view (among 8 rendered views) and the input image's alpha mask. Both are cropped to bounding box and resized to 256x256.

**Formula**: `Dice = 2 * |A intersect B| / (|A| + |B|)`, scored as raw percentage (0-100).

**Current failure modes** (from per-image analysis):
- **Complex topology** (crown at 67.5): Model fundamentally misinterprets the shape. Stage 1 sparse structure generates wrong topology.
- **Thin features** (scores 78-84): 32^3 voxel quantization loses boundary detail. Each voxel projects to ~2.25 pixels on 256x256.
- **Compact objects** (scores 86-89): Near ceiling for 32^3 resolution but ~5 pts below theoretical max.

**Root cause hierarchy**:
1. 32^3 discretization (5-8 pts lost for complex shapes)
2. Conservative silhouette corrector (3-5 pts lost: max_disp=0.02 too small, BCE loss weak, w_lap=50 too rigid)
3. Topology errors on hard images (5-15 pts for worst cases, addressable only by Best-of-N)

### 2.2 C3 Detail Richness (Gap: 2.10 weighted pts)

**What V4.1 measures**: Three-component composite:
- 40%: UV texture-map Laplacian energy (log-scale: 3->30pts, 6->60pts, 9->78pts, 12->90pts, 15->100pts)
- 40%: Mesh dihedral angle std (log-scale: 8->30pts, 12->65pts, 16->85pts, 20->100pts)
- 20%: View-based Shannon entropy (grayscale histogram)

**Current bottleneck analysis**:
- `tex_lap_energy` currently ~7.7-8.3 (mapping to ~60-70 on the log scale)
- `dihed_std` currently ~10-12 (mapping to ~50-65)
- `entropy` currently ~4.0-5.0 (mapping to ~90-97)

The tex_detail and geo_detail components are the bottlenecks. Entropy is near ceiling.

**Key insight for C3**: The 40% geometry weight means that just increasing texture detail is insufficient; we also need geometric surface complexity (dihedral angle variation). The `split_sched` finding (+5.0 pts C3) works because quadratic time-stepping near t=0 improves BOTH the fine geometry and texture simultaneously.

### 2.3 A2 Color Distribution (Gap: 1.91 weighted pts)

**What V4.1 measures**: Three-component blend:
- 50%: HSV histogram correlation (H with 36 bins, S with 32 bins)
- 35%: Saturation-adaptive mean color proximity (hue + saturation)
- 15%: LAB chrominance distance (delta-E on a*, b* only)

**Key weakness**: The metric compares the rendered front view's color distribution against the input image. For objects with complex or unfamiliar color palettes (crown = gold, steampunk = mixed metals), the model's hallucinated back-side colors can shift the histogram.

**Root cause**: Single-view generation inherently leaves color ambiguous for occluded surfaces. The texture model sometimes desaturates or shifts hue on non-visible regions, which affects the histogram even when the front-view appearance is correct.

### 2.4 C1 Tex Coherence (Gap: 0.80 weighted pts)

This is already at 94.7-96.7 and near ceiling. The `split_sched` finding showed a -2.0 trade-off here, suggesting that optimizing C3 may slightly hurt C1. The narrow guidance interval [0.05, 0.85] gave +1.9 on C1. This dimension is likely best addressed by the same guidance schedule tuning already explored.

---

## 3. Technique-by-Technique Deep Dive

### 3.1 Render-and-Compare Texture Refinement (Rank 1)

**Papers**:
- GTR (Zhuang et al., ICLR 2025): Per-instance texture refinement via MSE loss, 20 iterations, 4 seconds on A100
- PBR-SR (Chen et al., NeurIPS 2025): DiffBIR + nvdiffrast for 4x texture super-resolution
- Generative Detail (Saeed et al., SIGGRAPH 2025): UV-consistent noise + projective attention for PBR enhancement

**What it does**: After the standard TRELLIS.2 pipeline produces a GLB mesh, render the front view differentiably via nvdiffrast, compare against the input image, and back-propagate through the UV texture to minimize the difference.

**Why this is Rank 1**: The `TextureRefiner` class already exists at `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/postprocessing/texture_refiner.py` with nvdiffrast rendering, LPIPS-VGG loss, chrominance L1, and multi-view support. It is implemented but UNTESTED. This is the single highest-ROI action available.

**How it improves C3**: The Laplacian energy of rendered views is dominated by the texture map content. By optimizing the texture to match the input image (which contains the "ground truth" detail), we inject high-frequency content that the flow model may have smoothed out.

**How it simultaneously improves A2**: Render-and-compare minimizes the chrominance distance between rendered and input views. This directly optimizes the A2 metric's LAB component and indirectly optimizes the histogram correlation.

**Expected V4.1 impact**:
- C3: +1.5-2.1 pts (tex_lap_energy 7.7 -> 9.5-11.0 from restored detail)
- A2: +0.5-1.0 pts (improved color fidelity from front-view optimization)
- C1: +0.2-0.5 pts (improved coherence from aligned front view)

**Critical modifications needed before testing**:
1. **Add high-frequency loss**: L1 on Laplacian-filtered images (directly targets C3's tex_lap_energy)
2. **Reduce TV weight**: From 0.01 to 0.001 (TV penalizes what C3 rewards)
3. **Increase iterations**: From default to 80 (GTR uses 20, but texture-only optimization needs more)
4. **Multi-view optimization**: Use 2-4 views (not just front) to avoid overfitting visible region

**Effort**: 1-2 days (modify TextureRefiner params, integrate into app.py, test)
**Risk**: Low (existing infrastructure, well-understood optimization)

**Key references**:
- [GTR](https://arxiv.org/abs/2406.05649) (ICLR 2025)
- [PBR-SR](https://arxiv.org/abs/2506.02846) (NeurIPS 2025)
- [Generative Detail](https://dl.acm.org/doi/10.1145/3721238.3730751) (SIGGRAPH 2025)

---

### 3.2 Sliced Optimal Transport Color Transfer (Rank 2)

**Papers**:
- Pitie, Kokaram, Dahyot, "Automated Colour Grading using Colour Distribution Transfer" (CVIU 2007)
- Bonneel et al., "Sliced and Radon Wasserstein Barycenters of Measures" (JMIV 2015)
- "Color Transfer via Sliced Optimal Transport" (Coeurjoly, 2024)
- "Color Transfer with Modulated Flows" (arXiv 2503.19062, March 2025)

**What it does**: After generating the 3D model, render it from the input viewpoint. Compute a color transfer function that maps the rendered color distribution to the input image's distribution. Apply this transfer to the UV texture map, but ONLY to texels visible from the input viewpoint.

**Why Sliced OT over Reinhard**: Classical Reinhard color transfer (1st moment matching in Lab space) only matches mean and std, losing multimodal structure. Sliced Optimal Transport preserves the full distribution shape by solving 1D transport along random projections. For an object like a gold crown with both bright and dark gold regions, Reinhard would shift everything toward the mean, while Sliced OT preserves the bimodal structure.

**Algorithm**:
```
1. Render 3D model from input camera pose at 512px
2. Extract foreground pixels from both rendered and input images
3. Convert to Lab color space
4. Compute Sliced OT transport plan (100 random 1D projections, 50 iterations)
5. Apply transport to UV texture: for each visible texel, transform its Lab values
6. Blend transferred texture (visible region) with original texture (occluded region)
```

**How it improves A2**: This directly optimizes the histogram correlation component (50% of A2). By matching the distribution of rendered colors to the input, the HSV histogram correlation approaches 1.0 for the front view. The mean color proximity (35%) and LAB distance (15%) also improve.

**Visibility-aware application is critical**: Only texels visible from the input view should be color-transferred. Back-side texels should remain as-is. This requires the nvdiffrast rasterization to identify which UV texels are visible, which is already implemented in the TextureRefiner's rendering pipeline.

**Expected V4.1 impact**:
- A2: +1.2-1.9 pts (directly optimizes the metric)
- C2: +0-0.2 pts (may improve color vitality if desaturated regions are corrected)
- C1: +0-0.3 pts (better color alignment improves coherence)

**Implementation**:
```python
import numpy as np
from scipy.ndimage import distance_transform_edt

def sliced_ot_color_transfer(source_pixels, target_pixels, n_projections=100, n_iters=50):
    """Sliced optimal transport color transfer in Lab space.

    Args:
        source_pixels: (N, 3) Lab values from rendered image
        target_pixels: (M, 3) Lab values from input image
    Returns:
        transferred: (N, 3) Lab values after transfer
    """
    transferred = source_pixels.copy()
    for _ in range(n_iters):
        # Random projection direction on unit sphere
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)

        # Project both sets
        src_proj = transferred @ direction  # (N,)
        tgt_proj = target_pixels @ direction  # (M,)

        # Sort and match quantiles
        src_idx = np.argsort(src_proj)
        tgt_sorted = np.sort(tgt_proj)

        # Interpolate target quantiles to source size
        tgt_interp = np.interp(
            np.linspace(0, 1, len(src_proj)),
            np.linspace(0, 1, len(tgt_sorted)),
            tgt_sorted
        )

        # Displacement along projection direction
        displacement = tgt_interp[np.argsort(src_idx)] - src_proj
        transferred += np.outer(displacement, direction)

    return transferred
```

**Effort**: 4-8 hours (pure numpy/opencv, no new dependencies)
**Risk**: Low (classical technique, well-understood)

**Key references**:
- [Sliced OT Color Transfer](https://dcoeurjo.github.io/OTColorTransfer/)
- [Color Transfer with Modulated Flows](https://arxiv.org/html/2503.19062v1)
- [Sequential Color Correction for 3D Mesh Texture](https://pmc.ncbi.nlm.nih.gov/articles/PMC9865480/)

---

### 3.3 Multi-Resolution Silhouette Corrector (Rank 3)

**Papers**:
- SoftRas (Liu et al., ICCV 2019): Soft differentiable rasterizer for silhouette fitting
- Palfinger, "Continuous remeshing for inverse rendering" (2022): Optimal Laplacian weight w=5-15
- DMesh++ (Son, ICCV 2025): Differentiable mesh optimization with proper regularization
- GTR (Zhuang et al., ICLR 2025): Joint geometry + texture refinement

**What it does**: The existing `SilhouetteCorrector` deforms mesh vertices to match the input image's silhouette. The current implementation FAILS (A1 unchanged, C3 degrades) due to: too-conservative max displacement (0.02), wrong loss function (BCE), excessive regularization (w_lap=50), and wrong pipeline ordering (after texture baking).

**Why this is Rank 3 (not Rank 1)**: The corrector exists but its previous test DEGRADED quality (A1 +0.1, C3 -10.5). The required fixes are well-understood but need careful implementation:

**Five critical upgrades**:

1. **Soft Dice loss** (replaces BCE): BCE distributes gradients uniformly across all pixels, which is wasteful since 90%+ of pixels are far from the boundary. Soft Dice concentrates signal on boundary pixels and directly optimizes the evaluation metric.
   ```python
   def soft_dice_loss(pred, target):
       p, t = pred.flatten(), target.flatten()
       return 1.0 - (2.0 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
   ```

2. **Coarse-to-fine pyramid**: Two stages -- (256px, 30 steps, lr=2e-3, max_disp=0.06) then (512px, 50 steps, lr=5e-4, max_disp=0.03). Coarse stage makes large corrections; fine stage refines boundaries.

3. **Reduced Laplacian weight**: w=10-15 (from 50). The literature (Palfinger 2022) shows that w=5-15 with cotangent weighting is optimal for silhouette fitting. At w=50, the mesh is essentially frozen.

4. **Pipeline reordering**: Apply correction BEFORE texture baking, not after. Since TRELLIS.2's texture representation is volumetric (sparse voxels), moving vertices and then re-sampling texture from the voxel grid gives correct colors at the new positions. This eliminates the C3 degradation.

5. **Edge-length regularization**: Instead of only Laplacian smoothness, add edge-length preservation to prevent triangle collapse:
   ```python
   edge_lengths = (vertices[edges[:, 0]] - vertices[edges[:, 1]]).norm(dim=1)
   edge_reg = ((edge_lengths - original_edge_lengths) ** 2).mean()
   ```

**How it improves A1**: Directly optimizes the Dice metric at the pixel level. The coarse-to-fine pyramid can handle both large shape corrections (wrong protrusions) and fine boundary alignment (voxel staircase).

**Expected V4.1 impact**:
- A1: +1.5-2.5 pts (from boundary refinement + large-displacement correction)
- C3: +0.0-0.5 pts (if pipeline reordering works correctly, geo_detail may improve)
- C1: -0.0-0.3 pts (slight risk of texture distortion if voxel re-sampling is imperfect)

**Theoretical ceiling**: ~90-92 on A1 (topology-limited: cannot add/remove faces, only move vertices).

**Effort**: 1-2 days (modify silhouette_corrector.py, change pipeline ordering)
**Risk**: Medium (pipeline reordering needs testing; voxel re-sampling at deformed positions is novel)

**Key references**:
- [SoftRas](https://arxiv.org/abs/1901.05567) (ICCV 2019)
- [Continuous remeshing](https://www.researchgate.net/publication/362099386) (Palfinger 2022)
- [DMesh++](https://arxiv.org/abs/2309.07126) (ICCV 2025)

---

### 3.4 64^3 Sparse Structure Cascade (Rank 4)

**Papers**:
- XCube (Ren et al., NeurIPS 2024): Hierarchical sparse voxel generation preserves fine detail
- SparseFlex (arXiv 2503.21732, ICCV 2025 Oral): 1024^3 sparse isosurface for detail
- Sparc3D (arXiv 2505.14521): Sparse representation for high-resolution 3D shapes

**What it does**: The TRELLIS.2 decoder already produces 64^3 occupancy (16^3 latent -> 2 learned upsamples -> 64^3). For cascade pipelines, this is max-pooled down to 32^3. Skipping the max_pool preserves boundary detail at the cost of ~4x more SLAT tokens.

**Why this is Rank 4 (not higher)**: While conceptually the most impactful for A1 (halving boundary error), the implementation has significant engineering risk:
- 4-8x more tokens requires max_num_tokens >= 98304 (feasible on 128GB but needs testing)
- The LR shape model was trained on 32^3-density coordinate distributions
- Generation time roughly doubles
- The per-image A1 improvement may be less than theoretical because the model's 16^3 latent already limits fine-boundary information

**Architecture constraint analysis**:
The flow model generates at 16^3 (4096 tokens, 8 channels = 32768 floats of information). The decoder upsamples to 64^3, but its learned upsampling from 16^3 is information-bottlenecked. The max_pool OR operation at 32^3 loses concavities (small voids are filled) but rarely loses convexities (occupied sub-voxels propagate up). For MOST objects, the 32^3 -> 64^3 difference is primarily at concavities and thin negative features.

**Where this helps most**: Objects with thin protruding features (antlers, crown filigree) or concave regions (bowls, cups). For compact convex objects, the improvement may be only 1-2 pts.

**Expected V4.1 impact**:
- A1: +0.5-1.5 pts average (+3-5 pts on complex objects like the crown)
- Generation time: +50-100% (acceptable for quality mode)

**Implementation**: Already coded as `ss_native_64=True` parameter. Needs testing with max_num_tokens=98304.

**Effort**: 8-16 hours (testing, token limit adjustment, timing validation)
**Risk**: Medium (token overflow, OOD coordinate density for SLAT model)

**Key references**:
- [XCube](https://arxiv.org/abs/2312.03806) (NeurIPS 2024)
- [SparseFlex](https://arxiv.org/abs/2503.21732) (ICCV 2025)

---

### 3.5 Staged Best-of-N Shape Selection (Rank 5)

**Papers**:
- Ma et al., "Scaling Inference Time Compute for Diffusion Models" (CVPR 2025)
- Tang et al., "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps" (arXiv:2501.09732)
- Kim et al., "Inference-Time Scaling for Flow Models via Stochastic Generation and RBF" (NeurIPS 2025)

**What it does**: Generate N shape candidates (different seeds), evaluate silhouette Dice for each WITHOUT texturing, then texture only the best shape.

**Why this is effective**: The A1 score has high per-seed variance (std ~6.2 across the test set). For the outlier image at 67.5, the score likely reflects a bad random draw. With N=4 candidates, P(at least one scores 10+ pts higher) is substantial for high-variance cases.

**Cost analysis**:
- Shape-only evaluation: Render occupancy silhouette at 256x256, compare Dice. Takes ~0.5 seconds per candidate.
- Shape generation (SS + shape SLAT): ~20-30% of total pipeline time.
- N=4 staged: 4 shapes + 1 texture = ~1.8x total time.

**Expected V4.1 impact**:
- A1: +0.5-1.5 pts average (+5-10 pts on outlier cases like the crown)
- A2: +0-0.5 pts (better shape -> better visible-region coverage)

**Implementation**: Already wired in `app.py` line 534 (full pipeline BoN). The STAGED variant (shape-only evaluation) needs a fast silhouette Dice verifier, which can be implemented by rendering the sparse structure occupancy grid directly.

**Effort**: 2-4 hours (add fast Dice evaluation of SS occupancy)
**Risk**: Very low (well-understood technique, no model changes)

**Key references**:
- [Inference-Time Scaling for Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf) (CVPR 2025)
- [Flow Inference-Time Scaling](https://arxiv.org/abs/2503.19385) (NeurIPS 2025)

---

### 3.6 UV Seam Laplacian Sharpening (Rank 6)

**Papers**:
- "GPU-Friendly Laplacian Texture Blending" (arXiv 2502.13945, February 2025)
- Hunyuan3D 2.1 (arXiv 2506.15442): 3D-Aware RoPE for UV seam reduction

**What it does**: After the o_voxel postprocess fills UV holes via Gaussian diffusion (sigma=20, blend=0.7), apply a per-frequency-band sharpening step to restore high-frequency content in the filled regions.

**Why this helps C3**: The fill_holes operation blurs texels in gap regions, reducing tex_lap_energy by 5-15% near seam boundaries. Since UV seams can cover 10-20% of the texture map area (xatlas produces 5000+ UV islands for complex objects), this blur effect is measurable.

**Implementation** (pure numpy/opencv, ~30 lines):
```python
def sharpen_filled_regions(texture_map, fill_mask):
    """Restore high-freq content where fill_holes blurred the texture."""
    tex_f = texture_map.astype(np.float32)
    gauss_low = cv2.GaussianBlur(tex_f, (0, 0), sigmaX=1.5)
    high_freq = tex_f - gauss_low
    sharpened = tex_f + 0.7 * high_freq
    sharpened = np.clip(sharpened, 0, 255)
    result = texture_map.astype(np.float32).copy()
    result[fill_mask] = 0.3 * result[fill_mask] + 0.7 * sharpened[fill_mask]
    return np.clip(result, 0, 255).astype(np.uint8)
```

**Expected V4.1 impact**:
- C3: +0.3-0.8 pts (tex_lap_energy improvement in seam regions)
- Runtime: <100ms (negligible)

**Effort**: 2-4 hours (add to postprocess.py, sync to container)
**Risk**: Very low (pure post-processing, no model interaction)

**Key references**:
- [GPU-Friendly Laplacian Texture Blending](https://arxiv.org/abs/2502.13945)

---

## 4. Techniques Considered but NOT Recommended

### 4.1 Elevate3D / HFS-SDEdit (HIGH potential, too slow)

Elevate3D (SIGGRAPH 2025) uses FLUX-based HFS-SDEdit to enhance rendered views, then back-projects to UV. Expected +3-8 on A2, +5-10 on C3. However:
- Requires FLUX model (~12GB VRAM, ~12GB download)
- Processing: 25 minutes per model (unoptimized)
- Container dependency management adds complexity

**Verdict**: Excellent for a future "quality premium" mode, but overkill for the 93->95 push. Save for 95->98 effort.

### 4.2 SuperCarver (Geometry detail, slow)

SuperCarver (arXiv 2503.09439) generates detail-boosted normal maps via diffusion, then applies noise-resistant SDF deformation. Could improve C3's geo_detail component. However:
- Requires training a normal diffusion model or using their pretrained one
- Multi-view processing adds 5-10 minutes
- Geometry detail is only 40% of C3

**Verdict**: Not cost-effective for 2.1 pt C3 gap. Would be valuable if C3 gap were 5+.

### 4.3 3DEnhancer (Multi-view diffusion enhancement)

3DEnhancer (CVPR 2025) uses a multi-view latent diffusion model to enhance coarse 3D inputs. Strong results on texture quality and consistency. However:
- Requires training or downloading a multi-view enhancement model
- Per-instance processing is significant
- The TRELLIS.2 output is already high quality; enhancement may not help vs. starting from scratch

**Verdict**: Better suited for enhancing outputs from weaker models (DreamGaussian, etc.), not for already-strong TRELLIS.2.

### 4.4 Flow Inference-Time Scaling with RBF (High potential, complex)

Kim et al. (NeurIPS 2025) convert the ODE to SDE with rollover budget forcing for particle-based quality search. Could help all three gap dimensions. However:
- ODE-to-SDE conversion for sparse 3D tensors is novel and untested
- SDE confirmed to have ZERO effect in A/B testing (Test 5 results)
- The reward function (decode SLAT to mesh) is expensive at each particle step

**Verdict**: The SDE approach was empirically tested and failed for TRELLIS.2. The specific structure of sparse 3D latents may not benefit from the stochastic perturbation that helps 2D image diffusion.

### 4.5 Rectified-CFG++ (Testing now, uncertain)

The predictor-corrector approach shows theoretical promise but has 1.5x cost and requires hyperparameter recalibration from 2D settings to 3D sparse latents. Currently under testing.

**Verdict**: Wait for A/B test results before investing further.

---

## 5. Combined Implementation Plan

### Phase 1: Quick Wins (Day 1) -- Expected: +1.5-3.0 pts

| Action | Time | Expected Impact |
|--------|------|----------------|
| Test TextureRefiner with default params | 2h | Baseline measurement |
| Tune TextureRefiner (reduce TV, add HF loss) | 4h | C3 +1.0-1.5 |
| Add UV seam sharpening to postprocess.py | 2h | C3 +0.3-0.8 |
| Test staged Best-of-N with N=4 shapes | 2h | A1 +0.5-1.5 |

### Phase 2: Color Transfer (Day 2) -- Expected: +1.2-1.9 pts

| Action | Time | Expected Impact |
|--------|------|----------------|
| Implement Sliced OT color transfer | 4h | A2 +1.2-1.9 |
| Add visibility masking via nvdiffrast | 2h | Prevents back-side corruption |
| Integrate into postprocess pipeline | 2h | End-to-end |

### Phase 3: Silhouette Correction (Days 3-4) -- Expected: +1.5-2.5 pts

| Action | Time | Expected Impact |
|--------|------|----------------|
| Upgrade SilhouetteCorrector (Soft Dice, coarse-to-fine) | 8h | A1 +1.0-2.0 |
| Move correction before texture baking | 8h | Prevents C3 degradation |
| Test 64^3 cascade | 4h | A1 +0.5-1.5 additional |

### Phase 4: Validation and Combination (Day 5)

| Action | Time | Expected Impact |
|--------|------|----------------|
| Full 10-image evaluation of combined config | 4h | Confirm total improvement |
| Resolve interaction effects between phases | 4h | Handle A1/C3 trade-offs |

### Projected Score Trajectory

| After Phase | A1 | A2 | C3 | C1 | Overall |
|-------------|-----|-----|-----|-----|---------|
| Current | 81.4 | 80.9 | 79.0 | 94.7 | ~93.0 |
| Phase 1 | 82.5 | 81.5 | 83.0 | 94.5 | **~94.3** |
| Phase 2 | 82.5 | 85.0 | 83.0 | 95.0 | **~95.1** |
| Phase 3 | 88.0 | 85.0 | 83.5 | 94.7 | **~96.3** |
| Optimistic | 90.0 | 88.0 | 86.0 | 95.0 | **~97.5** |

---

## 6. Critical Interactions and Trade-Offs

### 6.1 TextureRefiner vs. Color Transfer Ordering

Both modify the UV texture map. The correct ordering is:
1. **First**: Silhouette correction (modifies geometry, must happen before texture)
2. **Second**: TextureRefiner (optimizes texture to match input, uses geometry as fixed)
3. **Third**: Color transfer (fine-tunes color distribution, operates on refined texture)

Reversing TextureRefiner and Color Transfer would cause the refiner to undo the transfer.

### 6.2 split_sched and TextureRefiner Interaction

The `split_sched` finding (+5.0 C3, -2.0 C1) concentrates ODE steps near t=0, producing more detailed textures at the expense of global coherence. If TextureRefiner is used, the coherence penalty from split_sched may be partially recovered, because the refiner re-aligns the front-view appearance. This suggests the combination `split_sched + TextureRefiner` may be super-additive.

### 6.3 Silhouette Correction and C3 Risk

The previous silhouette corrector test showed C3 -10.5 pts because correction was applied AFTER texture baking. If pipeline reordering (correction BEFORE texture) cannot be cleanly implemented, an alternative is to:
1. Apply correction with max_disp=0.03 (smaller displacement reduces UV distortion)
2. Re-bake texture from the original voxel grid at the deformed vertex positions
3. This requires access to the voxel grid at the point of correction, which is available in the pipeline

### 6.4 64^3 and Best-of-N Interaction

These are complementary: 64^3 reduces systematic boundary error (helps all images), while Best-of-N reduces stochastic topology error (helps worst-case images). The combination addresses both failure modes.

---

## 7. Monitoring and Evaluation

### Key Metrics to Track Per Dimension

For each technique, run the full 10-image V4.1 evaluation and monitor:

| Technique | Primary KPI | Secondary KPIs | Failure Signal |
|-----------|------------|----------------|----------------|
| TextureRefiner | tex_lap_energy delta | LPIPS (front view), C1 delta | C1 < 92 (over-fitting) |
| Color Transfer | hist_correl delta | delta_E reduction, C2 stability | C2 < 95 (over-desaturation) |
| Silhouette Corr | Dice delta | C3 delta, dihed_std delta | C3 < 75 (texture damage) |
| 64^3 Cascade | A1 delta (complex imgs) | Gen time, token count | OOM or time > 15min |
| Best-of-N | A1 std reduction | Outlier A1 improvement | No improvement on best seed |
| UV Sharpening | tex_lap_energy delta | Visual artifact check | Ringing artifacts in UV |

### Success Criteria

- **95/100**: Achieved if Phase 1 + Phase 2 deliver expected improvements
- **96/100**: Requires Phase 3 silhouette correction to work correctly
- **97+/100**: Requires all techniques to be additive (optimistic)

---

## 8. Paper References (Full List)

### Render-and-Compare / Texture Refinement
1. Zhuang, P. et al. "GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement." ICLR 2025. [Paper](https://arxiv.org/abs/2406.05649) | [Code](https://github.com/snap-research/GTR)
2. Chen et al. "PBR-SR: Mesh PBR Texture Super Resolution from 2D Image Priors." NeurIPS 2025. [Paper](https://arxiv.org/abs/2506.02846) | [Website](https://terencecyj.github.io/projects/PBR-SR/)
3. Saeed et al. "Generative Detail Enhancement for Physically Based Materials." SIGGRAPH 2025. [Paper](https://dl.acm.org/doi/10.1145/3721238.3730751) | [Code](https://github.com/saeedhd96/generative-detail)
4. Ryu et al. "Elevating 3D Models: High-Quality Texture and Geometry Refinement." SIGGRAPH 2025. [Paper](https://arxiv.org/abs/2507.11465) | [Code](https://github.com/ryunuri/Elevate3D)

### Color Transfer
5. Pitie, Kokaram, Dahyot. "Automated Colour Grading using Colour Distribution Transfer." CVIU 2007.
6. Bonneel et al. "Sliced and Radon Wasserstein Barycenters of Measures." JMIV 2015. [Code](https://dcoeurjo.github.io/OTColorTransfer/)
7. "Color Transfer with Modulated Flows." arXiv:2503.19062, March 2025. [Paper](https://arxiv.org/abs/2503.19062)
8. "A Sequential Color Correction Approach for Texture Mapping of 3D Meshes." MDPI Sensors 2023. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9865480/)

### Silhouette Correction and Differentiable Rendering
9. Liu et al. "Soft Rasterizer: Differentiable Rendering for Unsupervised Single-View Mesh Reconstruction." ICCV 2019. [Paper](https://arxiv.org/abs/1901.05567)
10. Palfinger. "Continuous remeshing for inverse rendering." 2022. [Paper](https://www.researchgate.net/publication/362099386)
11. Son. "DMesh++: An Efficient Differentiable Mesh for Complex Shapes." ICCV 2025.
12. Laine et al. "Nvdiffrast: Modular Primitives for High-Performance Differentiable Rendering." SIGGRAPH Asia 2020. [Code](https://github.com/NVlabs/nvdiffrast)

### Inference-Time Scaling
13. Ma et al. "Scaling Inference Time Compute for Diffusion Models." CVPR 2025.
14. Kim et al. "Inference-Time Scaling for Flow Models via Stochastic Generation and RBF." NeurIPS 2025. [Paper](https://arxiv.org/abs/2503.19385) | [Code](https://github.com/KAIST-Visual-AI-Group/Flow-Inference-Time-Scaling)
15. Tang et al. "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps." arXiv:2501.09732.

### 3D Generation Quality
16. Chen et al. "MeshGen: Generating PBR Textured Mesh with Render-Enhanced Auto-Encoder." CVPR 2025 Highlight. [Paper](https://arxiv.org/abs/2505.04656) | [Code](https://github.com/heheyas/MeshGen)
17. Luo et al. "3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement." CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_3DEnhancer_Consistent_Multi-View_Diffusion_for_3D_Enhancement_CVPR_2025_paper.pdf) | [Code](https://github.com/Luo-Yihang/3DEnhancer)
18. "SuperCarver: Texture-Consistent 3D Geometry Super-Resolution." arXiv:2503.09439. [Paper](https://arxiv.org/abs/2503.09439)
19. Tencent. "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material." arXiv:2506.15442. [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
20. "GPU-Friendly Laplacian Texture Blending." arXiv:2502.13945, February 2025. [Paper](https://arxiv.org/abs/2502.13945)

### Sparse Voxel Representations
21. Ren et al. "XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies." NeurIPS 2024.
22. "SparseFlex: High-Resolution Sparse Isosurface." arXiv:2503.21732, ICCV 2025.
23. Liang et al. "UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes." arXiv:2505.23253. [Code](https://github.com/YixunLiang/UniTEX)

### Meta 3D TextureGen / Multi-View Texture
24. "Meta 3D TextureGen: Fast and Consistent Texture Generation for 3D Objects." arXiv:2407.02430.
25. "MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D." arXiv:2411.02336.

---

## 9. Conclusion

The path from 93 to 95+ requires a fundamentally different approach than the guidance-tuning and sampler modifications explored so far. All guidance-based methods (CFG-Zero*, APG, beta schedules, SDE sampling, guidance annealing) have been exhaustively tested and yield at most +0.6 pts. The remaining gains must come from **post-generation refinement** -- techniques that operate on the generated mesh/texture rather than the generation process.

The three most impactful techniques are:
1. **Render-and-Compare Texture Refinement** (addresses C3 + A2 simultaneously)
2. **Sliced OT Color Transfer** (directly optimizes A2 metric)
3. **Multi-Resolution Silhouette Correction** (addresses A1, the largest gap)

These three techniques are orthogonal (operate on different dimensions), complementary (can be applied sequentially), and well-grounded in the literature. Combined with the lower-effort UV sharpening and Best-of-N selection, the projected improvement of +3.5-5.0 pts would push the V4.1 score to 96-98/100.
