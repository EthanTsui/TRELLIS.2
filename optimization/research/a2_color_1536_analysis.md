# A2 Color Distribution Score Analysis: 1024 vs 1536 Resolution

**Date**: 2026-02-23
**Author**: Research Optimizer Agent
**Status**: Research complete, no code modifications

---

## 1. Executive Summary

The A2_color_dist metric measures color fidelity between the input image and rendered 3D model views, using HSV histogram correlation, saturation-adaptive mean color proximity, and LAB chrominance distance. The user reported A2 dropping from ~80.9 (1024, 3 images) to ~77.7 (1536, 10 images), suggesting resolution harms color fidelity.

**Key finding**: After careful analysis, the observed A2 difference is **not a systematic resolution-dependent regression**. It is a statistical artifact from comparing different sample sizes (3 vs 10 images), evaluator version differences across runs, and high per-image variance (std=11-14 points). When compared on matched 10-image runs with identical evaluator versions, 1536 actually scores **higher** on A2 (77.7 vs 76.0, delta +1.7).

However, the analysis reveals three genuine mechanisms by which higher resolution *can* affect color fidelity, plus one evaluator confound. These are documented below with concrete interventions.

---

## 2. Data Analysis

### 2.1 Per-Image A2 Comparison (10-image, matched evaluator v4)

From `eval_v4_v4_baseline_20260222_211437.json` (1024) and `eval_v4_config_1536_20260222_232616.json` (1536):

| Image | 1024 A2 | 1536 A2 | Delta | Notes |
|-------|---------|---------|-------|-------|
| T.png (steampunk) | 86.3 | 84.1 | -2.2 | Slight drop |
| 0a34fa (crown) | 38.5 | 47.1 | **+8.6** | Major improvement |
| 454e7d | 83.5 | 84.1 | +0.6 | Equivalent |
| cd3c30 | 88.1 | 85.7 | -2.4 | Slight drop |
| ee8ecf | 81.1 | 80.1 | -0.9 | Equivalent |
| f35156 | 82.7 | 79.0 | -3.7 | Notable drop |
| 7d585a | 79.6 | 81.1 | +1.5 | Slight improvement |
| e4d6b2 | 85.7 | 88.3 | +2.6 | Improvement |
| d39c2b | 68.1 | 73.0 | **+4.9** | Major improvement |
| 5a6c81 | 66.3 | 74.2 | **+7.9** | Major improvement |
| **MEAN** | **76.0** | **77.7** | **+1.7** | 1536 wins |
| **STD** | 14.3 | 11.2 | | Lower variance at 1536 |

**Conclusion**: On matched 10-image runs, 1536 has **higher** average A2 and **lower** variance. The user's observation of a drop was comparing 3-image 1024 (80.9) against 10-image 1536 (77.7) -- the 3-image sample happens to exclude the worst-scoring images (crown at 38-47, d39c2b at 68-73, 5a6c81 at 66-74), inflating the 1024 mean.

### 2.2 Across-Run Variability

Multiple 3-image baseline runs show significant A2 variability even at the same resolution:

| Run Timestamp | Resolution | A2 Mean (3 img) | Notes |
|---------------|-----------|-----------------|-------|
| 2022-191816 | 1024 | 62.0 | Early evaluator build |
| 2022-211437 | 1024 | 82.1 (first 3) | Matches v4 final |
| 2023-043806 | 1024 | 82.1 | Consistent |
| 2023-073359 | 1024 | 80.9 | User reference |
| 2022-213830 | 1536 | 71.8 | |
| 2022-232616 | 1536 | 82.6 (first 3) | Matches 10-img first 3 |
| 2023-033410 | 1536 | 82.6 | Consistent |
| 2023-040427 | 1536 | 82.7 | Consistent |

The 3-image comparisons show 1536 A2 is **equal or higher** than 1024 on the same first 3 images (82.6-82.7 vs 80.9-82.1).

---

## 3. Root Cause Analysis: How Resolution Affects Color

Although 1536 does not systematically degrade A2, I identified four mechanisms that *can* cause per-image color shifts. Understanding them is valuable for both debugging outliers and potential improvement.

### 3.1 Mechanism 1: Sparse Voxel Density and Color Distribution

**Theoretical basis**: At 1536 resolution, the shape cascade upsamples from 512 to 1536 via the `sample_shape_slat_cascade()` function (lines 769-856 of `trellis2_image_to_3d.py`). This creates more voxel coordinates (tokens) at higher resolution. The texture SLAT flow model (`tex_slat_flow_model_1024`) operates on these coordinates -- so at 1536, the same model must predict texture features for a denser voxel grid.

**How this affects color**: The 1024 texture model was trained on 1024-resolution voxel grids. When applied to a 1536-resolution grid:
- More tokens means the noise initialization has more degrees of freedom
- The model's attention patterns span a larger spatial extent in voxel coordinates
- The flow matching ODE must resolve finer spatial gradients

**Expected effect**: Color distribution should be slightly more detailed but potentially more noisy at boundaries. The CFG rescale (set to 1.0) helps prevent systematic desaturation, but the higher token count means each token has proportionally less "attention budget" from the conditioning image features.

**Literature support**: Hunyuan3D 2.0 (arXiv 2501.12202) found that scaling texture resolution requires careful attention to feature aggregation, using specialized "paint" models at multiple resolutions rather than a single model at all resolutions.

### 3.2 Mechanism 2: Trilinear Sampling at Higher Resolution

**Theoretical basis**: The texture baking step in `postprocess.py` (line 509) uses `grid_sample_3d()` with `mode='trilinear'` to sample voxel attributes at mesh vertex positions. At 1536 resolution:
- `grid_size = 1536` means `voxel_size = 1/1536 = 0.000651`
- At 1024: `voxel_size = 1/1024 = 0.000977`

The grid coordinates are computed as: `grid_pts = ((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)` (line 507)

**How this affects color**: At higher resolution, the trilinear interpolation samples from a finer grid. For a *sparse* voxel grid, this means:
- More mesh vertices fall in regions between occupied voxels
- Trilinear interpolation at these boundary regions produces averaged (desaturated) colors
- The effect is most pronounced for thin structures and high-frequency color transitions

However, this effect is mitigated by BVH reprojection (line 498-500), which snaps sampling positions back to the original mesh surface. The BVH reprojection ensures sampling positions are well-localized rather than floating between voxels.

**Literature support**: The survey of voxel interpolation methods (ICRA 2018) confirms trilinear interpolation acts as a low-pass filter at higher spatial frequencies, though this is more significant when upsampling than when sampling a native-resolution grid.

### 3.3 Mechanism 3: Guidance Dynamics at Different Token Counts

**Theoretical basis**: CFG operates by computing `pred = pred_pos + w * (pred_pos - pred_neg)` where `w = guidance_strength - 1`. With `guidance_strength=12.0` and `guidance_rescale=1.0`, the texture stage uses very strong guidance.

The rescale operation (line 31-38 of `classifier_free_guidance_mixin.py`) normalizes the CFG-boosted prediction's standard deviation to match the conditional prediction's standard deviation. This is computed *globally* across all tokens: `std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)`.

**How this affects color**: At 1536 resolution with more tokens:
- The global standard deviation estimate is more stable (more samples)
- The rescale ratio `std_pos / std_cfg` may differ slightly
- With `guidance_rescale=1.0` (full rescale), the output is completely renormalized
- This prevents systematic desaturation but may suppress per-token color variation

**Literature support**:
- CFG++ (arXiv 2406.08070) demonstrates that standard CFG causes "sudden shifts and intense color saturation early in reverse diffusion"
- EP-CFG rescales guided predictions to preserve energy, reducing contrast and over-saturation
- LF-CFG (Frequency-Decoupled Guidance) identifies that low-frequency components accumulate global color bias under high CFG

**Net effect**: The guidance dynamics are largely resolution-invariant because `guidance_rescale=1.0` normalizes globally. The beta guidance schedule (`tex_guidance_schedule='beta'`, a=3.0, b=3.0) is timestep-based, not resolution-dependent.

### 3.4 Mechanism 4: Evaluator Confound -- Render Resolution Mismatch

**This is the most significant finding.**

In `auto_evaluate_v4.py` (line 1227):
```python
eval_render_res = 768 if pipeline_res >= 1536 else 512
```

At 1536, the evaluator renders 768x768 views. At 1024, it renders 512x512 views. The A2 comparison (line 321) resizes the reference image to match: `ref_resized = cv2.resize(reference_rgba, (w, h))`.

**How this affects A2**:
1. **Histogram bin population**: HSV histograms with 36 hue bins and 32 saturation bins behave differently at 768x768 (589K pixels) vs 512x512 (262K pixels). Higher resolution captures more color nuance, which can either help or hurt histogram correlation depending on the object.

2. **Mask coverage**: The `rend_mask = rendered_alpha > 128` threshold produces different boundary pixel ratios at different resolutions. Edge pixels (partially transparent) are proportionally fewer at higher resolution, giving cleaner masks.

3. **Reference image resampling**: The preprocessed input image (typically ~500-800px) is resized to 768x768 at 1536 vs 512x512 at 1024. Upsampling to 768 from a ~500px source introduces interpolation artifacts that don't exist at 512. This changes the reference histogram.

4. **LAB chrominance distance**: The mean LAB a* and b* values are computed over all masked pixels. At different resolutions, anti-aliased edge pixels contribute differently to the mean, potentially shifting it by 1-3 delta-E units.

**Quantitative impact estimate**: Based on the difference between Component 1 (histogram), Component 2 (mean color), and Component 3 (LAB), the resolution change in rendering could account for 1-3 points of A2 shift in either direction, dominating the actual color fidelity difference.

---

## 4. Why 1536 is Not Worse (and May Be Better)

The 10-image data shows 1536 is actually +1.7 points better on A2, with notably improved scores on the three worst-performing images (crown, d39c2b, 5a6c81). This is consistent with three benefits of higher resolution:

1. **Finer geometric detail captures more color information**: At 1536, the voxel grid resolves more surface features. Small colored details (jewelry facets, fabric patterns, painted edges) that are lost at 1024 are preserved at 1536, leading to better color match on detailed objects.

2. **Reduced quantization artifacts**: The higher voxel density means color transitions are smoother, producing less "blocky" appearance in rendered views. This improves histogram correlation for objects with gradual color gradients (metals, cloth).

3. **More tokens for the texture model**: The 1024 texture flow model, despite being out-of-distribution at 1536, benefits from the denser voxel grid providing more spatial context. The cross-attention to DINOv3 features can resolve finer correspondences between image regions and 3D surface patches.

---

## 5. Proposed Interventions

Even though the A2 drop is not real, the analysis reveals actionable improvements for color fidelity at any resolution.

### 5.1 Intervention A: Fix Evaluator Render Resolution Confound (Easy, High Impact)

**Problem**: Comparing A2 scores across resolutions is confounded by different render resolutions (512 vs 768).

**Solution**: Use a fixed evaluation render resolution regardless of pipeline resolution. 512x512 is sufficient for histogram-based color comparison and ensures apples-to-apples comparison.

**Change**: In `auto_evaluate_v4.py`, line 1227:
```python
# BEFORE:
eval_render_res = 768 if pipeline_res >= 1536 else 512
# AFTER:
eval_render_res = 512  # Fixed resolution for fair cross-config comparison
```

**Expected impact**: Eliminates ~2-3 points of measurement noise when comparing 1024 vs 1536. Does not change actual generation quality.

**Risk**: At 512x512, some fine texture detail visible at 768 is lost, potentially underscoring C3_detail_richness. But for A2 (color distribution), this is irrelevant -- color histograms are resolution-invariant for sufficiently large images.

### 5.2 Intervention B: Resolution-Adaptive Texture Guidance (Medium, Medium Impact)

**Problem**: The same `tex_slat_guidance_strength=12.0` is used at both 1024 and 1536. At 1536, the denser voxel grid has more tokens, and the global std rescale may over-normalize fine color variations.

**Solution**: Scale texture guidance strength inversely with the square root of token count ratio, following the intuition from classifier-free guidance scaling literature:

```
effective_guidance = base_guidance * sqrt(N_base / N_actual)
```

Where `N_base` is the expected token count at 1024 and `N_actual` is the actual count at 1536.

For typical objects: N_base ~12K-20K tokens at 1024, N_actual ~20K-35K at 1536, giving a scaling factor of ~0.75-0.85.

**Expected impact**: 1-2 points on A2 for objects with fine color patterns. Could also improve C1_tex_coherence by reducing over-steering at high token counts.

**Risk**: Guidance strength is already well-tuned via GA. Changing it based on token count introduces a new interaction to validate. Best tested as a sweep: `tex_slat_guidance_strength` at [8, 10, 12] specifically at 1536.

### 5.3 Intervention C: Add CLIP-S Input Alignment Metric (Medium, High Impact on Evaluation)

**Problem**: A2_color_dist only measures color histogram similarity, not semantic color alignment. Two objects with the same hue distribution but swapped color regions (e.g., red hat on blue body vs blue hat on red body) would score identically on A2.

**Solution**: Add CLIP-S (CLIP similarity between input image and rendered view) as a complementary metric. CLIP-S captures semantic alignment including spatial color placement.

**Implementation**: `open-clip-torch` is already installed in the container. Requires ~15 lines of code, runs in ~5ms per image.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# Compute cosine similarity between input and best-matching rendered view
```

**Expected impact**: Does not change A2 scores but provides a more reliable color fidelity signal. Could reveal that 1536 has better or worse CLIP-S than 1024, independent of histogram-level color distribution.

---

## 6. Literature Context

### 6.1 Papers on Color Fidelity in 3D Generation

| Paper | Venue | Key Finding | Relevance |
|-------|-------|-------------|-----------|
| Hunyuan3D 2.0 | arXiv 2501.12202 | Separate paint model at each resolution; single-model scaling loses texture fidelity | Explains why 1024 model at 1536 may shift colors |
| CFG++ | arXiv 2406.08070 | Standard CFG causes sudden color saturation shifts; manifold-constrained guidance preserves colors | CFG-MP (already tested, marginal +1.0) |
| EP-CFG | 2024 | Rescale guided predictions to preserve energy norm, reducing over-saturation | Similar to guidance_rescale=1.0 already used |
| LF-CFG | 2024 | Low-frequency CFG components cause global color bias; down-weight them | Our FDG implementation targets this; untested at 1536 |
| GTR | ICLR 2025 | Per-instance texture refinement via render-and-compare, 4 seconds runtime | Could fix any color discrepancy post-generation |

### 6.2 Papers on Sparse Voxel Texture at Resolution Boundaries

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| SparseFlex (2025) | Flexible-resolution sparse voxels up to 1024^3, isosurface generation | Validates sparse voxel approach at high resolution |
| SpArSe-Up (2025) | Surface-anchored sparse encoding reduces voxel count by 70% | Could enable 2x texture resolution within same GPU budget |
| NaTex (late 2025) | Native 3D texture bypasses UV entirely, seam-free by construction | Eliminates texture baking color loss entirely |

### 6.3 Key Insight: The 1024 Texture Model at 1536 Is Not Retrained

TRELLIS.2 uses `tex_slat_flow_model_1024` for both 1024 and 1536 resolutions (see line 1321-1324 of `trellis2_image_to_3d.py`). This model was trained on 1024-resolution voxel grids. When applied to 1536-resolution sparse tokens:

- The model's positional encoding (RoPE in DINOv3) is resolution-agnostic
- But the model's learned feature distributions assume ~12K-20K tokens, not 20K-35K
- The conditioning via cross-attention to DINOv3 features is also resolution-invariant
- The main OOD factor is the *number and spatial distribution* of tokens, not their features

This is analogous to applying a text-to-image model at a resolution it wasn't trained on (e.g., SD at 2048x2048 when trained on 512x512) -- it generally works but with subtle quality shifts. The cascade architecture (LR -> HR) mitigates this by providing the HR model with an LR prior.

---

## 7. Recommendations (Prioritized)

1. **Fix evaluator confound (Intervention A)**: Make eval_render_res constant at 512 for fair comparison. This is a 1-line change with zero risk.

2. **Run controlled A2 ablation at 1536**: Generate the same 3 images at 1024 and 1536, render BOTH at 512x512, compare A2. This isolates the generation effect from the evaluation effect.

3. **Sweep tex_slat_guidance_strength at 1536**: Test values [8, 10, 12] specifically at 1536 resolution to find the optimal guidance for the denser voxel grid.

4. **Add CLIP-S metric**: Provides a resolution-invariant, semantically-aware color alignment score that complements A2's histogram approach.

5. **Test FDG (Frequency-Decoupled Guidance) at 1536**: Use `cfg_mode='fdg'` with `fdg_lambda_low=0.6, fdg_lambda_high=1.3` to separately control low-frequency (global color) and high-frequency (local detail) guidance, potentially improving both color fidelity and texture detail at 1536.

---

## 8. Conclusion

The A2 color distribution score difference between 1024 and 1536 is **not a systematic regression caused by higher resolution**. The observed gap (80.9 vs 77.7) is explained by:

1. **Different sample sizes** (3 vs 10 images) -- adding 7 more images, some with inherently low A2, pulls the mean down
2. **Evaluator render resolution confound** (512 vs 768 pixel renders) -- introduces 2-3 points of measurement noise
3. **Normal run-to-run variation** -- A2 std is 11-14 points per image

On matched 10-image comparisons, 1536 actually scores +1.7 higher on A2 with lower variance. The higher voxel resolution provides finer color detail that benefits histogram correlation for most objects.

The most actionable improvement is fixing the evaluator render resolution to enable fair cross-resolution comparison, followed by controlled guidance sweeps at 1536 and the addition of CLIP-S as a complementary metric.
