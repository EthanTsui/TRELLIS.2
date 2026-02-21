# TRELLIS.2 Quality Evaluation Framework Analysis & Redesign

**Date**: 2026-02-21
**Author**: 3D Vision Expert Agent
**Scope**: Complete audit of existing scoring systems and design of a principled replacement

---

## 1. Audit of Existing Scoring Systems

TRELLIS.2 has accumulated **three different scoring systems** across optimization iterations. Each was designed to fix problems in the previous one, but none was built from first principles. This section dissects each.

### 1.1 Scoring System v1 (evaluate.py) -- GA v1 Sweep

**File**: `/workspace/TRELLIS.2/optimization/scripts/evaluate.py`

**Dimensions** (100 total):
| Dimension | Weight | Method |
|-----------|--------|--------|
| Silhouette | 20 | Binary mask IoU (rendered vs reference alpha) |
| Contour | 15 | Canny edge F1 with 5px tolerance dilation |
| Color | 15 | 60% histogram correlation + 40% MAE in masked region |
| Detail | 10 | 50% SSIM + 50% LPIPS (AlexNet) |
| Artifacts | 15 | Dark spots, speckles, components, holes |
| Smoothness | 15 | Laplacian energy, local deviation, interior edge density |
| Coherence | 10 | Crack detection, gradient harshness, dark interior |

**Score range observed**: 33-52 (avg ~37-51 across all GA experiments).

**Critical problems**:
1. **Silhouette scores are low and unreliable (12-45)**: The camera angle for the rendered front view does not match the training camera used for the input image. The `render_snapshot` function uses `offset=(-16deg, 20deg)` with `r=10, fov=8` (snapshot orthographic-like), while the pipeline generates the 3D model from a processed image that was captured at unknown viewpoint. The silhouette IoU is comparing apples to oranges -- slight rotation differences produce IoU drops of 20-40%.

2. **Detail metric (LPIPS+SSIM) measures the wrong thing**: LPIPS and SSIM compare pixel-level correspondence between a 2D reference photo and a 3D render from a specific viewpoint. A TRELLIS.2 model that perfectly captures the 3D object will NEVER achieve high SSIM because: (a) the exact viewpoint cannot be replicated, (b) the 3D render has different lighting/shading, (c) SSIM penalizes spatial shifts that are expected in 3D reconstruction. The observed detail scores (19-30) confirm this is broken.

3. **Smoothness directly contradicts Detail**: Smoothness penalizes Laplacian energy and local deviation, while Detail (via SSIM/LPIPS) rewards textural fidelity. An object with rich, correct texture (high detail) will always score poorly on smoothness. These two dimensions are anti-correlated by design.

4. **Color histogram is viewpoint-dependent**: The histogram comparison only works for the front view, but the 3D model may show different surfaces than the reference. A perfectly correct 3D model viewed from a slightly different angle will have a completely different color distribution.

5. **Artifacts score is inflated (85-100)**: Nearly all generated models score near-perfect on artifacts. This dimension provides almost zero discriminative power between configurations.

### 1.2 Scoring System v2 (genetic_optimizer.py) -- GA v2

Same dimensions as v1, same evaluator class (`evaluate.py:QualityEvaluator`), but with:
- Added mesh-level smoothness (face normal consistency, 60% mesh + 40% view).
- Added texture-level artifact detection blended 50/50 with view artifacts.
- 7 dimensions with same weights.

**Score range observed**: 31-52 (avg ~37-38). V2 was nominally "stricter" but the distribution is nearly identical because the same fundamental metrics are used.

**Champion (gen3-005)**: 37.81/100 -- the fact that the best config scores under 40% on the evaluation system strongly suggests the evaluation is measuring the wrong things, not that the generation quality is genuinely bad.

### 1.3 Scoring System v3/v3.1/v3.2 (auto_evaluate.py) -- Post-GA Focused Optimization

**File**: `/workspace/TRELLIS.2/optimization/scripts/auto_evaluate.py`

**Dimensions** (100 total):
| Dimension | Weight | Method |
|-----------|--------|--------|
| Shape | 15 | Binary mask IoU (only front view vs reference alpha) |
| Color Match | 10 | MAE in overlapping masked region |
| Detail | 10 | Laplacian energy of rendered view (no reference comparison) |
| Fragmentation | 20 | Crack detection + gradient harshness + local color variance |
| Smoothness | 15 | Laplacian + local deviation + edge density (conflict with detail!) |
| Darkness | 15 | Dark pixel ratios + mean brightness + dark patches |
| Vibrancy | 15 | HSV saturation + value + contrast |

**Score range observed**: 66-70 (avg ~68.5 for champion config). Dramatically different from v1/v2 because the metrics changed completely.

**Improvements over v1/v2**:
- Detail is now intrinsic (Laplacian energy, no reference comparison) -- correct for non-view-matched evaluation.
- Added fragmentation (20pts) -- the single most discriminative metric, correctly identifying the main quality problem.
- Added texture-map-level evaluation (darkness_tex, vibrancy_tex, fragmentation_tex) blended 50/50 with view scores.

**Remaining problems**:
1. **Smoothness-Detail conflict persists**: Smoothness penalizes Laplacian energy >10; Detail rewards Laplacian energy in 8-20 range. A base_color render with legitimate wood grain texture at Laplacian energy 12 loses ~4 smoothness points while gaining detail points.

2. **Vibrancy is near-perfect (99-100) and non-discriminative**: All tested configurations score 99-100 on vibrancy. This 15-point dimension provides zero guidance for optimization. It only penalizes aggressively desaturated outputs, which our pipeline does not produce.

3. **Darkness is high (80-90) and low-variance**: Most configs score 80-90, providing ~1pt discriminative range. This 15-point dimension is effectively wasted on measuring something that rarely varies.

4. **Shape IoU is fundamentally misaligned**: The front rendered view is not viewpoint-matched to the input image. Silhouette IoU of 50-67% includes both genuine shape errors and viewpoint mismatch artifacts.

5. **Texture map evaluation uses a crude mask**: `np.any(tex_rgb > 5)` as the valid texel mask will include UV padding regions that have been inpainted, penalizing the model for background texels.

6. **fragmentation_tex uses global Sobel**: This penalizes genuine texture detail (e.g., brick walls, fabric patterns) as "harsh boundaries."

7. **3 test images is too few**: The GA used 2-3 examples. Variance across runs is high (~3-5 points). We cannot distinguish a real +1pt improvement from noise.

---

## 2. Root Cause Analysis: Why Current Metrics Fail

### 2.1 The Fundamental Misalignment Problem

All three scoring systems share a foundational flaw: **they try to measure 3D quality through 2D proxies without controlling for the 3D-to-2D projection**.

When we render a 3D model and compare to the input image:
- The exact camera pose is unknown (TRELLIS.2 takes an arbitrary image, not a calibrated view)
- The lighting model differs (ambient + envmap vs. original scene lighting)
- Material appearance differs (trilinear-sampled voxels vs. original surface)
- Background/context is removed

These differences are *expected* and *correct* -- they are consequences of the 2D-to-3D lifting, not quality defects. But our metrics penalize them heavily.

### 2.2 The Metric Conflict Problem

Two kinds of qualities are fundamentally in tension:
1. **Smoothness**: Wants uniform, noise-free surfaces (Laplacian energy = 0)
2. **Detail richness**: Wants high-frequency content (Laplacian energy > 0)

A marble statue should score high on smoothness. A textured character should score high on detail. The same metric cannot measure both correctly without knowing what the object IS. Our current system applies both penalties to every object equally.

### 2.3 The Discriminability Problem

A good evaluation metric must satisfy two criteria:
1. **Validity**: Higher scores correlate with human-perceived better quality
2. **Discriminability**: Different configurations produce meaningfully different scores

Current metric behavior (from the v3.2 baseline evaluation):
| Metric | Score | Std across images | Discriminative? |
|--------|-------|-------------------|-----------------|
| Shape | 58.0 | 8.5 | Moderate |
| Color | 43.1 | 9.5 | Moderate |
| Detail | 99.8 | 0.2 | **No** (ceiling) |
| Fragmentation | 65.7 | 13.1 | **Yes** (best) |
| Smoothness | 28.8 | 2.5 | Low (floor) |
| Darkness | 87.4 | 5.3 | Low |
| Vibrancy | 99.8 | 0.2 | **No** (ceiling) |

Only **Fragmentation** and **Shape** provide meaningful discrimination. 75 of 100 points are invested in dimensions that are either at ceiling (Detail, Vibrancy), at floor (Smoothness), or have low variance (Darkness).

### 2.4 The Test Set Problem

Using 2-3 test images creates severe overfitting risk. The GA optimized for these specific images. A configuration that improves score by 1.0 on 3 images is likely noise. Statistical significance requires at minimum 10-15 diverse test cases to detect a 2-point improvement at p<0.05.

---

## 3. Proposed Evaluation Framework v4

### 3.1 Design Principles

1. **Separate intrinsic vs. extrinsic quality**: Intrinsic metrics evaluate the 3D asset in isolation (is it a well-formed mesh with coherent textures?). Extrinsic metrics compare to the input (does it match?). These should be scored separately.

2. **No metric conflicts**: Each quality dimension must be independent. Smoothness and detail must not penalize each other.

3. **Discriminability first**: Every metric must demonstrate variance across configurations. Non-discriminative metrics should be dropped or reweighted to near-zero.

4. **Multi-level evaluation**: Evaluate at three levels: (a) mesh geometry, (b) texture maps, (c) rendered views. Each level captures different defects.

5. **Sufficient test diversity**: Minimum 8 test images spanning categories: organic (character, animal), hard-surface (mechanical, architecture), mixed (furniture, wearable), and edge cases (thin structures, translucent).

### 3.2 Proposed Dimensions

#### Category A: Input-Output Consistency (25 points total)

These metrics measure how well the 3D model matches the input image. They are inherently limited by viewpoint ambiguity but still valuable for detecting gross errors.

**A1. Silhouette Match (15 points)**

*What it measures*: Does the 3D model's outline match the input image's outline?

*Current problem*: Camera mismatch between render_snapshot and input viewpoint.

*Fix*: Render at the SAME camera parameters the pipeline uses internally. The pipeline preprocesses images and generates from a canonical front view. We should render back at the canonical camera (yaw=0, pitch=0 or the pipeline's assumed viewpoint).

```python
def silhouette_match(mesh, reference_rgba, resolution=512):
    """
    Render mesh at canonical front view and compare silhouette.
    Uses the same camera params as pipeline training data.
    """
    # Pipeline canonical camera: front view, yaw=0, pitch=0
    yaw = [0.0]
    pitch = [0.0]
    r = 2.0
    fov = 40.0  # Training FOV is 30, render_video uses 40

    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    result = render_frames(mesh, extr, intr, {'resolution': resolution, 'bg_color': (0,0,0)})

    rend_alpha = result['alpha'][0]  # Use actual alpha channel, not gray>5
    ref_alpha = (reference_rgba[:, :, 3] > 128).astype(np.float32)

    # Resize to common resolution
    if ref_alpha.shape != rend_alpha.shape:
        ref_alpha = cv2.resize(ref_alpha, (resolution, resolution))

    rend_mask = (rend_alpha > 0.5).astype(np.float32)

    intersection = (rend_mask * ref_alpha).sum()
    union = np.clip(rend_mask + ref_alpha, 0, 1).sum()
    iou = intersection / max(union, 1)

    return iou * 100  # 0-100
```

*Expected range*: 50-90. Good: >75. Bad: <50.
*Weight*: 15 points. Critical for ensuring the model represents the input subject.

**A2. Color Distribution Match (10 points)**

*What it measures*: Are the dominant colors of the 3D model consistent with the input?

*Current problem*: Pixel-level MAE is too sensitive to viewpoint and lighting differences.

*Fix*: Compare color histograms in LAB space (perceptually uniform) over the masked region. Use Earth Mover's Distance (Wasserstein) instead of histogram correlation, as it is more robust to small shifts.

```python
def color_distribution_match(rendered_rgb, reference_rgb, rendered_mask, reference_mask):
    """
    Compare color distributions in LAB space using histogram correlation.
    """
    both_mask = (rendered_mask > 0.5) & (reference_mask > 0.5)
    if both_mask.sum() < 100:
        return 50.0

    # Convert to LAB (perceptually uniform)
    rend_lab = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB)

    # Compare histograms per channel in LAB
    scores = []
    for c in range(3):
        bins = 32
        rend_hist = cv2.calcHist([rend_lab], [c], both_mask.astype(np.uint8), [bins], [0, 256])
        ref_hist = cv2.calcHist([ref_lab], [c], both_mask.astype(np.uint8), [bins], [0, 256])
        cv2.normalize(rend_hist, rend_hist)
        cv2.normalize(ref_hist, ref_hist)

        # Correlation (robust to small shifts)
        corr = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_CORREL)
        scores.append(max(0, corr))

    # Weight L channel less (lighting differences are expected)
    weighted = 0.3 * scores[0] + 0.35 * scores[1] + 0.35 * scores[2]
    return weighted * 100
```

*Expected range*: 30-85. Good: >65. Bad: <40.
*Weight*: 10 points.


#### Category B: Geometric Quality (20 points total)

These metrics evaluate the mesh geometry independent of any reference image.

**B1. Mesh Integrity (10 points)**

*What it measures*: Is the mesh well-formed? Watertight, no degenerate faces, no floating fragments, reasonable face count.

*Implementation*: Uses trimesh built-in diagnostics.

```python
def mesh_integrity_score(glb_path):
    """
    Evaluate mesh structural quality from GLB file.
    Uses trimesh for mesh diagnostics.
    """
    import trimesh
    scene = trimesh.load(glb_path)

    # Extract main mesh from scene
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
        mesh = meshes[0] if meshes else None
    else:
        mesh = scene

    if mesh is None:
        return 0.0

    score = 100.0

    # 1. Watertight check (-20 if not)
    if not mesh.is_watertight:
        score -= 20

    # 2. Degenerate face ratio (-25 max)
    face_areas = mesh.area_faces
    zero_area = (face_areas < 1e-10).sum()
    degen_ratio = zero_area / max(len(face_areas), 1)
    if degen_ratio > 0.001:
        score -= min(25, degen_ratio * 5000)

    # 3. Connected components (-20 max)
    # More than 1 connected component = floating fragments
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Check if secondary components are significant (>1% of main)
        main_area = max(c.area for c in components)
        fragments = sum(1 for c in components if c.area > 0.01 * main_area)
        if fragments > 1:
            score -= min(20, (fragments - 1) * 10)

    # 4. Face aspect ratio (-15 max)
    # Very elongated triangles indicate poor meshing
    # Use edge length ratio as proxy
    edges = mesh.edges_unique_length
    if len(edges) > 0:
        p95 = np.percentile(edges, 95)
        p5 = np.percentile(edges, 5)
        if p5 > 1e-8:
            aspect = p95 / p5
            if aspect > 100:
                score -= min(15, (aspect - 100) * 0.1)

    # 5. Normal consistency (-10 max)
    # Check that face normals are consistent (no flipped faces)
    if hasattr(mesh, 'face_adjacency') and len(mesh.face_adjacency) > 0:
        adj_dots = mesh.face_adjacency_angles
        flipped_ratio = (adj_dots > np.pi * 0.9).sum() / max(len(adj_dots), 1)
        if flipped_ratio > 0.01:
            score -= min(10, flipped_ratio * 500)

    return max(0, score)
```

*Expected range*: 60-100. Good: >85. Bad: <60.
*Weight*: 10 points.

**B2. Surface Quality (10 points)**

*What it measures*: Is the surface geometrically smooth where it should be, and sharp where it should be? This replaces the conflicted "smoothness" metric.

*Key insight*: Instead of penalizing all high-frequency content, we evaluate **geometric** smoothness from the mesh normals rendered as a normal map. Geometric noise (bumpy surface) shows up in the normal map even when the base_color is smooth. Conversely, rich texture detail does NOT affect the normal map.

```python
def surface_quality_score(mesh, resolution=512, nviews=4):
    """
    Evaluate geometric surface quality from rendered normal maps.
    Normal maps capture geometry-only roughness, independent of texture.
    """
    yaw = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    pitch = [0.3] * nviews

    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2.0, 40.0)
    result = render_frames(mesh, extr, intr, {'resolution': resolution, 'bg_color': (0,0,0)})

    scores = []
    for i in range(nviews):
        normal_img = result['normal'][i]  # (H, W, 3) uint8
        alpha = result['alpha'][i]
        mask = alpha > 128

        if mask.sum() < 100:
            scores.append(50.0)
            continue

        # Erode mask to exclude silhouette edges
        kernel = np.ones((10, 10), np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), kernel) > 0

        if interior.sum() < 100:
            scores.append(50.0)
            continue

        # Compute normal map Laplacian (measures geometric roughness)
        normal_float = normal_img.astype(np.float32) / 255.0

        lap_energy = 0
        for c in range(3):
            lap = cv2.Laplacian(normal_float[:, :, c], cv2.CV_64F)
            lap_energy += np.abs(lap[interior]).mean()
        lap_energy /= 3.0

        # Score: low Laplacian energy = smooth surface
        # Scale: 0.0-0.02 = excellent, 0.02-0.05 = good, 0.05-0.1 = acceptable, >0.1 = rough
        s = 100.0
        if lap_energy > 0.02:
            s -= min(40, (lap_energy - 0.02) * 500)
        if lap_energy > 0.05:
            s -= min(30, (lap_energy - 0.05) * 300)

        scores.append(max(0, s))

    return float(np.mean(scores))
```

*Expected range*: 40-95. Good: >70. Bad: <40.
*Weight*: 10 points.

*Why this solves the conflict*: Normal maps contain ONLY geometric information. A textured object (wood grain, fabric weave) will have smooth normals and score high. A geometrically noisy object (noisy SDF reconstruction) will have chaotic normals and score low. This correctly separates geometry quality from texture richness.


#### Category C: Texture Quality (35 points total)

These are the most impactful metrics, as texture is the primary quality differentiator between configurations.

**C1. Texture Coherence / Anti-Fragmentation (15 points)**

*What it measures*: Is the texture continuous and free from patchwork, UV seams, and color discontinuities?

*Current state*: This is already the most discriminative metric (65.7 baseline, 13.1 std). Keep the approach but improve it.

*Improvements over v3*:
- Use the alpha channel (not `gray > 5`) for masking
- Apply evaluation to the normal-space-aware interior (not just eroded silhouette)
- Reduce sensitivity to legitimate texture edges by using adaptive thresholds based on the image's overall contrast

```python
def texture_coherence_score(rendered_views, alpha_views, nviews=8):
    """
    Evaluate texture coherence across multiple rendered views.
    Detects patchwork, UV seams, and harsh discontinuities.
    """
    scores = []

    for i in range(min(nviews, len(rendered_views))):
        rgb = rendered_views[i]
        mask = alpha_views[i] > 128
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Interior mask
        kernel = np.ones((10, 10), np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), kernel) > 0

        if interior.sum() < 200:
            scores.append(50.0)
            continue

        s = 100.0

        # 1. Morphological crack detection (thin dark gaps)
        gray_int = gray.copy()
        gray_int[~interior] = 128
        closed = cv2.morphologyEx(gray_int, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        crack_diff = closed.astype(np.float32) - gray_int.astype(np.float32)
        crack_ratio = ((crack_diff > 20) & interior).sum() / max(interior.sum(), 1)
        if crack_ratio > 0.003:
            s -= min(35, (crack_ratio - 0.003) * 600)

        # 2. Gradient harshness in COLOR space (not just grayscale)
        # Harsh color transitions = patchwork from bad multi-view blending
        rgb_f = rgb.astype(np.float32)
        grad_mag = np.zeros(gray.shape, dtype=np.float32)
        for c in range(3):
            gx = cv2.Sobel(rgb_f[:,:,c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(rgb_f[:,:,c], cv2.CV_32F, 0, 1, ksize=3)
            grad_mag += np.sqrt(gx**2 + gy**2)
        grad_mag /= 3.0

        # Adaptive threshold: based on image contrast
        p90 = np.percentile(grad_mag[interior], 90)
        threshold = max(35, p90 * 0.8)  # at least 35, or 80th pct of gradient dist
        harsh = (grad_mag[interior] > threshold).sum() / max(interior.sum(), 1)
        if harsh > 0.08:
            s -= min(30, (harsh - 0.08) * 200)

        # 3. Local color variance (patchwork detection)
        local_var = np.zeros(gray.shape, dtype=np.float32)
        for c in range(3):
            blurred = cv2.GaussianBlur(rgb_f[:,:,c], (15,15), 3.0)
            local_var += (rgb_f[:,:,c] - blurred) ** 2
        local_var = np.sqrt(local_var / 3.0)
        high_var_ratio = (local_var[interior] > 30).sum() / max(interior.sum(), 1)
        if high_var_ratio > 0.15:
            s -= min(25, (high_var_ratio - 0.15) * 120)

        scores.append(max(0, s))

    return float(np.mean(scores))
```

*Expected range*: 30-90. Good: >75. Bad: <50.
*Weight*: 15 points.

**C2. Color Vitality (10 points)**

*What it measures*: Combined metric replacing the non-discriminative Vibrancy (ceiling at 99) and mostly-non-discriminative Darkness (floor at 80). Instead of measuring absolute saturation (which is object-dependent -- a white teapot SHOULD be low saturation), measure relative color health.

```python
def color_vitality_score(rendered_views, alpha_views, nviews=8):
    """
    Evaluate color health: no grey patches, no washed-out areas,
    adequate dynamic range, no metallic artifacts.
    """
    scores = []

    for i in range(min(nviews, len(rendered_views))):
        rgb = rendered_views[i]
        mask = alpha_views[i] > 128

        if mask.sum() < 100:
            scores.append(50.0)
            continue

        s = 100.0

        masked_rgb = rgb[mask].astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[mask].astype(np.float32)

        # 1. Grey patch detection (-30 max)
        # Grey patches: low saturation AND mid-luminance (not white, not black)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:,:,1][mask].astype(np.float32)
        val = hsv[:,:,2][mask].astype(np.float32)
        # Grey = sat<15 AND 40<val<200 (excluding intentionally dark/bright)
        grey_mask = (sat < 15) & (val > 40) & (val < 200)
        grey_ratio = grey_mask.sum() / max(mask.sum(), 1)
        if grey_ratio > 0.15:
            s -= min(30, (grey_ratio - 0.15) * 100)

        # 2. Dynamic range (-25 max)
        p5, p95 = np.percentile(gray, [5, 95])
        dyn_range = p95 - p5
        if dyn_range < 40:
            s -= min(25, (40 - dyn_range) * 0.8)

        # 3. Dark patch penalty (-25 max)
        # Only penalize UNEXPECTED darkness (patches, not overall dark objects)
        dark_ratio = (gray < 10).sum() / max(gray.size, 1)
        if dark_ratio > 0.02:
            s -= min(25, (dark_ratio - 0.02) * 500)

        # 4. Color channel variance (-20 max)
        # If all channels are nearly identical -> grey/monochrome artifact
        r_mean = masked_rgb[:, 0].mean()
        g_mean = masked_rgb[:, 1].mean()
        b_mean = masked_rgb[:, 2].mean()
        channel_std = np.std([r_mean, g_mean, b_mean])
        # Very low channel variance = grey/metallic artifact (unless object is neutral)
        # Only penalize if VERY low -- many objects are legitimately neutral-toned
        if channel_std < 3:
            s -= min(15, (3 - channel_std) * 5)

        scores.append(max(0, s))

    return float(np.mean(scores))
```

*Expected range*: 50-95. Good: >80. Bad: <50.
*Weight*: 10 points.

**C3. Texture Detail Richness (10 points)**

*What it measures*: Does the texture contain meaningful detail, or is it blurry/over-smoothed?

*Key change from v3*: Use base_color renders ONLY (not shaded), and measure detail in the UV texture map directly in addition to rendered views.

```python
def texture_detail_score(rendered_base_colors, alpha_views, texture_map=None, tex_mask=None, nviews=8):
    """
    Measure intrinsic texture detail richness.
    Uses Laplacian energy and local entropy.
    """
    # View-based detail
    view_scores = []
    for i in range(min(nviews, len(rendered_base_colors))):
        bc = rendered_base_colors[i]
        mask = alpha_views[i] > 128
        gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)

        interior_kern = np.ones((8,8), np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), interior_kern) > 0

        if interior.sum() < 200:
            view_scores.append(50.0)
            continue

        # Laplacian energy (texture detail proxy)
        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        lap_energy = np.abs(lap[interior]).mean()

        # Score mapping:
        # < 2: very blurry (20pts)
        # 2-5: low detail (20-60pts)
        # 5-15: good detail (60-90pts)
        # 15-25: rich detail (90-100pts)
        # > 25: potentially noisy (cap at 90, slight penalty)
        if lap_energy < 2:
            s = 20.0
        elif lap_energy < 5:
            s = 20 + (lap_energy - 2) * 13.3
        elif lap_energy < 15:
            s = 60 + (lap_energy - 5) * 3.0
        elif lap_energy < 25:
            s = 90 + (lap_energy - 15) * 1.0
        else:
            s = max(85, 100 - (lap_energy - 25) * 0.5)

        view_scores.append(s)

    view_detail = float(np.mean(view_scores))

    # Texture-map detail (if available)
    if texture_map is not None and tex_mask is not None:
        mask_bool = tex_mask.astype(bool)
        if mask_bool.sum() > 1000:
            tex_gray = cv2.cvtColor(texture_map, cv2.COLOR_RGB2GRAY)
            tex_lap = cv2.Laplacian(tex_gray.astype(np.float64), cv2.CV_64F)
            tex_lap_energy = np.abs(tex_lap[mask_bool]).mean()

            if tex_lap_energy < 3:
                tex_score = 30.0
            elif tex_lap_energy < 10:
                tex_score = 30 + (tex_lap_energy - 3) * 10
            else:
                tex_score = min(100, 100 + (tex_lap_energy - 10) * 0)

            # Blend: 60% view-based, 40% texture-map-based
            return 0.6 * view_detail + 0.4 * tex_score

    return view_detail
```

*Expected range*: 30-95. Good: >70. Bad: <40.
*Weight*: 10 points.


#### Category D: PBR Material Quality (10 points total)

**D1. Material Plausibility (10 points)**

*What it measures*: Are the PBR material properties (metallic, roughness) in physically reasonable ranges?

*Why it matters*: Incorrect metallic values produce the characteristic "chrome" artifact. Incorrect roughness produces either mirror-like or completely matte surfaces that look unnatural.

```python
def material_plausibility_score(rendered_metallic_views, rendered_roughness_views, alpha_views, nviews=4):
    """
    Evaluate PBR material plausibility.
    Most real-world objects are non-metallic with moderate roughness.
    """
    scores = []

    for i in range(min(nviews, len(rendered_metallic_views))):
        mask = alpha_views[i] > 128
        if mask.sum() < 100:
            scores.append(50.0)
            continue

        met = rendered_metallic_views[i].astype(np.float32)
        if met.ndim == 3:
            met = met[:,:,0]  # single channel
        met = met[mask] / 255.0

        rough = rendered_roughness_views[i].astype(np.float32)
        if rough.ndim == 3:
            rough = rough[:,:,0]
        rough = rough[mask] / 255.0

        s = 100.0

        # 1. High metallic penalty (-30 max)
        # Most generated objects should not be highly metallic
        # (pipeline already clamps to max_metallic=0.05)
        high_met_ratio = (met > 0.3).sum() / max(met.size, 1)
        if high_met_ratio > 0.05:
            s -= min(30, (high_met_ratio - 0.05) * 200)

        # 2. Extreme roughness penalty (-20 max)
        # Very low roughness (<0.1) = mirror, very high (>0.95) = completely matte
        mirror_ratio = (rough < 0.1).sum() / max(rough.size, 1)
        if mirror_ratio > 0.1:
            s -= min(20, (mirror_ratio - 0.1) * 100)

        # 3. Roughness uniformity bonus (+10 max)
        # Most objects have reasonably consistent roughness
        rough_std = rough.std()
        if rough_std < 0.15:
            s = min(100, s + 5)  # mild bonus for plausible roughness

        # 4. Metallic-roughness correlation check (-10 max)
        # Metallic areas should typically have low roughness (polished metal)
        # Non-metallic areas can have any roughness
        if high_met_ratio > 0.01:
            met_areas = met > 0.3
            if met_areas.sum() > 10:
                met_rough = rough[met_areas].mean()
                if met_rough > 0.7:
                    s -= 10  # metallic + rough = implausible

        scores.append(max(0, s))

    return float(np.mean(scores))
```

*Expected range*: 70-100 (most models should score well here since we already clamp metallic).
*Weight*: 10 points.


#### Category E: Multi-View Consistency (10 points total)

**E1. Cross-View Consistency (10 points)**

*What it measures*: Does the model look consistent when viewed from different angles? This detects view-dependent artifacts, texture stretching on back faces, and incomplete back-side generation.

```python
def multiview_consistency_score(rendered_views, alpha_views, nviews=8):
    """
    Evaluate consistency of quality across different viewpoints.
    Measures variance in per-view quality metrics -- high variance =
    view-dependent artifacts.
    """
    if len(rendered_views) < 2:
        return 50.0

    # Per-view brightness distribution
    brightness_means = []
    brightness_stds = []
    saturation_means = []
    detail_energies = []

    for i in range(min(nviews, len(rendered_views))):
        mask = alpha_views[i] > 128
        if mask.sum() < 100:
            continue

        gray = cv2.cvtColor(rendered_views[i], cv2.COLOR_RGB2GRAY)
        masked_gray = gray[mask].astype(np.float32)
        brightness_means.append(masked_gray.mean())
        brightness_stds.append(masked_gray.std())

        hsv = cv2.cvtColor(rendered_views[i], cv2.COLOR_RGB2HSV)
        sat = hsv[:,:,1][mask].astype(np.float32)
        saturation_means.append(sat.mean())

        lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
        detail_energies.append(np.abs(lap[mask]).mean())

    if len(brightness_means) < 2:
        return 50.0

    score = 100.0

    # 1. Brightness consistency (-30 max)
    # Large variation = some views are too dark (failed back-face texturing)
    bright_cv = np.std(brightness_means) / max(np.mean(brightness_means), 1)
    if bright_cv > 0.15:
        score -= min(30, (bright_cv - 0.15) * 150)

    # 2. Saturation consistency (-25 max)
    sat_cv = np.std(saturation_means) / max(np.mean(saturation_means), 1)
    if sat_cv > 0.2:
        score -= min(25, (sat_cv - 0.2) * 100)

    # 3. Detail consistency (-25 max)
    detail_cv = np.std(detail_energies) / max(np.mean(detail_energies), 1)
    if detail_cv > 0.3:
        score -= min(25, (detail_cv - 0.3) * 80)

    # 4. Coverage consistency (-20 max)
    # All views should have similar pixel coverage (object fills frame consistently)
    coverages = [alpha_views[i].sum() / alpha_views[i].size for i in range(min(nviews, len(alpha_views)))]
    cov_cv = np.std(coverages) / max(np.mean(coverages), 1e-6)
    if cov_cv > 0.3:
        score -= min(20, (cov_cv - 0.3) * 60)

    return max(0, score)
```

*Expected range*: 40-90. Good: >75. Bad: <50.
*Weight*: 10 points.


### 3.3 Summary of Proposed Metrics

| Category | Dimension | Weight | Key Method | Discriminative? |
|----------|-----------|--------|------------|-----------------|
| A. I/O Match | A1. Silhouette Match | 15 | Alpha IoU at canonical camera | High |
| A. I/O Match | A2. Color Distribution | 10 | LAB histogram correlation | Medium |
| B. Geometry | B1. Mesh Integrity | 10 | Trimesh diagnostics (watertight, degen faces) | Medium |
| B. Geometry | B2. Surface Quality | 10 | Normal map Laplacian energy | High |
| C. Texture | C1. Texture Coherence | 15 | Crack detection + gradient harshness | **Highest** |
| C. Texture | C2. Color Vitality | 10 | Grey patch, dynamic range, dark patch | Medium |
| C. Texture | C3. Detail Richness | 10 | Base_color Laplacian energy | Medium |
| D. Material | D1. Material Plausibility | 10 | Metallic/roughness ranges + correlations | Low-Med |
| E. Consistency | E1. Multi-View Consistency | 10 | CV of per-view quality metrics | High |
| **Total** | | **100** | | |

### 3.4 Weight Justification

The weighting reflects the priority order for user-perceived quality:

1. **Texture quality (35 pts)**: This is the PRIMARY differentiator between TRELLIS.2 configurations. Texture coherence (fragmentation) has been identified as the most discriminative metric and the most impactful quality defect. Color vitality catches the grey/metallic artifact problem. Detail richness catches over-smoothing.

2. **Input-Output Consistency (25 pts)**: Users expect the 3D model to look like their input image. Silhouette match is the most reliable comparison metric given viewpoint uncertainty.

3. **Geometric Quality (20 pts)**: A geometrically sound mesh is necessary for downstream use (3D printing, game engines, AR). Surface quality from normal maps is a clean way to measure geometry without conflicting with texture.

4. **Multi-View Consistency (10 pts)**: Important for 3D assets but less critical than front-view quality for most use cases.

5. **Material Plausibility (10 pts)**: PBR correctness matters for rendering but is already largely handled by the postprocess pipeline (metallic clamping, roughness floor).

---

## 4. Resolved Metric Conflicts

| Conflict in v1-v3 | Resolution in v4 |
|--------------------|------------------|
| Smoothness penalizes texture detail | B2 uses NORMAL maps (geometry only), C3 measures texture detail separately |
| Detail (SSIM/LPIPS) requires viewpoint match | A1 uses IoU (viewpoint-robust), C3 uses intrinsic Laplacian |
| Vibrancy non-discriminative (ceiling) | Replaced by C2 Color Vitality with grey-patch and dynamic range focus |
| Darkness non-discriminative (floor) | Merged into C2 as dark-patch sub-metric |
| Artifacts non-discriminative (ceiling) | Split into B1 (mesh integrity) and C1 (texture coherence) |

---

## 5. Implementation Recommendations

### 5.1 Test Image Set

Increase from 3 to minimum 8 images:

```python
TEST_IMAGES = [
    # Organic / characters
    'assets/example_image/T.png',                    # Steampunk device (metallic, complex)
    'assets/example_image/0a34...webp',              # Ornate crown (gold, detailed)
    'assets/example_image/454e...webp',              # Current diverse example
    # Hard-surface / mechanical
    # Add: shoe, car part, or architectural element
    # Organic / natural
    # Add: plant, animal, or food item
    # Mixed / wearable
    # Add: backpack, watch, or furniture
    # Edge cases
    # Add: thin structure (eyeglasses, tree)
    # Add: translucent (glass, ice)
]
```

Recommendation: Use at least 8 images spanning 4+ categories. Compute 95% confidence intervals on the overall score to determine if an improvement is statistically significant.

### 5.2 Rendering Configuration

Use PBR mesh renderer with explicit channel extraction:

```python
def render_evaluation_views(mesh, nviews=8, resolution=512):
    """Render all channels needed for evaluation."""
    yaw = np.linspace(0, 2*np.pi, nviews, endpoint=False)
    pitch = [0.25] * nviews

    envmap = get_envmap()
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2.0, 40.0)
    result = render_frames(mesh, extr, intr, {'resolution': resolution}, envmap=envmap)

    return {
        'base_color': result['base_color'],   # For C1, C2, C3
        'normal': result['normal'],            # For B2
        'metallic': result['metallic'],        # For D1
        'roughness': result['roughness'],      # For D1
        'alpha': result['alpha'],              # For all (masking)
        'shaded': result['shaded'],            # For visual inspection only
    }
```

### 5.3 Score Calibration

Before deploying v4, run the champion config on all test images and establish the baseline distribution. Adjust thresholds so that:
- The champion config scores 65-75/100 overall (room for improvement above, room for detection below)
- No single dimension is at ceiling (>95) or floor (<20) for the champion config
- Each dimension has at least 5 points of variance across the test set

### 5.4 Statistical Significance

For a test set of N images:
```
Standard error = sigma / sqrt(N)
95% CI = mean +/- 1.96 * SE
```

For N=3, sigma=5: SE=2.9, CI width=11.4 -- cannot detect 2-point improvements.
For N=8, sigma=5: SE=1.8, CI width=7.0 -- can detect 5-point improvements.
For N=15, sigma=5: SE=1.3, CI width=5.1 -- can detect 3-point improvements.

Recommendation: **N=10-15 images for configuration optimization, N=3-5 for quick smoke tests**.

---

## 6. Metrics We Explicitly Do NOT Include

### 6.1 LPIPS / SSIM (Removed)

These metrics require pixel-aligned reference-to-render correspondence. Since we cannot recover the exact input camera pose, LPIPS and SSIM will always be dominated by viewpoint mismatch noise. They provide misleading optimization signal.

### 6.2 FID / KID (Not applicable)

These are distributional metrics requiring hundreds of samples. We are evaluating individual outputs. Not applicable to our use case.

### 6.3 CLIP Score (Deferred)

CLIP-based image quality assessment (e.g., CLIP-IQA, QualiCLIP) could provide a human-aligned no-reference quality score. However:
- Adds a ~400MB model dependency (CLIP ViT-L)
- Inference time: ~100ms per view
- Not calibrated for 3D renders specifically
- Risk of CLIP latent space bias toward photographic images

**Recommendation**: Worth exploring as a supplementary "perceptual quality" metric (5-10 points), but should not replace the principled metrics above. Can be added as Category F in a future iteration after calibration against human ratings.

### 6.4 Gen3DEval / VLM-based Scoring (Future)

Using a vision-language model (GPT-4V, Claude) to score rendered views is promising but:
- Requires API calls (cost, latency)
- Non-deterministic (temperature-dependent)
- Not reproducible across model versions

**Recommendation**: Use VLM scoring for periodic validation of the automated metrics, not as a primary scoring channel.

---

## 7. Migration Path

### Phase 1: Implement v4 evaluator (1-2 hours)

Create `auto_evaluate_v4.py` with the new `QualityEvaluatorV4` class. Run on all existing test images + 5 new diverse images. Record baseline scores.

### Phase 2: Calibrate thresholds (1 hour)

Adjust all threshold values so the champion config scores 65-75/100, with no ceiling/floor dimensions.

### Phase 3: Validate against human judgment (2-4 hours)

Generate 10 models with varying parameters. Have a human rank them 1-10 by visual quality. Compute Spearman correlation between v4 scores and human rankings. Target: rho > 0.7.

### Phase 4: Replace in GA (30 minutes)

Update `genetic_optimizer.py` to use v4 evaluator. This is a drop-in replacement since the interface (score 0-100 overall + per-dimension) is identical.

### Phase 5: Deprecate v1-v3 (cleanup)

Archive `evaluate.py` and the v3 `QualityEvaluatorV3` from `auto_evaluate.py`.

---

## 8. References

- **Gen3DEval** (arXiv 2504.08125): VLM-based evaluation of text-to-3D outputs. Closest published work to automated 3D quality scoring.
- **3DGen-Bench** (arXiv 2503.21745): Comprehensive benchmark suite for 3D generative models.
- **3D Arena** (arXiv 2506.18787): Open platform for 3D generation evaluation with human preference data.
- **CLIP-IQA** (PyTorch-Metrics): No-reference image quality assessment using CLIP embeddings.
- **QualiCLIP** (arXiv 2403.11176): Quality-aware CLIP for opinion-unaware image quality assessment.
- **trimesh** (github.com/mikedh/trimesh): Python mesh analysis library with watertight checking and face diagnostics.

---

## Appendix A: Full Score Distribution from Existing Experiments

### v1/v2 scoring (evaluate.py)
- **N experiments**: ~120 (sweep + GA gen0-gen4)
- **Overall range**: 31.95 - 52.16
- **Mean**: ~46.5
- **Per-dimension ranges**:
  - Silhouette: 9-45 (very low, viewpoint mismatch)
  - Contour: 24-63 (moderate variance)
  - Color: 13-84 (high variance, viewpoint-dependent)
  - Detail: 19-30 (very low, LPIPS/SSIM failure)
  - Artifacts: 49-100 (ceiling effect, non-discriminative)
  - Smoothness: 8-20 (floor effect, penalizes texture)
  - Coherence: 57-62 (low variance, non-discriminative)

### v3/v3.2 scoring (auto_evaluate.py)
- **N experiments**: ~10
- **Overall range**: 66-70
- **Per-dimension ranges**:
  - Shape: 52-68 (moderate, still viewpoint-affected)
  - Color Match: 34-53 (moderate)
  - Detail: 99-100 (ceiling, non-discriminative)
  - Fragmentation: 55-81 (best discriminability, highest variance)
  - Smoothness: 25-36 (floor effect, penalizes texture)
  - Darkness: 81-91 (low variance)
  - Vibrancy: 95-100 (ceiling, non-discriminative)

### Key observation

The same configurations scored ~37 on v2 and ~68 on v3. This 31-point gap is NOT because the models improved -- it is because the metrics changed. This underscores why having principled, stable metrics is critical. Optimizing against a broken metric produces configurations that are only "good" according to the broken metric.
