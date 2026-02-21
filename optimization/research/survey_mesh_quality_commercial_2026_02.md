# Survey: 3D Mesh Quality Evaluation for Commercial-Grade Product Rendering

**Date**: 2026-02-22
**Author**: Research Optimizer Agent
**Scope**: Mesh integrity, surface quality, texture coherence, and commercial quality standards for AI-generated 3D assets
**Focus**: Detecting 5 specific defects in TRELLIS.2 outputs: holes, surface peeling/fragmentation, surface roughness, texture bleeding, grey/desaturated patches

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Defect-by-Defect Analysis](#2-defect-by-defect-analysis)
3. [Mesh Integrity Metrics](#3-mesh-integrity-metrics)
4. [Surface Smoothness Metrics](#4-surface-smoothness-metrics)
5. [Texture Quality Metrics](#5-texture-quality-metrics)
6. [Commercial 3D Quality Standards](#6-commercial-3d-quality-standards)
7. [Recent Papers (2024-2026)](#7-recent-papers-2024-2026)
8. [Concrete Metrics to Implement](#8-concrete-metrics-to-implement)
9. [Prioritized Implementation Plan](#9-prioritized-implementation-plan)
10. [References](#10-references)

---

## 1. Executive Summary

This survey targets a gap between our existing evaluation system and the actual defects we observe in TRELLIS.2 outputs. While our v4 evaluator covers 9 dimensions across 5 categories, it fails to detect several critical quality issues that are immediately visible to human observers. The five defect categories listed below require new, targeted metrics.

**Key finding**: The most impactful improvements come not from adding learned metrics (DreamSim, CLIP-S) but from implementing straightforward geometric and texture analysis that directly measures the failure modes we observe. Trimesh and OpenCV provide nearly everything needed with zero additional model downloads.

### Gap Analysis: What Our Evaluator Misses

| Defect | Currently Detected? | Why Not | Fix Difficulty |
|--------|---------------------|---------|----------------|
| Mesh holes | Partial (v4 watertight check) | Binary check, no hole sizing/counting | Easy |
| Surface peeling/fragmentation | Partial (v4 coherence) | Detects cracks, not peeling patterns | Medium |
| Surface roughness on smooth objects | Partial (v4 normal Laplacian) | No object-type awareness, no curvature analysis | Medium |
| Texture bleeding across UV seams | No | UV seam analysis not implemented | Medium |
| Grey/desaturated patches | Partial (v4 color vitality) | Thresholds too lenient, spatial clustering missing | Easy |

---

## 2. Defect-by-Defect Analysis

### 2.1 Mesh Holes

**What it looks like**: Gaps in the geometry, typically at shoe soles, thin protruding parts, or areas where the sparse voxel grid has insufficient resolution.

**Root cause in TRELLIS.2**: The 64^3 sparse structure grid cannot represent thin features. During FDG mesh extraction, some voxels near thin parts fail to produce valid faces. Decimation (xatlas simplification) can also create holes by collapsing thin geometry.

**Detection methods**:

1. **Boundary edge counting** (trimesh): Every edge in a watertight mesh is shared by exactly 2 faces. Boundary edges (shared by only 1 face) indicate holes. `mesh.outline()` returns the boundary loops. The number and total length of boundary edges quantifies hole severity.

2. **Boundary loop analysis**: Count the number of distinct boundary loops. A single connected boundary loop suggests one hole (potentially a deliberate opening like the top of a cup). Multiple disjoint loops suggest multiple unintended holes.

3. **Hole area estimation**: For each boundary loop, compute the area of the convex hull of the loop vertices. Larger holes are more visually damaging. Normalize by total mesh surface area.

4. **Flood-fill on rendered silhouette** (already in v1 evaluator): Render the mesh, flood-fill from corners, invert to find interior holes. This catches holes visible from the camera viewpoint but misses holes on the back/bottom.

**Implementation with trimesh**:
```python
import trimesh
import numpy as np

def mesh_hole_metrics(mesh):
    """Compute hole-related metrics for a trimesh mesh."""
    metrics = {}

    # Boundary edges: edges belonging to only 1 face
    # In trimesh: mesh.edges_unique gives unique edges
    # mesh.faces_unique_edges gives face-edge mapping
    # Boundary edges are those referenced by exactly 1 face

    # Method 1: Use mesh.is_watertight (binary)
    metrics['is_watertight'] = mesh.is_watertight

    # Method 2: Count boundary edges directly
    # Each edge should appear in exactly 2 faces for watertight mesh
    edges_sorted = np.sort(mesh.edges, axis=1)
    edge_tuples = [tuple(e) for e in edges_sorted]
    from collections import Counter
    edge_counts = Counter(edge_tuples)
    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    metrics['n_boundary_edges'] = len(boundary_edges)
    metrics['boundary_edge_ratio'] = len(boundary_edges) / max(len(edge_counts), 1)

    # Method 3: Boundary loops (connected boundary edge chains)
    # trimesh.path.segments can group boundary edges into loops
    # For simplicity, use connected components on boundary edge graph
    if len(boundary_edges) > 0:
        from scipy import sparse
        boundary_verts = set()
        for e in boundary_edges:
            boundary_verts.add(e[0])
            boundary_verts.add(e[1])
        boundary_verts = sorted(boundary_verts)
        vert_map = {v: i for i, v in enumerate(boundary_verts)}
        n = len(boundary_verts)
        rows, cols = [], []
        for e in boundary_edges:
            rows.append(vert_map[e[0]])
            cols.append(vert_map[e[1]])
        graph = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        n_loops = sparse.csgraph.connected_components(graph, directed=False)[0]
        metrics['n_boundary_loops'] = n_loops
    else:
        metrics['n_boundary_loops'] = 0

    # Method 4: Hole area estimation
    # For each boundary loop, sum edge lengths as proxy for hole size
    total_boundary_length = sum(
        np.linalg.norm(
            mesh.vertices[e[0]] - mesh.vertices[e[1]]
        ) for e in boundary_edges
    )
    metrics['total_boundary_length'] = total_boundary_length
    metrics['boundary_length_ratio'] = total_boundary_length / max(mesh.area, 1e-8)

    return metrics
```

**Scoring formula**:
```
hole_score = 100
if n_boundary_loops > 0:
    hole_score -= min(30, n_boundary_loops * 10)
if boundary_length_ratio > 0.01:
    hole_score -= min(30, (boundary_length_ratio - 0.01) * 3000)
if not is_watertight:
    hole_score -= 10
```

### 2.2 Surface Peeling/Fragmentation

**What it looks like**: Texture appears to be flaking off the surface. Discontinuous color patches that look like paper peeling from a wet surface.

**Root cause in TRELLIS.2**: UV fragmentation from xatlas unwrapping creates many small UV islands. When texture resolution is insufficient to cover all islands, some get very few texels, leading to low-resolution patches next to high-resolution ones. Additionally, the grid_sample_3d interpolation at voxel boundaries can produce dark or desaturated strips at UV island boundaries.

**Detection methods**:

1. **UV island analysis**: Count the number of UV islands. More islands = more seam boundaries = more potential peeling. Measure texel density variance across islands -- high variance indicates fragmentation.

2. **Local color variance at edges**: Within the interior of the rendered view, compute the ratio of pixels with high local color variance (Sobel gradient > threshold). Peeling creates sharp edges where smooth surfaces should exist.

3. **Morphological thin-structure detection**: Apply morphological opening to the rendered base_color. The difference between original and opened version reveals thin, high-contrast structures that look like peeling or cracking.

4. **Connected region analysis in color space**: Segment the rendered view into superpixels or use simple thresholding. Count the number of small, disconnected color regions within areas that should be continuous. High fragmentation count = peeling.

5. **Texture map island fragmentation metric**: Analyze the UV texture map directly. Count UV islands, measure average island area, compute fragmentation index = n_islands / sqrt(total_area).

**Implementation**:
```python
def texture_fragmentation_score(texture_map, tex_mask):
    """Measure UV texture map fragmentation."""
    # 1. Count connected components in texture mask (UV islands)
    mask_u8 = (tex_mask * 255).astype(np.uint8)
    n_islands, labels = cv2.connectedComponents(mask_u8)
    n_islands -= 1  # subtract background

    # 2. Compute island area distribution
    island_areas = []
    for label_id in range(1, n_islands + 1):
        area = (labels == label_id).sum()
        island_areas.append(area)
    island_areas = np.array(island_areas)

    # 3. Fragmentation index
    total_texels = tex_mask.sum()
    frag_index = n_islands / max(np.sqrt(total_texels), 1)

    # 4. Small island ratio (islands < 1% of total)
    small_threshold = total_texels * 0.01
    small_ratio = (island_areas < small_threshold).sum() / max(n_islands, 1)

    # 5. Area coefficient of variation
    area_cv = np.std(island_areas) / max(np.mean(island_areas), 1) if len(island_areas) > 1 else 0

    score = 100.0
    # High island count penalty
    if n_islands > 50:
        score -= min(25, (n_islands - 50) * 0.5)
    # High small-island ratio penalty
    if small_ratio > 0.3:
        score -= min(25, (small_ratio - 0.3) * 100)
    # High area CV penalty
    if area_cv > 2.0:
        score -= min(25, (area_cv - 2.0) * 10)
    # Overall fragmentation
    if frag_index > 0.5:
        score -= min(25, (frag_index - 0.5) * 50)

    return max(0, score), {
        'n_islands': n_islands,
        'frag_index': frag_index,
        'small_island_ratio': small_ratio,
        'area_cv': area_cv,
    }
```

### 2.3 Surface Roughness on Manufactured Objects

**What it looks like**: A shoe should have smooth leather/fabric surfaces, but the generated mesh has bumpy, noisy geometry that looks like the surface of an orange peel.

**Root cause in TRELLIS.2**: The sparse voxel representation introduces staircase artifacts at voxel boundaries. The FDG mesh extraction produces geometry that follows the voxel grid rather than the intended smooth surface. Normal maps can amplify this by baking the voxel staircase into per-texel normals.

**Detection methods**:

1. **Gaussian curvature variance** (Lavoue 2007): Local roughness at each vertex is defined based on Gaussian curvature. The roughness metric LRGC = weighted difference between vertex Gaussian curvature and its neighbors' curvatures. Higher LRGC = rougher surface.

2. **Dihedral angle distribution**: Measure angles between normals of adjacent faces. For a smooth manufactured object, most dihedral angles should be close to 180 degrees (flat). A bimodal distribution with a peak at large angles indicates rough geometry.

3. **Laplacian smoothing residual**: Apply one step of Laplacian smoothing to the mesh. The L2 distance between original and smoothed vertex positions measures roughness. Large residuals = rough surface.

4. **Normal map Laplacian energy** (already in v4 evaluator): The Laplacian energy of the rendered normal map measures geometric frequency content. Our v4 implementation uses this but lacks context-awareness (a shoe should be smooth, but a tree bark should be rough).

5. **Curvature histogram analysis**: Compute principal curvatures at each vertex. For manufactured objects, the curvature distribution should have a narrow peak near zero (flat areas) with controlled spread (curved areas). Wide spread = noisy geometry.

**Implementation with trimesh**:
```python
def mesh_roughness_metrics(mesh):
    """Compute surface roughness metrics."""
    metrics = {}

    # 1. Face adjacency angle statistics
    adj_angles = mesh.face_adjacency_angles  # radians
    metrics['mean_dihedral'] = float(np.degrees(adj_angles.mean()))
    metrics['std_dihedral'] = float(np.degrees(adj_angles.std()))
    # Ratio of "rough" faces (angle > 30 degrees from flat)
    rough_threshold = np.radians(30)
    metrics['rough_face_ratio'] = float((adj_angles > rough_threshold).sum() / max(len(adj_angles), 1))

    # 2. Laplacian smoothing residual
    # Use trimesh's vertex neighbors to compute Laplacian
    from scipy import sparse
    adj_matrix = mesh.vertex_adjacency_graph
    # ... or use vertex_neighbors

    # Simpler: compute via face normal consistency
    face_normals = mesh.face_normals

    # 3. Vertex normal consistency
    # Area-weighted vertex normals vs face normals
    face_areas = mesh.area_faces
    vertex_normals = mesh.vertex_normals

    # For each face, compare face normal to average of its vertex normals
    v0n = vertex_normals[mesh.faces[:, 0]]
    v1n = vertex_normals[mesh.faces[:, 1]]
    v2n = vertex_normals[mesh.faces[:, 2]]
    avg_vn = (v0n + v1n + v2n) / 3.0
    avg_vn_norm = avg_vn / (np.linalg.norm(avg_vn, axis=1, keepdims=True) + 1e-8)

    consistency = np.sum(face_normals * avg_vn_norm, axis=1)  # cos(angle)
    metrics['mean_normal_consistency'] = float(consistency.mean())
    metrics['rough_ratio_nc'] = float((consistency < 0.95).sum() / max(len(consistency), 1))

    return metrics
```

**Scoring formula**:
```
roughness_score = 100
# Dihedral angle spread
if std_dihedral > 15:
    roughness_score -= min(30, (std_dihedral - 15) * 2)
# Rough face ratio
if rough_face_ratio > 0.05:
    roughness_score -= min(30, (rough_face_ratio - 0.05) * 500)
# Normal consistency
if mean_normal_consistency < 0.95:
    roughness_score -= min(20, (0.95 - mean_normal_consistency) * 500)
# Very rough faces
if rough_ratio_nc > 0.1:
    roughness_score -= min(20, (rough_ratio_nc - 0.1) * 200)
```

### 2.4 Texture Bleeding Across UV Seams

**What it looks like**: Colors from one part of the mesh appear on adjacent UV islands. A red shoe might have red bleeding into the white sole along the UV seam boundary.

**Root cause in TRELLIS.2**: During texture baking (o_voxel/postprocess.py), the 3D voxel color field is projected onto the UV atlas. At UV seam boundaries, bilinear sampling can pull colors from the wrong side of the seam. The xatlas UV unwrapping creates many island boundaries, each a potential bleeding site.

**Detection methods**:

1. **Checker texture test**: Replace the base color texture with a checker pattern aligned to UV coordinates. Render from multiple views. Visible seam discontinuities in the checker pattern indicate UV quality problems. This is the industry-standard UV quality visualization.

2. **Seam gradient analysis**: Identify UV seam edges in the mesh. For each seam edge, compare the texel colors on either side. Large color differences at seam edges (relative to the local color gradient) indicate bleeding.

3. **Texel margin analysis**: Measure the bleed margin (gutter/padding) around each UV island. Standard practice requires 0.5-1 pixel margin at the working resolution. Insufficient margin causes visible bleeding at lower mip levels.

4. **Cross-seam color discontinuity**: For each pair of UV seam edges that share 3D vertices, compare the texture colors at corresponding points. In a perfect UV mapping, colors should match exactly at shared vertices.

**Implementation**:
```python
def uv_seam_quality(mesh, texture_map):
    """Analyze UV seam quality and detect bleeding."""
    # This requires access to the UV coordinates and face-UV mapping
    # which trimesh stores in mesh.visual.uv

    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return 50.0, {'error': 'no UV data'}

    uv = mesh.visual.uv  # (n_vertices_uv, 2) normalized [0,1]
    h, w = texture_map.shape[:2]

    # Find UV seam edges: edges where two faces share a 3D vertex
    # but have different UV coordinates
    # This requires the face-UV index mapping

    # Simpler proxy: detect high-gradient edges in texture map
    # at UV island boundaries
    gray_tex = cv2.cvtColor(texture_map, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_tex.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_tex.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Detect island boundaries (edges of valid texel mask)
    mask = (texture_map.sum(axis=2) > 0).astype(np.uint8)
    boundary = mask - cv2.erode(mask, np.ones((3,3), np.uint8))

    # Measure gradient at island boundaries vs interior
    boundary_bool = boundary > 0
    interior_bool = cv2.erode(mask, np.ones((5,5), np.uint8)) > 0

    if boundary_bool.sum() < 10 or interior_bool.sum() < 100:
        return 50.0, {'insufficient_data': True}

    boundary_grad_mean = grad_mag[boundary_bool].mean()
    interior_grad_mean = grad_mag[interior_bool].mean()

    # Bleeding ratio: how much stronger are boundary gradients
    bleed_ratio = boundary_grad_mean / max(interior_grad_mean, 1.0)

    score = 100.0
    if bleed_ratio > 3.0:
        score -= min(40, (bleed_ratio - 3.0) * 10)
    if bleed_ratio > 5.0:
        score -= min(30, (bleed_ratio - 5.0) * 10)

    return max(0, score), {
        'boundary_grad_mean': float(boundary_grad_mean),
        'interior_grad_mean': float(interior_grad_mean),
        'bleed_ratio': float(bleed_ratio),
    }
```

### 2.5 Grey/Desaturated Patches

**What it looks like**: Regions that should be colorful (e.g., a bright red shoe) appear grey or washed out. Often occurs in patches, not uniformly.

**Root cause in TRELLIS.2**: At sparse voxel boundaries, grid_sample_3d interpolation toward zero produces dark/grey texels. The grey recovery in postprocess.py (Gaussian diffusion from colorful neighbors) partially fixes this, but can miss large grey regions or those surrounded by other grey areas. Additionally, CFG guidance that is too low (especially tex_slat_guidance_strength < 3) produces globally desaturated outputs.

**Detection methods**:

1. **Spatial grey cluster detection**: Instead of just counting grey pixels globally (as v4 does), detect spatially connected clusters of grey pixels. A single large grey patch is more visually damaging than scattered individual grey pixels.

2. **Local saturation variance**: Compute HSV saturation in a sliding window. Areas where saturation drops sharply compared to neighbors are grey patches. Use the ratio of low-saturation windows to total windows.

3. **Chroma analysis in LAB space**: Convert to CIELAB. Compute chroma C* = sqrt(a*^2 + b*^2). Low chroma regions are desaturated. Threshold: C* < 10 for "grey" pixels in most contexts.

4. **Grey patch area and count**: Find connected components of grey pixels (saturation < 15 AND value in [40, 200]). Count the number of patches and measure the largest patch area as a fraction of total object area.

5. **Texture map grey analysis**: Analyze the UV texture directly. Grey regions in the texture map correspond to failed voxel sampling. Measure grey_texel_ratio and largest_grey_component_area.

**Implementation**:
```python
def grey_patch_analysis(rgb_image, mask):
    """Detect and measure grey/desaturated patches."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Grey pixels: low saturation, mid-range value (not black, not white)
    grey_pixels = (sat < 15) & (val > 40) & (val < 200) & mask

    # Connected component analysis on grey pixels
    grey_u8 = grey_pixels.astype(np.uint8) * 255
    n_patches, labels, stats, centroids = cv2.connectedComponentsWithStats(grey_u8)
    n_patches -= 1  # subtract background

    mask_area = mask.sum()

    if n_patches == 0:
        return 100.0, {'n_grey_patches': 0, 'grey_ratio': 0.0}

    # Patch areas (skip label 0 = background)
    patch_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_patch = patch_areas.max()
    total_grey = patch_areas.sum()

    grey_ratio = total_grey / max(mask_area, 1)
    largest_patch_ratio = largest_patch / max(mask_area, 1)

    score = 100.0
    # Overall grey ratio penalty
    if grey_ratio > 0.1:
        score -= min(30, (grey_ratio - 0.1) * 200)
    # Large patch penalty (worse than scattered small patches)
    if largest_patch_ratio > 0.05:
        score -= min(30, (largest_patch_ratio - 0.05) * 400)
    # Number of patches penalty
    if n_patches > 5:
        score -= min(20, (n_patches - 5) * 2)

    return max(0, score), {
        'n_grey_patches': n_patches,
        'grey_ratio': float(grey_ratio),
        'largest_patch_ratio': float(largest_patch_ratio),
        'patch_areas': patch_areas.tolist(),
    }
```

---

## 3. Mesh Integrity Metrics

### 3.1 Watertightness and Manifoldness

**Definition**: A mesh is watertight if every edge is shared by exactly 2 faces (no boundary edges). A mesh is manifold if additionally no edge is shared by more than 2 faces and every vertex star is connected.

**Tools**:
- `trimesh.Trimesh.is_watertight` -- binary check
- `trimesh.Trimesh.is_volume` -- True if watertight and has consistent winding
- `pymeshlab` filter `compute_geometric_measures` -- returns volume (only valid for watertight meshes)

**Our v4 status**: Binary watertight check with -20 penalty. This is too coarse -- a mesh with one tiny hole at the bottom gets the same penalty as one with 50 large holes.

**Recommended upgrade**: Boundary edge counting + loop analysis + area estimation (see Section 2.1).

### 3.2 Self-Intersection Detection

**Definition**: Self-intersections occur when faces of the mesh penetrate other faces, creating impossible geometry.

**Detection methods**:
- `pymeshlab` filter `compute_self_intersecting_faces` -- returns list of intersecting face pairs
- BVH-accelerated ray casting (available in trimesh via `mesh.ray`)
- PyMesh `detect_self_intersection()` -- dedicated function

**Relevance to TRELLIS.2**: Self-intersections are uncommon in FDG extraction outputs but can occur during mesh decimation. They create visual artifacts when rendering with backface culling.

**Implementation**:
```python
import pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh('output.glb')
ms.compute_selection_by_self_intersections_per_face()
n_self_intersecting = ms.current_mesh().selected_face_number()
```

### 3.3 Connected Components (Floating Fragments)

**Definition**: Count the number of disconnected mesh components. Multiple components indicate floating geometry fragments.

**Detection methods**:
- `mesh.split()` -- splits mesh into connected components (memory-intensive for large meshes)
- `scipy.sparse.csgraph.connected_components()` on face adjacency graph (more efficient)
- Our v4 implementation already uses the sparse graph approach

**Recommended upgrade**: In addition to counting components, measure the relative size of each. The main component should contain >95% of the faces. Small components (<100 faces) are almost certainly floating artifacts.

**Implementation**:
```python
def connected_component_analysis(mesh):
    """Analyze mesh connected components."""
    from scipy import sparse

    nfaces = len(mesh.faces)
    adj = mesh.face_adjacency
    graph = sparse.coo_matrix(
        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
        shape=(nfaces, nfaces)).tocsr()
    n_comp, labels = sparse.csgraph.connected_components(graph, directed=False)

    # Component sizes
    comp_sizes = np.bincount(labels)
    main_comp = comp_sizes.max()
    main_ratio = main_comp / nfaces
    small_comps = (comp_sizes < 100).sum()

    return {
        'n_components': n_comp,
        'main_component_ratio': float(main_ratio),
        'n_small_fragments': int(small_comps),
        'fragment_face_count': int(nfaces - main_comp),
    }
```

### 3.4 Degenerate Face Detection

**Definition**: Faces with near-zero area (collapsed triangles), or extremely high aspect ratios (needle-like triangles).

**Detection methods**:
- `mesh.area_faces` -- check for near-zero areas
- `mesh.face_angles` -- check for near-zero angles
- `mesh.edges_unique_length` -- check for extreme length ratios

**Our v4 status**: Already implemented (degenerate ratio and aspect ratio checks). Adequate for current needs.

---

## 4. Surface Smoothness Metrics

### 4.1 Gaussian Curvature-Based Roughness (Lavoue 2007)

**Reference**: Lavoue, "A Roughness Measure for 3D Mesh Visual Masking" (APGV 2007)

**Method**: Define local roughness LRGC at each vertex as the weighted difference between the vertex's Gaussian curvature and its neighbors' curvatures. The global roughness R is the surface-integral-weighted sum of LRGC.

**Formula**:
```
LRGC(v) = |K(v) - sum_j(w_j * K(n_j))|
where K(v) = Gaussian curvature at vertex v
      n_j = j-th neighbor vertex
      w_j = area-based weight
```

**Applicability**: HIGH. Directly measures the "orange peel" effect on manufactured objects. Gaussian curvature is invariant to rigid transformations.

**Implementation**: Trimesh provides `mesh.vertex_defects` (discrete Gaussian curvature = 2*pi - sum of face angles at vertex).

### 4.2 Dihedral Angle Distribution

**Reference**: Standard mesh quality metric used in computational geometry and FEM.

**Method**: For each pair of adjacent faces, compute the dihedral angle. Smooth surfaces have dihedral angles close to 180 degrees (pi radians). The distribution of dihedral angles characterizes surface roughness.

**Implementation**: `mesh.face_adjacency_angles` in trimesh.

**Key statistics**:
- Mean angle: Should be close to pi for smooth objects
- Standard deviation: Lower = smoother
- 95th percentile: Captures worst-case roughness
- Ratio of angles > 30 degrees from flat: "rough face ratio"

### 4.3 Laplacian Smoothing Residual

**Reference**: Sorkine, "Laplacian Mesh Processing" (Eurographics 2005)

**Method**: Apply one step of uniform Laplacian smoothing. The L2 distance between original and smoothed positions is the roughness measure. High residual = rough surface.

**Formula**:
```
delta_v = v - (1/|N(v)|) * sum_{u in N(v)} u
roughness(v) = ||delta_v||
```

**Applicability**: HIGH. Can be computed efficiently with sparse matrix operations. Provides per-vertex roughness values that can be visualized as a heatmap.

### 4.4 Normal Map Laplacian Energy (Already Implemented)

**Current implementation**: v4 evaluator computes Laplacian energy of rendered normal maps. This captures geometric roughness visible in the rendered view.

**Limitation**: View-dependent (only captures roughness visible from the camera angle). Does not account for object type (a shoe should be smoother than a tree).

**Recommended upgrade**: Add a global mesh roughness metric (4.1 or 4.2) that is view-independent, and use it as the primary roughness signal. Keep the normal map Laplacian as a secondary check.

### 4.5 FMPD: Fast Mesh Perceptual Distance (Lavoue 2011)

**Reference**: Lavoue et al., "A fast roughness-based approach to the assessment of 3D mesh visual quality" (CG 2012)

**Method**: Full-reference metric that computes roughness on both distorted and reference meshes at multiple scales, then combines the roughness differences into a perceptual distance.

**Applicability**: LOW for our use case (requires reference mesh). But the roughness computation component can be used standalone.

---

## 5. Texture Quality Metrics

### 5.1 UV Quality Metrics

#### 5.1.1 Texel Density Uniformity

**Definition**: Measure the ratio of texels per world-space unit area across the mesh. Uniform texel density means consistent detail level everywhere.

**Computation**: For each face, compute the ratio of UV-space area to 3D-space area. The coefficient of variation (std/mean) of this ratio measures uniformity.

```python
def texel_density_uniformity(mesh, texture_size):
    """Compute texel density coefficient of variation."""
    uv = mesh.visual.uv
    faces = mesh.faces

    # 3D face areas
    area_3d = mesh.area_faces

    # UV face areas (in texel space)
    uv_verts = uv * texture_size
    v0_uv = uv_verts[faces[:, 0]]
    v1_uv = uv_verts[faces[:, 1]]
    v2_uv = uv_verts[faces[:, 2]]
    cross = np.cross(v1_uv - v0_uv, v2_uv - v0_uv)
    area_uv = 0.5 * np.abs(cross)

    # Texel density per face
    density = area_uv / np.maximum(area_3d, 1e-10)

    cv = np.std(density) / max(np.mean(density), 1e-10)
    return float(cv)
```

**Threshold**: CV < 0.5 is good, 0.5-1.0 acceptable, >1.0 poor.

#### 5.1.2 UV Space Utilization

**Definition**: Percentage of UV space [0,1]^2 covered by mesh geometry.

**Computation**: Rasterize UV triangles into a binary mask at texture resolution, count covered pixels.

**Threshold**: >50% is good for complex objects, >70% is excellent.

#### 5.1.3 Conformal Energy (Distortion)

**Definition**: Measure of UV mapping distortion -- how much triangles are stretched or compressed in UV space relative to 3D space.

**Computation**: For each face, compute the singular values of the Jacobian of the 3D-to-UV mapping. Perfect mapping has singular values = (1, 1). Distortion = max(s1/s2, s2/s1).

**Threshold**: Mean distortion < 2.0 is good, < 1.5 is excellent.

### 5.2 Seam Visibility Metrics

#### 5.2.1 Checker Texture Test

**Description**: Replace base color with a procedural checker pattern. Render from multiple views. Analyze rendered checkers for discontinuities at UV seam boundaries.

**Implementation**: Straightforward with trimesh + any renderer. Most informative single test for UV quality.

#### 5.2.2 Seam Color Discontinuity

**Description**: At UV seam edges, compare texel colors on either side. In a well-baked texture, colors should match at seam vertices.

**Implementation**: Requires UV seam edge identification, which is available from the xatlas output.

### 5.3 Perceptual Crack Detection (PCD)

**Reference**: Shafiei et al., "Perceptual Crack Detection for Rendered 3D Textured Meshes" (IEEE QoMEX 2024)

**Method**: Uses contrast and Laplacian measurement modules to characterize crack artifacts in rendered 3D meshes. Generates a crack map and incorporates it into quality assessment via a weighting mechanism.

**Code**: https://github.com/arshafiee/crack-detection-VVM

**Applicability**: MEDIUM-HIGH. Directly addresses our "surface peeling/fragmentation" defect. Full-reference method (needs reference), but the crack detection module can be used standalone.

### 5.4 HybridMQA (2024)

**Reference**: "HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment" (arXiv 2412.01986)

**Method**: Hybrid model-based + projection-based approach that uses graph learning for 3D topology-aware features, projects them to 2D, and uses cross-attention to capture geometry-texture interactions.

**Applicability**: MEDIUM. Full-reference, requires training, but conceptually shows that joint geometry-texture analysis outperforms either alone.

### 5.5 GeodesicPSIM (2023/2025)

**Reference**: "GeodesicPSIM: Predicting the Quality of Static Mesh with Texture Map via Geodesic Patch Similarity" (arXiv 2308.04928)

**Method**: Uses geodesic distance on the mesh surface (rather than Euclidean distance) to define patches for perceptual similarity computation. More accurate for curved surfaces.

**Applicability**: MEDIUM. Novel concept but requires implementation from scratch.

### 5.6 FMQM: Field Mesh Quality Metric (2025)

**Reference**: "Textured Mesh Quality Assessment Using Geometry and Color Field Similarity" (arXiv 2505.10824)

**Method**: Uses signed distance fields and color fields to extract four features: geometry similarity, geometry gradient similarity, space color distribution similarity, and space color gradient similarity.

**Applicability**: MEDIUM. Four-dimensional quality representation is principled but requires SDF computation infrastructure.

---

## 6. Commercial 3D Quality Standards

### 6.1 TurboSquid CheckMate Pro v2

TurboSquid's CheckMate certification represents the highest commercially enforced quality standard for 3D models. Requirements include:

**Topology**:
- Grid arrangement for edges (quad-dominant, 90-degree edge angles where possible)
- Supporting edges to hold shape during subdivision
- No poles with 6+ edges on curved surfaces
- No unnecessary edge detail
- Objects separated into pieces corresponding to real-world components
- Models must subdivide cleanly at least one level
- No crease settings above 0 (3ds Max / Maya)

**UV Mapping**:
- UV coordinates must open without errors
- Texture placement must match thumbnails
- All texture paths stripped from model files
- Proper UV island organization

**Rendering**:
- Must render correctly in standard lighting
- Thumbnails showing subdivision level 0 and level 1

**File Quality**:
- No degenerate geometry
- Proper naming conventions for all objects and materials
- Clean scene hierarchy

**Relevance to TRELLIS.2**: The topology requirements (grid edge flow, clean subdivision) are aspirational for AI-generated meshes. Current AI generators, including TRELLIS.2, produce triangle soups that fail topology requirements. However, the UV mapping and texture quality requirements are immediately relevant and measurable.

### 6.2 Game Industry Standards

Standard game asset quality criteria:

| Criterion | Acceptable | Good | Excellent |
|-----------|------------|------|-----------|
| Polygon count | < 100K | < 50K | < 20K |
| UV utilization | > 40% | > 60% | > 80% |
| Texel density variance (CV) | < 2.0 | < 1.0 | < 0.5 |
| Max texture resolution | 2048 | 4096 | 4096+ |
| Manifold | Required | Required | Required |
| No self-intersections | Required | Required | Required |
| Clean normals | Required | Required | Required |
| LOD support | Optional | 2 levels | 3+ levels |
| PBR textures | Albedo+Normal | +Rough+Metal | +AO+Emissive |

**Relevance to TRELLIS.2**: TRELLIS.2 outputs at ~500K-800K faces (after decimation) are far above game-industry polygon counts. UV utilization is typically 30-50% due to xatlas fragmentation. These are known limitations.

### 6.3 AI 3D Generator Comparisons (Industry Reviews)

Based on comprehensive reviews (SimInsights 2025, Cyber-Fox.net 2025):

**Evaluation dimensions used by reviewers**:
1. Geometry fidelity (silhouette match, overall shape)
2. Topology quality (holes, floating parts, degenerate faces)
3. Texture quality (resolution, seam visibility, color accuracy)
4. Editability (can an artist modify the output?)
5. Failure rate (how often does generation fail completely?)
6. Iteration speed (can you refine the result?)

**Comparative rankings** (approximate, from multiple sources):
- Rodin (Hyper3D): 8.5-9.5/10 -- best photorealism
- Tripo: Best topology/editability, captures form well
- TRELLIS: Good geometry, competitive CLIP scores, but texture quality lags
- Meshy: Consistent but lower individual quality
- CSM: Struggles with complex geometry

**Key insight**: Industry reviewers emphasize that **no single metric captures quality**. They use a dashboard approach with 5-8 dimensions and always include human visual inspection.

---

## 7. Recent Papers (2024-2026)

### 7.1 Directly Relevant to Our Defects

| Paper | Year | Venue | Relevance | Key Contribution |
|-------|------|-------|-----------|------------------|
| PCD (Perceptual Crack Detection) | 2024 | QoMEX | HIGH | Crack/peeling artifact detection in rendered meshes |
| Robust Hole Detection | 2024 | ScienceDirect | HIGH | Boundary traversal for hole detection with singular vertices |
| HybridMQA | 2024 | arXiv | MEDIUM | Joint geometry-texture quality with cross-attention |
| SeamCrafter | 2025 | arXiv | MEDIUM | RL-based seam placement to minimize visibility |
| ArtUV | 2025 | arXiv | MEDIUM | Artist-style UV unwrapping with semantic visibility awareness |
| FMQM | 2025 | arXiv | MEDIUM | SDF-based textured mesh quality metric |
| GeodesicPSIM | 2025 | arXiv | MEDIUM | Geodesic patch similarity for mesh+texture quality |
| TMQA | 2023 | ACM TOG | MEDIUM | Large-scale learned mesh quality metric (343K stimuli) |
| NCD | 2025 | CGF | MEDIUM | Normal-guided Chamfer distance for watertight reconstruction |

### 7.2 Evaluation Frameworks

| Framework | Year | Venue | Approach | Human Correlation |
|-----------|------|-------|----------|-------------------|
| Hi3DEval | 2025 | NeurIPS | Video+3D learned scorer | tau = 0.774 |
| HyperScore | 2025 | ICCV | CLIP + hypernetwork | tau ~ 0.68 |
| 3DGen-Score | 2025 | arXiv | CLIP + dimension heads | tau ~ 0.65 |
| GPTEval3D | 2024 | CVPR | GPT-4V pairwise | tau = 0.710 |
| Rank2Score | 2025 | arXiv | Contrastive rank learning | -- |
| SRAM | 2025 | arXiv | PointBERT+LLM no-reference | PLCC = 0.689 |

### 7.3 Perceptual Metrics

| Metric | Year | Human Correlation | Compute Cost | Our Status |
|--------|------|-------------------|--------------|------------|
| DreamSim | 2023 | tau = 0.77 | 15ms/pair | Not implemented |
| CLIP-S | 2021 | tau = 0.55 | 5ms/image | Not implemented |
| LAION Aesthetic | 2022 | tau = 0.55 | 5ms/image | Not implemented |
| VQAScore | 2024 | tau = 0.65 | 100ms/eval | Not implemented |
| LPIPS | 2018 | tau = 0.58 | 10ms/pair | Implemented |
| MEt3R | 2025 | -- | 2-3s/pair | Not implemented |

---

## 8. Concrete Metrics to Implement

Based on this survey, here are the specific metrics to add to our evaluation system, organized by the defect they detect:

### 8.1 For Mesh Holes

| Metric | Implementation | Compute Cost | Dependencies |
|--------|---------------|--------------|--------------|
| `n_boundary_edges` | trimesh edge analysis | <100ms | trimesh (already available) |
| `boundary_edge_ratio` | boundary_edges / total_edges | <100ms | trimesh |
| `n_boundary_loops` | scipy connected_components on boundary edge graph | <100ms | scipy |
| `total_boundary_length` | sum of boundary edge lengths | <100ms | trimesh |
| `boundary_length_ratio` | total_boundary_length / mesh.area | <100ms | trimesh |
| `silhouette_hole_area` | flood-fill on rendered alpha (multi-view) | <50ms/view | cv2 |

### 8.2 For Surface Peeling/Fragmentation

| Metric | Implementation | Compute Cost | Dependencies |
|--------|---------------|--------------|--------------|
| `n_uv_islands` | connected_components on texture mask | <50ms | cv2 |
| `fragmentation_index` | n_islands / sqrt(total_texels) | <50ms | cv2 |
| `small_island_ratio` | islands < 1% of total / n_islands | <50ms | cv2 |
| `island_area_cv` | std(island_areas) / mean(island_areas) | <50ms | cv2 |
| `morphological_peel_ratio` | (original - morphologically opened) > threshold | <100ms/view | cv2 |

### 8.3 For Surface Roughness

| Metric | Implementation | Compute Cost | Dependencies |
|--------|---------------|--------------|--------------|
| `mean_dihedral_angle` | trimesh face_adjacency_angles | <200ms | trimesh |
| `dihedral_std` | std of dihedral angles | <200ms | trimesh |
| `rough_face_ratio` | faces with dihedral > 30deg / total | <200ms | trimesh |
| `mean_normal_consistency` | face normal vs vertex normal alignment | <200ms | trimesh |
| `laplacian_residual_mean` | distance between original and smoothed positions | <500ms | scipy |
| `gaussian_curvature_variance` | std of vertex_defects | <200ms | trimesh |

### 8.4 For Texture Bleeding

| Metric | Implementation | Compute Cost | Dependencies |
|--------|---------------|--------------|--------------|
| `seam_gradient_ratio` | boundary_grad / interior_grad | <100ms | cv2 |
| `boundary_color_discontinuity` | color diff at UV island boundaries | <100ms | cv2 |
| `texel_density_cv` | std(texel_density) / mean(texel_density) | <200ms | trimesh UV data |

### 8.5 For Grey/Desaturated Patches

| Metric | Implementation | Compute Cost | Dependencies |
|--------|---------------|--------------|--------------|
| `grey_patch_count` | connected_components of grey pixels | <50ms/view | cv2 |
| `grey_patch_ratio` | total_grey_area / object_area | <50ms/view | cv2 |
| `largest_grey_patch_ratio` | max_grey_patch / object_area | <50ms/view | cv2 |
| `chroma_floor_ratio` | C* < 10 pixel ratio in LAB space | <50ms/view | cv2 |
| `texture_grey_ratio` | grey texels in UV map / total texels | <50ms | cv2 |

---

## 9. Prioritized Implementation Plan

### Tier 1: Immediate (1-2 hours, no new dependencies)

These use only trimesh (already available) and OpenCV (already available). Zero model downloads.

1. **Enhanced hole detection** (replace binary watertight check)
   - boundary_edge_ratio, n_boundary_loops, boundary_length_ratio
   - Multi-view silhouette hole detection (flood-fill from 8 views, not just front)
   - Integrate into B1_mesh_integrity, replacing the binary watertight check

2. **Enhanced grey patch detection** (upgrade C2_color_vitality)
   - Connected component analysis of grey pixels (spatial clustering)
   - Largest grey patch ratio (more penalizing than scattered grey)
   - LAB chroma floor analysis
   - Texture map grey analysis

3. **UV fragmentation analysis** (new sub-metric in C1_tex_coherence)
   - n_uv_islands, small_island_ratio, island_area_cv
   - Fragmentation index
   - Integrate into C1_tex_coherence or create new dimension

### Tier 2: Short-term (1 day, no new dependencies)

4. **Mesh roughness analysis** (upgrade B2_surface_quality)
   - Dihedral angle statistics (mean, std, rough ratio)
   - Normal consistency metric
   - Gaussian curvature variance (vertex_defects)
   - Replace or complement normal map Laplacian with mesh-level metrics

5. **UV seam bleeding detection** (new sub-metric)
   - Seam gradient ratio (boundary vs interior gradient magnitude)
   - Boundary color discontinuity
   - Integrate into C1_tex_coherence

6. **Connected component quality** (upgrade B1_mesh_integrity)
   - Component size distribution
   - Main component ratio
   - Small fragment count
   - Fragment face percentage

### Tier 3: Medium-term (1 week, requires model downloads)

7. **CLIP-S input alignment** (new dimension, 10-15 pts)
   - Replaces A2_color_dist as primary input matching metric
   - `pip install open-clip-torch`, ~400MB model
   - Compute CLIP cosine similarity between input image and each rendered view

8. **DreamSim perceptual quality** (upgrade or supplement LPIPS)
   - `pip install dreamsim`, ~200MB model
   - Better human alignment than LPIPS (tau 0.77 vs 0.58)
   - Apply to multi-view renders

9. **LAION Aesthetic score** (new dimension or bonus)
   - CLIP features + small linear head, ~5MB additional
   - Cheap "looks good" proxy

### Tier 4: Long-term (2-4 weeks, significant engineering)

10. **PCD crack detection** (specialized peeling/fragmentation)
    - Port from https://github.com/arshafiee/crack-detection-VVM
    - Requires understanding of the contrast+Laplacian crack characterization

11. **MEt3R multi-view consistency** (upgrade E1_multiview)
    - DUSt3R-based 3D consistency metric
    - ~1.5GB model, 15s per evaluation
    - Replaces CV-of-per-view-stats with actual geometric consistency

12. **HyperScore learned evaluator** (replace entire scoring system)
    - CLIP ViT-L/14 + trained dimension-specific heads
    - Requires training on human preference data or using pre-trained weights if released

### Revised Scoring System (After Tier 1+2)

| Dimension | Pts | Metrics | Status |
|-----------|-----|---------|--------|
| A1 Silhouette Match | 12 | Alpha IoU (multi-view) | Existing |
| A2 Input Alignment | 13 | CLIP-S (Tier 3) or color dist | Upgrade |
| B1 Mesh Integrity | 12 | Watertight + holes + components + fragments | **Enhanced** |
| B2 Surface Roughness | 10 | Dihedral angles + curvature + normal consistency | **Enhanced** |
| C1 Texture Coherence | 12 | Cracks + UV fragmentation + seam bleeding | **Enhanced** |
| C2 Grey Patch Detection | 8 | Spatial grey clusters + chroma floor | **Enhanced** |
| C3 Detail Richness | 8 | Laplacian energy (view + texture) | Existing |
| D1 Material Plausibility | 8 | Metallic/roughness checks | Existing |
| D2 UV Quality | 7 | Texel density CV + utilization | **New** |
| E1 Multi-View Consistency | 10 | CV of per-view stats (or MEt3R) | Existing |
| **Total** | **100** | | |

---

## 10. References

### Mesh Integrity and Hole Detection
1. Shafiei et al., "Robust Hole-Detection in Triangular Meshes Irrespective of the Presence of Singular Vertices" (ScienceDirect, 2024) -- https://www.sciencedirect.com/science/article/pii/S001044852400023X | https://arxiv.org/abs/2311.12466
2. Li et al., "Normal-Guided Chamfer Distance Loss for Watertight Mesh Reconstruction" (CGF 2025) -- https://onlinelibrary.wiley.com/doi/10.1111/cgf.70088
3. trimesh documentation: https://trimesh.org/trimesh.html -- `is_watertight`, `outline()`, `split()`, `face_adjacency`
4. PyMeshLab filters: https://pymeshlab.readthedocs.io/en/latest/filter_list.html

### Surface Roughness and Smoothness
5. Lavoue, "A Roughness Measure for 3D Mesh Visual Masking" (APGV 2007) -- https://perso.liris.cnrs.fr/guillaume.lavoue/conference/APGV2007.pdf
6. Lavoue et al., "A fast roughness-based approach to the assessment of 3D mesh visual quality" (CG 2012) -- https://www.sciencedirect.com/science/article/abs/pii/S0097849312001203
7. Sorkine, "Laplacian Mesh Processing" (Eurographics STAR 2005)
8. "Blind Mesh Assessment Based on Graph Spectral Entropy and Spatial Features" (PMC 2020) -- https://pmc.ncbi.nlm.nih.gov/articles/PMC7516613/

### Texture Quality and UV Assessment
9. Shafiei et al., "Perceptual Crack Detection for Rendered 3D Textured Meshes" (IEEE QoMEX 2024) -- https://arxiv.org/abs/2405.06143 | https://github.com/arshafiee/crack-detection-VVM
10. "HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment" (arXiv 2024) -- https://arxiv.org/abs/2412.01986
11. "GeodesicPSIM: Predicting Quality of Static Mesh with Texture Map" (arXiv 2023/2025) -- https://arxiv.org/abs/2308.04928
12. "FMQM: Textured Mesh Quality Using Geometry and Color Field Similarity" (arXiv 2025) -- https://arxiv.org/abs/2505.10824
13. Nehme et al., "Textured Mesh Quality Assessment" (ACM TOG 2023) -- https://dl.acm.org/doi/10.1145/3592786
14. Li et al., "SeamCrafter: Enhancing Mesh Seam Generation via Reinforcement Learning" (arXiv 2025) -- https://arxiv.org/abs/2509.20725
15. "ArtUV: Artist-style UV Unwrapping" (arXiv 2025) -- https://arxiv.org/abs/2509.20710
16. Sloyd, "Texture Quality Metrics for 3D Models" -- https://www.sloyd.ai/blog/texture-quality-metrics-for-3d-models

### Evaluation Frameworks
17. Hi3DEval (NeurIPS 2025) -- https://arxiv.org/abs/2508.05609
18. GPTEval3D (CVPR 2024) -- https://arxiv.org/abs/2401.04092
19. HyperScore / MATE-3D (ICCV 2025) -- https://arxiv.org/abs/2412.11170
20. 3DGen-Bench (2025) -- https://arxiv.org/abs/2503.21745
21. SRAM (2025) -- https://arxiv.org/abs/2512.01373
22. Rank2Score / T23D-CompBench (2025) -- https://arxiv.org/abs/2509.23841

### Perceptual Metrics
23. DreamSim (NeurIPS 2023) -- https://arxiv.org/abs/2306.09344
24. MEt3R (CVPR 2025) -- https://github.com/mohammadasim98/met3r
25. VQAScore (ECCV 2024) -- https://github.com/linzhiqiu/t2v_metrics
26. LAION Aesthetic Predictor -- https://github.com/LAION-AI/aesthetic-predictor

### Commercial Standards
27. TurboSquid CheckMate Pro v2 Specifications -- https://resources.turbosquid.com/checkmate/checkmate-specifications/checkmate-specifications-overview/
28. TurboSquid CheckMate Topology Requirements -- https://blog.turbosquid.com/2013/06/13/checkmate-pro-v2-submissions-and-topology-requirements/
29. SimInsights, "Is AI Ready for High-Quality 3D Assets?" (2025) -- https://www.siminsights.com/ai-3d-generators-2025-production-readiness/
30. 3D Arena (2025) -- https://arxiv.org/abs/2506.18787

### Libraries and Tools
31. trimesh (Python): https://trimesh.org/ -- `pip install trimesh`
32. PyMeshLab (Python): https://pymeshlab.readthedocs.io/ -- `pip install pymeshlab`
33. PCD crack detection code: https://github.com/arshafiee/crack-detection-VVM
34. Open-CLIP: https://github.com/mlfoundations/open_clip
35. DreamSim: `pip install dreamsim`
