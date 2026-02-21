# Practical Mesh Quality Detection & Scoring: Tools, Libraries, and Implementation Guide

**Date**: 2026-02-22
**Author**: Research Optimizer Agent
**Scope**: Code-level solutions for 3D mesh defect detection and quality scoring applicable to TRELLIS.2 GLB outputs
**Target**: Building a comprehensive automated quality evaluation pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Library Capabilities Matrix](#2-library-capabilities-matrix)
3. [Trimesh: Complete Mesh Quality API](#3-trimesh-complete-mesh-quality-api)
4. [Open3D: Topology and Manifold Analysis](#4-open3d-topology-and-manifold-analysis)
5. [PyVista/VTK: Cell Quality Metrics](#5-pyvistavtk-cell-quality-metrics)
6. [PyMeshLab: Quality Filters](#6-pymeshlab-quality-filters)
7. [PyTorch3D: Differentiable Mesh Losses](#7-pytorch3d-differentiable-mesh-losses)
8. [No-Reference Image Quality (pyiqa)](#8-no-reference-image-quality-pyiqa)
9. [VLM-Based 3D Evaluation](#9-vlm-based-3d-evaluation)
10. [Texture and PBR Quality Checks](#10-texture-and-pbr-quality-checks)
11. [Industry Quality Standards](#11-industry-quality-standards)
12. [Recommended Evaluation Pipeline](#12-recommended-evaluation-pipeline)
13. [Implementation Plan](#13-implementation-plan)

---

## 1. Executive Summary

This survey identifies **47 concrete, implementable quality metrics** across 6 libraries and 4 evaluation paradigms. The metrics fall into three tiers:

**Tier 1 - Already Available (trimesh 4.11 installed in container)**:
- Watertight/manifold checks, Euler number, body count, degenerate face detection
- Face angle distribution, edge length ratio, dihedral angle analysis
- Normal consistency, face area statistics, boundary detection

**Tier 2 - Easy to Add (pip install, <1 day)**:
- pyiqa: 30+ no-reference image quality metrics (MUSIQ, TOPIQ, CLIPIQA, BRISQUE, NIQE)
- DreamSim: human-aligned perceptual similarity (already installed)
- Open3D: non-manifold edge/vertex detection, self-intersection, orientability
- PyMeshLab: Hausdorff distance, aspect ratio per face, curvature analysis

**Tier 3 - Medium Effort (1-3 days)**:
- Hi3DEval: hierarchical 3D evaluation (5 object dims + material dims)
- GPTEval3D: VLM-based pairwise comparison with ELO scoring
- Q-Align: VLM-based quality scoring
- PyTorch3D: differentiable mesh losses (Laplacian smoothing, normal consistency, edge loss)

### Key Finding from Real GLB Analysis

Running trimesh analysis on an actual TRELLIS.2 output (`e6_round0_baseline.glb`) revealed:
- **NOT watertight** (is_watertight=False, body_count=5631)
- 44 degenerate faces out of 490K (0.01% -- acceptable)
- 1,136 skinny triangles (min angle < 5 deg, 0.23% -- acceptable)
- 17,937 non-smooth face pairs (2.69% -- moderate concern)
- 16 inverted normals (0.002% -- acceptable)
- Edge length ratio 120:1 (high, indicates non-uniform tessellation)

These are **expected characteristics** of generated meshes (not CAD models), but the non-watertight status and high body count are actionable quality signals.

---

## 2. Library Capabilities Matrix

| Capability | trimesh | Open3D | PyVista | PyMeshLab | PyTorch3D | pyiqa |
|------------|---------|--------|---------|-----------|-----------|-------|
| **Already installed** | YES | no | no | no | no | no |
| Watertight check | YES | YES | - | YES | - | - |
| Non-manifold edges | YES* | YES | - | YES | - | - |
| Self-intersection | - | YES | - | YES | - | - |
| Degenerate faces | YES | YES | - | YES | - | - |
| Face angle quality | YES | - | YES | YES | - | - |
| Aspect ratio | partial | - | YES | YES | - | - |
| Hausdorff distance | - | YES | - | YES | - | - |
| Curvature analysis | integral | - | - | YES | - | - |
| Normal consistency | manual | manual | - | - | YES | - |
| Laplacian smoothness | - | - | - | - | YES | - |
| Edge length loss | - | - | - | - | YES | - |
| No-ref image quality | - | - | - | - | - | YES |
| Perceptual similarity | - | - | - | - | - | YES |
| Aesthetic scoring | - | - | - | - | - | YES |

*trimesh: inferred from is_watertight (edges in exactly 2 faces)

---

## 3. Trimesh: Complete Mesh Quality API

**Status**: Already installed (v4.11.1) in the trellis2 container.

### 3.1 Topology and Manifold Checks

```python
import trimesh
import numpy as np

def mesh_topology_report(glb_path: str) -> dict:
    """Complete topology quality report for a GLB file."""
    scene = trimesh.load(glb_path)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene

    report = {
        # --- Topology ---
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'edge_count': len(mesh.edges_unique),
        'is_watertight': bool(mesh.is_watertight),
        'is_volume': bool(mesh.is_volume),
        'is_winding_consistent': bool(mesh.is_winding_consistent),
        'is_convex': bool(mesh.is_convex),
        'euler_number': int(mesh.euler_number),
        'body_count': int(mesh.body_count),
        'surface_area': float(mesh.area),
    }

    if mesh.is_watertight:
        report['volume'] = float(mesh.volume)
        report['compactness'] = float(
            (36 * np.pi * mesh.volume**2) / mesh.area**3
        )  # sphere=1, lower=more complex

    return report
```

### 3.2 Face Quality Analysis

```python
def face_quality_report(mesh: trimesh.Trimesh) -> dict:
    """Per-face quality metrics."""
    areas = mesh.area_faces
    angles_deg = np.degrees(mesh.face_angles)

    report = {
        # --- Face areas ---
        'face_area_min': float(areas.min()),
        'face_area_max': float(areas.max()),
        'face_area_mean': float(areas.mean()),
        'face_area_std': float(areas.std()),
        'face_area_cv': float(areas.std() / max(areas.mean(), 1e-12)),  # coefficient of variation
        'degenerate_faces': int((areas < 1e-10).sum()),
        'degenerate_pct': float((areas < 1e-10).mean() * 100),

        # --- Face angles ---
        'min_angle_global': float(angles_deg.min()),
        'max_angle_global': float(angles_deg.max()),
        'mean_angle': float(angles_deg.mean()),  # ideal = 60 for equilateral
        'skinny_triangles': int((angles_deg.min(axis=1) < 5).sum()),  # min angle < 5 deg
        'skinny_pct': float((angles_deg.min(axis=1) < 5).mean() * 100),
        'obtuse_triangles': int((angles_deg.max(axis=1) > 150).sum()),  # max angle > 150 deg
        'obtuse_pct': float((angles_deg.max(axis=1) > 150).mean() * 100),
    }

    # Aspect ratio proxy: ratio of largest to smallest face area
    sorted_areas = np.sort(areas)
    p05, p95 = np.percentile(areas, [5, 95])
    report['face_area_ratio_p95_p05'] = float(p95 / max(p05, 1e-12))

    return report
```

### 3.3 Edge Quality Analysis

```python
def edge_quality_report(mesh: trimesh.Trimesh) -> dict:
    """Edge length distribution and quality."""
    lengths = mesh.edges_unique_length

    return {
        'edge_length_min': float(lengths.min()),
        'edge_length_max': float(lengths.max()),
        'edge_length_mean': float(lengths.mean()),
        'edge_length_ratio': float(lengths.max() / max(lengths.min(), 1e-12)),
        'edge_length_cv': float(lengths.std() / max(lengths.mean(), 1e-12)),
    }
```

### 3.4 Normal Consistency (Surface Smoothness)

```python
def normal_consistency_report(mesh: trimesh.Trimesh) -> dict:
    """Face normal alignment across adjacent faces."""
    fn = mesh.face_normals
    adj = mesh.face_adjacency

    if len(adj) == 0:
        return {'error': 'no face adjacency'}

    # Cosine similarity between adjacent face normals
    n1 = fn[adj[:, 0]]
    n2 = fn[adj[:, 1]]
    cos_sim = (n1 * n2).sum(axis=1)

    # Dihedral angles
    dihedral_deg = np.degrees(mesh.face_adjacency_angles)

    return {
        'normal_cos_mean': float(cos_sim.mean()),
        'normal_cos_min': float(cos_sim.min()),
        'normal_cos_std': float(cos_sim.std()),
        'non_smooth_pct': float((cos_sim < 0.9).mean() * 100),  # > ~26 deg dihedral
        'inverted_pct': float((cos_sim < 0).mean() * 100),       # > 90 deg dihedral
        'dihedral_mean_deg': float(dihedral_deg.mean()),
        'dihedral_max_deg': float(dihedral_deg.max()),
        'sharp_crease_pct': float((dihedral_deg > 45).mean() * 100),
    }
```

### 3.5 Body/Component Analysis

```python
def component_report(mesh: trimesh.Trimesh) -> dict:
    """Connected component analysis -- critical for detecting fragmentation."""
    components = mesh.split(only_watertight=False)

    if len(components) == 0:
        return {'body_count': 0}

    areas = [c.area for c in components]
    verts = [len(c.vertices) for c in components]
    total_area = sum(areas)

    # Largest component dominance
    max_area = max(areas)

    return {
        'body_count': len(components),
        'largest_body_area_pct': float(max_area / max(total_area, 1e-12) * 100),
        'small_bodies': int(sum(1 for a in areas if a < total_area * 0.01)),  # < 1% of total
        'tiny_bodies': int(sum(1 for v in verts if v < 10)),  # < 10 vertices
        'body_area_distribution': [float(a / max(total_area, 1e-12) * 100) for a in sorted(areas, reverse=True)[:10]],
    }
```

---

## 4. Open3D: Topology and Manifold Analysis

**Status**: NOT installed. Install: `pip install open3d`

### 4.1 Key Quality Methods

```python
import open3d as o3d

def open3d_topology_report(glb_path: str) -> dict:
    """Open3D provides explicit non-manifold and self-intersection detection."""
    mesh = o3d.io.read_triangle_mesh(glb_path)
    mesh.compute_vertex_normals()

    return {
        'is_edge_manifold': mesh.is_edge_manifold(allow_boundary_edges=True),
        'is_vertex_manifold': mesh.is_vertex_manifold(),
        'is_watertight': mesh.is_watertight(),
        'is_orientable': mesh.is_orientable(),
        'is_self_intersecting': mesh.is_self_intersecting(),
        'non_manifold_edges': len(mesh.get_non_manifold_edges(allow_boundary_edges=True)),
        'non_manifold_vertices': len(mesh.get_non_manifold_vertices()),
        'self_intersecting_triangles': len(mesh.get_self_intersecting_triangles()),
    }
```

### 4.2 Distance Metrics (Open3D 0.19+)

```python
# Chamfer Distance, Hausdorff Distance, F-Score between two meshes
import open3d as o3d

def mesh_distance_metrics(mesh_a, mesh_b, threshold=0.01):
    """Compare two meshes using Open3D's distance metrics."""
    pcd_a = mesh_a.sample_points_uniformly(number_of_points=10000)
    pcd_b = mesh_b.sample_points_uniformly(number_of_points=10000)

    # Point-to-point distances
    dists_a = pcd_a.compute_point_cloud_distance(pcd_b)
    dists_b = pcd_b.compute_point_cloud_distance(pcd_a)

    return {
        'chamfer_distance': float(np.mean(dists_a) + np.mean(dists_b)) / 2,
        'hausdorff_distance': float(max(np.max(dists_a), np.max(dists_b))),
        'precision': float(np.mean(np.array(dists_b) < threshold)),
        'recall': float(np.mean(np.array(dists_a) < threshold)),
    }
```

---

## 5. PyVista/VTK: Cell Quality Metrics

**Status**: NOT installed. Install: `pip install pyvista`

### 5.1 Available Quality Measures for Triangles

PyVista wraps VTK's `vtkMeshQuality` filter, providing these per-cell metrics:

| Measure | What it computes | Ideal | Bad |
|---------|-----------------|-------|-----|
| `area` | Triangle area | uniform | near-zero |
| `aspect_ratio` | Longest edge / shortest altitude | 1.0 | >20 |
| `aspect_frobenius` | Frobenius norm ratio | 1.0 | >3 |
| `condition` | Condition number of Jacobian | 1.0 | >10 |
| `distortion` | Distortion metric | 1.0 | <0.5 |
| `max_angle` | Maximum interior angle | 60 | >170 |
| `min_angle` | Minimum interior angle | 60 | <5 |
| `radius_ratio` | Circumradius / (2 * inradius) | 1.0 | >10 |
| `scaled_jacobian` | Scaled Jacobian determinant | 1.0 | <0.3 |
| `shape` | Shape metric (2D) | 1.0 | <0.3 |
| `shape_and_size` | Shape + size quality | 1.0 | <0.3 |

```python
import pyvista as pv

def pyvista_quality_report(glb_path: str) -> dict:
    """Cell-level quality analysis using VTK."""
    mesh = pv.read(glb_path)

    report = {}
    for measure in ['aspect_ratio', 'min_angle', 'max_angle', 'scaled_jacobian', 'shape']:
        qual = mesh.compute_cell_quality(quality_measure=measure)
        values = qual.cell_data['CellQuality']
        report[f'{measure}_mean'] = float(np.mean(values))
        report[f'{measure}_p05'] = float(np.percentile(values, 5))
        report[f'{measure}_p95'] = float(np.percentile(values, 95))

    return report
```

---

## 6. PyMeshLab: Quality Filters

**Status**: NOT installed. Install: `pip install pymeshlab`

### 6.1 Key Quality Filters

```python
import pymeshlab

def pymeshlab_quality_report(glb_path: str) -> dict:
    """PyMeshLab's comprehensive quality analysis."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(glb_path)

    report = {}

    # 1. Geometric measures (volume, surface area, etc.)
    geo = ms.get_geometric_measures()
    report['geo_measures'] = geo

    # 2. Topological measures (genus, manifold properties)
    topo = ms.get_topological_measures()
    report['topo_measures'] = topo

    # 3. Compute aspect ratio per face
    ms.compute_scalar_by_aspect_ratio_per_face()

    # 4. Detect non-manifold edges
    ms.compute_selection_by_non_manifold_edges_per_face()
    non_manifold_count = ms.current_mesh().selected_face_number()
    report['non_manifold_faces'] = non_manifold_count
    ms.set_selection_none()

    # 5. Detect self-intersections
    ms.compute_selection_by_self_intersections_per_face()
    self_intersect_count = ms.current_mesh().selected_face_number()
    report['self_intersecting_faces'] = self_intersect_count
    ms.set_selection_none()

    # 6. Detect bad faces (degenerate triangles)
    ms.compute_selection_bad_faces()
    bad_count = ms.current_mesh().selected_face_number()
    report['bad_faces'] = bad_count

    # 7. Hausdorff distance (if comparing two meshes)
    # ms.load_new_mesh(reference_path)
    # hausdorff = ms.get_hausdorff_distance()

    return report
```

---

## 7. PyTorch3D: Differentiable Mesh Losses

**Status**: NOT installed. Heavy dependency (CUDA build required).

These are differentiable and can be used as quality metrics OR optimization objectives.

```python
from pytorch3d.loss import (
    mesh_edge_loss,         # Penalizes long edges
    mesh_laplacian_smoothing,  # Penalizes non-smooth surfaces
    mesh_normal_consistency,   # Penalizes adjacent-face normal deviation
)
from pytorch3d.structures import Meshes

def pytorch3d_quality_metrics(verts_tensor, faces_tensor) -> dict:
    """Differentiable mesh quality metrics."""
    meshes = Meshes(verts=[verts_tensor], faces=[faces_tensor])

    return {
        'edge_loss': float(mesh_edge_loss(meshes).item()),
        'laplacian_smoothing': float(mesh_laplacian_smoothing(meshes, method="uniform").item()),
        'normal_consistency': float(mesh_normal_consistency(meshes).item()),
    }
```

**Note**: For our use case (evaluation, not training), the trimesh-based normal consistency computation in Section 3.4 achieves the same result without the PyTorch3D dependency.

---

## 8. No-Reference Image Quality (pyiqa)

**Status**: NOT installed. Install: `pip install pyiqa`

### 8.1 Key Metrics for 3D Rendered Views

The `pyiqa` library provides 30+ no-reference image quality metrics through a unified API. For evaluating rendered 3D views:

```python
import pyiqa
import torch

class RenderQualityScorer:
    """Score rendered 3D views using no-reference image quality metrics."""

    def __init__(self, device='cuda'):
        self.device = device
        self.metrics = {
            # General quality
            'musiq': pyiqa.create_metric('musiq', device=device),        # Multi-scale, higher=better
            'topiq_nr': pyiqa.create_metric('topiq_nr', device=device),  # SOTA NR metric, higher=better
            'clipiqa': pyiqa.create_metric('clipiqa+', device=device),   # CLIP-based, higher=better

            # Traditional
            'brisque': pyiqa.create_metric('brisque', device=device),    # NSS-based, lower=better
            'niqe': pyiqa.create_metric('niqe', device=device),          # Blind, lower=better

            # Aesthetic
            'nima': pyiqa.create_metric('nima', device=device),          # Neural aesthetic, higher=better
        }

    def score_render(self, image_tensor: torch.Tensor) -> dict:
        """Score a single rendered view. image_tensor: (1, 3, H, W), 0-1 range."""
        scores = {}
        for name, metric in self.metrics.items():
            with torch.no_grad():
                score = metric(image_tensor)
            scores[name] = float(score.item())
            # Normalize direction (all to higher=better)
            if metric.lower_better:
                scores[f'{name}_quality'] = -scores[name]  # Invert for consistent comparison
            else:
                scores[f'{name}_quality'] = scores[name]
        return scores
```

### 8.2 Recommended Metrics for 3D Evaluation

| Metric | Type | Speed | Why use it |
|--------|------|-------|-----------|
| **MUSIQ** | NR | ~20ms | Multi-scale, handles different resolutions well |
| **TOPIQ-NR** | NR | ~15ms | SOTA accuracy, trained on large datasets |
| **CLIPIQA+** | NR | ~10ms | CLIP-based, semantic awareness, good for generated content |
| **BRISQUE** | NR | ~5ms | Fast baseline, NSS-based, well-studied |
| **NIQE** | NR | ~8ms | Completely blind (no training on distorted images) |
| **NIMA** | NR | ~10ms | Aesthetic quality, trained on AVA dataset |
| **DreamSim** | FR | ~15ms | Best human alignment for perceptual similarity (already installed) |

---

## 9. VLM-Based 3D Evaluation

### 9.1 GPTEval3D

**Repository**: https://github.com/3DTopia/GPTEval3D (CVPR 2024)

**Method**: Renders 120 evenly-spaced views (512x512, RGB + normals), sends pairs to GPT-4V for comparison, computes ELO ratings.

**Evaluation Dimensions**:
1. Text-Asset Alignment
2. 3D Plausibility
3. Texture Detail
4. Geometry Detail
5. Texture-Geometry Coherency

**Practical for us**: Too expensive for per-config evaluation (requires many API calls), but useful for periodic benchmarking.

### 9.2 Hi3DEval

**Repository**: https://github.com/3DTopia/Hi3DEval (NeurIPS 2025)

**Hierarchical Evaluation**:
- **Object-level** (5 dims): Geometry Plausibility, Geometry Details, Texture Quality, Geometry-Texture Coherence, 3D-Prompt Alignment
- **Material-level**: Albedo, Saturation, Metallicness assessment

**Scoring**: Video-based representations processed by InternVideo2.5 encoder + PartField 3D features. Achieves Kendall tau 0.774 (vs GPTEval3D 0.690).

**Practical for us**: More practical than GPTEval3D (open-source, runs locally), but requires significant setup.

### 9.3 Q-Align

**Repository**: https://github.com/Q-Future/Q-Align (ICML 2024)

**Method**: Fine-tuned LMM that outputs text-defined quality levels ("good", "fair", "poor") from visual input. Supports IQA, IAA, and VQA modes.

**Practical for us**: Could score rendered views, but heavy model (~13B parameters).

---

## 10. Texture and PBR Quality Checks

### 10.1 Texture Map Analysis (implementable now with trimesh + numpy)

```python
import numpy as np
import cv2
from PIL import Image

def texture_quality_report(glb_path: str) -> dict:
    """Analyze texture quality from a GLB file."""
    import trimesh
    scene = trimesh.load(glb_path)
    geom = list(scene.geometry.values())[0]

    report = {}

    # --- Base Color Texture ---
    if hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'baseColorTexture'):
        tex = np.array(geom.visual.material.baseColorTexture)
        rgb = tex[:, :, :3].astype(np.float32)

        # 1. Resolution
        report['texture_resolution'] = f'{tex.shape[1]}x{tex.shape[0]}'
        report['texture_megapixels'] = float(tex.shape[0] * tex.shape[1] / 1e6)

        # 2. Utilization (non-black area)
        luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        utilized = luminance > 5
        report['utilization_pct'] = float(utilized.mean() * 100)

        # 3. Grey/desaturated detection
        max_c = np.maximum(np.maximum(rgb[:,:,0], rgb[:,:,1]), rgb[:,:,2])
        min_c = np.minimum(np.minimum(rgb[:,:,0], rgb[:,:,1]), rgb[:,:,2])
        chroma = max_c - min_c
        saturation = np.where(max_c > 0, chroma / max_c, 0)
        report['grey_texel_pct'] = float((saturation[utilized] < 0.05).mean() * 100)
        report['mean_saturation'] = float(saturation[utilized].mean())

        # 4. Dark texel detection (within utilized area)
        report['dark_texel_pct'] = float((luminance[utilized] < 30).mean() * 100)
        report['very_dark_texel_pct'] = float((luminance[utilized] < 10).mean() * 100)

        # 5. Texture detail (Laplacian energy)
        gray = cv2.cvtColor(tex[:,:,:3], cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        report['laplacian_energy'] = float(np.abs(lap[utilized]).mean())

        # 6. Speckle/noise detection
        median = cv2.medianBlur(tex[:,:,:3], 3)
        diff = np.abs(tex[:,:,:3].astype(np.float32) - median.astype(np.float32)).max(axis=-1)
        report['speckle_ratio'] = float(((diff > 40) & utilized).sum() / max(utilized.sum(), 1) * 100)

        # 7. Color distribution entropy (higher = more diverse colors)
        for i, ch_name in enumerate(['r', 'g', 'b']):
            hist = np.histogram(rgb[utilized, i], bins=64, range=(0, 256))[0]
            hist = hist / max(hist.sum(), 1)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            report[f'{ch_name}_entropy'] = float(entropy)

    # --- PBR MetallicRoughness Texture ---
    if hasattr(geom.visual.material, 'metallicRoughnessTexture') and \
       geom.visual.material.metallicRoughnessTexture is not None:
        mr = np.array(geom.visual.material.metallicRoughnessTexture)

        # glTF PBR convention: G=roughness, B=metallic
        roughness = mr[:, :, 1].astype(np.float32) / 255.0
        metallic = mr[:, :, 2].astype(np.float32) / 255.0

        report['roughness_mean'] = float(roughness.mean())
        report['roughness_std'] = float(roughness.std())
        report['metallic_mean'] = float(metallic.mean())
        report['metallic_max'] = float(metallic.max())

        # PBR plausibility checks
        report['pbr_metallic_high_pct'] = float((metallic > 0.5).mean() * 100)  # Should be low for most objects
        report['pbr_roughness_low_pct'] = float((roughness < 0.2).mean() * 100)  # Mirror-like, unusual
        report['pbr_roughness_high_pct'] = float((roughness > 0.9).mean() * 100)  # Very rough

    # --- UV Quality ---
    if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
        uv = geom.visual.uv
        report['uv_range_min'] = float(uv.min())
        report['uv_range_max'] = float(uv.max())
        report['uv_in_01_pct'] = float(
            (np.all(uv >= 0, axis=1) & np.all(uv <= 1, axis=1)).mean() * 100
        )

        # UV edge discontinuity (proxy for seam density)
        edges = geom.edges_unique
        uv_edge = uv[edges]
        uv_edge_len = np.linalg.norm(uv_edge[:, 0] - uv_edge[:, 1], axis=1)
        report['uv_edge_len_mean'] = float(uv_edge_len.mean())
        report['uv_edge_len_max'] = float(uv_edge_len.max())

    return report
```

### 10.2 PBR Plausibility Scoring

```python
def pbr_plausibility_score(report: dict) -> float:
    """Score PBR material plausibility (0-100)."""
    score = 100.0

    # High metallic is unusual for most generated objects
    if report.get('pbr_metallic_high_pct', 0) > 10:
        score -= min(20, (report['pbr_metallic_high_pct'] - 10) * 2)

    # Very low roughness (mirror-like) is unusual
    if report.get('pbr_roughness_low_pct', 0) > 20:
        score -= min(15, (report['pbr_roughness_low_pct'] - 20) * 1)

    # Grey/desaturated texture is a known TRELLIS.2 artifact
    if report.get('grey_texel_pct', 0) > 5:
        score -= min(20, (report['grey_texel_pct'] - 5) * 2)

    # Low utilization means wasted UV space
    if report.get('utilization_pct', 100) < 80:
        score -= min(15, (80 - report['utilization_pct']) * 1)

    # Very dark texels within utilized area (often artifacts)
    if report.get('very_dark_texel_pct', 0) > 2:
        score -= min(15, (report['very_dark_texel_pct'] - 2) * 3)

    # Low texture detail
    if report.get('laplacian_energy', 15) < 5:
        score -= min(15, (5 - report['laplacian_energy']) * 3)

    return max(0, score)
```

---

## 11. Industry Quality Standards

### 11.1 Game-Ready Quality Checklist (consolidated from Threedium, Polycount, Sketchfab)

**Topology**:
- No non-manifold edges
- No degenerate faces (area = 0)
- All normals facing outward, consistent winding
- Polygon budget: 20K-120K for hero assets, 500-5K for background

**UV Mapping**:
- All UV shells packed in [0,1] space
- No overlapping UVs (unless intentional mirroring)
- Uniform texel density across surface
- 0.5-1 pixel padding between UV islands to prevent bleeding

**Textures**:
- Minimum 2048x2048 for hero assets
- PBR textures: albedo, normal, roughness, metallic
- BC7/ASTC compression for real-time
- Consistent texel density (no stretch/compression artifacts)

**Materials**:
- Albedo should NOT contain lighting information (no baked shadows)
- Metallic should be mostly 0 or 1 (not intermediate for non-conductors)
- Roughness should have spatial variation (not flat values)

### 11.2 3D Printing Requirements

- Must be watertight (manifold)
- No self-intersections
- Minimum wall thickness > 0.5mm
- No inverted normals

### 11.3 Sketchfab Seller Guidelines

- FBX/OBJ preferred formats
- Optimized polycount for real-time rendering
- Clean topology (no n-gons, no hidden faces, no overlapping geometry)
- Proper UV unwrapping with no stretching
- PBR materials configured correctly

---

## 12. Recommended Evaluation Pipeline

### Pipeline Order (what to check, in what order)

```
Phase 1: Mesh Integrity (fast, binary pass/fail)
  1. Load GLB → extract mesh
  2. Watertight check                    [trimesh.is_watertight]
  3. Winding consistency                 [trimesh.is_winding_consistent]
  4. Degenerate face count               [trimesh.area_faces < 1e-10]
  5. Body count                          [trimesh.body_count]

Phase 2: Geometry Quality (fast, continuous scores)
  6. Face angle distribution             [trimesh.face_angles]
  7. Edge length ratio                   [trimesh.edges_unique_length]
  8. Normal consistency                  [manual from face_normals + face_adjacency]
  9. Dihedral angle distribution         [trimesh.face_adjacency_angles]
  10. Component analysis                 [trimesh.split()]

Phase 3: Texture Quality (fast, continuous scores)
  11. Texture resolution check           [material.baseColorTexture.shape]
  12. Utilization / dead texel ratio     [luminance > threshold]
  13. Grey/desaturated texel ratio       [saturation < 0.05]
  14. Dark artifact detection            [luminance < 10]
  15. Speckle/noise detection            [median filter difference]
  16. PBR plausibility                   [metallic/roughness distributions]
  17. Texture detail (Laplacian energy)  [cv2.Laplacian]

Phase 4: Rendered View Quality (medium, requires GPU rendering)
  18. Multi-view renders (8 views)       [render_mesh_at_views()]
  19. CLIP-S input alignment             [open_clip_torch]
  20. DreamSim perceptual similarity     [dreamsim, already installed]
  21. No-reference quality (MUSIQ/TOPIQ) [pyiqa]
  22. Aesthetic score (NIMA/LAION)       [pyiqa]

Phase 5: Input Alignment (requires reference image)
  23. Silhouette IoU                     [existing evaluate.py]
  24. Contour/edge alignment             [existing evaluate.py]
  25. Color fidelity                     [existing evaluate.py]
```

### Scoring Weights (recommended)

```python
QUALITY_WEIGHTS = {
    # Phase 1: Mesh Integrity (15 points)
    'watertight_bonus': 3,        # +3 if watertight
    'winding_consistent': 3,      # +3 if consistent
    'degenerate_penalty': -3,     # -3 if > 0.1% degenerate
    'body_count_penalty': -3,     # -3 per 1000 extra bodies
    'component_integrity': 3,     # +3 if single dominant body > 95%

    # Phase 2: Geometry Quality (20 points)
    'normal_consistency': 8,      # 0-8 based on mean cos similarity
    'angle_quality': 4,           # 0-4 based on skinny/obtuse %
    'edge_uniformity': 4,         # 0-4 based on edge length CV
    'smoothness': 4,              # 0-4 from dihedral angle distribution

    # Phase 3: Texture Quality (20 points)
    'texture_resolution': 4,      # 0-4 based on megapixels
    'utilization': 4,             # 0-4 based on utilized area %
    'color_health': 4,            # 0-4 (grey + dark + speckle penalties)
    'pbr_plausibility': 4,        # 0-4 from pbr_plausibility_score()
    'texture_detail': 4,          # 0-4 from Laplacian energy

    # Phase 4: Rendered View Quality (25 points)
    'clip_s_alignment': 8,        # 0-8 CLIP similarity to input
    'perceptual_quality': 7,      # 0-7 from MUSIQ/TOPIQ average
    'dreamsim_similarity': 5,     # 0-5 DreamSim to input views
    'aesthetic_score': 5,         # 0-5 from NIMA/LAION

    # Phase 5: Input Alignment (20 points)
    'silhouette_iou': 8,          # 0-8 from existing evaluator
    'contour_alignment': 6,       # 0-6 from existing evaluator
    'color_fidelity': 6,          # 0-6 from existing evaluator
}
# Total: 100 points
```

---

## 13. Implementation Plan

### Phase 1: Trimesh-Based Mesh Quality (0 days -- already installed)

Add mesh quality scoring to existing `evaluate.py`. No new dependencies.

```python
# Add to evaluate.py

def evaluate_mesh_quality(glb_path: str) -> dict:
    """Complete mesh quality evaluation using trimesh (already installed)."""
    import trimesh
    scene = trimesh.load(glb_path)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene

    topo = mesh_topology_report_from_mesh(mesh)
    faces = face_quality_report(mesh)
    edges = edge_quality_report(mesh)
    normals = normal_consistency_report(mesh)
    components = component_report(mesh)

    # Combine into overall mesh quality score (0-100)
    score = 100.0

    # Topology penalties
    if not topo['is_watertight']:
        score -= 5
    if not topo['is_winding_consistent']:
        score -= 10
    if topo['body_count'] > 100:
        score -= min(15, topo['body_count'] / 100)

    # Face quality penalties
    score -= min(10, faces['degenerate_pct'] * 100)
    score -= min(10, faces['skinny_pct'] * 2)

    # Normal consistency
    smooth_bonus = max(0, (normals['normal_cos_mean'] - 0.9) * 100)
    score += min(10, smooth_bonus) - 5  # baseline at 0.95

    # Non-smooth penalty
    score -= min(10, normals['non_smooth_pct'])

    return {
        'mesh_quality_score': max(0, min(100, score)),
        'topology': topo,
        'face_quality': faces,
        'edge_quality': edges,
        'normal_consistency': normals,
        'components': components,
    }
```

### Phase 2: Texture Quality (0 days -- trimesh + numpy)

```python
# Add texture_quality_report() and pbr_plausibility_score() from Section 10
# These use only trimesh (installed) + numpy + cv2 (installed)
```

### Phase 3: No-Reference Image Quality (pip install pyiqa, ~30 min)

```bash
docker exec -u root trellis2 pip install pyiqa
```

```python
# Add RenderQualityScorer from Section 8
# Score each rendered view with MUSIQ + TOPIQ + CLIPIQA
```

### Phase 4: CLIP-S Input Alignment (open_clip already installed, ~1 hr)

```python
import open_clip

def clip_similarity(input_image, rendered_views) -> float:
    """CLIP-based input-output alignment score."""
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    input_features = model.encode_image(preprocess(input_image).unsqueeze(0))

    scores = []
    for view in rendered_views:
        view_features = model.encode_image(preprocess(view).unsqueeze(0))
        sim = torch.nn.functional.cosine_similarity(input_features, view_features)
        scores.append(sim.item())

    return float(np.mean(scores))
```

### Phase 5: DreamSim Perceptual (already installed, ~30 min)

```python
from dreamsim import dreamsim

def dreamsim_similarity(input_image, rendered_views) -> float:
    """DreamSim perceptual similarity (already installed)."""
    model, preprocess = dreamsim(pretrained=True)

    input_tensor = preprocess(input_image).unsqueeze(0)

    distances = []
    for view in rendered_views:
        view_tensor = preprocess(view).unsqueeze(0)
        dist = model(input_tensor, view_tensor)
        distances.append(dist.item())

    # Convert distance to similarity score (0-1)
    mean_dist = np.mean(distances)
    similarity = max(0, 1.0 - mean_dist)
    return similarity
```

---

## Appendix: Sources

### Libraries
- [trimesh documentation](https://trimesh.org/trimesh.html)
- [Open3D TriangleMesh API](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)
- [PyVista mesh quality](https://docs.pyvista.org/examples/01-filter/mesh_quality.html)
- [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)
- [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch)
- [PyTorch3D mesh losses](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html)

### Evaluation Frameworks
- [GPTEval3D](https://github.com/3DTopia/GPTEval3D) (CVPR 2024)
- [Hi3DEval](https://github.com/3DTopia/Hi3DEval) (NeurIPS 2025)
- [Q-Align](https://github.com/Q-Future/Q-Align) (ICML 2024)

### Industry Standards
- [Sketchfab Seller Guidelines](https://help.sketchfab.com/hc/en-us/articles/115004276366-Seller-Guidelines)
- [Game-Ready Asset Checklist](https://medium.com/@prasenkakade21/is-your-3d-asset-actually-game-ready-a-no-nonsense-checklist-4557e50508f0)
- [Sloyd.ai Texture Quality Metrics](https://www.sloyd.ai/blog/texture-quality-metrics-for-3d-models)
- [Sloyd.ai 7 Key Metrics for 3D Quality](https://www.sloyd.ai/blog/top-7-metrics-for-evaluating-3d-model-quality)

### No-Reference IQA
- [pyiqa Model Card](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md)
- [pybrisque](https://pypi.org/project/pybrisque/)
- [NIQA algorithms](https://github.com/EadCat/NIQA)
