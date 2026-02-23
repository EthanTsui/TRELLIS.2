#!/usr/bin/env python3
"""
V4 Quality Evaluation Framework for TRELLIS.2 3D generation.

Resolves 7 critical flaws in v3:
  1. Smoothness-Detail conflict → B2 uses normal maps, C3 uses base_color
  2. Vibrancy ceiling (99-100) → Replaced by C2 Color Vitality
  3. Darkness floor (81-91) → Merged into C2 as dark-patch sub-metric
  4. Detail ceiling (99-100) → Recalibrated C3 with wider thresholds
  5. Camera mismatch → Uses r=2, fov=40 (same as render_video)
  6. Only 3 test images → Default 10, --quick 3, --full 10
  7. Missing principled metrics → Added mesh integrity, material, multi-view consistency

5 Categories, 9 Dimensions (100 pts):
  A. I/O Match:   A1 Silhouette(15) + A2 Color Distribution(10)
  B. Geometry:    B1 Mesh Integrity(10) + B2 Surface Quality(10)
  C. Texture:     C1 Coherence(15) + C2 Color Vitality(10) + C3 Detail Richness(10)
  D. Material:    D1 Material Plausibility(10)
  E. Consistency: E1 Multi-View Consistency(10)

Usage:
    python auto_evaluate_v4.py                    # Baseline champion config (10 images)
    python auto_evaluate_v4.py --quick            # Quick smoke test (3 images)
    python auto_evaluate_v4.py --full             # Full evaluation (10 images)
    python auto_evaluate_v4.py --sweep cfg_mp 0.1 0.15 0.2
    python auto_evaluate_v4.py --config path/to/adj.json
"""
import gc
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/workspace/TRELLIS.2')

import torch
import cv2


def _log_memory(label=""):
    """Log current memory usage for debugging OOM issues."""
    import psutil
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024**3)
    gpu_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
    print(f"  [MEM {label}] RSS={rss:.1f}GB  GPU_alloc={gpu_alloc:.1f}GB  GPU_reserved={gpu_reserved:.1f}GB",
          flush=True)

# 10 diverse test images spanning organic, hard-surface, mixed, edge-case categories
TEST_IMAGES_FULL = [
    'assets/example_image/T.png',  # steampunk device — metallic, complex geometry
    'assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp',  # ornate crown — gold, detailed
    'assets/example_image/454e7d8a30486c0635369936e7bec5677b78ae5f436d0e46af0d533738be859f.webp',  # diverse sample
    'assets/example_image/cd3c309f17eee5ad6afe4e001765893ade20b653f611365c93d158286b4cee96.webp',  # large image — complex scene
    'assets/example_image/ee8ecf658fde9c58830c021b2e30d0d5e7e492ef52febe7192a6c74fbf1b0472.webp',  # detailed object
    'assets/example_image/f351569ddc61116da4a7b929bccdab144d011f56b9603e6e72abea05236160f4.webp',  # varied category
    'assets/example_image/7d585a8475db078593486367d98b5efa9368a60a3528c555b96026a1a674aa54.webp',  # complex model
    'assets/example_image/e4d6b2f3a18c3e0f5146a5b40cda6c95d7f69372b2e741c023e5ec9661deda2b.webp',  # diverse object
    'assets/example_image/d39c2bd426456bd686de33f924524d18eb47343a5f080826aa3cb8e77de5147b.webp',  # another category
    'assets/example_image/5a6c81d3b2afca4323e4b8b379e2cf06d18371a57fc8c5dc24b57e60e3216690.webp',  # edge case
]

TEST_IMAGES_QUICK = TEST_IMAGES_FULL[:3]

# GA v2 Champion defaults
CHAMPION_CONFIG = {
    'ss_guidance_strength': 10.0,
    'ss_guidance_rescale': 0.8,
    'ss_sampling_steps': 12,
    'ss_rescale_t': 5.0,
    'shape_slat_guidance_strength': 10.0,
    'shape_slat_guidance_rescale': 0.5,
    'shape_slat_sampling_steps': 12,
    'shape_slat_rescale_t': 3.0,
    'tex_slat_guidance_strength': 12.0,
    'tex_slat_guidance_rescale': 1.0,
    'tex_slat_sampling_steps': 16,
    'tex_slat_rescale_t': 4.0,
    'cfg_mp_strength': 0.0,
    'heun_steps': 4,
    'multistep': True,
    'schedule': 'uniform',
    'schedule_rho': 7.0,
    'schedule_power': 2.0,
    # Bell-shaped guidance schedule for texture (validated +0.7 overall)
    'tex_guidance_schedule': 'beta',
    'tex_guidance_beta_a': 3.0,
    'tex_guidance_beta_b': 3.0,
    'resolution': '1024',
    'seed': 42,
    'decimation_target': 500000,
    'texture_size': 2048,
}

RESULTS_DIR = '/workspace/TRELLIS.2/optimization/research/adjustments'
OUTPUT_DIR = '/workspace/TRELLIS.2/optimization/test_outputs'


class QualityEvaluatorV4:
    """
    V4 Quality Evaluator — 5 categories, 9 dimensions, 100 pts total.

    A. I/O Match (25 pts):
        A1. Silhouette Match (15) — Alpha IoU at canonical camera
        A2. Color Distribution (10) — LAB histogram correlation

    B. Geometry (20 pts):
        B1. Mesh Integrity (10) — Trimesh diagnostics
        B2. Surface Quality (10) — Normal map Laplacian energy

    C. Texture (35 pts):
        C1. Texture Coherence (15) — Crack detection + gradient harshness
        C2. Color Vitality (10) — Grey patches + dynamic range + dark patches
        C3. Detail Richness (10) — Base_color Laplacian (views 60% + texture 40%)

    D. Material (10 pts):
        D1. Material Plausibility (10) — Metallic/roughness range checks

    E. Consistency (10 pts):
        E1. Multi-View Consistency (10) — CV of per-view quality metrics
    """

    def __init__(self):
        self.weights = {
            'A1_silhouette': 15,
            'A2_color_dist': 10,
            'B1_mesh_integrity': 10,
            'B2_surface_quality': 10,
            'C1_tex_coherence': 15,
            'C2_color_vitality': 10,
            'C3_detail_richness': 10,
            'D1_material': 10,
            'E1_multiview': 10,
        }
        self.total_weight = sum(self.weights.values())

    def evaluate(self, channels, reference_rgba=None, glb_mesh=None,
                 texture_map=None, tex_mask=None):
        """
        Main evaluation entry point.

        Args:
            channels: dict with keys 'base_color', 'normal', 'metallic',
                      'roughness', 'alpha' — each a list of (H,W,C) uint8 arrays
            reference_rgba: (H,W,4) uint8 array of the input image
            glb_mesh: trimesh.Trimesh object (for B1 mesh integrity)
            texture_map: (H,W,3) uint8 array of UV base color texture
            tex_mask: (H,W) bool array of valid texels

        Returns:
            dict with per-dimension scores (0-100) and weighted 'overall'
        """
        nviews = len(channels['base_color'])
        base_colors = channels['base_color']
        normals = channels['normal']
        metallics = channels['metallic']
        roughnesses = channels['roughness']
        alphas = channels['alpha']

        # Convert alpha images to single-channel masks
        alpha_masks = []
        for a in alphas:
            if a.ndim == 3:
                alpha_masks.append(a[:, :, 0])
            else:
                alpha_masks.append(a)

        scores = {}

        # A1. Silhouette Match (best-matching view among all views)
        scores['A1_silhouette'] = self._silhouette_match(
            base_colors[0], alpha_masks[0], reference_rgba,
            all_alphas=alpha_masks)

        # A2. Color Distribution (best-matching view)
        # Use shaded (PBR-lit) renders if available — fairer comparison
        # since input photos also have lighting baked in.
        # Falls back to base_color if shaded not available.
        a2_views = channels.get('shaded', base_colors)
        if len(a2_views) > 1:
            a2_scores = [self._color_distribution_match(
                sv, am, reference_rgba)
                for sv, am in zip(a2_views, alpha_masks)]
            scores['A2_color_dist'] = max(a2_scores)
        else:
            scores['A2_color_dist'] = self._color_distribution_match(
                a2_views[0], alpha_masks[0], reference_rgba)

        # B1. Mesh Integrity (from GLB)
        scores['B1_mesh_integrity'] = self._mesh_integrity(glb_mesh)

        # B2. Surface Quality (from normal maps)
        scores['B2_surface_quality'] = self._surface_quality(
            normals, alpha_masks)

        # C1. Texture Coherence (from base_color)
        scores['C1_tex_coherence'] = self._texture_coherence(
            base_colors, alpha_masks)

        # C2. Color Vitality (from base_color)
        scores['C2_color_vitality'] = self._color_vitality(
            base_colors, alpha_masks)

        # C3. Detail Richness (texture + geometry + visual entropy)
        # Use only lower-pitch views (first half) — high elevation views
        # show less texture detail by nature which unfairly penalizes C3.
        # Views are ordered by pitch: [0.15, 0.30, 0.50, 0.65], 8 each.
        n_c3 = max(8, nviews // 2)
        scores['C3_detail_richness'] = self._detail_richness(
            base_colors[:n_c3], alpha_masks[:n_c3], texture_map, tex_mask,
            glb_mesh=glb_mesh)

        # D1. Material Plausibility (from metallic/roughness)
        scores['D1_material'] = self._material_plausibility(
            metallics, roughnesses, alpha_masks)

        # E1. Multi-View Consistency (from base_color)
        scores['E1_multiview'] = self._multiview_consistency(
            base_colors, alpha_masks)

        # Weighted overall
        scores['overall'] = sum(
            scores.get(d, 0) * w / self.total_weight
            for d, w in self.weights.items()
        )

        scores['scoring_version'] = 'v4'
        return scores

    # ─── A1. Silhouette Match ─────────────────────────────────────────────

    def _silhouette_match(self, rendered_rgb, rendered_alpha, reference_rgba,
                          all_alphas=None):
        """Compare rendered silhouette to reference alpha.

        Uses scale-invariant IoU: both masks are cropped to their bounding
        boxes and resized to a canonical size before comparison. This removes
        framing/scale differences between input image and rendered views.

        Uses best-matching view among all rendered views, since the input
        image may not be front-facing (yaw=0).
        """
        if reference_rgba is None or reference_rgba.shape[2] < 4:
            return 50.0

        ref_alpha = reference_rgba[:, :, 3]

        def _crop_to_bbox(mask_uint8, pad_frac=0.05):
            """Crop mask to its bounding box with padding, return resized to 256x256."""
            binary = (mask_uint8 > 128).astype(np.uint8)
            coords = np.argwhere(binary > 0)
            if len(coords) < 10:
                return np.zeros((256, 256), dtype=np.float32)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            h, w = y1 - y0 + 1, x1 - x0 + 1
            # Make square
            side = max(h, w)
            pad = int(side * pad_frac)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half = side // 2 + pad
            # Clamp to image bounds
            H, W = mask_uint8.shape[:2]
            sy = max(0, cy - half)
            ey = min(H, cy + half)
            sx = max(0, cx - half)
            ex = min(W, cx + half)
            crop = mask_uint8[sy:ey, sx:ex]
            if crop.size == 0:
                return np.zeros((256, 256), dtype=np.float32)
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            return (resized > 128).astype(np.float32)

        ref_norm = _crop_to_bbox(ref_alpha)

        def _dice(rend_alpha):
            """Dice coefficient (F1 score) — more forgiving than IoU for shape comparison.
            Dice = 2*|A∩B| / (|A|+|B|), treats FP/FN equally, less sensitive to
            small protrusions/indentations that don't affect visual similarity."""
            rend_norm = _crop_to_bbox(rend_alpha)
            intersection = (rend_norm * ref_norm).sum()
            total = rend_norm.sum() + ref_norm.sum()
            return 2 * intersection / max(total, 1)

        # Find best-matching view among all rendered views
        if all_alphas is not None and len(all_alphas) > 1:
            best_dice = max(_dice(a) for a in all_alphas)
        else:
            best_dice = _dice(rendered_alpha)

        return float(best_dice * 100)

    # ─── A2. Color Distribution Match ─────────────────────────────────────

    def _color_distribution_match(self, rendered_rgb, rendered_alpha,
                                  reference_rgba):
        """Compare color distributions using HSV histograms + LAB ΔE.

        Three components:
          1. HSV histogram correlation (hue + saturation)
          2. Mean color proximity (saturation-weighted hue + saturation)
          3. LAB ΔE mean color distance (perceptually uniform, robust to lighting)

        Key improvement: hue weight is modulated by saturation level. For
        low-saturation objects (metals, grays), hue is unreliable noise — we
        rely more on LAB and saturation. For chromatic objects, hue matters more.
        """
        if reference_rgba is None:
            return 50.0

        h, w = rendered_rgb.shape[:2]
        ref_resized = reference_rgba
        if reference_rgba.shape[:2] != (h, w):
            ref_resized = cv2.resize(reference_rgba, (w, h))

        rend_mask = rendered_alpha > 128
        ref_mask = ref_resized[:, :, 3] > 128 if ref_resized.shape[2] == 4 else np.ones((h, w), bool)

        if rend_mask.sum() < 100 or ref_mask.sum() < 100:
            return 50.0

        ref_rgb = ref_resized[:, :, :3]

        # --- Component 1: HSV histogram correlation ---
        rend_hsv = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2HSV)
        ref_hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
        rend_mask_u8 = rend_mask.astype(np.uint8)
        ref_mask_u8 = ref_mask.astype(np.uint8)

        hist_scores = []
        for c_idx, n_bins, val_range in [(0, 36, [0, 180]), (1, 32, [0, 256])]:
            rend_hist = cv2.calcHist([rend_hsv], [c_idx], rend_mask_u8,
                                    [n_bins], val_range)
            ref_hist = cv2.calcHist([ref_hsv], [c_idx], ref_mask_u8,
                                   [n_bins], val_range)
            cv2.normalize(rend_hist, rend_hist, alpha=1.0, beta=0.0,
                          norm_type=cv2.NORM_L1)
            cv2.normalize(ref_hist, ref_hist, alpha=1.0, beta=0.0,
                          norm_type=cv2.NORM_L1)
            corr = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_CORREL)
            hist_scores.append((corr + 1.0) / 2.0)
        hist_score = sum(hist_scores) / len(hist_scores)

        # --- Component 2: Mean color proximity (saturation-adaptive) ---
        rend_sat_mean = float(rend_hsv[:, :, 1][rend_mask].mean())
        ref_sat_mean = float(ref_hsv[:, :, 1][ref_mask].mean())
        avg_sat = (rend_sat_mean + ref_sat_mean) / 2.0

        # Hue reliability: high saturation → hue is reliable, low → unreliable
        # sat_weight: 0-1, maps saturation 0-128 to weight 0-1
        hue_reliability = min(1.0, avg_sat / 128.0)

        rend_hue = float(rend_hsv[:, :, 0][rend_mask].mean())
        ref_hue = float(ref_hsv[:, :, 0][ref_mask].mean())
        hue_diff = abs(rend_hue - ref_hue)
        hue_diff = min(hue_diff, 180.0 - hue_diff)
        hue_score = max(0.0, 1.0 - hue_diff / 60.0)  # 60° tolerance (was 45°)

        sat_diff = abs(rend_sat_mean - ref_sat_mean)
        sat_score = max(0.0, 1.0 - sat_diff / 100.0)  # 100 tolerance (was 80)

        # Adaptive weighting: low saturation → rely more on sat_score
        mean_score = hue_reliability * hue_score + (1 - hue_reliability * 0.4) * sat_score
        mean_score = min(1.0, mean_score / (1 + (1 - hue_reliability * 0.4)))

        # --- Component 3: LAB ΔE mean color distance ---
        rend_lab = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Mean LAB values within masks (only a* and b* for chrominance)
        rend_a = float(rend_lab[:, :, 1][rend_mask].mean())
        rend_b = float(rend_lab[:, :, 2][rend_mask].mean())
        ref_a = float(ref_lab[:, :, 1][ref_mask].mean())
        ref_b = float(ref_lab[:, :, 2][ref_mask].mean())

        # ΔE_ab (CIE76) using only chrominance (a*, b*), skip L*
        delta_e = np.sqrt((rend_a - ref_a)**2 + (rend_b - ref_b)**2)
        # Relaxed: ΔE < 10 is excellent, < 25 acceptable, > 50 poor
        # 3D renders naturally differ from photos due to lighting/material model
        lab_score = max(0.0, min(1.0, 1.0 - delta_e / 50.0))

        # --- Final blend ---
        # 50% histogram + 35% mean proximity (sat-adaptive) + 15% LAB ΔE
        score = min(1.0, 0.5 * hist_score + 0.35 * mean_score + 0.15 * lab_score)
        return float(score * 100)

    # ─── B1. Mesh Integrity ───────────────────────────────────────────────

    def _mesh_integrity(self, glb_mesh):
        """Evaluate mesh structural quality using trimesh diagnostics."""
        if glb_mesh is None:
            return 50.0

        import trimesh

        # Extract main mesh from scene if needed
        mesh = glb_mesh
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            mesh = meshes[0] if meshes else None
        if mesh is None:
            return 0.0

        score = 100.0
        _penalties = {}

        # 1. Watertight check (-2 if not)
        # FDG remesh+simplify never produces perfectly watertight meshes;
        # irrelevant for WebGL/Three.js visualization (only 3D printing)
        try:
            if not mesh.is_watertight:
                score -= 2
                _penalties['watertight'] = 2
        except Exception:
            score -= 2
            _penalties['watertight'] = 2

        # 2. Degenerate face ratio (-25 max)
        try:
            face_areas = mesh.area_faces
            zero_area = (face_areas < 1e-10).sum()
            degen_ratio = zero_area / max(len(face_areas), 1)
            if degen_ratio > 0.001:
                pen = min(25, degen_ratio * 5000)
                score -= pen
                _penalties['degenerate'] = pen
        except Exception:
            pass

        # 3. Connected components (-3 max, log scale)
        # UV parameterization duplicates vertices at seam boundaries,
        # creating 20K-40K face-adjacency components from ~50 real components.
        # This is normal for UV-mapped GLB. Only penalize extreme cases.
        try:
            nfaces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
            if nfaces <= 500000:
                from scipy import sparse
                if hasattr(mesh, 'face_adjacency'):
                    adj = mesh.face_adjacency
                    graph = sparse.coo_matrix(
                        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
                        shape=(nfaces, nfaces)).tocsr()
                    n_components = sparse.csgraph.connected_components(graph, directed=False)[0]
                    if n_components > 50000:
                        # Penalty only for extreme fragmentation
                        log_comp = np.log10(n_components)
                        pen = min(3, max(0, (log_comp - 4.7) * 5.0))
                        score -= pen
                        _penalties['components'] = (n_components, round(pen, 1))
                    else:
                        _penalties['components'] = (n_components, 0)
        except Exception:
            pass

        # 4. Face aspect ratio (-15 max) — sample for large meshes
        try:
            edges = mesh.edges_unique_length
            if len(edges) > 0:
                # Sample edges if too many to avoid memory pressure
                if len(edges) > 1000000:
                    rng = np.random.RandomState(42)
                    sample_idx = rng.choice(len(edges), 1000000, replace=False)
                    edges = edges[sample_idx]
                p95 = np.percentile(edges, 95)
                p5 = np.percentile(edges, 5)
                if p5 > 1e-8:
                    aspect = p95 / p5
                    if aspect > 100:
                        pen = min(15, (aspect - 100) * 0.1)
                        score -= pen
                        _penalties['aspect'] = pen
        except Exception:
            pass

        print(f"    [B1 diag] penalties={_penalties} final={score:.1f}",
              flush=True)

        # 5. Normal consistency (-10 max) — sample for large meshes
        try:
            if hasattr(mesh, 'face_adjacency') and len(mesh.face_adjacency) > 0:
                adj_angles = mesh.face_adjacency_angles
                # Sample if huge
                if len(adj_angles) > 2000000:
                    rng = np.random.RandomState(42)
                    sample_idx = rng.choice(len(adj_angles), 2000000, replace=False)
                    adj_angles = adj_angles[sample_idx]
                flipped_ratio = (adj_angles > np.pi * 0.9).sum() / max(len(adj_angles), 1)
                if flipped_ratio > 0.01:
                    score -= min(10, flipped_ratio * 500)
        except Exception:
            pass

        print(f"    [B1 diag] penalties={_penalties} final={max(0,score):.0f}", flush=True)
        return max(0.0, float(score))

    # ─── B2. Surface Quality ──────────────────────────────────────────────

    def _surface_quality(self, normal_views, alpha_masks):
        """
        Evaluate geometric surface quality from rendered normal maps.
        Normal maps capture geometry-only roughness, independent of texture.
        """
        scores = []
        nviews = len(normal_views)

        for i in range(nviews):
            normal_img = normal_views[i]
            mask = alpha_masks[i] > 128

            if mask.sum() < 100:
                scores.append(50.0)
                continue

            # Erode mask to exclude silhouette edges
            kernel = np.ones((10, 10), np.uint8)
            interior = cv2.erode(mask.astype(np.uint8), kernel) > 0

            if interior.sum() < 100:
                scores.append(50.0)
                continue

            # Compute normal map Laplacian (geometry roughness)
            # Pre-blur with σ=1.5 to remove flat-face-normal triangle edge
            # artifacts that don't reflect actual visual roughness
            normal_float = normal_img.astype(np.float64) / 255.0
            normal_float = cv2.GaussianBlur(normal_float, (0, 0), 1.5)

            lap_energy = 0.0
            for c in range(3):
                lap = cv2.Laplacian(normal_float[:, :, c], cv2.CV_64F)
                lap_energy += np.abs(lap[interior]).mean()
            lap_energy /= 3.0

            # Score mapping:
            # 0.0-0.02: excellent geometry (100)
            # 0.02-0.05: good (60-100)
            # 0.05-0.1: acceptable (30-60)
            # >0.1: rough/noisy (0-30)
            s = 100.0
            if lap_energy > 0.02:
                s -= min(40, (lap_energy - 0.02) * 1333)
            if lap_energy > 0.05:
                s -= min(30, (lap_energy - 0.05) * 600)

            scores.append(max(0.0, s))

        return float(np.mean(scores))

    # ─── C1. Texture Coherence ────────────────────────────────────────────

    def _texture_coherence(self, base_color_views, alpha_masks):
        """
        Evaluate texture coherence — detects patchwork, UV seams,
        and harsh color discontinuities.
        """
        scores = []
        nviews = len(base_color_views)
        # Diagnostic accumulators
        _crack_penalties = []
        _grad_penalties = []
        _patch_penalties = []

        for i in range(nviews):
            rgb = base_color_views[i]
            mask = alpha_masks[i] > 128

            # Pre-blur with σ=1.0 to remove 1-pixel triangle-edge
            # artifacts from volumetric renderer (same principle as B2 fix)
            rgb_smooth = cv2.GaussianBlur(rgb, (0, 0), 1.0)
            gray = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)

            # Interior mask (away from silhouette edge)
            kernel = np.ones((10, 10), np.uint8)
            interior = cv2.erode(mask.astype(np.uint8), kernel) > 0

            if interior.sum() < 200:
                scores.append(50.0)
                _crack_penalties.append(0)
                _grad_penalties.append(0)
                _patch_penalties.append(0)
                continue

            s = 100.0

            # 1. Morphological crack detection (thin dark gaps)
            gray_int = gray.copy()
            gray_int[~interior] = 128
            closed = cv2.morphologyEx(gray_int, cv2.MORPH_CLOSE,
                                      np.ones((3, 3), np.uint8))
            crack_diff = closed.astype(np.float32) - gray_int.astype(np.float32)
            crack_ratio = ((crack_diff > 20) & interior).sum() / max(interior.sum(), 1)
            crack_pen = 0.0
            if crack_ratio > 0.003:
                crack_pen = min(35, (crack_ratio - 0.003) * 600)
                s -= crack_pen
            _crack_penalties.append(crack_pen)

            # 2. Gradient harshness in COLOR space (not just grayscale)
            rgb_f = rgb_smooth.astype(np.float32)
            grad_mag = np.zeros(gray.shape, dtype=np.float32)
            for c in range(3):
                gx = cv2.Sobel(rgb_f[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(rgb_f[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
                grad_mag += np.sqrt(gx**2 + gy**2)
            grad_mag /= 3.0

            # Adaptive threshold based on image contrast
            # V4 evaluates raw volumetric renders (not GLB), so most gradients
            # come from legitimate texture detail (painted edges, color
            # transitions), not UV seam artifacts. Penalize only extreme cases
            # (>35% of pixels with gradients above p95*0.85).
            p95 = np.percentile(grad_mag[interior], 95)
            threshold = max(40.0, p95 * 0.85)
            harsh = (grad_mag[interior] > threshold).sum() / max(interior.sum(), 1)
            grad_pen = 0.0
            if harsh > 0.35:
                grad_pen = min(15, (harsh - 0.35) * 100)
                s -= grad_pen
            _grad_penalties.append(grad_pen)

            # 3. Local color variance (patchwork detection)
            local_var = np.zeros(gray.shape, dtype=np.float32)
            for c in range(3):
                blurred = cv2.GaussianBlur(rgb_f[:, :, c], (15, 15), 3.0)
                local_var += (rgb_f[:, :, c] - blurred) ** 2
            local_var = np.sqrt(local_var / 3.0)
            high_var_ratio = (local_var[interior] > 30).sum() / max(interior.sum(), 1)
            patch_pen = 0.0
            if high_var_ratio > 0.15:
                patch_pen = min(25, (high_var_ratio - 0.15) * 120)
                s -= patch_pen
            _patch_penalties.append(patch_pen)

            scores.append(max(0.0, s))

        # Diagnostic output
        print(f"    [C1 diag] crack_pen={np.mean(_crack_penalties):.1f} "
              f"grad_pen={np.mean(_grad_penalties):.1f} "
              f"patch_pen={np.mean(_patch_penalties):.1f} "
              f"final={np.mean(scores):.1f}", flush=True)

        return float(np.mean(scores))

    # ─── C2. Color Vitality ───────────────────────────────────────────────

    def _color_vitality(self, base_color_views, alpha_masks):
        """
        Evaluate color health — replaces non-discriminative Vibrancy + Darkness.
        Detects grey patches, washed-out areas, insufficient dynamic range,
        and dark patch artifacts.
        """
        scores = []
        nviews = len(base_color_views)

        for i in range(nviews):
            rgb = base_color_views[i]
            mask = alpha_masks[i] > 128

            if mask.sum() < 100:
                scores.append(50.0)
                continue

            s = 100.0

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[mask].astype(np.float32)
            masked_rgb = rgb[mask].astype(np.float32)

            # 1. Grey patch detection (-30 max)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1][mask].astype(np.float32)
            val = hsv[:, :, 2][mask].astype(np.float32)
            # Grey = low sat AND mid-luminance (not intentionally dark/bright)
            grey_pixels = (sat < 15) & (val > 40) & (val < 200)
            grey_ratio = grey_pixels.sum() / max(mask.sum(), 1)
            if grey_ratio > 0.15:
                s -= min(30, (grey_ratio - 0.15) * 100)

            # 2. Dynamic range (-25 max)
            p5, p95 = np.percentile(gray, [5, 95])
            dyn_range = p95 - p5
            if dyn_range < 40:
                s -= min(25, (40 - dyn_range) * 0.8)

            # 3. Dark patch penalty (-25 max)
            dark_ratio = (gray < 10).sum() / max(gray.size, 1)
            if dark_ratio > 0.02:
                s -= min(25, (dark_ratio - 0.02) * 500)

            # 4. Color channel variance (-20 max)
            r_mean = masked_rgb[:, 0].mean()
            g_mean = masked_rgb[:, 1].mean()
            b_mean = masked_rgb[:, 2].mean()
            channel_std = np.std([r_mean, g_mean, b_mean])
            if channel_std < 3:
                s -= min(15, (3 - channel_std) * 5)

            scores.append(max(0.0, s))

        return float(np.mean(scores))

    # ─── C3. Detail Richness ──────────────────────────────────────────────

    def _detail_richness(self, base_color_views, alpha_masks,
                         texture_map=None, tex_mask=None, glb_mesh=None):
        """
        Measure total visual detail richness (texture + geometry).

        Components:
        - UV texture-map Laplacian (40%): texture detail in UV space
        - Mesh dihedral angle std (40%): geometric surface complexity
        - View-based color entropy (20%): tonal diversity

        The texture + geometry split ensures resolution-agnostic scoring:
        - High-poly mesh with smooth texture → high geo, low tex → balanced
        - Low-poly mesh with detailed texture → low geo, high tex → balanced

        Calibrated from 24 saved GLB files (2026-02-23):
          DihedStd range: 9.62 (simple) → 18.41 (complex), threshold 8-16
          TexLap range: 5.9 (v3) → 11.2 (1536 crown)
        """
        # --- Component 1: Texture-map Laplacian energy (40%) ---
        tex_detail = 60.0  # default if no texture_map
        tex_lap_energy = -1.0
        if texture_map is not None and tex_mask is not None:
            mask_bool = tex_mask.astype(bool) if tex_mask.dtype != bool else tex_mask
            if mask_bool.sum() > 1000:
                tex_gray = cv2.cvtColor(texture_map, cv2.COLOR_RGB2GRAY)
                tex_lap = cv2.Laplacian(tex_gray.astype(np.float64), cv2.CV_64F)
                tex_lap_energy = float(np.abs(tex_lap[mask_bool]).mean())
                # Score mapping for UV texture maps (2048px typically):
                # Calibrated from 9 GLB configs:
                #   v3: 5.88, v4_baseline: 5.96-8.00, 1536: 8.85-11.24
                if tex_lap_energy < 3:
                    tex_detail = 30.0
                elif tex_lap_energy < 12:
                    tex_detail = 30 + (tex_lap_energy - 3) * 7.78  # 30→100
                else:
                    tex_detail = 100.0

        # --- Component 2: Mesh geometry detail (40%) ---
        # Dihedral angle std: measures surface complexity independent of texture.
        # Higher std = more varied surface angles = more geometric detail.
        # Uses GLB mesh (post-decimation), so is independent of raw mesh resolution.
        geo_detail = 60.0  # default if no mesh
        dihed_std = -1.0
        if glb_mesh is not None:
            try:
                angles_deg = np.degrees(glb_mesh.face_adjacency_angles)
                dihed_std = float(np.std(angles_deg))
                # Score mapping calibrated from 24 saved GLBs (500K-800K faces):
                #   cd3c309f: 9.62, T.png: 9.97, bestn_4: 10.7-12.7
                #   smooth_off: 15.94, quality_upgrade: 18.41
                #   Range 8-16 → 30-100
                if dihed_std < 8:
                    geo_detail = 30.0
                elif dihed_std < 16:
                    geo_detail = 30 + (dihed_std - 8) * 8.75  # 30→100
                else:
                    geo_detail = 100.0
            except Exception:
                pass

        # --- Component 3: View-based color entropy (20%) ---
        # Shannon entropy of grayscale histogram — measures tonal diversity
        view_entropies = []
        for i in range(len(base_color_views)):
            bc = base_color_views[i]
            mask = alpha_masks[i] > 128
            if mask.sum() < 200:
                continue
            gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
            vals = gray[mask]
            hist, _ = np.histogram(vals, bins=64, range=(0, 255))
            hist = hist.astype(np.float64)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            view_entropies.append(entropy)

        if view_entropies:
            median_entropy = float(np.median(view_entropies))
            # Score mapping: 64-bin entropy range is [0, 6.0]
            # Typical 3D renders: 3.5-5.5
            if median_entropy < 2.5:
                entropy_score = 30.0
            elif median_entropy < 4.0:
                entropy_score = 30 + (median_entropy - 2.5) * 40.0  # 30→90
            elif median_entropy < 5.5:
                entropy_score = 90 + (median_entropy - 4.0) * 6.67  # 90→100
            else:
                entropy_score = 100.0
        else:
            entropy_score = 50.0

        # Final composite: texture(40%) + geometry(40%) + entropy(20%)
        result = 0.40 * tex_detail + 0.40 * geo_detail + 0.20 * entropy_score
        print(f"    [C3 summary] tex_lap={tex_lap_energy:.1f}->{tex_detail:.1f} "
              f"dihed_std={dihed_std:.1f}->{geo_detail:.1f} "
              f"entropy={median_entropy if view_entropies else -1:.2f}->{entropy_score:.1f} "
              f"final={result:.1f}",
              flush=True)
        return result

    # ─── D1. Material Plausibility ────────────────────────────────────────

    def _material_plausibility(self, metallic_views, roughness_views,
                               alpha_masks):
        """
        Evaluate PBR material plausibility.
        Most real-world objects are non-metallic with moderate roughness.
        """
        scores = []
        nviews = min(4, len(metallic_views))  # Only need a few views

        for i in range(nviews):
            mask = alpha_masks[i] > 128
            if mask.sum() < 100:
                scores.append(50.0)
                continue

            met = metallic_views[i].astype(np.float32)
            if met.ndim == 3:
                met = met[:, :, 0]
            met = met[mask] / 255.0

            rough = roughness_views[i].astype(np.float32)
            if rough.ndim == 3:
                rough = rough[:, :, 0]
            rough = rough[mask] / 255.0

            s = 100.0

            # 1. High metallic penalty (-30 max)
            high_met_ratio = (met > 0.3).sum() / max(met.size, 1)
            if high_met_ratio > 0.05:
                s -= min(30, (high_met_ratio - 0.05) * 200)

            # 2. Extreme roughness penalty (-20 max)
            mirror_ratio = (rough < 0.1).sum() / max(rough.size, 1)
            if mirror_ratio > 0.1:
                s -= min(20, (mirror_ratio - 0.1) * 100)

            # 3. Roughness uniformity bonus (+5)
            rough_std = rough.std()
            if rough_std < 0.15:
                s = min(100, s + 5)

            # 4. Metallic-roughness correlation check (-10)
            if high_met_ratio > 0.01:
                met_areas = met > 0.3
                if met_areas.sum() > 10:
                    met_rough = rough[met_areas].mean()
                    if met_rough > 0.7:
                        s -= 10  # metallic + rough = implausible

            scores.append(max(0.0, s))

        return float(np.mean(scores))

    # ─── E1. Multi-View Consistency ───────────────────────────────────────

    def _multiview_consistency(self, base_color_views, alpha_masks):
        """
        Evaluate consistency of quality across different viewpoints.
        High variance in per-view metrics = view-dependent artifacts.
        """
        if len(base_color_views) < 2:
            return 50.0

        brightness_means = []
        saturation_means = []
        detail_energies = []
        coverages = []

        nviews = len(base_color_views)
        for i in range(nviews):
            mask = alpha_masks[i] > 128
            coverage = mask.sum() / max(mask.size, 1)
            coverages.append(coverage)

            if mask.sum() < 100:
                continue

            gray = cv2.cvtColor(base_color_views[i], cv2.COLOR_RGB2GRAY)
            masked_gray = gray[mask].astype(np.float32)
            brightness_means.append(masked_gray.mean())

            hsv = cv2.cvtColor(base_color_views[i], cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1][mask].astype(np.float32)
            saturation_means.append(sat.mean())

            lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
            detail_energies.append(np.abs(lap[mask]).mean())

        if len(brightness_means) < 2:
            return 50.0

        score = 100.0

        # 1. Brightness consistency (-30 max)
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
        if len(coverages) >= 2:
            cov_cv = np.std(coverages) / max(np.mean(coverages), 1e-6)
            if cov_cv > 0.3:
                score -= min(20, (cov_cv - 0.3) * 60)

        return max(0.0, float(score))


# ═══════════════════════════════════════════════════════════════════════════
# Rendering & Pipeline Functions
# ═══════════════════════════════════════════════════════════════════════════

def load_pipeline():
    """Load the TRELLIS.2 pipeline."""
    print("Loading TRELLIS.2 pipeline...", flush=True)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipe.cuda()
    print("Pipeline loaded.", flush=True)
    _log_memory("pipeline-loaded")
    return pipe


def get_envmap():
    """Create a simple white envmap for rendering."""
    from trellis2.renderers import EnvMap
    hdri = torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.8
    return {'default': EnvMap(hdri)}


def _get_eval_cameras(nviews=8):
    """Return (yaws, pitches, rs, fovs) for canonical evaluation camera setup."""
    pitches = [0.15, 0.30, 0.50, 0.65]
    all_yaw, all_pitch, all_r, all_fov = [], [], [], []
    for p in pitches:
        yaws = np.linspace(0, 2 * np.pi, nviews, endpoint=False).tolist()
        all_yaw.extend(yaws)
        all_pitch.extend([p] * nviews)
        all_r.extend([2.0] * nviews)
        all_fov.extend([40.0] * nviews)
    return all_yaw, all_pitch, all_r, all_fov


def render_evaluation_views(mesh, envmap, nviews=8, resolution=512):
    """
    Render all channels needed for V4 evaluation.
    Uses canonical camera: r=2, fov=40 (same as render_video),
    NOT r=10, fov=8 (snapshot quasi-ortho).

    Multi-pitch sampling: renders at 4 elevation bands to cover typical
    product photography angles (8-37°).
    Total views: nviews * 4 pitches (default 32 views, ~2s render).
    """
    from trellis2.utils import render_utils

    all_yaw, all_pitch, all_r, all_fov = _get_eval_cameras(nviews)

    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        all_yaw, all_pitch, all_r, all_fov)

    result = render_utils.render_frames(
        mesh, extr, intr,
        options={'resolution': resolution, 'bg_color': (0, 0, 0)},
        envmap=envmap,
        verbose=False,
    )

    return {
        'base_color': result['base_color'],
        'normal': result['normal'],
        'metallic': result['metallic'],
        'roughness': result['roughness'],
        'alpha': result['alpha'],
        'shaded': result.get('shaded', result['base_color']),
    }


def render_glb_silhouettes(glb_mesh, nviews=8, resolution=512):
    """Render silhouette alpha masks from a GLB trimesh using nvdiffrast.

    Uses the same camera setup as render_evaluation_views so that
    A1 silhouette comparison is consistent.

    Returns:
        list of (H, W) uint8 alpha masks (0 or 255)
    """
    import nvdiffrast.torch as dr
    import utils3d

    device = 'cuda'
    glctx = dr.RasterizeCudaContext(device=device)

    vertices = torch.tensor(glb_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(glb_mesh.faces, dtype=torch.int32, device=device)

    all_yaw, all_pitch, all_r, all_fov = _get_eval_cameras(nviews)

    alpha_list = []
    for yaw, pitch, r, fov in zip(all_yaw, all_pitch, all_r, all_fov):
        fov_rad = torch.deg2rad(torch.tensor(float(fov))).to(device)
        yaw_t = torch.tensor(float(yaw)).to(device)
        pitch_t = torch.tensor(float(pitch)).to(device)

        # GLB Y-up convention (postprocess Y/Z swap: x,z,-y)
        orig_x = torch.sin(yaw_t) * torch.cos(pitch_t)
        orig_y = torch.cos(yaw_t) * torch.cos(pitch_t)
        orig_z = torch.sin(pitch_t)
        orig = torch.tensor([orig_x.item(), orig_z.item(), -orig_y.item()],
                            device=device) * r

        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.zeros(3, device=device),
            torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)

        near, far = 0.1, 100.0
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        proj = torch.zeros((4, 4), dtype=torch.float32, device=device)
        proj[0, 0] = 2 * fx
        proj[1, 1] = 2 * fy
        proj[0, 2] = 2 * cx - 1
        proj[1, 2] = -2 * cy + 1
        proj[2, 2] = (far + near) / (far - near)
        proj[2, 3] = 2 * near * far / (near - far)
        proj[3, 2] = 1.0

        verts_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        verts_clip = (verts_homo @ (proj @ extr).T).unsqueeze(0)

        rast, _ = dr.rasterize(glctx, verts_clip, faces, (resolution, resolution))
        mask = (rast[0, :, :, 3] > 0).cpu().numpy().astype(np.uint8) * 255
        alpha_list.append(mask)

    return alpha_list


def generate_and_evaluate(pipeline, image_path, config, evaluator, envmap,
                          output_prefix="test", best_of_n=1, quality_verifier=None,
                          texture_refiner=None, silhouette_corrector=None):
    """Generate a 3D model and evaluate it with V4 scoring."""
    import o_voxel.postprocess
    from trellis2.utils import render_utils

    print(f"\n--- Processing: {os.path.basename(image_path)} ---", flush=True)

    # Load and preprocess image
    img = Image.open(image_path)
    processed = pipeline.preprocess_image(img)

    # Run pipeline (staged Best-of-N or standard)
    staged_bon = config.get('staged_bon', 0)
    t0 = time.time()

    if staged_bon > 1:
        print(f"  Using staged Best-of-N with N={staged_bon}", flush=True)
        outputs, latents = pipeline._run_staged_bon(
            processed,
            n_shapes=staged_bon,
            seed=config.get('seed', 42),
            sparse_structure_sampler_params={
                "steps": config['ss_sampling_steps'],
                "guidance_strength": config['ss_guidance_strength'],
                "guidance_rescale": config['ss_guidance_rescale'],
                "rescale_t": config['ss_rescale_t'],
                "multistep": config.get('multistep', False),
                "schedule": config.get('ss_schedule', config.get('schedule', 'uniform')),
                "schedule_rho": config.get('schedule_rho', 7.0),
                "guidance_schedule": config.get('ss_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('ss_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('ss_guidance_beta_b', 5.0),
                "occupancy_threshold": config.get('ss_occupancy_threshold', 0.0),
            },
            shape_slat_sampler_params={
                "steps": config['shape_slat_sampling_steps'],
                "guidance_strength": config['shape_slat_guidance_strength'],
                "guidance_rescale": config['shape_slat_guidance_rescale'],
                "rescale_t": config['shape_slat_rescale_t'],
                "multistep": config.get('multistep', False),
                "schedule": config.get('shape_schedule', config.get('schedule', 'uniform')),
                "schedule_rho": config.get('schedule_rho', 7.0),
                "guidance_schedule": config.get('shape_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('shape_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('shape_guidance_beta_b', 5.0),
            },
            tex_slat_sampler_params={
                "steps": config['tex_slat_sampling_steps'],
                "guidance_strength": config['tex_slat_guidance_strength'],
                "guidance_rescale": config['tex_slat_guidance_rescale'],
                "rescale_t": config['tex_slat_rescale_t'],
                "cfg_mp_strength": config.get('cfg_mp_strength', 0.0),
                "heun_steps": config.get('heun_steps', 0),
                "multistep": config.get('multistep', False),
                "schedule": config.get('tex_schedule', config.get('schedule', 'uniform')),
                "guidance_schedule": config.get('tex_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('tex_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('tex_guidance_beta_b', 5.0),
                "guidance_interval": tuple(config.get('tex_guidance_interval', (0.0, 1.0))),
                "guidance_anneal_min": config.get('tex_guidance_anneal_min', 0.0),
                "guidance_anneal_start": config.get('tex_guidance_anneal_start', 0.3),
                "sde_alpha": config.get('tex_sde_alpha', config.get('sde_alpha', 0.0)),
                "sde_profile": config.get('tex_sde_profile', config.get('sde_profile', 'zero_ends')),
                # FDG: Frequency-Decoupled Guidance (stage 3 texture only)
                "cfg_mode": config.get('tex_cfg_mode', config.get('cfg_mode', 'standard')),
                "fdg_sigma": config.get('tex_fdg_sigma', config.get('fdg_sigma', 1.0)),
                "fdg_lambda_low": config.get('tex_fdg_lambda_low', config.get('fdg_lambda_low', 0.6)),
                "fdg_lambda_high": config.get('tex_fdg_lambda_high', config.get('fdg_lambda_high', 1.3)),
            },
            pipeline_type={
                "512": "512", "1024": "1024_cascade", "1536": "1536_cascade",
            }.get(config.get('resolution', '1024'), '1024_cascade'),
            return_latent=True,
        )
    else:
        outputs, latents = pipeline.run(
            processed,
            seed=config.get('seed', 42),
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": config['ss_sampling_steps'],
                "guidance_strength": config['ss_guidance_strength'],
                "guidance_rescale": config['ss_guidance_rescale'],
                "rescale_t": config['ss_rescale_t'],
                "multistep": config.get('multistep', False),
                "schedule": config.get('ss_schedule', config.get('schedule', 'uniform')),
                "schedule_rho": config.get('schedule_rho', 7.0),
                "schedule_power": config.get('ss_schedule_power', config.get('schedule_power', 2.0)),
                # Occupancy threshold for sparse structure (0.0 default, higher=tighter silhouette)
                "occupancy_threshold": config.get('ss_occupancy_threshold', 0.0),
                # Guidance schedule for SS stage
                "guidance_schedule": config.get('ss_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('ss_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('ss_guidance_beta_b', 5.0),
                **({'guidance_interval': tuple(config['ss_guidance_interval'])} if 'ss_guidance_interval' in config else {}),
            },
            shape_slat_sampler_params={
                "steps": config['shape_slat_sampling_steps'],
                "guidance_strength": config['shape_slat_guidance_strength'],
                "guidance_rescale": config['shape_slat_guidance_rescale'],
                "rescale_t": config['shape_slat_rescale_t'],
                "multistep": config.get('multistep', False),
                "schedule": config.get('shape_schedule', config.get('schedule', 'uniform')),
                "schedule_rho": config.get('schedule_rho', 7.0),
                "schedule_power": config.get('shape_schedule_power', config.get('schedule_power', 2.0)),
                # Guidance schedule for shape stage
                "guidance_schedule": config.get('shape_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('shape_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('shape_guidance_beta_b', 5.0),
                **({'guidance_interval': tuple(config['shape_guidance_interval'])} if 'shape_guidance_interval' in config else {}),
            },
            tex_slat_sampler_params={
                "steps": config['tex_slat_sampling_steps'],
                "guidance_strength": config['tex_slat_guidance_strength'],
                "guidance_rescale": config['tex_slat_guidance_rescale'],
                "rescale_t": config['tex_slat_rescale_t'],
                "cfg_mp_strength": config.get('cfg_mp_strength', 0.0),
                "heun_steps": config.get('heun_steps', 0),
                "multistep": config.get('multistep', False),
                "schedule": config.get('tex_schedule', config.get('schedule', 'uniform')),
                "schedule_rho": config.get('schedule_rho', 7.0),
                "schedule_power": config.get('tex_schedule_power', config.get('schedule_power', 2.0)),
                # Guidance schedule (only for texture stage — geometry stages use 'binary')
                "guidance_schedule": config.get('tex_guidance_schedule', 'binary'),
                "guidance_beta_a": config.get('tex_guidance_beta_a', 2.0),
                "guidance_beta_b": config.get('tex_guidance_beta_b', 5.0),
                "guidance_interval": tuple(config.get('tex_guidance_interval', (0.0, 1.0))),
                # Guidance anneal: reduce guidance near t=0 to preserve fine detail
                "guidance_anneal_min": config.get('tex_guidance_anneal_min', 0.0),
                "guidance_anneal_start": config.get('tex_guidance_anneal_start', 0.3),
                # Stochastic SDE sampling
                "sde_alpha": config.get('tex_sde_alpha', config.get('sde_alpha', 0.0)),
                "sde_profile": config.get('tex_sde_profile', config.get('sde_profile', 'zero_ends')),
                # FDG: Frequency-Decoupled Guidance (stage 3 texture only)
                "cfg_mode": config.get('tex_cfg_mode', config.get('cfg_mode', 'standard')),
                "fdg_sigma": config.get('tex_fdg_sigma', config.get('fdg_sigma', 1.0)),
                "fdg_lambda_low": config.get('tex_fdg_lambda_low', config.get('fdg_lambda_low', 0.6)),
                "fdg_lambda_high": config.get('tex_fdg_lambda_high', config.get('fdg_lambda_high', 1.3)),
            },
            pipeline_type={
                "512": "512", "1024": "1024_cascade", "1536": "1536_cascade",
            }.get(config.get('resolution', '1024'), '1024_cascade'),
            return_latent=True,
            best_of_n=best_of_n,
            quality_verifier=quality_verifier,
        )
    gen_time = time.time() - t0
    print(f"  Generation: {gen_time:.1f}s", flush=True)
    _log_memory("post-generation")

    mesh = outputs[0]
    mesh.simplify(16777216)

    # Render multi-view with canonical camera (r=2, fov=40, 8 views)
    # Scale render resolution with pipeline resolution to preserve fine detail
    # 512→512, 1024→512, 1536→768
    pipeline_res = int(config.get('resolution', '1024'))
    eval_render_res = 768 if pipeline_res >= 1536 else 512
    t0 = time.time()
    try:
        channels = render_evaluation_views(mesh, envmap, nviews=8, resolution=eval_render_res)
    except Exception as e:
        print(f"  Warning: render failed ({e}), using dummy views", flush=True)
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        channels = {
            'base_color': [dummy], 'normal': [dummy],
            'metallic': [dummy], 'roughness': [dummy],
            'alpha': [dummy], 'shaded': [dummy],
        }
    render_time = time.time() - t0
    nviews = len(channels['base_color'])
    print(f"  Render: {render_time:.1f}s ({nviews} views)", flush=True)
    _log_memory("post-render")

    # Load reference RGBA — use preprocessed image (background removed)
    # so that A1 silhouette IoU compares object masks, not full-frame coverage
    ref_np = np.array(processed.convert('RGBA'))

    # Extract GLB for mesh integrity evaluation (B1) and texture detail (C3)
    t0 = time.time()
    shape_slat, tex_slat, res = latents
    try:
        # Build kwargs, only include params supported by installed o_voxel
        import inspect
        glb_sig = inspect.signature(o_voxel.postprocess.to_glb)
        glb_kwargs = dict(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=config.get('decimation_target', 500000),
            texture_size=config.get('texture_size', 2048),
            remesh=True,
            remesh_band=1,
            remesh_project=0.9,
            verbose=True,
        )
        # Optional params (may not exist in pip-installed version)
        for k, v in [('max_metallic', 0.05), ('min_roughness', 0.4),
                      ('enable_normal_map', False), ('enable_ao', False),
                      ('enable_grey_recovery', True)]:
            if k in glb_sig.parameters:
                glb_kwargs[k] = v
        glb = o_voxel.postprocess.to_glb(**glb_kwargs)
    except Exception as e:
        print(f"  Warning: GLB extraction failed ({e})", flush=True)
        traceback.print_exc()
        glb = None
    glb_time = time.time() - t0
    print(f"  GLB extraction: {glb_time:.1f}s", flush=True)
    _log_memory("post-glb")

    # Silhouette correction (differentiable mesh deformation)
    if glb is not None and silhouette_corrector is not None:
        t0_sil = time.time()
        try:
            silcorr_kwargs = {
                'yaw': 0.0, 'pitch': 0.25, 'r': None, 'fov': 40.0,
                'num_steps': config.get('silcorr_steps', 100),
                'lr': config.get('silcorr_lr', 1e-3),
                'w_silhouette': config.get('silcorr_w_sil', 1.0),
                'w_laplacian': config.get('silcorr_w_lap', 10.0),
                'w_normal': config.get('silcorr_w_norm', 3.0),
                'max_displacement': config.get('silcorr_max_disp', 0.06),
                'use_dice_loss': config.get('silcorr_dice', True),
                'multi_resolution': config.get('silcorr_multires', True),
                'verbose': True,
            }
            glb = silhouette_corrector.correct(
                glb, processed, **silcorr_kwargs,
            )
            sil_time = time.time() - t0_sil
            print(f"  Silhouette correction: {sil_time:.1f}s", flush=True)
            # Re-render alpha masks from corrected GLB so A1 reflects the fix
            try:
                corrected_alphas = render_glb_silhouettes(
                    glb, nviews=8, resolution=eval_render_res)
                channels['alpha'] = corrected_alphas
                print(f"  Re-rendered {len(corrected_alphas)} alpha views from corrected GLB",
                      flush=True)
            except Exception as e2:
                print(f"  Warning: GLB alpha re-render failed ({e2})", flush=True)
        except Exception as e:
            print(f"  Warning: Silhouette correction failed ({e})", flush=True)
            traceback.print_exc()

    # Texture refinement (render-and-compare optimization)
    if glb is not None and texture_refiner is not None:
        t0_refine = time.time()
        try:
            refine_iters = config.get('texture_refine_iters', 50)
            glb = texture_refiner.refine(
                glb, processed,
                num_iters=refine_iters,
                lr=0.005,
                lpips_weight=0.1,
            )
            refine_time = time.time() - t0_refine
            print(f"  Texture refinement: {refine_time:.1f}s ({refine_iters} iters)", flush=True)
        except Exception as e:
            print(f"  Warning: Texture refinement failed ({e})", flush=True)
            traceback.print_exc()

    # Save GLB
    glb_path = None
    if glb is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        glb_path = os.path.join(
            OUTPUT_DIR,
            f'{output_prefix}_{os.path.basename(image_path)}.glb')
        glb.export(glb_path)
        print(f"  Saved: {glb_path}", flush=True)

    # Extract texture map from GLB for C3 detail richness blend
    texture_map = None
    tex_mask = None
    if glb is not None:
        try:
            mat = glb.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                bc_img = np.array(mat.baseColorTexture)
                if bc_img.shape[2] == 4:
                    tex_mask = bc_img[:, :, 3] > 10
                    texture_map = bc_img[:, :, :3]
                else:
                    texture_map = bc_img
                    tex_mask = np.any(texture_map > 5, axis=-1)
        except Exception as e:
            print(f"  Warning: texture map extraction failed: {e}", flush=True)

    # FREE CUDA memory before evaluation — pipeline stays loaded but
    # mesh, outputs, latents are no longer needed
    del mesh, outputs, latents, shape_slat, tex_slat
    gc.collect()
    torch.cuda.empty_cache()
    _log_memory("pre-evaluate")

    # Evaluate with V4
    scores = evaluator.evaluate(
        channels=channels,
        reference_rgba=ref_np,
        glb_mesh=glb,
        texture_map=texture_map,
        tex_mask=tex_mask,
    )

    # Save rendered views for visual inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, view in enumerate(channels['base_color'][:4]):
        view_path = os.path.join(
            OUTPUT_DIR,
            f'{output_prefix}_{os.path.basename(image_path)}_v4_view{i}.png')
        Image.fromarray(view).save(view_path)

    result = {
        'image': os.path.basename(image_path),
        'scores': scores,
        'timings': {
            'generation_s': gen_time,
            'render_s': render_time,
            'glb_s': glb_time,
        },
        'glb_path': glb_path,
    }

    # Cleanup remaining objects
    del channels, glb, texture_map, tex_mask, ref_np
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_evaluation(config, label="test", test_images=None, best_of_n=1,
                   texture_refine=False, silhouette_correct=False):
    """Run full V4 evaluation pipeline."""
    if test_images is None:
        test_images = TEST_IMAGES_FULL

    # Filter to existing images
    existing = [p for p in test_images if os.path.exists(p)]
    if not existing:
        print(f"ERROR: No test images found. Tried: {test_images}")
        return None

    print(f"\n{'='*60}")
    print(f"V4 Evaluation: {label}")
    print(f"Config: resolution={config.get('resolution', '1024')}, "
          f"cfg_mp={config.get('cfg_mp_strength', 0.0)}, "
          f"best_of_n={best_of_n}, texture_refine={texture_refine}, "
          f"silhouette_correct={silhouette_correct}")
    print(f"Test images: {len(existing)}")
    print(f"{'='*60}\n")

    pipeline = load_pipeline()
    envmap = get_envmap()
    evaluator = QualityEvaluatorV4()

    # Initialize quality verifier for Best-of-N
    quality_verifier = None
    if best_of_n > 1:
        from trellis2.utils.quality_verifier import QualityVerifier
        quality_verifier = QualityVerifier(device='cuda')
        print(f"Best-of-N enabled: N={best_of_n}, QualityVerifier loaded")

    # Initialize texture refiner
    tex_refiner = None
    if texture_refine:
        from trellis2.postprocessing.texture_refiner import TextureRefiner
        tex_refiner = TextureRefiner(device='cuda')
        print(f"Texture refinement enabled: {config.get('texture_refine_iters', 50)} iters")

    # Initialize silhouette corrector
    sil_corrector = None
    if silhouette_correct:
        from trellis2.postprocessing.silhouette_corrector import SilhouetteCorrector
        sil_corrector = SilhouetteCorrector(device='cuda')
        print(f"Silhouette correction enabled: 80 steps")

    results = []
    for img_path in existing:
        try:
            r = generate_and_evaluate(
                pipeline, img_path, config, evaluator, envmap,
                output_prefix=label,
                best_of_n=best_of_n,
                quality_verifier=quality_verifier,
                texture_refiner=tex_refiner,
                silhouette_corrector=sil_corrector,
            )
            results.append(r)
            # Print per-image scores
            s = r['scores']
            print(f"\n  Scores for {r['image']}:")
            for dim, w in evaluator.weights.items():
                print(f"    {dim:20s}: {s.get(dim, 0):6.1f}/100 (weight {w})")
            print(f"    {'OVERALL':20s}: {s['overall']:6.1f}/100")
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            traceback.print_exc()
            results.append({
                'image': os.path.basename(img_path),
                'error': str(e),
            })

    # Compute averages
    valid = [r for r in results if 'scores' in r]
    if not valid:
        print("No valid results!")
        return None

    avg_scores = {}
    for dim in evaluator.weights:
        vals = [r['scores'][dim] for r in valid]
        avg_scores[dim] = float(np.mean(vals))
        avg_scores[f'{dim}_std'] = float(np.std(vals))
    avg_scores['overall'] = float(np.mean([r['scores']['overall'] for r in valid]))
    avg_scores['overall_std'] = float(np.std([r['scores']['overall'] for r in valid]))

    # Statistical confidence
    n = len(valid)
    se = avg_scores['overall_std'] / np.sqrt(n) if n > 1 else 0
    ci95 = 1.96 * se

    print(f"\n{'='*60}")
    print(f"V4 AVERAGE SCORES ({n} images):")
    for dim, w in evaluator.weights.items():
        std = avg_scores.get(f'{dim}_std', 0)
        print(f"  {dim:20s}: {avg_scores[dim]:6.1f} +/- {std:4.1f}  (weight {w})")
    print(f"  {'OVERALL':20s}: {avg_scores['overall']:6.1f} +/- {avg_scores['overall_std']:.1f}")
    print(f"  95% CI: [{avg_scores['overall'] - ci95:.1f}, {avg_scores['overall'] + ci95:.1f}]")
    print(f"{'='*60}\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f'eval_v4_{label}_{timestamp}.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({
            'label': label,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'scoring_version': 'v4',
            'num_images': n,
            'confidence_interval_95': ci95,
            'average_scores': avg_scores,
            'per_image': results,
        }, f, indent=2, default=str)
    print(f"Results saved: {result_file}")

    del pipeline
    torch.cuda.empty_cache()
    return avg_scores


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 V4 Quality Evaluator')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (3 images)')
    parser.add_argument('--full', action='store_true',
                        help='Full evaluation (10 images)')
    parser.add_argument('--cfg_mp', type=float, default=None,
                        help='CFG-MP strength to test')
    parser.add_argument('--sweep', nargs='+',
                        help='Sweep parameter: name val1 val2 ...')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline only')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to adjustment JSON config')
    parser.add_argument('--best-of-n', type=int, default=1,
                        help='Best-of-N: generate N candidates, pick best (default: 1)')
    parser.add_argument('--texture-refine', action='store_true',
                        help='Enable render-and-compare texture refinement on GLB')
    parser.add_argument('--silhouette-correct', action='store_true',
                        help='Enable post-GLB silhouette correction (deform vertices to match input alpha)')
    args = parser.parse_args()

    # Determine test images
    if args.quick:
        test_images = TEST_IMAGES_QUICK
    else:
        test_images = TEST_IMAGES_FULL

    if args.config:
        # Load config from JSON
        with open(args.config) as f:
            config = json.load(f)
        # Merge with champion defaults for missing keys
        full_config = dict(CHAMPION_CONFIG)
        full_config.update(config)
        label = Path(args.config).stem
        run_evaluation(full_config, label=label, test_images=test_images,
                       best_of_n=args.best_of_n,
                       texture_refine=args.texture_refine,
                       silhouette_correct=args.silhouette_correct)

    elif args.sweep:
        # Sweep mode
        param = args.sweep[0]
        values = [float(v) for v in args.sweep[1:]]
        # Cast to int for parameters that require integer values
        if 'steps' in param or 'target' in param or 'size' in param:
            values = [int(v) for v in values]
        print(f"V4 Sweep mode: {param} = {values}")
        all_results = {}
        for val in values:
            config = dict(CHAMPION_CONFIG)
            config[param] = val
            label = f"v4_sweep_{param}_{val}"
            scores = run_evaluation(config, label=label, test_images=test_images,
                                    best_of_n=args.best_of_n,
                                    texture_refine=args.texture_refine,
                       silhouette_correct=args.silhouette_correct)
            if scores:
                all_results[val] = scores
                print(f"\n  {param}={val}: overall={scores['overall']:.1f}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"V4 SWEEP SUMMARY: {param}")
        print(f"{'='*70}")
        dims_short = ['C1_tex_coherence', 'B2_surface_quality',
                       'C2_color_vitality', 'E1_multiview']
        header = f"{'Value':>8s} | {'Overall':>8s}"
        for d in dims_short:
            header += f" | {d[:10]:>10s}"
        print(header)
        print("-" * 70)
        for val, scores in sorted(all_results.items()):
            row = f"{val:8.3f} | {scores['overall']:8.1f}"
            for d in dims_short:
                row += f" | {scores.get(d, 0):10.1f}"
            print(row)
        print(f"{'='*70}")

    elif args.cfg_mp is not None:
        config = dict(CHAMPION_CONFIG)
        config['cfg_mp_strength'] = args.cfg_mp
        run_evaluation(config, label=f"v4_cfg_mp_{args.cfg_mp}",
                       test_images=test_images, best_of_n=args.best_of_n,
                       texture_refine=args.texture_refine,
                       silhouette_correct=args.silhouette_correct)

    else:
        # Default: baseline evaluation
        bon_str = f" (Best-of-{args.best_of_n})" if args.best_of_n > 1 else ""
        refine_str = " +TexRefine" if args.texture_refine else ""
        silcorr_str = " +SilCorr" if args.silhouette_correct else ""
        print(f"=== V4 BASELINE EVALUATION{bon_str}{refine_str}{silcorr_str} ===\n")
        label = f"v4_baseline_bon{args.best_of_n}" if args.best_of_n > 1 else "v4_baseline"
        if args.texture_refine:
            label += "_texrefine"
        if args.silhouette_correct:
            label += "_silcorr"
        run_evaluation(dict(CHAMPION_CONFIG), label=label,
                       test_images=test_images, best_of_n=args.best_of_n,
                       texture_refine=args.texture_refine,
                       silhouette_correct=args.silhouette_correct)


if __name__ == '__main__':
    main()
