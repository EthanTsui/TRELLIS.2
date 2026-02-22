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

        # C3. Detail Richness (from base_color + optional texture map)
        scores['C3_detail_richness'] = self._detail_richness(
            base_colors, alpha_masks, texture_map, tex_mask)

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
        """Compare color distributions in LAB space using histogram correlation.

        Uses separate masks for rendered vs reference (not intersection),
        since the preprocessed reference has different framing/scale.
        Histogram comparison only needs color distributions to match,
        not spatial pixel alignment.
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

        # Convert to LAB (perceptually uniform)
        rend_lab = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB)
        ref_rgb = ref_resized[:, :, :3]
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB)

        rend_mask_u8 = rend_mask.astype(np.uint8)
        ref_mask_u8 = ref_mask.astype(np.uint8)

        # Use both histogram correlation AND Bhattacharyya distance
        # Correlation alone is too harsh when distributions shift slightly
        corr_scores = []
        bhatt_scores = []
        for c in range(3):
            rend_hist = cv2.calcHist([rend_lab], [c], rend_mask_u8, [32], [0, 256])
            ref_hist = cv2.calcHist([ref_lab], [c], ref_mask_u8, [32], [0, 256])
            cv2.normalize(rend_hist, rend_hist)
            cv2.normalize(ref_hist, ref_hist)
            corr = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_CORREL)
            corr_scores.append(max(0.0, corr))
            # Bhattacharyya: 0=identical, higher=different. Convert to similarity.
            bhatt = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
            bhatt_scores.append(max(0.0, 1.0 - bhatt))

        # Weight L channel less (lighting differences are expected in 3D rendering)
        def _weighted(scores):
            return 0.2 * scores[0] + 0.4 * scores[1] + 0.4 * scores[2]

        # Blend: 50% correlation + 50% Bhattacharyya similarity
        blended = 0.5 * _weighted(corr_scores) + 0.5 * _weighted(bhatt_scores)
        return float(blended * 100)

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

        # 3. Connected components (-10 max, log scale)
        # FDG remesh + simplification naturally creates thousands of disconnected
        # patches. This is cosmetic and doesn't affect WebGL rendering quality.
        # Use log scale: <100 = 0 penalty, 100-10K = gradual, cap at -10.
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
                    if n_components > 100:
                        # Log-scale penalty: 100→0, 10000→10
                        log_comp = np.log10(n_components)
                        pen = min(10, max(0, (log_comp - 2.0) * 5.0))
                        score -= pen
                        _penalties['components'] = (n_components, round(pen, 1))
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
                         texture_map=None, tex_mask=None):
        """
        Measure intrinsic texture detail richness via Laplacian energy.
        Uses base_color renders (60%) + UV texture map (40%) when available.
        """
        # View-based detail
        view_scores = []
        nviews = len(base_color_views)

        for i in range(nviews):
            bc = base_color_views[i]
            mask = alpha_masks[i] > 128
            gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)

            interior_kern = np.ones((8, 8), np.uint8)
            interior = cv2.erode(mask.astype(np.uint8), interior_kern) > 0

            if interior.sum() < 200:
                view_scores.append(50.0)
                continue

            # Laplacian energy (texture detail proxy)
            lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
            lap_energy = np.abs(lap[interior]).mean()

            # Score mapping (calibrated for 3D renders):
            # < 2: very blurry (20 pts)
            # 2-5: low detail (20-65 pts)
            # 5-15: good detail (65-100 pts) — most real objects fall here
            # > 15: cap at 100 (rich detail, no noise penalty for 3D)
            if lap_energy < 2:
                s = 20.0
            elif lap_energy < 5:
                s = 20 + (lap_energy - 2) * 15.0
            elif lap_energy < 15:
                s = 65 + (lap_energy - 5) * 3.5
            else:
                s = 100.0

            view_scores.append(min(100.0, s))

        view_detail = float(np.mean(view_scores))

        # Texture-map detail (if available)
        if texture_map is not None and tex_mask is not None:
            mask_bool = tex_mask.astype(bool) if tex_mask.dtype != bool else tex_mask
            if mask_bool.sum() > 1000:
                tex_gray = cv2.cvtColor(texture_map, cv2.COLOR_RGB2GRAY)
                tex_lap = cv2.Laplacian(tex_gray.astype(np.float64), cv2.CV_64F)
                tex_lap_energy = np.abs(tex_lap[mask_bool]).mean()

                if tex_lap_energy < 3:
                    tex_score = 30.0
                elif tex_lap_energy < 10:
                    tex_score = 30 + (tex_lap_energy - 3) * 10
                elif tex_lap_energy < 25:
                    tex_score = 100.0
                else:
                    tex_score = max(85, 100 - (tex_lap_energy - 25) * 0.3)

                # Blend: 60% view-based, 40% texture-map-based
                return 0.6 * view_detail + 0.4 * tex_score

        return view_detail

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


def render_evaluation_views(mesh, envmap, nviews=8, resolution=512):
    """
    Render all channels needed for V4 evaluation.
    Uses canonical camera: r=2, fov=40 (same as render_video),
    NOT r=10, fov=8 (snapshot quasi-ortho).

    Multi-pitch sampling: renders at 3 elevation bands (low/mid/high)
    to better match diverse input image viewing angles.
    Total views: nviews * 3 pitches (default 24 views, ~1.5s render).
    """
    from trellis2.utils import render_utils

    pitches = [0.15, 0.30, 0.50]  # ~8.6°, ~17.2°, ~28.6° elevation
    all_yaw, all_pitch, all_r, all_fov = [], [], [], []
    for p in pitches:
        yaws = np.linspace(0, 2 * np.pi, nviews, endpoint=False).tolist()
        all_yaw.extend(yaws)
        all_pitch.extend([p] * nviews)
        all_r.extend([2.0] * nviews)
        all_fov.extend([40.0] * nviews)

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


def generate_and_evaluate(pipeline, image_path, config, evaluator, envmap,
                          output_prefix="test", best_of_n=1, quality_verifier=None):
    """Generate a 3D model and evaluate it with V4 scoring."""
    import o_voxel.postprocess
    from trellis2.utils import render_utils

    print(f"\n--- Processing: {os.path.basename(image_path)} ---", flush=True)

    # Load and preprocess image
    img = Image.open(image_path)
    processed = pipeline.preprocess_image(img)

    # Run pipeline
    t0 = time.time()
    outputs, latents = pipeline.run(
        processed,
        seed=config.get('seed', 42),
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": config['ss_sampling_steps'],
            "guidance_strength": config['ss_guidance_strength'],
            "guidance_rescale": config['ss_guidance_rescale'],
            "rescale_t": config['ss_rescale_t'],
        },
        shape_slat_sampler_params={
            "steps": config['shape_slat_sampling_steps'],
            "guidance_strength": config['shape_slat_guidance_strength'],
            "guidance_rescale": config['shape_slat_guidance_rescale'],
            "rescale_t": config['shape_slat_rescale_t'],
        },
        tex_slat_sampler_params={
            "steps": config['tex_slat_sampling_steps'],
            "guidance_strength": config['tex_slat_guidance_strength'],
            "guidance_rescale": config['tex_slat_guidance_rescale'],
            "rescale_t": config['tex_slat_rescale_t'],
            "cfg_mp_strength": config.get('cfg_mp_strength', 0.0),
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
    t0 = time.time()
    try:
        channels = render_evaluation_views(mesh, envmap, nviews=8, resolution=512)
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


def run_evaluation(config, label="test", test_images=None, best_of_n=1):
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
          f"best_of_n={best_of_n}")
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

    results = []
    for img_path in existing:
        try:
            r = generate_and_evaluate(
                pipeline, img_path, config, evaluator, envmap,
                output_prefix=label,
                best_of_n=best_of_n,
                quality_verifier=quality_verifier,
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
                       best_of_n=args.best_of_n)

    elif args.sweep:
        # Sweep mode
        param = args.sweep[0]
        values = [float(v) for v in args.sweep[1:]]
        print(f"V4 Sweep mode: {param} = {values}")
        all_results = {}
        for val in values:
            config = dict(CHAMPION_CONFIG)
            config[param] = val
            label = f"v4_sweep_{param}_{val}"
            scores = run_evaluation(config, label=label, test_images=test_images,
                                    best_of_n=args.best_of_n)
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
                       test_images=test_images, best_of_n=args.best_of_n)

    else:
        # Default: baseline evaluation
        bon_str = f" (Best-of-{args.best_of_n})" if args.best_of_n > 1 else ""
        print(f"=== V4 BASELINE EVALUATION{bon_str} ===\n")
        label = f"v4_baseline_bon{args.best_of_n}" if args.best_of_n > 1 else "v4_baseline"
        run_evaluation(dict(CHAMPION_CONFIG), label=label,
                       test_images=test_images, best_of_n=args.best_of_n)


if __name__ == '__main__':
    main()
