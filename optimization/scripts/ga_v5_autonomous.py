#!/usr/bin/env python3
"""
V5 Autonomous GA Optimizer for TRELLIS.2 3D Generation Quality.

Self-contained: V5 evaluator + GA operators + autonomous runner.
Designed for overnight unattended operation.

V5 Evaluator improvements over V4:
  - Mesh defect detection via trimesh (holes, roughness, fragmentation)
  - 3-view angle-matched comparison (shape, color, material, gloss)
  - Better calibrated scoring (V4 gave ~37/100 which was too harsh)
  - Spatial grey cluster detection (connected components)

Scoring (100 pts):
  A. Input Match (25):  A1 Silhouette(8) + A2 Color(8) + A3 Detail(9)
  B. Mesh Quality (35): B1 Smoothness(15) + B2 Holes(10) + B3 Fragmentation(10)
  C. Texture (25):      C1 Coherence(10) + C2 Vitality(10) + C3 Consistency(5)
  D. Material (15):     D1 Contour(10) + D2 Plausibility(5)

Usage:
    python ga_v5_autonomous.py --generations 25 --pop-size 6
    python ga_v5_autonomous.py --baseline-only  # Just evaluate champion config
"""

import gc
import os
import sys
import json
import math
import time
import random
import argparse
import traceback
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from PIL import Image

TRELLIS_ROOT = '/workspace/TRELLIS.2'
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, 'optimization', 'scripts'))

# Override o_voxel.postprocess with live-mounted version
import o_voxel
_ovoxel_live = os.path.join(TRELLIS_ROOT, 'o-voxel', 'o_voxel', 'postprocess.py')
if os.path.isfile(_ovoxel_live):
    import importlib.util
    _spec = importlib.util.spec_from_file_location('o_voxel.postprocess', _ovoxel_live)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    o_voxel.postprocess = _mod
    sys.modules['o_voxel.postprocess'] = _mod

# Paths
RESEARCH_DIR = os.path.join(TRELLIS_ROOT, 'optimization', 'research')
POP_V5_PATH = os.path.join(RESEARCH_DIR, 'population_v5.json')
LOG_V5_PATH = os.path.join(RESEARCH_DIR, 'experiment_log_v5.json')
PROGRESS_PATH = os.path.join(RESEARCH_DIR, 'ga_v5_progress.json')
UPLOAD_DIR = os.path.join(TRELLIS_ROOT, 'optimization', 'test_images')

# ============================================================
# TEST EXAMPLES (from auto_optimize.py)
# ============================================================

TEST_EXAMPLES = {
    4: {
        'name': 'floral_shoes',
        'image': 'example4/ChatGPT Image 2026年2月14日 上午01_57_12.png',
        'layout': '1x3',
        'view_names': ['front', 'left', 'right'],
        'challenge': 'Fine floral patterns, bright white surface, smooth manufactured',
    },
    7: {
        'name': 'ultraman_mickey',
        'image': 'example7/image (4).webp',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Dual-character design, mixed colors',
    },
    1: {
        'name': 'bikini_woman',
        'image': 'example1/Gemini_Generated_Image_5yov5c5yov5c5yov.png',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Human body, skin tones, fine detail',
    },
}

VIEW_ANGLES = {
    'front': (0.0, 0.0),
    'back': (math.pi, 0.0),
    'left': (math.pi / 2, 0.0),
    'right': (-math.pi / 2, 0.0),
}

# Champion config from GA v2
CHAMPION_CONFIG = {
    'tex_slat_guidance_strength': 12.0,
    'tex_slat_guidance_rescale': 1.0,
    'tex_slat_sampling_steps': 16,
    'tex_slat_rescale_t': 4.0,
    'shape_slat_guidance_strength': 10.0,
    'shape_slat_guidance_rescale': 0.5,
    'shape_slat_sampling_steps': 12,
    'ss_guidance_strength': 10.0,
    'ss_guidance_rescale': 0.8,
}

FIXED_PARAMS = {
    'resolution': '512',
    'multiview_mode': 'concat',
    'texture_multiview_mode': 'tapa',
    'ss_rescale_t': 5.0,
    'shape_slat_rescale_t': 3.0,
    'ss_sampling_steps': 12,
    'cfg_mode': 'standard',
    'guidance_schedule': 'binary',
}

PARAM_BOUNDS = {
    'tex_slat_guidance_strength': {'min': 5.0, 'max': 15.0, 'type': 'float', 'step': 0.5},
    'tex_slat_guidance_rescale': {'min': 0.7, 'max': 1.0, 'type': 'float', 'step': 0.05},
    'tex_slat_sampling_steps': {'min': 10, 'max': 20, 'type': 'int', 'step': 2},
    'tex_slat_rescale_t': {'min': 2.0, 'max': 6.0, 'type': 'float', 'step': 0.5},
    'shape_slat_guidance_strength': {'min': 7.0, 'max': 14.0, 'type': 'float', 'step': 0.5},
    'shape_slat_guidance_rescale': {'min': 0.3, 'max': 0.8, 'type': 'float', 'step': 0.05},
    'shape_slat_sampling_steps': {'min': 10, 'max': 20, 'type': 'int', 'step': 2},
    'ss_guidance_strength': {'min': 7.0, 'max': 14.0, 'type': 'float', 'step': 0.5},
    'ss_guidance_rescale': {'min': 0.5, 'max': 0.9, 'type': 'float', 'step': 0.05},
}


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _log_memory(label=""):
    gpu_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
    _log(f"  [MEM {label}] GPU_alloc={gpu_alloc:.1f}GB GPU_reserved={gpu_reserved:.1f}GB")


# ============================================================
# V5 QUALITY EVALUATOR
# ============================================================

class QualityEvaluatorV5:
    """
    V5 Quality Evaluator — mesh defects + view comparison + calibration.

    A. Input Match (25 pts):
        A1. Silhouette (8)  — Binary mask IoU per view
        A2. Color (8)       — LAB histogram correlation per view
        A3. Detail (9)      — LPIPS perceptual distance per view

    B. Mesh Quality (35 pts):
        B1. Smoothness (15) — Dihedral angle roughness (trimesh)
        B2. Holes (10)      — Boundary edge loop counting (trimesh)
        B3. Fragmentation (10) — Connected component ratio (trimesh)

    C. Texture (25 pts):
        C1. Coherence (10)  — Crack detection + gradient harshness
        C2. Vitality (10)   — Grey clusters + dynamic range
        C3. Consistency (5) — Multi-view metric variance

    D. Material/Contour (15 pts):
        D1. Contour (10)    — Edge alignment accuracy
        D2. Plausibility (5) — Metallic/roughness sanity
    """

    WEIGHTS = {
        'A1_silhouette': 10, 'A2_color': 10, 'A3_detail': 10,
        'B1_smoothness': 8, 'B2_holes': 7, 'B3_fragmentation': 8,
        'C1_coherence': 12, 'C2_vitality': 10, 'C3_consistency': 5,
        'D1_contour': 12, 'D2_material': 8,
    }

    def __init__(self, device='cuda'):
        self.device = device
        self._lpips_model = None

    @property
    def lpips_model(self):
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
        return self._lpips_model

    def evaluate(self, rendered_views, reference_views,
                 mesh_vertices=None, mesh_faces=None):
        """Main evaluation entry point."""
        scores = {}
        common_views = set(rendered_views.keys()) & set(reference_views.keys())
        if not common_views:
            return {'overall': 0, 'error': 'No matching views'}

        # Per-view scores
        per_view = {}
        for vname in common_views:
            per_view[vname] = self._score_single_view(
                rendered_views[vname], reference_views[vname])

        # Aggregate per-view → dimension scores
        for dim in ['silhouette', 'color', 'detail', 'coherence', 'vitality', 'contour']:
            vals = [pv.get(dim, 50) for pv in per_view.values()]
            scores[f'{"ABCD"["silhouette color detail".split().index(dim)] if dim in "silhouette color detail".split() else "C" if dim in ("coherence","vitality") else "D"}'] = float(np.mean(vals))

        # Simpler aggregation
        scores['A1_silhouette'] = float(np.mean([pv['silhouette'] for pv in per_view.values()]))
        scores['A2_color'] = float(np.mean([pv['color'] for pv in per_view.values()]))
        scores['A3_detail'] = float(np.mean([pv['detail'] for pv in per_view.values()]))
        scores['C1_coherence'] = float(np.mean([pv['coherence'] for pv in per_view.values()]))
        scores['C2_vitality'] = float(np.mean([pv['vitality'] for pv in per_view.values()]))
        scores['D1_contour'] = float(np.mean([pv['contour'] for pv in per_view.values()]))

        # Multi-view consistency
        scores['C3_consistency'] = self._multiview_consistency(
            list(rendered_views.values()))

        # Mesh defect scores
        mesh_scores = self._mesh_defect_scores(mesh_vertices, mesh_faces)
        scores['B1_smoothness'] = float(mesh_scores['smoothness'])
        scores['B2_holes'] = float(mesh_scores['holes'])
        scores['B3_fragmentation'] = float(mesh_scores['fragmentation'])

        # Material plausibility (from rendered views — check for obvious PBR issues)
        scores['D2_material'] = self._material_score(list(rendered_views.values()))

        # Weighted overall
        total_w = sum(self.WEIGHTS.values())
        scores['overall'] = sum(
            scores.get(d, 50) * w / total_w
            for d, w in self.WEIGHTS.items()
        )
        scores['per_view'] = per_view
        scores['mesh_metrics'] = mesh_scores
        return scores

    # ─── Per-View Scoring ──────────────────────────────────────────

    def _score_single_view(self, rendered, reference):
        """Score a single rendered view against its reference."""
        h, w = rendered.shape[:2]
        if reference.shape[:2] != (h, w):
            reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Extract masks
        if reference.shape[2] == 4:
            ref_alpha = reference[:, :, 3] / 255.0
            ref_rgb = reference[:, :, :3]
        else:
            ref_rgb = reference
            ref_alpha = np.ones((h, w), dtype=np.float32)

        rend_rgb = rendered[:, :, :3] if rendered.shape[2] >= 3 else rendered
        rend_gray = cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY)
        rend_mask = (rend_gray > 5).astype(np.float32)
        ref_mask = (ref_alpha > 0.5).astype(np.float32)

        scores = {}
        scores['silhouette'] = self._silhouette_iou(rend_mask, ref_mask)
        scores['color'] = self._color_score(rend_rgb, ref_rgb, rend_mask, ref_mask)
        scores['detail'] = self._detail_score(rend_rgb, ref_rgb, rend_mask, ref_mask)
        scores['contour'] = self._contour_score(rend_mask, ref_mask, rend_rgb, ref_rgb)
        scores['coherence'] = self._coherence_score(rend_rgb, rend_mask)
        scores['vitality'] = self._vitality_score(rend_rgb, rend_mask)
        return scores

    def _silhouette_iou(self, pred, gt):
        intersection = (pred * gt).sum()
        union = np.clip(pred + gt, 0, 1).sum()
        if union < 1:
            return 0.0
        return float(intersection / union * 100)

    def _color_score(self, rend, ref, rend_mask, ref_mask):
        both = ((rend_mask > 0.5) & (ref_mask > 0.5)).astype(np.uint8)
        if both.sum() < 100:
            return 50.0
        corrs = []
        for c in range(3):
            hr = cv2.calcHist([rend], [c], both, [64], [0, 256])
            hg = cv2.calcHist([ref], [c], both, [64], [0, 256])
            cv2.normalize(hr, hr)
            cv2.normalize(hg, hg)
            corrs.append(max(0, cv2.compareHist(hr, hg, cv2.HISTCMP_CORREL)))
        hist_score = np.mean(corrs)
        # MAE
        rend_m = rend[both > 0].astype(np.float32)
        ref_m = ref[both > 0].astype(np.float32)
        mae = np.abs(rend_m - ref_m).mean()
        mae_score = max(0, 1.0 - mae / 128.0)
        return float((0.6 * hist_score + 0.4 * mae_score) * 100)

    def _detail_score(self, rend, ref, rend_mask, ref_mask):
        both = ((rend_mask > 0.5) & (ref_mask > 0.5))
        if both.sum() < 100:
            return 50.0
        # SSIM
        ssim_val = self._ssim(rend, ref, both)
        # LPIPS
        lpips_val = self._lpips(rend, ref)
        return float((0.5 * max(0, ssim_val) + 0.5 * max(0, 1 - lpips_val)) * 100)

    def _ssim(self, a, b, mask):
        a_f = a.astype(np.float64) / 255.0
        b_f = b.astype(np.float64) / 255.0
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        k = cv2.getGaussianKernel(11, 1.5)
        w = k @ k.T
        vals = []
        for c in range(3):
            mu1 = cv2.filter2D(a_f[:, :, c], -1, w)
            mu2 = cv2.filter2D(b_f[:, :, c], -1, w)
            s1 = cv2.filter2D(a_f[:, :, c] ** 2, -1, w) - mu1 ** 2
            s2 = cv2.filter2D(b_f[:, :, c] ** 2, -1, w) - mu2 ** 2
            s12 = cv2.filter2D(a_f[:, :, c] * b_f[:, :, c], -1, w) - mu1 * mu2
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
            valid = mask.astype(bool)
            if valid.sum() > 0:
                vals.append(ssim_map[valid].mean())
        return float(np.mean(vals)) if vals else 0.5

    @torch.no_grad()
    def _lpips(self, a, b):
        t1 = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
        t2 = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
        t1 = F.interpolate(t1, size=(256, 256), mode='bilinear', align_corners=False)
        t2 = F.interpolate(t2, size=(256, 256), mode='bilinear', align_corners=False)
        return float(self.lpips_model(t1, t2).item())

    def _contour_score(self, rend_mask, ref_mask, rend_rgb, ref_rgb):
        rend_edges = cv2.Canny((rend_mask * 255).astype(np.uint8), 50, 150)
        ref_edges = cv2.Canny((ref_mask * 255).astype(np.uint8), 50, 150)
        rend_ce = cv2.Canny(rend_rgb, 30, 100)
        ref_ce = cv2.Canny(ref_rgb, 30, 100)
        rend_all = np.clip(rend_edges.astype(np.float32) + rend_ce.astype(np.float32), 0, 255)
        ref_all = np.clip(ref_edges.astype(np.float32) + ref_ce.astype(np.float32), 0, 255)
        k = np.ones((5, 5), np.uint8)
        ref_d = cv2.dilate(ref_all.astype(np.uint8), k)
        rend_d = cv2.dilate(rend_all.astype(np.uint8), k)
        rp = rend_all > 0
        if rp.sum() < 10:
            return 50.0
        prec = (ref_d[rp] > 0).astype(np.float32).mean()
        rpe = ref_all > 0
        if rpe.sum() < 10:
            return 50.0
        recall = (rend_d[rpe] > 0).astype(np.float32).mean()
        if prec + recall < 1e-6:
            return 0.0
        return float(2 * prec * recall / (prec + recall) * 100)

    def _coherence_score(self, rgb, mask):
        """Texture coherence: cracks + gradient harshness."""
        mask_bool = mask > 0.5
        if mask_bool.sum() < 200:
            return 50.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        k = np.ones((10, 10), np.uint8)
        interior = cv2.erode(mask_bool.astype(np.uint8), k) > 0
        if interior.sum() < 200:
            return 50.0
        s = 100.0
        # Crack detection
        gi = gray.copy()
        gi[~interior] = 128
        closed = cv2.morphologyEx(gi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        crack_diff = closed.astype(np.float32) - gi.astype(np.float32)
        cr = ((crack_diff > 20) & interior).sum() / max(interior.sum(), 1)
        if cr > 0.003:
            s -= min(40, (cr - 0.003) * 700)
        # Gradient harshness
        grad = np.zeros(gray.shape, dtype=np.float32)
        rgbf = rgb.astype(np.float32)
        for c in range(3):
            gx = cv2.Sobel(rgbf[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(rgbf[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
            grad += np.sqrt(gx ** 2 + gy ** 2)
        grad /= 3.0
        p90 = np.percentile(grad[interior], 90)
        thresh = max(35.0, p90 * 0.8)
        harsh = (grad[interior] > thresh).sum() / max(interior.sum(), 1)
        if harsh > 0.08:
            s -= min(30, (harsh - 0.08) * 200)
        return max(0, s)

    def _vitality_score(self, rgb, mask):
        """Color vitality: grey clusters + dynamic range + dark patches."""
        mask_bool = mask > 0.5
        if mask_bool.sum() < 100:
            return 50.0
        s = 100.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[mask_bool].astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1][mask_bool].astype(np.float32)
        val = hsv[:, :, 2][mask_bool].astype(np.float32)
        # Grey ratio (global)
        grey_pix = (sat < 15) & (val > 40) & (val < 200)
        grey_ratio = grey_pix.sum() / max(mask_bool.sum(), 1)
        if grey_ratio > 0.1:
            s -= min(25, (grey_ratio - 0.1) * 80)
        # Grey spatial clusters (connected components on the 2D mask)
        grey_2d = np.zeros(mask.shape, dtype=np.uint8)
        grey_2d[mask_bool] = grey_pix.astype(np.uint8) * 255
        n_clusters, labels, stats, _ = cv2.connectedComponentsWithStats(grey_2d)
        if n_clusters > 1:
            # Largest grey cluster (excluding background 0)
            cluster_sizes = stats[1:, cv2.CC_STAT_AREA]
            largest = cluster_sizes.max() if len(cluster_sizes) > 0 else 0
            largest_ratio = largest / max(mask_bool.sum(), 1)
            if largest_ratio > 0.03:
                s -= min(20, (largest_ratio - 0.03) * 200)
        # Dynamic range
        p5, p95 = np.percentile(gray, [5, 95])
        dr = p95 - p5
        if dr < 40:
            s -= min(20, (40 - dr) * 0.7)
        # Dark patches
        dark_r = (gray < 10).sum() / max(gray.size, 1)
        if dark_r > 0.02:
            s -= min(20, (dark_r - 0.02) * 400)
        return max(0, s)

    def _multiview_consistency(self, views):
        """Score consistency of quality across views."""
        if len(views) < 2:
            return 50.0
        brightness = []
        saturations = []
        for v in views:
            gray = cv2.cvtColor(v[:, :, :3], cv2.COLOR_RGB2GRAY)
            m = gray > 5
            if m.sum() < 100:
                continue
            brightness.append(gray[m].astype(np.float32).mean())
            hsv = cv2.cvtColor(v[:, :, :3], cv2.COLOR_RGB2HSV)
            saturations.append(hsv[:, :, 1][m].astype(np.float32).mean())
        if len(brightness) < 2:
            return 50.0
        s = 100.0
        b_cv = np.std(brightness) / max(np.mean(brightness), 1)
        if b_cv > 0.15:
            s -= min(40, (b_cv - 0.15) * 180)
        s_cv = np.std(saturations) / max(np.mean(saturations), 1)
        if s_cv > 0.2:
            s -= min(30, (s_cv - 0.2) * 120)
        return max(0, s)

    def _material_score(self, views):
        """Basic material plausibility from rendered views."""
        s = 100.0
        # Check for unnaturally uniform or extreme colors
        for v in views[:3]:
            gray = cv2.cvtColor(v[:, :, :3], cv2.COLOR_RGB2GRAY)
            m = gray > 5
            if m.sum() < 100:
                continue
            # Highlight detection (specular-like)
            highlight_ratio = (gray[m] > 250).sum() / max(m.sum(), 1)
            if highlight_ratio > 0.1:
                s -= min(15, (highlight_ratio - 0.1) * 80)
        return max(0, s)

    # ─── Mesh Defect Detection (trimesh) ────────────────────────────

    def _mesh_defect_scores(self, vertices, faces):
        """Compute mesh quality scores using trimesh.

        CALIBRATION NOTE: Raw TRELLIS.2 meshes (before GLB postprocessing) are
        inherently fragmented (body_count ~1000-5000) and have many boundary edges.
        Scores are calibrated relative to the TRELLIS.2 output distribution, not
        absolute mesh quality standards.
        """
        defaults = {'smoothness': 50.0, 'holes': 50.0, 'fragmentation': 50.0,
                    'n_boundary_edges': -1, 'n_boundary_loops': -1,
                    'body_count': -1, 'dihedral_std_deg': -1,
                    'rough_face_ratio': -1, 'n_faces': -1}

        if vertices is None or faces is None:
            return defaults

        try:
            import trimesh
            verts = vertices.cpu().numpy() if torch.is_tensor(vertices) else np.array(vertices)
            fcs = faces.cpu().numpy() if torch.is_tensor(faces) else np.array(faces)
            n_total_faces = len(fcs)

            # Always use full mesh — random face subsampling destroys
            # connectivity and boundary topology, making B2/B3 meaningless.
            # Dihedral angle analysis already subsamples angles internally.
            mesh = trimesh.Trimesh(vertices=verts, faces=fcs, process=False)
            _log(f"    [mesh] Full mesh: {n_total_faces} faces, {len(verts)} verts")

            result = dict(defaults)
            result['n_faces'] = n_total_faces

            # B1. Surface Smoothness — dihedral angle roughness
            try:
                if hasattr(mesh, 'face_adjacency') and len(mesh.face_adjacency) > 0:
                    angles = mesh.face_adjacency_angles
                    if len(angles) > 2000000:
                        rng = np.random.RandomState(42)
                        angles = angles[rng.choice(len(angles), 2000000, replace=False)]
                    deg = np.degrees(angles)
                    dih_std = float(np.std(deg))
                    dih_mean = float(np.mean(deg))
                    rough_ratio = float((deg > 30).sum() / max(len(deg), 1))
                    result['dihedral_std_deg'] = round(dih_std, 2)
                    result['dihedral_mean_deg'] = round(dih_mean, 2)
                    result['rough_face_ratio'] = round(rough_ratio, 4)
                    _log(f"    [mesh] Dihedral: mean={dih_mean:.1f}° std={dih_std:.1f}° rough={rough_ratio:.3f}")

                    # TRELLIS.2 calibration: raw meshes typically have std=20-70
                    # (high std from voxel staircase + marching cubes artifacts)
                    # Lower std = smoother geometry = better parameters
                    # Map std [15, 80] → score [90, 15]
                    sm = np.clip(90 - (dih_std - 15) * 1.15, 15, 95)
                    # Bonus for very smooth
                    if dih_std < 20:
                        sm = min(100, sm + 5)
                    # Penalty for very rough
                    if rough_ratio > 0.25:
                        sm -= min(15, (rough_ratio - 0.25) * 25)
                    result['smoothness'] = max(0, min(100, float(sm)))
            except Exception as e:
                _log(f"    [mesh] WARNING smoothness: {e}")

            # B2. Holes — boundary edge ratio (vectorized for speed)
            try:
                # Use face_adjacency to find boundary: edges shared by only 1 face
                total_edges = len(mesh.edges_unique)
                # Number of interior edges = face_adjacency pairs
                n_interior = len(mesh.face_adjacency) if hasattr(mesh, 'face_adjacency') else 0
                n_boundary = total_edges - n_interior
                boundary_ratio = n_boundary / max(total_edges, 1)
                result['n_boundary_edges'] = n_boundary
                result['boundary_ratio'] = round(boundary_ratio, 4)
                _log(f"    [mesh] Boundary: {n_boundary}/{total_edges} edges ({boundary_ratio:.3f})")

                # TRELLIS.2 calibration: raw meshes have boundary_ratio 0.3-0.9
                # (they're NOT watertight). Lower ratio = fewer holes = better params.
                # Map ratio [0.2, 0.8] → score [80, 25]
                hs = np.clip(80 - (boundary_ratio - 0.2) * 91.7, 15, 90)
                if boundary_ratio < 0.1:
                    hs = min(100, hs + 10)
                result['holes'] = max(0, min(100, float(hs)))
            except Exception as e:
                _log(f"    [mesh] WARNING holes: {e}")

            # B3. Fragmentation — connected components
            try:
                from scipy import sparse as sp
                nf = len(mesh.faces)
                if hasattr(mesh, 'face_adjacency') and len(mesh.face_adjacency) > 0:
                    adj = mesh.face_adjacency
                    graph = sp.coo_matrix(
                        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
                        shape=(nf, nf)).tocsr()
                    n_comp = sp.csgraph.connected_components(graph, directed=False)[0]
                else:
                    n_comp = 1
                result['body_count'] = n_comp
                _log(f"    [mesh] Components: {n_comp}")

                # TRELLIS.2 calibration: raw meshes have 2000-15000 components
                # Fewer = better geometry convergence = better params
                # Use log scale: log10(comp) [2.5, 4.5] → score [85, 20]
                if n_comp <= 1:
                    fs = 100.0
                else:
                    log_c = math.log10(max(n_comp, 1))
                    fs = np.clip(85 - (log_c - 2.5) * 32.5, 15, 95)
                    if n_comp < 500:
                        fs = min(100, fs + 10)
                result['fragmentation'] = max(0, min(100, float(fs)))
            except Exception as e:
                _log(f"    [mesh] WARNING fragmentation: {e}")

            return result

        except Exception as e:
            _log(f"    [mesh] WARNING defect analysis failed: {e}")
            traceback.print_exc()
            return defaults


# ============================================================
# RENDERING
# ============================================================

def get_envmap():
    from trellis2.renderers import EnvMap
    hdri = torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.8
    return EnvMap(hdri)


def render_mesh_at_views(mesh, view_angles, resolution=512, r=2.0, fov=36.0, envmap=None):
    """Render mesh from specific angles. Returns dict of view_name -> RGB array."""
    from trellis2.utils.render_utils import (
        yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames)

    yaws = [a[0] for a in view_angles.values()]
    pitchs = [a[1] for a in view_angles.values()]

    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    result = render_frames(
        mesh, extr, intr,
        {'resolution': resolution, 'bg_color': (0, 0, 0)},
        envmap=envmap,
    )

    renders = {}
    key = 'base_color' if 'base_color' in result else 'shaded'
    for i, vname in enumerate(view_angles.keys()):
        renders[vname] = result[key][i]
    return renders


def split_input_image(image, layout='1x3', view_names=None):
    """Split composite input image into individual views."""
    img_np = np.array(image)
    w, h = image.size
    if layout == '1x3':
        if view_names is None:
            view_names = ['front', 'left', 'right']
        cell_w = w // 3
        return {name: img_np[:, i * cell_w:(i + 1) * cell_w]
                for i, name in enumerate(view_names)}
    return {'front': img_np}


# ============================================================
# GA OPERATORS
# ============================================================

def tournament_select(individuals, k=3):
    evald = [i for i in individuals if i.get('fitness')]
    if not evald:
        return random.choice(individuals)
    t = random.sample(evald, min(k, len(evald)))
    return max(t, key=lambda x: x['fitness']['overall'])


def blend_crossover(p1, p2, bounds, alpha=0.3):
    child = {}
    for name, bound in bounds.items():
        v1 = p1['params'].get(name)
        v2 = p2['params'].get(name)
        if v1 is None or v2 is None:
            child[name] = v1 if v1 is not None else v2
            if child[name] is None:
                child[name] = (bound['min'] + bound['max']) / 2
            continue
        lo = max(bound['min'], min(v1, v2) - alpha * abs(v2 - v1))
        hi = min(bound['max'], max(v1, v2) + alpha * abs(v2 - v1))
        val = random.uniform(lo, hi)
        step = bound.get('step', 0.01)
        val = round(val / step) * step
        child[name] = round(val, 4) if bound['type'] == 'float' else int(val)
    return child


def gaussian_mutate(params, bounds, rate=0.15, sigma=0.2):
    m = deepcopy(params)
    mutated_any = False
    for name, bound in bounds.items():
        if name not in m:
            m[name] = (bound['min'] + bound['max']) / 2
        if random.random() > rate:
            continue
        rng = bound['max'] - bound['min']
        noise = random.gauss(0, sigma * rng)
        val = m[name] + noise
        val = max(bound['min'], min(bound['max'], val))
        step = bound.get('step', 0.01)
        val = round(val / step) * step
        m[name] = round(val, 4) if bound['type'] == 'float' else int(val)
        mutated_any = True
    if not mutated_any:
        name = random.choice(list(bounds.keys()))
        bound = bounds[name]
        rng = bound['max'] - bound['min']
        m[name] = m.get(name, (bound['min'] + bound['max']) / 2)
        val = m[name] + random.gauss(0, sigma * rng)
        val = max(bound['min'], min(bound['max'], val))
        step = bound.get('step', 0.01)
        m[name] = round(round(val / step) * step, 4) if bound['type'] == 'float' else int(round(val / step) * step)
    return m


def create_next_generation(pop):
    bounds = pop['parameter_bounds']
    inds = pop['individuals']
    size = pop['population_size']
    gen = pop['generation'] + 1
    evald = sorted(
        [i for i in inds if i.get('fitness')],
        key=lambda x: x['fitness']['overall'], reverse=True)
    if not evald:
        return inds

    # Elites
    n_elite = max(1, int(size * 0.25))
    new_gen = []
    for e in evald[:n_elite]:
        p = deepcopy(e)
        p['generation'] = gen
        p['origin'] = f'elite_from_gen{gen - 1}'
        p['status'] = 'evaluated'
        new_gen.append(p)

    # Offspring
    for i in range(size - len(new_gen)):
        if random.random() < 0.6 and len(evald) >= 2:
            p1 = tournament_select(evald)
            p2 = tournament_select(evald)
            for _ in range(5):
                if p1['id'] != p2['id']:
                    break
                p2 = tournament_select(evald)
            child_params = blend_crossover(p1, p2, bounds)
            origin = f'cross({p1["id"][:8]},{p2["id"][:8]})'
        else:
            parent = tournament_select(evald)
            child_params = deepcopy(parent['params'])
            origin = f'mut({parent["id"][:8]})'
        child_params = gaussian_mutate(child_params, bounds)
        new_gen.append({
            'id': f'v5g{gen}-{i + 1:03d}',
            'generation': gen,
            'origin': origin,
            'params': child_params,
            'fitness': None,
            'seed': random.randint(1, 10000),
            'status': 'queued',
        })
    return new_gen


# ============================================================
# EVALUATION ENGINE
# ============================================================

def evaluate_individual(ind, pipeline, evaluator, examples, envmap, render_dir):
    """Evaluate a single individual."""
    config = dict(FIXED_PARAMS)
    config.update(ind['params'])
    seed = ind.get('seed', 42)

    total_score = 0
    n_ex = 0
    per_example = {}
    dim_sums = {}

    for eid, ex in examples.items():
        try:
            _log(f"    Ex{eid} ({ex['info']['name']})...")
            t0 = time.time()

            outputs = pipeline.run(
                ex['pipeline_input'],
                seed=seed,
                preprocess_image=True,
                sparse_structure_sampler_params={
                    'steps': config.get('ss_sampling_steps', 12),
                    'guidance_strength': config.get('ss_guidance_strength', 10.0),
                    'guidance_rescale': config.get('ss_guidance_rescale', 0.8),
                    'rescale_t': config.get('ss_rescale_t', 5.0),
                },
                shape_slat_sampler_params={
                    'steps': config.get('shape_slat_sampling_steps', 12),
                    'guidance_strength': config.get('shape_slat_guidance_strength', 10.0),
                    'guidance_rescale': config.get('shape_slat_guidance_rescale', 0.5),
                    'rescale_t': config.get('shape_slat_rescale_t', 3.0),
                },
                tex_slat_sampler_params={
                    'steps': config.get('tex_slat_sampling_steps', 16),
                    'guidance_strength': config.get('tex_slat_guidance_strength', 12.0),
                    'guidance_rescale': config.get('tex_slat_guidance_rescale', 1.0),
                    'guidance_interval': [0.0, 1.0],
                    'rescale_t': config.get('tex_slat_rescale_t', 4.0),
                },
                pipeline_type='512',
                return_latent=True,
                multiview_mode=config.get('multiview_mode', 'concat'),
                texture_multiview_mode=config.get('texture_multiview_mode', 'tapa'),
            )
            meshes, latents = outputs
            mesh = meshes[0]
            gen_time = time.time() - t0

            # Render at matching view angles
            vnames = ex['info']['view_names']
            va = {vn: VIEW_ANGLES[vn] for vn in vnames if vn in VIEW_ANGLES}
            rendered = render_mesh_at_views(mesh, va, resolution=512, r=2.0, fov=36.0, envmap=envmap)

            # Evaluate
            mesh_v = mesh.vertices if hasattr(mesh, 'vertices') else None
            mesh_f = mesh.faces if hasattr(mesh, 'faces') else None
            scores = evaluator.evaluate(
                rendered, ex['reference_views'],
                mesh_vertices=mesh_v, mesh_faces=mesh_f)

            overall = scores.get('overall', 0)
            per_example[str(eid)] = {k: round(v, 2) for k, v in scores.items()
                                      if isinstance(v, (int, float))}
            total_score += overall
            n_ex += 1

            for dim in evaluator.WEIGHTS:
                dim_sums[dim] = dim_sums.get(dim, 0) + scores.get(dim, 0)

            # Save renders
            for vn, img in rendered.items():
                rp = render_dir / f'{ind["id"]}_ex{eid}_{vn}.png'
                Image.fromarray(img).save(str(rp))

            _log(f"    → {overall:.1f}/100 ({gen_time:.0f}s)")

            # Cleanup
            del mesh, meshes, latents, outputs, rendered
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            _log(f"    ERROR ex{eid}: {e}")
            traceback.print_exc()
            per_example[str(eid)] = {'overall': 0, 'error': str(e)}
            # Try CUDA recovery
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

    avg = total_score / max(n_ex, 1)
    per_dim = {d: round(dim_sums.get(d, 0) / max(n_ex, 1), 2) for d in evaluator.WEIGHTS}

    return {
        'overall': round(avg, 2),
        'per_example': per_example,
        'per_dimension': per_dim,
    }


# ============================================================
# MAIN GA LOOP
# ============================================================

class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_progress(pop, log_data, start_time, gen):
    """Save current state for monitoring."""
    evald = [i for i in pop['individuals'] if i.get('fitness')]
    evald.sort(key=lambda x: x['fitness']['overall'], reverse=True)
    best = evald[0] if evald else None

    progress = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_min': round((time.time() - start_time) / 60, 1),
        'generation': gen,
        'total_evaluated': len(evald),
        'best_id': best['id'] if best else None,
        'best_score': best['fitness']['overall'] if best else 0,
        'best_params': best['params'] if best else {},
        'top5': [{
            'id': i['id'], 'score': i['fitness']['overall'],
            'dims': i['fitness'].get('per_dimension', {}),
        } for i in evald[:5]],
    }
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2, cls=_NumpySafeEncoder)

    with open(POP_V5_PATH, 'w') as f:
        json.dump(pop, f, indent=2, cls=_NumpySafeEncoder)

    with open(LOG_V5_PATH, 'w') as f:
        json.dump(log_data, f, indent=2, cls=_NumpySafeEncoder)


def main():
    parser = argparse.ArgumentParser(description='V5 GA Optimizer')
    parser.add_argument('--generations', type=int, default=25)
    parser.add_argument('--pop-size', type=int, default=6)
    parser.add_argument('--examples', type=str, default='4,7,1')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    example_ids = [int(x) for x in args.examples.split(',')]
    start_time = time.time()

    _log("=" * 60)
    _log("V5 AUTONOMOUS GA OPTIMIZER")
    _log(f"Generations: {args.generations}, Pop: {args.pop_size}")
    _log(f"Examples: {example_ids}")
    _log("=" * 60)

    # Load pipeline
    _log("Loading pipeline...")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    _log("Pipeline loaded.")
    _log_memory("pipeline")

    envmap = get_envmap()
    evaluator = QualityEvaluatorV5(device='cuda')

    # Load test examples
    examples = {}
    for eid in example_ids:
        if eid not in TEST_EXAMPLES:
            continue
        info = TEST_EXAMPLES[eid]
        img_path = os.path.join(UPLOAD_DIR, info['image'])
        if not os.path.exists(img_path):
            _log(f"  WARNING: {img_path} not found, skipping example {eid}")
            continue
        image = Image.open(img_path).convert('RGBA')
        views = split_input_image(image, info['layout'], info['view_names'])
        pipeline_input = {vn: Image.fromarray(va) for vn, va in views.items()}
        examples[eid] = {
            'info': info,
            'reference_views': views,
            'pipeline_input': pipeline_input,
        }
        _log(f"  Loaded example {eid}: {info['name']}")

    if not examples:
        _log("ERROR: No examples loaded!")
        return

    # Output directories
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(TRELLIS_ROOT) / 'optimization' / f'ga_v5_run_{ts}'
    render_dir = out_dir / 'renders'
    render_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or load population
    if os.path.exists(POP_V5_PATH):
        with open(POP_V5_PATH) as f:
            pop = json.load(f)
        _log(f"Loaded existing V5 population (gen {pop['generation']})")
    else:
        # Fresh population seeded with champion + random variations
        individuals = [{
            'id': 'v5-champion',
            'generation': 0,
            'origin': 'ga_v2_champion',
            'params': dict(CHAMPION_CONFIG),
            'fitness': None,
            'seed': 42,
            'status': 'queued',
        }]
        # Generate random variations
        for i in range(args.pop_size - 1):
            params = gaussian_mutate(dict(CHAMPION_CONFIG), PARAM_BOUNDS, rate=0.5, sigma=0.3)
            individuals.append({
                'id': f'v5g0-{i + 1:03d}',
                'generation': 0,
                'origin': 'random_init',
                'params': params,
                'fitness': None,
                'seed': random.randint(1, 10000),
                'status': 'queued',
            })
        pop = {
            'version': 'v5.0',
            'description': 'V5 GA: mesh defects + view comparison + better calibration',
            'generation': 0,
            'population_size': args.pop_size,
            'elite_ratio': 0.25,
            'mutation_rate': 0.15,
            'crossover_rate': 0.6,
            'parameter_bounds': PARAM_BOUNDS,
            'fixed_params': FIXED_PARAMS,
            'individuals': individuals,
        }
        _log(f"Created fresh V5 population ({args.pop_size} individuals)")

    # Initialize experiment log
    if os.path.exists(LOG_V5_PATH):
        with open(LOG_V5_PATH) as f:
            log_data = json.load(f)
    else:
        log_data = {'version': 'v5.0', 'experiments': []}

    # Baseline-only mode
    if args.baseline_only:
        _log("\n=== BASELINE EVALUATION ===")
        champion_ind = {
            'id': 'v5-baseline',
            'params': dict(CHAMPION_CONFIG),
            'seed': 42,
        }
        fitness = evaluate_individual(
            champion_ind, pipeline, evaluator, examples, envmap, render_dir)
        _log(f"\nV5 Baseline Score: {fitness['overall']:.1f}/100")
        for d, s in fitness['per_dimension'].items():
            _log(f"  {d:20s}: {s:.1f}")
        for eid, es in fitness['per_example'].items():
            _log(f"  Example {eid}: {es.get('overall', 0):.1f}")
        return

    # ── GA LOOP ──────────────────────────────────────────────
    _log(f"\nStarting GA loop: {args.generations} generations")

    for gen_idx in range(args.generations):
        gen_start = time.time()
        _log(f"\n{'=' * 60}")
        _log(f"GENERATION {gen_idx + 1}/{args.generations} (pop gen {pop['generation']})")
        _log(f"{'=' * 60}")

        # Evaluate queued individuals
        queued = [i for i in pop['individuals'] if i['status'] == 'queued']
        _log(f"Evaluating {len(queued)} queued individuals...")

        for qi, ind in enumerate(queued):
            _log(f"\n  [{qi + 1}/{len(queued)}] {ind['id']} ({ind['origin']})")
            _log(f"    Params: " + ", ".join(f"{k}={v}" for k, v in sorted(ind['params'].items())))

            try:
                fitness = evaluate_individual(
                    ind, pipeline, evaluator, examples, envmap, render_dir)
            except Exception as e:
                _log(f"  FATAL: {e}")
                fitness = {'overall': 0, 'per_example': {}, 'per_dimension': {},
                           'error': str(e)}
                torch.cuda.empty_cache()

            ind['fitness'] = fitness
            ind['status'] = 'evaluated'

            # Log experiment
            log_data['experiments'].append({
                'id': f'v5-{ind["id"]}',
                'timestamp': datetime.now().isoformat(),
                'individual': ind['id'],
                'generation': ind['generation'],
                'config': {**FIXED_PARAMS, **ind['params']},
                'fitness': fitness,
            })

            _log(f"  → Score: {fitness['overall']:.1f}/100")

            # Save progress after each evaluation
            save_progress(pop, log_data, start_time, pop['generation'])

        # Generation summary
        evald = sorted(
            [i for i in pop['individuals'] if i.get('fitness')],
            key=lambda x: x['fitness']['overall'], reverse=True)
        gen_time = (time.time() - gen_start) / 60

        _log(f"\n--- Generation {pop['generation']} Summary ({gen_time:.1f} min) ---")
        _log(f"  Best: {evald[0]['id']} = {evald[0]['fitness']['overall']:.1f}/100")
        if len(evald) > 1:
            _log(f"  2nd:  {evald[1]['id']} = {evald[1]['fitness']['overall']:.1f}/100")
        _log(f"  Best dimensions:")
        for d, s in evald[0]['fitness'].get('per_dimension', {}).items():
            _log(f"    {d:20s}: {s:.1f}")

        # Evolve
        if gen_idx < args.generations - 1:
            _log("\nEvolving next generation...")
            new_inds = create_next_generation(pop)
            pop['individuals'] = new_inds
            pop['generation'] += 1

            queued_new = [i for i in new_inds if i['status'] == 'queued']
            elites = [i for i in new_inds if 'elite' in i.get('origin', '')]
            _log(f"  Gen {pop['generation']}: {len(elites)} elites + {len(queued_new)} new")

        save_progress(pop, log_data, start_time, pop['generation'])

        # Elapsed time check
        elapsed_h = (time.time() - start_time) / 3600
        _log(f"\nTotal elapsed: {elapsed_h:.1f}h")

    # ── FINAL SUMMARY ──────────────────────────────────────────
    elapsed_total = (time.time() - start_time) / 3600
    evald = sorted(
        [i for i in pop['individuals'] if i.get('fitness')],
        key=lambda x: x['fitness']['overall'], reverse=True)

    _log(f"\n{'=' * 60}")
    _log(f"V5 GA OPTIMIZATION COMPLETE")
    _log(f"{'=' * 60}")
    _log(f"Total time: {elapsed_total:.1f}h")
    _log(f"Generations: {pop['generation']}")
    _log(f"Total evaluated: {len(log_data['experiments'])}")

    if evald:
        _log(f"\nBest Individual: {evald[0]['id']}")
        _log(f"Best Score: {evald[0]['fitness']['overall']:.1f}/100")
        _log(f"Best Params:")
        for k, v in sorted(evald[0]['params'].items()):
            _log(f"  {k}: {v}")
        _log(f"\nPer-Dimension Breakdown:")
        for d, s in evald[0]['fitness'].get('per_dimension', {}).items():
            _log(f"  {d:20s}: {s:.1f}")
        _log(f"\nTop 5:")
        for i in evald[:5]:
            _log(f"  {i['id']:15s}: {i['fitness']['overall']:.1f}")

    _log(f"\nResults saved to:")
    _log(f"  Population: {POP_V5_PATH}")
    _log(f"  Experiments: {LOG_V5_PATH}")
    _log(f"  Progress: {PROGRESS_PATH}")
    _log(f"  Renders: {render_dir}")


if __name__ == '__main__':
    main()
