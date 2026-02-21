"""
Multi-dimensional GLB quality evaluation system.
Compares rendered 3D views against input reference images.

Scoring Dimensions (100 points total):
  1. Silhouette/Shape    (20 pts) - Binary mask IoU
  2. Contour/Edge        (15 pts) - Edge alignment accuracy
  3. Color Fidelity      (15 pts) - Color distribution match
  4. Detail/Texture      (10 pts) - Perceptual similarity (LPIPS + SSIM)
  5. Artifacts           (15 pts) - Dark spots, speckles, holes, disconnection
  6. Surface Smoothness  (15 pts) - Mesh normal consistency + rendered view smoothness
  7. Texture Coherence   (10 pts) - Crack detection, color continuity, gradient harshness

Priority: Shape/Contour must be high first, then texture quality, then detail/color.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import os


class QualityEvaluator:
    """Multi-dimensional quality scorer for 3D generation outputs."""

    def __init__(self, device='cuda'):
        self.device = device
        self._lpips_model = None

    @property
    def lpips_model(self):
        """Lazy-load LPIPS model."""
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    def evaluate(
        self,
        rendered_views: Dict[str, np.ndarray],
        reference_views: Dict[str, np.ndarray],
        texture_map: Optional[np.ndarray] = None,
        texture_mask: Optional[np.ndarray] = None,
        mesh_vertices=None,
        mesh_faces=None,
    ) -> Dict:
        """
        Evaluate rendered views against reference images.

        Args:
            rendered_views: dict of view_name -> RGB numpy array (H, W, 3)
            reference_views: dict of view_name -> RGBA numpy array (H, W, 4) from input
            texture_map: Optional base_color texture (H, W, 3) for artifact detection
            texture_mask: Optional valid texel mask (H, W) bool
            mesh_vertices: Optional (N, 3) tensor of vertex positions for mesh-level metrics
            mesh_faces: Optional (M, 3) tensor of face indices for mesh-level metrics

        Returns:
            Dict with per-dimension scores and overall score
        """
        scores = {}
        view_scores = {}

        # Match views
        common_views = set(rendered_views.keys()) & set(reference_views.keys())
        if not common_views:
            return {'overall': 0, 'error': 'No matching views'}

        for view_name in common_views:
            rendered = rendered_views[view_name]
            reference = reference_views[view_name]
            vs = self._evaluate_single_view(rendered, reference, view_name)
            view_scores[view_name] = vs

        # Aggregate across views (weighted average)
        dim_names = ['silhouette', 'contour', 'color', 'detail', 'artifacts',
                     'smoothness', 'coherence']
        for dim in dim_names:
            vals = [vs[dim] for vs in view_scores.values() if dim in vs]
            scores[dim] = np.mean(vals) if vals else 0.0

        # Add texture-level artifact detection if available
        if texture_map is not None and texture_mask is not None:
            tex_score = self._evaluate_texture_artifacts(texture_map, texture_mask)
            # Blend view-based and texture-based artifact scores
            scores['artifacts'] = 0.5 * scores['artifacts'] + 0.5 * tex_score['artifact_score']
            scores['texture_metrics'] = tex_score

        # Add mesh-level smoothness if mesh data is available
        if mesh_vertices is not None and mesh_faces is not None:
            mesh_smooth = self._mesh_smoothness_score(mesh_vertices, mesh_faces)
            # Blend view-based and mesh-based smoothness (60% mesh, 40% view)
            scores['smoothness'] = 0.6 * mesh_smooth + 0.4 * scores.get('smoothness', 50.0)

        # Weighted overall score
        weights = {
            'silhouette': 20,
            'contour': 15,
            'color': 15,
            'detail': 10,
            'artifacts': 15,
            'smoothness': 15,
            'coherence': 10,
        }
        total_weight = sum(weights.values())
        scores['overall'] = sum(
            scores.get(dim, 0) * w / total_weight
            for dim, w in weights.items()
        )
        scores['view_scores'] = view_scores
        return scores

    def _evaluate_single_view(
        self,
        rendered: np.ndarray,
        reference: np.ndarray,
        view_name: str,
    ) -> Dict:
        """Evaluate a single rendered view against its reference."""
        scores = {}

        # Ensure same size
        h, w = rendered.shape[:2]
        if reference.shape[:2] != (h, w):
            reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Extract alpha masks
        if reference.shape[2] == 4:
            ref_alpha = reference[:, :, 3] / 255.0
            ref_rgb = reference[:, :, :3]
        else:
            ref_rgb = reference
            ref_alpha = np.ones((h, w), dtype=np.float32)

        # Rendered mask: assume black background means empty
        rend_rgb = rendered[:, :, :3] if rendered.shape[2] >= 3 else rendered
        rend_gray = cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY)
        rend_mask = (rend_gray > 5).astype(np.float32)

        ref_mask = (ref_alpha > 0.5).astype(np.float32)

        # 1. SILHOUETTE / SHAPE (IoU of binary masks)
        scores['silhouette'] = self._silhouette_iou(rend_mask, ref_mask)

        # 2. CONTOUR / EDGE (edge alignment)
        scores['contour'] = self._contour_score(rend_mask, ref_mask, rend_rgb, ref_rgb)

        # 3. COLOR FIDELITY (histogram + L2 in masked region)
        scores['color'] = self._color_score(rend_rgb, ref_rgb, rend_mask, ref_mask)

        # 4. DETAIL / TEXTURE (LPIPS + SSIM in masked region)
        scores['detail'] = self._detail_score(rend_rgb, ref_rgb, rend_mask, ref_mask)

        # 5. ARTIFACTS (dark spots, noise in rendered image)
        scores['artifacts'] = self._artifact_score_view(rend_rgb, rend_mask)

        # 6. SURFACE SMOOTHNESS (high-frequency noise, sparkle detection)
        scores['smoothness'] = self._surface_smoothness_score_view(rend_rgb, rend_mask)

        # 7. TEXTURE COHERENCE (cracks, color discontinuities, gradient harshness)
        scores['coherence'] = self._texture_coherence_score_view(rend_rgb, rend_mask)

        return scores

    def _silhouette_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Binary mask IoU, scaled to 0-100."""
        intersection = (pred_mask * gt_mask).sum()
        union = np.clip(pred_mask + gt_mask, 0, 1).sum()
        if union < 1:
            return 0.0
        iou = intersection / union
        return float(iou * 100)

    def _contour_score(
        self,
        rend_mask: np.ndarray,
        ref_mask: np.ndarray,
        rend_rgb: np.ndarray,
        ref_rgb: np.ndarray,
    ) -> float:
        """Edge alignment score using Canny edges."""
        # Get edges from masks
        rend_edges = cv2.Canny((rend_mask * 255).astype(np.uint8), 50, 150)
        ref_edges = cv2.Canny((ref_mask * 255).astype(np.uint8), 50, 150)

        # Also get content edges
        rend_content_edges = cv2.Canny(rend_rgb, 30, 100)
        ref_content_edges = cv2.Canny(ref_rgb, 30, 100)

        # Combine silhouette and content edges
        rend_all_edges = np.clip(rend_edges.astype(np.float32) + rend_content_edges.astype(np.float32), 0, 255)
        ref_all_edges = np.clip(ref_edges.astype(np.float32) + ref_content_edges.astype(np.float32), 0, 255)

        # Dilate reference edges for tolerance
        kernel = np.ones((5, 5), np.uint8)
        ref_dilated = cv2.dilate(ref_all_edges.astype(np.uint8), kernel, iterations=1)

        # Precision: how many rendered edges are near reference edges
        rend_edge_pixels = rend_all_edges > 0
        if rend_edge_pixels.sum() < 10:
            return 50.0  # No edges to evaluate
        precision = (rend_all_edges[rend_edge_pixels > 0] > 0).astype(np.float32)
        near_ref = ref_dilated[rend_edge_pixels] > 0
        edge_precision = near_ref.astype(np.float32).mean()

        # Recall: how many reference edges are near rendered edges
        rend_dilated = cv2.dilate(rend_all_edges.astype(np.uint8), kernel, iterations=1)
        ref_edge_pixels = ref_all_edges > 0
        if ref_edge_pixels.sum() < 10:
            return 50.0
        near_rend = rend_dilated[ref_edge_pixels] > 0
        edge_recall = near_rend.astype(np.float32).mean()

        # F1 score
        if edge_precision + edge_recall < 1e-6:
            return 0.0
        f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)
        return float(f1 * 100)

    def _color_score(
        self,
        rend_rgb: np.ndarray,
        ref_rgb: np.ndarray,
        rend_mask: np.ndarray,
        ref_mask: np.ndarray,
    ) -> float:
        """Color fidelity score using histogram correlation + masked L2."""
        # Intersection mask (both have content)
        both_mask = ((rend_mask > 0.5) & (ref_mask > 0.5)).astype(np.uint8)
        if both_mask.sum() < 100:
            return 50.0

        # Histogram correlation (per channel)
        hist_corrs = []
        for c in range(3):
            hist_r = cv2.calcHist([rend_rgb], [c], both_mask, [64], [0, 256])
            hist_g = cv2.calcHist([ref_rgb], [c], both_mask, [64], [0, 256])
            cv2.normalize(hist_r, hist_r)
            cv2.normalize(hist_g, hist_g)
            corr = cv2.compareHist(hist_r, hist_g, cv2.HISTCMP_CORREL)
            hist_corrs.append(max(0, corr))
        hist_score = np.mean(hist_corrs)

        # Mean absolute error in masked region
        rend_masked = rend_rgb[both_mask > 0].astype(np.float32)
        ref_masked = ref_rgb[both_mask > 0].astype(np.float32)
        mae = np.abs(rend_masked - ref_masked).mean()
        mae_score = max(0, 1.0 - mae / 128.0)  # 128 = half range

        # Combine: 60% histogram, 40% MAE
        score = 0.6 * hist_score + 0.4 * mae_score
        return float(score * 100)

    def _detail_score(
        self,
        rend_rgb: np.ndarray,
        ref_rgb: np.ndarray,
        rend_mask: np.ndarray,
        ref_mask: np.ndarray,
    ) -> float:
        """Perceptual detail score using LPIPS and custom SSIM."""
        both_mask = ((rend_mask > 0.5) & (ref_mask > 0.5))
        if both_mask.sum() < 100:
            return 50.0

        # Custom SSIM (windowed)
        ssim_val = self._compute_ssim(rend_rgb, ref_rgb, both_mask)

        # LPIPS (perceptual)
        lpips_val = self._compute_lpips(rend_rgb, ref_rgb)

        # Combine: 50% SSIM, 50% LPIPS
        ssim_score = max(0, ssim_val)  # already 0-1
        lpips_score = max(0, 1.0 - lpips_val)  # lower is better, convert to 0-1

        score = 0.5 * ssim_score + 0.5 * lpips_score
        return float(score * 100)

    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute SSIM in masked region."""
        # Convert to float
        a = img1.astype(np.float64) / 255.0
        b = img2.astype(np.float64) / 255.0

        # Simple windowed SSIM
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T

        ssim_vals = []
        for c in range(3):
            mu1 = cv2.filter2D(a[:, :, c], -1, window)
            mu2 = cv2.filter2D(b[:, :, c], -1, window)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(a[:, :, c] ** 2, -1, window) - mu1_sq
            sigma2_sq = cv2.filter2D(b[:, :, c] ** 2, -1, window) - mu2_sq
            sigma12 = cv2.filter2D(a[:, :, c] * b[:, :, c], -1, window) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            # Only in masked region
            valid = mask.astype(bool)
            if valid.sum() > 0:
                ssim_vals.append(ssim_map[valid].mean())

        return float(np.mean(ssim_vals)) if ssim_vals else 0.5

    def _compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute LPIPS distance."""
        # Convert to tensor [-1, 1]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1

        # Resize to 256x256 for LPIPS
        t1 = F.interpolate(t1, size=(256, 256), mode='bilinear', align_corners=False)
        t2 = F.interpolate(t2, size=(256, 256), mode='bilinear', align_corners=False)

        with torch.no_grad():
            dist = self.lpips_model(t1, t2)
        return float(dist.item())

    def _artifact_score_view(self, rend_rgb: np.ndarray, rend_mask: np.ndarray) -> float:
        """Detect artifacts in rendered view."""
        score = 100.0

        mask_bool = rend_mask > 0.5
        if mask_bool.sum() < 100:
            return 50.0

        masked_rgb = rend_rgb[mask_bool]
        gray = cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY)
        masked_gray = gray[mask_bool]

        # 1. Dark spot penalty (-20 max)
        dark_ratio = (masked_gray < 10).sum() / max(masked_gray.size, 1)
        score -= min(20, dark_ratio * 200)

        # 2. Very dark patch penalty (-15 max)
        dark30_ratio = (masked_gray < 30).sum() / max(masked_gray.size, 1)
        if dark30_ratio > 0.05:
            score -= min(15, (dark30_ratio - 0.05) * 150)

        # 3. Speckle detection (-15 max)
        median = cv2.medianBlur(rend_rgb, 3)
        diff = np.abs(rend_rgb.astype(np.float32) - median.astype(np.float32)).max(axis=-1)
        speckle_ratio = ((diff > 40) & mask_bool).sum() / max(mask_bool.sum(), 1)
        score -= min(15, speckle_ratio * 300)

        # 4. Disconnected components penalty (-10 max)
        mask_u8 = (rend_mask * 255).astype(np.uint8)
        n_components, _ = cv2.connectedComponents(mask_u8)
        if n_components > 3:  # 1 background + main object + some tolerance
            score -= min(10, (n_components - 3) * 2)

        # 5. Hole detection (-10 max)
        # Fill the mask and compare
        filled = mask_u8.copy()
        h, w = filled.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        filled_inv = cv2.bitwise_not(filled)
        holes = cv2.bitwise_and(filled_inv, cv2.bitwise_not(mask_u8))
        hole_ratio = holes.sum() / 255 / max(mask_bool.sum(), 1)
        if hole_ratio > 0.01:
            score -= min(10, hole_ratio * 200)

        return max(0, score)

    def _surface_smoothness_score_view(self, rend_rgb: np.ndarray, rend_mask: np.ndarray) -> float:
        """Evaluate surface smoothness from a rendered view.
        Detects high-frequency noise and sparkle indicating normal map issues
        or noisy texture sampling.
        """
        mask_bool = rend_mask > 0.5
        if mask_bool.sum() < 100:
            return 50.0

        score = 100.0
        gray = cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # 1. High-frequency energy via Laplacian (-30 max)
        # Laplacian captures second-order derivatives; high values = noisy surface
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_energy = np.abs(lap[mask_bool]).mean()
        # Smooth surface: < 5, acceptable: 5-10, noisy: > 15
        if lap_energy > 5:
            score -= min(30, (lap_energy - 5) * 3)

        # 2. Local deviation from smoothed version (-35 max)
        # Compare original to Gaussian-blurred version; large diffs = surface noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
        local_diff = np.abs(gray - blurred)
        avg_local_diff = local_diff[mask_bool].mean()
        # Good: < 3, noisy: > 6
        if avg_local_diff > 3:
            score -= min(35, (avg_local_diff - 3) * 7)

        # 3. Interior edge density (-25 max)
        # Erode mask to exclude silhouette edges; interior edges indicate surface bumps
        kernel = np.ones((15, 15), np.uint8)
        interior = cv2.erode(mask_bool.astype(np.uint8), kernel) > 0
        if interior.sum() > 100:
            edges = cv2.Canny(rend_rgb, 30, 80)
            interior_edge_density = edges[interior].sum() / 255 / interior.sum()
            # Smooth: < 0.03, bumpy: > 0.06
            if interior_edge_density > 0.03:
                score -= min(25, (interior_edge_density - 0.03) * 500)

        return max(0, score)

    def _texture_coherence_score_view(self, rend_rgb: np.ndarray, rend_mask: np.ndarray) -> float:
        """Evaluate texture coherence from a rendered view.
        Detects cracking, UV seam artifacts, color discontinuities, and
        texture fragmentation.
        """
        mask_bool = rend_mask > 0.5
        if mask_bool.sum() < 100:
            return 50.0

        score = 100.0
        gray = cv2.cvtColor(rend_rgb, cv2.COLOR_RGB2GRAY)

        # Erode mask to get interior region (away from silhouette)
        kernel = np.ones((10, 10), np.uint8)
        interior = cv2.erode(mask_bool.astype(np.uint8), kernel) > 0

        if interior.sum() < 100:
            return 50.0

        # 1. Dark crack detection (-35 max)
        # Morphological close fills thin dark gaps; the difference reveals cracks
        gray_interior = gray.copy()
        gray_interior[~interior] = 128  # neutral fill for non-interior
        closed = cv2.morphologyEx(gray_interior, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        crack_diff = closed.astype(np.float32) - gray_interior.astype(np.float32)
        crack_ratio = ((crack_diff > 25) & interior).sum() / max(interior.sum(), 1)
        if crack_ratio > 0.005:
            score -= min(35, (crack_ratio - 0.005) * 600)

        # 2. Color gradient harshness (-35 max)
        # Sobel gradient magnitude; harsh gradients = texture discontinuities
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        interior_grad = gradient_mag[interior]
        # Ratio of pixels with harsh gradients (> 40)
        harsh_ratio = (interior_grad > 40).sum() / max(interior.sum(), 1)
        if harsh_ratio > 0.08:
            score -= min(35, (harsh_ratio - 0.08) * 200)

        # 3. Dark region penalty in interior (-20 max)
        # Interior dark spots often indicate failed voxel sampling
        interior_gray = gray[interior].astype(np.float32)
        dark_interior_ratio = (interior_gray < 20).sum() / max(interior.sum(), 1)
        if dark_interior_ratio > 0.01:
            score -= min(20, (dark_interior_ratio - 0.01) * 400)

        return max(0, score)

    def _mesh_smoothness_score(self, vertices, faces) -> float:
        """Evaluate mesh surface smoothness from geometry.
        Measures face normal consistency — smooth surfaces have faces whose normals
        align well with area-weighted vertex normals. Rough/noisy meshes have
        high angular deviation between adjacent faces.

        Args:
            vertices: (N, 3) tensor of vertex positions
            faces: (M, 3) tensor of face indices

        Returns:
            Score 0-100, higher = smoother
        """
        import torch
        import torch.nn.functional as F_t

        vertices = vertices.float().cuda()
        faces = faces.long().cuda()

        # Compute face normals and areas
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_areas = fn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        fn = fn / face_areas  # normalized face normals

        # Area-weighted vertex normals
        vn = torch.zeros_like(vertices)
        weighted_fn = fn * face_areas
        for i in range(3):
            vn.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), weighted_fn)
        vn = F_t.normalize(vn, dim=-1)

        # Face-vertex normal consistency
        # Average vertex normal per face, compared to face normal
        avg_vn = F_t.normalize(
            vn[faces[:, 0]] + vn[faces[:, 1]] + vn[faces[:, 2]], dim=-1
        )
        consistency = (fn * avg_vn).sum(-1).clamp(-1, 1)  # cos(angle)

        # Statistics
        mean_cos = consistency.mean().item()
        rough_face_ratio = (consistency < 0.9).float().mean().item()

        # Score: mean_cos 0.95-1.0 = very smooth, 0.85-0.95 = acceptable, <0.85 = rough
        # Map [0.7, 1.0] → [0, 100]
        score = max(0, min(100, (mean_cos - 0.7) / 0.3 * 100))

        # Extra penalty for high ratio of rough faces
        if rough_face_ratio > 0.1:
            score -= min(30, (rough_face_ratio - 0.1) * 100)

        return max(0, score)

    def _evaluate_texture_artifacts(
        self,
        texture_map: np.ndarray,
        mask: np.ndarray,
    ) -> Dict:
        """Evaluate texture map for artifacts."""
        metrics = {}
        mask_bool = mask.astype(bool)
        valid_texels = texture_map[mask_bool]

        if valid_texels.size == 0:
            return {'artifact_score': 50.0}

        # Luminance
        lum = valid_texels.astype(np.float32).mean(axis=-1)
        metrics['mean_luminance'] = float(lum.mean())

        # Dark ratios
        metrics['dark_pct_10'] = float((lum < 10).sum() / max(lum.size, 1) * 100)
        metrics['dark_pct_30'] = float((lum < 30).sum() / max(lum.size, 1) * 100)

        # Speckle
        median = cv2.medianBlur(texture_map, 3)
        diff = np.abs(texture_map.astype(np.float32) - median.astype(np.float32)).max(axis=-1)
        speckle = ((diff > 40) & mask_bool).sum()
        metrics['speckle_ratio'] = float(speckle / max(mask_bool.sum(), 1) * 100)

        # Spatial frequency (detail level)
        gray = cv2.cvtColor(texture_map, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['spatial_freq'] = float(np.abs(lap[mask_bool]).mean())

        # Utilization
        metrics['utilization'] = float(mask_bool.sum() / max(mask.size, 1) * 100)

        # Compute artifact score from metrics
        artifact_score = 100.0
        if metrics['dark_pct_10'] > 2:
            artifact_score -= min(20, (metrics['dark_pct_10'] - 2) * 5)
        if metrics['dark_pct_30'] > 10:
            artifact_score -= min(15, (metrics['dark_pct_30'] - 10) * 1.5)
        if metrics['speckle_ratio'] > 3:
            artifact_score -= min(15, (metrics['speckle_ratio'] - 3) * 5)
        if metrics['mean_luminance'] < 80:
            artifact_score -= min(10, (80 - metrics['mean_luminance']) * 0.2)

        metrics['artifact_score'] = max(0, artifact_score)
        return metrics


def _get_default_envmap():
    """Create a default uniform white EnvMap for rendering."""
    import torch
    from trellis2.renderers import EnvMap
    # Uniform white HDRI (latlong format, 256x512x3)
    hdri = torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 1.0
    return {'default': EnvMap(hdri)}


_DEFAULT_ENVMAP = None

def render_mesh_at_views(
    mesh,
    view_angles: Dict[str, Tuple[float, float]],
    resolution: int = 512,
    r: float = 2.0,
    fov: float = 36.0,
    envmap=None,
) -> Dict[str, np.ndarray]:
    """
    Render mesh from specific view angles.

    Args:
        mesh: TRELLIS.2 mesh object
        view_angles: dict of view_name -> (yaw_rad, pitch_rad)
        resolution: render resolution
        r: camera distance
        fov: field of view in degrees
        envmap: optional environment map dict (EnvMap name -> EnvMap)

    Returns:
        Dict of view_name -> RGB numpy array
    """
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames

    # Ensure we have an envmap
    global _DEFAULT_ENVMAP
    if envmap is None:
        if _DEFAULT_ENVMAP is None:
            _DEFAULT_ENVMAP = _get_default_envmap()
        envmap = _DEFAULT_ENVMAP

    yaw_list = [angles[0] for angles in view_angles.values()]
    pitch_list = [angles[1] for angles in view_angles.values()]

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaw_list, pitch_list, r, fov
    )

    result = render_frames(
        mesh, extrinsics, intrinsics,
        {'resolution': resolution, 'bg_color': (0, 0, 0)},
        envmap=envmap,
    )

    renders = {}
    for i, view_name in enumerate(view_angles.keys()):
        # Prefer base_color (unlit) for evaluation; fall back to shaded
        key = 'base_color' if 'base_color' in result else 'shaded'
        renders[view_name] = result[key][i]  # RGB numpy array

    return renders


def split_input_image(
    image: Image.Image,
    layout: str = '1x3',
    view_names: List[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Split a composite multi-view input image into individual views.

    Args:
        image: Composite PIL image
        layout: '1x3', '2x3', '2x2', etc.
        view_names: Names for each view cell

    Returns:
        Dict of view_name -> RGBA numpy array
    """
    img_np = np.array(image)
    w, h = image.size

    if layout == '1x3':
        if view_names is None:
            view_names = ['front', 'left', 'right']
        cell_w = w // 3
        views = {}
        for i, name in enumerate(view_names):
            views[name] = img_np[:, i * cell_w:(i + 1) * cell_w]
        return views

    elif layout == '2x3':
        if view_names is None:
            view_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        cell_w, cell_h = w // 3, h // 2
        views = {}
        for idx, name in enumerate(view_names):
            r, c = divmod(idx, 3)
            views[name] = img_np[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
        return views

    elif layout == '2x2':
        if view_names is None:
            view_names = ['front', 'right', 'back', 'left']
        cell_w, cell_h = w // 2, h // 2
        views = {}
        for idx, name in enumerate(view_names):
            r, c = divmod(idx, 2)
            views[name] = img_np[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
        return views

    else:
        return {'front': img_np}
