"""
Quality verification for Best-of-N candidate selection.

V4-aligned scoring: uses the same metrics as auto_evaluate_v4.py
(silhouette Dice, LAB histogram, texture coherence, detail richness)
so that Best-of-N selection maximizes V4 evaluation scores.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Union


class QualityVerifier:
    """V4-aligned quality scoring for Best-of-N candidate selection.

    Scores a generated mesh against the input reference image using:
    - Silhouette Dice (A1-aligned: scale-invariant, best-view)
    - LAB color histogram (A2-aligned: best-view, L weighted low)
    - Geometric quality (B1-aligned: log-scale component penalty)
    - Texture coherence (C1-aligned: crack + gradient detection)
    - Color richness (C2-aligned: chroma coverage)
    - Detail richness (C3-aligned: Laplacian energy)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def render_views(self, mesh, num_views: int = 6, resolution: int = 512) -> Dict:
        """Render multi-view images of the mesh.

        Uses multi-pitch sampling (3 elevations) for better coverage of diverse
        input image viewing angles, matching the V4 evaluator.

        Args:
            mesh: MeshWithVoxel or renderable mesh object.
            num_views: Number of evenly-spaced azimuth views PER PITCH.
            resolution: Render resolution.

        Returns:
            Dict mapping render key (e.g. 'base_color') to list of [H,W,3] uint8 arrays.
        """
        from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
        from ..renderers import EnvMap

        pitches = [0.15, 0.30, 0.50]  # ~8.6°, ~17.2°, ~28.6° elevation
        all_yaw, all_pitch = [], []
        for p in pitches:
            yaws = [i * 2 * np.pi / num_views for i in range(num_views)]
            all_yaw.extend(yaws)
            all_pitch.extend([p] * num_views)

        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            all_yaw, all_pitch, 2, 40
        )
        envmap = EnvMap(torch.ones(16, 32, 3, device=self.device))
        return render_frames(
            mesh, extrinsics, intrinsics,
            {'resolution': resolution},
            verbose=False,
            envmap=envmap,
        )

    def compute_geometric_score(self, mesh) -> float:
        """Score geometric quality of the mesh.

        Checks vertex/face count, degenerate faces, bounding box aspect ratio,
        and connected components (fragmentation).

        Returns:
            Score in [0, 1] where 1 = excellent geometry.
        """
        score = 1.0

        n_verts = mesh.vertices.shape[0]
        if n_verts < 10000:
            score *= 0.5
        elif n_verts < 50000:
            score *= 0.8

        n_faces = mesh.faces.shape[0]
        if n_faces < 20000:
            score *= 0.5

        # Degenerate face ratio
        v0 = mesh.vertices[mesh.faces[:, 0]]
        v1 = mesh.vertices[mesh.faces[:, 1]]
        v2 = mesh.vertices[mesh.faces[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
        areas = torch.norm(cross, dim=-1) * 0.5
        degen_ratio = (areas < 1e-10).float().mean().item()
        score *= max(0.5, 1.0 - degen_ratio * 5)

        # Bounding box aspect ratio
        bbox_min = mesh.vertices.min(dim=0).values
        bbox_max = mesh.vertices.max(dim=0).values
        bbox_size = bbox_max - bbox_min
        if bbox_size.min() > 0:
            aspect = (bbox_size.max() / bbox_size.min()).item()
            if aspect > 10:
                score *= 0.7

        # Connected components — TRELLIS.2 FDG meshing produces 1000s of tiny
        # fragments; this is normal. Use log-scale penalty matching V4 B1.
        # Only penalize if face count is low (decimation didn't remove fragments).
        try:
            if n_faces < 500000:  # Only check if mesh is small enough to matter
                from scipy import sparse
                faces_np = mesh.faces.cpu().numpy()
                v0, v1, v2 = faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]
                rows = np.concatenate([v0, v1, v2, v1, v2, v0])
                cols = np.concatenate([v1, v2, v0, v0, v1, v2])
                graph = sparse.coo_matrix(
                    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                    shape=(n_verts, n_verts)
                ).tocsr()
                n_components = sparse.csgraph.connected_components(graph, directed=False)[0]
                if n_components > 100:
                    # Log-scale: 100→1.0, 1000→0.9, 10000→0.8, 100000→0.7
                    log_penalty = max(0.0, (np.log10(n_components) - 2) * 0.1)
                    score *= max(0.5, 1.0 - log_penalty)
        except Exception:
            pass

        return score

    def compute_color_richness(self, renders: Dict) -> float:
        """Score color richness across views.

        Penalizes meshes with large grey/monochrome regions.

        Returns:
            Score in [0, 1] where 1 = rich, colorful textures.
        """
        if 'base_color' not in renders:
            return 0.5

        scores = []
        for bc in renders['base_color']:
            bc_float = bc.astype(np.float32)
            max_c = np.maximum(np.maximum(bc_float[..., 0], bc_float[..., 1]), bc_float[..., 2])
            min_c = np.minimum(np.minimum(bc_float[..., 0], bc_float[..., 1]), bc_float[..., 2])
            chroma = max_c - min_c
            # Exclude background (black = 0)
            fg_mask = max_c > 10
            if fg_mask.sum() == 0:
                scores.append(0.0)
                continue
            colorful_ratio = float((chroma[fg_mask] > 15).mean())
            scores.append(colorful_ratio)

        return float(np.mean(scores)) if scores else 0.5

    def compute_color_match(self, renders: Dict, reference_image) -> float:
        """Compare color distributions in HSV space (A2-aligned metric).

        Uses HSV hue/saturation histograms with correlation, plus mean
        chrominance proximity. Matches V4's A2_color_dist methodology exactly.

        Returns:
            Score in [0, 1] where 1 = identical color distribution.
        """
        import cv2

        views = renders.get('base_color', [])
        alphas = renders.get('alpha', [])
        if not views:
            return 0.5

        # Prepare reference
        if isinstance(reference_image, Image.Image):
            ref = reference_image.convert('RGBA')
            ref_np = np.array(ref)
        elif isinstance(reference_image, np.ndarray):
            ref_np = reference_image
        else:
            return 0.5

        def _ensure_mask(m):
            """Ensure mask is 2D contiguous uint8 for cv2.calcHist."""
            if hasattr(m, 'numpy'):
                m = m.cpu().numpy()
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[:, :, 0]
            return np.ascontiguousarray(m.astype(np.uint8))

        def _score_view(render_rgb, alpha_mask):
            try:
                h, w = render_rgb.shape[:2]
                ref_r = ref_np
                if ref_np.shape[:2] != (h, w):
                    ref_r = cv2.resize(ref_np, (w, h))

                if alpha_mask is not None:
                    rend_mask = _ensure_mask(alpha_mask > 128)
                else:
                    gray = cv2.cvtColor(render_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
                    rend_mask = _ensure_mask(gray > 5)

                ref_mask = _ensure_mask(ref_r[:, :, 3] > 128) if ref_r.shape[2] >= 4 else np.ones((h, w), dtype=np.uint8)

                if rend_mask.sum() < 100 or ref_mask.sum() < 100:
                    return 0.0

                rend_mask_bool = rend_mask > 0
                ref_mask_bool = ref_mask > 0

                # Convert to HSV (H,S are lighting-invariant)
                rgb_3ch = render_rgb[:, :, :3] if render_rgb.shape[2] > 3 else render_rgb
                rend_hsv = cv2.cvtColor(np.ascontiguousarray(rgb_3ch), cv2.COLOR_RGB2HSV)
                ref_hsv = cv2.cvtColor(np.ascontiguousarray(ref_r[:, :, :3]), cv2.COLOR_RGB2HSV)

                # Hue (36 bins, [0,180]) + Saturation (32 bins, [0,256])
                hist_scores = []
                for c_idx, n_bins, val_range in [(0, 36, [0, 180]), (1, 32, [0, 256])]:
                    rend_hist = cv2.calcHist([rend_hsv], [c_idx], rend_mask, [n_bins], val_range)
                    ref_hist = cv2.calcHist([ref_hsv], [c_idx], ref_mask, [n_bins], val_range)
                    cv2.normalize(rend_hist, rend_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                    cv2.normalize(ref_hist, ref_hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
                    corr = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_CORREL)
                    hist_scores.append((corr + 1.0) / 2.0)

                hist_score = sum(hist_scores) / len(hist_scores)

                # Mean hue proximity (circular distance)
                rend_hue = float(rend_hsv[:, :, 0][rend_mask_bool].mean())
                ref_hue = float(ref_hsv[:, :, 0][ref_mask_bool].mean())
                hue_diff = abs(rend_hue - ref_hue)
                hue_diff = min(hue_diff, 180.0 - hue_diff)
                hue_score = max(0.0, 1.0 - hue_diff / 45.0)

                # Mean saturation proximity
                rend_sat = float(rend_hsv[:, :, 1][rend_mask_bool].mean())
                ref_sat = float(ref_hsv[:, :, 1][ref_mask_bool].mean())
                sat_score = max(0.0, 1.0 - abs(rend_sat - ref_sat) / 80.0)

                mean_score = 0.6 * hue_score + 0.4 * sat_score
                return min(1.0, 0.6 * hist_score + 0.4 * mean_score)
            except Exception:
                return 0.0

        # Best-matching view (matches V4 A2 behavior)
        best = 0.0
        for i, view in enumerate(views):
            am = alphas[i] if i < len(alphas) else None
            if am is not None and am.ndim == 3:
                am = am[:, :, 0]
            best = max(best, _score_view(view, am))

        return float(best)

    @torch.no_grad()
    def compute_depth_consistency(
        self,
        mesh,
        reference_image: Union[Image.Image, np.ndarray],
        depth_estimator=None,
    ) -> float:
        """Score depth consistency between rendered mesh and estimated depth.

        Uses DepthEstimator to get a monocular depth map of the reference image,
        then compares it with the rendered depth of the mesh.

        Returns:
            Score in [0, 1] where 1 = perfect depth match.
        """
        if depth_estimator is None:
            return 0.5

        try:
            from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
            from ..renderers import MeshRenderer

            # Get estimated depth
            if isinstance(reference_image, np.ndarray):
                ref_pil = Image.fromarray(reference_image)
            else:
                ref_pil = reference_image
            est_depth = depth_estimator.estimate(ref_pil)  # [H, W] normalized

            # Render mesh depth from front view
            extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(0.0, 0.25, 2.0, 40.0)
            renderer = MeshRenderer({'resolution': 256, 'near': 0.1, 'far': 100.0})
            result = renderer.render(mesh, extr, intr, return_types=['depth', 'mask'])

            if 'depth' not in result or 'mask' not in result:
                return 0.5

            rendered_depth = result['depth']  # [H, W]
            rendered_mask = result['mask']  # [H, W]

            # Normalize rendered depth
            fg = rendered_mask > 0.5
            if fg.sum() < 100:
                return 0.5
            d_min = rendered_depth[fg].min()
            d_max = rendered_depth[fg].max()
            if d_max - d_min < 1e-6:
                return 0.5
            norm_depth = (rendered_depth - d_min) / (d_max - d_min)

            # Resize estimated depth to match
            est_resized = torch.nn.functional.interpolate(
                est_depth.unsqueeze(0).unsqueeze(0),
                size=norm_depth.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Compare in foreground region
            diff = (norm_depth[fg] - est_resized[fg]).abs()
            consistency = 1.0 - diff.mean().item()
            return max(0.0, min(1.0, consistency))

        except Exception:
            return 0.5

    def compute_silhouette_dice(self, renders: Dict,
                               reference_image: Union[Image.Image, np.ndarray]) -> float:
        """Scale-invariant Dice coefficient between rendered and reference silhouettes.

        Mirrors V4 A1_silhouette: crops both masks to bounding box, resizes to
        256x256, then computes Dice = 2*|A∩B| / (|A|+|B|).
        Takes the best-matching view among all rendered views.

        Returns:
            Score in [0, 1] where 1 = perfect shape match.
        """
        import cv2

        # Extract reference alpha
        if isinstance(reference_image, Image.Image):
            ref_np = np.array(reference_image.convert('RGBA'))
        elif isinstance(reference_image, np.ndarray):
            ref_np = reference_image
        else:
            return 0.5

        if ref_np.shape[2] < 4:
            return 0.5

        ref_alpha = ref_np[:, :, 3]

        def _crop_to_bbox(mask_uint8, pad_frac=0.05):
            binary = (mask_uint8 > 128).astype(np.uint8)
            coords = np.argwhere(binary > 0)
            if len(coords) < 10:
                return np.zeros((256, 256), dtype=np.float32)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            side = max(y1 - y0 + 1, x1 - x0 + 1)
            pad = int(side * pad_frac)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half = side // 2 + pad
            H, W = mask_uint8.shape[:2]
            sy, ey = max(0, cy - half), min(H, cy + half)
            sx, ex = max(0, cx - half), min(W, cx + half)
            crop = mask_uint8[sy:ey, sx:ex]
            if crop.size == 0:
                return np.zeros((256, 256), dtype=np.float32)
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            return (resized > 128).astype(np.float32)

        ref_norm = _crop_to_bbox(ref_alpha)

        alphas = renders.get('alpha', [])
        if not alphas:
            return 0.5

        best_dice = 0.0
        for alpha_view in alphas:
            if alpha_view.ndim == 3:
                alpha_view = alpha_view[:, :, 0]
            rend_norm = _crop_to_bbox(alpha_view)
            intersection = (rend_norm * ref_norm).sum()
            total = rend_norm.sum() + ref_norm.sum()
            dice = 2 * intersection / max(total, 1)
            best_dice = max(best_dice, float(dice))

        return best_dice

    def compute_texture_coherence(self, renders: Dict) -> float:
        """Simplified V4 C1 texture coherence check.

        Detects morphological cracks and harsh gradient discontinuities
        in the base_color renders. Returns worst-view score.

        Returns:
            Score in [0, 1] where 1 = perfectly coherent texture.
        """
        import cv2

        if 'base_color' not in renders:
            return 0.5

        scores = []
        for bc in renders['base_color']:
            alpha = renders.get('alpha', [None])[0]
            rgb_smooth = cv2.GaussianBlur(bc, (0, 0), 1.0)
            gray = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)

            # Use simple foreground mask from brightness
            fg = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY) > 5
            kernel = np.ones((8, 8), np.uint8)
            interior = cv2.erode(fg.astype(np.uint8), kernel) > 0
            if interior.sum() < 200:
                scores.append(0.5)
                continue

            s = 1.0
            # Morphological crack detection
            gray_int = gray.copy()
            gray_int[~interior] = 128
            closed = cv2.morphologyEx(gray_int, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            crack_diff = closed.astype(np.float32) - gray_int.astype(np.float32)
            crack_ratio = ((crack_diff > 20) & interior).sum() / max(interior.sum(), 1)
            if crack_ratio > 0.003:
                s -= min(0.35, (crack_ratio - 0.003) * 6.0)

            # Color gradient harshness
            rgb_f = rgb_smooth.astype(np.float32)
            grad_mag = np.zeros(gray.shape, dtype=np.float32)
            for c in range(3):
                gx = cv2.Sobel(rgb_f[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(rgb_f[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
                grad_mag += np.sqrt(gx**2 + gy**2)
            grad_mag /= 3.0
            p95 = np.percentile(grad_mag[interior], 95)
            threshold = max(40.0, p95 * 0.85)
            harsh = (grad_mag[interior] > threshold).sum() / max(interior.sum(), 1)
            if harsh > 0.35:
                s -= min(0.15, (harsh - 0.35) * 1.0)

            scores.append(max(0.0, s))

        return float(np.mean(scores)) if scores else 0.5

    def compute_detail_richness(self, renders: Dict) -> float:
        """Simplified V4 C3 detail richness check.

        Measures Laplacian energy of base_color as a proxy for texture detail.

        Returns:
            Score in [0, 1] where 1 = rich texture detail.
        """
        import cv2

        if 'base_color' not in renders:
            return 0.5

        energies = []
        for bc in renders['base_color']:
            gray = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
            fg = gray > 5
            kernel = np.ones((8, 8), np.uint8)
            interior = cv2.erode(fg.astype(np.uint8), kernel) > 0
            if interior.sum() < 200:
                continue
            lap = cv2.Laplacian(gray.astype(np.float64) / 255.0, cv2.CV_64F)
            energies.append(np.abs(lap[interior]).mean() * 100)

        if not energies:
            return 0.5

        lap_energy = float(np.mean(energies))
        # Same curve as V4 C3 but mapped to [0, 1]
        if lap_energy < 2:
            return 0.2
        elif lap_energy < 5:
            return 0.2 + (lap_energy - 2) * 0.15
        elif lap_energy < 15:
            return 0.65 + (lap_energy - 5) * 0.035
        else:
            return 1.0

    @torch.no_grad()
    def score(
        self,
        mesh,
        reference_image: Union[Image.Image, np.ndarray],
        num_views: int = 6,
        render_resolution: int = 512,
        use_dreamsim: bool = False,
        depth_estimator=None,
    ) -> Dict[str, float]:
        """Score a generated mesh against the reference image.

        V4-aligned scoring: uses the same metrics as auto_evaluate_v4.py
        so Best-of-N selection picks candidates that maximize V4 score.

        Args:
            mesh: MeshWithVoxel or similar renderable mesh.
            reference_image: Input image (PIL RGBA or numpy array).
            num_views: Number of views to render for scoring.
            render_resolution: Resolution for renders.
            use_dreamsim: Whether to use DreamSim (unused, kept for API compat).
            depth_estimator: Optional DepthEstimator (unused, kept for API compat).

        Returns:
            Dict with 'total' score (0-1) and individual component scores.
        """
        # Create a shallow copy for scoring to avoid mutating the original mesh.
        from ..representations.mesh import MeshWithVoxel
        score_mesh = MeshWithVoxel(
            vertices=mesh.vertices.clone(),
            faces=mesh.faces.clone(),
            origin=mesh.origin.tolist() if hasattr(mesh.origin, 'tolist') else list(mesh.origin),
            voxel_size=mesh.voxel_size,
            coords=mesh.coords,
            attrs=mesh.attrs,
            voxel_shape=mesh.voxel_shape,
            layout=mesh.layout,
        )
        # Quick simplify to 500K for fast rendering during candidate scoring
        score_mesh.simplify(500000)

        renders = self.render_views(score_mesh, num_views, render_resolution)

        # V4-aligned scoring components:
        # A1: Silhouette Dice (scale-invariant, best-view)
        silhouette = self.compute_silhouette_dice(renders, reference_image)

        # A2: Color distribution match (LAB histogram)
        color_match = self.compute_color_match(renders, reference_image)

        # B1: Geometric quality (connected components, degenerate faces)
        geo_score = self.compute_geometric_score(score_mesh)

        # C1: Texture coherence (crack + gradient detection)
        tex_coherence = self.compute_texture_coherence(renders)

        # C2: Color richness (chroma)
        color_richness = self.compute_color_richness(renders)

        # C3: Detail richness (Laplacian energy)
        detail = self.compute_detail_richness(renders)

        # Weighted total — mirrors V4 weight distribution:
        # A1(15) + A2(10) + B1(10) + C1(15) + C2(10) + C3(10) = 70 of 100
        # (B2, D1, E1 excluded: low variance between seeds)
        total = (0.22 * silhouette       # A1: 15/70
                 + 0.14 * color_match     # A2: 10/70
                 + 0.14 * geo_score       # B1: 10/70
                 + 0.22 * tex_coherence   # C1: 15/70
                 + 0.14 * color_richness  # C2: 10/70
                 + 0.14 * detail)         # C3: 10/70

        result = {
            'total': total,
            'silhouette_dice': silhouette,
            'color_match': color_match,
            'geometric': geo_score,
            'tex_coherence': tex_coherence,
            'color_richness': color_richness,
            'detail_richness': detail,
        }

        return result

    def cleanup(self):
        """Release GPU memory."""
        torch.cuda.empty_cache()
