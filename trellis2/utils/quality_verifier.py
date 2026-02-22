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
    """Multi-view quality scoring for Best-of-N selection.

    Scores a generated mesh against the input reference image using:
    - LPIPS perceptual distance (front view vs reference)
    - DreamSim perceptual distance (optional, if installed)
    - Geometric quality heuristics (face count, degeneracy, aspect ratio)
    - Color richness across multiple views (penalizes grey/monochrome)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._lpips_fn = None
        self._dreamsim_model = None
        self._dreamsim_preprocess = None

    @torch.no_grad()
    def _get_lpips(self):
        """Lazy-load LPIPS model."""
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net='alex').to(self.device).eval()
        return self._lpips_fn

    @torch.no_grad()
    def _get_dreamsim(self):
        """Lazy-load DreamSim model. Returns (None, None) if not installed."""
        if self._dreamsim_model is None:
            try:
                from dreamsim import dreamsim
                self._dreamsim_model, self._dreamsim_preprocess = dreamsim(
                    pretrained=True, device=self.device
                )
                self._dreamsim_model.eval()
            except ImportError:
                return None, None
        return self._dreamsim_model, self._dreamsim_preprocess

    def render_views(self, mesh, num_views: int = 6, resolution: int = 512) -> Dict:
        """Render multi-view images of the mesh.

        Args:
            mesh: MeshWithVoxel or renderable mesh object.
            num_views: Number of evenly-spaced azimuth views.
            resolution: Render resolution.

        Returns:
            Dict mapping render key (e.g. 'base_color') to list of [H,W,3] uint8 arrays.
        """
        from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, render_frames
        from ..renderers import EnvMap

        yaws = [i * 2 * np.pi / num_views for i in range(num_views)]
        pitchs = [0.25] * num_views  # slightly elevated
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, 2, 40
        )
        # PbrMeshRenderer.render() requires an envmap; use a plain white one
        envmap = EnvMap(torch.ones(16, 32, 3, device=self.device))
        return render_frames(
            mesh, extrinsics, intrinsics,
            {'resolution': resolution},
            verbose=False,
            envmap=envmap,
        )

    def _image_to_lpips_tensor(self, img, size: int = 224) -> torch.Tensor:
        """Convert image to LPIPS-compatible tensor: [1, 3, H, W] in [-1, 1]."""
        if isinstance(img, Image.Image):
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            img = img.convert('RGB').resize((size, size), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((size, size), Image.LANCZOS)
            arr = np.array(pil).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        return tensor.to(self.device)

    @torch.no_grad()
    def compute_lpips(self, render, reference) -> float:
        """Compute LPIPS perceptual similarity (higher = more similar).

        Args:
            render: Rendered view (numpy uint8 or PIL Image).
            reference: Reference input image (PIL RGBA or numpy).

        Returns:
            Score in [0, 1] where 1 = identical.
        """
        lpips_fn = self._get_lpips()
        render_t = self._image_to_lpips_tensor(render)
        ref_t = self._image_to_lpips_tensor(reference)
        dist = lpips_fn(render_t, ref_t).item()
        return max(0.0, 1.0 - dist)

    @torch.no_grad()
    def compute_dreamsim(self, render, reference) -> Optional[float]:
        """Compute DreamSim perceptual similarity (higher = more similar).

        Returns None if DreamSim is not installed.
        """
        model, preprocess = self._get_dreamsim()
        if model is None:
            return None

        if isinstance(render, np.ndarray):
            render = Image.fromarray(render)
        if isinstance(reference, np.ndarray):
            reference = Image.fromarray(np.clip(reference, 0, 255).astype(np.uint8))

        # Composite RGBA on black background
        if hasattr(render, 'mode') and render.mode == 'RGBA':
            bg = Image.new('RGB', render.size, (0, 0, 0))
            bg.paste(render, mask=render.split()[3])
            render = bg
        if hasattr(reference, 'mode') and reference.mode == 'RGBA':
            bg = Image.new('RGB', reference.size, (0, 0, 0))
            bg.paste(reference, mask=reference.split()[3])
            reference = bg

        render_t = preprocess(render.convert('RGB')).to(self.device)
        ref_t = preprocess(reference.convert('RGB')).to(self.device)
        dist = model(render_t, ref_t).item()
        return max(0.0, 1.0 - dist)

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

        # Connected components via vertex adjacency graph — fast O(V+F)
        try:
            from scipy import sparse
            faces_np = mesh.faces.cpu().numpy()
            # Build vertex adjacency graph from triangle edges
            v0, v1, v2 = faces_np[:, 0], faces_np[:, 1], faces_np[:, 2]
            rows = np.concatenate([v0, v1, v2, v1, v2, v0])
            cols = np.concatenate([v1, v2, v0, v0, v1, v2])
            graph = sparse.coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_verts, n_verts)
            ).tocsr()
            n_components = sparse.csgraph.connected_components(graph, directed=False)[0]
            if n_components > 1:
                # Harsh penalty: fragmented mesh is unusable
                # 2 components -> 0.3, 5+ -> 0.1
                score *= max(0.1, 0.5 / n_components)
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
        """Compare color distributions in LAB space (A2-aligned metric).

        Uses histogram correlation between rendered front view and reference
        image in LAB color space, matching V4's A2_color_dist methodology.

        Returns:
            Score in [0, 1] where 1 = identical color distribution.
        """
        import cv2

        if 'base_color' not in renders or len(renders['base_color']) == 0:
            return 0.5

        front_render = renders['base_color'][0]  # [H, W, 3] uint8 RGB
        alpha = renders.get('alpha', [None])[0]

        # Prepare reference
        if isinstance(reference_image, Image.Image):
            ref = reference_image.convert('RGBA')
            ref_np = np.array(ref)
        elif isinstance(reference_image, np.ndarray):
            ref_np = reference_image
        else:
            return 0.5

        h, w = front_render.shape[:2]
        if ref_np.shape[:2] != (h, w):
            ref_np = cv2.resize(ref_np, (w, h))

        # Masks
        if alpha is not None:
            rend_mask = (alpha > 128).astype(np.uint8)
        else:
            gray = cv2.cvtColor(front_render, cv2.COLOR_RGB2GRAY)
            rend_mask = (gray > 5).astype(np.uint8)

        if ref_np.shape[2] == 4:
            ref_mask = (ref_np[:, :, 3] > 128).astype(np.uint8)
        else:
            ref_mask = np.ones((h, w), dtype=np.uint8)

        if rend_mask.sum() < 100 or ref_mask.sum() < 100:
            return 0.5

        # Convert to LAB
        rend_lab = cv2.cvtColor(front_render, cv2.COLOR_RGB2LAB)
        ref_rgb = ref_np[:, :, :3]
        ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB)

        # Histogram correlation (A and B channels weighted higher)
        corr_scores = []
        for c in range(3):
            rend_hist = cv2.calcHist([rend_lab], [c], rend_mask, [32], [0, 256])
            ref_hist = cv2.calcHist([ref_lab], [c], ref_mask, [32], [0, 256])
            cv2.normalize(rend_hist, rend_hist)
            cv2.normalize(ref_hist, ref_hist)
            corr = cv2.compareHist(rend_hist, ref_hist, cv2.HISTCMP_CORREL)
            corr_scores.append(max(0.0, corr))

        # Weight: L=0.2, A=0.4, B=0.4 (lighting less important)
        weighted = 0.2 * corr_scores[0] + 0.4 * corr_scores[1] + 0.4 * corr_scores[2]
        return float(weighted)

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
        score_mesh.simplify(16777216)

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
        """Release model memory."""
        if self._lpips_fn is not None:
            del self._lpips_fn
            self._lpips_fn = None
        if self._dreamsim_model is not None:
            del self._dreamsim_model
            del self._dreamsim_preprocess
            self._dreamsim_model = None
            self._dreamsim_preprocess = None
        torch.cuda.empty_cache()
