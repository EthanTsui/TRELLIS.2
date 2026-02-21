"""
Quality verification for Best-of-N candidate selection.

Uses LPIPS perceptual distance, color richness analysis, and geometric
heuristics to score generated 3D meshes against input reference images.
DreamSim support is optional (lazy-loaded if available).
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

        Checks vertex/face count, degenerate faces, bounding box aspect ratio.

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

        Args:
            mesh: MeshWithVoxel or similar renderable mesh.
            reference_image: Input image (PIL RGBA or numpy array).
            num_views: Number of views to render for scoring.
            render_resolution: Resolution for renders.
            use_dreamsim: Whether to use DreamSim (requires dreamsim package).
            depth_estimator: Optional DepthEstimator for depth consistency scoring.

        Returns:
            Dict with 'total' score (0-1) and individual component scores.
        """
        # Create a shallow copy for scoring to avoid mutating the original mesh.
        # Mesh.simplify() replaces .vertices and .faces in-place.
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

        # 1. Perceptual similarity (front view vs reference)
        front_render = renders['base_color'][0]
        lpips_score = self.compute_lpips(front_render, reference_image)

        # 2. Optional DreamSim
        dreamsim_score = None
        if use_dreamsim:
            dreamsim_score = self.compute_dreamsim(front_render, reference_image)

        # 3. Geometric quality
        geo_score = self.compute_geometric_score(score_mesh)

        # 4. Color richness
        color_score = self.compute_color_richness(renders)

        # 5. Optional depth consistency
        depth_score = None
        if depth_estimator is not None:
            depth_score = self.compute_depth_consistency(mesh, reference_image, depth_estimator)

        # Weighted total
        if dreamsim_score is not None:
            perceptual = 0.5 * lpips_score + 0.5 * dreamsim_score
        else:
            perceptual = lpips_score

        if depth_score is not None:
            total = 0.4 * perceptual + 0.15 * geo_score + 0.25 * color_score + 0.2 * depth_score
        else:
            total = 0.5 * perceptual + 0.2 * geo_score + 0.3 * color_score

        result = {
            'total': total,
            'lpips': lpips_score,
            'geometric': geo_score,
            'color_richness': color_score,
        }
        if dreamsim_score is not None:
            result['dreamsim'] = dreamsim_score
        if depth_score is not None:
            result['depth_consistency'] = depth_score

        return result

    def cleanup(self):
        """Release model memory."""
        del self._lpips_fn
        del self._dreamsim_model
        del self._dreamsim_preprocess
        self._lpips_fn = None
        self._dreamsim_model = None
        self._dreamsim_preprocess = None
        torch.cuda.empty_cache()
