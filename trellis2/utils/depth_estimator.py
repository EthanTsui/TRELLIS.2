"""
Depth estimation using Depth Anything V2 for geometry guidance.

Provides monocular depth maps that can be used to:
1. Improve visual hull computation (depth-aware carving)
2. Score geometry quality in Best-of-N selection (depth consistency)
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2.

    Uses the HuggingFace transformers pipeline for depth estimation.
    The model is lazy-loaded on first use.

    Args:
        model_name: HuggingFace model ID for depth estimation.
        device: CUDA device string.
    """

    def __init__(
        self,
        model_name: str = 'depth-anything/Depth-Anything-V2-Large-hf',
        device: str = 'cuda',
    ):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        """Lazy-load the depth estimation pipeline."""
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                'depth-estimation',
                model=self.model_name,
                device=self.device,
            )
        return self._pipe

    @torch.no_grad()
    def estimate(self, image: Image.Image) -> torch.Tensor:
        """Estimate depth map from a single image.

        Args:
            image: PIL Image (RGB or RGBA).

        Returns:
            Depth tensor [H, W] on CUDA, values normalized to [0, 1]
            where 0 = closest, 1 = farthest.
        """
        if image.mode == 'RGBA':
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        image = image.convert('RGB')

        pipe = self._get_pipe()
        result = pipe(image)
        depth = result['depth']

        if isinstance(depth, Image.Image):
            depth_np = np.array(depth).astype(np.float32)
        elif isinstance(depth, np.ndarray):
            depth_np = depth.astype(np.float32)
        else:
            depth_np = depth.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return torch.from_numpy(depth_np).float().to(self.device)

    @torch.no_grad()
    def estimate_pointcloud(
        self,
        image: Image.Image,
        fov: float = 40.0,
    ) -> torch.Tensor:
        """Estimate a pseudo-point cloud from depth map.

        Unprojects depth pixels to 3D using estimated camera intrinsics.

        Args:
            image: PIL Image.
            fov: Assumed camera field of view in degrees.

        Returns:
            Point cloud tensor [N, 3] where N is the number of foreground pixels.
        """
        depth = self.estimate(image)
        H, W = depth.shape

        # Build pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # Normalize to [-1, 1]
        cx, cy = W / 2.0, H / 2.0
        fov_rad = torch.deg2rad(torch.tensor(fov, device=self.device))
        f = 0.5 / torch.tan(fov_rad / 2)

        x_norm = (x - cx) / (W * f)
        y_norm = (y - cy) / (H * f)

        # Unproject
        z = depth
        pts_x = x_norm * z
        pts_y = y_norm * z
        pts = torch.stack([pts_x, pts_y, z], dim=-1)  # [H, W, 3]

        # Filter: only foreground (where depth is meaningful)
        # Use alpha mask if available, otherwise use depth variation
        fg_mask = depth > 0.01
        points = pts[fg_mask]

        return points

    def cleanup(self):
        """Release model memory."""
        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()
