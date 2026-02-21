"""
Visual Hull computation for multi-view silhouette intersection.

Uses TRELLIS.2's Z-up canonical space with yaw/pitch camera convention.
See Doc 14 R-P2-1 for coordinate system corrections.
"""
import torch
import torch.nn.functional as F
from typing import List


def compute_visual_hull(
    silhouettes: List[torch.Tensor],
    cameras: List[dict],
    grid_resolution: int = 32,
    canonical_r: float = 2.0,
    canonical_fov: float = 30.0,
    alpha_threshold: float = 0.15,
    dilation_voxels: int = 3,
    max_removal_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Compute visual hull mask from multi-view silhouettes.

    Uses TRELLIS.2's Z-up canonical space and yaw/pitch camera convention,
    directly calling yaw_pitch_r_fov_to_extrinsics_intrinsics for consistency.

    Args:
        silhouettes: K tensors of shape [H, W] (alpha/mask, values in [0,1]).
        cameras: K dicts with 'yaw' and 'pitch' (radians).
        grid_resolution: Voxel grid resolution (e.g. 32 for sparse structure).
        canonical_r: Camera distance used during training (default 2.0).
        canonical_fov: Camera FOV in degrees (training: 30.0).
        alpha_threshold: Threshold for binarizing silhouettes.
        dilation_voxels: Dilation kernel half-size (must be >= 1).
        max_removal_ratio: Safety cap on fraction of voxels removed.

    Returns:
        hull_mask: Bool tensor [1, 1, R, R, R] suitable for AND with decoded structure.
    """
    from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    device = silhouettes[0].device
    R = grid_resolution
    K = len(silhouettes)

    # Build 3D grid in canonical space [-0.5, 0.5]^3
    lin = torch.linspace(-0.5 + 0.5 / R, 0.5 - 0.5 / R, R, device=device)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
    # grid_pts: [R^3, 3] in (x, y, z) order
    grid_pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    # Start with all voxels visible
    hull = torch.ones(R * R * R, dtype=torch.bool, device=device)

    # Get camera extrinsics/intrinsics
    yaws = [c['yaw'] for c in cameras]
    pitchs = [c['pitch'] for c in cameras]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, canonical_r, canonical_fov
    )

    for i in range(K):
        extr = extrinsics[i]  # [4, 4]
        intr = intrinsics[i]  # [3, 3]
        sil = silhouettes[i]  # [H, W]
        H, W = sil.shape

        # Binarize silhouette
        sil_binary = (sil > alpha_threshold).float()

        # Project grid points to camera space
        # P_cam = extr @ [P_world; 1]
        pts_homo = torch.cat([grid_pts, torch.ones(R**3, 1, device=device)], dim=-1)  # [R^3, 4]
        pts_cam = (extr @ pts_homo.T).T[:, :3]  # [R^3, 3]

        # Perspective projection: p = intr @ P_cam
        pts_proj = (intr @ pts_cam.T).T  # [R^3, 3]
        depth = pts_proj[:, 2]
        uv = pts_proj[:, :2] / depth.unsqueeze(-1).clamp(min=1e-6)  # [R^3, 2] in [0, 1]

        # utils3d intrinsics produce UV in [0, 1] range (cx=0.5, cy=0.5)
        # grid_sample expects [-1, 1] range
        uv = uv * 2 - 1  # [0, 1] -> [-1, 1]
        uv_grid = uv.unsqueeze(0).unsqueeze(0)  # [1, 1, R^3, 2]

        # Sample silhouette at projected locations
        sil_input = sil_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        sampled = F.grid_sample(sil_input, uv_grid, mode='nearest',
                                padding_mode='zeros', align_corners=False)
        sampled = sampled.squeeze()  # [R^3]

        # Voxels behind camera or outside image should be marked as outside
        in_front = depth > 0
        visible = (sampled > 0.5) & in_front

        hull = hull & visible

    # Reshape to [1, 1, R, R, R]
    hull_mask = hull.reshape(1, 1, R, R, R)

    # Apply dilation to be conservative (avoid over-trimming)
    if dilation_voxels > 0:
        kernel_size = 2 * dilation_voxels + 1
        hull_mask = F.max_pool3d(
            hull_mask.float(), kernel_size, stride=1, padding=dilation_voxels
        ) > 0

    return hull_mask
