"""
Visibility-weighted multi-view fusion for multi-view diffusion.

Breaks the mathematical equivalence between x_0 and velocity averaging
by introducing per-token, per-view weights based on camera visibility.
See Doc 14 R-P1-B for theoretical motivation.
"""
import math
import torch
from typing import List, Dict

# Standard view name -> camera parameter mapping for TRELLIS.2 Z-up canonical space.
# Camera convention: cam_pos = [sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)] * r
VIEW_CAMERA_PARAMS: Dict[str, dict] = {
    # Cardinal views (90° apart)
    'front':  {'yaw': 0.0, 'pitch': 0.0},
    'back':   {'yaw': math.pi, 'pitch': 0.0},
    'left':   {'yaw': math.pi / 2, 'pitch': 0.0},
    'right':  {'yaw': -math.pi / 2, 'pitch': 0.0},
    'top':    {'yaw': 0.0, 'pitch': math.pi / 2 * 0.95},
    'bottom': {'yaw': 0.0, 'pitch': -math.pi / 2 * 0.95},
    # Intermediate views (45° from cardinal)
    'front_left':  {'yaw': math.pi / 4, 'pitch': 0.0},
    'front_right': {'yaw': -math.pi / 4, 'pitch': 0.0},
    'back_left':   {'yaw': 3 * math.pi / 4, 'pitch': 0.0},
    'back_right':  {'yaw': -3 * math.pi / 4, 'pitch': 0.0},
    # 120° spacing views (for 3-view evenly spaced setups)
    'left_120':  {'yaw': 2 * math.pi / 3, 'pitch': 0.0},
    'right_120': {'yaw': -2 * math.pi / 3, 'pitch': 0.0},
}


def get_camera_params_from_views(view_names: List[str]) -> List[dict]:
    """Map view names to camera parameters. Unknown names default to front."""
    return [VIEW_CAMERA_PARAMS.get(name, VIEW_CAMERA_PARAMS['front']) for name in view_names]


def compute_visibility_weights(
    coords: torch.Tensor,
    camera_params: List[dict],
    grid_resolution: int = 64,
    canonical_range: float = 0.5,
    temperature: float = 0.3,
) -> torch.Tensor:
    """
    Compute per-view, per-token visibility weights using direction-based heuristic.

    Each voxel gets higher weight from cameras that are closer/more aligned to it,
    so front-facing voxels prefer the front camera, back-facing prefer back, etc.

    Args:
        coords: SparseTensor.coords [N, 4] (batch, x, y, z) integer coords.
        camera_params: K dicts with 'yaw' and 'pitch' (radians).
        grid_resolution: Voxel grid resolution (inferred from coords if 0).
        canonical_range: Half-extent of canonical space (0.5 for [-0.5, 0.5]^3).
        temperature: Softmax temperature. Higher = more uniform weights.

    Returns:
        weights: [K, N] tensor, sum over K dim = 1 for each token.
    """
    N = coords.shape[0]
    K = len(camera_params)
    device = coords.device

    if grid_resolution <= 0:
        grid_resolution = coords[:, 1:4].max().item() + 1

    # Convert discrete voxel coords to continuous canonical space [-0.5, 0.5]^3
    voxel_pos = (coords[:, 1:4].float() + 0.5) / grid_resolution - canonical_range  # [N, 3]

    # Build camera direction matrix [K, 3]
    cam_dirs = torch.zeros(K, 3, device=device, dtype=torch.float32)
    for i, cam in enumerate(camera_params):
        yaw = cam['yaw']
        pitch = cam['pitch']
        cam_dirs[i] = torch.tensor([
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ])

    # Dot product: [N, 3] @ [3, K] -> [N, K] -> transpose to [K, N]
    raw_weights = (voxel_pos @ cam_dirs.T).T / temperature

    # Softmax over views for each token: [K, N]
    weights = torch.softmax(raw_weights, dim=0)

    return weights


def compute_sharp_visibility_weights(
    coords: torch.Tensor,
    camera_params: List[dict],
    grid_resolution: int = 64,
    temperature: float = 0.1,
    power: float = 3.0,
) -> torch.Tensor:
    """
    Compute visibility weights with power sharpening for crisper view boundaries.

    Applies power sharpening after softmax to concentrate weight on the
    dominant view per token, reducing blending in transition zones.

    Args:
        coords: SparseTensor.coords [N, 4] (batch, x, y, z) integer coords.
        camera_params: K dicts with 'yaw' and 'pitch' (radians).
        grid_resolution: Voxel grid resolution.
        temperature: Softmax temperature (lower = sharper before power).
        power: Exponent for post-softmax sharpening (higher = sharper).

    Returns:
        weights: [K, N] tensor, sum over K dim = 1 for each token.
    """
    weights = compute_visibility_weights(
        coords, camera_params, grid_resolution, temperature=temperature,
    )
    weights = weights ** power
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    return weights
