"""
Advanced CFG (Classifier-Free Guidance) strategies for flow matching models.

Implements:
- Standard CFG (default)
- CFG-Zero* (Fan et al., arXiv 2503.18886): Adaptive s* projection + zero-init
- APG (Wizadwongsa & Chinchuthakun, ICLR 2025): Orthogonal decomposition
- FDG (Sadat et al., arXiv 2506.19713): Frequency-Decoupled Guidance

All modes are backward-compatible: cfg_mode='standard' reproduces original behavior.
"""

import torch
import torch.nn.functional as F
import math


def _make_gaussian_kernel_1d(sigma, device, dtype):
    """Create a 1D Gaussian kernel for separable convolution."""
    radius = int(math.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    x = torch.arange(kernel_size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_3d(tensor, sigma):
    """
    Apply separable 3D Gaussian blur to a dense [B, C, D, H, W] tensor.
    Uses depthwise separable convolution for efficiency.
    """
    if sigma <= 0:
        return tensor
    kernel = _make_gaussian_kernel_1d(sigma, tensor.device, tensor.dtype)
    ks = kernel.shape[0]
    pad = ks // 2
    B, C = tensor.shape[:2]

    # Reshape to [B*C, 1, D, H, W] for depthwise conv
    x = tensor.reshape(B * C, 1, *tensor.shape[2:])

    # Separable: blur along each spatial dimension sequentially
    # Dim 2 (depth)
    k = kernel.reshape(1, 1, ks, 1, 1)
    x = F.conv3d(x, k, padding=(pad, 0, 0))
    # Dim 3 (height)
    k = kernel.reshape(1, 1, 1, ks, 1)
    x = F.conv3d(x, k, padding=(0, pad, 0))
    # Dim 4 (width)
    k = kernel.reshape(1, 1, 1, 1, ks)
    x = F.conv3d(x, k, padding=(0, 0, pad))

    return x.reshape(B, C, *tensor.shape[2:])


def _sparse_gaussian_blur_3d(feats, coords, sigma):
    """
    Gaussian blur on sparse 3D features via scatter-gather on dense grid.

    Args:
        feats: [N, D] feature tensor
        coords: [N, 4] integer coordinates (batch_idx, x, y, z)
        sigma: Gaussian blur sigma in voxel units
    Returns:
        [N, D] blurred features at original sparse locations
    """
    if sigma <= 0:
        return feats

    N, D = feats.shape
    B = coords[:, 0].max().item() + 1
    # Infer grid resolution from max coordinate
    R = coords[:, 1:].max().item() + 1

    # Safety: if grid is too large (>128^3), downsample coordinates to cap memory
    # 128^3 * D_max(64) * 4 bytes * 2 (grid+occ) ≈ 1.1 GB — acceptable
    MAX_R = 128
    scale = 1
    if R > MAX_R:
        scale = R / MAX_R
        R = MAX_R

    # Scatter features to dense grid using advanced indexing
    grid = torch.zeros(B, D, R, R, R, device=feats.device, dtype=feats.dtype)
    occupancy = torch.zeros(B, 1, R, R, R, device=feats.device, dtype=feats.dtype)

    b_idx = coords[:, 0].long()
    if scale > 1:
        x_idx = (coords[:, 1].float() / scale).long().clamp(0, R - 1)
        y_idx = (coords[:, 2].float() / scale).long().clamp(0, R - 1)
        z_idx = (coords[:, 3].float() / scale).long().clamp(0, R - 1)
    else:
        x_idx = coords[:, 1].long()
        y_idx = coords[:, 2].long()
        z_idx = coords[:, 3].long()

    # Scatter: grid[b, :, x, y, z] = feats[i]
    grid[b_idx, :, x_idx, y_idx, z_idx] = feats
    occupancy[b_idx, 0, x_idx, y_idx, z_idx] = 1.0

    # Blur both the features and the occupancy mask
    blurred_grid = _gaussian_blur_3d(grid, sigma)
    blurred_occ = _gaussian_blur_3d(occupancy, sigma)

    # Normalize by blurred occupancy to get proper weighted average
    blurred_grid = blurred_grid / blurred_occ.clamp(min=1e-8)

    # Gather blurred features at original sparse locations
    blurred_feats = blurred_grid[b_idx, :, x_idx, y_idx, z_idx]

    return blurred_feats


def compute_cfg_prediction(pred_pos, pred_neg, guidance_strength,
                           cfg_mode='standard', apg_alpha=0.3,
                           fdg_sigma=1.0, fdg_lambda_low=0.6, fdg_lambda_high=1.3):
    """
    Compute classifier-free guided prediction using the specified mode.

    Args:
        pred_pos: Conditional model prediction (dense Tensor or SparseTensor).
        pred_neg: Unconditional model prediction (same type as pred_pos).
        guidance_strength: CFG scale (w). Standard: pred = w * pos + (1-w) * neg.
        cfg_mode: One of 'standard', 'cfg_zero_star', 'apg', 'fdg'.
        apg_alpha: Parallel component damping for APG (0=full suppression, 1=standard CFG).
        fdg_sigma: Gaussian blur sigma for FDG frequency decomposition.
        fdg_lambda_low: Low-frequency guidance weight for FDG.
        fdg_lambda_high: High-frequency guidance weight for FDG.

    Returns:
        Guided prediction (same type as inputs).
    """
    if cfg_mode == 'cfg_zero_star':
        # CFG-Zero*: compute optimal projection scalar s*
        #   s* = (v_cond^T v_uncond) / ||v_uncond||^2
        #   v_guided = (1-w) * s* * v_uncond + w * v_cond
        is_sparse = hasattr(pred_pos, 'feats')
        if is_sparse:
            pos_flat = pred_pos.feats.reshape(-1)
            neg_flat = pred_neg.feats.reshape(-1)
        else:
            pos_flat = pred_pos.reshape(-1)
            neg_flat = pred_neg.reshape(-1)

        s_star = torch.dot(pos_flat, neg_flat) / (torch.dot(neg_flat, neg_flat) + 1e-8)
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * (s_star * pred_neg)

    elif cfg_mode == 'apg':
        # APG: Adaptive Projected Guidance
        #   delta = standard_cfg_pred - pred_pos
        #   delta_parallel = proj(delta, pred_pos)  -- causes oversaturation
        #   delta_orthogonal = delta - delta_parallel  -- enhances quality
        #   result = pred_pos + alpha * delta_parallel + delta_orthogonal
        std_pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
        delta = std_pred - pred_pos

        is_sparse = hasattr(delta, 'feats')
        if is_sparse:
            d = delta.feats          # (N_tokens, D)
            p = pred_pos.feats       # (N_tokens, D)
            dot = (d * p).sum(dim=-1, keepdim=True)
            norm_sq = (p * p).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            d_parallel = (dot / norm_sq) * p
            d_orthogonal = d - d_parallel
            guided_feats = p + apg_alpha * d_parallel + d_orthogonal
            pred = pred_pos.replace(guided_feats)
        else:
            d = delta
            p = pred_pos
            # Per-sample projection (collapse all dims except batch)
            dims = list(range(1, d.ndim))
            dot = (d * p).sum(dim=dims, keepdim=True)
            norm_sq = (p * p).sum(dim=dims, keepdim=True).clamp(min=1e-8)
            d_parallel = (dot / norm_sq) * p
            d_orthogonal = d - d_parallel
            pred = p + apg_alpha * d_parallel + d_orthogonal

    elif cfg_mode == 'fdg':
        # FDG: Frequency-Decoupled Guidance (Sadat et al., arXiv 2506.19713)
        #   delta = pred_pos - pred_neg (the CFG direction)
        #   delta_low = GaussianBlur(delta, sigma)
        #   delta_high = delta - delta_low
        #   pred = pred_neg + w * (lambda_low * delta_low + lambda_high * delta_high)
        is_sparse = hasattr(pred_pos, 'feats')
        if is_sparse:
            delta_feats = pred_pos.feats - pred_neg.feats
            delta_low = _sparse_gaussian_blur_3d(
                delta_feats, pred_pos.coords, fdg_sigma
            )
            delta_high = delta_feats - delta_low
            guided_feats = pred_neg.feats + guidance_strength * (
                fdg_lambda_low * delta_low + fdg_lambda_high * delta_high
            )
            pred = pred_neg.replace(guided_feats)
        else:
            delta = pred_pos - pred_neg
            delta_low = _gaussian_blur_3d(delta, fdg_sigma)
            delta_high = delta - delta_low
            pred = pred_neg + guidance_strength * (
                fdg_lambda_low * delta_low + fdg_lambda_high * delta_high
            )

    else:
        # Standard CFG
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

    return pred
