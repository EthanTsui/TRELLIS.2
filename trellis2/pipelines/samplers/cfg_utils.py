"""
Advanced CFG (Classifier-Free Guidance) strategies for flow matching models.

Implements:
- Standard CFG (default)
- CFG-Zero* (Fan et al., arXiv 2503.18886): Adaptive s* projection + zero-init
- APG (Wizadwongsa & Chinchuthakun, ICLR 2025): Orthogonal decomposition

All modes are backward-compatible: cfg_mode='standard' reproduces original behavior.
"""

import torch


def compute_cfg_prediction(pred_pos, pred_neg, guidance_strength,
                           cfg_mode='standard', apg_alpha=0.3):
    """
    Compute classifier-free guided prediction using the specified mode.

    Args:
        pred_pos: Conditional model prediction (dense Tensor or SparseTensor).
        pred_neg: Unconditional model prediction (same type as pred_pos).
        guidance_strength: CFG scale (w). Standard: pred = w * pos + (1-w) * neg.
        cfg_mode: One of 'standard', 'cfg_zero_star', 'apg'.
        apg_alpha: Parallel component damping for APG (0=full suppression, 1=standard CFG).

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

    else:
        # Standard CFG
        pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

    return pred
