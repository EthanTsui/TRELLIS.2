from typing import *
import math


def _beta_weight(t_norm: float, a: float, b: float) -> float:
    """
    Compute normalized beta PDF weight at t_norm in [0, 1].

    Returns a value in [0, 1] where 1.0 is at the mode of the distribution.
    Used for smooth guidance scheduling (replaces binary on/off).

    Args:
        t_norm: Normalized timestep in [0, 1] within the guidance interval.
        a: Beta shape parameter alpha (>1 for unimodal).
        b: Beta shape parameter beta (>1 for unimodal).
    """
    if t_norm <= 0 or t_norm >= 1:
        return 0.0
    if a <= 1 or b <= 1:
        return 1.0  # degenerate: fall back to uniform

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_pdf = (a - 1) * math.log(t_norm) + (b - 1) * math.log(1 - t_norm) - log_beta

    # Normalize by the peak (mode) value so output is in [0, 1]
    mode = (a - 1) / (a + b - 2)
    log_max = (a - 1) * math.log(mode) + (b - 1) * math.log(1 - mode) - log_beta

    return math.exp(log_pdf - log_max)


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.

    Supports three scheduling modes:
    - 'binary' (default): guidance is fully on within interval, off outside.
    - 'beta': guidance strength is modulated by a beta distribution curve.
              guidance_beta_a/b control the shape (a=2,b=5 peaks early).
    - 'triangular': symmetric triangle peaking at interval midpoint (TV-CFG style).
              From "Stage-wise Dynamics of CFG" (arXiv:2509.22007).
    """

    def _inference_model(self, model, x_t, t, cond, guidance_strength, guidance_interval,
                         guidance_schedule='binary', guidance_beta_a=2.0, guidance_beta_b=5.0,
                         **kwargs):
        if guidance_interval[0] <= t <= guidance_interval[1]:
            if guidance_schedule == 'beta':
                # Smooth guidance modulation via beta distribution
                interval_range = guidance_interval[1] - guidance_interval[0]
                if interval_range > 0:
                    t_norm = (t - guidance_interval[0]) / interval_range
                    weight = _beta_weight(t_norm, guidance_beta_a, guidance_beta_b)
                    effective_strength = 1.0 + (guidance_strength - 1.0) * weight
                else:
                    effective_strength = guidance_strength
                return super()._inference_model(model, x_t, t, cond,
                                                guidance_strength=effective_strength, **kwargs)
            elif guidance_schedule == 'triangular':
                # TV-CFG: symmetric triangle peaking at interval midpoint
                interval_range = guidance_interval[1] - guidance_interval[0]
                if interval_range > 0:
                    t_norm = (t - guidance_interval[0]) / interval_range
                    weight = 2.0 * t_norm if t_norm <= 0.5 else 2.0 * (1.0 - t_norm)
                    effective_strength = 1.0 + (guidance_strength - 1.0) * weight
                else:
                    effective_strength = guidance_strength
                return super()._inference_model(model, x_t, t, cond,
                                                guidance_strength=effective_strength, **kwargs)
            else:
                # Binary: full guidance within interval
                return super()._inference_model(model, x_t, t, cond,
                                                guidance_strength=guidance_strength, **kwargs)
        else:
            return super()._inference_model(model, x_t, t, cond,
                                            guidance_strength=1, **kwargs)
