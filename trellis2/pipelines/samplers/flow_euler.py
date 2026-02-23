from typing import *
import math
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


def build_timestep_schedule(steps, schedule='uniform', rescale_t=1.0, sigma_min=1e-5, **kwargs):
    """Build a timestep schedule from t=1 (noise) to t=0 (data).

    Args:
        steps: Number of ODE steps.
        schedule: 'uniform' (default), 'edm', 'logsnr', or 'quadratic'.
        rescale_t: Rescaling factor (only used with 'uniform' schedule).
        sigma_min: Minimum noise scale for flow matching.
        **kwargs: 'rho' for EDM schedule (default 7.0),
                  'power' for quadratic schedule exponent (default 2.0).

    Returns:
        List of float timestep values (length steps+1) from ~1 to ~0.
    """
    if schedule == 'uniform':
        t_seq = np.linspace(1, 0, steps + 1)
        if rescale_t != 1.0:
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

    elif schedule == 'edm':
        # Karras et al. (2022): σ_i = (σ_max^(1/ρ) + i/N*(σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        rho = kwargs.get('rho', 7.0)
        sigma_max = 1.0
        i_frac = np.linspace(0, 1, steps + 1)
        sigmas = (sigma_max ** (1 / rho) + i_frac * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_seq = np.clip((sigmas - sigma_min) / (1 - sigma_min), 0, 1)

    elif schedule == 'logsnr':
        # Uniform spacing in log signal-to-noise ratio
        eps = 1e-4
        t_hi, t_lo = 1 - eps, eps
        sigma_hi = sigma_min + (1 - sigma_min) * t_hi
        sigma_lo = sigma_min + (1 - sigma_min) * t_lo
        logsnr_hi = 2 * np.log((1 - t_hi) / sigma_hi)  # low SNR (noisy)
        logsnr_lo = 2 * np.log((1 - t_lo) / sigma_lo)   # high SNR (clean)
        logsnr_seq = np.linspace(logsnr_hi, logsnr_lo, steps + 1)
        r = np.exp(logsnr_seq / 2)
        t_seq = (1 - r * sigma_min) / (1 + r * (1 - sigma_min))
        t_seq = np.clip(t_seq, 0, 1)

    elif schedule == 'quadratic':
        # t_i = (i/N)^p — more steps near t=0 (fine detail)
        # power=2.0 is classic quadratic; 1.5 is gentler; 3.0 is more aggressive
        power = kwargs.get('power', 2.0)
        i_frac = np.linspace(1, 0, steps + 1)
        t_seq = i_frac ** power

    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'uniform', 'edm', 'logsnr', or 'quadratic'.")

    return t_seq.tolist()


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # CFG-Zero*: skip first K ODE steps (zero velocity)
        zero_init_steps = kwargs.pop('zero_init_steps', 0)
        # CFG-MP: manifold projection strength (0=off, recommended 0.1-0.3)
        cfg_mp_strength = kwargs.pop('cfg_mp_strength', 0.0)
        # Heun: use 2nd-order Heun method for the last N steps (0=off)
        heun_steps = kwargs.pop('heun_steps', 0)
        # AB2: Adams-Bashforth 2nd order multistep (free accuracy, no extra model calls)
        multistep = kwargs.pop('multistep', False)
        # Rectified-CFG++ (Saini et al., arXiv 2510.07631, NeurIPS 2025)
        # 3-NFE predictor-corrector: conditional predictor → evaluate cond+uncond
        # at predicted point → interpolative correction. Keeps trajectories on the
        # data manifold. When active, bypasses CFG mixin (Heun/AB2/CFG-MP disabled).
        # Backward compat: cfg_pp_steps > 0 also enables R-CFG++
        cfg_pp_steps = kwargs.pop('cfg_pp_steps', 0)
        rectified_cfgpp = kwargs.pop('rectified_cfgpp', False) or cfg_pp_steps > 0
        rcfgpp_lambda_max = kwargs.pop('rcfgpp_lambda_max', 4.5)
        rcfgpp_gamma = kwargs.pop('rcfgpp_gamma', 0.0)
        rcfgpp_sigma_noise = kwargs.pop('rcfgpp_sigma_noise', 0.005)
        # Timestep schedule: 'uniform' (default), 'edm', 'logsnr', 'quadratic'
        schedule = kwargs.pop('schedule', 'uniform')
        schedule_rho = kwargs.pop('schedule_rho', 7.0)
        schedule_power = kwargs.pop('schedule_power', 2.0)
        # Guidance anneal: reduce guidance near t=0 to preserve fine detail
        # guidance_anneal_min = minimum fraction of guidance at t=0 (0=off, 0.25=reduce to 25%)
        guidance_anneal_min = kwargs.pop('guidance_anneal_min', 0.0)
        guidance_anneal_start = kwargs.pop('guidance_anneal_start', 0.3)
        # Stochastic SDE: convert ODE to SDE with controlled noise injection
        # sde_alpha = 0 (off, pure ODE), 0.1-0.5 recommended for diversity/detail
        sde_alpha = kwargs.pop('sde_alpha', 0.0)
        sde_profile = kwargs.pop('sde_profile', 'zero_ends')  # 'zero_ends' or 'sqrt_t'
        # Guidance rescale anneal: reduce rescale near t=0 to preserve high-frequency CFG detail
        # rescale_anneal_min = minimum rescale at t=0 (0=off, 0.7=reduce to 70% near t=0)
        rescale_anneal_min = kwargs.pop('rescale_anneal_min', 0.0)
        rescale_anneal_start = kwargs.pop('rescale_anneal_start', 0.3)

        sample = noise
        t_seq = build_timestep_schedule(
            steps, schedule=schedule, rescale_t=rescale_t,
            sigma_min=self.sigma_min, rho=schedule_rho, power=schedule_power,
        )
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        prev_v = None  # for AB2 multistep
        prev_dt = None
        orig_guidance = kwargs.get('guidance_strength', None)
        orig_rescale = kwargs.get('guidance_rescale', None)
        for step_idx, (t, t_prev) in enumerate(tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)):
            # Convert numpy scalars to Python float to avoid numpy*SparseTensor errors
            # (numpy tries to convert SparseTensor to ndarray, causing ValueError)
            t, t_prev = float(t), float(t_prev)
            # Apply guidance anneal: linearly reduce guidance from full to min_frac as t → 0
            if guidance_anneal_min > 0 and orig_guidance is not None and t < guidance_anneal_start:
                frac = t / guidance_anneal_start  # 1.0 at start, 0.0 at t=0
                kwargs['guidance_strength'] = orig_guidance * (guidance_anneal_min + (1 - guidance_anneal_min) * frac)
            elif orig_guidance is not None and guidance_anneal_min > 0:
                kwargs['guidance_strength'] = orig_guidance
            # Apply guidance rescale anneal: reduce rescale near t=0
            # Lower rescale at tail preserves CFG's high-frequency amplification
            if rescale_anneal_min > 0 and orig_rescale is not None and t < rescale_anneal_start:
                frac = t / rescale_anneal_start
                kwargs['guidance_rescale'] = orig_rescale * (rescale_anneal_min + (1 - rescale_anneal_min) * frac)
            elif orig_rescale is not None and rescale_anneal_min > 0:
                kwargs['guidance_rescale'] = orig_rescale
            if step_idx < zero_init_steps:
                # CFG-Zero*: hold position for early steps
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(sample)
                continue
            dt = t - t_prev
            # Check if Rectified-CFG++ should be used for this step
            _cur_guidance = kwargs.get('guidance_strength', 1.0)
            use_rcfgpp = (rectified_cfgpp and _cur_guidance is not None
                          and _cur_guidance > 1.0 and step_idx >= zero_init_steps)
            if step_idx == 0 and rectified_cfgpp:
                print(f"  [R-CFG++ debug] rectified_cfgpp={rectified_cfgpp}, guidance={_cur_guidance}, "
                      f"use_rcfgpp={use_rcfgpp}, neg_cond={'present' if kwargs.get('neg_cond') is not None else 'MISSING'}", flush=True)

            if use_rcfgpp:
                # Rectified-CFG++: 3-NFE predictor-corrector (bypasses CFG mixin)
                neg_cond_rc = kwargs.get('neg_cond', None)

                # Alpha schedule: lambda_max * (1-t)^gamma
                alpha_t = rcfgpp_lambda_max
                if rcfgpp_gamma > 0:
                    alpha_t = rcfgpp_lambda_max * (1.0 - t) ** rcfgpp_gamma

                # Respect guidance interval
                gi = kwargs.get('guidance_interval', None)
                if gi is not None and (t < gi[0] or t > gi[1]):
                    alpha_t = 0.0

                # Phase 1: Conditional predictor (full Euler step)
                v_cond = FlowEulerSampler._inference_model(
                    self, model, sample, t, cond)
                x_pred = sample - dt * v_cond

                # Optional noise injection (disabled near t=0)
                if rcfgpp_sigma_noise > 0 and t_prev > 0.1:
                    if hasattr(x_pred, 'replace') and hasattr(x_pred, 'feats'):
                        noise_eps = x_pred.replace(
                            torch.randn_like(x_pred.feats))
                    else:
                        noise_eps = torch.randn_like(x_pred)
                    x_pred = x_pred + rcfgpp_sigma_noise * noise_eps

                # Phase 2: Evaluate cond + uncond at predicted point
                v_cond_pred = FlowEulerSampler._inference_model(
                    self, model, x_pred, t_prev, cond)
                if neg_cond_rc is not None and alpha_t > 0:
                    v_uncond_pred = FlowEulerSampler._inference_model(
                        self, model, x_pred, t_prev, neg_cond_rc)
                else:
                    v_uncond_pred = v_cond_pred

                # Phase 3: Corrector — Eq. 8: v_guided = v_cond + α(v_c_pred - v_u_pred)
                if hasattr(v_cond, 'feats'):
                    corr = alpha_t * (v_cond_pred.feats - v_uncond_pred.feats)
                    v_guided = v_cond.replace(v_cond.feats + corr)
                else:
                    v_guided = v_cond + alpha_t * (v_cond_pred - v_uncond_pred)

                pred_x_prev = sample - dt * v_guided
                pred_x_0, _ = self._v_to_xstart_eps(sample, t, v_guided)
                out = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
                self._last_cond_pred_v = v_cond

            else:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                # Heun correction: 2nd-order method for final steps
                use_heun = (heun_steps > 0 and step_idx >= steps - heun_steps
                            and t_prev > 0)
                if use_heun:
                    v1 = (sample - out.pred_x_prev) / dt
                    _, _, v2 = self._get_model_prediction(
                        model, out.pred_x_prev, t_prev, cond, **kwargs)
                    out.pred_x_prev = sample - dt / 2 * (v1 + v2)
                elif multistep and prev_v is not None and prev_dt is not None:
                    # AB2: variable step-size Adams-Bashforth 2nd order
                    v_curr = (sample - out.pred_x_prev) / dt
                    r = dt / (2 * prev_dt)
                    out.pred_x_prev = sample - dt * ((1 + r) * v_curr - r * prev_v)
                    prev_v = v_curr
                    prev_dt = dt
                elif multistep:
                    # First step: store velocity for AB2
                    prev_v = (sample - out.pred_x_prev) / dt
                    prev_dt = dt
                if cfg_mp_strength > 0 and t_prev > 0:
                    # CFG-MP: project toward data manifold using conditional pred_x_0
                    cond_pred_v = getattr(self, '_last_cond_pred_v', None)
                    if cond_pred_v is not None:
                        cond_x_0, _ = self._v_to_xstart_eps(sample, t, cond_pred_v)
                    else:
                        cond_x_0 = out.pred_x_0
                    sigma_t = self.sigma_min + (1 - self.sigma_min) * t
                    sigma_t_prev = self.sigma_min + (1 - self.sigma_min) * t_prev
                    ratio = sigma_t_prev / sigma_t
                    coeff_x0 = (1 - t_prev) - (1 - t) * ratio
                    x_manifold = ratio * sample + coeff_x0 * cond_x_0
                    out.pred_x_prev = out.pred_x_prev + cfg_mp_strength * (x_manifold - out.pred_x_prev)
            # Stochastic SDE noise injection: adds diversity and can improve detail
            # Only inject noise at intermediate steps (not the final step to t≈0)
            _sde_alpha = float(sde_alpha) if sde_alpha else 0.0
            _t_prev_f = float(t_prev)
            if _sde_alpha > 0 and _t_prev_f > 1e-3:
                try:
                    # Compute score function: ∇log p_t(x) ≈ -eps / σ(t)
                    _t = float(t)
                    _sigma_t = float(self.sigma_min + (1 - self.sigma_min) * _t)
                    pred_eps = self._xstart_to_eps(sample, _t, out.pred_x_0)
                    score = -pred_eps / _sigma_t
                    # Diffusion coefficient g̃(t)
                    if sde_profile == 'zero_ends':
                        g_tilde = float(_sde_alpha * math.sqrt(max(_t * (1 - _t), 0)))
                    else:  # sqrt_t
                        g_tilde = float(_sde_alpha * math.sqrt(max(_t, 0)))
                    _abs_dt = float(abs(dt))
                    # SDE: drift correction + noise injection
                    # Drift: -½g̃²·score·dt (Stochastic Interpolants, Albergo et al.)
                    _dt = float(dt)
                    out.pred_x_prev = out.pred_x_prev - 0.5 * g_tilde**2 * score * _dt
                    # Noise: g̃·√dt·z — handle SparseTensor (has .feats/.replace)
                    if hasattr(out.pred_x_prev, 'replace') and hasattr(out.pred_x_prev, 'feats'):
                        noise = out.pred_x_prev.replace(torch.randn_like(out.pred_x_prev.feats))
                    else:
                        noise = torch.randn_like(out.pred_x_prev)
                    out.pred_x_prev = out.pred_x_prev + g_tilde * math.sqrt(_abs_dt) * noise
                except (ValueError, TypeError) as sde_err:
                    import sys
                    print(f"[SDE WARNING] step={step_idx} t={t} t_prev={t_prev} "
                          f"sde_alpha={sde_alpha}({type(sde_alpha).__name__}) "
                          f"t_prev_type={type(t_prev).__name__}: {sde_err}",
                          file=sys.stderr, flush=True)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)
