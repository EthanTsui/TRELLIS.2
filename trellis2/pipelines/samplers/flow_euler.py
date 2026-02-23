from typing import *
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
        for step_idx, (t, t_prev) in enumerate(tqdm(t_pairs, desc=tqdm_desc, disable=not verbose)):
            # Apply guidance anneal: linearly reduce guidance from full to min_frac as t → 0
            if guidance_anneal_min > 0 and orig_guidance is not None and t < guidance_anneal_start:
                frac = t / guidance_anneal_start  # 1.0 at start, 0.0 at t=0
                kwargs['guidance_strength'] = orig_guidance * (guidance_anneal_min + (1 - guidance_anneal_min) * frac)
            elif orig_guidance is not None and guidance_anneal_min > 0:
                kwargs['guidance_strength'] = orig_guidance
            if step_idx < zero_init_steps:
                # CFG-Zero*: hold position for early steps
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(sample)
                continue
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            dt = t - t_prev
            # Heun correction: 2nd-order method for final steps where detail forms
            use_heun = heun_steps > 0 and step_idx >= steps - heun_steps and t_prev > 0
            if use_heun:
                v1 = (sample - out.pred_x_prev) / dt
                _, _, v2 = self._get_model_prediction(model, out.pred_x_prev, t_prev, cond, **kwargs)
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
                # CFG-MP: project Euler step result back toward data manifold.
                # Use conditional-only pred_x_0 (not CFG-boosted) for true manifold target.
                cond_pred_v = getattr(self, '_last_cond_pred_v', None)
                if cond_pred_v is not None:
                    cond_x_0, _ = self._v_to_xstart_eps(sample, t, cond_pred_v)
                else:
                    cond_x_0 = out.pred_x_0  # fallback to CFG pred_x_0
                sigma_t = self.sigma_min + (1 - self.sigma_min) * t
                sigma_t_prev = self.sigma_min + (1 - self.sigma_min) * t_prev
                ratio = sigma_t_prev / sigma_t
                coeff_x0 = (1 - t_prev) - (1 - t) * ratio
                x_manifold = ratio * sample + coeff_x0 * cond_x_0
                out.pred_x_prev = out.pred_x_prev + cfg_mp_strength * (x_manifold - out.pred_x_prev)
            # Stochastic SDE noise injection: adds diversity and can improve detail
            # Only inject noise at intermediate steps (not the final step to t≈0)
            if sde_alpha > 0 and t_prev > 1e-3:
                # Compute score function: ∇log p_t(x) ≈ -eps / σ(t)
                sigma_t = self.sigma_min + (1 - self.sigma_min) * t
                pred_eps = self._xstart_to_eps(sample, t, out.pred_x_0)
                score = -pred_eps / sigma_t
                # Diffusion coefficient g̃(t)
                if sde_profile == 'zero_ends':
                    g_tilde = sde_alpha * np.sqrt(max(t * (1 - t), 0))
                else:  # sqrt_t
                    g_tilde = sde_alpha * np.sqrt(max(t, 0))
                abs_dt = abs(dt)
                # SDE: drift correction + noise injection
                # Drift: -½g̃²·score·dt (Stochastic Interpolants, Albergo et al.)
                out.pred_x_prev = out.pred_x_prev - 0.5 * g_tilde**2 * score * dt
                # Noise: g̃·√dt·z — handle SparseTensor (has .feats/.replace)
                if hasattr(out.pred_x_prev, 'replace') and hasattr(out.pred_x_prev, 'feats'):
                    noise = out.pred_x_prev.replace(torch.randn_like(out.pred_x_prev.feats))
                else:
                    noise = torch.randn_like(out.pred_x_prev)
                out.pred_x_prev = out.pred_x_prev + g_tilde * np.sqrt(abs_dt) * noise
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
