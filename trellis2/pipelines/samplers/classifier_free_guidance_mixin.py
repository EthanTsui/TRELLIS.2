from typing import *
from .cfg_utils import compute_cfg_prediction


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength,
                         guidance_rescale=0.0, cfg_mode='standard', apg_alpha=0.3,
                         fdg_sigma=1.0, fdg_lambda_low=0.6, fdg_lambda_high=1.3,
                         **kwargs):
        if guidance_strength == 1:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            self._last_cond_pred_v = pred  # Store for CFG-MP
            return pred
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            self._last_cond_pred_v = pred_pos  # Store conditional pred for CFG-MP
            pred = compute_cfg_prediction(pred_pos, pred_neg, guidance_strength,
                                          cfg_mode=cfg_mode, apg_alpha=apg_alpha,
                                          fdg_sigma=fdg_sigma,
                                          fdg_lambda_low=fdg_lambda_low,
                                          fdg_lambda_high=fdg_lambda_high)

            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
