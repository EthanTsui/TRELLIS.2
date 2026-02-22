from typing import *
from contextlib import contextmanager, nullcontext
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from .samplers.flow_euler import FlowEulerSampler
from .samplers.cfg_utils import compute_cfg_prediction
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output_np = np.array(output).astype(np.float32) / 255
        rgb = output_np[:, :, :3] * output_np[:, :, 3:4]
        rgba = np.concatenate([rgb, output_np[:, :, 3:4]], axis=2)
        output = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_cond_multiview(
        self,
        images: Dict[str, Image.Image],
        resolution: int,
    ) -> dict:
        """
        Get per-view conditioning from multiple views.

        Returns a dict with:
            'cond': stacked per-view features (K, N, D)
            'neg_cond': zeros tensor (1, N, D)
        """
        image_list = list(images.values())

        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        all_cond = self.image_cond_model(image_list)  # (K, N, D)
        if self.low_vram:
            self.image_cond_model.cpu()

        neg_cond = torch.zeros_like(all_cond[:1])  # (1, N, D)

        return {
            'cond': all_cond,
            'neg_cond': neg_cond,
        }

    @contextmanager
    def inject_sampler_multi_image(self, sampler, num_images, mode='multidiffusion',
                                    camera_params=None):
        """
        Context manager that monkey-patches a sampler's _inference_model to
        handle multi-view conditioning via prediction averaging (multidiffusion)
        or view cycling (stochastic).

        Ported from TRELLIS 1's inject_sampler_multi_image.

        Args:
            sampler: The sampler object to patch (e.g. self.shape_slat_sampler).
            num_images: Number of views (K).
            mode: 'multidiffusion' (average K predictions per step) or
                  'stochastic' (cycle through views, 1 prediction per step).
            camera_params: List of K dicts with 'yaw'/'pitch' for visibility weighting.
                          If provided, uses per-token weighted averaging instead of uniform.
        """
        if mode == 'multidiffusion':
            def _patched_inference_model(
                model, x_t, t, cond,
                neg_cond=None, guidance_strength=1.0,
                guidance_interval=(0.0, 1.0), guidance_rescale=0.0,
                cfg_mode='standard', apg_alpha=0.3,
                **kwargs
            ):
                # cond is (K, N, D) — run raw model once per view, then fuse
                preds = []
                for i in range(num_images):
                    pred_i = FlowEulerSampler._inference_model(
                        sampler, model, x_t, t, cond[i:i+1], **kwargs
                    )
                    preds.append(pred_i)

                # Visibility-weighted averaging for SparseTensors (breaks x_0/v equivalence)
                if camera_params is not None and hasattr(x_t, 'coords'):
                    from ..utils.visibility import compute_visibility_weights
                    grid_res = x_t.coords[:, 1:].max().item() + 1
                    weights = compute_visibility_weights(x_t.coords, camera_params, grid_res)
                    # Power sharpening: (0.50, 0.30, 0.20) → ~(0.85, 0.12, 0.03)
                    weights = weights ** 3
                    weights = weights / weights.sum(dim=0, keepdim=True)
                    weighted_feats = torch.zeros_like(preds[0].feats)
                    for i in range(num_images):
                        weighted_feats += weights[i].unsqueeze(-1) * preds[i].feats
                    pred = preds[0].replace(weighted_feats)

                    # Per-token variance rescaling: restore variance compressed by averaging
                    avg_std = sum(p.feats.std(dim=-1, keepdim=True) for p in preds) / num_images
                    actual_std = pred.feats.std(dim=-1, keepdim=True).clamp(min=1e-6)
                    feat_mean = pred.feats.mean(dim=-1, keepdim=True)
                    rescaled = feat_mean + (avg_std / actual_std) * (pred.feats - feat_mean)
                    pred = pred.replace(rescaled)
                else:
                    # Uniform averaging (dense tensors or no camera params)
                    pred = sum(preds) / num_images

                # Apply CFG within guidance interval
                if guidance_interval[0] <= t <= guidance_interval[1] and guidance_strength != 1:
                    if guidance_strength == 0:
                        return FlowEulerSampler._inference_model(
                            sampler, model, x_t, t, neg_cond, **kwargs
                        )
                    neg_pred = FlowEulerSampler._inference_model(
                        sampler, model, x_t, t, neg_cond, **kwargs
                    )
                    result = compute_cfg_prediction(pred, neg_pred, guidance_strength,
                                                    cfg_mode=cfg_mode, apg_alpha=apg_alpha)

                    if guidance_rescale > 0:
                        # Use average of per-view x_0 stds instead of std of averaged
                        # prediction, which is systematically lower due to variance
                        # compression from averaging (see Doc 14 R-P1-A)
                        stds_pos = []
                        for i in range(num_images):
                            x_0_i = sampler._pred_to_xstart(x_t, t, preds[i])
                            std_i = x_0_i.std(dim=list(range(1, x_0_i.ndim)), keepdim=True)
                            stds_pos.append(std_i)
                        std_pos = sum(stds_pos) / num_images

                        x_0_cfg = sampler._pred_to_xstart(x_t, t, result)
                        std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                        x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                        result = sampler._xstart_to_pred(x_t, t, x_0)

                    return result
                return pred

        elif mode == 'stochastic':
            step_counter = [0]

            def _patched_inference_model(model, x_t, t, cond, **kwargs):
                # cond is (K, N, D) — pick one view per step, cycling
                idx = step_counter[0] % num_images
                single_cond = cond[idx:idx+1]
                step_counter[0] += 1
                # Delegate to original class chain with single-view cond
                return type(sampler)._inference_model(
                    sampler, model, x_t, t, single_cond, **kwargs
                )
        else:
            raise ValueError(f"Unknown multi-image mode: {mode}")

        # Monkey-patch: instance attribute shadows the class method
        sampler._inference_model = _patched_inference_model
        try:
            yield
        finally:
            # Remove instance attribute to restore the class method
            if '_inference_model' in sampler.__dict__:
                del sampler._inference_model

    @contextmanager
    def inject_tapa_conditioning(self, sampler, concat_cond, single_cond,
                                  threshold=0.7, blend_width=0.2):
        """
        Timestep-Adaptive Partial Averaging (TAPA) with smooth transition.

        Switches conditioning based on timestep with optional smooth blending:
        - t > threshold + blend_width/2: uses concat_cond (all views via cross-attention)
        - t < threshold - blend_width/2: uses single_cond (front-view only)
        - In between: linearly blends predictions from both

        This preserves multi-view geometric consistency in early steps while
        allowing single-view fine detail in later steps. The smooth transition
        avoids hard artifacts at the switching boundary.

        Args:
            sampler: The sampler to monkey-patch.
            concat_cond: Dict with 'cond' and 'neg_cond' for multi-view concat.
            single_cond: Dict with 'cond' and 'neg_cond' for single view.
            threshold: Timestep center for switching (default 0.7).
            blend_width: Width of smooth transition zone (default 0.2).
                         0 = hard switch (original behavior).
        """
        concat_cond_tensor = concat_cond['cond']
        concat_neg_tensor = concat_cond['neg_cond']
        single_cond_tensor = single_cond['cond']
        single_neg_tensor = single_cond['neg_cond']

        t_high = threshold + blend_width / 2  # above this: pure concat
        t_low = threshold - blend_width / 2   # below this: pure single

        def _run_with_cfg(model, x_t, t, cond, neg, guidance_strength,
                          guidance_interval, guidance_rescale,
                          cfg_mode='standard', apg_alpha=0.3, **kwargs):
            """Run model prediction with CFG and optional rescaling."""
            pred = FlowEulerSampler._inference_model(
                sampler, model, x_t, t, cond, **kwargs
            )
            if guidance_interval[0] <= t <= guidance_interval[1] and guidance_strength != 1:
                if guidance_strength == 0:
                    return FlowEulerSampler._inference_model(
                        sampler, model, x_t, t, neg, **kwargs
                    )
                neg_pred = FlowEulerSampler._inference_model(
                    sampler, model, x_t, t, neg, **kwargs
                )
                result = compute_cfg_prediction(pred, neg_pred, guidance_strength,
                                                cfg_mode=cfg_mode, apg_alpha=apg_alpha)
                if guidance_rescale > 0:
                    x_0_pos = sampler._pred_to_xstart(x_t, t, pred)
                    x_0_cfg = sampler._pred_to_xstart(x_t, t, result)
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    result = sampler._xstart_to_pred(x_t, t, x_0)
                return result
            return pred

        def _tapa_inference_model(
            model, x_t, t, cond,
            neg_cond=None, guidance_strength=1.0,
            guidance_interval=(0.0, 1.0), guidance_rescale=0.0,
            cfg_mode='standard', apg_alpha=0.3,
            **kwargs
        ):
            # Compute blend weight: 1.0 = pure concat, 0.0 = pure single
            if blend_width <= 0 or t >= t_high:
                alpha = 1.0
            elif t <= t_low:
                alpha = 0.0
            else:
                alpha = (t - t_low) / (t_high - t_low)

            if alpha >= 1.0:
                # Pure concat
                return _run_with_cfg(
                    model, x_t, t, concat_cond_tensor, concat_neg_tensor,
                    guidance_strength, guidance_interval, guidance_rescale,
                    cfg_mode=cfg_mode, apg_alpha=apg_alpha, **kwargs
                )
            elif alpha <= 0.0:
                # Pure single
                return _run_with_cfg(
                    model, x_t, t, single_cond_tensor, single_neg_tensor,
                    guidance_strength, guidance_interval, guidance_rescale,
                    cfg_mode=cfg_mode, apg_alpha=apg_alpha, **kwargs
                )
            else:
                # Blend zone: run both and lerp predictions
                pred_concat = _run_with_cfg(
                    model, x_t, t, concat_cond_tensor, concat_neg_tensor,
                    guidance_strength, guidance_interval, guidance_rescale,
                    cfg_mode=cfg_mode, apg_alpha=apg_alpha, **kwargs
                )
                pred_single = _run_with_cfg(
                    model, x_t, t, single_cond_tensor, single_neg_tensor,
                    guidance_strength, guidance_interval, guidance_rescale,
                    cfg_mode=cfg_mode, apg_alpha=apg_alpha, **kwargs
                )
                return alpha * pred_concat + (1 - alpha) * pred_single

        sampler._inference_model = _tapa_inference_model
        try:
            yield
        finally:
            if '_inference_model' in sampler.__dict__:
                del sampler._inference_model

    @contextmanager
    def inject_view_weighted_cross_attn(
        self, sampler, mv_cond, camera_params,
        mv_cond_secondary=None,
        temperature=0.1, power=3.0,
    ):
        """
        Per-view cross-attention with visibility-weighted merge.

        Runs cross-attention separately for each view, then merges outputs
        using direction-based visibility weights. Self-attention and FFN
        run only once per block — only cross-attention is multiplied by K.

        Args:
            sampler: The sampler to monkey-patch.
            mv_cond: Primary multiview cond dict {'cond': (K,N,D), 'neg_cond': (1,N,D)}.
            camera_params: List of K camera param dicts with 'yaw'/'pitch'.
            mv_cond_secondary: Optional second-resolution cond for cascade pipelines.
            temperature: Visibility softmax temperature.
            power: Visibility weight sharpening exponent.
        """
        from ..utils.visibility import compute_sharp_visibility_weights
        from ..modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock

        K = mv_cond['cond'].shape[0]

        # Build per-view cond lookup keyed by sequence length N
        # so the patched blocks can find the right resolution's features
        per_view_by_n = {}
        neg_by_n = {}
        N_primary = mv_cond['cond'].shape[1]
        per_view_by_n[N_primary] = [mv_cond['cond'][i:i+1] for i in range(K)]
        neg_by_n[N_primary] = mv_cond['neg_cond']

        if mv_cond_secondary is not None:
            N_sec = mv_cond_secondary['cond'].shape[1]
            per_view_by_n[N_sec] = [mv_cond_secondary['cond'][i:i+1] for i in range(K)]
            neg_by_n[N_sec] = mv_cond_secondary['neg_cond']

        # Shared mutable state for communication between _inference_model and block patches
        view_state = {
            'active': False,
            'per_view_cond': None,  # Set per forward pass based on resolution
            'weights': None,        # Lazily computed [K, T] visibility weights
            'weights_coords_id': None,
        }

        # --- Block-level patching ---
        patched_blocks = []

        def patch_model_blocks(model):
            for block in model.blocks:
                if not isinstance(block, ModulatedSparseTransformerCrossBlock):
                    continue
                if hasattr(block, '_orig_forward_vw'):
                    continue  # Already patched

                orig_fn = block._forward
                block._orig_forward_vw = orig_fn

                def make_patched(blk, orig):
                    def patched_forward(x, mod, context):
                        if not view_state['active']:
                            return orig(x, mod, context)

                        # --- Modulation (same as original lines 143-146) ---
                        dtype = x.feats.dtype
                        if blk.share_mod:
                            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                                (blk.modulation + mod).type(mod.dtype).chunk(6, dim=1)
                        else:
                            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                                blk.adaLN_modulation(mod).chunk(6, dim=1)
                        # Ensure modulation outputs match model dtype to prevent float32 promotion
                        shift_msa, scale_msa, gate_msa = shift_msa.to(dtype), scale_msa.to(dtype), gate_msa.to(dtype)
                        shift_mlp, scale_mlp, gate_mlp = shift_mlp.to(dtype), scale_mlp.to(dtype), gate_mlp.to(dtype)

                        # --- Self-attention (unchanged, lines 147-151) ---
                        h = x.replace(blk.norm1(x.feats).to(dtype=dtype))
                        h = h * (1 + scale_msa) + shift_msa
                        h = blk.self_attn(h)
                        h = h * gate_msa
                        x = x + h

                        # --- Per-view cross-attention + visibility merge ---
                        h_normed = x.replace(blk.norm2(x.feats).to(dtype=dtype))

                        # Compute/reuse visibility weights (same coords across all blocks in one forward)
                        coords_id = id(x.coords)
                        if view_state['weights'] is None or view_state['weights_coords_id'] != coords_id:
                            grid_res = x.coords[:, 1:].max().item() + 1
                            view_state['weights'] = compute_sharp_visibility_weights(
                                x.coords, camera_params, grid_res,
                                temperature=temperature, power=power,
                            )
                            view_state['weights_coords_id'] = coords_id
                        per_view = view_state['per_view_cond']
                        # Cast visibility weights to block dtype (model may use bfloat16)
                        w = view_state['weights'].to(dtype=dtype)  # [K, T]

                        cross_feats = []
                        for k in range(K):
                            cond_k = per_view[k].to(dtype=dtype)
                            h_k = blk.cross_attn(h_normed, cond_k)
                            cross_feats.append(h_k.feats)

                        stacked = torch.stack(cross_feats, dim=0)  # [K, T, D]
                        merged = (w.unsqueeze(-1) * stacked).sum(dim=0).to(dtype=dtype)  # [T, D]
                        x = x + x.replace(merged)

                        # --- FFN (unchanged, lines 155-159) ---
                        h = x.replace(blk.norm3(x.feats).to(dtype=dtype))
                        h = h * (1 + scale_mlp) + shift_mlp
                        h = blk.mlp(h)
                        h = h * gate_mlp
                        x = x + h

                        return x
                    return patched_forward

                block._forward = make_patched(block, orig_fn)
                patched_blocks.append(block)

        def unpatch_all():
            for block in patched_blocks:
                if hasattr(block, '_orig_forward_vw'):
                    block._forward = block._orig_forward_vw
                    del block._orig_forward_vw
            patched_blocks.clear()

        # --- Sampler-level patching ---
        def _vw_inference_model(
            model, x_t, t, cond,
            neg_cond=None, guidance_strength=1.0,
            guidance_interval=(0.0, 1.0), guidance_rescale=0.0,
            cfg_mode='standard', apg_alpha=0.3,
            **kwargs
        ):
            # Patch model blocks on first encounter
            patch_model_blocks(model)

            # Determine which resolution's per-view cond to use
            cond_n = cond.shape[1] if cond.dim() >= 2 else -1
            if cond_n in per_view_by_n:
                per_view = per_view_by_n[cond_n]
                neg = neg_by_n[cond_n]
            else:
                # Fallback to primary
                per_view = per_view_by_n[N_primary]
                neg = neg_by_n[N_primary]

            # Positive prediction with per-view cross-attention
            view_state['active'] = True
            view_state['per_view_cond'] = per_view
            view_state['weights'] = None  # Reset cache per timestep
            pred = FlowEulerSampler._inference_model(
                sampler, model, x_t, t, per_view[0], **kwargs
            )
            view_state['active'] = False

            # CFG within guidance interval
            if guidance_interval[0] <= t <= guidance_interval[1] and guidance_strength != 1:
                if guidance_strength == 0:
                    return FlowEulerSampler._inference_model(
                        sampler, model, x_t, t, neg, **kwargs
                    )
                neg_pred = FlowEulerSampler._inference_model(
                    sampler, model, x_t, t, neg, **kwargs
                )
                result = compute_cfg_prediction(pred, neg_pred, guidance_strength,
                                                cfg_mode=cfg_mode, apg_alpha=apg_alpha)

                if guidance_rescale > 0:
                    x_0_pos = sampler._pred_to_xstart(x_t, t, pred)
                    x_0_cfg = sampler._pred_to_xstart(x_t, t, result)
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    result = sampler._xstart_to_pred(x_t, t, x_0)

                return result
            return pred

        sampler._inference_model = _vw_inference_model
        try:
            yield
        finally:
            if '_inference_model' in sampler.__dict__:
                del sampler._inference_model
            unpatch_all()

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        hull_mask: torch.Tensor = None,
        max_removal_ratio: float = 0.3,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            hull_mask (torch.Tensor): Optional visual hull mask [1, 1, R, R, R] bool.
            max_removal_ratio (float): Safety cap on fraction of voxels hull can remove.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        # Visual Hull masking (applied after max_pool, see Doc 14 R-P2-2)
        if hull_mask is not None:
            decoded_before = decoded.clone()
            decoded = decoded & hull_mask
            before_count = decoded_before.sum().item()
            after_count = decoded.sum().item()
            removal = 1.0 - after_count / max(before_count, 1)
            if removal > max_removal_ratio:
                # Fall back to dilated hull to avoid over-trimming
                hull_dilated = torch.nn.functional.max_pool3d(
                    hull_mask.float(), 5, stride=1, padding=2
                ) > 0
                decoded = decoded_before & hull_dilated
            # If hull masking removed everything, skip it entirely
            if decoded.sum().item() == 0:
                decoded = decoded_before

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 65536,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution <= 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes(max_hole_perimeter=0.3)
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def _run_single(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 65536,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run(
        self,
        image: Union[Image.Image, Dict[str, Image.Image]],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 65536,
        multiview_mode: str = 'multidiffusion',
        texture_multiview_mode: Optional[str] = None,
        custom_yaw_angles: Optional[list] = None,
        best_of_n: int = 1,
        quality_verifier = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image: Single PIL image or dict mapping view names to images.
                   View names: 'front', 'back', 'left', 'right', 'top', 'bottom'.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline.
            max_num_tokens (int): The maximum number of tokens to use.
            multiview_mode (str): 'concat' (cross-attention feature concatenation, recommended),
                                  'view_weighted' (per-view cross-attn with visibility merge),
                                  'multidiffusion' (average K predictions per step), or
                                  'stochastic' (cycle views, fastest but lowest quality).
            texture_multiview_mode (str): Override multiview mode for texture stage only.
                                          Options: 'concat', 'view_weighted', 'single' (front-view only),
                                          'tapa' (TAPA: concat for t>0.5, single for t<=0.5),
                                          'multidiffusion'. If None, defaults to 'single'.
            best_of_n (int): Generate N candidates and return the best one (1 = no selection).
            quality_verifier: QualityVerifier instance for scoring candidates. Required when best_of_n > 1.
        """
        # --- Best-of-N selection loop ---
        if best_of_n > 1 and quality_verifier is not None:
            # Preprocess once before the loop to avoid redundant work
            if preprocess_image:
                if isinstance(image, Image.Image):
                    image = self.preprocess_image(image)
                elif isinstance(image, dict):
                    image = {k: self.preprocess_image(v) for k, v in image.items()}

            # Extract reference image for scoring
            if isinstance(image, Image.Image):
                ref_image = image
            elif isinstance(image, dict):
                ref_image = image.get('front', next(iter(image.values())))
            else:
                ref_image = image

            best_result = None
            best_score = -float('inf')
            all_scores = []

            for i in range(best_of_n):
                print(f"[Best-of-{best_of_n}] Generating candidate {i+1}/{best_of_n} (seed={seed+i})")
                result = self.run(
                    image,
                    num_samples=num_samples,
                    seed=seed + i,
                    sparse_structure_sampler_params=sparse_structure_sampler_params,
                    shape_slat_sampler_params=shape_slat_sampler_params,
                    tex_slat_sampler_params=tex_slat_sampler_params,
                    preprocess_image=False,  # already preprocessed
                    return_latent=return_latent,
                    pipeline_type=pipeline_type,
                    max_num_tokens=max_num_tokens,
                    multiview_mode=multiview_mode,
                    texture_multiview_mode=texture_multiview_mode,
                    custom_yaw_angles=custom_yaw_angles,
                    best_of_n=1,  # prevent recursion
                )

                if return_latent:
                    meshes, latents = result
                else:
                    meshes = result

                mesh = meshes[0]
                score_dict = quality_verifier.score(mesh, ref_image, use_dreamsim=True)
                all_scores.append(score_dict)
                ds_str = f", dreamsim={score_dict['dreamsim']:.3f}" if 'dreamsim' in score_dict else ""
                print(f"  Candidate {i+1}: total={score_dict['total']:.3f} "
                      f"(lpips={score_dict['lpips']:.3f}, geo={score_dict['geometric']:.3f}, "
                      f"color={score_dict['color_richness']:.3f}{ds_str})")

                if score_dict['total'] > best_score:
                    old_result = best_result
                    best_result = result
                    best_score = score_dict['total']
                    del old_result
                else:
                    del result

                # Drop local references before cache cleanup
                del meshes, mesh
                if return_latent:
                    del latents
                torch.cuda.empty_cache()

            print(f"[Best-of-{best_of_n}] Selected best candidate: score={best_score:.3f}")
            return best_result
        # Single image: delegate directly
        if isinstance(image, Image.Image):
            return self._run_single(
                image, num_samples, seed,
                sparse_structure_sampler_params, shape_slat_sampler_params,
                tex_slat_sampler_params, preprocess_image, return_latent,
                pipeline_type, max_num_tokens,
            )

        # Dict with single view: unwrap and delegate
        if isinstance(image, dict) and len(image) == 1:
            only_image = next(iter(image.values()))
            return self._run_single(
                only_image, num_samples, seed,
                sparse_structure_sampler_params, shape_slat_sampler_params,
                tex_slat_sampler_params, preprocess_image, return_latent,
                pipeline_type, max_num_tokens,
            )

        # Multi-view path (dict with >=2 views)
        assert isinstance(image, dict) and len(image) >= 2, \
            f"Expected dict with >=2 views, got {type(image)} with {len(image)} entries"

        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'tex_slat_flow_model_512' in self.models
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models
            assert 'shape_slat_flow_model_1024' in self.models
            assert 'tex_slat_flow_model_1024' in self.models
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        # Preprocess all views
        if preprocess_image:
            image = {k: self.preprocess_image(v) for k, v in image.items()}

        torch.manual_seed(seed)

        # Get per-view conditioning: cond shape (K, N, D)
        mv_cond_512 = self.get_cond_multiview(image, 512)
        mv_cond_1024 = self.get_cond_multiview(image, 1024) if pipeline_type != '512' else None
        num_views = mv_cond_512['cond'].shape[0]

        # Compute Visual Hull mask from multi-view silhouettes
        from ..utils.visual_hull import compute_visual_hull
        from ..utils.visibility import get_camera_params_from_views
        if custom_yaw_angles and len(custom_yaw_angles) == len(image):
            cam_params = [{'yaw': y, 'pitch': 0.0} for y in custom_yaw_angles]
        else:
            cam_params = get_camera_params_from_views(list(image.keys()))
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]

        # Skip Visual Hull when all views are horizontal (no vertical constraint)
        all_horizontal = all(abs(cam.get('pitch', 0.0)) < 0.1 for cam in cam_params)
        if all_horizontal and len(cam_params) <= 3:
            hull_mask = None
        else:
            silhouettes = []
            for img in image.values():
                img_np = np.array(img).astype(np.float32) / 255.0
                if img_np.ndim == 3 and img_np.shape[2] == 4:
                    alpha = img_np[:, :, 3]
                else:
                    alpha = img_np.max(axis=-1) if img_np.ndim == 3 else img_np
                silhouettes.append(torch.from_numpy(alpha).float().to(self.device))
            hull_mask = compute_visual_hull(
                silhouettes, cam_params, grid_resolution=ss_res,
            )

        # Prepare per-stage conditioning based on multiview_mode
        front_cond_512 = {'cond': mv_cond_512['cond'][:1], 'neg_cond': mv_cond_512['neg_cond']}
        front_cond_1024 = {'cond': mv_cond_1024['cond'][:1], 'neg_cond': mv_cond_1024['neg_cond']} if mv_cond_1024 is not None else None

        # Cross-attention concatenation: (K, N, D) → (1, K*N, D)
        # Each sparse token attends to ALL views' features — cross-attention natively
        # selects relevant views per-token. No averaging, no variance collapse.
        concat_cond_512 = None
        concat_cond_1024 = None
        tex_mode = texture_multiview_mode or 'single'
        if multiview_mode == 'concat' or tex_mode == 'concat':
            K, N, D = mv_cond_512['cond'].shape
            concat_cond_512 = {
                'cond': mv_cond_512['cond'].reshape(1, K * N, D),
                'neg_cond': torch.zeros(1, K * N, D, device=self.device),
            }
            if mv_cond_1024 is not None:
                K2, N2, D2 = mv_cond_1024['cond'].shape
                concat_cond_1024 = {
                    'cond': mv_cond_1024['cond'].reshape(1, K2 * N2, D2),
                    'neg_cond': torch.zeros(1, K2 * N2, D2, device=self.device),
                }

        # Stage 1 (sparse structure): ALWAYS use front-view only.
        # The SparseStructureFlowModel was trained with single-view conditioning (1, N, D).
        # Feeding K*N tokens (concat) is out-of-distribution and causes the model to
        # generate multiple separate objects instead of one coherent shape.
        # Multi-view information is used in Stage 2 (shape) and Stage 3 (texture) only.
        ss_cond = front_cond_512
        coords = self.sample_sparse_structure(
            ss_cond, ss_res,
            num_samples, sparse_structure_sampler_params,
            hull_mask=hull_mask,
        )

        # Shape stage: concat bypasses monkey-patching entirely (K× faster, no variance collapse)
        if multiview_mode == 'concat':
            shape_ctx = nullcontext()
            shape_lr_cond = concat_cond_512
            shape_hr_cond = concat_cond_1024
        elif multiview_mode == 'view_weighted':
            shape_ctx = self.inject_view_weighted_cross_attn(
                self.shape_slat_sampler, mv_cond_512, cam_params,
                mv_cond_secondary=mv_cond_1024,
                temperature=0.1, power=3.0,
            )
            shape_lr_cond = front_cond_512
            shape_hr_cond = front_cond_1024
        else:
            shape_ctx = self.inject_sampler_multi_image(
                self.shape_slat_sampler, num_views, multiview_mode, cam_params
            )
            shape_lr_cond = mv_cond_512
            shape_hr_cond = mv_cond_1024

        # Texture stage: 'single' uses front-view only (safest default),
        # 'concat' uses all views via cross-attention, 'multidiffusion' uses averaging,
        # 'tapa' switches from concat to single mid-generation
        if tex_mode == 'concat':
            tex_ctx = nullcontext()
            tex_cond_512 = concat_cond_512
            tex_cond_1024 = concat_cond_1024
        elif tex_mode == 'tapa':
            # TAPA: uses concat for structural timesteps, single for detail
            # Ensure concat cond is available
            if concat_cond_512 is None:
                K, N, D = mv_cond_512['cond'].shape
                concat_cond_512 = {
                    'cond': mv_cond_512['cond'].reshape(1, K * N, D),
                    'neg_cond': torch.zeros(1, K * N, D, device=self.device),
                }
            if concat_cond_1024 is None and mv_cond_1024 is not None:
                K2, N2, D2 = mv_cond_1024['cond'].shape
                concat_cond_1024 = {
                    'cond': mv_cond_1024['cond'].reshape(1, K2 * N2, D2),
                    'neg_cond': torch.zeros(1, K2 * N2, D2, device=self.device),
                }
            # The inject_tapa_conditioning uses the appropriate resolution cond
            tex_cond_512 = front_cond_512  # placeholder, overridden by TAPA
            tex_cond_1024 = front_cond_1024
            tex_ctx = self.inject_tapa_conditioning(
                self.tex_slat_sampler,
                concat_cond_1024 or concat_cond_512,
                front_cond_1024 or front_cond_512,
                threshold=0.7,
                blend_width=0.2,
            )
        elif tex_mode == 'view_weighted':
            tex_ctx = self.inject_view_weighted_cross_attn(
                self.tex_slat_sampler,
                mv_cond_1024 or mv_cond_512, cam_params,
                mv_cond_secondary=mv_cond_512 if mv_cond_1024 is not None else None,
                temperature=0.1, power=3.0,
            )
            tex_cond_512 = front_cond_512
            tex_cond_1024 = front_cond_1024
        elif tex_mode == 'single':
            tex_ctx = nullcontext()
            tex_cond_512 = front_cond_512
            tex_cond_1024 = front_cond_1024
        else:
            tex_ctx = self.inject_sampler_multi_image(
                self.tex_slat_sampler, num_views, tex_mode, cam_params
            )
            tex_cond_512 = mv_cond_512
            tex_cond_1024 = mv_cond_1024

        if pipeline_type == '512':
            with shape_ctx:
                shape_slat = self.sample_shape_slat(
                    shape_lr_cond, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params,
                )
            with tex_ctx:
                tex_slat = self.sample_tex_slat(
                    tex_cond_512, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params,
                )
            res = 512
        elif pipeline_type == '1024':
            with shape_ctx:
                shape_slat = self.sample_shape_slat(
                    shape_hr_cond, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params,
                )
            with tex_ctx:
                tex_slat = self.sample_tex_slat(
                    tex_cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params,
                )
            res = 1024
        elif pipeline_type == '1024_cascade':
            with shape_ctx:
                shape_slat, res = self.sample_shape_slat_cascade(
                    shape_lr_cond, shape_hr_cond,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1024,
                    coords, shape_slat_sampler_params,
                    max_num_tokens,
                )
            with tex_ctx:
                tex_slat = self.sample_tex_slat(
                    tex_cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params,
                )
        elif pipeline_type == '1536_cascade':
            with shape_ctx:
                shape_slat, res = self.sample_shape_slat_cascade(
                    shape_lr_cond, shape_hr_cond,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1536,
                    coords, shape_slat_sampler_params,
                    max_num_tokens,
                )
            with tex_ctx:
                tex_slat = self.sample_tex_slat(
                    tex_cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params,
                )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
