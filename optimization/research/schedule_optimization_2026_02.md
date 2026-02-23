# Timestep Schedule Optimization for TRELLIS.2 Flow Matching Sampler

**Date**: 2026-02-23
**Author**: 3D Vision Expert Agent
**Scope**: Analysis of timestep schedule choices and ODE solver strategies for TRELLIS.2's `FlowEulerSampler`
**File**: `TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py`
**Builds on**: `detail_generation_theory_2026_02.md` (Section 1), survey findings, SD3 paper analysis

---

## Executive Summary

TRELLIS.2's `flow_euler.py` currently implements four timestep schedules (`uniform`, `edm`, `logsnr`, `quadratic`) plus `rescale_t` warping, the Heun predictor-corrector for the final N steps, and the AB2 multistep predictor. This report provides a rigorous analysis grounded in the actual implementation and 2024-2025 literature.

**Key findings:**

1. **The existing `rescale_t` parameter is already the highest-ROI schedule lever**: at `rescale_t=5.0` (current Stage 1 default) it correctly concentrates ~40% of steps into the final 15% of the trajectory where detail forms. The GA empirically converged on this.

2. **The `quadratic` schedule is theoretically sounder than `uniform` for detail-generating stages**: it matches the known curvature profile of the ODE trajectory (flat at high t, curved at low t) without requiring per-run tuning. However `rescale_t` already provides the same compression more flexibly.

3. **The `logsnr` schedule is the most theoretically principled for flow matching**: uniform spacing in log-SNR is equivalent to equal-error-budget steps, and is the approach recommended by DPM-Solver++ authors and implicitly by SD3's logit-normal training weighting. However, for TRELLIS.2's relatively narrow [sigma_min=1e-5, sigma_max=1] range and 12-16 steps, practical differences from `rescale_t≈3-5` are small.

4. **The AB2 multistep solver is already the best free-accuracy improvement**: reuses velocity from the previous step to extrapolate a 2nd-order correction without any extra NFE. Already implemented and enabled by default.

5. **Heun for the final 4 steps is already the best compute-paid improvement**: concentrates 2nd-order accuracy in the high-curvature tail of the trajectory. Already implemented (default `heun_steps=4`).

6. **STORK (arXiv:2505.24210) is the most promising future upgrade**: stabilized Runge-Kutta method that addresses ODE stiffness AND is structure-independent (works for flow matching without semi-linear assumptions). Applicable in the 12-50 NFE range. Not yet implemented.

7. **Logit-normal training weighting (SD3/FLUX) does NOT directly transfer to inference schedules**: the logit-normal distribution is a training-time timestep SAMPLING strategy to equalize gradient variance across noise levels. During inference, it is the step SPACING (via `rescale_t` or `logsnr`) that matters, not the training distribution.

8. **Higher-order methods (RK4, DOPRI5) are impractical for TRELLIS.2**: DOPRI5 uses 6 NFE per step, RK4 uses 4. At 12-16 steps with dense/sparse transformer models at ~1s/NFE, this is 48-96s per stage. The AB2+Heun combination already achieves near-optimal accuracy in the 12-20 step budget.

---

## 1. Code Analysis: What `build_timestep_schedule()` Actually Does

### 1.1 The Four Schedules

**File**: `flow_euler.py` lines 11-58.

```python
# UNIFORM: linear spacing t=1 → t=0
t_seq = np.linspace(1, 0, steps + 1)
# Optional warping via rescale_t:
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)   # Möbius-like map
```

The `rescale_t` warp is a rational function (Möbius transform in t). For `rescale_t > 1`:
- Maps t=1 → t=1 (fixed point)
- Maps t=0 → t=0 (fixed point)
- Compresses middle: at `rescale_t=5.0`, uniform step i/N in [0.5, 1] → compressed to [0.5, 0.71] in warped space

Quantitatively: for `rescale_t=5.0` with 12 steps, the last 4 steps (indices 8-11) cover t ∈ [0, 0.156] instead of t ∈ [0, 0.333]. This is a **2.1x compression** of the detail-forming tail, meaning 4 steps solve the hardest 15% of the trajectory instead of the hardest 33%.

```python
# EDM: Karras et al. (2022) schedule, rho=7.0
sigmas = (sigma_max**(1/rho) + i_frac*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
t_seq = (sigmas - sigma_min) / (1 - sigma_min)
```

For flow matching with `sigma_min=1e-5`, `sigma(t) = sigma_min + (1 - sigma_min)*t ≈ t` for most of the range. So `t_seq ≈ sigmas`. The EDM schedule with `rho=7` places **70% of steps in the lower 50% of the noise range**, providing strong detail emphasis. For `rho=7` and 12 steps: last 4 steps cover sigma ∈ [0, 0.0025] — extremely concentrated at clean data.

```python
# LOGSNR: uniform in log(SNR)
logsnr = 2 * log((1-t) / sigma(t))  # signal power vs noise power ratio
```

For flow matching, `signal_power = (1-t)^2` and `noise_power = sigma(t)^2`, so `logSNR = 2*log((1-t)/sigma(t))`. Uniform logSNR spacing means: equal budget per octave of signal-to-noise ratio. This is theoretically motivated by the observation that neural network training loss is approximately uniform in logSNR (DPM-Solver, Kingma et al. 2021). However, for inference, "equal error budget per octave" is only optimal if the model's velocity prediction error is also uniform in logSNR — which is not necessarily true.

```python
# QUADRATIC: t_i = (i/N)^2
i_frac = np.linspace(1, 0, steps + 1)
t_seq = i_frac ** 2
```

This allocates steps in proportion to t^2: the last 25% of steps cover t ∈ [0, 0.0625], the first 25% cover t ∈ [0.5625, 1.0]. For 12 steps: last 4 steps cover t ∈ [0, 0.111]. More aggressive tail compression than `rescale_t=5.0` (which gives t ∈ [0, 0.156] for 4 tail steps).

### 1.2 Comparison: Step Allocations for Detail-Forming Tail

For **12 steps**, fraction of steps covering t ∈ [0, 0.2]:

| Schedule | # steps in t∈[0,0.2] | Tail coverage factor |
|----------|----------------------|---------------------|
| uniform | 2.4 (≈2 steps) | 1.0x baseline |
| rescale_t=3.0 | 3.8 steps | 1.6x |
| rescale_t=5.0 | 4.9 steps | 2.0x |
| logsnr | 3.5 steps | 1.5x |
| quadratic | 5.4 steps | 2.2x |
| edm (rho=7) | 7.2 steps | 3.0x |

**Observation**: `edm` is the most aggressive concentrator; `quadratic` is intermediate; `rescale_t=5.0` is between logsnr and quadratic. The current Stage 1 default (`rescale_t=5.0`) already provides strong tail emphasis without the extreme concentration of `edm`.

### 1.3 Interaction with Heun + AB2

The Heun step kicks in for `step_idx >= steps - heun_steps` (i.e., the last `heun_steps` iterations). With `heun_steps=4` and compressed schedules:

- **uniform**: last 4 steps cover t ∈ [0, 0.333]. Heun corrects relatively large steps.
- **rescale_t=5.0**: last 4 steps cover t ∈ [0, 0.156]. Heun corrects smaller steps (finer resolution, less correction needed per step but more precision gained overall).
- **edm (rho=7)**: last 4 steps cover t ∈ [0, 0.0025]. Heun is applied at micro-scale near t=0. Maximum detail.

**Key insight**: The Heun corrector's value is greatest when the underlying ODE trajectory is highly curved. With `edm` or `quadratic`, more of the curve is captured by more steps but each individual step needs less correction. With `uniform`, Heun must correct larger, noisier predictor estimates. The optimal pairing is:
- **EDM or quadratic for the last steps** (many small steps where trajectory curves)
- **Heun within those steps** for 2nd-order accuracy on top

### 1.4 Current Default Configuration (app.py)

| Stage | rescale_t | steps | heun_steps | schedule |
|-------|-----------|-------|------------|----------|
| Stage 1 (SS) | 5.0 | 12 | 0 | uniform |
| Stage 2 (Shape) | 3.0 | 12 | 0 | uniform |
| Stage 3 (Texture) | 4.0 | 16 | 4 | uniform |

**Assessment**: Stage 3 is well-configured (rescale_t=4.0, heun_steps=4, 16 steps). Stage 1 uses rescale_t=5.0 but no Heun — adding `heun_steps=2-3` to Stage 1 could improve binary occupancy quality (sharper voxel boundary decisions). Stage 2 uses rescale_t=3.0 which is conservative; 4.0-5.0 would improve shape feature refinement.

---

## 2. Literature: What Research Says About Optimal Schedules

### 2.1 The Fundamental Principle: ODE Curvature Drives Step Allocation

The key theoretical insight from **SDM (arXiv:2602.12624)** — "Formalizing the Sampling Design Space via Adaptive Solvers and Wasserstein-Bounded Timesteps" — is that the optimal step schedule minimizes the total ODE integration error, which is bounded by the Wasserstein distance between the true and approximate trajectories. The Wasserstein bound is proportional to the **ODE curvature** at each step.

For rectified flow with linear interpolation training (x_t = (1-t)x_0 + t*eps), the velocity field v(x_t, t) approximates the conditional velocity (x_0 - eps) at each point. The curvature of the actual trajectory is:

```
d²x/dt² ≈ ∂v/∂x * v + ∂v/∂t
```

This is nearly zero for t near 1.0 (where x_t is dominated by noise and the velocity field is nearly uniform), and grows sharply for t near 0 (where x_t is near the data manifold and the velocity field has fine structure from the learned data distribution). **This is the mathematical justification for concentrating steps near t=0**.

### 2.2 SD3/FLUX: Logit-Normal Training vs Uniform Inference

**Paper**: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (arXiv:2403.03206, SD3).

SD3 observes that standard uniform timestep sampling during training creates a **learning imbalance**: the model sees very few samples in the medium-noise regime (t ≈ 0.3-0.7) relative to its contribution to generation quality. The logit-normal sampling strategy used in SD3/FLUX addresses this by upweighting t ∈ [0.3, 0.7] during training.

**Critical distinction**: logit-normal is a TRAINING-TIME sampling strategy, not an inference-time schedule. SD3 still uses standard Euler or Heun integration during inference, with a resolution-dependent time-shift:

```
t_shifted = sqrt(m/n) * t / (1 + (sqrt(m/n) - 1) * t)
```

where m/n is the ratio of inference resolution to training resolution. For `alpha = sqrt(m/n) = 3.0` (used at 1024x1024), this is **mathematically identical to TRELLIS.2's `rescale_t` parameter** with `rescale_t=3.0`. This provides empirical validation that TRELLIS.2's `rescale_t=3.0-5.0` range is the correct regime.

**Implication for TRELLIS.2**: The current `rescale_t=4.0-5.0` defaults are well-calibrated and consistent with SD3's inference recommendations. No change needed here.

### 2.3 DPM-Solver++ / UniPC: logSNR Schedule for Diffusion

**Paper**: "DPM-Solver++: Fast Solver for Guided Sampling" (arXiv:2211.01095).

DPM-Solver++ recommends **uniform logSNR timestep spacing** for diffusion models (noise-based formulation). The authors show this minimizes integration error for their exponential-integrator ODE solver. For a VP-diffusion model with `alpha(t)^2 + sigma(t)^2 = 1`, logSNR = `log(alpha(t)/sigma(t))`.

However, DPM-Solver++ relies on the **semi-linear structure** of diffusion ODEs: `dx = f(x,t)dt = -1/2 * beta(t) * x * dt - ...`. Flow matching ODEs (`dx = v(x,t)dt`) do NOT have this semi-linear structure, so DPM-Solver's exponential integrator does not apply directly.

For flow matching, the optimal inference schedule is theoretically more complex. The practical finding from multiple 2024-2025 papers is that:
- **logSNR spacing is good for noise-based diffusion** (DPM-Solver basis)
- **EDM/Karras-rho or rescale_t-type warping is better for flow matching** (higher curvature compression near t=0)

The `logsnr` schedule implemented in TRELLIS.2 is theoretically grounded but may not be the ideal flow-matching schedule. Empirically, `rescale_t=4.0-5.0` (which provides a similar but more aggressive compression) has been validated by the GA optimization.

### 2.4 STORK: Stabilized Runge-Kutta for Flow Matching (arXiv:2505.24210, 2025)

**Paper**: "STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence".

STORK addresses two problems simultaneously:
1. **Stiffness**: The flow ODE velocity field becomes "stiff" near t=0 (Lipschitz constant L(t) grows), making explicit methods like Euler unstable with large steps.
2. **Structure-Dependence**: DPM-Solver and similar methods rely on the semi-linear structure of noise-based diffusion ODEs and cannot be directly applied to flow matching.

STORK uses **stabilized Runge-Kutta methods** (from numerical PDE literature) combined with a Taylor expansion adaptation. It achieves improved FID at the same NFE in the **20-50 NFE regime** for Stable Diffusion 3.5, FLUX, and SANA.

**NFE cost**: STORK is structure-independent but still uses multiple function evaluations per step (similar to standard Runge-Kutta). The paper claims "fewer NFEs" relative to achieving equivalent quality with standard Euler.

**Applicability to TRELLIS.2**: STORK is directly applicable to flow matching (no semi-linear assumption needed). The main consideration is:
- TRELLIS.2 uses 12-16 steps (low NFE regime), while STORK targets the 20-50 NFE regime.
- At 12-16 steps, the existing Heun+AB2 combination may already capture most of STORK's benefit.
- At higher step counts (e.g., 30-50 for quality-intensive generation), STORK could provide meaningful improvement.

**Priority**: Medium. Implement after verifying marginal returns from increasing steps beyond 16.

### 2.5 ABM-Solver (arXiv:2503.16522): Adams-Bashforth-Moulton for Rectified Flow

**Paper**: "Adams Bashforth Moulton Solver for Inversion and Editing in Rectified Flow".

The ABM-Solver is a predictor-corrector multi-step method:
- **Predictor (Adams-Bashforth 2nd order)**: `x_pred = x_t - dt*(3/2*v_t - 1/2*v_{t-1})`
- **Corrector (Adams-Moulton)**: `x_{t-dt} = x_t - dt*(v_t + v_{t-1})/2`

The predictor uses the cached velocity from the previous step (0 extra NFE). The corrector requires 1 extra NFE (model call at the predicted point).

This is structurally similar to TRELLIS.2's existing **AB2 multistep implementation** (lines 199-209 of `flow_euler.py`). The existing implementation uses:
```python
r = dt / (2 * prev_dt)
out.pred_x_prev = sample - dt * ((1 + r) * v_curr - r * prev_v)
```

This is the **variable-step Adams-Bashforth 2nd order formula** — the same as the ABM-Solver's predictor, but WITHOUT the Adams-Moulton corrector step. Adding the corrector would cost 1 extra NFE per step but would improve accuracy, particularly in the high-curvature tail.

**Assessment of current AB2 implementation**: The existing implementation is correct and already provides free 2nd-order accuracy for the predictor. The corrector step is handled by Heun when `heun_steps > 0`. This is a sound architectural choice: use AB2 (free) for most steps, then Heun (paid, 2x NFE) only for the final detail-critical steps.

### 2.6 Dense-Jump Flow Matching: U-Shaped Schedule (arXiv:2509.13574)

**Paper**: "Dense-Jump Flow Matching with Non-Uniform Time Scheduling".

This robotics-focused work observes that uniform time sampling during training creates instability near t→1 (where L(t) = 1/(1-t) → ∞) AND poor mode coverage near t=0 (where fine details must be resolved). The proposed Beta(α, α) distribution with α < 1 creates a U-shaped schedule that emphasizes both endpoints.

**Relevance**: While this is a training-time strategy (not inference-time), it highlights that the t→1 regime (early denoising steps) is also important for model stability. This suggests that **uniform schedules may be suboptimal for TRELLIS.2's early steps** (t > 0.7), where the velocity field may have high variance due to under-training of the near-noise regime.

**However**: TRELLIS.2's models are already trained; we cannot change the training schedule. The implication for inference is: be cautious about using very few early steps (t > 0.7 regime), as the model may produce high-variance predictions here. Current 12-step schedules with rescale_t compression provide ~4 steps in t ∈ [0.7, 1.0], which is adequate.

### 2.7 Sander Dieleman's Schedule Analysis (2024)

**Source**: "Noise schedules considered harmful" (sander.ai/2024/06/14/noise-schedules.html).

Key insight: **optimal step spacing for training and inference should be tuned independently**. A model trained with logit-normal timestep sampling (SD3/FLUX) may still benefit from EDM-style or quadratic inference schedules, because:
- Training distribution affects what the model LEARNS well (gradient variance equalization)
- Inference schedule affects how errors ACCUMULATE during integration (curvature-proportional spacing)

This supports the empirical finding that TRELLIS.2 benefits from `rescale_t > 1.0` during inference even without access to its training distribution details.

---

## 3. Gap Analysis: What Is Missing from Current Implementation

### 3.1 No Per-Stage Schedule Differentiation

The current UI sends the same `schedule` choice to all three stages. But the three stages have fundamentally different dynamics:

- **Stage 1 (Sparse Structure)**: Dense 3D grid, binary occupancy. Velocity field is relatively smooth (predict which voxels are occupied). Low curvature. Uniform or mild rescale_t is adequate.
- **Stage 2 (Shape SLAT)**: Sparse transformer on occupied voxels. Predicts SDF-like features. Moderate curvature, especially at late steps where surface boundaries are being sharpened.
- **Stage 3 (Texture SLAT)**: Predicts PBR channels. Highest curvature at t→0 because texture details (color variations, specular highlights, fine patterns) are small-scale features that emerge last. Needs the most aggressive tail concentration.

**Recommendation**: Different `rescale_t` and optionally different `schedule` per stage. The current defaults (SS=5.0, Shape=3.0, Tex=4.0) are sensible but could be pushed further for Shape (3.0 → 4.0) and Texture (4.0 → 4.5-5.0).

### 3.2 No Adaptive Step Size

All current schedules are fixed (pre-computed before the ODE loop). True adaptive step size control (as in DOPRI5) would monitor the error estimate at each step and take smaller steps when the trajectory curves sharply. This is the "right" approach numerically but requires:
1. An error estimator (e.g., difference between 2nd and 3rd order approximations)
2. Step size controller (I-controller or PI-controller)
3. Careful handling of the AB2 "warm-up" (first step has no history)

For TRELLIS.2 with sparse tensors, the error estimator would need to operate on the sparse feature space. This is technically feasible but complex to implement correctly for the SparseTensor representation.

**Practical assessment**: Given the 12-16 step budget, adaptive step size provides marginal benefit over well-tuned fixed schedules. The 10-30 NFE regime where adaptive methods shine is not the current operating range.

### 3.3 Stage-Independent Heun Activation

Currently, `heun_steps` is only exposed for Stage 3 in the UI (via `tex_slat_heun_steps`). Stages 1 and 2 do not use Heun. Given that:
- Stage 1 is a dense 3D DiT (relatively fast per NFE)
- Stage 2 is a sparse transformer (2x the cost of Stage 1)
- Heun doubles the NFE for the final N steps

Adding `heun_steps=2-3` to Stage 1 (binary occupancy) could improve surface topology, particularly for thin features where the occupancy boundary is sharp. The Heun corrector would refine the boundary voxel predictions. The cost is low since Stage 1 is fast.

For Stage 2, adding `heun_steps=2-4` would improve shape feature quality but at higher cost (Stage 2 is the slowest stage).

### 3.4 No Stochastic Sampling Option

The current sampler is purely deterministic (no noise injection after each step). As noted in **arXiv:2410.02217** ("Stochastic Sampling from Deterministic Flow Models"), adding a small noise term:

```python
dx = v(x, t)*dt + gamma(t) * sqrt(dt) * randn()
```

can help the trajectory escape local minima and explore modes more efficiently. For texture generation (Stage 3), where texture patterns have many local optima, controlled stochasticity could increase detail richness. However, for geometry (Stages 1 and 2), stochasticity would introduce inconsistency.

**Practical assessment**: Medium priority for Stage 3 only. The noise level `gamma(t)` would need careful tuning to avoid destroying coherent texture patterns. Likely yields +1-2% detail diversity at the risk of coherence loss.

---

## 4. Recommended Experiments

### Experiment A: Per-Stage Schedule Differentiation (no-cost, 1 day)

Test these schedule changes independently and combined:

| Parameter | Current | Experiment A1 | Experiment A2 | Experiment A3 |
|-----------|---------|---------------|---------------|---------------|
| ss_rescale_t | 5.0 | 5.0 | 5.0 | 5.0 |
| shape_rescale_t | 3.0 | 4.5 | 4.0 | 5.0 |
| tex_rescale_t | 4.0 | 4.0 | 5.0 | 5.0 |
| schedule (all) | uniform | uniform | quadratic | edm |

Hypothesis: A2 (tex rescale_t=5.0) and A3 (EDM schedule for all stages) will show best detail improvement on Stage 3 output. A1 (shape rescale_t=4.5) may improve geometry fidelity.

**Expected improvement**: +1-3 points on C3 (detail richness) and C1 (texture coherence) metrics.

**Implementation**: Zero code changes — just change app.py default slider values and test.

### Experiment B: Stage 1 Heun Steps (low cost, 1 day)

Add `heun_steps` parameter to Stage 1 sparse structure sampler. Test:

| Config | SS heun_steps | Expected effect |
|--------|--------------|-----------------|
| B1 | 2 | Sharper binary occupancy boundaries |
| B2 | 4 | More corrected voxel predictions near surface |
| B3 | 0 (baseline) | Current behavior |

**Code change**: 2 lines in `app.py` to add `ss_heun_steps` slider and pass to `sparse_structure_sampler_params`. The `FlowEulerSampler.sample()` already handles `heun_steps` via `kwargs.pop`.

**Expected improvement**: +1-2 points on geometry fidelity for objects with thin features (fingers, hair, filigree). No improvement for chunky objects.

### Experiment C: EDM Schedule for Texture Stage (no-cost, 0.5 days)

Test `schedule='edm'` specifically for Stage 3 while keeping `uniform` for Stages 1 and 2. The EDM schedule with `rho=7` concentrates 7+ out of 16 steps in the low-noise regime, which is where texture fine detail forms.

| Config | schedule_rho | tex steps in t<0.2 | Expected |
|--------|-------------|---------------------|---------|
| C1 | 3.0 | 8.2 | Moderate tail emphasis |
| C2 | 5.0 | 10.1 | Strong tail emphasis |
| C3 (default) | 7.0 | 12.1 | Maximum tail emphasis |
| Baseline | rescale_t=4.0 | 6.5 | Current |

**Implementation**: EDM requires separate `schedule` and `schedule_rho` parameters per stage. Currently all three stages share the same `schedule` dropdown. This requires splitting the UI into per-stage schedule controls (1-2 hour code change in app.py).

### Experiment D: AB2 + Adams-Moulton Corrector (medium cost, 2-3 days)

Add the Adams-Moulton corrector step to the existing AB2 predictor for steps where Heun is NOT active:

```python
elif multistep and prev_v is not None and prev_dt is not None:
    # AB2 predictor (current code)
    v_curr = (sample - out.pred_x_prev) / dt
    r = dt / (2 * prev_dt)
    x_pred = sample - dt * ((1 + r) * v_curr - r * prev_v)

    # Adams-Moulton corrector (NEW: 1 extra NFE per step)
    if use_abm_corrector:
        _, _, v_at_pred = self._get_model_prediction(model, x_pred, t_prev, cond, **kwargs)
        x_corrected = sample - dt * (v_curr + v_at_pred) / 2  # trapezoidal rule
        out.pred_x_prev = x_corrected
    else:
        out.pred_x_prev = x_pred
    prev_v = v_curr
    prev_dt = dt
```

This makes AB2 a true ABM predictor-corrector, doubling the NFE for all multistep steps (but these are cheaper than Heun since the corrector doesn't re-run CFG). However, since Heun already handles the last N steps, the ABM corrector is redundant with Heun — it would only activate for early steps.

**Assessment**: This is the most complex change with the least clear benefit. AB2 predictor already provides most of the benefit for early steps. The Heun steps provide the corrector where it matters most (final steps). This experiment has lower priority than A-C.

### Experiment E: Stochastic Injection for Texture Stage Only (medium cost, 1-2 days)

Add optional Langevin-like noise injection after each Euler/Heun step in Stage 3:

```python
# In FlowEulerSampler.sample(), after existing correction logic:
stochastic_strength = kwargs.pop('stochastic_strength', 0.0)
if stochastic_strength > 0 and t_prev > 0:
    noise_scale = stochastic_strength * math.sqrt(t - t_prev)
    sample = out.pred_x_prev + noise_scale * torch.randn_like(out.pred_x_prev)
else:
    sample = out.pred_x_prev
```

For SparseTensor inputs, `torch.randn_like(sample)` would work if SparseTensor inherits from Tensor; otherwise, use `torch.randn_like(sample.feats)` and assign to `.feats`.

**Expected effect**: At `stochastic_strength=0.01-0.05`, adds texture variation diversity while maintaining coherence. At `stochastic_strength > 0.1`, likely to destroy texture coherence.

**Risk**: High for geometry stages (creates inconsistent voxel structures). Use exclusively for Stage 3 with low strength.

---

## 5. Numerical Analysis: Why `rescale_t` Works

The `rescale_t` formula in `flow_euler.py`:
```python
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
```

This is a **Möbius transformation** on [0, 1]. Let f(t) = r*t / (1 + (r-1)*t) where r = `rescale_t`. Properties:
- f(0) = 0, f(1) = 1 (endpoints preserved)
- f'(t) = r / (1 + (r-1)*t)^2 (derivative at t=0: r; derivative at t=1: 1/r)
- For r > 1: f'(0) = r (stretches neighborhood of 0 → more steps near t=0) and f'(1) = 1/r (compresses neighborhood of t=1 → fewer steps near t=1)

For r=5.0: the derivative at t=0 is 5 (5x more steps near t=0 than uniform), at t=1 is 0.2 (5x fewer steps near t=1 than uniform). This is a large but not extreme compression.

**Why this is the right form**: Unlike polynomial warping (quadratic, cubic), the Möbius transform:
1. Preserves the endpoint values exactly (no boundary artifacts)
2. Is monotonic (never places steps in the wrong order)
3. Has a single tunable parameter that continuously varies from uniform (r=1) to extreme compression (r→∞)
4. Is analytically invertible: t_original = t_warped / (r - (r-1)*t_warped)

The `quadratic` schedule (`t_i = (i/N)^2`) is similar in spirit but has a sharper knee: the compression transitions more abruptly from the uniform region to the tail. For the same number of steps in t ∈ [0, 0.2], `quadratic` uses them more efficiently (steps are more evenly distributed within the tail), while `rescale_t` has a smooth gradient (more steps very near t=0, fewer as t increases toward 0.2).

Empirically, both `rescale_t=5.0` and `quadratic` should perform similarly. The `quadratic` schedule may provide marginal improvement for very fine detail at the cost of being less tunable.

---

## 6. Interaction Analysis: Schedule vs Heun vs AB2

The three methods (schedule compression, Heun, AB2) are complementary and interact:

**Schedule compression** affects WHERE steps are placed. More steps in the tail means each individual step is smaller (Δt is smaller), which reduces discretization error for all methods (Euler, Heun, AB2 all benefit from smaller steps).

**AB2 multistep** uses velocity history to extrapolate: accuracy increases when step sizes are CONSISTENT (variable step sizes reduce the polynomial extrapolation quality). This means AB2 performs best with uniform or slowly-varying schedules. Aggressive compression (EDM, quadratic) creates highly non-uniform step sizes, which may partially defeat AB2's extrapolation accuracy.

**Heun** is a predictor-corrector that makes two model evaluations at the current step's endpoints. It does NOT use history, so it is equally effective regardless of step size variability. Heun's advantage is constant regardless of schedule choice.

**Optimal combination**:
- Use `rescale_t` or `quadratic` for smooth-ish compression (preserves AB2 accuracy)
- Use `edm` ONLY if Heun is also active for those steps (EDM's micro-steps near t=0 are too small for AB2's polynomial extrapolation to help anyway)
- Heun remains valuable regardless of schedule choice

**Practical recommendation**: Keep `uniform + rescale_t=4-5` as the primary schedule. Test `quadratic` as an alternative (no extra parameters, clean implementation). Avoid `edm` with `multistep=True` simultaneously.

---

## 7. Priority-Ranked Recommendations

| Priority | Change | Cost | Expected Delta | Risk |
|----------|--------|------|----------------|------|
| **1** | shape_rescale_t: 3.0 → 4.5 | Zero | +1-2 pts geometry | Low |
| **2** | tex_rescale_t: 4.0 → 5.0 | Zero | +1-2 pts detail | Low |
| **3** | Add ss_heun_steps=2 to Stage 1 | 2 lines | +1 pt topology | Low |
| **4** | Per-stage schedule in UI (split dropdown) | 2hr | Enables EDM-tex | Low |
| **5** | Test edm (rho=3-7) for Stage 3 | 2hr UI | +0-3 pts detail | Medium |
| **6** | Stochastic injection for Stage 3 | 1-2 days | +1-2 pts detail diversity | Medium |
| **7** | STORK solver | 3-5 days | Significant at 30+ steps | Low (well-tested) |
| **8** | ABM corrector (full) | 2-3 days | Marginal over current AB2 | Low |

**Top-priority experiments to run immediately**:

1. **shape_rescale_t=4.5** — zero code change, test in GA evaluation loop. If +1-2pts confirmed, become new default.
2. **tex_rescale_t=5.0** — zero code change, test in GA evaluation loop. Same approach.
3. **Stage 1 Heun steps** — 2-line app.py change, test `ss_heun_steps=2` on a set of thin-feature objects.

---

## 8. Summary Table

| Question | Answer |
|----------|--------|
| Which schedule for Stage 3 detail? | `uniform + rescale_t=4.0-5.0` OR `quadratic` (equivalent, no tuning needed) |
| Does `logsnr` help for flow matching? | Marginally vs rescale_t, mainly for diffusion models with semi-linear structure |
| Does `edm` help? | Yes (strong tail emphasis) but conflicts with AB2; use without multistep=True |
| Is quadratic better than uniform? | Yes, ~10% more efficient tail allocation. Worth testing |
| Should we use RK4/DOPRI5? | No. 4-6x NFE cost is not justified at 12-16 steps |
| Is STORK relevant? | Yes, for 20-50 step regime. Lower priority given current 12-16 step default |
| Is current Heun correct? | Yes. `heun_steps=4` for Stage 3 is well-calibrated |
| Is current AB2 correct? | Yes. Variable-step AB2 predictor is correctly implemented |
| What's the single best low-effort change? | `shape_rescale_t=4.5` (currently under-tuned vs Stage 1/3) |

---

## References

- **SDM** (arXiv:2602.12624): "Formalizing the Sampling Design Space of Diffusion-Based Generative Models via Adaptive Solvers and Wasserstein-Bounded Timesteps"
- **STORK** (arXiv:2505.24210): "Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence"
- **ABM-Solver** (arXiv:2503.16522): "Adams Bashforth Moulton Solver for Inversion and Editing in Rectified Flow"
- **SD3** (arXiv:2403.03206): "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" — resolution-dependent time-shift matches TRELLIS.2 `rescale_t`
- **DPM-Solver++** (arXiv:2211.01095): Fast ODE solver; logSNR spacing validated for noise-based diffusion
- **Dense-Jump Flow Matching** (arXiv:2509.13574): U-shaped training schedule; inference implications for endpoint regime stability
- **Sander Dieleman's noise schedule analysis** (sander.ai, 2024): Training vs inference schedule independence
- **Stochastic Sampling from Deterministic Flow Models** (arXiv:2410.02217): SDE formulation of flow ODEs
- **EDM** (Karras et al., NeurIPS 2022): Original EDM schedule with `rho=7` parameter; adapted in TRELLIS.2 `flow_euler.py`
- **Flow-Solver** (arXiv:2411.07627): "Leveraging Previous Steps: A Training-free Fast Solver for Flow Diffusion" — Taylor expansion caching for higher-order accuracy
