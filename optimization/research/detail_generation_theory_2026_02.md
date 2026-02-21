# Fine Detail Generation in 3D Models: Theoretical Foundations and Practical Techniques

**Date**: 2026-02-21
**Author**: Deep Learning Researcher Agent
**Scope**: Techniques for generating finer geometric and texture details in sparse-voxel + flow-matching architectures (TRELLIS.2)
**Context**: TRELLIS.2 uses sparse structured latents at ~1024 voxel resolution, rectified flow with Euler ODE sampling, CFG guidance, and a 3-stage cascade pipeline (sparse structure -> shape SLAT -> texture SLAT).

---

## Executive Summary

This survey covers 40+ papers from 2024-2026 organized around five research questions on fine detail generation. The findings are ranked by applicability to TRELLIS.2's frozen-weight, inference-time constraints.

**Key insight**: The single highest-impact improvement for detail generation in TRELLIS.2 is *not* a single technique but a *stack* of complementary approaches operating at different levels:

1. **ODE Solver Level**: Higher-order solvers (Heun/midpoint) or multi-step methods that reduce discretization error, especially in the final (near-data) timesteps where detail is formed.
2. **Guidance Level**: Frequency-decoupled guidance (FDG) with stronger high-frequency guidance, combined with attention-based guidance (PAG/NAG/ERG) that specifically enhances structural coherence.
3. **Cascade Level**: Adaptive compute allocation -- more solver steps in the high-resolution cascade stage, fewer in the low-resolution stage.
4. **Post-Generation Level**: DetailGen3D-style coarse-to-fine flow refinement or render-and-compare texture optimization.

The theoretical foundation is that **detail is a late-timestep phenomenon**: in flow matching, the trajectory evolves from globally correct structure (t near 1.0) to locally detailed features (t near 0.0). Most detail is determined in the final 20-30% of the ODE trajectory, making solver accuracy and guidance behavior in this regime critical.

---

## Table of Contents

1. [Flow Matching Solvers and Detail](#1-flow-matching-solvers-and-detail)
2. [Guidance Techniques for Detail](#2-guidance-techniques-for-detail)
3. [Multi-Resolution and Cascade Generation](#3-multi-resolution-and-cascade-generation)
4. [Attention Mechanisms for Detail](#4-attention-mechanisms-for-detail)
5. [Training-Free Detail Enhancement](#5-training-free-detail-enhancement)
6. [Synthesis: Recommended Implementation Stack](#6-synthesis-recommended-implementation-stack)
7. [References](#7-references)

---

## 1. Flow Matching Solvers and Detail

### 1.1 Theoretical Background: Why Euler Loses Detail

TRELLIS.2 uses first-order Euler integration for its flow ODE:

```
x_{t-dt} = x_t - dt * v(x_t, t)
```

where `v(x_t, t)` is the learned velocity field. The local truncation error of Euler is O(dt^2), and the global error is O(dt). For a typical 12-16 step schedule, dt ~ 0.06-0.08, giving a non-trivial approximation error.

**The critical insight from SDM (arXiv:2602.12624)**: The ODE trajectory is *not uniformly curved*. Early timesteps (high noise, t near 1.0) have nearly linear trajectories -- Euler is sufficient. Late timesteps (low noise, t near 0.0) have highly nonlinear, curved trajectories where detail is formed -- higher-order methods are critical.

This means detail loss from Euler is *concentrated in the final steps*, exactly where fine features are being resolved. A non-uniform solver strategy that allocates higher accuracy to late steps is theoretically optimal.

### 1.2 Higher-Order ODE Solvers for Flow Matching

#### Heun's Method (Second-Order, Predictor-Corrector)

The simplest improvement over Euler. For flow matching:

```
# Predict
v_1 = model(x_t, t)
x_pred = x_t - dt * v_1

# Correct
v_2 = model(x_pred, t - dt)
x_{t-dt} = x_t - dt * (v_1 + v_2) / 2
```

**Cost**: 2 NFE per step (double Euler).
**Error**: O(dt^3) local, O(dt^2) global -- one order better than Euler.
**Quality impact**: EDM (Karras et al.) uses Heun and demonstrates significant quality improvement over Euler. For Stable Diffusion 3 (a flow matching model), Heun improves FID measurably at the same total NFE budget.

**Application to TRELLIS.2**: Can be implemented in `flow_euler.py`'s `sample_once()` method by adding a correction step. The key question is whether the 2x compute cost is worthwhile. Based on the SDM analysis, the answer is: **use Heun only for the final K steps** (e.g., last 4-6 steps out of 16), keeping Euler for early steps. This gives most of Heun's quality at ~1.25-1.4x the compute cost.

**Implementation sketch** (for `FlowEulerSampler.sample()`):
```python
# After existing code:
heun_final_steps = kwargs.pop('heun_final_steps', 0)
# ...
for step_idx, (t, t_prev) in enumerate(t_pairs):
    if steps - step_idx <= heun_final_steps:
        # Heun: predictor-corrector
        pred_v1 = self._inference_model(model, sample, t, cond, **kwargs)
        x_pred = sample - (t - t_prev) * pred_v1
        pred_v2 = self._inference_model(model, x_pred, t_prev, cond, **kwargs)
        sample = sample - (t - t_prev) * (pred_v1 + pred_v2) / 2
    else:
        # Euler
        out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
        sample = out.pred_x_prev
```

**Difficulty**: Easy (10 lines). **Retraining**: No. **Expected impact**: +2-5% detail quality on final mesh geometry and texture sharpness.

#### ABM-Solver (Adams-Bashforth-Moulton, Multi-Step)

**Paper**: "Adams Bashforth Moulton Solver for Inversion and Editing in Rectified Flow" (arXiv:2503.16522)

A multi-step predictor-corrector method that achieves second-order accuracy by reusing velocity evaluations from previous timesteps, avoiding the extra NFE cost of Heun:

```
# Adams-Bashforth (predictor, uses v from previous step)
x_pred = x_t - dt * (3/2 * v_t - 1/2 * v_{t+dt})

# Adams-Moulton (corrector)
v_pred = model(x_pred, t - dt)
x_{t-dt} = x_t - dt * (v_t + v_pred) / 2
```

**Cost**: 1 NFE per step (same as Euler, after the first step) for the predictor; 2 NFE with correction.
**Key advantage**: Adaptive step size adjustment -- larger steps where the trajectory is straight, smaller steps where it curves.

**Application to TRELLIS.2**: Requires caching the velocity from the previous step. Straightforward to implement in the sample loop.

**Difficulty**: Easy-Medium (15-20 lines). **Retraining**: No. **Expected impact**: +2-4% quality with same NFE budget as Euler.

#### Flow-Solver (Multi-Step with Previous Step Leverage)

**Paper**: "Leveraging Previous Steps: A Training-free Fast Solver for Flow Diffusion" (arXiv:2411.07627)

Uses Taylor expansion + polynomial interpolation of cached previous-step velocities to approximate higher-order derivatives without extra NFE. Reports FID improvements from 13.79 to 6.75 on CIFAR-10 at NFE=10. Outperforms Heun on Stable Diffusion 3.

**Application to TRELLIS.2**: Cache previous velocity predictions and use polynomial interpolation for the current step. The number of cached steps determines the approximation order.

**Difficulty**: Medium (requires careful caching with SparseTensor). **Retraining**: No. **Expected impact**: +3-6% quality at same NFE.

#### SDM: Adaptive Solver with Wasserstein-Bounded Timesteps

**Paper**: "Formalizing the Sampling Design Space of Diffusion-Based Generative Models via Adaptive Solvers and Wasserstein-Bounded Timesteps" (arXiv:2602.12624)

**Key result**: Achieves FID 1.93 on CIFAR-10 by:
1. Using Euler for early (high-noise) steps where dynamics are simple
2. Progressively deploying higher-order solvers for later steps
3. Optimizing timestep schedules via Wasserstein distance bounds

The Wasserstein-bounded strategy automatically concentrates more steps near t=0 where detail is formed, providing a *principled justification* for TRELLIS.2's existing `rescale_t` parameter which already compresses more steps near t=0.

**Application to TRELLIS.2**: The `rescale_t` parameter already implements a rudimentary version of this. The formula in `flow_euler.py`:
```python
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
```
maps uniform spacing to a schedule that concentrates steps at low t. Higher `rescale_t` values (3.0-5.0) already push more steps toward the detail-forming regime. The SDM paper suggests that **combining this with a Heun solver for the final steps** is the optimal strategy.

**Difficulty**: Already partially implemented. **Retraining**: No.

### 1.3 Stochastic Sampling for Flow Matching

#### SDE Formulation of Flow Models

**Paper**: "Stochastic Sampling from Deterministic Flow Models" (arXiv:2410.02217)

Converts the deterministic flow ODE into a family of SDEs with the same marginal distributions:

```
dx = v(x, t) dt + g(t) dW
```

where `g(t)` is the diffusion coefficient controlling stochasticity. Key findings:
- Stochastic samplers tend to produce **better FIDs** than deterministic ones
- The diffusion coefficient provides a "knob" for diversity vs. quality
- Non-singular (partially stochastic) samplers outperform both fully deterministic and fully stochastic

**Why this helps detail**: Stochasticity during sampling can help the trajectory explore nearby modes, potentially finding sharper, more detailed solutions that the deterministic ODE's straight-line path misses. However, too much noise destroys coherence.

**Application to TRELLIS.2**: Add optional noise injection during the ODE sampling:
```python
# After Euler step
noise_level = diffusion_coeff * math.sqrt(t - t_prev)
sample = out.pred_x_prev + noise_level * torch.randn_like(sample)
```

The optimal `diffusion_coeff` would need to be tuned per stage (lower for shape, higher for texture where diversity of texture patterns is desirable).

**Difficulty**: Easy (5 lines). **Retraining**: No. **Expected impact**: +1-3% texture diversity/detail, risk of coherence loss if over-applied.

### 1.4 Summary: Solver Recommendations for TRELLIS.2

| Method | NFE Cost | Detail Impact | Difficulty | Priority |
|--------|----------|---------------|------------|----------|
| Heun final K steps | +25-50% | +2-5% | Easy | **HIGH** |
| ABM-Solver | +0-50% | +2-4% | Easy-Med | Medium |
| Flow-Solver (cache) | +0% | +3-6% | Medium | **HIGH** |
| Higher rescale_t | +0% | +1-2% | Trivial | Already done |
| Stochastic injection | +0% | +1-3% | Easy | Medium |
| SDM adaptive solver | +0% | +3-5% | Medium | Medium |

**Top recommendation**: Implement **Heun for the final 4 steps** combined with **Flow-Solver caching** for the early steps. This gives the best quality-compute tradeoff for detail generation.

---

## 2. Guidance Techniques for Detail

### 2.1 The Guidance-Detail Paradox

Standard CFG enhances prompt alignment but introduces artifacts at high guidance scales:
- **Oversaturation**: Colors become unnaturally vivid
- **Mode collapse**: Details converge to "average" textures
- **Spatial frequency bias**: CFG amplifies low-frequency structure more than high-frequency detail

TRELLIS.2 already implements standard CFG, CFG-Zero*, APG, and guidance rescale. The question is: what guidance specifically *enhances fine detail*?

### 2.2 Frequency-Decoupled Guidance (FDG)

**Paper**: "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales" (arXiv:2506.19713, June 2025)
**Venue**: NeurIPS 2025 submission

**Core insight**: Standard CFG applies the same guidance strength to all spatial frequencies. This is suboptimal because:
- **Low-frequency components** (global structure, color palette) need moderate guidance for condition alignment
- **High-frequency components** (texture detail, edges, fine patterns) benefit from *stronger* guidance

FDG decomposes the velocity prediction into frequency bands using **Laplacian pyramids**, then applies separate guidance strengths:

```python
# Standard CFG
v_guided = w * v_cond + (1-w) * v_uncond

# FDG
v_low_cond, v_high_cond = laplacian_decompose(v_cond)
v_low_uncond, v_high_uncond = laplacian_decompose(v_uncond)
v_guided = (w_low * v_low_cond + (1-w_low) * v_low_uncond) +
           (w_high * v_high_cond + (1-w_high) * v_high_uncond)
```

**Application to TRELLIS.2**: The challenge is that TRELLIS.2's velocity predictions are **sparse tensors** (not regular grids), so standard Laplacian pyramids don't directly apply. However, for the **texture SLAT** stage (which predicts PBR channels), a frequency decomposition can be approximated by:

1. **For dense stages** (sparse structure flow): Apply directly on the dense 3D grid
2. **For sparse stages**: Convert sparse velocity to dense, decompose, convert back; OR use a learned decomposition via the existing windowed attention (windows = local/high-freq, global attention = low-freq)

A simpler approximation: Apply a Gaussian blur to the velocity field and subtract to get high-frequency component:
```python
v_low = gaussian_blur_sparse(v_cond, sigma=2.0)
v_high = v_cond - v_low
# Apply separate guidance
```

**Difficulty**: Medium (sparse frequency decomposition is non-trivial). **Retraining**: No. **Expected impact**: +3-5% texture detail quality. **Code**: ComfyUI implementation at github.com/sdtana/ComfyUI-FDG.

### 2.3 Perturbed Attention Guidance (PAG) and Variants

#### PAG (ECCV 2024)

**Paper**: "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" (arXiv:2403.17377)

Replaces self-attention maps with identity matrices to create a "degraded structure" prediction, then guides away from it:

```
v_pag = v_model + w_pag * (v_model - v_degraded)
```

This specifically enhances *structural coherence* -- the relationship between parts of the generated object. In 3D generation, this translates to more consistent geometry and texture patterns.

**For TRELLIS.2**: Would require modifying the `SparseMultiHeadAttention` to optionally return identity-attention outputs. The sparse windowed attention (`sparse_windowed_scaled_dot_product_self_attention`) can be modified by replacing the attention computation with an identity mapping (averaging within windows). However, this requires an extra forward pass per step.

#### HeadHunter: Fine-Grained PAG (2025)

**Paper**: "Fine-Grained Perturbation Guidance via Attention Head Selection" (arXiv:2506.10978)

Discovery: Different attention heads control different visual properties (structure, style, texture). **SoftPAG** linearly interpolates each selected head's attention map toward identity, providing continuous control without binary on/off.

**Key for detail**: Identify which attention heads in TRELLIS.2's sparse transformer control *texture detail* vs. *global structure* and selectively perturb only the structure heads, preserving texture detail heads.

**Difficulty**: Hard (requires head analysis + selective perturbation in sparse attention). **Retraining**: No. **Expected impact**: +2-4% if right heads identified.

#### Normalized Attention Guidance (NAG, NeurIPS 2025)

**Paper**: "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models" (arXiv:2505.21179)
**Code**: github.com/ChenDarYen/Normalized-Attention-Guidance

Applies extrapolation in attention space with L1-based normalization:
- Works across UNet and DiT architectures
- Universal: improves both conditional and unconditional generation
- Minimal computational overhead
- "Higher scales reveal sharper structure and more fine-grained details"

**Key advantage over PAG**: L1 normalization constrains feature deviation, preventing the out-of-manifold drift that can occur with raw attention perturbation. This is particularly important for 3D generation where drift causes geometry artifacts.

**Difficulty**: Medium (adapt to sparse attention). **Retraining**: No. **Expected impact**: +2-4% detail and coherence.

#### Token Perturbation Guidance (TPG, NeurIPS 2025)

**Paper**: "Token Perturbation Guidance for Diffusion Models" (arXiv:2506.10036)
**Code**: github.com/TaatiTeam/Token-Perturbation-Guidance

Applies **norm-preserving shuffling** to intermediate token embeddings:
```python
# Shuffle tokens along spatial dimension (orthogonal perturbation)
perturbed_tokens = tokens[random_permutation]
v_perturbed = model(perturbed_tokens, t)
v_tpg = v_model + w_tpg * (v_model - v_perturbed)
```

The shuffling disrupts local spatial correlations while preserving global statistics. Nearly 2x FID improvement for unconditional generation.

**Application to TRELLIS.2**: Particularly natural for sparse tensors -- shuffle the sparse token ordering (which is already somewhat arbitrary in SparseTensor). The perturbation disrupts local 3D spatial patterns, guiding toward more coherent local-global structure.

**Difficulty**: Easy (shuffle sparse tensor feats by permuting indices). **Retraining**: No. **Expected impact**: +2-3% structural detail.

#### Smoothed Energy Guidance (SEG, NeurIPS 2024)

**Paper**: "Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention" (arXiv:2408.00760)
**Code**: github.com/SusungHong/SEG-SDXL

Blurs attention weights (Gaussian kernel on query projections) to reduce energy curvature:
- "Consistently produces sharper details, more realistic textures"
- Does not make images grayish (unlike PAG at high scales)
- Continuously controllable via Gaussian kernel parameter

**Application to TRELLIS.2**: Apply Gaussian blur to Q projections in the sparse self-attention. In windowed attention, this means blurring within each window.

**Difficulty**: Medium (blur in sparse attention). **Retraining**: No. **Expected impact**: +2-3% texture sharpness.

### 2.4 Entropy Rectifying Guidance (ERG, NeurIPS 2025)

**Paper**: "Entropy Rectifying Guidance for Diffusion and Flow Models" (arXiv:2504.13987)

Uses temperature-scaled softmax in attention layers to create weak/strong prediction branches:

```python
# Strong branch (low temperature -> sharp attention)
attn_strong = softmax(QK^T / (sqrt(d) * tau_strong))
# Weak branch (high temperature -> diffuse attention)
attn_weak = softmax(QK^T / (sqrt(d) * tau_weak))

# Guidance
v_erg = v_strong + w_erg * (v_strong - v_weak)
```

**Key advantages for TRELLIS.2**:
1. Works explicitly with flow matching models (tested on DiT)
2. No separate unconditional forward pass needed -- halves compute vs. CFG
3. Can be **combined with CFG** for additive improvements
4. The attention mechanism in TRELLIS.2's `ModulatedSparseTransformerCrossBlock` uses standard softmax that can be temperature-scaled

**Theoretical connection**: Low-temperature attention concentrates on the most relevant tokens (detailed features), while high-temperature attention distributes uniformly (blurred features). The guidance pushes toward the detailed extreme.

**Difficulty**: Medium (temperature parameter in sparse attention). **Retraining**: No. **Expected impact**: +3-5% detail + reduced compute if replacing separate uncond pass.

### 2.5 Sliding Window Guidance (M-SWG, BMVC 2025)

**Paper**: "Guiding a Diffusion Model with Itself Using Sliding Windows" (arXiv:2411.10257)

Restricts the model's receptive field via sliding window crops, then guides toward the full-receptive-field prediction:

```
v_restricted = model_with_restricted_receptive_field(x_t, t)
v_swg = v_model + w_swg * (v_model - v_restricted)
```

This upweights long-range spatial dependencies, improving global coherence. Achieves state-of-the-art FD-DINOv2 on ImageNet with EDM2-XXL and DiT-XL.

**Application to TRELLIS.2**: TRELLIS.2 already uses **windowed self-attention** with shifted windows (Swin-style). M-SWG can be implemented by:
1. Using a smaller window size for the "restricted" branch
2. Computing guidance as the difference between normal and small-window predictions

This is naturally aligned with TRELLIS.2's existing architecture -- the windowed attention *already* restricts receptive field, and varying the window size gives the restricted branch for free.

**Difficulty**: Medium (requires modifying window size per-step or additional forward pass). **Retraining**: No. **Expected impact**: +2-4% global coherence and detail consistency.

### 2.6 PLADIS (ICCV 2025)

**Paper**: "Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity" (arXiv:2503.07677)

Computes both sparse and dense attention within cross-attention modules and uses the difference as a guidance signal:

```
out = out_dense + lambda * (out_dense - out_sparse)
```

Compatible with all existing guidance methods. Works with both U-Net and Transformer architectures. No extra inference paths needed.

**Application to TRELLIS.2**: TRELLIS.2's `ModulatedSparseTransformerCrossBlock` already has both windowed self-attention (sparse) and full cross-attention (dense). PLADIS suggests using the *difference* between these as a signal -- but in TRELLIS.2 they attend over different key/value sets (3D self vs. 2D cross), so the analogy is not direct. However, computing sparse vs. dense *self-attention* within the same block is feasible.

**Difficulty**: Medium. **Retraining**: No. **Expected impact**: +2-3% text/condition alignment (detail alignment with input image).

### 2.7 Summary: Guidance Recommendations for TRELLIS.2

| Method | Extra NFE | Detail Impact | Difficulty | Priority |
|--------|-----------|---------------|------------|----------|
| FDG (freq-decoupled) | +0% | +3-5% | Medium | **HIGH** |
| ERG (entropy) | -50% (no uncond) | +3-5% | Medium | **HIGH** |
| NAG (normalized attn) | +100% (extra pass) | +2-4% | Medium | Medium |
| TPG (token shuffle) | +100% (extra pass) | +2-3% | Easy | Medium |
| PAG (perturbed attn) | +100% (extra pass) | +2-3% | Medium | Low-Med |
| SEG (smoothed energy) | +100% (extra pass) | +2-3% | Medium | Low-Med |
| M-SWG (sliding window) | +100% (extra pass) | +2-4% | Medium | Medium |
| PLADIS | +0% | +2-3% | Medium | Medium |

**Top recommendation**: Implement **FDG** for frequency-aware guidance (stronger high-frequency guidance = more detail) + **ERG** to replace or supplement CFG with lower compute cost and better detail.

---

## 3. Multi-Resolution and Cascade Generation

### 3.1 TRELLIS.2's Current Cascade Architecture

TRELLIS.2 already implements a sophisticated cascade:

1. **Stage 1**: Sparse Structure Flow at 32x32x32 (512 pipeline) or 64x64x64 (1024 direct)
2. **Stage 2**: Shape SLAT Flow at 512 resolution (low-res) -> upsample 4x -> Shape SLAT Flow at 1024/1536 (high-res)
3. **Stage 3**: Texture SLAT Flow at 1024/1536 resolution

The cascade is implemented in `sample_shape_slat_cascade()` (lines 745-832 of `trellis2_image_to_3d.py`), where the low-res shape is generated, upsampled with `decoder.upsample()`, quantized to the target resolution, and then refined at high resolution.

### 3.2 XCube: Hierarchical Sparse Voxel Generation (CVPR 2024 Highlight)

**Paper**: "XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies" (arXiv:2312.03806)
**Code**: github.com/nv-tlabs/XCube (NVIDIA)

Uses hierarchical voxel latent diffusion, generating progressively from coarse to fine (up to 1024^3). Built on the VDB data structure for memory-efficient sparse operations.

**Key insight**: Each cascade level uses its own independent diffusion model trained at that resolution, with conditioning from the previous level. The coarse level establishes structure; fine levels add geometric detail.

**Application to TRELLIS.2**: TRELLIS.2 already follows this paradigm. The improvement opportunity is in **how information flows between cascade levels**:
- Currently: Upsample geometry, quantize to grid, generate fresh noise at high-res
- Better: Use the upsampled low-res prediction as a *partial noise initialization* (SDEdit-style) rather than pure noise, preserving structural coherence while allowing detail generation

### 3.3 ShapeShifter: Multiscale Sparse Point-Voxel Diffusion (CVPR 2025)

**Paper**: "ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion" (arXiv:2502.02187)

**Key innovation**: All cascade levels are *independent and trainable in parallel*. Each level:
1. Takes the output of the previous level
2. Prunes inactive voxels (saves compute on empty space)
3. Upsamples with a level-specific upsampler
4. Adds noise at an intermediate level (not full noise)
5. Denoises with a level-specific diffusion model

**The pruning step is crucial**: By removing voxels far from the surface, each subsequent level focuses compute on *surface detail* rather than empty space. This is directly relevant to TRELLIS.2 where many sparse tokens represent near-surface regions.

**Application to TRELLIS.2**: The pruning concept could be applied to TRELLIS.2's cascade by filtering out tokens that are far from the predicted surface after the low-res stage, concentrating the high-res compute budget on surface-adjacent voxels.

**Difficulty**: Medium (requires surface proximity estimation). **Retraining**: Yes (for optimal results), but a training-free variant using low-res surface prediction is possible.

### 3.4 SparseFlex: High-Resolution Sparse Isosurface (ICCV 2025 Oral)

**Paper**: "SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling" (arXiv:2503.21732)
**Code**: github.com/VAST-AI-Research/TripoSF

Combines FlexiCubes accuracy with sparse voxel structure. Key results:
- 82% reduction in Chamfer Distance vs. previous methods
- 88% increase in F-score
- Operates at 1024^3 resolution with frustum-aware training

**Key technique**: Frustum-aware sectional voxel training -- only activates voxels visible from the current rendering viewpoint, dramatically reducing memory. This enables training at very high resolutions.

**Application to TRELLIS.2**: SparseFlex's isosurface extraction could replace or augment TRELLIS.2's current FlexiCubes/FDG mesh extraction, potentially producing meshes with finer geometric detail. The frustum-aware approach could also be used during texture generation to allocate more compute to visible regions.

**Difficulty**: Hard (replacement of mesh extraction pipeline). **Retraining**: Requires VAE retraining.

### 3.5 DetailGen3D: Coarse-to-Fine Flow Refinement

**Paper**: "DetailGen3D: Generative 3D Geometry Enhancement via Data-Dependent Flow" (arXiv:2411.16820)
**Code**: github.com/VAST-AI-Research/DetailGen3D (released)

**This is the most directly applicable cascade improvement**. Instead of noise-to-data flow, it models *coarse-to-fine* flow:

```
z_t = (1-t) * z_coarse + t * z_fine
v(z_t, t) predicts (z_fine - z_coarse)
```

Key features:
- **Data-dependent rectified flow**: Uses optimal transport coupling between coarse and fine geometry
- **Token matching**: Nearest-neighbor spatial correspondence ensures local refinement
- **DiT architecture**: 24 blocks, 768 width, 368M parameters
- **Fast**: "A few seconds" per refinement
- **User study**: 94% prefer refined geometry

**Application to TRELLIS.2**: Can be used as a *post-processing step* on TRELLIS.2's generated shape:
1. Generate coarse shape with TRELLIS.2's existing pipeline
2. Encode to DetailGen3D's latent space
3. Run the coarse-to-fine flow to add geometric detail
4. Decode to high-resolution mesh

This is the **theoretically optimal approach** for adding geometric detail because it specifically trains the flow to model the coarse-to-fine transformation, rather than hoping the general-purpose flow captures it.

**Difficulty**: Medium (integration of separate model). **Retraining**: Uses pretrained DetailGen3D model. **Expected impact**: +5-10% geometric detail.

### 3.6 LATTICE: VoxSet + Progressive Training

**Paper**: "LATTICE: Democratize High-Fidelity 3D Generation at Scale" (arXiv:2512.03052)
**Code**: github.com/Zeqiang-Lai/LATTICE

Uses VoxSet (sparse voxel set tokenizer) with rectified flow transformer. Key insight: anchoring latent vectors to a coarse voxel grid injects position information via positional embedding, which is "proven essential in model scaling."

**For detail**: LATTICE uses progressive training (increasing resolution during training) which teaches the model to add detail at each scale. While TRELLIS.2's model is frozen, the concept suggests that **progressive denoising** (starting at lower effective resolution and gradually increasing) could help at inference time.

### 3.7 VoxSet: Fixed-Length Sparse Voxel Tokenizer (ICLR 2026)

**Paper**: "VoxSet: Sparse Voxel Set Tokenizer for 3D Shape Generation" (OpenReview, ICLR 2026)

Combines sparse voxels (outer layers, for surface detail) with vector set bottleneck (inner layers, for compression). The key innovation: sparse voxels in the outer layers **explicitly capture fine surface details** while the compressed bottleneck ensures efficient generation.

This validates TRELLIS.2's SLAT approach conceptually but suggests that the outer decoder layers should have higher spatial resolution than the inner generation layers.

### 3.8 Cascade Improvement Recommendations

| Method | Detail Impact | Difficulty | Priority |
|--------|---------------|------------|----------|
| DetailGen3D post-refinement | +5-10% geom | Medium | **HIGH** |
| SDEdit-style cascade init | +2-4% | Easy | **HIGH** |
| Surface-aware token pruning | +1-3% | Medium | Medium |
| SparseFlex mesh extraction | +3-5% | Hard | Low |
| Progressive denoising | +1-2% | Easy | Low |

**Top recommendation**: Integrate **DetailGen3D** as a post-processing geometry refinement step. For the existing cascade, implement **SDEdit-style initialization** at the high-res stage (start from partially noised upsampled prediction rather than pure noise).

---

## 4. Attention Mechanisms for Detail

### 4.1 TRELLIS.2's Current Attention Architecture

TRELLIS.2's `ModulatedSparseTransformerCrossBlock` uses:
1. **Windowed self-attention** (Swin-style): Local context within 3D windows of size `window_size`, with optional shifted windows
2. **Full cross-attention**: 3D sparse tokens attend to all 2D image feature tokens (DINOv3)
3. **FFN**: Standard feed-forward with adaptive layer norm modulation

The windowed self-attention limits each token's receptive field to its local 3D neighborhood. While this is memory-efficient, it means that fine detail in one part of the object cannot directly influence fine detail in another part (only through cascaded layers).

### 4.2 Direct3D-S2: Spatial Sparse Attention (NeurIPS 2025)

**Paper**: "Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention" (arXiv:2505.17412)
**Code**: github.com/DreamTechAI/Direct3D-S2

Introduces Spatial Sparse Attention (SSA) that achieves:
- 3.9x speedup in forward pass
- 9.6x speedup in backward pass
- Training at 1024^3 with only 8 GPUs

**Key technique**: Instead of dense attention over all sparse tokens or windowed attention, SSA identifies *spatially meaningful* groups of tokens and performs attention within these groups. This is more efficient than full attention while capturing longer-range dependencies than windowed attention.

**Application to TRELLIS.2**: Could replace the current windowed self-attention with SSA for better detail coherence across distant but spatially related tokens (e.g., symmetric features of an object). However, this requires modifying the attention kernel, which is tightly integrated with flash_attn.

**Difficulty**: Hard (attention kernel modification). **Retraining**: Ideally yes. **Expected impact**: +2-4% long-range detail consistency.

### 4.3 SpArSe-Up: Learnable Sparse Upsampling (2025)

**Paper**: "SpArSe-Up: Learnable Sparse Upsampling for 3D Generation with High-Fidelity Textures" (arXiv:2509.23646)

Key techniques for detail in sparse representations:
1. **Surface anchoring**: Constrains voxels to mesh surface, removing >70% redundant voxels
2. **View-domain partitioning**: Supervises only visible local patches, reducing compute

**Application to TRELLIS.2**: The surface anchoring concept is directly applicable to TRELLIS.2's cascade upsampling step. After low-res shape prediction, constrain the high-res voxels to the predicted surface rather than filling the entire bounding volume.

### 4.4 Multi-Scale Attention Strategies

**Insight from literature**: Several papers (MHVTNet, 3DGTN, XCube) converge on the idea that **dual attention** -- combining local (windowed/voxel-level) and global (downsampled/pooled) attention -- captures both fine detail and global structure better than either alone.

**Application to TRELLIS.2** (training-free): Insert a global attention step every K windowed attention steps by:
1. Downsampling the sparse tokens (e.g., max-pool within each window)
2. Running full attention on the downsampled tokens
3. Broadcasting the global context back to the original tokens

This provides long-range coherence without the quadratic cost of full attention on all tokens.

**Difficulty**: Hard (requires architectural modification). **Retraining**: Preferably. **Expected impact**: +3-5% detail coherence.

### 4.5 Summary: Attention Recommendations

| Method | Detail Impact | Difficulty | Priority |
|--------|---------------|------------|----------|
| Multi-scale dual attention | +3-5% | Hard | Low (needs retraining) |
| SSA from Direct3D-S2 | +2-4% | Hard | Low (needs retraining) |
| Surface-aware token filtering | +1-3% | Medium | Medium |
| Global attention every K layers | +2-3% | Hard | Low (needs retraining) |

**These modifications generally require retraining and are not recommended for the frozen-weight setting.** They are documented here for completeness and future training runs.

---

## 5. Training-Free Detail Enhancement

### 5.1 Inference-Time Scaling (Compute Search)

#### Noise Search (CVPR 2025)

**Paper**: "Scaling Inference Time Compute for Diffusion Models" (Ma et al., CVPR 2025)

**Core idea**: Some initial noise seeds produce better results than others. By generating multiple candidates and selecting the best using a verifier, quality improves with more compute.

Two axes of design:
1. **Verifiers**: Models that score generation quality (CLIP score, aesthetic predictor, VLM)
2. **Search algorithms**: Best-of-N, particle filtering, trajectory search

**Application to TRELLIS.2**: Generate N=4-8 shape SLATs from different noise seeds, render each, score with a quality verifier (DINOv2 similarity to input, or CLIP aesthetic score), keep the best. This is particularly effective for the **sparse structure stage** where the compute cost per sample is low.

**Difficulty**: Easy (generate multiple, pick best). **Retraining**: No. **Expected impact**: +3-8% (depending on N and verifier quality). **Compute**: Linear in N.

#### LoTTS: Localized Test-Time Scaling (2025)

**Paper**: "Scale Where It Matters: Training-Free Localized Scaling for Diffusion Models" (arXiv:2511.19917)

Instead of improving the entire generation, identifies **defective regions** and resamples only those:
1. Contrast cross/self-attention signals under quality-aware prompts ("high-quality" vs. "low-quality")
2. Detect low-quality regions from attention differences
3. Perturb only defective regions at intermediate timesteps
4. Run a few global denoising steps for coherence

Reduces GPU cost by 2-4x vs. Best-of-N while matching or exceeding quality.

**Application to TRELLIS.2**: Adapt the region detection to 3D: use the attention weights from cross-attention (image features -> sparse tokens) to identify tokens that have low attention confidence (high entropy). These tokens likely represent poorly conditioned regions. Re-noise and re-denoise only these tokens.

**Difficulty**: Hard (3D region detection + selective re-denoising in sparse tensors). **Retraining**: No. **Expected impact**: +2-5% on problem regions.

#### Reflect-DiT (ICCV 2025)

**Paper**: "Reflect-DiT: Inference-Time Scaling for Text-to-Image Diffusion Transformers via In-Context Reflection" (arXiv:2503.12271)
**Code**: github.com/jacklishufan/Reflect-DiT

Uses a VLM to critique generations and feeds the critique back to the DiT for iterative refinement. +0.19 on GenEval benchmark.

**Application to TRELLIS.2**: After generating a 3D mesh, render it, use a VLM (e.g., GPT-4V or LLaVA) to identify quality issues, then regenerate with modified parameters. This is more of a *system-level* approach than an algorithm.

**Difficulty**: Medium (VLM integration). **Retraining**: No. **Expected impact**: +2-5% with good VLM.

#### Flow Map Trajectory Tilting (FMTT, 2025)

**Paper**: "Test-time Scaling of Diffusions with Flow Maps" (arXiv:2511.22688)

Uses flow maps (distilled single-step predictors) as look-aheads to optimize the trajectory toward a reward:
1. At each timestep, use the flow map to predict the final sample in one step
2. Compute a reward on the predicted final sample
3. Use the reward gradient to tilt the trajectory

**Key advantage**: The flow map prediction is much more accurate than the standard denoiser prediction (especially early in sampling), giving useful gradients from the start.

**Application to TRELLIS.2**: Would require a distilled single-step version of TRELLIS.2 (or use the existing model with aggressive timestep skipping as an approximation). The reward could be 2D rendering similarity to the input image.

**Difficulty**: Hard (flow map distillation). **Retraining**: Yes (for flow map). **Expected impact**: +5-10% with good reward.

### 5.2 SDEdit-Style Refinement on Generated Output

**Technique**: After generating a 3D model, add noise to the latent at an intermediate timestep (e.g., t=0.3), then re-denoise. The re-denoising path can diverge from the original, potentially finding a higher-quality solution while preserving the overall structure.

**Elevate3D's HFS-SDEdit variant**: Decompose into low/high frequency bands. Let low-frequency evolve freely (finds better overall appearance), constrain high-frequency to match original (preserves structure).

**Application to TRELLIS.2**: After the shape SLAT flow completes:
1. Re-encode the result to an intermediate timestep: `x_t = (1-t)*x_0 + (sigma_min + (1-sigma_min)*t)*eps`
2. Re-run the flow from t to 0 with potentially different guidance/steps
3. Use the re-denoised result if quality improves

This is training-free and can be applied to any stage of the pipeline.

**Difficulty**: Easy (5-10 lines in the sample loop). **Retraining**: No. **Expected impact**: +1-3% on texture quality, slight risk of divergence.

### 5.3 Render-and-Compare Texture Optimization

**Technique**: After mesh extraction, optimize the UV texture by rendering from multiple views and comparing to the input image(s) via differentiable rendering.

TRELLIS.2 already has nvdiffrast installed. The optimization loop:

```python
for iteration in range(100):
    rendered = nvdiffrast.render(mesh, camera_params)
    loss = perceptual_loss(rendered, target_image) + tv_loss(texture)
    loss.backward()
    optimizer.step()
```

**Application to TRELLIS.2**: This is the most reliable way to ensure the texture matches the input image. Combined with multi-view rendering (back, sides), it can also add detail from hallucinated views.

**Difficulty**: Medium (render pipeline setup). **Retraining**: No. **Expected impact**: +3-5% texture fidelity.

### 5.4 Summary: Training-Free Enhancement Recommendations

| Method | Detail Impact | Compute | Difficulty | Priority |
|--------|---------------|---------|------------|----------|
| Noise search (best-of-N) | +3-8% | Nx | Easy | **HIGH** |
| SDEdit refinement | +1-3% | +30-50% | Easy | Medium |
| Render-and-compare | +3-5% | +2-5 min | Medium | **HIGH** |
| LoTTS localized | +2-5% | +50% | Hard | Low-Med |
| Reflect-DiT (VLM loop) | +2-5% | +variable | Medium | Low |
| FMTT trajectory tilting | +5-10% | +variable | Hard | Low |

**Top recommendations**: (1) **Best-of-N noise search** for the structure/shape stages (cheap, high impact); (2) **Render-and-compare texture optimization** as a post-processing step for final texture quality.

---

## 6. Synthesis: Recommended Implementation Stack

Based on the full survey, here is the recommended stack of techniques for maximizing detail generation in TRELLIS.2, ordered by priority:

### Tier 1: Quick Wins (1-2 days each, high confidence)

1. **Heun solver for final steps**
   - Add `heun_final_steps` parameter to `FlowEulerSampler.sample()`
   - Use Heun for the last 4 steps of the 16-step ODE
   - Expected: +2-5% geometric detail quality
   - Files: `trellis2/pipelines/samplers/flow_euler.py`

2. **Best-of-N noise search for sparse structure**
   - Generate N=4-8 sparse structures, render each, pick best by DINOv2 similarity
   - Expected: +3-8% overall quality
   - Files: `trellis2/pipelines/trellis2_image_to_3d.py` (pipeline `run()`)

3. **FDG: Frequency-Decoupled Guidance**
   - Implement Laplacian decomposition of velocity predictions
   - Apply w_high > w_low (e.g., w_high=1.5*w, w_low=0.7*w)
   - For sparse tensors: approximate via Gaussian blur on dense projection
   - Expected: +3-5% texture detail
   - Files: `trellis2/pipelines/samplers/cfg_utils.py`, `classifier_free_guidance_mixin.py`

### Tier 2: Medium-Effort Improvements (3-5 days each)

4. **ERG: Entropy Rectifying Guidance**
   - Add temperature scaling to sparse attention softmax
   - Replace CFG's unconditional pass with temperature-modulated weak branch
   - Expected: +3-5% detail + 50% compute reduction
   - Files: `trellis2/modules/sparse/attention/windowed_attn.py`, `full_attn.py`

5. **Flow-Solver with velocity caching**
   - Cache previous-step velocities in the sample loop
   - Use polynomial interpolation for current step prediction
   - Expected: +3-6% quality at same NFE
   - Files: `trellis2/pipelines/samplers/flow_euler.py`

6. **DetailGen3D post-processing**
   - Integrate as optional geometry refinement after mesh generation
   - Expected: +5-10% geometric detail
   - Files: New integration script; github.com/VAST-AI-Research/DetailGen3D

7. **Render-and-compare texture optimization**
   - Post-processing step using nvdiffrast for multi-view texture refinement
   - Expected: +3-5% texture fidelity to input
   - Files: New post-processing function in `o_voxel/postprocess.py`

### Tier 3: Research Explorations (1-2 weeks each)

8. **SDEdit cascade initialization**
   - Initialize high-res stage from partially noised upsampled low-res, not pure noise
   - Expected: +2-4% structural consistency

9. **Stochastic flow sampling**
   - Optional noise injection during ODE integration for texture diversity
   - Expected: +1-3% texture variety/detail

10. **TPG/NAG attention guidance**
    - Token perturbation or normalized attention as additional guidance signals
    - Expected: +2-4% detail coherence

### Theoretical Detail Budget

Combining Tier 1 + Tier 2 techniques (assuming partial additivity with ~50% overlap in effects):

| Component | Individual Impact | Cumulative (after overlap) |
|-----------|------------------|---------------------------|
| Heun final steps | +3.5% | +3.5% |
| Best-of-N search | +5.5% | +7.5% |
| FDG guidance | +4.0% | +9.5% |
| ERG guidance | +4.0% | +11.5% |
| Flow-Solver cache | +4.5% | +14.0% |
| DetailGen3D | +7.5% | +18.0% |
| Render-compare | +4.0% | +20.0% |

**Total estimated improvement: ~15-20% detail quality** with all Tier 1+2 techniques, though actual gains depend on the evaluation metric, test objects, and interaction effects between techniques.

---

## 7. References

### Flow Matching Solvers
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al., ICLR 2023
- [SDM: Adaptive Solvers and Wasserstein-Bounded Timesteps](https://arxiv.org/abs/2602.12624) - February 2026
- [ABM-Solver for Rectified Flow](https://arxiv.org/abs/2503.16522) - March 2025
- [Flow-Solver: Leveraging Previous Steps](https://arxiv.org/abs/2411.07627) - November 2024
- [Rectified Diffusion: Straightness Is Not Your Need](https://arxiv.org/abs/2410.07303) - ICLR 2025
- [Stochastic Sampling from Deterministic Flow Models](https://arxiv.org/abs/2410.02217) - October 2024
- [Flow Diverse and Efficient: Momentum Flow Matching](https://arxiv.org/abs/2506.08796) - June 2025
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) - SD3

### Guidance Techniques
- [CFG-Zero*](https://arxiv.org/abs/2503.18886) - Fan et al., March 2025
- [APG: Eliminating Oversaturation](https://arxiv.org/abs/2410.02416) - ICLR 2025
- [FDG: Frequency-Decoupled Guidance](https://arxiv.org/abs/2506.19713) - June 2025
- [PAG: Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377) - ECCV 2024
- [HeadHunter: Fine-Grained PAG](https://arxiv.org/abs/2506.10978) - June 2025
- [NAG: Normalized Attention Guidance](https://arxiv.org/abs/2505.21179) - NeurIPS 2025
- [TPG: Token Perturbation Guidance](https://arxiv.org/abs/2506.10036) - NeurIPS 2025
- [SEG: Smoothed Energy Guidance](https://arxiv.org/abs/2408.00760) - NeurIPS 2024
- [ERG: Entropy Rectifying Guidance](https://arxiv.org/abs/2504.13987) - NeurIPS 2025
- [M-SWG: Sliding Window Guidance](https://arxiv.org/abs/2411.10257) - BMVC 2025
- [PLADIS: Sparse vs Dense Attention Guidance](https://arxiv.org/abs/2503.07677) - ICCV 2025
- [CFG Manifold Projection for Flow Matching](https://arxiv.org/abs/2601.21892) - January 2026

### Multi-Resolution & Cascade
- [XCube: Sparse Voxel Hierarchies](https://arxiv.org/abs/2312.03806) - CVPR 2024 Highlight
- [ShapeShifter: Multiscale Point-Voxel Diffusion](https://arxiv.org/abs/2502.02187) - CVPR 2025
- [SparseFlex: High-Resolution Isosurface](https://arxiv.org/abs/2503.21732) - ICCV 2025 Oral
- [DetailGen3D: Data-Dependent Flow](https://arxiv.org/abs/2411.16820) - VAST, 2024-2025
- [LATTICE: VoxSet 3D Generation](https://arxiv.org/abs/2512.03052) - December 2025
- [VoxSet: Sparse Voxel Set Tokenizer](https://openreview.net/forum?id=7cLvFw1ZGu) - ICLR 2026
- [SpArSe-Up: Learnable Sparse Upsampling](https://arxiv.org/abs/2509.23646) - September 2025
- [Direct3D-S2: Spatial Sparse Attention](https://arxiv.org/abs/2505.17412) - NeurIPS 2025

### Attention Mechanisms
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Liu et al., ICCV 2021
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [DiT: Scalable Diffusion Transformers](https://arxiv.org/abs/2212.09748) - Peebles & Xie, ICCV 2023

### Inference-Time Scaling
- [Scaling Inference Time Compute for Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.pdf) - CVPR 2025
- [LoTTS: Localized Test-Time Scaling](https://arxiv.org/abs/2511.19917) - November 2025
- [Reflect-DiT: In-Context Reflection](https://arxiv.org/abs/2503.12271) - ICCV 2025
- [FMTT: Flow Map Trajectory Tilting](https://arxiv.org/abs/2511.22688) - November 2025

### 3D Quality Enhancement
- [Elevate3D: HFS-SDEdit](https://arxiv.org/abs/2507.11465) - SIGGRAPH 2025
- [PBR-SR: Texture Super-Resolution](https://arxiv.org/abs/2506.02846) - NeurIPS 2025
- [3DEnhancer: Multi-View Enhancement](https://arxiv.org/abs/2412.18565) - CVPR 2025
- [SuperCarver: Geometry Super-Resolution](https://arxiv.org/abs/2503.09439) - March 2025
- [Generative Detail Enhancement](https://arxiv.org/abs/2502.13994) - SIGGRAPH 2025 (NVIDIA)

---

## Appendix: TRELLIS.2 Architecture Quick Reference

### Relevant Code Locations

| Component | File |
|-----------|------|
| Flow sampler (Euler) | `trellis2/pipelines/samplers/flow_euler.py` |
| CFG implementation | `trellis2/pipelines/samplers/classifier_free_guidance_mixin.py` |
| CFG utilities (APG, Zero*) | `trellis2/pipelines/samplers/cfg_utils.py` |
| Guidance interval | `trellis2/pipelines/samplers/guidance_interval_mixin.py` |
| Sparse transformer blocks | `trellis2/modules/sparse/transformer/modulated.py` |
| Windowed self-attention | `trellis2/modules/sparse/attention/windowed_attn.py` |
| Full cross-attention | `trellis2/modules/sparse/attention/full_attn.py` |
| Image features (DINOv3) | `trellis2/modules/image_feature_extractor.py` |
| Pipeline (cascade) | `trellis2/pipelines/trellis2_image_to_3d.py` |
| Mesh post-processing | `o-voxel/o_voxel/postprocess.py` |

### Key Equations (TRELLIS.2 Flow Matching)

```
# Forward process: x_t = (1-t)*x_0 + (sigma_min + (1-sigma_min)*t)*eps
# Velocity field: v(x_t, t) = dx_t/dt

# Velocity to x_0:
x_0 = (1-sigma_min)*x_t - (sigma_min + (1-sigma_min)*t)*v

# Velocity to eps:
eps = (1-t)*v + x_t

# Euler step:
x_{t-dt} = x_t - dt * v(x_t, t)

# Rescaled timestep schedule:
t_rescaled = rescale_t * t / (1 + (rescale_t - 1) * t)
```

### Cascade Pipeline Flow (1024_cascade)

```
1. Image -> DINOv3 features (512, 1024 resolution)
2. Silhouette -> 32x32x32 sparse structure (SparseStructureFlowModel)
3. Sparse structure -> coords
4. Coords -> Shape SLAT at 512 (SLatFlowModel_512) -- LR stage
5. Shape SLAT -> upsample 4x -> quantize to HR resolution
6. HR coords -> Shape SLAT at 1024/1536 (SLatFlowModel_1024) -- HR stage
7. Shape SLAT -> decode to mesh (FlexiCubes/FDG)
8. Mesh coords -> Texture SLAT at 1024/1536 (SLatFlowModel_1024)
9. Texture SLAT -> decode to PBR textures
10. Mesh + PBR -> GLB (postprocess.py)
```
