# Silhouette Accuracy (A1) Improvement Research
## Training-Free Methods for TRELLIS.2 Shape-Image Alignment
### Date: 2026-02-23

**Status**: Current A1 = 81.4/100 (Dice IoU between input silhouette and best-matching rendered 3D view)
**Target**: A1 >= 90/100
**Constraint**: Frozen model weights (no retraining)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Method 1: Post-Generation Silhouette Deformation](#method-1-post-generation-silhouette-deformation)
3. [Method 2: Staged Best-of-N with Shape-Only Selection](#method-2-staged-best-of-n-with-shape-only-selection)
4. [Method 3: Noise Trajectory Optimization](#method-3-noise-trajectory-optimization)
5. [Method 4: FlowDPS Posterior Sampling with Silhouette Likelihood](#method-4-flowdps-posterior-sampling)
6. [Method 5: Sparse Structure Visual Hull Tightening](#method-5-sparse-structure-visual-hull-tightening)
7. [Method 6: Training-Free Guidance with Silhouette Reward](#method-6-training-free-guidance)
8. [Method 7: Multi-View Silhouette Constraints](#method-7-multi-view-silhouette-constraints)
9. [Method 8: Test-Time Mesh Refinement (GTR-style)](#method-8-test-time-mesh-refinement-gtr-style)
10. [Method 9: Noise Inversion Stability Selection](#method-9-noise-inversion-stability)
11. [Method 10: Dynamic Guidance for Shape Fidelity](#method-10-dynamic-guidance-for-shape-fidelity)
12. [Comparison Table](#comparison-table)
13. [Implementation Priority](#implementation-priority)
14. [References](#references)

---

## Executive Summary

Silhouette accuracy (A1) measures how well the generated 3D shape matches the input image's outline when rendered from the best-matching viewpoint. At 81.4/100, the primary failure modes are:

1. **Shape ambiguity**: The model generates plausible but incorrect back-side geometry
2. **Volumetric over-smoothing**: Fine protrusions (ears, limbs, tails) are under-represented in the sparse structure
3. **Sparse structure discretization**: 32^3 voxel grid limits thin feature fidelity
4. **Camera alignment**: The best-view search may not find the optimal viewpoint

This report surveys 10 methods organized by approach category. The recommended priority stack is:

| Priority | Method | Expected A1 Gain | Difficulty | Time Cost |
|----------|--------|-------------------|------------|-----------|
| 1 | Post-gen silhouette deformation (already implemented) | +2-6 pts | Easy (tune) | +3-8s |
| 2 | Staged Best-of-N (shape-only early pruning) | +3-5 pts | Easy | +3-8x gen |
| 3 | Sparse structure visual hull tightening | +1-3 pts | Medium | +0.5s |
| 4 | Dynamic guidance schedule (shape stage) | +1-3 pts | Easy | +0s |
| 5 | Noise trajectory optimization (shape stage) | +2-5 pts | Hard | +5-10x gen |
| 6 | FlowDPS posterior sampling | +3-8 pts | Hard | +2-4x gen |

---

## Method 1: Post-Generation Silhouette Deformation

### Overview
Already implemented in `trellis2/postprocessing/silhouette_corrector.py`. Renders the mesh from the input viewpoint, compares silhouette against reference alpha, optimizes vertex positions via gradient descent through nvdiffrast.

### Theory
Given mesh vertices V, faces F, and reference silhouette S_ref:

```
L = w_sil * BCE(render(V, F), S_ref, dt_weights)
  + w_lap * ||L(V) - L(V_orig)||^2
  + w_norm * max(0, -n_new . n_orig)
```

Where L(V) is the Laplacian coordinate (per-vertex average of neighbors subtracted from vertex position), and the distance-transform weights `dt_weights` emphasize boundary regions.

### Current Implementation Analysis

The `SilhouetteCorrector` at `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/postprocessing/silhouette_corrector.py` has several well-designed features:

**Strengths:**
- Scale-invariant comparison via crop-to-bbox matching V4 evaluator
- Auto-calibrated camera distance (binary search for coverage match)
- DT-weighted BCE loss (higher weight near boundaries)
- Laplacian regularization preserving mesh smoothness
- Normal consistency preventing face flips
- Max displacement clamping (0.02 of unit cube)
- Best-Dice tracking with rollback

**Weaknesses / Improvement Opportunities:**

1. **Single-view only**: Only optimizes from one camera angle (yaw=0, pitch=0.25). Geometry off-screen is unconstrained and can degrade.

2. **High Laplacian weight**: w_laplacian=50 is very conservative. Research (Palfinger 2022, MIT thesis on mesh regularization) shows that for silhouette fitting, w_laplacian=5-15 with cotangent weighting yields better results while preventing self-intersection.

3. **Max displacement too conservative**: 0.02 limits correction to ~2% of the bounding box. For protrusions that are significantly wrong (rotated arms, missing ears), 0.05-0.08 is needed.

4. **No IoU/Dice loss component**: BCE is the wrong loss for silhouette matching. Soft Dice loss `1 - 2*sum(p*t)/(sum(p)+sum(t))` directly optimizes the evaluation metric and has better gradient behavior at boundaries.

5. **No multi-resolution**: Operating at a single 512px resolution misses both coarse outline errors and fine boundary details.

6. **Not using SoftRas-style gradients**: `dr.antialias()` provides limited gradients only at visible silhouette edges. SoftRas (Shichen Liu, ICCV 2019) provides smoother gradients via soft rasterization, enabling long-range vertex movement.

### Recommended Improvements

```python
# Key changes to SilhouetteCorrector.correct():

# 1. Add Soft Dice loss (directly optimizes evaluation metric)
def _soft_dice_loss(self, rendered, target):
    p = rendered[0, :, :, 0]
    t = target
    intersection = (p * t).sum()
    return 1.0 - (2.0 * intersection + 1) / (p.sum() + t.sum() + 1)

# 2. Multi-resolution pyramid (coarse-to-fine)
for stage, (res, steps, lr, max_disp) in enumerate([
    (256, 30, 2e-3, 0.06),   # coarse: large movements
    (512, 50, 5e-4, 0.03),   # fine: precision
]):
    ...

# 3. Reduce Laplacian weight, add edge-length regularization
w_laplacian = 10.0  # was 50.0
w_edge = 5.0  # prevent triangle collapse

# 4. Multi-view regularization (render from 4 views, penalize extreme deformation)
views = [(0, 0.25), (np.pi/2, 0.25), (np.pi, 0.25), (3*np.pi/2, 0.25)]
for yaw, pitch in views:
    mask = self._render_silhouette(vertices, faces, ...)
    l_reg += variance_of_deformation_at_view(mask)
```

### Estimated Impact
- Current implementation (tuned): +1-3 pts on A1
- With Soft Dice + multi-resolution + relaxed constraints: +3-6 pts on A1
- Ceiling: ~88-90 A1 (limited by topology; can move vertices but cannot add/remove them)

### Flow Matching Compatibility
N/A -- post-processing, fully compatible.

### Computational Cost
80 steps at 512px: ~3-8 seconds on A100/Blackwell.

### Key Papers
- Palfinger (2022): "Continuous remeshing for inverse rendering"
- Shichen Liu (ICCV 2019): SoftRas differentiable renderer
- NVIDIA FlexiCubes (SIGGRAPH 2023): Flexible isosurface for gradient-based optimization

---

## Method 2: Staged Best-of-N with Shape-Only Selection

### Overview
Generate N shape candidates (varying only the shape seed), evaluate silhouette accuracy for each before running texture, then texture only the best shape(s). This amortizes the expensive texture stage.

### Theory
The key insight from CVPR'25 research on Best-of-N (see our quality-aware generation survey) is that verifier choice matters more than N for moderate budgets. For silhouette accuracy specifically:

```
For each seed s_i, i = 1..N:
    shape_i = run_shape_pipeline(seed=s_i)
    mesh_i = extract_quick_mesh(shape_i)  # low-res, no texture
    dice_i = silhouette_dice(render(mesh_i), reference_alpha)
best = argmax(dice_i)
texture = run_texture_pipeline(shape_best)
```

### Implementation Design

The pipeline already has `best_of_n` support in `run()` but it runs the FULL pipeline (shape+texture+decode+postprocess) for each candidate. A **staged** variant that only runs shape is far more efficient:

```python
def run_staged_best_of_n(self, image, N_shape=4, N_tex=2, seed=42, ...):
    # Stage 1: Generate N shapes, quick-decode, evaluate silhouette
    shape_candidates = []
    for i in range(N_shape):
        torch.manual_seed(seed + i)
        coords = self.sample_sparse_structure(cond_512, ...)
        shape_slat = self.sample_shape_slat(cond_1024, ..., coords)
        # Quick mesh extraction (no texture, fast decode)
        quick_mesh = self.decode_shape_only(shape_slat, res)
        dice = compute_silhouette_dice(quick_mesh, reference_alpha)
        shape_candidates.append((dice, shape_slat, coords, seed+i))

    # Stage 2: Keep top-K shapes, texture each
    shape_candidates.sort(key=lambda x: x[0], reverse=True)
    top_shapes = shape_candidates[:N_tex]

    # Stage 3: Texture top shapes, full decode, pick best
    results = []
    for dice, slat, coords, s in top_shapes:
        tex_slat = self.sample_tex_slat(cond_1024, ..., slat)
        mesh = self.decode_latent(slat, tex_slat, res)
        results.append(mesh)

    return results[0]  # or further selection
```

### Estimated Impact
- N=4 shapes, keep top-1: +3-5 pts on A1 (variance reduction)
- N=8 shapes, keep top-2 + pick best texture: +4-7 pts on A1
- The gain follows sqrt(N) diminishing returns

### Cost Analysis
| Config | Shape evals | Texture evals | Total cost (relative) |
|--------|------------|---------------|----------------------|
| Baseline | 1 | 1 | 1.0x |
| N=4 shapes, top-1 | 4 | 1 | ~3.2x (shape ~55% of total) |
| N=8 shapes, top-2 | 8 | 2 | ~6.4x |

Shape stage (512+1024 cascade) takes ~3.5 min; texture takes ~2 min. Quick decode for silhouette evaluation takes ~5s. So N=4 staged = ~14min + 2min + 0.3min = ~16min vs ~5.5min baseline = 3x cost.

### Flow Matching Compatibility
Fully compatible -- just varies the initial noise seed.

### Key Finding
The OFER method (CVPR'25 faces) confirms that ranking multiple shape hypotheses from a conditional diffusion model significantly improves reconstruction accuracy. Their IdRank network selects the best shape from N candidates, but for our case the silhouette Dice is an even stronger signal since it directly measures the target metric.

---

## Method 3: Noise Trajectory Optimization

### Overview
Instead of using a single random noise as initialization for the shape flow, search over noise vectors to find one that produces a shape with high silhouette accuracy. Based on "Not All Noises Are Created Equally" (ICLR 2025 submission) and "Test-Time Scaling of Diffusion Models via Noise Trajectory Search" (arXiv 2506.03164).

### Theory: Noise Inversion Stability

The ICLR submission establishes that noise inversion stability -- measured as the cosine similarity `s(epsilon) = cos(epsilon, epsilon')` between sampled noise and re-inverted noise -- correlates strongly with generation quality. Noises that are "stable" under forward-reverse cycling produce better samples.

For flow matching, the analogous concept is ODE trajectory stability:
```
x_1 = noise (initial)
x_0 = ODE_solve(v_theta, x_1, t=1->0)  # generate
x_1' = ODE_solve(v_theta, x_0, t=0->1)  # re-noise
stability = cos_sim(x_1, x_1')
```

High-stability noises correspond to regions where the learned velocity field is locally smooth and the ODE trajectory is well-conditioned.

### Theory: Noise Trajectory Search (NTS)

NTS (arXiv 2506.03164) formulates diffusion as an MDP with terminal reward:
- **State**: x_t at each denoising step
- **Action**: noise injection epsilon_t (for stochastic samplers) or branching choice
- **Reward**: R(x_0) = quality of final sample

The paper relaxes this to independent contextual bandits per timestep:
```
For step k: choose epsilon_k to maximize E[R(x_0) | x_{t_k}, epsilon_k]
```

This is impractical for full MDP but the contextual bandit relaxation is tractable: at each step, evaluate M candidate noises and keep the best.

### Adaptation for TRELLIS.2

**Challenge**: TRELLIS.2's shape stage uses deterministic ODE sampling (Euler/Heun), not stochastic. There is no noise injection at intermediate steps -- only the initial noise x_1 matters.

**Approach 1 -- Initial noise optimization**:
```python
# Gradient-free: evaluate N random noises, pick best
noises = [torch.randn_like(template) for _ in range(N)]
scores = [evaluate_shape(noise_i) for noise_i in noises]
best_noise = noises[argmax(scores)]

# Gradient-based: CMA-ES or Adam on noise space
noise = torch.randn(..., requires_grad=True)
for step in range(M):
    shape = forward_pass(noise)  # with torch.enable_grad()
    mesh = quick_decode(shape)
    dice = differentiable_silhouette_dice(mesh, ref_alpha)
    dice.backward()
    noise = noise + lr * noise.grad  # maximize
    noise = noise / noise.norm() * expected_norm  # re-normalize
```

**Critical issue**: The forward pass through the flow model is 12 steps of transformer inference on sparse 3D data. Backpropagating through all 12 steps requires gradient checkpointing and is extremely memory-intensive for sparse transformers. The gradient-free approach (Best-of-N at the noise level) is more practical.

**Approach 2 -- CMA-ES on initial noise (gradient-free)**:
```python
import cma
es = cma.CMAEvolutionStrategy(noise.flatten().cpu().numpy(), sigma0=0.1)
while not es.stop():
    candidates = es.ask()
    fitnesses = [-evaluate_shape(c.reshape(noise_shape)) for c in candidates]
    es.tell(candidates, fitnesses)
```

CMA-ES is known to work well for noise optimization in 2D diffusion (arXiv 2506.12036). For 3D, the noise dimension is much smaller (~50K tokens x 8 channels = 400K dims for shape) than 2D images (1M+ pixels), making CMA-ES more feasible.

### Estimated Impact
- N=8 random noise search: +2-4 pts A1 (equivalent to Best-of-N shape)
- CMA-ES optimization (50 evaluations): +3-5 pts A1
- Gradient-based noise optimization: +4-7 pts A1 (if feasible)

### Computational Cost
- N=8 random: 8x shape pipeline cost
- CMA-ES 50 evals: 50x shape pipeline cost (~3 hours for 1536_cascade)
- Gradient-based: 12-step backprop + gradient checkpointing, ~4x memory, 3x time per eval

### Flow Matching Compatibility
Fully compatible. Initial noise is a standard input to the ODE solver.

### Key Papers
- "Not All Noises Are Created Equally" (arXiv 2407.14041, ICLR 2025)
- "Test-Time Scaling of Diffusion Models via Noise Trajectory Search" (arXiv 2506.03164)
- "Golden Noise for Diffusion Models" (ICCV 2025, arXiv 2411.09502)
- "FIND: Fine-tuning Initial Noise Distribution" (arXiv 2407.19453)

---

## Method 4: FlowDPS Posterior Sampling with Silhouette Likelihood

### Overview
FlowDPS (ICCV 2025, arXiv 2503.08136) extends Diffusion Posterior Sampling to flow matching by deriving the flow-version of Tweedie's formula. It injects likelihood gradients into the flow ODE to enforce data consistency constraints.

### Theory

**Flow Tweedie's formula** (clean data estimate from noisy state):
```
x_hat_0|t = x_t - t * v_theta(x_t, t)
```

For TRELLIS.2's sigma_min variant:
```
x_hat_0|t = (1 - sigma_min) * x_t - (sigma_min + (1-sigma_min)*t) * v_theta(x_t, t)
```
(This is exactly `_pred_to_xstart()` in `flow_euler.py`)

**Posterior velocity** (flow ODE conditioned on observation y):
```
v_t(x_t | y) = v_t(x_t) - zeta_t * grad_x log p(y | x_hat_0|t)
```

where `zeta_t` is a time-dependent scaling factor.

**For silhouette constraint**:
```
y = S_ref  (reference silhouette image)
x_hat_0 = decode_sparse_latent_to_mesh(x_hat_0|t)  # THE BOTTLENECK
S_rendered = render_silhouette(mesh)
log p(y | x_hat_0) = -||S_ref - S_rendered||^2 / (2 * sigma_y^2)
```

### Critical Feasibility Issue

The fundamental problem: computing `grad_x log p(y | x_hat_0|t)` requires:
1. Predict `x_hat_0|t` from the current noisy sparse latent (fast, already done)
2. **Decode x_hat_0|t to a mesh** (expensive: FlexiCubes + marching cubes)
3. **Render the mesh** (differentiable: nvdiffrast)
4. **Backpropagate through steps 3, 2, and the flow model** to get gradients w.r.t. x_t

Step 2 is the bottleneck. The SLAT decoder includes:
- Sparse 3D convolutions
- FlexiCubes isosurface extraction (inherently differentiable if using FlexiCubes)
- Vertex position computation

The decoder IS differentiable (it uses FlexiCubes which was designed for gradient-based optimization), but running it at every ODE step (12 steps) would add ~5-10 seconds per step, totaling ~1-2 minutes additional for the shape stage alone.

### Practical Adaptation

**Approximation 1 -- Decode only at select timesteps**:
Only compute the silhouette gradient at t = {0.5, 0.3, 0.1} (3 steps instead of 12):
```python
guidance_timesteps = {0.5, 0.3, 0.1}
for t, t_prev in t_pairs:
    out = self.sample_once(model, sample, t, t_prev, cond)
    if t in guidance_timesteps:
        x_hat_0 = out.pred_x_0
        with torch.enable_grad():
            x_hat_0_grad = x_hat_0.clone().requires_grad_(True)
            mesh = decode_shape(x_hat_0_grad)
            sil = render_silhouette(mesh, ref_camera)
            loss = bce_loss(sil, ref_silhouette)
            grad = torch.autograd.grad(loss, x_hat_0_grad)[0]
        # Apply guidance
        sample = out.pred_x_prev - guidance_scale * grad * (t - t_prev)
```

**Approximation 2 -- Voxel-space silhouette (bypass mesh decoding)**:
Instead of full mesh decode, project the sparse structure occupancy directly:
```python
# x_hat_0 is sparse latent; its occupancy can be approximated from the coords
# Project occupied voxels to 2D using the camera
# Compare with reference silhouette in voxel projection space
```

This is MUCH faster but coarser.

### Estimated Impact
- Full FlowDPS (3 guided steps): +3-8 pts A1
- Voxel-space approximation: +1-3 pts A1
- Best combined with post-gen correction: +5-10 pts total

### Computational Cost
- Full FlowDPS: +60-120s per generation (3 decode+render+backprop cycles)
- Voxel-space: +2-5s per generation
- Memory: +4-8 GB for gradient computation through decoder

### Flow Matching Compatibility
FlowDPS is specifically designed for flow matching (ICCV 2025). The flow Tweedie formula maps directly to TRELLIS.2's `_pred_to_xstart()`. Fully compatible.

### Key Papers
- FlowDPS (ICCV 2025, arXiv 2503.08136)
- DPS (Chung et al., NeurIPS 2023)
- "Inference-Time Alignment in Diffusion Models" (arXiv 2501.09685, tutorial/review)
- "Fast constrained sampling in pre-trained diffusion models" (arXiv 2410.18804)

---

## Method 5: Sparse Structure Visual Hull Tightening

### Overview
The sparse structure stage (Stage 1) produces a 32^3 binary voxel grid. Currently, a visual hull from multi-view silhouettes can constrain this (already implemented in `visual_hull.py` for multi-view mode). For single-view, we can use the input image silhouette to create a 2D projection constraint.

### Theory

Visual hull carving intersects back-projected viewing cones from multiple silhouettes. With a single view, we get a viewing frustum constraint:
```
For each voxel (i,j,k) in 32^3 grid:
    project to camera pixel (u, v)
    if ref_silhouette[u, v] == 0:
        voxel[i,j,k] = 0  (outside object)
```

This removes voxels that project outside the reference silhouette, tightening the sparse structure before shape generation.

### Implementation

```python
def apply_single_view_hull(decoded_structure, reference_image,
                           camera_params, grid_resolution=32):
    """Carve sparse structure using single-view silhouette projection."""
    # Get reference silhouette
    alpha = extract_alpha(reference_image)
    sil = (alpha > 0.15).float()

    # Build voxel grid in canonical space
    R = grid_resolution
    lin = torch.linspace(-0.5 + 0.5/R, 0.5 - 0.5/R, R)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
    grid_pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    # Project to camera
    extr, intr = get_front_camera(camera_params)
    pts_cam = project_to_camera(grid_pts, extr, intr)

    # Sample silhouette at projected locations
    visible = grid_sample(sil, pts_cam.uv) > 0.5

    # Apply with dilation (conservative)
    hull_mask = visible.reshape(1, 1, R, R, R)
    hull_mask = max_pool3d(hull_mask.float(), 3, stride=1, padding=1) > 0

    return decoded_structure & hull_mask
```

### Estimated Impact
- Single-view hull: +1-2 pts A1 (removes obvious outlier voxels)
- Combined with depth prior: +2-3 pts A1
- Main benefit is for objects with concavities or thin features visible in silhouette

### Limitations
- Single view provides only a 2D projection constraint -- ambiguity along the depth axis
- Conservative dilation needed to avoid removing valid geometry
- Already in the pipeline for multi-view; single-view extension is straightforward

### Flow Matching Compatibility
Applied before shape flow, to the sparse structure output. Fully compatible.

### Computational Cost
Negligible (<0.5s).

---

## Method 6: Training-Free Guidance with Silhouette Reward

### Overview
TFG (NeurIPS 2024, arXiv 2409.15761) provides a unified framework for training-free guidance in diffusion models. The idea is to inject a reward signal (silhouette accuracy) into the sampling process without modifying model weights.

### Theory

TFG unifies four guidance mechanisms:
1. **Mean Guidance**: Shift the predicted mean toward higher reward
2. **Variance Guidance**: Modify the noise injection based on reward gradient
3. **Implicit Dynamic**: Momentum-based guidance accumulation
4. **Recurrence**: Re-denoise from a corrected intermediate state

For deterministic flow matching (no variance), only Mean Guidance applies:
```
v_guided(x_t, t) = v_theta(x_t, t) - lambda * grad_x R(x_hat_0|t)
```

where R is the silhouette reward computed on the predicted clean data `x_hat_0|t`.

### Critical Issue: Same as FlowDPS

The reward gradient `grad_x R(x_hat_0|t)` requires differentiating through:
1. Flow Tweedie prediction (cheap)
2. SLAT decoder (expensive)
3. Mesh rendering (moderate)
4. Silhouette comparison (cheap)

This is mathematically identical to FlowDPS's likelihood gradient injection.

### Voxel-Space Approximation

A cheaper alternative avoids full mesh decoding:
```python
# At each step, predict x_hat_0 (clean sparse latent)
x_hat_0 = self._pred_to_xstart(x_t, t, pred_v)

# Convert x_hat_0 features to occupancy probability
# (average over channels, sigmoid)
occupancy = sigmoid(x_hat_0.feats.mean(dim=-1))

# Project occupied tokens to 2D (using sparse coords)
projected = scatter_to_2d(x_hat_0.coords, occupancy, camera)

# Compare with reference silhouette
reward = -F.binary_cross_entropy(projected, ref_sil)

# Compute gradient w.r.t x_t (through the affine Tweedie transform)
grad = autograd.grad(reward, x_t)[0]

# Apply guidance
v_guided = pred_v - lambda * grad
```

**Advantage**: Only requires gradient through the Tweedie formula (affine) and a 2D scatter+comparison. No mesh decoding.

**Disadvantage**: Occupancy from raw latent features is a crude approximation -- the latent space is not directly interpretable as occupancy.

### Estimated Impact
- Full reward guidance (3 guided steps): +2-5 pts A1
- Voxel-space approximation: +0.5-2 pts A1
- Combined with post-gen correction: synergistic

### Computational Cost
- Full: Same as FlowDPS (~60-120s additional)
- Voxel-space: +2-5s per generation

### Flow Matching Compatibility
TFG was designed for DDPM/DDIM but the Mean Guidance component maps directly to flow matching via the velocity correction formula above.

### Key Papers
- TFG (NeurIPS 2024, arXiv 2409.15761)
- Universal Guidance (CVPR 2023)
- "Towards a unified framework for guided diffusion models" (arXiv 2512.04985)

---

## Method 7: Multi-View Silhouette Constraints

### Overview
Generate a synthetic back/side view from the input image using a multi-view diffusion model (Zero-1-to-3++, SV3D, Era3D), then use the multi-view silhouettes to constrain generation via visual hull and/or multi-view Best-of-N selection.

### Theory

Single-view 3D is inherently ambiguous along the depth axis. Adding even one more view (e.g., side or back) dramatically reduces the solution space. Multi-view diffusion models like SV3D can generate consistent novel views from a single image.

```
Step 1: img_back = SV3D(img_front, target_view=180deg)
Step 2: silhouettes = [alpha(img_front), alpha(img_back)]
Step 3: hull_mask = compute_visual_hull(silhouettes, cameras)
Step 4: coords = sample_sparse_structure(..., hull_mask=hull_mask)
Step 5: shape = sample_shape_slat(...)  # with visual hull constraint
```

### Practical Considerations

1. **Multi-view diffusion quality**: SV3D/Zero-1-to-3++ can produce inconsistent views, especially for complex objects. The visual hull should be conservative (dilated).

2. **Cost**: Running SV3D for 1-2 additional views adds ~10-30 seconds.

3. **Integration**: The multi-view visual hull code already exists at `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/utils/visual_hull.py` and is integrated in the multi-view pipeline path.

4. **Single-view path**: Currently `_run_single()` does NOT use any visual hull. Adding the front-view-only hull (Method 5) plus one generated side view would significantly constrain the sparse structure.

### Estimated Impact
- Front + generated back view hull: +2-4 pts A1
- Front + 2 side views: +3-5 pts A1
- Diminishing returns beyond 3 views

### Computational Cost
+10-30s per novel view generation (SV3D)

### Flow Matching Compatibility
Operates on the sparse structure stage (pre-flow for shape). Fully compatible.

### Key Limitation
Requires SV3D or similar model, which is not currently in the TRELLIS.2 Docker image. Would need to add ~3-5GB model weights.

---

## Method 8: Test-Time Mesh Refinement (GTR-style)

### Overview
GTR (ICLR 2025, arXiv 2406.05649) demonstrates that a lightweight per-instance optimization (20 iterations, 4 seconds on A100) can significantly improve both geometry and texture. The core idea is to fine-tune the NeRF/latent representation using a render-and-compare loss with the input image(s).

### Adaptation for TRELLIS.2

TRELLIS.2 uses a voxel-based representation (MeshWithVoxel) rather than NeRF. The `attrs` field contains per-voxel PBR features. We can optimize these attrs + vertex positions jointly:

```python
# After generation, before GLB export:
vertices = mesh.vertices.clone().requires_grad_(True)
attrs = mesh.attrs.clone().requires_grad_(True)

optimizer = torch.optim.Adam([vertices, attrs], lr=1e-3)

for step in range(20):
    # Render from input viewpoint
    rendered = render_with_voxel_attrs(vertices, faces, attrs, camera)

    # Compare with input image
    loss = (
        F.l1_loss(rendered.rgb, ref_rgb)
        + 0.5 * lpips_loss(rendered.rgb, ref_rgb)
        + 0.3 * F.mse_loss(rendered.alpha, ref_alpha)  # silhouette
        + laplacian_regularization(vertices, faces)
    )
    loss.backward()
    optimizer.step()
```

### Key Difference from SilhouetteCorrector

The SilhouetteCorrector ONLY optimizes vertex positions using silhouette loss. GTR-style refinement additionally:
1. Optimizes appearance (attrs/texture) simultaneously
2. Uses photometric loss (RGB comparison), not just silhouette
3. Can use LPIPS for perceptual quality

For A1 specifically, the silhouette component of the loss is what matters. But joint optimization avoids the issue of deforming geometry to match silhouette at the cost of texture alignment.

### Estimated Impact
- Geometry-only (silhouette focus): Same as Method 1 (+2-6 pts)
- Joint geometry+appearance: +3-6 pts A1, +2-4 pts A2 (color)
- The LPIPS component helps prevent degenerate geometry

### Computational Cost
20 iterations at 512px: ~4-8 seconds on A100
Requires nvdiffrast (already available) + LPIPS (needs pip install)

### Flow Matching Compatibility
Post-processing; fully compatible.

### Key Papers
- GTR (ICLR 2025, arXiv 2406.05649)
- MeTTA (BMVC 2024, arXiv 2408.11465): Test-time adaptation with viewpoint self-calibration
- SITTO (ICLR 2024): Single-image mesh reconstruction through test-time optimization

---

## Method 9: Noise Inversion Stability Selection

### Overview
Select the initial noise vector that has the highest "inversion stability" -- meaning the noise that, when used to generate a shape and then re-noised (inverted), produces a noise closest to the original. Stable noises produce higher-quality generations.

### Theory

From "Not All Noises Are Created Equally" (arXiv 2407.14041):

```
For flow matching:
1. Sample noise epsilon ~ N(0, I)
2. Generate: x_0 = ODE_solve(v_theta, epsilon, t: 1->0)
3. Re-noise: epsilon' = ODE_solve(v_theta, x_0, t: 0->1)
4. Stability: s(epsilon) = cos_sim(epsilon, epsilon')
```

Select epsilon with highest s(epsilon) from a pool of candidates.

### Critical Issue for TRELLIS.2

**The re-noising step (step 3) requires a reverse ODE from t=0 to t=1**, which means running the flow model in the FORWARD direction (noise to data is already forward in rectified flow convention). In TRELLIS.2:

- Forward: x_t = (1-t)*x_0 + (sigma_min + (1-sigma_min)*t)*epsilon
- The velocity field v_theta predicts the flow direction from noise to data
- Re-noising requires integrating v_theta from t=0 to t=1, which IS the standard sampling direction

Wait -- this is wrong. In TRELLIS.2, t=1 is noise and t=0 is data. The standard ODE is from t=1 to t=0 (denoising). Re-noising would go t=0 to t=1, which means integrating:

```
dx/dt = -v_theta(x_t, t)  # note: negative sign for reverse direction
```

This can be done with the same model but the trajectory may not be well-approximated with Euler due to the curvature near t=0.

### Practical Concerns

1. **Cost**: Each stability evaluation requires a full forward + reverse ODE pass = 2x generation cost per candidate
2. **100 candidates** as in the original paper = 200x the cost of a single generation
3. **Correlation with silhouette accuracy is unproven for 3D**: The paper only validates on 2D image generation (SDXL)
4. **No flow matching validation**: The paper is exclusively on DDPM-based models

### Estimated Impact
- Highly uncertain for 3D flow matching
- If it transfers from 2D: +2-4 pts A1
- Risk: May select noises that are "smooth" but not silhouette-accurate

### Flow Matching Compatibility
Theoretically applicable but requires implementing the reverse ODE, which is non-trivial for the sparse transformer backbone.

### Recommendation
**Low priority** -- the cost is too high (200x) for uncertain benefit. Better to use simple Best-of-N with direct silhouette evaluation (Method 2) which has guaranteed relevance.

---

## Method 10: Dynamic Guidance for Shape Fidelity

### Overview
Use timestep-dependent guidance schedules that emphasize shape fidelity at the critical timesteps. Based on our extensive dynamic CFG research (15 papers surveyed, see `dynamic_cfg_schedule_2026_02.md`).

### Theory

The consensus from the literature is that constant CFG is suboptimal. For shape fidelity specifically:

1. **Early timesteps (t near 1)**: Coarse structure is being determined. Low guidance allows diversity; high guidance enforces structure.
2. **Mid timesteps (t ~0.3-0.7)**: Shape details are forming. This is where guidance has the most impact on silhouette accuracy.
3. **Late timesteps (t near 0)**: Fine details. Over-guidance here causes saturation without improving silhouette.

### Recommended Schedule for Shape Stage

```python
# Bell-shaped guidance schedule for shape fidelity
def shape_guidance_schedule(t, w_base=10.0):
    """Higher guidance at mid-timesteps where shape is determined."""
    if t > 0.8:
        return w_base * 0.3  # low early (let structure emerge)
    elif t > 0.3:
        return w_base * 1.2  # high mid (enforce shape)
    else:
        return w_base * 0.6  # moderate late (preserve detail)
```

Or using the beta distribution approach:
```python
from scipy.stats import beta
# Beta(3,3) peaks at t=0.5, symmetric bell
schedule = beta.pdf(t, 3, 3) / beta.pdf(0.5, 3, 3) * w_base
```

### Implementation

The `flow_euler.py` already supports `guidance_anneal_min` and `guidance_anneal_start` for linear decay. A more general schedule hook is needed:

```python
# In FlowEulerSampler.sample():
guidance_schedule = kwargs.pop('guidance_schedule', None)
# ...
for step_idx, (t, t_prev) in enumerate(t_pairs):
    if guidance_schedule is not None:
        kwargs['guidance_strength'] = guidance_schedule(t)
    # ...
```

### Estimated Impact
- Bell-shaped vs constant: +1-2 pts A1
- Combined with higher base guidance for shape: +1-3 pts A1
- Zero risk (just a schedule change)

### Computational Cost
Zero additional cost.

### Flow Matching Compatibility
Fully compatible. Guidance schedules work identically in flow matching.

### Key Papers
- "Applying Guidance in a Limited Interval" (Wang et al., arXiv 2404.13040)
- ReCFG (CVPR 2025): Closed-form optimal timestep coefficients
- TCFG (CVPR 2025): Direction-aware SVD guidance

---

## Comparison Table

| # | Method | A1 Gain | Difficulty | Time Cost | Memory | Flow Compatible | Risk |
|---|--------|---------|------------|-----------|--------|-----------------|------|
| 1 | Silhouette Deformation (tuned) | +3-6 | Easy | +3-8s | +0.5 GB | Yes (post) | Low |
| 2 | Staged Best-of-N | +3-5 | Easy | 3-8x | +0 GB | Yes | Low |
| 3 | Noise Trajectory Opt | +3-5 | Hard | 8-50x | +4 GB | Yes | Medium |
| 4 | FlowDPS Posterior | +3-8 | Hard | +60-120s | +4-8 GB | Yes (native) | Medium |
| 5 | Visual Hull Tightening | +1-3 | Medium | +0.5s | +0 GB | Yes (pre) | Low |
| 6 | TFG Reward Guidance | +1-5 | Hard | +60s or +5s | +0-8 GB | Partial | High |
| 7 | Multi-View Constraints | +2-5 | Medium | +10-30s | +3-5 GB | Yes (pre) | Medium |
| 8 | GTR Test-Time Refine | +3-6 | Medium | +4-8s | +1 GB | Yes (post) | Low |
| 9 | Noise Inversion Stability | +2-4 | Hard | 200x | +0 GB | Uncertain | High |
| 10 | Dynamic Guidance | +1-3 | Easy | +0s | +0 GB | Yes | Very Low |

---

## Implementation Priority

### Tier 1: Quick wins (implement this week)

**P1: Tune SilhouetteCorrector** (Method 1)
- File: `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/postprocessing/silhouette_corrector.py`
- Changes: Add Soft Dice loss, reduce w_laplacian to 10-15, increase max_displacement to 0.04, add multi-resolution (256 then 512)
- Expected: +2-4 pts A1
- Time: 2-4 hours implementation

**P2: Dynamic guidance schedule** (Method 10)
- File: `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/pipelines/samplers/flow_euler.py`
- Changes: Add guidance_schedule callable parameter, implement beta(3,3) schedule
- Expected: +1-2 pts A1
- Time: 1-2 hours implementation

**P3: Single-view visual hull** (Method 5)
- File: New function in `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/utils/visual_hull.py`
- Changes: Add single-view projection carving to sparse structure stage in `_run_single()`
- Expected: +1-2 pts A1
- Time: 2-3 hours implementation

### Tier 2: High-impact, moderate effort (next sprint)

**P4: Staged Best-of-N** (Method 2)
- File: `/home/ethan/projects/Trellis2-DGX-Spark-Docker/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`
- Changes: Add `run_staged_best_of_n()` method, quick mesh decode for silhouette eval
- Expected: +3-5 pts A1
- Time: 4-8 hours implementation + testing

**P5: GTR-style joint optimization** (Method 8)
- File: Extend `SilhouetteCorrector` or new `MeshRefiner` class
- Changes: Add RGB photometric loss + LPIPS, joint vertex+attr optimization
- Expected: +3-6 pts A1 + bonus A2 improvement
- Time: 8-12 hours implementation

### Tier 3: Research-grade, high potential (future)

**P6: FlowDPS posterior sampling** (Method 4)
- File: New sampler or modify `flow_euler.py`
- Changes: Add silhouette likelihood gradient injection at select timesteps
- Expected: +3-8 pts A1
- Time: 20-40 hours (research + implementation + debugging)
- Dependency: Need to verify gradient flow through SLAT decoder + FlexiCubes

**P7: Multi-view constraint generation** (Method 7)
- Dependency: SV3D or Zero-1-to-3++ model integration
- Time: 16-24 hours (model integration + pipeline changes)

### Combined Trajectory

Assuming additive effects with diminishing returns:

| Phase | Methods | Expected A1 | Cumulative |
|-------|---------|-------------|------------|
| Baseline | - | 81.4 | 81.4 |
| Phase 1 | P1 + P2 + P3 | +3-5 | 84-87 |
| Phase 2 | + P4 + P5 | +3-5 | 87-91 |
| Phase 3 | + P6 | +2-4 | 89-94 |

**Conservative estimate**: 87-90 after Tier 1+2
**Optimistic estimate**: 92-95 after all tiers

---

## References

### Differentiable Rendering & Mesh Optimization
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) - NVIDIA differentiable rasterizer (available in TRELLIS.2 Docker)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes) (SIGGRAPH 2023) - Gradient-based mesh optimization
- [SoftRas](https://github.com/ShichenLiu/SoftRas) (ICCV 2019) - Soft rasterizer for smoother gradients
- [PyTorch3D mesh losses](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html) - Laplacian, edge, chamfer

### Flow Matching Guidance
- [FlowDPS](https://arxiv.org/abs/2503.08136) (ICCV 2025) - Flow-driven posterior sampling
- [CFG-Zero*](https://arxiv.org/abs/2503.18886) - Improved CFG for flow matching
- [TFG](https://arxiv.org/abs/2409.15761) (NeurIPS 2024) - Unified training-free guidance
- [Inference-Time Alignment Tutorial](https://arxiv.org/abs/2501.09685) - Comprehensive review

### Noise Optimization
- [Not All Noises Are Created Equally](https://arxiv.org/abs/2407.14041) - Noise inversion stability
- [Golden Noise](https://arxiv.org/abs/2411.09502) (ICCV 2025) - Learned noise perturbation
- [Noise Trajectory Search](https://arxiv.org/abs/2506.03164) - Test-time scaling via noise MDP
- [FIND](https://arxiv.org/abs/2407.19453) - Fine-tuning noise distribution

### Test-Time Optimization for 3D
- [GTR](https://arxiv.org/abs/2406.05649) (ICLR 2025) - Geometry and texture refinement
- [MeTTA](https://arxiv.org/abs/2408.11465) (BMVC 2024) - Test-time adaptation for 3D
- [SITTO](https://openreview.net/forum?id=hkWHdI8ss5) - Single-image test-time optimization

### Best-of-N and Quality Selection
- [OFER](https://cvpr.thecvf.com/Conferences/2025) (CVPR 2025) - Shape hypothesis ranking
- [Diffusion Tree Sampling](https://arxiv.org/abs/2506.20701) - Scalable inference-time alignment
- [Dynamic Search](https://arxiv.org/abs/2503.02039) - Inference-time alignment search

### Dynamic Guidance Schedules
- [Wang et al.](https://arxiv.org/abs/2404.13040) - Limited interval guidance
- [ReCFG](https://cvpr.thecvf.com/Conferences/2025) (CVPR 2025) - Optimal timestep coefficients
- [TCFG](https://cvpr.thecvf.com/Conferences/2025) (CVPR 2025) - Direction-aware guidance

### 3D Generation Quality
- [Geometry in Style](https://openaccess.thecvf.com/content/CVPR2025/) (CVPR 2025) - Surface normal deformation
- [R2-Mesh](https://arxiv.org/abs/2408.10135) - RL-powered mesh reconstruction
- [REPARO](https://openaccess.thecvf.com/content/ICCV2025/) (ICCV 2025) - Compositional 3D with differentiable layout

---

## Appendix: Existing Infrastructure

### Already Implemented
1. `SilhouetteCorrector` at `trellis2/postprocessing/silhouette_corrector.py` (510 lines)
2. `QualityVerifier.compute_silhouette_dice()` at `trellis2/utils/quality_verifier.py`
3. `compute_visual_hull()` at `trellis2/utils/visual_hull.py` (multi-view only)
4. `MeshRenderer` at `trellis2/renderers/mesh_renderer.py` (nvdiffrast-based)
5. Best-of-N selection in `run()` method of pipeline
6. Multiple test scripts for silhouette correction evaluation

### Available Dependencies
- nvdiffrast (installed in Docker, differentiable rendering)
- cumesh (CUDA mesh operations -- simplify, fill_holes)
- FlexiCubes (via FlexGEMM, differentiable isosurface)
- utils3d (camera utilities)
- trimesh (mesh I/O)
- PyTorch with autograd (gradient computation)

### Missing Dependencies (need to add)
- LPIPS (for perceptual loss in GTR-style refinement): `pip install lpips`
- SV3D/Zero-1-to-3++ (for multi-view generation): separate model download
- CMA-ES (for noise optimization): `pip install cma`
