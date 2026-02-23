# Implementation Status — 2026-02-23

## Current Champion Config (V4.1 Score: ~92.7/100)

| Parameter | Value | Notes |
|-----------|-------|-------|
| resolution | 1024 | 1536 available but slower |
| ss_guidance_strength | 10.0 | |
| ss_guidance_rescale | 0.8 | |
| ss_sampling_steps | 12 | |
| ss_rescale_t | 3.0 | |
| shape_slat_guidance_strength | 10.0 | |
| shape_slat_guidance_rescale | 0.5 | |
| shape_slat_sampling_steps | 12 | |
| shape_slat_rescale_t | 3.0 | |
| tex_slat_guidance_strength | 12.0 | |
| tex_slat_guidance_rescale | 1.0 | Most important param |
| tex_slat_sampling_steps | 16 | |
| tex_slat_rescale_t | 4.0 | |
| heun_steps | 4 | Last 4 steps use 2nd-order Heun |
| multistep | True | AB2 (free accuracy) |
| tex_guidance_schedule | triangular | +0.6 vs beta(3,3), +1.1 vs constant |
| tex_guidance_interval | (0.0, 1.0) | Full interval |

## Completed This Session

### 1. Time-varying FDG Lambda Schedule (Task #72)
- **Files**: `cfg_utils.py`, `classifier_free_guidance_mixin.py`, `guidance_interval_mixin.py`, `trellis2_image_to_3d.py`
- **What**: FDG lambdas now ramp from (1.0, 1.0) to (λ_low, λ_high) within the guidance interval
- **Schedules**: `fixed` (default), `cosine`, `quadratic`, `late_only`
- **Thread-through**: `guidance_interval` now passed from GuidanceIntervalMixin → ClassifierFreeGuidanceMixin → compute_cfg_prediction → _fdg_effective_lambdas
- **All 4 CFG sites updated**: mixin, multidiffusion, TAPA, VW-concat
- **Backward compatible**: `fdg_time_schedule='fixed'` reproduces original behavior

### 2. Rectified-CFG++ (Task #73)
- **Files**: `flow_euler.py`, `app.py`, `auto_evaluate_v4.py`
- **What**: Predictor-corrector where corrector uses conditional-only velocity (guidance_strength=1)
- **Parameter**: `cfg_pp_steps` (0=off, N>0 = apply to last N steps)
- **Key insight**: CFG overshoots; cond-only corrector pulls back to data manifold
- **Cost**: Each CFG++ step costs 1 extra model evaluation (like Heun but with cond-only corrector)
- **Interaction**: CFG++ takes priority over Heun for the same step; compatible with AB2 on non-PP steps
- **UI**: Slider in Gradio (0-16, default 0)

### 3. Guidance Schedule Update
- Champion config updated: beta(3,3) → triangular
- Validated by 3-image A/B test: +0.6 overall, +0.8 disc_score

## Completed A/B Test Results (Round 1)

| Test | Key Finding | Best Config |
|------|-------------|-------------|
| guidance_anneal | ZERO EFFECT | N/A |
| best_combos | split_sched C3 +5.0 | quadratic SS+shape, uniform tex |
| guidance_interval | Narrow [0.05,0.85] = A2+2.2, C1+1.9 | tex_narrow |
| guidance_sched2 | triangular +0.6 overall | triangular |
| sde_sampling | ZERO EFFECT (5 configs) | N/A |

## Pending / Not Yet Tested

### High Priority (expected impact)
1. **Rectified-CFG++ A/B test** — `cfg_pp_steps=4,8,16` vs baseline
2. **Time-varying FDG A/B test** — cosine/quadratic/late_only schedules
3. **split_sched + beta(4,2) combo** — C3 potentially +7 (if additive)
4. **64³ native cascade** — A1 potentially +3-5
5. **Staged BON4** — A1 improvement via shape selection

### Medium Priority
6. **Narrow interval + triangular combo** — A2/C1 optimized
7. **Shape stage guidance schedule** — triangular/beta for geometry
8. **FDG with time-varying lambda** — texture detail at fine timesteps

### Confirmed Zero-Effect (skip)
- SDE sampling (all profiles/alphas)
- Guidance strength annealing near t=0
- rescale_t changes (3.0→4.5)
- Texture resolution 4096 (no improvement over 2048)
- APG/CFG-Zero* for Stage 1/2

## Score Ceiling Analysis

| Metric | Current Best | Weight | Gap to 100 |
|--------|-------------|--------|-----------|
| A1 silhouette | 81.4 | 15% | 18.6 |
| A2 color_dist | 82.9 | 10% | 17.1 |
| C3 detail_richness | 86.1 | 10% | 13.9 |
| C1 tex_coherence | 96.7 | 15% | 3.3 |
| B1 mesh_integrity | 98.0 | 10% | 2.0 |
| B2, C2, D1, E1 | 100.0 | 40% | 0.0 |

**Total recoverable: ~6.6 pts → theoretical max ~99/100**
**Realistic target: 95-96/100**

## Batch Test Status
- Batch `run_batch_tests.py` was interrupted by container restart (was running occ_threshold + sde_sampling + staged_bon + silcorr)
- Results from completed tests in `optimization/test_outputs/ab_test_*.json`
- Need to re-run remaining tests (occ_threshold, staged_bon, silcorr) in next session

## Architecture of Implemented Features

### Sampler Features (flow_euler.py sample() kwargs)
```
zero_init_steps    — CFG-Zero* hold (0=off)
cfg_mp_strength    — Manifold projection (0=off, 0.1-0.3)
heun_steps         — 2nd-order Heun for last N steps
cfg_pp_steps       — Rectified-CFG++ for last N steps (NEW)
multistep          — AB2 Adams-Bashforth 2nd order
schedule           — uniform/edm/logsnr/quadratic
guidance_anneal_*  — Linear guidance reduction near t=0
rescale_anneal_*   — Linear rescale reduction near t=0
sde_alpha/profile  — Stochastic SDE noise injection
```

### CFG Modes (cfg_utils.py compute_cfg_prediction)
```
standard           — Classic CFG: (1+w)*cond - w*uncond
apg                — Perpendicular decomposition (alpha dampens parallel)
fdg                — Frequency-decoupled: separate low/high freq guidance
  fdg_time_schedule: fixed/cosine/quadratic/late_only
  guidance_interval: interval-relative lambda ramp (NEW)
```

### Guidance Schedules (guidance_interval_mixin.py)
```
binary             — On/off within interval (original)
beta               — Beta distribution curve (parameterized by a, b)
triangular         — Symmetric triangle peaking at interval midpoint (champion)
```

## Files Modified This Session
1. `trellis2/pipelines/samplers/flow_euler.py` — cfg_pp_steps implementation
2. `trellis2/pipelines/samplers/cfg_utils.py` — guidance_interval for FDG time-varying
3. `trellis2/pipelines/samplers/classifier_free_guidance_mixin.py` — accept guidance_interval
4. `trellis2/pipelines/samplers/guidance_interval_mixin.py` — pass guidance_interval to super()
5. `trellis2/pipelines/trellis2_image_to_3d.py` — guidance_interval to all 3 compute_cfg_prediction calls
6. `optimization/scripts/auto_evaluate_v4.py` — cfg_pp_steps pass-through (both locations)
7. `app.py` — CFG++ slider + wiring
