#!/usr/bin/env python3
"""Efficient A/B test runner — loads pipeline ONCE, reuses across configs.

Usage:
    python ab_test_runner.py --configs '{"label": {"param": value}, ...}'
    python ab_test_runner.py --preset schedule  # predefined test sets

Saves results to /tmp/ab_results.json and logs to /tmp/ab_test.log
"""
import sys, os, time, json, gc, argparse
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
from PIL import Image

LOG = '/tmp/ab_test.log'


def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')


# Import V4 evaluator components
from optimization.scripts.auto_evaluate_v4 import (
    CHAMPION_CONFIG, QualityEvaluatorV4, generate_and_evaluate,
    load_pipeline, get_envmap, OUTPUT_DIR,
)

# Test images
DEFAULT_IMAGES = ['assets/example_image/T.png']
FULL_IMAGES = [
    'assets/example_image/T.png',
    'assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp',
    'assets/example_image/454e7d8a30486c0635369936e7bec5677b78ae5f436d0e46af0d533738be859f.webp',
]

# Predefined test presets
PRESETS = {
    'schedule': {
        'sched_uniform': {'schedule': 'uniform'},
        'sched_edm7':    {'schedule': 'edm', 'schedule_rho': 7.0},
        'sched_quad':    {'schedule': 'quadratic'},
        'sched_logsnr':  {'schedule': 'logsnr'},
    },
    'rescale_t': {
        'rt_baseline':   {'shape_slat_rescale_t': 3.0, 'tex_slat_rescale_t': 4.0},
        'rt_shape4.5':   {'shape_slat_rescale_t': 4.5, 'tex_slat_rescale_t': 4.0},
        'rt_tex5.0':     {'shape_slat_rescale_t': 3.0, 'tex_slat_rescale_t': 5.0},
        'rt_both':       {'shape_slat_rescale_t': 4.5, 'tex_slat_rescale_t': 5.0},
    },
    'tex_steps': {
        'steps16': {'tex_slat_sampling_steps': 16},
        'steps24': {'tex_slat_sampling_steps': 24},
        'steps32': {'tex_slat_sampling_steps': 32},
    },
    # Best combinations from schedule + rescale_t results
    'best_combos': {
        'champion':      {},  # baseline champion
        'split_sched':   {    # quadratic geometry + uniform texture
            'ss_schedule': 'quadratic',
            'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
        },
        'shape_rt4.5':   {'shape_slat_rescale_t': 4.5},
        'split+rt4.5':   {   # best schedule + best rescale_t
            'ss_schedule': 'quadratic',
            'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'shape_slat_rescale_t': 4.5,
        },
    },
    # Power schedule family: t^p with different exponents (shape stages only)
    'power_sched': {
        'power_baseline': {},  # uniform baseline
        'power_1.5': {  # gentler curve
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'ss_schedule_power': 1.5, 'shape_schedule_power': 1.5,
        },
        'power_2.0': {  # classic quadratic
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'ss_schedule_power': 2.0, 'shape_schedule_power': 2.0,
        },
        'power_3.0': {  # aggressive detail focus
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'ss_schedule_power': 3.0, 'shape_schedule_power': 3.0,
        },
    },
    # Quadratic shape with more steps (same wall time as uniform 12-step)
    'quad_steps': {
        'qs_baseline': {},  # uniform 12-step baseline
        'qs_quad24': {  # quadratic 24-step shape (~same time as uniform 12)
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'ss_sampling_steps': 24, 'shape_slat_sampling_steps': 24,
        },
        'qs_quad32': {  # quadratic 32-step shape
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'ss_sampling_steps': 32, 'shape_slat_sampling_steps': 32,
        },
    },
    # Guidance schedule: dynamic CFG for texture stage
    'guidance_sched': {
        'gs_baseline':    {},  # standard constant guidance
        'gs_interval':    {   # trim both ends (NeurIPS 2024 interval paper)
            'tex_guidance_interval': [0.05, 0.9],
        },
        'gs_bell':        {   # symmetric bell curve (Stage-wise paper)
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 3.0,
            'tex_guidance_beta_b': 3.0,
        },
        'gs_early_peak':  {   # peak early, decay toward data (RectifiedHR)
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 5.0,
            'tex_guidance_beta_b': 2.0,
        },
    },
    # Guidance interval and schedule tests
    'guidance': {
        'gi_champion':     {},  # baseline: tex interval=(0.0,1.0), SS/shape use pipeline default [0.6,1.0]
        'gi_tex_narrow':   {    # Narrow texture interval — skip harmful early/late steps
            'tex_guidance_interval': (0.05, 0.85),
        },
        'gi_tex_original': {    # Restore original pipeline default for texture
            'tex_guidance_interval': (0.6, 0.9),
        },
        'gi_tex_beta33':   {    # Smooth bell-curve guidance schedule (symmetric)
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 3.0,
            'tex_guidance_beta_b': 3.0,
            'tex_guidance_interval': (0.0, 1.0),
        },
    },
    # Guidance schedule v2: triangular + asymmetric beta + combinations
    # NOTE: t_norm = t for interval (0,1). t=1 is noise, t=0 is data.
    # Beta(a,b) mode = (a-1)/(a+b-2). Beta(4,2): mode=0.75 (noise end).
    'guidance_sched2': {
        'gs2_baseline':    {},  # constant guidance (champion)
        'gs2_triangular':  {   # TV-CFG triangular (Stage-wise Dynamics paper, 4.2x ImageReward)
            'tex_guidance_schedule': 'triangular',
        },
        'gs2_beta42':      {   # peaks at t=0.75 (noise end), low at data end — anti-oversaturation
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 4.0,
            'tex_guidance_beta_b': 2.0,
        },
        'gs2_tri_narrow':  {   # triangular + narrow interval (skip harmful early/late)
            'tex_guidance_schedule': 'triangular',
            'tex_guidance_interval': (0.05, 0.9),
        },
    },
    # Guidance anneal: reduce guidance near t=0 to preserve fine detail
    'guidance_anneal': {
        'ga_baseline': {},  # constant guidance (champion)
        'ga_25pct': {  # reduce to 25% at t=0 (12→3)
            'tex_guidance_anneal_min': 0.25,
            'tex_guidance_anneal_start': 0.3,
        },
        'ga_50pct': {  # reduce to 50% at t=0 (12→6)
            'tex_guidance_anneal_min': 0.50,
            'tex_guidance_anneal_start': 0.3,
        },
        'ga_25pct_wide': {  # reduce to 25%, start at t=0.5
            'tex_guidance_anneal_min': 0.25,
            'tex_guidance_anneal_start': 0.5,
        },
    },
    # Combined: split schedule + guidance anneal
    'split_anneal': {
        'sa_baseline': {},  # champion
        'sa_split': {  # split sched only
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
        },
        'sa_split_anneal25': {  # split + guidance anneal to 25%
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'tex_guidance_anneal_min': 0.25,
            'tex_guidance_anneal_start': 0.3,
        },
        'sa_split_anneal50': {  # split + guidance anneal to 50%
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'tex_guidance_anneal_min': 0.50,
            'tex_guidance_anneal_start': 0.3,
        },
    },
    # Shape guidance schedule: dynamic guidance for shape stage (targeting A1 silhouette)
    # Research consensus: bell-shaped guidance helps shape accuracy
    'shape_guidance': {
        'sg_baseline':    {},  # constant guidance for shape
        'sg_bell33':      {   # symmetric bell for shape (beta(3,3))
            'shape_guidance_schedule': 'beta',
            'shape_guidance_beta_a': 3.0,
            'shape_guidance_beta_b': 3.0,
        },
        'sg_tri':         {   # triangular for shape
            'shape_guidance_schedule': 'triangular',
        },
        'sg_bell_narrow': {   # bell + narrow interval (skip very early/late)
            'shape_guidance_schedule': 'beta',
            'shape_guidance_beta_a': 3.0,
            'shape_guidance_beta_b': 3.0,
            'shape_guidance_interval': (0.05, 0.95),
        },
    },
    # Shape/SS sampling steps: more steps for better shape quality
    'shape_steps': {
        'steps_baseline': {},  # 12 SS + 12 shape
        'steps_ss16': {'ss_sampling_steps': 16},  # SS steps 12→16
        'steps_shape16': {'shape_slat_sampling_steps': 16},  # shape steps 12→16
        'steps_both16': {'ss_sampling_steps': 16, 'shape_slat_sampling_steps': 16},  # both 16
    },
    # Staged Best-of-N: generate N shape candidates, pick best by silhouette Dice
    'staged_bon': {
        'sbon_baseline':  {},  # single shape (N=1)
        'sbon_n4':   {'staged_bon': 4},   # 4 shape candidates
        'sbon_n8':   {'staged_bon': 8},   # 8 shape candidates
    },
    # Silhouette correction: test improved corrector (Dice loss, lower reg, higher displacement)
    'silcorr': {
        'sc_baseline':  {},  # no correction
        'sc_dice_v2':   {'enable_silcorr': True},  # new defaults: Dice, w_lap=10, max_disp=0.06
        'sc_aggressive': {'enable_silcorr': True, 'silcorr_w_lap': 5.0, 'silcorr_max_disp': 0.08},
        'sc_conservative': {'enable_silcorr': True, 'silcorr_w_lap': 20.0, 'silcorr_max_disp': 0.04, 'silcorr_dice': False},  # closer to old but with multires
    },
    # Occupancy threshold: tighten/loosen sparse structure for A1 silhouette
    'occ_threshold': {
        'occ_baseline':  {},  # threshold=0.0 (default)
        'occ_tight_05':  {'ss_occupancy_threshold': 0.5},   # tighter silhouette
        'occ_tight_10':  {'ss_occupancy_threshold': 1.0},   # much tighter
        'occ_loose_m05': {'ss_occupancy_threshold': -0.5},  # looser (recover thin features)
    },
    # Stochastic SDE sampling: convert ODE to SDE for diversity + detail
    # sde_alpha controls noise injection strength; 0=pure ODE
    'sde_sampling': {
        'sde_baseline': {},  # pure ODE (sde_alpha=0)
        'sde_ze_01':    {'tex_sde_alpha': 0.1, 'tex_sde_profile': 'zero_ends'},
        'sde_ze_02':    {'tex_sde_alpha': 0.2, 'tex_sde_profile': 'zero_ends'},
        'sde_ze_03':    {'tex_sde_alpha': 0.3, 'tex_sde_profile': 'zero_ends'},
        'sde_sqrt_02':  {'tex_sde_alpha': 0.2, 'tex_sde_profile': 'sqrt_t'},
    },
    # FDG: Frequency-Decoupled Guidance for texture stage
    # Boosts high-freq detail while dampening low-freq oversaturation
    'fdg': {
        'fdg_baseline': {},  # standard CFG (no FDG)
        'fdg_default':  {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 1.0, 'tex_fdg_lambda_low': 0.6, 'tex_fdg_lambda_high': 1.3},
        'fdg_hi15':     {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 1.0, 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},
        'fdg_hi20':     {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 1.0, 'tex_fdg_lambda_low': 0.4, 'tex_fdg_lambda_high': 2.0},
    },
    # FDG + sigma sweep (detail at different spatial scales)
    'fdg_sigma': {
        'fdgs_baseline': {},
        'fdgs_s05':     {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 0.5, 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},
        'fdgs_s10':     {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 1.0, 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},
        'fdgs_s20':     {'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 2.0, 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},
    },
    # FDG time-varying: lambda ramps from (1,1) at t=1 to (lambda_low, lambda_high) at t=0
    # Tests whether gradual FDG avoids early-step artifacts while boosting late-step detail
    'fdg_tv': {
        'fdgtv_baseline': {},  # standard CFG (no FDG)
        'fdgtv_fixed':    {'tex_cfg_mode': 'fdg', 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},  # constant FDG
        'fdgtv_linear':   {'tex_cfg_mode': 'fdg', 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5, 'tex_fdg_time_schedule': 'linear'},
        'fdgtv_cosine':   {'tex_cfg_mode': 'fdg', 'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5, 'tex_fdg_time_schedule': 'cosine'},
    },
    # Rectified-CFG++ (NeurIPS 2025): 3-NFE predictor-corrector
    # Evaluates guidance at predicted point (closer to manifold) instead of current point
    # DEPRECATED: use 'rcfgpp' preset instead (uses correct param names)
    'rectified_cfgpp': {
        'rcfg_baseline':  {},  # standard CFG
        'rcfg_default':   {'rectified_cfgpp': True},  # R-CFG++ with default lambda=4.5
        'rcfg_lambda3':   {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 3.0},  # lower lambda
        'rcfg_lambda6':   {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 6.0},  # higher lambda
    },
    # Mesh quality: remesh, projection, fragment cleanup
    # FDG analysis (agent aa8867c) found remesh discards learned sub-voxel offsets
    'mesh_quality': {
        'mq_baseline':   {},  # remesh=True, project=0.9, frag=50 (current)
        'mq_proj098':    {'remesh_project': 0.98},  # closer projection preserves FDG detail
        'mq_noremesh':   {'remesh': False},  # standard path preserves FDG vertex positions
        'mq_frag200':    {'min_fragment_faces': 200},  # remove more floating debris
    },
    # Color transfer: LAB chrominance histogram matching to improve A2
    # Directly targets histogram correlation + LAB ΔE components of A2 metric
    'color_transfer': {
        'ct_baseline':    {},  # no color transfer
        'ct_hist_07':     {'color_transfer': 'histogram', 'color_transfer_strength': 0.7},   # moderate
        'ct_hist_10':     {'color_transfer': 'histogram', 'color_transfer_strength': 1.0},   # full
        'ct_hist_04':     {'color_transfer': 'histogram', 'color_transfer_strength': 0.4},   # gentle
    },
    # Decimation target: higher → more faces → more geometric detail
    # Champion config uses 800K (matching app.py default)
    'decimation': {
        'dec_500k':  {'decimation_target': 500000},   # lower (old default)
        'dec_800k':  {},                                # champion default (800K)
        'dec_1200k': {'decimation_target': 1200000},   # higher
    },
    # BON4 + best guidance configs: combine staged Best-of-N with champion triangular
    # Standard BON4 gave 93.34. Triangular gave 92.6. Potentially additive.
    'bon4_combined': {
        'b4_baseline':  {'staged_bon': 4},  # staged BON4 alone (uses champion triangular)
        'b4_fdg':       {'staged_bon': 4,   # BON4 + FDG for max detail
                         'tex_cfg_mode': 'fdg', 'tex_fdg_sigma': 1.0,
                         'tex_fdg_lambda_low': 0.5, 'tex_fdg_lambda_high': 1.5},
        'b4_narrow':    {'staged_bon': 4,   # BON4 + narrow interval
                         'tex_guidance_interval': (0.05, 0.85)},
    },
    # Shape guidance schedule for improving A1 silhouette
    # A1 is the largest weighted gap (81.4/100, weight 15)
    'shape_gs_combined': {
        'sgc_baseline':      {},  # champion (tex=triangular, shape=constant)
        'sgc_shape_tri':     {   # triangular for shape stage only
            'shape_guidance_schedule': 'triangular',
        },
        'sgc_shape_bell':    {   # bell-curve beta(3,3) for shape stage only
            'shape_guidance_schedule': 'beta',
            'shape_guidance_beta_a': 3.0,
            'shape_guidance_beta_b': 3.0,
        },
        'sgc_shape_tri_bon4': {  # triangular shape + BON4
            'staged_bon': 4,
            'shape_guidance_schedule': 'triangular',
        },
    },
    # Texture refinement: test fixed refiner (no TV, proximity loss, lower iters/lr)
    'tex_refine': {
        'tr_baseline':      {},  # no refinement
        'tr_fixed':         {'enable_texture_refine': True},  # new defaults (20 iters, prox=0.5, tv=0)
        'tr_aggressive':    {'enable_texture_refine': True, 'refine_iters': 40, 'refine_proximity_weight': 0.2},
        'tr_conservative':  {'enable_texture_refine': True, 'refine_iters': 10, 'refine_proximity_weight': 1.0},
    },
    # Guidance rescale anneal: reduce rescale near t=0 to preserve CFG detail
    'rescale_anneal': {
        'ra_baseline':  {},  # no anneal (rescale=1.0 throughout)
        'ra_07':        {'tex_rescale_anneal_min': 0.7, 'tex_rescale_anneal_start': 0.3},  # taper to 70%
        'ra_05':        {'tex_rescale_anneal_min': 0.5, 'tex_rescale_anneal_start': 0.3},  # taper to 50%
        'ra_07_wide':   {'tex_rescale_anneal_min': 0.7, 'tex_rescale_anneal_start': 0.5},  # wider anneal range
    },
    # Single-view visual hull: carve voxels outside silhouette for A1 improvement
    'sv_hull': {
        'svh_baseline':  {},  # no hull carving
        'svh_enabled':   {'enable_single_view_hull': True},  # single-view hull carving
        'svh_tight':     {'enable_single_view_hull': True, 'ss_occupancy_threshold': 0.5},  # hull + tight threshold
    },
    # 64³ native cascade: skip max_pool, preserve decoder boundary detail for A1
    # WARNING: LR pass uses 1024 model on 64³ coords (~39K tokens) — much slower
    'cascade_64': {
        'c64_baseline':  {},  # default 32³ cascade
        'c64_native':    {'ss_native_64': True},  # 64³ native (no max_pool)
        'c64_nat_occ05': {'ss_native_64': True, 'ss_occupancy_threshold': 0.5},  # 64³ + tight occ
    },
    # 64³ cascade at 1536: the A1 improvement should be most visible at higher res
    'cascade_64_1536': {
        'c64h_baseline':   {'resolution': '1536', 'decimation_target': 800000},
        'c64h_native':     {'resolution': '1536', 'decimation_target': 800000, 'ss_native_64': True},
        'c64h_nat_occ05':  {'resolution': '1536', 'decimation_target': 800000,
                            'ss_native_64': True, 'ss_occupancy_threshold': 0.5},
    },
    # Round 3: Best combos from all prior tests (1024), updated 2026-02-23
    # VERIFIED FINDINGS (V4.1):
    # tri_narrow [0.05,0.9]: A2 +2.1, C1 +2.0, overall +0.6 (BEST for A2/C1)
    # beta(4,2) guidance: C3 +2.9, overall +0.5 (BEST for C3)
    # beta(3,3) guidance: A2 +0.6, C1 +1.3, overall +0.7 (BEST overall single change)
    # V4.1 findings: split_sched C3=86.0 (+5.0!), beta(4,2) C3=82.2 (+2.9), tri_narrow A2+2.1
    # SDE: ±0 (confirmed zero effect across all 5 configs)
    'round3_best': {
        'r3_baseline':     {},  # champion baseline (1024)
        'r3_split_b42':    {    # BEST C3 combo: split_sched (+5.0) + beta42 (+2.9)
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 4.0,
            'tex_guidance_beta_b': 2.0,
        },
        'r3_split_tri_n':  {    # C3 + A2/C1: split_sched + triangular + narrow
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'tex_guidance_schedule': 'triangular',
            'tex_guidance_interval': (0.05, 0.9),
        },
        'r3_full_combo':   {    # everything: split + beta42 + narrow + BON4
            'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic',
            'tex_schedule': 'uniform',
            'tex_guidance_schedule': 'beta',
            'tex_guidance_beta_a': 4.0,
            'tex_guidance_beta_b': 2.0,
            'tex_guidance_interval': (0.05, 0.9),
            'staged_bon': 4,
        },
    },
    # Round 4: 1536 resolution combos
    # config_1536 baseline: overall 93.96, C3=96.9 (+15.9!), A1=83.8 (+2.4), A2=77.7 (-3.2)
    # Key finding: A2 drop concentrated on crown (-26.3); excl. crown, A2 is HIGHER at 1536
    # Hypothesis: 1536 needs higher guidance to prevent hallucination on complex objects
    '1536_combos': {
        'hd_baseline':     {'resolution': '1536', 'decimation_target': 800000},
        'hd_triangular':   {'resolution': '1536', 'decimation_target': 800000,
                            'tex_guidance_schedule': 'triangular'},
        'hd_hi_guide':     {'resolution': '1536', 'decimation_target': 800000,
                            'tex_slat_guidance_strength': 15.0,
                            'shape_slat_guidance_strength': 12.0,
                            'tex_guidance_schedule': 'triangular'},
        'hd_bon4_tri':     {'resolution': '1536', 'decimation_target': 800000,
                            'tex_guidance_schedule': 'triangular',
                            'staged_bon': 4},
    },
    # Rectified-CFG++ (Saini et al., arXiv 2510.07631, NeurIPS 2025)
    # 3-NFE predictor-corrector: cond predictor → eval cond+uncond at predicted → interpolative correction
    # Replaces standard CFG extrapolation (omega>1) with manifold-preserving interpolation
    # true_cfg (lambda_max) ~4.5 for SD3; needs tuning for TRELLIS.2 (typically guidance/2)
    'rcfgpp': {
        'rc_baseline':     {},  # standard CFG (champion defaults)
        'rc_lam3':         {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 3.0},   # conservative
        'rc_lam5':         {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 5.0},   # moderate (SD3 default ≈4.5)
        'rc_lam8':         {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 8.0},   # aggressive
    },
    # R-CFG++ with gamma schedule and combinations
    'rcfgpp_combos': {
        'rcc_baseline':    {},  # standard CFG
        'rcc_g0_l5':      {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 5.0, 'rcfgpp_gamma': 0.0},   # constant alpha
        'rcc_g1_l5':      {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 5.0, 'rcfgpp_gamma': 1.0},   # linear ramp
        'rcc_split_l5':   {'rectified_cfgpp': True, 'rcfgpp_lambda_max': 5.0,                          # R-CFG++ + split_sched
                           'ss_schedule': 'quadratic', 'shape_schedule': 'quadratic', 'tex_schedule': 'uniform'},
    },
    'all': {},  # Combines all presets
}


def run_ab_test(configs: dict, test_images: list, label_prefix: str = 'ab'):
    """Run A/B test with shared pipeline.

    Args:
        configs: {label: {param_overrides}} dict
        test_images: list of image paths
        label_prefix: prefix for output labels

    Returns:
        {label: {scores}} dict
    """
    # Load pipeline ONCE
    log(f"Loading pipeline...")
    pipeline = load_pipeline()
    envmap = get_envmap()
    evaluator = QualityEvaluatorV4()

    existing = [p for p in test_images if os.path.exists(p)]
    if not existing:
        log(f"ERROR: No test images found: {test_images}")
        return {}

    # Check if any config needs silhouette correction
    needs_silcorr = any(
        v.get('enable_silcorr', False) for v in configs.values()
    )
    sil_corrector = None
    if needs_silcorr:
        from trellis2.postprocessing.silhouette_corrector import SilhouetteCorrector
        sil_corrector = SilhouetteCorrector(device='cuda')
        log("SilhouetteCorrector initialized for A/B test")

    # Check if any config needs texture refinement
    needs_texrefine = any(
        v.get('enable_texture_refine', False) for v in configs.values()
    )
    tex_refiner = None
    if needs_texrefine:
        from trellis2.postprocessing.texture_refiner import TextureRefiner
        tex_refiner = TextureRefiner(device='cuda')
        log("TextureRefiner initialized for A/B test")

    results = {}

    for label, overrides in configs.items():
        cfg = dict(CHAMPION_CONFIG)
        cfg.update(overrides)
        use_silcorr = cfg.pop('enable_silcorr', False)
        use_texrefine = cfg.pop('enable_texture_refine', False)
        desc = " ".join(f"{k}={v}" for k, v in overrides.items()) or "(champion defaults)"
        log(f"\n{'='*50}")
        log(f"=== {label} === {desc}")
        log(f"{'='*50}")

        image_results = []
        for img_path in existing:
            t0 = time.time()
            try:
                r = generate_and_evaluate(
                    pipeline, img_path, cfg, evaluator, envmap,
                    output_prefix=f"{label_prefix}_{label}",
                    silhouette_corrector=sil_corrector if use_silcorr else None,
                    texture_refiner=tex_refiner if use_texrefine else None,
                )
                image_results.append(r)
                s = r['scores']
                elapsed = time.time() - t0
                disc = s.get('disc_score', s['overall'])
                log(f"  {os.path.basename(img_path)}: overall={s['overall']:.1f} "
                    f"A1={s.get('A1_silhouette',0):.1f} A2={s.get('A2_color_dist',0):.1f} "
                    f"C3={s.get('C3_detail_richness',0):.1f} disc={disc:.1f} ({elapsed:.0f}s)")
            except Exception as e:
                log(f"  ERROR {os.path.basename(img_path)}: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()

        if image_results:
            # Average scores
            avg = {}
            score_keys = [k for k in image_results[0]['scores']
                         if not k.endswith('_std') and isinstance(image_results[0]['scores'][k], (int, float))]
            for k in score_keys:
                vals = [r['scores'][k] for r in image_results if k in r['scores']]
                if vals:
                    avg[k] = float(np.mean(vals))
                    if len(vals) > 1:
                        avg[f'{k}_std'] = float(np.std(vals))
            results[label] = avg
            disc_avg = avg.get('disc_score', avg['overall'])
            log(f"  AVG: overall={avg['overall']:.1f} disc={disc_avg:.1f}")

    return results


def print_comparison(results: dict, baseline_label: str = None):
    """Print formatted comparison table."""
    if not results:
        return

    if baseline_label is None:
        baseline_label = list(results.keys())[0]

    base = results.get(baseline_label, {}).get('overall', 0)
    key_dims = ['A1_silhouette', 'A2_color_dist', 'C1_tex_coherence', 'C3_detail_richness']

    log(f"\n{'='*80}")
    header = f"{'Config':20s} {'Overall':>8s}"
    for d in key_dims:
        short = d.split('_')[0]
        header += f" {short:>6s}"
    header += f" {'Delta':>7s}"
    log(header)
    log("-" * 80)

    for label, scores in sorted(results.items(), key=lambda x: -x[1].get('overall', 0)):
        delta = scores.get('overall', 0) - base
        line = f"{label:20s} {scores.get('overall',0):8.1f}"
        for d in key_dims:
            line += f" {scores.get(d,0):6.1f}"
        line += f" {delta:+7.1f}"
        log(line)


def main():
    parser = argparse.ArgumentParser(description='Efficient A/B test runner')
    parser.add_argument('--preset', type=str, default='all',
                        choices=list(PRESETS.keys()),
                        help='Predefined test set')
    parser.add_argument('--configs', type=str, default=None,
                        help='JSON dict of {label: {overrides}}')
    parser.add_argument('--images', type=str, default='1',
                        help='Number of test images: 1 or 3')
    args = parser.parse_args()

    t_start = time.time()
    # Clear log
    with open(LOG, 'w') as f:
        f.write('')

    log(f"=== A/B TEST RUNNER === {time.strftime('%H:%M:%S')}")

    # Select test images
    test_images = DEFAULT_IMAGES if args.images == '1' else FULL_IMAGES

    # Build config set
    if args.configs:
        configs = json.loads(args.configs)
    elif args.preset == 'all':
        configs = {}
        for preset_name, preset_configs in PRESETS.items():
            if preset_name == 'all':
                continue
            configs.update(preset_configs)
    else:
        configs = PRESETS[args.preset]

    log(f"Configs: {len(configs)}, Images: {len(test_images)}")
    log(f"Estimated time: {len(configs) * len(test_images) * 4:.0f} min")

    results = run_ab_test(configs, test_images)

    # Print comparison for each group
    for preset_name, preset_configs in PRESETS.items():
        if preset_name == 'all':
            continue
        group_results = {k: v for k, v in results.items() if k in preset_configs}
        if len(group_results) > 1:
            baseline = list(preset_configs.keys())[0]
            log(f"\n--- {preset_name.upper()} ---")
            print_comparison(group_results, baseline)

    # Overall ranking
    log(f"\n--- ALL CONFIGS RANKED ---")
    print_comparison(results)

    elapsed = time.time() - t_start
    log(f"\n=== DONE === {time.strftime('%H:%M:%S')} (total {elapsed/60:.1f} min)")

    # Save
    out_path = '/tmp/ab_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Results saved: {out_path}")

    # Also save to persistent location
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    persist_path = os.path.join(OUTPUT_DIR, f'ab_test_{ts}.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(persist_path, 'w') as f:
        json.dump({'results': results, 'configs': {k: dict(CHAMPION_CONFIG, **v) for k, v in configs.items()},
                   'test_images': [os.path.basename(p) for p in test_images],
                   'timestamp': ts}, f, indent=2, default=str)
    log(f"Persistent: {persist_path}")


if __name__ == '__main__':
    main()
