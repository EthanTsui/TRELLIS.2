#!/usr/bin/env python3
"""
Automated quality evaluation for TRELLIS.2 3D generation.
Generates 3D models from test images, renders multi-view snapshots,
evaluates quality with v3 scoring (includes fragmentation, darkness, vibrancy),
and records results as JSON in the adjustments/ folder.

Usage:
    python auto_evaluate.py                          # Baseline champion config
    python auto_evaluate.py --cfg_mp 0.15            # Test CFG-MP strength
    python auto_evaluate.py --sweep cfg_mp 0.05 0.1 0.15 0.2 0.3
    python auto_evaluate.py --config path/to/adj.json  # Test from adjustment JSON
"""
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

# Add TRELLIS.2 to path
sys.path.insert(0, '/workspace/TRELLIS.2')

import torch
import torch.nn.functional as F
import cv2

# Test images (2 diverse examples for speed)
TEST_IMAGES = [
    'assets/example_image/T.png',  # steampunk device — complex metallic
    'assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp',  # ornate crown
    'assets/example_image/454e7d8a30486c0635369936e7bec5677b78ae5f436d0e46af0d533738be859f.webp',  # diverse third sample
]

# GA v2 Champion defaults
CHAMPION_CONFIG = {
    'ss_guidance_strength': 10.0,
    'ss_guidance_rescale': 0.8,
    'ss_sampling_steps': 12,
    'ss_rescale_t': 5.0,
    'shape_slat_guidance_strength': 10.0,
    'shape_slat_guidance_rescale': 0.5,
    'shape_slat_sampling_steps': 12,
    'shape_slat_rescale_t': 3.0,
    'tex_slat_guidance_strength': 12.0,
    'tex_slat_guidance_rescale': 1.0,
    'tex_slat_sampling_steps': 16,
    'tex_slat_rescale_t': 4.0,
    'cfg_mp_strength': 0.0,
    'resolution': '1024',
    'seed': 42,
    'decimation_target': 500000,
    'texture_size': 2048,
}

RESULTS_DIR = '/workspace/TRELLIS.2/optimization/research/adjustments'
OUTPUT_DIR = '/workspace/TRELLIS.2/optimization/test_outputs'


class QualityEvaluatorV3:
    """
    V3 Quality Evaluator — focuses on user-visible defects:
      1. Shape (15)       - Silhouette IoU + contour match
      2. Color Match (10) - Color fidelity to input
      3. Detail (10)      - Perceptual similarity
      4. Fragmentation (20) - Patchwork, UV seams, texture breakup (碎片化)
      5. Smoothness (15)  - Surface evenness (不平整)
      6. Darkness (15)    - Dark patches, overall brightness (黑掉)
      7. Vibrancy (15)    - Saturation, freshness, not worn/faded (破舊感)
    Total: 100 pts
    """

    def __init__(self):
        self.weights = {
            'shape': 15,
            'color_match': 10,
            'detail': 10,
            'fragmentation': 20,
            'smoothness': 15,
            'darkness': 15,
            'vibrancy': 15,
        }

    def evaluate_rendered_views(self, rendered_views, reference_image=None):
        """
        Evaluate rendered multi-view snapshots of a 3D model.

        Args:
            rendered_views: list of RGB numpy arrays (H, W, 3) from different angles
            reference_image: optional RGBA numpy array of the input image

        Returns:
            dict with per-dimension scores and overall
        """
        all_scores = []
        for i, view in enumerate(rendered_views):
            scores = self._evaluate_single_view(view, reference_image if i == 0 else None)
            all_scores.append(scores)

        # Average across views — but reference-dependent metrics only use view 0
        dim_names = list(self.weights.keys())
        ref_dependent = {'shape', 'color_match'}  # need reference image
        avg = {}
        for dim in dim_names:
            if dim in ref_dependent:
                # Use only the front view (which has real reference comparison)
                avg[dim] = float(all_scores[0].get(dim, 50.0))
            else:
                # Average quality metrics across all views
                vals = [float(s[dim]) for s in all_scores if dim in s]
                avg[dim] = float(np.mean(vals)) if vals else 50.0

        # Overall weighted score
        total_w = sum(self.weights.values())
        avg['overall'] = sum(avg.get(d, 0) * w / total_w for d, w in self.weights.items())
        avg['per_view'] = all_scores
        avg['scoring_version'] = 'v3'
        return avg

    def evaluate_texture_map(self, texture_rgb, mask):
        """Evaluate the UV texture map directly for defects."""
        scores = {}
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        if mask_bool.sum() < 100:
            return {'fragmentation_tex': 50, 'darkness_tex': 50, 'vibrancy_tex': 50}

        valid = texture_rgb[mask_bool].astype(np.float32)

        # Darkness in texture space
        lum = valid.mean(axis=-1)
        mean_lum = lum.mean()
        dark10_pct = (lum < 10).sum() / max(lum.size, 1) * 100
        dark30_pct = (lum < 30).sum() / max(lum.size, 1) * 100
        dark_score = 100.0
        if dark10_pct > 1:
            dark_score -= min(30, (dark10_pct - 1) * 8)
        if dark30_pct > 5:
            dark_score -= min(25, (dark30_pct - 5) * 4)
        if mean_lum < 90:
            dark_score -= min(20, (90 - mean_lum) * 0.4)
        scores['darkness_tex'] = max(0, dark_score)

        # Vibrancy in texture space (saturation)
        hsv = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1][mask_bool].astype(np.float32)
        mean_sat = sat.mean()
        low_sat_pct = (sat < 30).sum() / max(sat.size, 1) * 100
        vib_score = 100.0
        if mean_sat < 60:
            vib_score -= min(40, (60 - mean_sat) * 1.0)
        if low_sat_pct > 40:
            vib_score -= min(30, (low_sat_pct - 40) * 1.0)
        scores['vibrancy_tex'] = max(0, vib_score)

        # Fragmentation in texture space
        gray = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2GRAY)
        # Morphological crack detection
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        crack_diff = closed.astype(np.float32) - gray.astype(np.float32)
        crack_pct = ((crack_diff > 20) & mask_bool).sum() / max(mask_bool.sum(), 1) * 100
        # Edge density (harsh boundaries)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        harsh_pct = ((grad_mag > 40) & mask_bool).sum() / max(mask_bool.sum(), 1) * 100
        frag_score = 100.0
        if crack_pct > 1:
            frag_score -= min(40, (crack_pct - 1) * 8)
        if harsh_pct > 10:
            frag_score -= min(30, (harsh_pct - 10) * 2)
        scores['fragmentation_tex'] = max(0, frag_score)

        return scores

    def _evaluate_single_view(self, rendered, reference=None):
        """Evaluate a single rendered view."""
        scores = {}
        h, w = rendered.shape[:2]
        rgb = rendered[:, :, :3] if rendered.shape[2] >= 3 else rendered
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = (gray > 5).astype(np.float32)
        mask_bool = mask > 0.5

        if mask_bool.sum() < 100:
            return {k: 50.0 for k in self.weights}

        # --- Shape (silhouette IoU with reference) ---
        if reference is not None and reference.shape[2] == 4:
            ref_mask = (reference[:, :, 3] > 128).astype(np.float32)
            if ref_mask.shape != mask.shape:
                ref_mask = cv2.resize(ref_mask, (w, h))
            intersection = (mask * ref_mask).sum()
            union = np.clip(mask + ref_mask, 0, 1).sum()
            iou = intersection / max(union, 1)
            scores['shape'] = float(iou * 100)
        else:
            scores['shape'] = 50.0  # Can't compare without reference

        # --- Color Match ---
        if reference is not None:
            ref_resized = reference
            if reference.shape[:2] != (h, w):
                ref_resized = cv2.resize(reference, (w, h))
            ref_rgb = ref_resized[:, :, :3]
            both = mask_bool & ((ref_resized[:, :, 3] > 128) if ref_resized.shape[2] == 4 else np.ones((h, w), bool))
            if both.sum() > 100:
                r_vals = rgb[both].astype(np.float32)
                g_vals = ref_rgb[both].astype(np.float32)
                mae = np.abs(r_vals - g_vals).mean()
                scores['color_match'] = float(max(0, (1.0 - mae / 100.0)) * 100)
            else:
                scores['color_match'] = 50.0
        else:
            scores['color_match'] = 50.0

        # --- Detail (intrinsic texture richness, not reference comparison) ---
        # Measures high-frequency content within the object. More detail = better.
        # This is independent of reference matching since 3D renders naturally differ.
        detail_score = 50.0  # baseline: no detail
        if mask_bool.sum() > 100:
            # High-frequency energy via Laplacian
            lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
            lap_energy = np.abs(lap[mask_bool]).mean()
            # Good detail range: lap_energy 5-25 (too low = blurry, too high = noisy)
            if lap_energy < 3:
                detail_score = 20.0  # very blurry
            elif lap_energy < 8:
                detail_score = 20.0 + (lap_energy - 3) * 12  # 20-80
            elif lap_energy < 20:
                detail_score = 80.0 + (lap_energy - 8) * 1.67  # 80-100
            else:
                detail_score = max(60, 100.0 - (lap_energy - 20) * 2)  # too noisy, penalize

            # Bonus for varied textures (local variance diversity)
            blurred = cv2.GaussianBlur(gray.astype(np.float64), (11, 11), 3.0)
            local_var = np.abs(gray.astype(np.float64) - blurred)
            var_std = local_var[mask_bool].std()
            if var_std > 5:
                detail_score = min(100, detail_score + 5)  # texture diversity bonus
        scores['detail'] = max(0, min(100, detail_score))

        # --- Fragmentation (碎片化) ---
        # Detect patchwork, UV seams, texture breakup
        frag_score = 100.0
        # Interior region (away from silhouette)
        kernel_int = np.ones((10, 10), np.uint8)
        interior = cv2.erode(mask_bool.astype(np.uint8), kernel_int) > 0
        if interior.sum() > 100:
            # Crack detection (morphological close reveals thin dark gaps)
            gray_int = gray.copy()
            gray_int[~interior] = 128
            closed = cv2.morphologyEx(gray_int, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            crack_diff = closed.astype(np.float32) - gray_int.astype(np.float32)
            crack_ratio = ((crack_diff > 20) & interior).sum() / max(interior.sum(), 1)
            if crack_ratio > 0.003:
                frag_score -= min(35, (crack_ratio - 0.003) * 700)

            # Gradient harshness (sharp color boundaries = patchwork)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(sobelx**2 + sobely**2)
            harsh = (grad[interior] > 35).sum() / max(interior.sum(), 1)
            if harsh > 0.06:
                frag_score -= min(30, (harsh - 0.06) * 250)

            # Color variance check (high local variance = fragmented texture)
            rgb_f = rgb.astype(np.float32)
            local_var = np.zeros(gray.shape, dtype=np.float32)
            for c in range(3):
                blurred = cv2.GaussianBlur(rgb_f[:, :, c], (15, 15), 3.0)
                diff = (rgb_f[:, :, c] - blurred) ** 2
                local_var += diff
            local_var = np.sqrt(local_var / 3.0)
            high_var_ratio = (local_var[interior] > 25).sum() / max(interior.sum(), 1)
            if high_var_ratio > 0.15:
                frag_score -= min(25, (high_var_ratio - 0.15) * 150)

        scores['fragmentation'] = max(0, frag_score)

        # --- Smoothness (不平整) ---
        # Note: base_color renders naturally have texture detail. Only penalize
        # extreme noise/artifacts, not legitimate texture variation.
        smooth_score = 100.0
        if interior.sum() > 100:
            # Laplacian energy (surface noise) — raised threshold for 3D renders
            lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
            lap_energy = np.abs(lap[mask_bool]).mean()
            if lap_energy > 10:
                smooth_score -= min(35, (lap_energy - 10) * 2)

            # Local deviation from smooth version — use larger kernel, higher threshold
            blurred = cv2.GaussianBlur(gray.astype(np.float64), (15, 15), 4.0)
            local_diff = np.abs(gray.astype(np.float64) - blurred)
            avg_diff = local_diff[mask_bool].mean()
            if avg_diff > 6:
                smooth_score -= min(30, (avg_diff - 6) * 4)

            # Interior edge density (bumps) — higher threshold
            edges = cv2.Canny(rgb, 40, 100)
            edge_density = edges[interior].sum() / 255 / max(interior.sum(), 1)
            if edge_density > 0.06:
                smooth_score -= min(25, (edge_density - 0.06) * 300)

        scores['smoothness'] = max(0, smooth_score)

        # --- Darkness (黑掉) ---
        dark_score = 100.0
        masked_gray = gray[mask_bool].astype(np.float32)
        mean_brightness = masked_gray.mean()
        # Dark pixels (< 15) — black patches
        dark_pct = (masked_gray < 15).sum() / max(masked_gray.size, 1) * 100
        if dark_pct > 1:
            dark_score -= min(30, (dark_pct - 1) * 5)
        # Very dark area (< 40) — shadow/darkening
        vdark_pct = (masked_gray < 40).sum() / max(masked_gray.size, 1) * 100
        if vdark_pct > 5:
            dark_score -= min(25, (vdark_pct - 5) * 2)
        # Overall low brightness
        if mean_brightness < 100:
            dark_score -= min(25, (100 - mean_brightness) * 0.5)
        # Dark patch uniformity (large contiguous dark areas)
        dark_mask = (gray < 30) & mask_bool
        if dark_mask.sum() > 100:
            n_dark, _ = cv2.connectedComponents(dark_mask.astype(np.uint8))
            if n_dark > 3:
                dark_score -= min(15, (n_dark - 3) * 3)

        scores['darkness'] = max(0, dark_score)

        # --- Vibrancy / Freshness (破舊感) ---
        vib_score = 100.0
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1][mask_bool].astype(np.float32)
        val = hsv[:, :, 2][mask_bool].astype(np.float32)
        mean_sat = sat.mean()
        mean_val = val.mean()
        # Low saturation penalty (desaturated = faded/worn)
        if mean_sat < 50:
            vib_score -= min(35, (50 - mean_sat) * 1.0)
        # Low value/brightness
        if mean_val < 100:
            vib_score -= min(20, (100 - mean_val) * 0.4)
        # Very low saturation pixels (gray/washed out)
        gray_pct = (sat < 20).sum() / max(sat.size, 1) * 100
        if gray_pct > 30:
            vib_score -= min(25, (gray_pct - 30) * 1.0)
        # Contrast check (low contrast = muddy)
        p5, p95 = np.percentile(masked_gray, [5, 95])
        contrast = p95 - p5
        if contrast < 80:
            vib_score -= min(20, (80 - contrast) * 0.5)

        scores['vibrancy'] = max(0, vib_score)

        return scores


def load_pipeline():
    """Load the TRELLIS.2 pipeline and create envmap for rendering."""
    print("Loading TRELLIS.2 pipeline...", flush=True)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipe.cuda()
    print("Pipeline loaded.", flush=True)
    return pipe


def get_envmap():
    """Create a simple white envmap for rendering."""
    from trellis2.renderers import EnvMap
    hdri = torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.8
    return {'default': EnvMap(hdri)}


def generate_and_evaluate(pipeline, image_path, config, evaluator, envmap, output_prefix="test"):
    """Generate a 3D model and evaluate it."""
    import o_voxel.postprocess
    from trellis2.utils import render_utils

    print(f"\n--- Processing: {os.path.basename(image_path)} ---", flush=True)

    # Load and preprocess image
    img = Image.open(image_path)
    processed = pipeline.preprocess_image(img)

    # Run pipeline
    t0 = time.time()
    outputs, latents = pipeline.run(
        processed,
        seed=config.get('seed', 42),
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": config['ss_sampling_steps'],
            "guidance_strength": config['ss_guidance_strength'],
            "guidance_rescale": config['ss_guidance_rescale'],
            "rescale_t": config['ss_rescale_t'],
        },
        shape_slat_sampler_params={
            "steps": config['shape_slat_sampling_steps'],
            "guidance_strength": config['shape_slat_guidance_strength'],
            "guidance_rescale": config['shape_slat_guidance_rescale'],
            "rescale_t": config['shape_slat_rescale_t'],
        },
        tex_slat_sampler_params={
            "steps": config['tex_slat_sampling_steps'],
            "guidance_strength": config['tex_slat_guidance_strength'],
            "guidance_rescale": config['tex_slat_guidance_rescale'],
            "rescale_t": config['tex_slat_rescale_t'],
            "cfg_mp_strength": config.get('cfg_mp_strength', 0.0),
        },
        pipeline_type={
            "512": "512", "1024": "1024_cascade", "1536": "1536_cascade",
        }.get(config.get('resolution', '1024'), '1024_cascade'),
        return_latent=True,
    )
    gen_time = time.time() - t0
    print(f"  Generation: {gen_time:.1f}s", flush=True)

    mesh = outputs[0]
    mesh.simplify(16777216)

    # Render multi-view snapshots with envmap
    t0 = time.time()
    try:
        images = render_utils.render_snapshot(
            mesh, resolution=512, r=2, fov=36, nviews=8, envmap=envmap
        )
        # Use base_color for evaluation (unlit, most reliable for scoring)
        key = 'base_color' if 'base_color' in images else list(images.keys())[0]
        rendered_views = [images[key][i] for i in range(min(8, len(images[key])))]
    except Exception as e:
        print(f"  Warning: render failed ({e}), using dummy views", flush=True)
        rendered_views = [np.zeros((512, 512, 3), dtype=np.uint8)]

    render_time = time.time() - t0
    print(f"  Render: {render_time:.1f}s ({len(rendered_views)} views)", flush=True)

    # Load reference for front-view comparison
    ref_np = np.array(img.convert('RGBA'))

    # Evaluate rendered views
    view_scores = evaluator.evaluate_rendered_views(rendered_views, ref_np)

    # Extract GLB
    t0 = time.time()
    shape_slat, tex_slat, res = latents

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=config.get('decimation_target', 500000),
        texture_size=config.get('texture_size', 2048),
        remesh=True,
        remesh_band=1,
        remesh_project=0.9,
        max_metallic=0.05,
        min_roughness=0.2,
        normal_strength=0.7,
        normal_blur_sigma=1.5,
        sharpen_texture=False,
        verbose=True,
    )
    glb_time = time.time() - t0
    print(f"  GLB extraction: {glb_time:.1f}s", flush=True)

    # Save GLB
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    glb_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_{os.path.basename(image_path)}.glb')
    glb.export(glb_path)
    print(f"  Saved: {glb_path}", flush=True)

    # Evaluate texture maps from GLB (trimesh object)
    tex_scores = {}
    try:
        mat = glb.visual.material
        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            bc_img = np.array(mat.baseColorTexture)
            if bc_img.shape[2] == 4:
                tex_mask = bc_img[:, :, 3] > 10
                tex_rgb = bc_img[:, :, :3]
            else:
                tex_rgb = bc_img
                tex_mask = np.any(tex_rgb > 5, axis=-1)
            tex_scores = evaluator.evaluate_texture_map(tex_rgb, tex_mask)
            print(f"  Texture scores: frag={tex_scores.get('fragmentation_tex',0):.0f} "
                  f"dark={tex_scores.get('darkness_tex',0):.0f} "
                  f"vib={tex_scores.get('vibrancy_tex',0):.0f}", flush=True)
    except Exception as e:
        print(f"  Warning: texture evaluation failed: {e}", flush=True)

    # Blend texture scores into view scores
    if tex_scores:
        for dim_pair in [('fragmentation', 'fragmentation_tex'),
                         ('darkness', 'darkness_tex'),
                         ('vibrancy', 'vibrancy_tex')]:
            view_dim, tex_dim = dim_pair
            if tex_dim in tex_scores and view_dim in view_scores:
                view_scores[view_dim] = 0.5 * view_scores[view_dim] + 0.5 * tex_scores[tex_dim]
        # Recompute overall
        total_w = sum(evaluator.weights.values())
        view_scores['overall'] = sum(
            view_scores.get(d, 0) * w / total_w
            for d, w in evaluator.weights.items()
        )

    # Save rendered views for visual inspection
    for i, view in enumerate(rendered_views[:4]):
        view_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_{os.path.basename(image_path)}_view{i}.png')
        Image.fromarray(view).save(view_path)

    torch.cuda.empty_cache()

    return {
        'image': os.path.basename(image_path),
        'scores': view_scores,
        'texture_scores': tex_scores,
        'timings': {
            'generation_s': gen_time,
            'render_s': render_time,
            'glb_s': glb_time,
        },
        'glb_path': glb_path,
    }


def run_evaluation(config, label="test", test_images=None):
    """Run full evaluation pipeline."""
    if test_images is None:
        test_images = TEST_IMAGES

    # Filter to existing images
    existing = [p for p in test_images if os.path.exists(p)]
    if not existing:
        print(f"ERROR: No test images found. Tried: {test_images}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluation: {label}")
    print(f"Config: cfg_mp={config.get('cfg_mp_strength', 0.0)}")
    print(f"Test images: {len(existing)}")
    print(f"{'='*60}\n")

    pipeline = load_pipeline()
    envmap = get_envmap()
    evaluator = QualityEvaluatorV3()

    results = []
    for img_path in existing:
        try:
            r = generate_and_evaluate(
                pipeline, img_path, config, evaluator, envmap,
                output_prefix=label
            )
            results.append(r)
            # Print per-image scores
            s = r['scores']
            print(f"\n  Scores for {r['image']}:")
            for dim, w in evaluator.weights.items():
                print(f"    {dim:15s}: {s.get(dim, 0):6.1f}/100 (weight {w})")
            print(f"    {'OVERALL':15s}: {s['overall']:6.1f}/100")
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            traceback.print_exc()
            results.append({'image': os.path.basename(img_path), 'error': str(e)})

    # Compute averages
    valid = [r for r in results if 'scores' in r]
    if not valid:
        print("No valid results!")
        return None

    avg_scores = {}
    for dim in evaluator.weights:
        vals = [r['scores'][dim] for r in valid]
        avg_scores[dim] = float(np.mean(vals))
    avg_scores['overall'] = float(np.mean([r['scores']['overall'] for r in valid]))

    print(f"\n{'='*60}")
    print(f"AVERAGE SCORES ({len(valid)} images):")
    for dim, w in evaluator.weights.items():
        print(f"  {dim:15s}: {avg_scores[dim]:6.1f}/100 (weight {w})")
    print(f"  {'OVERALL':15s}: {avg_scores['overall']:6.1f}/100")
    print(f"{'='*60}\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f'eval_{label}_{timestamp}.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({
            'label': label,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'scoring_version': 'v3.1',
            'average_scores': avg_scores,
            'per_image': results,
        }, f, indent=2, default=str)
    print(f"Results saved: {result_file}")

    del pipeline
    torch.cuda.empty_cache()
    return avg_scores


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Auto Evaluator')
    parser.add_argument('--cfg_mp', type=float, default=None, help='CFG-MP strength to test')
    parser.add_argument('--sweep', nargs='+', help='Sweep parameter: name val1 val2 ...')
    parser.add_argument('--baseline', action='store_true', help='Run baseline only')
    parser.add_argument('--full', action='store_true', help='Run full optimization cycle')
    args = parser.parse_args()

    if args.sweep:
        # Sweep mode: test multiple values
        param = args.sweep[0]
        values = [float(v) for v in args.sweep[1:]]
        print(f"Sweep mode: {param} = {values}")
        all_results = {}
        for val in values:
            config = dict(CHAMPION_CONFIG)
            config[param] = val
            label = f"sweep_{param}_{val}"
            scores = run_evaluation(config, label=label)
            if scores:
                all_results[val] = scores
                print(f"\n  {param}={val}: overall={scores['overall']:.1f}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY: {param}")
        print(f"{'='*60}")
        for val, scores in sorted(all_results.items()):
            dims = " | ".join(f"{d[:4]}={scores[d]:.0f}" for d in ['fragmentation', 'smoothness', 'darkness', 'vibrancy'])
            print(f"  {param}={val:6.3f}: overall={scores['overall']:5.1f} | {dims}")

    elif args.cfg_mp is not None:
        config = dict(CHAMPION_CONFIG)
        config['cfg_mp_strength'] = args.cfg_mp
        run_evaluation(config, label=f"cfg_mp_{args.cfg_mp}")

    elif args.full:
        # Full cycle: baseline + CFG-MP sweep
        print("=== FULL OPTIMIZATION CYCLE ===\n")

        # 1. Baseline
        baseline = run_evaluation(dict(CHAMPION_CONFIG), label="baseline_v3")

        # 2. CFG-MP sweep
        best_mp = 0.0
        best_score = baseline['overall'] if baseline else 0
        for mp in [0.05, 0.1, 0.15, 0.2, 0.3]:
            config = dict(CHAMPION_CONFIG)
            config['cfg_mp_strength'] = mp
            scores = run_evaluation(config, label=f"cfg_mp_{mp}")
            if scores and scores['overall'] > best_score:
                best_score = scores['overall']
                best_mp = mp
                print(f"  NEW BEST: cfg_mp={mp} -> {best_score:.1f}")

        print(f"\n=== BEST: cfg_mp_strength={best_mp}, score={best_score:.1f} ===")

    else:
        # Default: focused experiments on impactful parameters
        print("=== FOCUSED OPTIMIZATION CYCLE v3.2 ===\n")
        results_all = {}

        # 1. Baseline (champion config, texture_size=2048)
        baseline = run_evaluation(dict(CHAMPION_CONFIG), label="baseline_v3.2")
        if baseline:
            results_all['baseline'] = baseline

        # 2. CFG-MP with fixed conditional pred_x_0 (0.15, 0.3)
        for mp in [0.15, 0.3]:
            config = dict(CHAMPION_CONFIG)
            config['cfg_mp_strength'] = mp
            scores = run_evaluation(config, label=f"cfgmp_fixed_{mp}")
            if scores:
                results_all[f'cfgmp_{mp}'] = scores

        # 3. Higher texture resolution (4096)
        config = dict(CHAMPION_CONFIG)
        config['texture_size'] = 4096
        scores = run_evaluation(config, label="texres_4096")
        if scores:
            results_all['texres_4096'] = scores

        # Print comparison table
        print(f"\n{'='*80}")
        print(f"COMPARISON TABLE (v3.2)")
        print(f"{'='*80}")
        header = f"{'Config':20s}"
        dims = ['shape', 'fragmentation', 'smoothness', 'darkness', 'detail', 'overall']
        for d in dims:
            header += f" | {d[:8]:>8s}"
        print(header)
        print("-" * 80)
        for name, scores in results_all.items():
            row = f"{name:20s}"
            for d in dims:
                row += f" | {scores.get(d, 0):8.1f}"
            print(row)
        print(f"{'='*80}")

        # Identify best
        best_name = max(results_all, key=lambda k: results_all[k].get('overall', 0))
        print(f"\n=== BEST: {best_name} -> {results_all[best_name]['overall']:.1f} ===")


if __name__ == '__main__':
    main()
