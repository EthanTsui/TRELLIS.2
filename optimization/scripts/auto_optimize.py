#!/usr/bin/env python3
"""
TRELLIS.2 Automated Quality Optimization Loop

Runs iterative optimization of 3D generation quality:
1. Generate 3D from test images with current config
2. Render from matching view angles
3. Evaluate with multi-dimensional scoring
4. Log results and save GLB
5. Try next config variation
6. Repeat

Usage:
    python optimization/scripts/auto_optimize.py --iterations 100 --examples 4,6,7
"""

import os
import sys
import json
import time
import math
import shutil
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

# Add TRELLIS.2 to path
TRELLIS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TRELLIS_ROOT)

import numpy as np
import cv2
import torch
from PIL import Image

# TRELLIS.2 imports
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.utils.visibility import VIEW_CAMERA_PARAMS
import o_voxel

# Override o_voxel.postprocess with live-mounted version
# (site-packages version may be stale if entrypoint sync failed due to permissions)
_ovoxel_live_pp = os.path.join(TRELLIS_ROOT, 'o-voxel', 'o_voxel', 'postprocess.py')
if os.path.isfile(_ovoxel_live_pp):
    import importlib.util
    _spec = importlib.util.spec_from_file_location('o_voxel.postprocess', _ovoxel_live_pp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    o_voxel.postprocess = _mod
    sys.modules['o_voxel.postprocess'] = _mod
    print(f"Loaded live o_voxel.postprocess from {_ovoxel_live_pp}")

# Local imports
from evaluate import QualityEvaluator, render_mesh_at_views, split_input_image


# ============================================================
# TEST EXAMPLE DEFINITIONS
# ============================================================

UPLOAD_DIR = os.path.join(TRELLIS_ROOT, 'optimization', 'test_images')

TEST_EXAMPLES = {
    1: {
        'name': 'bikini_woman',
        'image': 'example1/Gemini_Generated_Image_5yov5c5yov5c5yov.png',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Human body, skin tones, fine detail',
    },
    2: {
        'name': 'elf_warrior',
        'image': 'example2/image (2).webp',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Dark subject, armor detail, flowing fabric',
    },
    3: {
        'name': 'park_bench',
        'image': 'example3/Gemini_Generated_Image_ppdaf3ppdaf3ppda.png',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Thin geometry, iron details',
    },
    4: {
        'name': 'floral_shoes',
        'image': 'example4/ChatGPT Image 2026年2月14日 上午01_57_12.png',
        'layout': '1x3',
        'view_names': ['front', 'left', 'right'],
        'challenge': 'Fine floral patterns, bright white surface',
    },
    6: {
        'name': 'treasure_chest',
        'image': 'example6/image (1).webp',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Dark subject, asymmetric (monster front, wood back)',
    },
    7: {
        'name': 'ultraman_mickey',
        'image': 'example7/image (4).webp',
        'layout': '1x3',
        'view_names': ['front', 'left', 'back'],
        'challenge': 'Dual-character design, mixed colors',
    },
}

# View name -> (yaw_rad, pitch_rad) for rendering
VIEW_ANGLES = {
    'front': (0.0, 0.0),
    'back': (math.pi, 0.0),
    'left': (math.pi / 2, 0.0),
    'right': (-math.pi / 2, 0.0),
    'top': (0.0, math.pi / 2 * 0.95),
    'bottom': (0.0, -math.pi / 2 * 0.95),
    'front_left': (math.pi / 4, 0.0),
    'front_right': (-math.pi / 4, 0.0),
}


# ============================================================
# CONFIGURATION SPACE
# ============================================================

# Baseline (best from R12 + remesh_project fix)
BASELINE_CONFIG = {
    # Pipeline params
    'resolution': '1024',
    'seed': 42,

    # Sparse structure sampler
    'ss_guidance_strength': 7.5,
    'ss_guidance_rescale': 0.7,
    'ss_sampling_steps': 12,
    'ss_rescale_t': 5.0,

    # Shape SLAT sampler
    'shape_slat_guidance_strength': 7.5,
    'shape_slat_guidance_rescale': 0.5,
    'shape_slat_sampling_steps': 12,
    'shape_slat_rescale_t': 3.0,

    # Texture SLAT sampler
    'tex_slat_guidance_strength': 5.0,
    'tex_slat_guidance_rescale': 0.85,
    'tex_slat_sampling_steps': 12,
    'tex_slat_rescale_t': 3.0,

    # Advanced guidance (CFG-Zero*, APG)
    'cfg_mode': 'standard',       # 'standard', 'cfg_zero_star', 'apg'
    'apg_alpha': 0.3,             # APG parallel damping (0=full, 1=standard)
    'zero_init_steps': 0,         # CFG-Zero* zero-init steps to skip

    # Guidance schedule (beta-shaped)
    'guidance_schedule': 'binary',  # 'binary' or 'beta'
    'guidance_beta_a': 2.0,         # Beta shape param alpha (>1 for unimodal)
    'guidance_beta_b': 5.0,         # Beta shape param beta (>1 for unimodal)

    # Multi-view
    'multiview_mode': 'concat',
    'texture_multiview_mode': 'tapa',

    # GLB export
    'decimation_target': 500000,
    'texture_size': 2048,
    'max_metallic': 0.05,
    'sharpen_texture': True,
    'projection_blend': 0.3,
    'remesh_project': 0.9,
}


def generate_param_variations(iteration: int) -> Dict:
    """
    Generate a single config variation for a given iteration number.
    Returns a modified copy of the baseline config.
    """
    cfg = deepcopy(BASELINE_CONFIG)

    # Phase 1 (iter 0): Baseline
    if iteration == 0:
        cfg['_variation_name'] = 'baseline_r17_remesh_fix'
        return cfg

    # Phase 2 (iter 1-24): Texture guidance sweep
    if iteration <= 24:
        tex_guidance_vals = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        tex_rescale_vals = [0.5, 0.7, 0.85, 0.95]
        idx = iteration - 1
        tg = tex_guidance_vals[idx // len(tex_rescale_vals) % len(tex_guidance_vals)]
        tr = tex_rescale_vals[idx % len(tex_rescale_vals)]
        cfg['tex_slat_guidance_strength'] = tg
        cfg['tex_slat_guidance_rescale'] = tr
        cfg['_variation_name'] = f'texg{tg}_texr{tr}'
        return cfg

    # Phase 3 (iter 25-44): Shape guidance sweep
    if iteration <= 44:
        shape_guidance_vals = [5.0, 7.5, 10.0, 12.0]
        shape_rescale_vals = [0.3, 0.5, 0.7, 0.9]
        steps_vals = [8, 12, 16]
        idx = iteration - 25
        sg_idx = idx // (len(shape_rescale_vals))
        sr_idx = idx % len(shape_rescale_vals)
        sg = shape_guidance_vals[sg_idx % len(shape_guidance_vals)]
        sr = shape_rescale_vals[sr_idx]
        # Also vary steps every 4 iterations
        st = steps_vals[(idx // 4) % len(steps_vals)]
        cfg['shape_slat_guidance_strength'] = sg
        cfg['shape_slat_guidance_rescale'] = sr
        cfg['shape_slat_sampling_steps'] = st
        cfg['_variation_name'] = f'shapeg{sg}_shaper{sr}_steps{st}'
        return cfg

    # Phase 4 (iter 45-56): Multi-view mode combinations
    if iteration <= 56:
        mv_modes = ['concat', 'view_weighted', 'multidiffusion']
        tex_mv_modes = ['single', 'concat', 'tapa', 'view_weighted']
        idx = iteration - 45
        mv = mv_modes[idx // len(tex_mv_modes) % len(mv_modes)]
        tmv = tex_mv_modes[idx % len(tex_mv_modes)]
        cfg['multiview_mode'] = mv
        cfg['texture_multiview_mode'] = tmv
        cfg['_variation_name'] = f'mv_{mv}_tex_{tmv}'
        return cfg

    # Phase 5 (iter 57-76): Postprocess parameter sweep
    if iteration <= 76:
        blend_vals = [0.1, 0.2, 0.3, 0.5, 0.7]
        gamma_vals = [100, 110, 115, 125]
        idx = iteration - 57
        bl = blend_vals[idx // len(gamma_vals) % len(blend_vals)]
        gt = gamma_vals[idx % len(gamma_vals)]
        cfg['projection_blend'] = bl
        cfg['_variation_name'] = f'blend{bl}_gamma{gt}'
        # Store gamma target for postprocess modification
        cfg['_gamma_target'] = gt
        return cfg

    # Phase 6 (iter 77-88): Resolution + texture size sweep
    if iteration <= 88:
        res_vals = ['512', '1024', '1536']
        tex_sizes = [1024, 2048, 4096]
        idx = iteration - 77
        res = res_vals[idx // len(tex_sizes) % len(res_vals)]
        tex_size = tex_sizes[idx % len(tex_sizes)]
        cfg['resolution'] = res
        cfg['texture_size'] = tex_size
        cfg['_variation_name'] = f'res{res}_texsize{tex_size}'
        return cfg

    # Phase 7 (iter 89-100): Seed diversity + combined best
    seed_base = 1000
    cfg['seed'] = seed_base + (iteration - 89)
    cfg['_variation_name'] = f'seed_{cfg["seed"]}'
    return cfg


# ============================================================
# MAIN PIPELINE
# ============================================================

class OptimizationRunner:
    """Main optimization loop runner."""

    def __init__(
        self,
        output_dir: str,
        example_ids: List[int],
        device: str = 'cuda',
    ):
        self.output_dir = Path(output_dir)
        self.example_ids = example_ids
        self.device = device
        self.pipeline = None
        self.evaluator = QualityEvaluator(device=device)
        self.results_log = []
        self.best_scores = {}
        self.best_configs = {}
        self.best_overall = 0.0

        # Create output directories
        self.glb_dir = self.output_dir / 'glb_outputs'
        self.render_dir = self.output_dir / 'renders'
        self.log_dir = self.output_dir / 'results'
        for d in [self.glb_dir, self.render_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_pipeline(self):
        """Load TRELLIS.2 pipeline."""
        print("Loading TRELLIS.2 pipeline...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
        self.pipeline.cuda()
        print("Pipeline loaded successfully.")

    def load_test_images(self) -> Dict[int, Dict]:
        """Load and preprocess test example images."""
        examples = {}
        for eid in self.example_ids:
            if eid not in TEST_EXAMPLES:
                print(f"Warning: Example {eid} not found, skipping")
                continue

            info = TEST_EXAMPLES[eid]
            img_path = os.path.join(UPLOAD_DIR, info['image'])
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}, skipping")
                continue

            image = Image.open(img_path).convert('RGBA')
            views = split_input_image(image, info['layout'], info['view_names'])

            # Prepare the dict of PIL images for the pipeline
            # Pipeline expects: Dict[str, PIL.Image] with view names as keys
            pipeline_input = {}
            for vname, varr in views.items():
                pipeline_input[vname] = Image.fromarray(varr)

            examples[eid] = {
                'info': info,
                'raw_image': image,
                'reference_views': views,  # numpy RGBA arrays
                'pipeline_input': pipeline_input,  # PIL images
            }
            print(f"  Loaded example {eid}: {info['name']} ({len(views)} views)")

        return examples

    def run_generation(
        self,
        pipeline_input: Dict[str, Image.Image],
        config: Dict,
    ):
        """Run the TRELLIS.2 pipeline with given config."""
        outputs = self.pipeline.run(
            pipeline_input,
            seed=config['seed'],
            preprocess_image=True,  # let pipeline handle preprocessing
            # NOTE: APG/beta schedule only safe for texture stage (Stage 3).
            # Shape stages (1 & 2) must use standard CFG -- APG's orthogonal
            # injection corrupts binary occupancy and SDF geometry features.
            sparse_structure_sampler_params={
                'steps': config['ss_sampling_steps'],
                'guidance_strength': config['ss_guidance_strength'],
                'guidance_rescale': config['ss_guidance_rescale'],
                'rescale_t': config['ss_rescale_t'],
                'zero_init_steps': config.get('zero_init_steps', 0),
            },
            shape_slat_sampler_params={
                'steps': config['shape_slat_sampling_steps'],
                'guidance_strength': config['shape_slat_guidance_strength'],
                'guidance_rescale': config['shape_slat_guidance_rescale'],
                'rescale_t': config['shape_slat_rescale_t'],
                'zero_init_steps': config.get('zero_init_steps', 0),
            },
            tex_slat_sampler_params={
                'steps': config['tex_slat_sampling_steps'],
                'guidance_strength': config['tex_slat_guidance_strength'],
                'guidance_rescale': config['tex_slat_guidance_rescale'],
                'guidance_interval': [0.0, 1.0],
                'rescale_t': config['tex_slat_rescale_t'],
                'cfg_mode': config.get('cfg_mode', 'standard'),
                'apg_alpha': config.get('apg_alpha', 0.3),
                'zero_init_steps': config.get('zero_init_steps', 0),
                'guidance_schedule': config.get('guidance_schedule', 'binary'),
                'guidance_beta_a': config.get('guidance_beta_a', 2.0),
                'guidance_beta_b': config.get('guidance_beta_b', 5.0),
            },
            pipeline_type={
                '512': '512',
                '1024': '1024_cascade',
                '1536': '1536_cascade',
            }[config['resolution']],
            return_latent=True,
            multiview_mode=config['multiview_mode'],
            texture_multiview_mode=config['texture_multiview_mode'],
        )
        # outputs is (mesh_list, (shape_slat, tex_slat, res))
        meshes, latents = outputs
        return meshes[0], latents

    def export_glb(
        self,
        mesh,
        config: Dict,
        input_images: List[Image.Image] = None,
        camera_params: List[dict] = None,
    ):
        """Export MeshWithVoxel to GLB with postprocess settings."""
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=self.pipeline.pbr_attr_layout,
            grid_size=int(mesh.voxel_shape[0]) if hasattr(mesh, 'voxel_shape') else 128,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=config['decimation_target'],
            texture_size=config['texture_size'],
            remesh=False,  # Disabled: CuMesh remesh has CUDA config error on SM 12.1
            remesh_band=1,
            remesh_project=config.get('remesh_project', 0.9),
            max_metallic=config['max_metallic'],
            sharpen_texture=config['sharpen_texture'],
            input_images=input_images,
            camera_params=camera_params,
            projection_blend=config['projection_blend'],
            use_tqdm=False,
        )
        return glb

    def render_and_evaluate(
        self,
        mesh,
        reference_views: Dict[str, np.ndarray],
        view_names: List[str],
    ) -> Tuple[Dict, Dict]:
        """Render mesh from matching views and evaluate."""
        view_angles = {}
        for vname in view_names:
            if vname in VIEW_ANGLES:
                view_angles[vname] = VIEW_ANGLES[vname]

        if not view_angles:
            return {'overall': 0, 'error': 'No valid view angles'}, {}

        # Render
        rendered = render_mesh_at_views(
            mesh, view_angles,
            resolution=512, r=2.0, fov=36.0,
        )

        # Evaluate
        scores = self.evaluator.evaluate(rendered, reference_views)
        return scores, rendered

    def run_single_iteration(
        self,
        iteration: int,
        config: Dict,
        examples: Dict,
    ) -> Dict:
        """Run one iteration: generate + evaluate for all examples."""
        var_name = config.get('_variation_name', f'iter_{iteration}')
        iteration_results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'config': {k: v for k, v in config.items() if not k.startswith('_')},
            'variation_name': var_name,
            'examples': {},
        }

        total_score = 0
        n_examples = 0

        for eid, example in examples.items():
            print(f"\n  --- Example {eid}: {example['info']['name']} ---")
            try:
                t_start = time.time()

                # Generate
                mesh, latents = self.run_generation(
                    example['pipeline_input'],
                    config,
                )
                t_gen = time.time() - t_start
                print(f"  Generation: {t_gen:.1f}s")

                # Render from matching views and evaluate
                view_names = example['info']['view_names']
                scores, rendered = self.render_and_evaluate(
                    mesh, example['reference_views'], view_names,
                )
                t_eval = time.time() - t_start - t_gen
                print(f"  Evaluation: {t_eval:.1f}s")
                print(f"  Score: {scores.get('overall', 0):.1f}/100")
                for dim in ['silhouette', 'contour', 'color', 'detail', 'artifacts']:
                    print(f"    {dim}: {scores.get(dim, 0):.1f}")

                # Export GLB
                try:
                    mv_images = [example['pipeline_input'][vn] for vn in view_names
                                 if vn in example['pipeline_input']]
                    cam_params = [VIEW_CAMERA_PARAMS.get(vn, VIEW_CAMERA_PARAMS['front'])
                                  for vn in view_names]
                    glb = self.export_glb(
                        mesh, config,
                        input_images=mv_images if mv_images else None,
                        camera_params=cam_params if mv_images else None,
                    )
                    glb_filename = f'iter{iteration:04d}_ex{eid}_{var_name}.glb'
                    glb_path = self.glb_dir / glb_filename
                    glb.export(str(glb_path), extension_webp=True)
                    glb_size_kb = os.path.getsize(glb_path) / 1024
                    print(f"  GLB saved: {glb_path.name} ({glb_size_kb:.0f} KB)")
                except Exception as e:
                    print(f"  GLB export failed (non-fatal): {e}")
                    glb_path = None
                    glb_size_kb = 0

                # Save renders
                for vname, rimg in rendered.items():
                    render_path = self.render_dir / f'iter{iteration:04d}_ex{eid}_{vname}.png'
                    Image.fromarray(rimg).save(str(render_path))

                # Record
                overall = scores.get('overall', 0)
                example_result = {
                    'example_id': eid,
                    'name': example['info']['name'],
                    'gen_time': round(t_gen, 1),
                    'eval_time': round(t_eval, 1),
                    'glb_path': str(glb_path) if glb_path else None,
                    'glb_size_kb': round(glb_size_kb, 0),
                    'scores': {k: round(v, 2) if isinstance(v, float) else v
                               for k, v in scores.items()
                               if k not in ('view_scores', 'texture_metrics')},
                }
                iteration_results['examples'][eid] = example_result
                total_score += overall
                n_examples += 1

                # Track best
                if eid not in self.best_scores or overall > self.best_scores[eid]:
                    self.best_scores[eid] = overall
                    self.best_configs[eid] = deepcopy(config)
                    print(f"  *** NEW BEST for example {eid}: {overall:.1f} ***")

            except Exception as e:
                print(f"  ERROR on example {eid}: {e}")
                traceback.print_exc()
                iteration_results['examples'][eid] = {
                    'example_id': eid,
                    'error': str(e),
                }

            # Clear GPU memory between examples
            torch.cuda.empty_cache()

        # Average score
        avg_score = total_score / max(n_examples, 1)
        iteration_results['avg_score'] = round(avg_score, 2)
        iteration_results['n_examples'] = n_examples

        if avg_score > self.best_overall:
            self.best_overall = avg_score
            print(f"\n  *** NEW BEST OVERALL: {avg_score:.1f} ***")

        return iteration_results

    def save_iteration_log(self, result: Dict, iteration: int):
        """Save iteration results to JSON."""
        log_path = self.log_dir / f'iter_{iteration:04d}.json'
        with open(log_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def save_summary(self):
        """Save overall summary to markdown."""
        summary_path = self.output_dir / 'optimization_summary.md'
        with open(summary_path, 'w') as f:
            f.write("# Automated Optimization Summary\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Best scores
            f.write("## Best Scores Per Example\n\n")
            f.write("| Example | Name | Best Score | Best Config |\n")
            f.write("|---------|------|-----------|-------------|\n")
            for eid in sorted(self.best_scores.keys()):
                info = TEST_EXAMPLES[eid]
                score = self.best_scores[eid]
                cfg = self.best_configs[eid]
                cfg_str = cfg.get('_variation_name', 'baseline')
                f.write(f"| {eid} | {info['name']} | {score:.1f} | {cfg_str} |\n")

            overall_avg = np.mean(list(self.best_scores.values())) if self.best_scores else 0
            f.write(f"\n**Best overall average: {overall_avg:.1f}**\n")

            # Iteration history
            f.write("\n## Iteration History\n\n")
            f.write("| Iter | Variation | Avg Score | Time |\n")
            f.write("|------|-----------|-----------|------|\n")
            for result in self.results_log:
                iter_n = result['iteration']
                var_name = result.get('variation_name', '')
                avg = result.get('avg_score', 0)
                ts = result.get('timestamp', '')
                f.write(f"| {iter_n} | {var_name} | {avg:.1f} | {ts} |\n")

    def run(self, max_iterations: int = 100, start_from: int = 0):
        """Main optimization loop."""
        print(f"\n{'='*60}")
        print(f"TRELLIS.2 Automated Quality Optimization")
        print(f"Max iterations: {max_iterations}")
        print(f"Test examples: {self.example_ids}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        # Load pipeline and examples
        self.load_pipeline()
        examples = self.load_test_images()

        if not examples:
            print("ERROR: No valid test examples found!")
            return

        print(f"\nLoaded {len(examples)} test examples.")
        print(f"Starting optimization from iteration {start_from}...\n")

        # Run iterations
        for iteration in range(start_from, start_from + max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{start_from + max_iterations - 1}")
            print(f"{'='*60}")

            # Get config for this iteration
            config = generate_param_variations(iteration)
            var_name = config.get('_variation_name', f'iter_{iteration}')
            print(f"Config: {var_name}")

            # Run
            result = self.run_single_iteration(iteration, config, examples)
            self.results_log.append(result)

            # Save progress
            self.save_iteration_log(result, iteration)
            self.save_summary()

            print(f"\nIteration {iteration} avg score: {result['avg_score']:.1f}")
            print(f"Best overall avg: {self.best_overall:.1f}")

            # Periodic cleanup
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                print(f"\n--- GPU memory cleaned ---")

        # Final summary
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {max_iterations}")
        print(f"Best overall average: {self.best_overall:.1f}")
        for eid in sorted(self.best_scores.keys()):
            info = TEST_EXAMPLES[eid]
            cfg = self.best_configs[eid]
            print(f"  Example {eid} ({info['name']}): {self.best_scores[eid]:.1f} [{cfg.get('_variation_name', '?')}]")
        print(f"\nResults saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Automated Quality Optimization')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--examples', type=str, default='4,6,7',
                        help='Comma-separated example IDs to test')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: optimization/run_YYYYMMDD_HHMMSS)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start iteration number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    example_ids = [int(x.strip()) for x in args.examples.split(',')]

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(TRELLIS_ROOT, 'optimization', f'run_{timestamp}')
    else:
        output_dir = args.output

    BASELINE_CONFIG['seed'] = args.seed

    runner = OptimizationRunner(
        output_dir=output_dir,
        example_ids=example_ids,
    )
    runner.run(
        max_iterations=args.iterations,
        start_from=args.start_from,
    )


if __name__ == '__main__':
    main()
