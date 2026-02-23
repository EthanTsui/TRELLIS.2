#!/usr/bin/env python3
"""A/B test: Standard CFG vs FDG for texture stage.

Generates the same images with identical seeds, comparing:
  A) cfg_mode='standard' (current default)
  B) cfg_mode='fdg' with lambda_low=0.6, lambda_high=1.3

Uses QualityVerifier for quick scoring (not full V4 eval).
"""
import sys, os, time, json, gc
sys.path.insert(0, '/workspace/TRELLIS.2')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
from PIL import Image


def main():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils.quality_verifier import QualityVerifier

    print("=== FDG A/B Test ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    verifier = QualityVerifier(device='cuda')

    # Champion config (shared across both A and B)
    ss_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.8, "rescale_t": 5.0, "multistep": True}
    shape_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.5, "rescale_t": 3.0, "multistep": True}

    # A: Standard CFG for texture
    tex_params_standard = {
        "steps": 16, "guidance_strength": 12.0, "guidance_rescale": 1.0,
        "rescale_t": 4.0, "heun_steps": 4, "multistep": True,
        "cfg_mode": "standard",
    }

    # B: FDG for texture (default params from paper)
    tex_params_fdg = {
        "steps": 16, "guidance_strength": 12.0, "guidance_rescale": 1.0,
        "rescale_t": 4.0, "heun_steps": 4, "multistep": True,
        "cfg_mode": "fdg", "fdg_sigma": 1.0,
        "fdg_lambda_low": 0.6, "fdg_lambda_high": 1.3,
    }

    # Test images
    example_dir = '/workspace/TRELLIS.2/assets/example_image'
    test_images = ['T.png',
                   '0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp',
                   '454e7d8a30486c0635369936e7bec5677b78ae5f436d0e46af0d533738be859f.webp']

    results = []

    for img_name in test_images:
        img_path = os.path.join(example_dir, img_name)
        if not os.path.exists(img_path):
            print(f"\nSkipping {img_name} (not found)", flush=True)
            continue

        short = img_name[:20] + '...' if len(img_name) > 20 else img_name
        print(f"\n{'='*60}", flush=True)
        print(f"Image: {short}", flush=True)
        print(f"{'='*60}", flush=True)

        img = Image.open(img_path)
        processed = pipeline.preprocess_image(img)

        scores = {}
        for label, tex_params in [("A_standard", tex_params_standard), ("B_fdg", tex_params_fdg)]:
            print(f"\n  [{label}] Generating (seed=42)...", flush=True)
            t0 = time.time()

            outputs = pipeline.run(
                processed, seed=42, preprocess_image=False,
                sparse_structure_sampler_params=ss_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                pipeline_type="1024_cascade",
                max_num_tokens=65536,
            )
            gen_time = time.time() - t0
            mesh = outputs[0]

            t1 = time.time()
            score_dict = verifier.score(mesh, processed, num_views=6, render_resolution=512)
            score_time = time.time() - t1

            scores[label] = {
                'scores': score_dict,
                'gen_time': gen_time,
                'score_time': score_time,
            }

            print(f"    Total: {score_dict['total']:.4f} "
                  f"(dice={score_dict['silhouette_dice']:.3f}, "
                  f"color={score_dict['color_match']:.3f}, "
                  f"geo={score_dict['geometric']:.3f}, "
                  f"tex={score_dict['tex_coherence']:.3f}, "
                  f"detail={score_dict['detail_richness']:.3f})",
                  flush=True)
            print(f"    Time: gen={gen_time:.1f}s, score={score_time:.1f}s", flush=True)

            del outputs, mesh
            gc.collect()
            torch.cuda.empty_cache()

        # Compare
        a = scores['A_standard']['scores']['total']
        b = scores['B_fdg']['scores']['total']
        diff = b - a
        print(f"\n  --- Comparison for {short} ---")
        print(f"  Standard: {a:.4f}")
        print(f"  FDG:      {b:.4f}")
        print(f"  Diff:     {diff:+.4f} ({diff*100:+.1f}%)")

        for key in ['silhouette_dice', 'color_match', 'geometric', 'tex_coherence', 'detail_richness']:
            va = scores['A_standard']['scores'][key]
            vb = scores['B_fdg']['scores'][key]
            print(f"    {key:20s}: {va:.3f} -> {vb:.3f} ({vb-va:+.3f})")

        results.append({
            'image': img_name,
            'standard': scores['A_standard']['scores'],
            'fdg': scores['B_fdg']['scores'],
            'diff': diff,
            'gen_time_std': scores['A_standard']['gen_time'],
            'gen_time_fdg': scores['B_fdg']['gen_time'],
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Standard CFG vs FDG")
    print(f"{'='*60}")

    diffs = [r['diff'] for r in results]
    print(f"Average improvement: {np.mean(diffs):+.4f} ({np.mean(diffs)*100:+.1f}%)")

    for r in results:
        short = r['image'][:20]
        print(f"  {short:22s}: std={r['standard']['total']:.4f}, "
              f"fdg={r['fdg']['total']:.4f} ({r['diff']:+.4f})")
        print(f"    Time: std={r['gen_time_std']:.1f}s, fdg={r['gen_time_fdg']:.1f}s "
              f"(overhead: {r['gen_time_fdg']-r['gen_time_std']:+.1f}s)")

    out_path = '/tmp/fdg_ab_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
