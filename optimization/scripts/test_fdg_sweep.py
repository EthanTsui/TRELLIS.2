#!/usr/bin/env python3
"""FDG lambda parameter sweep on single image.

Tests multiple (lambda_low, lambda_high) combinations to find optimal values.
Uses T.png with seed=42 at 1024 resolution for speed.
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

    print("=== FDG Lambda Sweep ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    verifier = QualityVerifier(device='cuda')

    # Shared params
    ss_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.8,
                 "rescale_t": 5.0, "multistep": True}
    shape_params = {"steps": 12, "guidance_strength": 10.0, "guidance_rescale": 0.5,
                    "rescale_t": 3.0, "multistep": True}
    base_tex = {"steps": 16, "guidance_strength": 12.0, "guidance_rescale": 1.0,
                "rescale_t": 4.0, "heun_steps": 4, "multistep": True}

    # Sweep configs: (label, cfg_mode, fdg_lambda_low, fdg_lambda_high)
    configs = [
        ("baseline",   "standard", 1.0, 1.0),   # standard CFG (control)
        ("fdg_06_13",  "fdg", 0.6, 1.3),         # paper defaults
        ("fdg_08_12",  "fdg", 0.8, 1.2),         # mild
        ("fdg_05_15",  "fdg", 0.5, 1.5),         # aggressive
        ("fdg_07_10",  "fdg", 0.7, 1.0),         # low-freq suppression only
        ("fdg_10_15",  "fdg", 1.0, 1.5),         # high-freq boost only
        ("fdg_04_18",  "fdg", 0.4, 1.8),         # very aggressive
        ("fdg_09_11",  "fdg", 0.9, 1.1),         # barely different from standard
    ]

    img = Image.open('/workspace/TRELLIS.2/assets/example_image/T.png')
    processed = pipeline.preprocess_image(img)

    results = []

    for label, cfg_mode, ll, lh in configs:
        print(f"\n--- {label} (mode={cfg_mode}, λ_low={ll}, λ_high={lh}) ---", flush=True)
        tex_params = {**base_tex, "cfg_mode": cfg_mode}
        if cfg_mode == 'fdg':
            tex_params["fdg_sigma"] = 1.0
            tex_params["fdg_lambda_low"] = ll
            tex_params["fdg_lambda_high"] = lh

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

        score_dict = verifier.score(mesh, processed, num_views=6, render_resolution=512)

        print(f"  Total: {score_dict['total']:.4f} "
              f"(dice={score_dict['silhouette_dice']:.3f}, "
              f"color={score_dict['color_match']:.3f}, "
              f"geo={score_dict['geometric']:.3f}, "
              f"tex={score_dict['tex_coherence']:.3f}, "
              f"detail={score_dict['detail_richness']:.3f}) "
              f"[{gen_time:.1f}s]",
              flush=True)

        results.append({
            'label': label,
            'cfg_mode': cfg_mode,
            'lambda_low': ll,
            'lambda_high': lh,
            'scores': score_dict,
            'gen_time': gen_time,
        })

        del outputs, mesh
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Label':15s} {'λL':>4s} {'λH':>4s} {'Total':>7s} {'Dice':>6s} {'Color':>6s} "
          f"{'Geo':>6s} {'Tex':>6s} {'Detail':>6s} {'Time':>6s}")
    print("-" * 80)

    baseline_total = results[0]['scores']['total']
    for r in results:
        s = r['scores']
        diff = s['total'] - baseline_total
        print(f"{r['label']:15s} {r['lambda_low']:4.1f} {r['lambda_high']:4.1f} "
              f"{s['total']:7.4f} {s['silhouette_dice']:6.3f} {s['color_match']:6.3f} "
              f"{s['geometric']:6.3f} {s['tex_coherence']:6.3f} {s['detail_richness']:6.3f} "
              f"{r['gen_time']:6.1f}s {diff:+.4f}")

    out_path = '/tmp/fdg_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
