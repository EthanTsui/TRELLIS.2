#!/usr/bin/env python3
"""Batch Round 2: test new features requiring container restart.

Must restart container first to pick up code changes:
- SDE sampling (flow_euler.py)
- Occupancy threshold (trellis2_image_to_3d.py)
- Fixed TextureRefiner (texture_refiner.py)

Usage: APP_SCRIPT=optimization/scripts/run_batch_round2.py docker compose up -d
"""
import sys
import time

BATCH = [
    ('sde_sampling', 3),       # stochastic SDE noise injection (new code)
    ('occ_threshold', 3),      # occupancy threshold for A1 silhouette (new code)
    ('sv_hull', 3),            # single-view visual hull carving (A1 improvement)
    ('rescale_anneal', 3),     # guidance rescale anneal near t=0 (new code, free)
    ('tex_refine', 1),         # fixed texture refiner (no TV, proximity loss)
    ('shape_gs_combined', 3),  # shape guidance schedules (A1 improvement)
    ('bon4_combined', 1),      # BON4 + best configs (1 image for cost)
]

for preset, n_images in BATCH:
    print(f"\n{'='*60}", flush=True)
    print(f"=== BATCH R2: Starting preset '{preset}' with {n_images} images ===", flush=True)
    print(f"{'='*60}\n", flush=True)

    sys.argv = [
        'ab_test_runner.py',
        '--preset', preset,
        '--images', str(n_images),
    ]

    try:
        exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
    except SystemExit:
        pass
    except Exception as e:
        print(f"ERROR in preset '{preset}': {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"\n=== BATCH R2: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)

print("\n=== ALL BATCH R2 TESTS COMPLETE ===", flush=True)
