#!/usr/bin/env python3
"""Round 5 batch: SEED VARIANCE + STAGED-BON + HIGHER STEPS.

KEY INSIGHT from R4: All post-processing methods (color_transfer, silcorr, tex_refine)
AND all sampler-level mods (APG, CFG-Zero*, CFG-MP, R-CFG++, FDG, SDE, guidance anneal)
have shown ZERO or negligible effect on TRELLIS.2 quality.

Remaining frontier: SELECTION (Best-of-N) and MORE COMPUTE (higher steps).

Priority order:
1. seed_variance: Measure cross-seed noise floor (critical for understanding BoN potential)
2. staged_bon_test: N=2,4,8 shape candidates with silhouette Dice selection
3. higher_steps: tex=20,24; all stages=16/16/20

Usage: APP_SCRIPT=optimization/scripts/run_batch_round5.py docker compose up -d
"""
import sys
import time

BATCH = [
    # Priority 1: Seed variance (4 configs × 3 images)
    # Must run FIRST to establish cross-seed noise floor
    # If seed variance is small (±0.5), BoN won't help much
    # If large (±2-3), BoN N=4-8 could give +3-5 pts
    ('seed_variance', 3),

    # Priority 2: Staged Best-of-N (4 configs × 3 images)
    # N=2 costs ~1.7x, N=4 costs ~2.5x, N=8 costs ~4x
    # Only tests SHAPE selection (texture is deterministic given shape)
    ('staged_bon_test', 3),

    # Priority 3: Higher sampling steps (4 configs × 3 images)
    # More steps = more detail, especially for texture
    # tex=24 costs ~50% more for texture stage only
    ('higher_steps', 3),
]

for preset, n_images in BATCH:
    print(f"\n{'='*60}", flush=True)
    print(f"=== BATCH R5: Starting preset '{preset}' with {n_images} images ===", flush=True)
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

    print(f"\n=== BATCH R5: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)

print("\n=== ALL BATCH R5 TESTS COMPLETE ===", flush=True)
