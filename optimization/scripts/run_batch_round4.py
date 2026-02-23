#!/usr/bin/env python3
"""Batch Round 4: FDG, texture refinement, shape guidance, BON combos.

Run AFTER Round 3 completes. Uses V4.1 evaluator (auto-loaded from live mount).

Key untested techniques:
- FDG: Frequency-Decoupled Guidance — separate high/low freq guidance weights
- tex_refine: Render-and-compare texture optimization via nvdiffrast
- shape_gs_combined: Bell-curve guidance schedule for shape stage (A1 target)
- bon_combos: BON4 + best guidance combos

Usage: APP_SCRIPT=optimization/scripts/run_batch_round4.py docker compose up -d
"""
import sys
import time

BATCH = [
    # FDG at 1024: promising for texture detail (C3), completely untested
    # lambda_high > 1 boosts high-frequency detail, lambda_low < 1 suppresses low-freq saturation
    ('fdg', 3),               # 4 configs × 3 images

    # Texture refinement: render-and-compare optimization, targets A2/C1
    # Uses nvdiffrast differentiable rendering + chrominance L1 + proximity loss
    ('tex_refine', 1),        # 4 configs × 1 image

    # Shape guidance schedule: bell-curve for shape stage, targets A1
    # Also tests combined shape+texture guidance + BON4
    ('shape_gs_combined', 1), # 4 configs × 1 image

    # BON + guidance combos: combines Best-of-N with best guidance findings
    ('bon_combos', 1),        # 4 configs × 1 image
]

for preset, n_images in BATCH:
    print(f"\n{'='*60}", flush=True)
    print(f"=== BATCH R4: Starting preset '{preset}' with {n_images} images ===", flush=True)
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

    print(f"\n=== BATCH R4: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)

print("\n=== ALL BATCH R4 TESTS COMPLETE ===", flush=True)
