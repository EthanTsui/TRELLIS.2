#!/usr/bin/env python3
"""Round 2 batch tests: 64³ cascade, round3 combos, 1536 variants.

Run AFTER the first batch (sde/occ/bon/silcorr) completes.

Usage: APP_SCRIPT=optimization/scripts/run_batch_r2.py docker compose up -d
"""
import sys
import time

BATCH = [
    # 64³ native cascade at 1024 (3 configs × 3 images)
    # Tests: skip max_pool, use full decoder output
    ('cascade_64', 3),

    # Round 3: best combos from V4.1 findings (4 configs × 3 images)
    # Tests: split_b42 (C3 +5+3), split_tri_narrow (C3+A2), full_combo (+BON4)
    ('round3_best', 3),

    # 1536 resolution combos (4 configs × 3 images)
    # Tests: triangular, hi guidance, BON4+tri
    ('1536_combos', 3),

    # 64³ cascade at 1536 (3 configs × 3 images, EXPENSIVE)
    # Tests: baseline vs native_64 vs native_64+occ0.5
    ('cascade_64_1536', 3),
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
