#!/usr/bin/env python3
"""Batch Round 3: 64³ cascade + best combos + 1536 combos.

Requires container restart to pick up trellis2_image_to_3d.py changes.

Usage: APP_SCRIPT=optimization/scripts/run_batch_round3.py docker compose up -d
"""
import sys
import time

BATCH = [
    # 64³ cascade at 1024: test A1 improvement from native 64³ occupancy
    # Expected: +3-5 pts A1, ~2x slower LR pass
    ('cascade_64', 3),

    # Round 3 best combos at 1024: combine best schedule/guidance findings
    ('round3_best', 3),

    # 64³ cascade at 1536: higher res amplifies A1 benefit
    # EXPENSIVE: 1536 + 64³ LR pass, ~15-20 min per run
    ('cascade_64_1536', 1),

    # 1536 combos: guidance/schedule tuning at 1536 resolution
    ('1536_combos', 1),
]

for preset, n_images in BATCH:
    print(f"\n{'='*60}", flush=True)
    print(f"=== BATCH R3: Starting preset '{preset}' with {n_images} images ===", flush=True)
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

    print(f"\n=== BATCH R3: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)

print("\n=== ALL BATCH R3 TESTS COMPLETE ===", flush=True)
