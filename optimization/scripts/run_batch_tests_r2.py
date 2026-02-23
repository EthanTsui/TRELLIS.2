#!/usr/bin/env python3
"""Round 2 batch tests — V4.1 evaluator, cascade_64, round3_best, 1536_combos.

Run AFTER restarting container to pick up V4.1 evaluator changes.

Usage: APP_SCRIPT=optimization/scripts/run_batch_tests_r2.py docker compose up -d
"""
import sys
import time

# Define the batch of presets to run
BATCH = [
    # (preset_name, n_images)
    ('cascade_64', 1),        # Quick: 64³ native cascade test (1 image, 3 configs)
    ('cascade_64_1536', 1),   # 64³ at 1536 resolution (expensive, 1 image)
    ('round3_best', 3),       # Round 3 combined (4 configs × 3 images)
    ('1536_combos', 1),       # 1536 resolution combos (4 configs × 1 image)
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
        pass  # ab_test_runner may call sys.exit()
    except Exception as e:
        print(f"ERROR in preset '{preset}': {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"\n=== BATCH R2: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)  # brief pause between presets

print("\n=== ALL BATCH R2 TESTS COMPLETE ===", flush=True)
