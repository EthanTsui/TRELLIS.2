#!/usr/bin/env python3
"""Run multiple A/B test presets sequentially.

After each preset completes, automatically starts the next one.
Results are saved to separate JSON files per preset.

Usage: APP_SCRIPT=optimization/scripts/run_batch_tests.py docker compose up -d
"""
import sys
import time

# Define the batch of presets to run
BATCH = [
    # ('guidance_sched2', 3),  # DONE — completed in prior run
    ('sde_sampling', 3),      # stochastic SDE with various alpha values
    ('occ_threshold', 3),     # occupancy threshold sweep for A1 silhouette
    ('staged_bon', 1),        # staged Best-of-N (expensive, 1 image only)
    ('silcorr', 1),           # silhouette corrector (1 image for quick check)
]

for preset, n_images in BATCH:
    print(f"\n{'='*60}", flush=True)
    print(f"=== BATCH: Starting preset '{preset}' with {n_images} images ===", flush=True)
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

    print(f"\n=== BATCH: Completed preset '{preset}' ===\n", flush=True)
    time.sleep(5)  # brief pause between presets

print("\n=== ALL BATCH TESTS COMPLETE ===", flush=True)
