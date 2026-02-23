#!/usr/bin/env python3
"""Test BON4 + best guidance configs (C4 combination).

BON4 alone = 93.34 overall. Beta(3,3) guidance = 92.66. Test if additive.
Each BON4 config takes ~10 min/image, so 3 configs × 1 image ≈ 30 min.

Usage: APP_SCRIPT=optimization/scripts/run_bon4_combined.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'bon4_combined',
    '--images', '1',  # BON4 is expensive, use 1 image first
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
