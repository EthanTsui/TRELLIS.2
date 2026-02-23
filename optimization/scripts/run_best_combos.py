#!/usr/bin/env python3
"""Wrapper to run ab_test_runner with best_combos preset, 3 images.

Usage: APP_SCRIPT=optimization/scripts/run_best_combos.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'best_combos',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
