#!/usr/bin/env python3
"""Test staged Best-of-N shape selection for A1 silhouette improvement.

Generates N shape candidates, picks best by silhouette Dice, then textures only the winner.
Cost: ~Nx shape + 1x texture (vs Nx full pipeline for regular Best-of-N).

Usage: APP_SCRIPT=optimization/scripts/run_staged_bon.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'staged_bon',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
