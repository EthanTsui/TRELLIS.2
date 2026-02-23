#!/usr/bin/env python3
"""Test improved SilhouetteCorrector: Dice loss, lower regularization, higher displacement.

Usage: APP_SCRIPT=optimization/scripts/run_silcorr.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'silcorr',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
