#!/usr/bin/env python3
"""Test increased shape/SS sampling steps for better geometry.

Usage: APP_SCRIPT=optimization/scripts/run_shape_steps.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'shape_steps',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
