#!/usr/bin/env python3
"""Wrapper to run ab_test_runner with guidance_sched preset, 3 images."""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'guidance_sched',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
