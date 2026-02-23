#!/usr/bin/env python3
"""Test guidance schedule v2: triangular, asymmetric beta, and combinations.

Usage: APP_SCRIPT=optimization/scripts/run_guidance_sched2.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'guidance_sched2',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
