#!/usr/bin/env python3
"""Test guidance anneal: reduce CFG near t=0 to preserve texture detail.

Usage: APP_SCRIPT=optimization/scripts/run_guidance_anneal.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'guidance_anneal',
    '--images', '1',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
