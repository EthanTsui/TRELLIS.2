#!/usr/bin/env python3
"""Test stochastic SDE sampling for texture detail improvement (C3).

Converts deterministic ODE to SDE with controlled noise injection.
Tests zero_ends and sqrt_t diffusion profiles at varying alpha strengths.

Usage: APP_SCRIPT=optimization/scripts/run_sde_test.py docker compose up -d
"""
import sys
sys.argv = [
    'ab_test_runner.py',
    '--preset', 'sde_sampling',
    '--images', '3',
]
exec(open('/workspace/TRELLIS.2/optimization/scripts/ab_test_runner.py').read())
