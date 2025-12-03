#!/usr/bin/env python3
"""Quick debug script to test linearop integration."""
import sys
sys.path.insert(0, '/home/theo1701/scratch/QNova/dicke')

import numpy as np
from test_integration_singlebin_linearop import build_case

try:
    t, Pee_b, Pee_l = build_case(n1=2, n2=0, mu=0.0, theta=np.pi/4, l=2.0, s=10)
    print("SUCCESS!")
    print(f"Time points: {len(t)}")
    print(f"Pee_baseline: {Pee_b[:3]}")
    print(f"Pee_linearop: {Pee_l[:3]}")
    print(f"Max diff: {np.max(np.abs(Pee_b - Pee_l))}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

