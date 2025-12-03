#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: baseline (dense) single-bin Dicke vs sparse memory-patched implementation.

We compare the survival probability P_ee(t) for a single bin across parameter sweeps:
  - (n1, n2) pairs with N = n1+n2 ∈ {2, 10, 50}
  - μ ∈ {0.0, 0.05, 0.5}
  - θ_v ∈ {pi/4, pi/2-0.2}
  - fixed: ω = 1.0
  - time grid: l=16, s=512  → t ∈ linspace(0, 16, 512)

The baseline uses exact diagonalization (dense); the mem_patched version uses sparse matrices
with the regular evolve_times method (no streaming).
We assert numerical agreement within tolerance: |Δ| ≤ atol + rtol·|baseline|, pointwise.
"""

import os, sys, math, itertools
import numpy as np


# ---- dynamic local import helpers --------------------------------------------------------------

def _import_module(name: str, filename: str):
    """
    Try importing by name; if it fails, import from file located next to this script.
    """
    try:
        return __import__(name)
    except Exception:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, filename)
        if not os.path.exists(path):
            raise
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


dc_base = _import_module("dicke_collective", "dicke_collective.py")
dc_patched = _import_module("dicke_linearop", "dicke_linearop.py")


# ---- testing core -----------------------------------------------------------------------------

def build_case(n1, n2, mu, theta, *, l=16.0, s=512, omega=1.0, dtype=None):
    """
    Build and run both implementations for a single bin, returning (t, Pee_base, Pee_patched).
    """
    N = int(n1 + n2)
    if N <= 0:
        raise ValueError("N must be > 0")

    if dtype is None:
        # Use complex128 by default - LinearOperator mode with complex64 has precision issues
        dtype = np.complex128

    t_grid = np.linspace(0.0, float(l), int(s), dtype=float)

    # ---- baseline (dense exact) ----
    psi0_b, S_b = dc_base.single_bin_initial_state(n1, n2)
    H_b, (Jx_b, Jy_b, Jz_b), S_list_b, dims_b = dc_base.build_single_bin_hamiltonian(
        N=N, omega=omega, theta_v=theta, mu=mu
    )
    states_b = dc_base.evolve_times(H_b, psi0_b, t_grid)
    _, Pee_b = dc_base.bin_observables(states_b, (Jx_b, Jy_b, Jz_b), S_list_b)

    # ---- sparse memory-patched (regular evolve_times, not streaming) ----
    psi0_p, S_p = dc_patched.single_bin_initial_state(n1, n2, dtype=dtype)
    H_p, (Jx_p, Jy_p, Jz_p), S_list_p, dims_p = dc_patched.build_single_bin_hamiltonian(
        N=N, omega=omega, theta_v=theta, mu=mu, dtype=dtype
    )
    
    # Use regular evolve_times method (not streaming)
    states_p = dc_patched.evolve_times(H_p, psi0_p, t_grid, dtype=dtype)
    _, Pee_p = dc_patched.bin_observables(states_p, (Jx_p, Jy_p, Jz_p), S_list_p)

    # Pee_b shape (T,1); Pee_p shape (T,1) -> flatten for comparison
    Pee_b_flat = Pee_b[:, 0]
    Pee_p_flat = Pee_p[:, 0]

    return t_grid, Pee_b_flat, Pee_p_flat


def max_errors(Pee_b, Pee_p):
    """Return max abs error and RMS error."""
    diff = Pee_p - Pee_b
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
    return max_abs, rms


def agrees_within(Pee_b, Pee_p, atol=2e-5, rtol=2e-5):
    """Pointwise check: |Δ| ≤ atol + rtol*|baseline|."""
    tol = atol + rtol * np.abs(Pee_b)
    ok = np.all(np.abs(Pee_p - Pee_b) <= tol)
    return bool(ok)


def run_suite(N_list=(2, 10, 50), mu_list=(0.0, 0.05, 0.5), theta_list=(math.pi/4, math.pi/2 - 0.2),
              l=16.0, s=512, omega=1.0, dtype=None,
              atol=2e-5, rtol=2e-5, max_cases=None):
    """
    Run all cases and print a compact report. Returns True if all pass.
    We sweep (N, polarization p) by generating (n1,n2) pairs: fully polarized, half-half, and minimally polarized.
    """
    if dtype is None:
        # Use complex128 by default - LinearOperator mode with complex64 has precision issues
        dtype = np.complex128
        
    cases_raw = list(itertools.product(N_list, mu_list, theta_list))
    # Generate (n1,n2) choices per N: (N,0), (N//2, N-N//2), (1, N-1) when valid
    cases = []
    for (N, mu, theta) in cases_raw:
        pols = []
        pols.append((N, 0))
        pols.append((N // 2, N - (N // 2)))
        if N >= 2:
            pols.append((1, N - 1))
        # de-dup
        seen = set()
        uniq = []
        for a, b in pols:
            key = (int(a), int(b))
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        for (n1, n2) in uniq:
            cases.append((n1, n2, mu, theta))

    if max_cases is not None:
        cases = cases[:int(max_cases)]

    header = f"{'n1':>4} {'n2':>4} {'N':>4} {'mu':>8} {'theta':>10} | {'max_abs':>12} {'rms':>10} | {'PASS?':>6}"
    print(header)
    print("-" * len(header))
    n_pass = 0
    for (n1, n2, mu, theta) in cases:
        try:
            t, Pee_b, Pee_p = build_case(n1, n2, mu, theta, l=l, s=s, omega=omega, dtype=dtype)
            max_abs, rms = max_errors(Pee_b, Pee_p)
            ok = agrees_within(Pee_b, Pee_p, atol=atol, rtol=rtol)
            print(f"{n1:4d} {n2:4d} {n1+n2:4d} {mu:8.3f} {theta:10.3f} | {max_abs:12.6e} {rms:10.6e} | {str(ok):>6}")
            n_pass += int(ok)
        except Exception as exc:
            print(f"{n1:4d} {n2:4d} {n1+n2:4d} {mu:8.3f} {theta:10.3f} | {'ERR':>12} {'ERR':>10} | False")
            print("   ->", repr(exc))
    total = len(cases)
    print("-" * len(header))
    print(f"Passed {n_pass}/{total} cases with atol={atol:g}, rtol={rtol:g}")
    return n_pass == total


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Integration test: dense vs sparse memory-patched single-bin (P_ee agreement).")
    p.add_argument("--N", nargs="*", type=int, default=[2, 10, 50], help="Total neutrino counts N to sweep.")
    p.add_argument("--mu", nargs="*", type=float, default=[0.0, 0.05, 0.5], help="Self-interaction strengths μ.")
    p.add_argument("--theta", nargs="*", type=float, default=[math.pi/4, math.pi/2 - 0.2], help="Vacuum mixing angles.")
    p.add_argument("--l", type=float, default=16.0, help="Baseline length / total time.")
    p.add_argument("--s", type=int, default=512, help="Number of sample steps along [0, l].")
    p.add_argument("--omega", type=float, default=1.0, help="Vacuum frequency ω.")
    p.add_argument("--dtype", type=str, default="complex128", choices=["complex64", "complex128"], help="Numeric dtype (default: complex128 for better precision with LinearOperators).")
    p.add_argument("--atol", type=float, default=2e-5, help="Absolute tolerance for agreement.")
    p.add_argument("--rtol", type=float, default=2e-5, help="Relative tolerance for agreement.")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (for quick smoke runs).")
    args = p.parse_args(argv)

    dtype = np.complex64 if args.dtype == "complex64" else np.complex128

    ok = run_suite(N_list=args.N, mu_list=args.mu, theta_list=args.theta,
                   l=args.l, s=args.s, omega=args.omega, dtype=dtype,
                   atol=args.atol, rtol=args.rtol, max_cases=args.max_cases)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

