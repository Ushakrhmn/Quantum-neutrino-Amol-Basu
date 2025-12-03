#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: baseline (dense) two-bin Dicke vs sparse memory-patched implementation.

We compare the survival probability P_ee(t) for both bins across parameter sweeps:
  - Fixed (e1, m1, e2, m2) cases: (1,1,1,1), (1,0,0,1), (5,0,0,5), (0,5,5,0), (5,5,5,5), (25,0,0,25)
  - μ ∈ {0.0, 0.05, 0.5}
  - θ_v ∈ {pi/4, pi/2-0.2}
  - fixed: ω = 1.0 (same for both bins)
  - time grid: l=16, s=512  → t ∈ linspace(0, 16, 512)

The baseline uses exact diagonalization (dense); the mem_patched version uses sparse matrices
with the regular evolve_times method (no streaming).
We assert numerical agreement within tolerance: |Δ| ≤ atol + rtol·|baseline|, pointwise for both bins.
"""

import os, sys, math
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

def build_case(e1, m1, e2, m2, mu, theta, *, l=16.0, s=512, omega=1.0, dtype=None):
    """
    Build and run both implementations for two bins, returning (t, Pee_bin1_base, Pee_bin1_patched, Pee_bin2_base, Pee_bin2_patched).
    """
    N1 = int(e1 + m1)
    N2 = int(e2 + m2)
    if N1 <= 0 or N2 <= 0:
        raise ValueError("N1 and N2 must be > 0")

    if dtype is None:
        # Use complex128 by default - LinearOperator mode with complex64 has precision issues
        dtype = np.complex128

    t_grid = np.linspace(0.0, float(l), int(s), dtype=float)

    # ---- baseline (dense exact) ----
    psi0_b, S_list_b = dc_base.multi_bin_initial_state([e1, e2], [m1, m2])
    H_b, (Jx_list_b, Jy_list_b, Jz_list_b), S_list_b, dims_b = dc_base.build_multi_bin_hamiltonian(
        N_list=[N1, N2], omega_list=[omega, omega], theta_v=theta, mu=mu
    )
    states_b = dc_base.evolve_times(H_b, psi0_b, t_grid)
    _, Pee_b = dc_base.bin_observables(states_b, Jz_list_b, S_list_b)

    # ---- sparse memory-patched (regular evolve_times, not streaming) ----
    psi0_p, S_list_p = dc_patched.multi_bin_initial_state([e1, e2], [m1, m2], dtype=dtype)
    H_p, (Jx_list_p, Jy_list_p, Jz_list_p), S_list_p, dims_p = dc_patched.build_multi_bin_hamiltonian(
        N_list=[N1, N2], omega_list=[omega, omega], theta_v=theta, mu=mu, dtype=dtype
    )
    
    # Use regular evolve_times method (not streaming)
    states_p = dc_patched.evolve_times(H_p, psi0_p, t_grid, dtype=dtype)
    _, Pee_p = dc_patched.bin_observables(states_p, Jz_list_p, S_list_p)

    # Pee_b shape (T,2); Pee_p shape (T,2) -> extract both bins
    Pee_bin1_b = Pee_b[:, 0]
    Pee_bin2_b = Pee_b[:, 1]
    Pee_bin1_p = Pee_p[:, 0]
    Pee_bin2_p = Pee_p[:, 1]

    return t_grid, Pee_bin1_b, Pee_bin1_p, Pee_bin2_b, Pee_bin2_p


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
    Tests only the specified (e1, m1, e2, m2) combinations with all mu and theta values.
    """
    if dtype is None:
        # Use complex128 by default - LinearOperator mode with complex64 has precision issues
        dtype = np.complex128
    
    # Fixed test cases: (e1, m1, e2, m2)
    fixed_cases = [
        (1, 1, 1, 1),
        (1, 0, 0, 1),
        (5, 0, 0, 5),
        (0, 5, 5, 0),
        (5, 5, 5, 5),
        (25, 0, 0, 25),
    ]
    
    # Generate all combinations with mu_list and theta_list
    cases = []
    for (e1, m1, e2, m2) in fixed_cases:
        for mu in mu_list:
            for theta in theta_list:
                cases.append((e1, m1, e2, m2, mu, theta))

    if max_cases is not None:
        cases = cases[:int(max_cases)]

    header = f"{'e1':>4} {'m1':>4} {'e2':>4} {'m2':>4} {'N1':>4} {'N2':>4} {'mu':>8} {'theta':>10} | {'max_abs':>12} {'rms':>10} | {'PASS?':>6}"
    print(header)
    print("-" * len(header))
    n_pass = 0
    for (e1, m1, e2, m2, mu, theta) in cases:
        try:
            t, Pee1_b, Pee1_p, Pee2_b, Pee2_p = build_case(e1, m1, e2, m2, mu, theta, l=l, s=s, omega=omega, dtype=dtype)
            # Check both bins
            max_abs1, rms1 = max_errors(Pee1_b, Pee1_p)
            max_abs2, rms2 = max_errors(Pee2_b, Pee2_p)
            ok1 = agrees_within(Pee1_b, Pee1_p, atol=atol, rtol=rtol)
            ok2 = agrees_within(Pee2_b, Pee2_p, atol=atol, rtol=rtol)
            ok = ok1 and ok2
            max_abs = max(max_abs1, max_abs2)
            rms = max(rms1, rms2)
            print(f"{e1:4d} {m1:4d} {e2:4d} {m2:4d} {e1+m1:4d} {e2+m2:4d} {mu:8.3f} {theta:10.3f} | {max_abs:12.6e} {rms:10.6e} | {str(ok):>6}")
            n_pass += int(ok)
        except Exception as exc:
            print(f"{e1:4d} {m1:4d} {e2:4d} {m2:4d} {e1+m1:4d} {e2+m2:4d} {mu:8.3f} {theta:10.3f} | {'ERR':>12} {'ERR':>10} | False")
            print("   ->", repr(exc))
    total = len(cases)
    print("-" * len(header))
    print(f"Passed {n_pass}/{total} cases with atol={atol:g}, rtol={rtol:g}")
    return n_pass == total


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Integration test: dense vs sparse memory-patched two-bin (P_ee agreement).")
    p.add_argument("--N", nargs="*", type=int, default=[2, 10, 50], help="Total neutrino counts N1, N2 to sweep (same list for both bins).")
    p.add_argument("--mu", nargs="*", type=float, default=[0.0, 0.05, 0.5], help="Self-interaction strengths μ.")
    p.add_argument("--theta", nargs="*", type=float, default=[math.pi/4, math.pi/2 - 0.2], help="Vacuum mixing angles.")
    p.add_argument("--l", type=float, default=16.0, help="Baseline length / total time.")
    p.add_argument("--s", type=int, default=512, help="Number of sample steps along [0, l].")
    p.add_argument("--omega", type=float, default=1.0, help="Vacuum frequency ω (same for both bins).")
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

