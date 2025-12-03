#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: baseline sparse vs LinearOperator-based implementation.

We compare the survival probability P_ee(t) for both single-bin and two-bin cases:
  
Single-bin tests:
  - (n1, n2) pairs with N = n1+n2 ∈ {2, 10, 50}
  - μ ∈ {0.0, 0.05, 0.5}
  - θ_v ∈ {pi/4, pi/2-0.2}
  - fixed: ω = 1.0
  - time grid: l=16, s=512  → t ∈ linspace(0, 16, 512)

Two-bin tests:
  - Fixed (e1, m1, e2, m2) cases: (1,1,1,1), (1,0,0,1), (5,0,0,5), (0,5,5,0), (5,5,5,5), (25,0,0,25)
  - μ ∈ {0.0, 0.05, 0.5}
  - θ_v ∈ {pi/4, pi/2-0.2}
  - fixed: ω = 1.0 (same for both bins)
  - time grid: l=16, s=512  → t ∈ linspace(0, 16, 512)

The baseline uses explicit sparse matrices (dicke_collective_sparse_opt);
the linop version uses LinearOperators (dicke_collective_sparse_opt_linop).
We assert numerical agreement within tolerance: |Δ| ≤ atol + rtol·|baseline|, pointwise.
"""

import os
import sys
import math
import itertools
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


dc_base = _import_module("dicke_collective_sparse_opt", "dicke_collective_sparse_opt.py")
dc_linop = _import_module("dicke_collective_sparse_opt_linop", "dicke_collective_sparse_opt_linop.py")


# ---- testing core -----------------------------------------------------------------------------

def build_case_single_bin(n1, n2, mu, theta, *, l=16.0, s=512, omega=1.0):
    """
    Build and run both implementations for a single bin, returning (t, Pee_base, Pee_linop).
    """
    N = int(n1 + n2)
    if N <= 0:
        raise ValueError("N must be > 0")

    t_grid = np.linspace(0.0, float(l), int(s), dtype=float)

    # ---- baseline (explicit sparse matrices) ----
    psi0_b, S_b = dc_base.single_bin_initial_state(n1, n2)
    H_b, (Jx_b, Jy_b, Jz_b), S_list_b, dims_b = dc_base.build_single_bin_hamiltonian(
        N=N, omega=omega, theta_v=theta, mu=mu
    )
    # Convert sparse initial state to dense if needed
    if hasattr(psi0_b, 'toarray'):
        psi0_b = psi0_b.toarray().flatten()
    else:
        psi0_b = np.asarray(psi0_b).flatten()
    states_b = dc_base.evolve_times(H_b, psi0_b, t_grid)
    _, Pee_b = dc_base.bin_observables(states_b, (Jx_b, Jy_b, Jz_b), S_list_b)

    # ---- linop (LinearOperator) ----
    H_l, (Jx_l, Jy_l, Jz_l), S_list_l, dims_l = dc_linop.build_single_bin_hamiltonian(
        N=N, omega=omega, theta_v=theta, mu=mu
    )
    # Use same initial state (convert to dense)
    psi0_l = np.asarray(psi0_b, dtype=np.complex128)
    states_l = dc_linop.evolve_times(H_l, psi0_l, t_grid)
    _, Pee_l = dc_base.bin_observables(states_l, (Jx_l, Jy_l, Jz_l), S_list_l)

    # Pee_b shape (T,1); Pee_l shape (T,1) -> extract first bin
    Pee_b_flat = Pee_b[:, 0]
    Pee_l_flat = Pee_l[:, 0]

    return t_grid, Pee_b_flat, Pee_l_flat


def build_case_two_bin(e1, m1, e2, m2, mu, theta, *, l=16.0, s=512, omega=1.0):
    """
    Build and run both implementations for two bins, returning (t, Pee1_base, Pee1_linop, Pee2_base, Pee2_linop).
    """
    N1 = int(e1 + m1)
    N2 = int(e2 + m2)
    if N1 <= 0 or N2 <= 0:
        raise ValueError("N1 and N2 must be > 0")

    t_grid = np.linspace(0.0, float(l), int(s), dtype=float)

    # ---- baseline (explicit sparse matrices) ----
    psi0_b, S_list_b = dc_base.multi_bin_initial_state([e1, e2], [m1, m2])
    H_b, (Jx_list_b, Jy_list_b, Jz_list_b), S_list_b2, dims_b = dc_base.build_multi_bin_hamiltonian(
        N_list=[N1, N2], omega_list=[omega, omega], theta_v=theta, mu=mu
    )
    # Convert sparse initial state to dense if needed
    if hasattr(psi0_b, 'toarray'):
        psi0_b = psi0_b.toarray().flatten()
    else:
        psi0_b = np.asarray(psi0_b).flatten()
    states_b = dc_base.evolve_times(H_b, psi0_b, t_grid)
    _, Pee_b = dc_base.bin_observables(states_b, Jz_list_b, S_list_b)

    # ---- linop (LinearOperator) ----
    H_l, (Jx_list_l, Jy_list_l, Jz_list_l), S_list_l, dims_l = dc_linop.build_multi_bin_hamiltonian(
        N_list=[N1, N2], omega_list=[omega, omega], theta_v=theta, mu=mu
    )
    # Use same initial state (convert to dense)
    psi0_l = np.asarray(psi0_b, dtype=np.complex128)
    states_l = dc_linop.evolve_times(H_l, psi0_l, t_grid)
    _, Pee_l = dc_base.bin_observables(states_l, Jz_list_l, S_list_l)

    # Pee_b shape (T,2); Pee_l shape (T,2) -> extract both bins
    Pee_bin1_b = Pee_b[:, 0]
    Pee_bin2_b = Pee_b[:, 1]
    Pee_bin1_l = Pee_l[:, 0]
    Pee_bin2_l = Pee_l[:, 1]

    return t_grid, Pee_bin1_b, Pee_bin1_l, Pee_bin2_b, Pee_bin2_l


def max_errors(Pee_b, Pee_l):
    """Return max abs error and RMS error."""
    diff = Pee_l - Pee_b
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
    return max_abs, rms


def agrees_within(Pee_b, Pee_l, atol=2e-5, rtol=2e-5):
    """Pointwise check: |Δ| ≤ atol + rtol*|baseline|."""
    tol = atol + rtol * np.abs(Pee_b)
    ok = np.all(np.abs(Pee_l - Pee_b) <= tol)
    return bool(ok)


# ---- Single-bin test suite -------------------------------------------------------------------

def run_suite_single_bin(N_list=(2, 10, 50), mu_list=(0.0, 0.05, 0.5), 
                         theta_list=(math.pi/4, math.pi/2 - 0.2),
                         l=16.0, s=512, omega=1.0, atol=2e-5, rtol=2e-5, max_cases=None):
    """
    Run all single-bin cases and print a compact report. Returns True if all pass.
    We sweep (N, polarization p) by generating (n1,n2) pairs: fully polarized, half-half, and minimally polarized.
    """
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

    print("=" * 80)
    print("SINGLE-BIN TESTS: Baseline (sparse) vs LinearOperator")
    print("=" * 80)
    header = f"{'n1':>4} {'n2':>4} {'N':>4} {'mu':>8} {'theta':>10} | {'max_abs':>12} {'rms':>10} | {'PASS?':>6}"
    print(header)
    print("-" * len(header))
    n_pass = 0
    for (n1, n2, mu, theta) in cases:
        try:
            t, Pee_b, Pee_l = build_case_single_bin(n1, n2, mu, theta, l=l, s=s, omega=omega)
            max_abs, rms = max_errors(Pee_b, Pee_l)
            ok = agrees_within(Pee_b, Pee_l, atol=atol, rtol=rtol)
            print(f"{n1:4d} {n2:4d} {n1+n2:4d} {mu:8.3f} {theta:10.3f} | {max_abs:12.6e} {rms:10.6e} | {str(ok):>6}")
            n_pass += int(ok)
        except Exception as exc:
            print(f"{n1:4d} {n2:4d} {n1+n2:4d} {mu:8.3f} {theta:10.3f} | {'ERR':>12} {'ERR':>10} | False")
            print("   ->", repr(exc))
    total = len(cases)
    print("-" * len(header))
    print(f"Passed {n_pass}/{total} cases with atol={atol:g}, rtol={rtol:g}")
    return n_pass == total


# ---- Two-bin test suite ----------------------------------------------------------------------

def run_suite_two_bin(mu_list=(0.0, 0.05, 0.5), theta_list=(math.pi/4, math.pi/2 - 0.2),
                       l=16.0, s=512, omega=1.0, atol=2e-5, rtol=2e-5, max_cases=None):
    """
    Run all two-bin cases and print a compact report. Returns True if all pass.
    Tests only the specified (e1, m1, e2, m2) combinations with all mu and theta values.
    """
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

    print("\n" + "=" * 80)
    print("TWO-BIN TESTS: Baseline (sparse) vs LinearOperator")
    print("=" * 80)
    header = f"{'e1':>4} {'m1':>4} {'e2':>4} {'m2':>4} {'N1':>4} {'N2':>4} {'mu':>8} {'theta':>10} | {'max_abs':>12} {'rms':>10} | {'PASS?':>6}"
    print(header)
    print("-" * len(header))
    n_pass = 0
    for (e1, m1, e2, m2, mu, theta) in cases:
        try:
            t, Pee1_b, Pee1_l, Pee2_b, Pee2_l = build_case_two_bin(e1, m1, e2, m2, mu, theta, l=l, s=s, omega=omega)
            # Check both bins
            max_abs1, rms1 = max_errors(Pee1_b, Pee1_l)
            max_abs2, rms2 = max_errors(Pee2_b, Pee2_l)
            ok1 = agrees_within(Pee1_b, Pee1_l, atol=atol, rtol=rtol)
            ok2 = agrees_within(Pee2_b, Pee2_l, atol=atol, rtol=rtol)
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


# ---- Main test runner -------------------------------------------------------------------------

def run_all_tests(N_list=(2, 10, 50), mu_list=(0.0, 0.05, 0.5), 
                   theta_list=(math.pi/4, math.pi/2 - 0.2),
                   l=16.0, s=512, omega=1.0, atol=2e-5, rtol=2e-5, 
                   max_cases_single=None, max_cases_two=None):
    """
    Run both single-bin and two-bin test suites.
    """
    ok_single = run_suite_single_bin(N_list=N_list, mu_list=mu_list, theta_list=theta_list,
                                      l=l, s=s, omega=omega, atol=atol, rtol=rtol, 
                                      max_cases=max_cases_single)
    
    ok_two = run_suite_two_bin(mu_list=mu_list, theta_list=theta_list,
                                l=l, s=s, omega=omega, atol=atol, rtol=rtol,
                                max_cases=max_cases_two)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Single-bin tests: {'PASS' if ok_single else 'FAIL'}")
    print(f"Two-bin tests:    {'PASS' if ok_two else 'FAIL'}")
    print("=" * 80)
    
    return ok_single and ok_two


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Integration test: baseline sparse vs LinearOperator (single-bin and two-bin).")
    p.add_argument("--N", nargs="*", type=int, default=[2, 10, 50], help="Total neutrino counts N to sweep (single-bin).")
    p.add_argument("--mu", nargs="*", type=float, default=[0.0, 0.05, 0.5], help="Self-interaction strengths μ.")
    p.add_argument("--theta", nargs="*", type=float, default=[math.pi/4, math.pi/2 - 0.2], help="Vacuum mixing angles.")
    p.add_argument("--l", type=float, default=16.0, help="Baseline length / total time.")
    p.add_argument("--s", type=int, default=512, help="Number of sample steps along [0, l].")
    p.add_argument("--omega", type=float, default=1.0, help="Vacuum frequency ω.")
    p.add_argument("--atol", type=float, default=2e-5, help="Absolute tolerance for agreement.")
    p.add_argument("--rtol", type=float, default=2e-5, help="Relative tolerance for agreement.")
    p.add_argument("--max-cases-single", type=int, default=None, help="Limit number of single-bin cases (for quick smoke runs).")
    p.add_argument("--max-cases-two", type=int, default=None, help="Limit number of two-bin cases (for quick smoke runs).")
    p.add_argument("--single-only", action="store_true", help="Run only single-bin tests.")
    p.add_argument("--two-only", action="store_true", help="Run only two-bin tests.")
    args = p.parse_args(argv)

    if args.single_only:
        ok = run_suite_single_bin(N_list=args.N, mu_list=args.mu, theta_list=args.theta,
                                   l=args.l, s=args.s, omega=args.omega,
                                   atol=args.atol, rtol=args.rtol, max_cases=args.max_cases_single)
    elif args.two_only:
        ok = run_suite_two_bin(mu_list=args.mu, theta_list=args.theta,
                                l=args.l, s=args.s, omega=args.omega,
                                atol=args.atol, rtol=args.rtol, max_cases=args.max_cases_two)
    else:
        ok = run_all_tests(N_list=args.N, mu_list=args.mu, theta_list=args.theta,
                           l=args.l, s=args.s, omega=args.omega,
                           atol=args.atol, rtol=args.rtol,
                           max_cases_single=args.max_cases_single,
                           max_cases_two=args.max_cases_two)
    
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

