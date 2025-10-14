#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: baseline (dense) Dicke vs streaming sparse-opt implementation.

We compare the survival probability P_ee(t) for 2 bins across parameter sweeps:
  - e1 ∈ {1, 10, 50}, m2 ∈ {1, 10, 50} with e2=m1=0
  - j   ∈ {0.0, 0.1, 0.5}
  - fixed: theta = pi/2 - 0.2, energies (E1=1.0, E2=1.2), dmsq=1.0
  - time grid: l=16, s=512  → t ∈ linspace(0, 16, 512)

The baseline uses exact diagonalization (dense); the sparse-opt uses expm_multiply streaming.
We assert numerical agreement within tolerance: |Δ| ≤ atol + rtol·|baseline|, pointwise.
"""

import os, sys, math, itertools, time
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
dc_opt  = _import_module("dicke_collective_sparse_opt", "dicke_collective_sparse_opt.py")

# ---- testing core -----------------------------------------------------------------------------

def build_case(e1, m2, j, l=16.0, s=512, E1=1.0, E2=1.2, theta=None, chunk=64, normalize=False):
    """
    Build and run both implementations, returning (t, Pee_base, Pee_opt).
    """
    # Physical setup (mirror emu_2bin defaults)
    if theta is None:
        theta = np.pi/2 - 0.2
    dmsq = 1.0
    omega1, omega2 = dmsq / (2*E1), dmsq / (2*E2)
    # Two-bin composition: (e1, m1=0) in bin1; (e2=0, m2) in bin2
    e2, m1 = 0, 0
    n1, n2 = e1 + m1, e2 + m2
    n_total = n1 + n2
    mu = (j / n_total) if n_total > 0 else 0.0
    t_grid = np.linspace(0.0, float(l), int(s), dtype=float)

    # ---- baseline (dense) ----
    psi0_b, S_list_b = dc_base.multi_bin_initial_state([e1, e2], [m1, m2])
    H_b, (Jx_b, Jy_b, Jz_b), S_list_b2, dims_b = dc_base.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list_b], omega_list=[omega1, omega2], theta_v=theta, mu=mu
    )
    # Consistency sanity
    assert np.allclose(S_list_b, S_list_b2), "Baseline S_list mismatch"
    states_b = dc_base.evolve_times(H_b, psi0_b, t_grid)
    _, Pee_b = dc_base.bin_observables(states_b, Jz_b, S_list_b)

    # ---- sparse-opt (streaming) ----
    psi0_o, S_list_o = dc_opt.multi_bin_initial_state([e1, e2], [m1, m2])
    H_o, (Jx_o, Jy_o, Jz_o), S_list_o2, dims_o = dc_opt.build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list_o], omega_list=[omega1, omega2], theta_v=theta, mu=mu
    )
    assert np.allclose(S_list_o, S_list_o2), "Opt S_list mismatch"
    t_stream, Pee_o = dc_opt.compute_pe_stream(H_o, psi0_o, t_grid, Jz_o, S_list_o, chunk=chunk, normalize=normalize)

    # Defensive: ensure matching time grids
    if not np.allclose(t_stream, t_grid):
        raise AssertionError("Time grids differ between implementations.")

    return t_grid, Pee_b, Pee_o


def max_errors(Pee_b, Pee_o):
    """Return per-bin and global max abs error, and RMS error."""
    diff = Pee_o - Pee_b
    max_abs_per_bin = np.max(np.abs(diff), axis=0)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(np.abs(diff)**2)))
    return max_abs_per_bin, max_abs, rms


def agrees_within(Pee_b, Pee_o, atol=5e-6, rtol=5e-6):
    """Pointwise check: |Δ| ≤ atol + rtol*|baseline|."""
    tol = atol + rtol * np.abs(Pee_b)
    ok = np.all(np.abs(Pee_o - Pee_b) <= tol)
    return bool(ok)


def run_suite(e_list=(1,10,50), m_list=(1,10,50), j_list=(0.0,0.1,0.5),
              l=16.0, s=512, E1=1.0, E2=1.2, theta=None, chunk=64, normalize=False,
              atol=5e-6, rtol=5e-6, max_cases=None):
    """
    Run all cases and print a compact report. Returns True if all pass.
    """
    cases = list(itertools.product(e_list, m_list, j_list))
    if max_cases is not None:
        cases = cases[:int(max_cases)]
    header = f"{'e1':>4} {'m2':>4} {'j':>6} | {'max_abs(bin1)':>14} {'max_abs(bin2)':>14} {'max_abs(all)':>12} {'rms':>10} | {'PASS?':>6}"
    print(header)
    print("-"*len(header))
    n_pass = 0
    for (e1, m2, j) in cases:
        try:
            t, Pee_b, Pee_o = build_case(e1, m2, j, l=l, s=s, E1=E1, E2=E2, theta=theta, chunk=chunk, normalize=normalize)
            per_bin, max_abs, rms = max_errors(Pee_b, Pee_o)
            ok = agrees_within(Pee_b, Pee_o, atol=atol, rtol=rtol)
            print(f"{e1:4d} {m2:4d} {j:6.3f} | {per_bin[0]:14.6e} {per_bin[1]:14.6e} {max_abs:12.6e} {rms:10.6e} | {str(ok):>6}")
            n_pass += int(ok)
        except Exception as exc:
            print(f"{e1:4d} {m2:4d} {j:6.3f} | {'ERR':>14} {'ERR':>14} {'ERR':>12} {'ERR':>10} | False")
            print("   ->", repr(exc))
    total = len(cases)
    print("-"*len(header))
    print(f"Passed {n_pass}/{total} cases with atol={atol:g}, rtol={rtol:g}")
    return n_pass == total


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Integration test: baseline dense vs streaming sparse-opt (P_ee agreement).")
    p.add_argument("--e1", nargs="*", type=int, default=[1,10,50], help="Values for e1 (bin1 electron count).")
    p.add_argument("--m2", nargs="*", type=int, default=[1,10,50], help="Values for m2 (bin2 muon count).")
    p.add_argument("--j", nargs="*", type=float, default=[0.0,0.1,0.5], help="Values for uniform coupling j (mu = j/N).")
    p.add_argument("--l", type=float, default=16.0, help="Baseline length / total time.")
    p.add_argument("--s", type=int, default=512, help="Number of sample steps along [0, l].")
    p.add_argument("--E1", type=float, default=1.0, help="Energy of bin 1 (controls omega1 = dmsq/(2E1)).")
    p.add_argument("--E2", type=float, default=1.2, help="Energy of bin 2.")
    p.add_argument("--theta", type=float, default=None, help="Vacuum mixing angle. Default pi/2-0.2.")
    p.add_argument("--chunk", type=int, default=64, help="Streaming expm_multiply block size.")
    p.add_argument("--normalize", action="store_true", help="L2-normalize |psi| at each step (streaming).")
    p.add_argument("--atol", type=float, default=5e-6, help="Absolute tolerance for agreement.")
    p.add_argument("--rtol", type=float, default=5e-6, help="Relative tolerance for agreement.")
    p.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (for quick smoke runs).")
    args = p.parse_args(argv)

    ok = run_suite(e_list=args.e1, m_list=args.m2, j_list=args.j,
                   l=args.l, s=args.s, E1=args.E1, E2=args.E2, theta=args.theta,
                   chunk=args.chunk, normalize=args.normalize,
                   atol=args.atol, rtol=args.rtol, max_cases=args.max_cases)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
