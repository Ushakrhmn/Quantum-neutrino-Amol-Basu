#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_nu_calibration.py

Calibration test: single-bin pure ν_e gas.

Compares:
  - Exact many-body Dicke evolution (spin-S representation)
  - Mean-field (Raffelt–Sigl polarization vectors, 2-flavour)

Physics setup:
  - One energy bin with N identical ν_e.
  - Vacuum mixing (θ, Δm², E).
  - Neutrino self-interaction with collective coupling μ.

Coupling convention:
  - Mean-field EOM (Raffelt–Sigl) uses:
        dP_k/dt = (ω_k B + μ D) × P_k
    where D = Σ P_j.
  - In this script, the CLI parameter `--mu` is this μ.

  - The Dicke Hamiltonian for one bin uses
        H_int = μ_H * J_tot^2
    with J_tot the collective flavour spin.
    Matching the mean-field μ gives
        μ_H = 2 * μ / N
    which is what we use below.
"""

import numpy as np
import matplotlib.pyplot as plt

import mft
import dicke_collective_sparse_opt as dc


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibration: pure ν_e gas (single bin) vs mean-field."
    )
    parser.add_argument(
        "--N", type=int, default=4,
        help="Number of neutrinos in the gas"
    )
    parser.add_argument(
        "--energy", type=float, default=1.0,
        help="Neutrino energy E (same units as |Δm²| / ω)"
    )
    parser.add_argument(
        "--dmsq", type=float, default=-1.0,
        help="Mass-squared splitting Δm² (consistent with E)"
    )
    parser.add_argument(
        "--theta", type=float, default=1e-3,
        help="Vacuum mixing angle θ in radians"
    )
    parser.add_argument(
        "--mu", type=float, default=1.0,
        help="Collective coupling μ in the mean-field EOM"
    )
    parser.add_argument(
        "--L", type=float, default=15.0,
        help="Maximum evolution time / baseline"
    )
    parser.add_argument(
        "--steps", type=int, default=512,
        help="Number of evolution steps"
    )
    parser.add_argument(
        "--savename", type=str, default="pure_nu_calibration",
        help="Base name for the output PNG figure"
    )

    args = parser.parse_args()

    # --- Basic kinematics ---
    N = args.N
    t_grid = np.linspace(0.0, args.L, args.steps)
    omega = args.dmsq / (2.0 * args.energy)

    print(f"# Pure ν_e gas calibration")
    print(f"# N = {N}, E = {args.energy}, Δm² = {args.dmsq}, θ = {args.theta}, μ = {args.mu}")
    print(f"# ω = Δm² / (2E) = {omega}")

    # ------------------------------------------------------------------
    # 1. Mean-field evolution (Raffelt–Sigl polarization vectors)
    # ------------------------------------------------------------------
    # Uniform all-to-all coupling: J_{ij} = μ / N
    J_mat = args.mu / N * np.ones((N, N), dtype=float)
    omega_vec = np.full(N, omega, dtype=float)
    init_flavors = ["e"] * N  # all ν_e initially

    sol = mft.P_osc_RS(
        t_table=t_grid,
        theta=args.theta,
        omega=omega_vec,
        lam=0.0,               # no ordinary matter here
        J=J_mat,
        initial_flavors=init_flavors
    )

    # Reshape: y has shape (3*N, len(t))
    P = np.reshape(sol.y, (N, 3, len(t_grid)))
    Pz = P[:, 2, :]                   # z-component per mode
    Pee_mft_modes = 0.5 * (1.0 + Pz)  # P_e for each neutrino
    Pee_mft = Pee_mft_modes.mean(axis=0)  # average over all neutrinos

    # ------------------------------------------------------------------
    # 2. Dicke many-body evolution (single bin)
    # ------------------------------------------------------------------
    # Initial state: one bin with N ν_e (spin-up), 0 ν_μ (spin-down)
    psi0, S = dc.single_bin_initial_state(n1=N, n2=0)

    # Matching μ between mean-field and Dicke:
    # H_int = μ_H * J_tot^2 with μ_H = 2 * μ / N
    mu_H = 2.0 * args.mu / N

    H, (Jx, Jy, Jz), S_list, dims = dc.build_single_bin_hamiltonian(
        N=N,
        omega=omega,
        theta_v=args.theta,
        mu=mu_H,
    )

    Y = dc.evolve_times(H, psi0, t_grid)

    # bin_observables returns <Jz> and P_ee = 1/2 * (1 + <Jz>/S)
    _, Pee_dicke = dc.bin_observables(
        states=Y,
        Jz_list=(Jx, Jy, Jz),
        S_list=S_list
    )
    Pee_dicke = Pee_dicke[:, 0]  # single bin

    # ------------------------------------------------------------------
    # 3. Residuals and diagnostics
    # ------------------------------------------------------------------
    residual = Pee_mft - Pee_dicke
    max_resid = np.max(np.abs(residual))
    mean_resid = np.mean(np.abs(residual))

    print(f"# Max |MFT - Dicke| = {max_resid:.3e}")
    print(f"# Mean |MFT - Dicke| = {mean_resid:.3e}")

    # ------------------------------------------------------------------
    # 4. Plot: P_e(t) and residuals
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        sharex=True,
        height_ratios=[2, 1]
    )

    # Top: probabilities
    ax1.plot(t_grid, Pee_dicke, label="Dicke (exact)", color="C0")
    ax1.plot(t_grid, Pee_mft,   label="MFT (polarization vectors)", color="C1", linestyle="--")
    ax1.set_ylabel(r"$P_{e}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax1.set_title(
        rf"Pure $\nu_e$ gas calibration: "
        rf"$N={N}$, $\theta={args.theta:.3g}$, "
        rf"$\Delta m^2={args.dmsq:.3g}$, $E={args.energy:.3g}$, "
        rf"$\mu={args.mu:.3g}$"
    )

    # Bottom: residuals
    ax2.plot(t_grid, residual, label="MFT - Dicke", color="C2")
    ax2.set_xlabel("t (baseline / arbitrary units)")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    ax2.text(
        0.01, 0.99,
        f"max |ΔP| = {max_resid:.2e}\nmean |ΔP| = {mean_resid:.2e}",
        transform=ax2.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    out_name = args.savename + ".png"
    fig.savefig(out_name, dpi=150)
    print(f"# Figure saved to {out_name}")
    plt.show()


if __name__ == "__main__":
    main()
