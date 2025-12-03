#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
micro_beam_background_test.py

Microphysical "beam in a background" test à la Friedland–Lunardini.

Goal
----
We model a single neutrino "beam" interacting with a background of many
(neutrinos in a 2-flavor toy model) and compare two forms of the 2-body
interaction between the beam bin and the background bin:

  1) "physical":
       H_int ∝ - Jx_beam Jx_bg + Jy_beam Jy_bg - Jz_beam Jz_bg

     This is the spin-1/2 uplift of the 4×4 matrix with diagonal
     [-2,-1,-1,-2] and corner elements -1, as derived from the
     Dedin Neto–Kemp forward-scattering formalism for ν–ν̄. Up to
     an overall factor and an additive identity, it reproduces the
     mean-field Hamiltonian used in density-matrix treatments. 

  2) "sigma":
       H_int ∝ Jx_beam Jx_bg + Jy_beam Jy_bg + Jz_beam Jz_bg

     This is the simple σ·σ / Heisenberg interaction used in many
     older toy models.

We focus on the short-time limit t → 0 and measure the initial
curvature of the beam survival probability,

    P_ee(t) ≈ 1 - C t^2 + O(t^3),

for different background sizes N_bg and different background flavor
angles α defined by

    |ν_x(α)⟩ = cos(α) |ν_e⟩ + sin(α) |ν_x⟩

with all background particles in the same state |ν_x(α)⟩.

According to the microscopic analysis of Friedland & Lunardini :

  - For generic α, coherent forward scattering produces an off-diagonal
    term ∝ sin(2α) in the effective one-particle Hamiltonian for the beam.

  - For a *background orthogonal to the beam flavor* (α = π/2, i.e.
    background = pure |ν_x⟩ if beam is |ν_e⟩), the coherent amplitude
    for flavor conversion vanishes; there should be no O(N_bg^2) coherent
    enhancement, and the forward-scattering Hamiltonian is diagonal in
    flavor.

This script numerically checks whether

  - the "physical" interaction above reproduces this behavior:
        C_phys(α = π/2) ≈ 0
  - the σ·σ interaction does *not*: it produces a non-zero curvature
        C_sigma(α = π/2) ≠ 0

Implementation notes
--------------------
- We work in a 2-bin Dicke basis:
    bin 0: "beam" (N_beam spins, usually N_beam = 1)
    bin 1: "background" (N_bg spins)

- The background initial state is constructed as a *spin coherent state*
  by starting from |S_bg, m=S_bg⟩ (all ν_e) and applying a rotation
  around the y-axis by angle 2α:

    |bg(α)⟩ = e^{-i (2α) J_y_bg} |S_bg, S_bg⟩

  which corresponds to all spins in the single-particle state
  cos(α)|ν_e⟩ + sin(α)|ν_x⟩.

- We *omit* vacuum mixing and self-interactions within a bin, so the
  only dynamics comes from the cross-bin 2-body term. This isolates
  the microscopic structure of the forward-scattering interaction.

Requirements
------------
This script assumes that `dicke_collective_sparse_opt.py` is available
in the Python path and exposes:

    - spin_matrices(S)
    - kron_on_slot(op, slot, dims)
    - multi_bin_initial_state(n1_list, n2_list)
    - evolve_times(H, psi0, t_grid)
    - bin_observables(states, Jz_list, S_list)

which are provided by the accompanying Dicke solver. 
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import pi

import dicke_collective_sparse_opt as dc
from scipy.sparse.linalg import expm as sparse_expm


# ----------------------------------------------------------------------
# Helpers: build operators, initial coherent background, Hamiltonians
# ----------------------------------------------------------------------

def prepare_system(N_beam, N_bg, alpha):
    """
    Build:
        - initial many-body state |ψ0(α)⟩ = |beam=ν_e⟩ ⊗ |bg(α)⟩
        - spin operators Jx0,Jy0,Jz0 for beam; Jx1,Jy1,Jz1 for background
        - spin magnitudes S_list = [S_beam, S_bg]

    Parameters
    ----------
    N_beam : int
        Number of neutrinos in the beam bin.
    N_bg : int
        Number of neutrinos in the background bin.
    alpha : float
        Background flavor angle (radians) in
            |ν_x(α)⟩ = cos(α)|ν_e⟩ + sin(α)|ν_x⟩.

    Returns
    -------
    psi0_rot : sparse or dense vector
        Initial state with rotated background.
    S_list : list[float]
        Spin magnitudes per bin [S_beam, S_bg].
    ops : tuple
        (Jx0, Jy0, Jz0, Jx1, Jy1, Jz1) as sparse matrices on full space.
    """
    # Start from product Dicke state: beam ν_e, background ν_e
    psi0, S_list = dc.multi_bin_initial_state(
        n1_list=[N_beam, N_bg],  # ν_e counts
        n2_list=[0,      0    ]  # ν_x counts
    )
    S_beam, S_bg = S_list
    dims = [int(2 * S_beam + 1), int(2 * S_bg + 1)]

    # Local spin operators for each bin
    Jx_b, Jy_b, Jz_b = dc.spin_matrices(S_beam)
    Jx_bg, Jy_bg, Jz_bg = dc.spin_matrices(S_bg)

    # Lift to full tensor space
    Jx0 = dc.kron_on_slot(Jx_b, 0, dims)
    Jy0 = dc.kron_on_slot(Jy_b, 0, dims)
    Jz0 = dc.kron_on_slot(Jz_b, 0, dims)

    Jx1 = dc.kron_on_slot(Jx_bg, 1, dims)
    Jy1 = dc.kron_on_slot(Jy_bg, 1, dims)
    Jz1 = dc.kron_on_slot(Jz_bg, 1, dims)

    # Rotate background by 2α around y:
    #   |bg(α)⟩ = exp(-i 2α J_y_bg) |S_bg, S_bg⟩
    theta_rot = 2.0 * alpha
    U_bg = sparse_expm(-1j * theta_rot * Jy1)
    psi0_rot = U_bg.dot(psi0)

    ops = (Jx0, Jy0, Jz0, Jx1, Jy1, Jz1)
    return psi0_rot, S_list, ops


def build_cross_hamiltonian(mu, ops, model="physical"):
    """
    Build the cross-bin Hamiltonian:

        H_int = μ * PairTerm(J_beam, J_bg)

    where

      - "physical": PairTerm = -Jx0 Jx1 + Jy0 Jy1 - Jz0 Jz1
      - "sigma":    PairTerm =  Jx0 Jx1 + Jy0 Jy1 + Jz0 Jz1

    Parameters
    ----------
    mu : float
        Interaction strength (sets overall scale of time).
    ops : tuple
        (Jx0, Jy0, Jz0, Jx1, Jy1, Jz1)
    model : str
        "physical" or "sigma".

    Returns
    -------
    H : sparse matrix
        Many-body Hamiltonian on the 2-bin Dicke space.
    """
    Jx0, Jy0, Jz0, Jx1, Jy1, Jz1 = ops

    if model.lower() == "physical":
        H = mu * (-Jx0 @ Jx1 + Jy0 @ Jy1 - Jz0 @ Jz1)
    elif model.lower() == "sigma":
        H = mu * (Jx0 @ Jx1 + Jy0 @ Jy1 + Jz0 @ Jz1)
    else:
        raise ValueError(f"Unknown model '{model}', use 'physical' or 'sigma'.")

    return H


def evolve_beam_probability(H, psi0, S_list, Jz0, t_max, steps):
    """
    Evolve |psi0> under H and compute P_ee(t) for the beam bin.

    Parameters
    ----------
    H : sparse matrix
        Hamiltonian.
    psi0 : sparse or dense vector
        Initial state.
    S_list : list[float]
        [S_beam, S_bg].
    Jz0 : sparse matrix
        J_z operator for the beam bin on full space.
    t_max : float
        Maximum evolution time.
    steps : int
        Number of time samples.

    Returns
    -------
    t_grid : ndarray
        Time samples.
    P_ee_beam : ndarray
        Beam survival probability P_ee(t).
    """
    t_grid = np.linspace(0.0, t_max, steps)
    states = dc.evolve_times(H, psi0, t_grid)

    # Use bin_observables to get P_ee for each bin
    Jz_list = [Jz0]  # we only care about beam, but could pass both bins as well
    # For consistency with API, we pass Jz for both bins:
    #   Jz_list = [Jz0, Jz1]
    # but bin_observables accepts fewer entries as well.
    Jz_t, Pee_t = dc.bin_observables(states, [Jz0], [S_list[0]])

    # Pee_t has shape (T, 1) in this call
    P_ee_beam = Pee_t[:, 0]
    return t_grid, P_ee_beam


def estimate_curvature(t, P, n_fit=10):
    """
    Estimate the initial curvature C in P(t) ≈ 1 - C t^2 by fitting
    a quadratic polynomial to the first n_fit points.

    Parameters
    ----------
    t : ndarray
        Time samples (t[0] should be 0).
    P : ndarray
        Probabilities P_ee(t).
    n_fit : int
        Number of early-time points to use in the fit.

    Returns
    -------
    C : float
        Estimated curvature (should be ≥ 0 for small oscillations).
    coefs : ndarray
        Polynomial coefficients [a2, a1, a0] such that
            P_fit(t) ≈ a2 t^2 + a1 t + a0.
    """
    n_fit = min(n_fit, len(t))
    x = t[:n_fit]
    y = P[:n_fit]
    coefs = np.polyfit(x, y, 2)
    a2, a1, a0 = coefs
    C = -a2
    return C, coefs


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Microphysical beam-in-background test (Friedland–Lunardini style)."
    )
    parser.add_argument("--N_beam", type=int, default=1,
                        help="Number of neutrinos in the beam bin (default: 1).")
    parser.add_argument("--Nbg_list", type=str, default="1,2,3,4",
                        help="Comma-separated list of background sizes, e.g. '1,2,4,8'.")
    parser.add_argument("--alphas", type=str,
                        default=f"{pi/4:.6f},{pi/2:.6f},{3*pi/4:.6f}",
                        help="Comma-separated list of α values (radians). "
                             "Default: π/4, π/2, 3π/4.")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Interaction strength μ (sets overall time scale).")
    parser.add_argument("--tmax", type=float, default=0.3,
                        help="Maximum evolution time (should keep μ tmax << 1).")
    parser.add_argument("--steps", type=int, default=81,
                        help="Number of time steps for evolution.")
    parser.add_argument("--savename", type=str, default="micro_beam_background",
                        help="Base name for output figures.")

    args = parser.parse_args()

    # Parse lists
    Nbg_list = [int(x) for x in args.Nbg_list.split(",") if x.strip()]
    alpha_list = [float(x) for x in args.alphas.split(",") if x.strip()]

    print("# Microphysical beam-in-background test")
    print(f"# N_beam = {args.N_beam}")
    print(f"# N_bg values = {Nbg_list}")
    print(f"# alphas (rad) = {alpha_list}")
    print(f"# mu = {args.mu}, tmax = {args.tmax}, steps = {args.steps}")

    for alpha in alpha_list:
        print("\n" + "=" * 70)
        print(f"# α = {alpha:.6f} rad  (~ {alpha/pi:.3f} π)")
        print("=" * 70)

        C_phys_list = []
        C_sigma_list = []

        last_t = None
        last_P_phys = None
        last_P_sigma = None
        last_Nbg = None

        for N_bg in Nbg_list:
            print(f"\n[alpha={alpha:.6f}] N_bg = {N_bg}")

            # Prepare system (operators + initial rotated background)
            psi0_rot, S_list, ops = prepare_system(args.N_beam, N_bg, alpha)
            Jx0, Jy0, Jz0, Jx1, Jy1, Jz1 = ops

            # Build "physical" and "sigma" Hamiltonians
            H_phys = build_cross_hamiltonian(args.mu, ops, model="physical")
            H_sigma = build_cross_hamiltonian(args.mu, ops, model="sigma")

            # Evolve and get P_ee(t) for the beam
            t_grid, P_phys = evolve_beam_probability(
                H_phys, psi0_rot, S_list, Jz0,
                t_max=args.tmax,
                steps=args.steps
            )
            _, P_sigma = evolve_beam_probability(
                H_sigma, psi0_rot, S_list, Jz0,
                t_max=args.tmax,
                steps=args.steps
            )

            # Estimate curvature near t=0
            C_phys, coefs_phys = estimate_curvature(t_grid, P_phys)
            C_sigma, coefs_sigma = estimate_curvature(t_grid, P_sigma)

            C_phys_list.append(C_phys)
            C_sigma_list.append(C_sigma)

            print(f"  physical:  C ≈ {C_phys:.3e},  P_min = {P_phys.min():.6f}")
            print(f"  sigma·sigma: C ≈ {C_sigma:.3e},  P_min = {P_sigma.min():.6f}")

            # Store the last one for plotting time series
            last_t = t_grid
            last_P_phys = P_phys
            last_P_sigma = P_sigma
            last_Nbg = N_bg

        # --- Plot for this α: time series (largest N_bg) + curvature vs N_bg ---

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8),
            sharex=False,
            height_ratios=[2, 1]
        )

        # Top: P_ee(t) for largest N_bg
        ax1.plot(last_t, last_P_phys, label=f"physical (N_bg={last_Nbg})", color="C0")
        ax1.plot(last_t, last_P_sigma, label=f"sigma·sigma (N_bg={last_Nbg})",
                 color="C1", linestyle="--")
        ax1.set_ylabel(r"$P_{ee}^{\rm beam}(t)$")
        ax1.set_title(
            rf"Beam survival $P_{{ee}}$ vs $t$  (α = {alpha:.3f}, N_bg={last_Nbg})"
        )
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best")

        # Bottom: curvature vs N_bg
        ax2.plot(Nbg_list, C_phys_list, "o-", label="physical", color="C0")
        ax2.plot(Nbg_list, C_sigma_list, "s--", label="sigma·sigma", color="C1")
        ax2.set_xlabel(r"$N_{\rm bg}$")
        ax2.set_ylabel(r"curvature $C$ in $P_{ee} \simeq 1 - C t^2$")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="best")

        ax2.text(
            0.02, 0.98,
            f"α = {alpha:.3f} rad\nμ = {args.mu:.3g}",
            transform=ax2.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        alpha_tag = f"{alpha/pi:.3f}pi".replace(".", "p")
        out_name = f"{args.savename}_alpha_{alpha_tag}.png"
        fig.savefig(out_name, dpi=150)
        print(f"# Saved figure for α={alpha:.6f} to {out_name}")

        plt.show()


if __name__ == "__main__":
    main()
