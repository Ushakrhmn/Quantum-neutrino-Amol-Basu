#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nunubar_beam_background_NxN.py

Neutrino–antineutrino "beam against background" test à la Friedland–Lunardini.

Goal
----
Starting from the micro_beam_background_test.py (ν beam in a ν background),
we now consider a beam of N_ν electron neutrinos interacting with a background
of N_{\barν} electron antineutrinos, all in a 2-flavor toy model.

We compare two choices for the 2-body ν–ν̄ interaction between the beam bin
and the background bin:

  1) "physical" ν–ν̄ interaction:
         H_int ∝ - Jx_ν Jx_{\barν} + Jy_ν Jy_{\barν} - Jz_ν Jz_{\barν}

     This is the spin-1/2 uplift of the 4×4 matrix with diagonal
     [-2,-1,-1,-2] and corner elements -1, as derived from the
     Dedin Neto–Kemp forward-scattering formalism for ν–ν̄.  Up to
     an overall factor and an additive identity, it reproduces the
     mean-field Hamiltonian used in density-matrix treatments and
     is consistent with the Friedland–Lunardini one-particle limit.

  2) "sigma":
         H_int ∝  Jx_ν Jx_{\barν} + Jy_ν Jy_{\barν} + Jz_ν Jz_{\barν}

     This is the naive σ·σ / Heisenberg interaction, often used in
     older toy models but with the wrong sign pattern for ν–ν̄.

We *omit* vacuum oscillations (ω = 0 for both bins) and any self-interaction
within a bin.  Dynamics is driven purely by the cross-bin 2-body term.

For each choice of N (with N_ν = N_{\barν} = N by default) and background
flavor angle α, we:

  - build the many-body Dicke state
      |ψ₀(α)⟩ = |beam = ν_e⟩^{⊗ N_ν} ⊗ |bg(α) = ν̄_x(α)⟩^{⊗ N_{\barν}}
  - evolve in time under H_phys and H_sigma
  - extract the beam ν_e survival probability P_ee^{(beam)}(t)
  - fit the early-time curvature C in
        P_ee(t) ≃ 1 - C t²
    to diagnose the presence (or absence) of an O(N²) coherent term.

Usage (example)
---------------
    python nunubar_beam_background_NxN.py \
        --N_list 1,2,4,8 \
        --alphas 1.570796 \   # e.g. α = π/2
        --mu 1.0 --tmax 0.3 --steps 81

This will scan N_ν = N_{\barν} ∈ {1,2,4,8} for one α value and compare the
"physical" and "sigma" interactions.

Requirements
------------
This script assumes that `dicke_collective_sparse_opt.py` is available in the
Python path and exposes:

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
# Helpers: build operators, initial ν–ν̄ coherent state, Hamiltonians
# ----------------------------------------------------------------------

def prepare_nunubar_system(N_nu, N_nubar, alpha):
    """
    Build a 2-bin Dicke system with
        bin 0: N_nu    neutrinos  (beam)
        bin 1: N_nubar antineutrinos (background)

    Initial state:
        - beam bin in pure ν_e (all spins up in flavor space)
        - background bin in a coherent flavor state
              |ν̄_x(α)⟩ = cos(α)|ν̄_e⟩ + sin(α)|ν̄_x⟩

      constructed by rotating the Dicke state |S_bg, m=S_bg⟩ around y:

              |bg(α)⟩ = exp(-i 2α J_y_bg) |S_bg, S_bg⟩.

    Parameters
    ----------
    N_nu : int
        Number of neutrinos in the beam bin.
    N_nubar : int
        Number of antineutrinos in the background bin.
    alpha : float
        Background flavor angle (radians).

    Returns
    -------
    psi0_rot : sparse or dense vector
        Initial state with rotated background.
    S_list : list[float]
        Spin magnitudes per bin [S_beam, S_bg].
    ops : tuple
        (Jx0, Jy0, Jz0, Jx1, Jy1, Jz1) as sparse matrices on full space.
    """
    # Start from product Dicke state: both bins "electron flavored"
    # n1_list counts ν_e / ν̄_e, n2_list counts ν_x / ν̄_x.
    psi0, S_list = dc.multi_bin_initial_state(
        n1_list=[N_nu, N_nubar],
        n2_list=[0,     0      ],
    )
    S_beam, S_bg = S_list
    dims = [int(2 * S_beam + 1), int(2 * S_bg + 1)]

    # Local spin operators for each bin
    Jx_beam, Jy_beam, Jz_beam = dc.spin_matrices(S_beam)
    Jx_bg,   Jy_bg,   Jz_bg   = dc.spin_matrices(S_bg)

    # Lift to full tensor space
    Jx0 = dc.kron_on_slot(Jx_beam, 0, dims)
    Jy0 = dc.kron_on_slot(Jy_beam, 0, dims)
    Jz0 = dc.kron_on_slot(Jz_beam, 0, dims)

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


def build_nunubar_cross_hamiltonian(mu, ops, model="physical"):
    """
    Build the ν–ν̄ cross-bin Hamiltonian:

        H_int = μ * PairTerm(J_beam, J_bg)

    with

      - "physical":
            PairTerm = -Jx0 Jx1 + Jy0 Jy1 - Jz0 Jz1

        which is the ν–ν̄ bilinear obtained by lifting the 4×4 forward
        scattering matrix of Dedin Neto & Kemp to flavor spins.

      - "sigma":
            PairTerm =  Jx0 Jx1 + Jy0 Jy1 + Jz0 Jz1

        the naive σ·σ interaction, kept here for comparison.

    Parameters
    ----------
    mu : float
        Interaction strength (sets overall time scale).
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


def evolve_probabilities(H, psi0, S_list, Jz0, Jz1, t_max, steps):
    """
    Evolve |psi0⟩ under H and compute P_ee(t) for both bins.

    Parameters
    ----------
    H : sparse matrix
        Hamiltonian.
    psi0 : sparse or dense vector
        Initial state.
    S_list : list[float]
        [S_beam, S_bg].
    Jz0, Jz1 : sparse matrices
        J_z operators for beam and background bins on full space.
    t_max : float
        Maximum evolution time.
    steps : int
        Number of time samples.

    Returns
    -------
    t_grid : ndarray
        Time samples.
    Pee_beam : ndarray
        Beam ν_e survival probability P_ee^{(beam)}(t).
    Pee_bg : ndarray
        Background ν̄_e survival probability P_ee^{(bg)}(t).
    """
    t_grid = np.linspace(0.0, t_max, steps)
    states = dc.evolve_times(H, psi0, t_grid)

    # bin_observables expects a list of Jz operators and the corresponding S's
    Jz_list = [Jz0, Jz1]
    Jz_t, Pee_t = dc.bin_observables(states, Jz_list, S_list)
    # Pee_t has shape (T, 2)
    Pee_beam = Pee_t[:, 0]
    Pee_bg   = Pee_t[:, 1]
    return t_grid, Pee_beam, Pee_bg


def estimate_curvature(t, P, n_fit=10):
    """
    Estimate the initial curvature C in P(t) ≃ 1 - C t² by fitting
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
            P_fit(t) ≃ a2 t² + a1 t + a0.
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
        description="Neutrino–antineutrino beam-in-background test (F&L style, N×N)."
    )
    parser.add_argument(
        "--N_list", type=str, default="1,2,3,4",
        help="Comma-separated list of N values with N_ν = N_{ν̄} = N, e.g. '1,2,4,8'."
    )
    parser.add_argument(
        "--alphas", type=str,
        default=f"{0.0:.6f},{0.0:.6f},{pi/4:.6f},{pi/2:.6f},{3*pi/4:.6f}",
        help="Comma-separated list of α values (radians). "
             "Default: 0, π/4, π/2, 3π/4."
    )
    parser.add_argument(
        "--mu", type=float, default=1.0,
        help="Interaction strength μ (sets overall time scale)."
    )
    parser.add_argument(
        "--tmax", type=float, default=0.3,
        help="Maximum evolution time (should keep μ tmax ≪ 1)."
    )
    parser.add_argument(
        "--steps", type=int, default=81,
        help="Number of time steps for evolution."
    )
    parser.add_argument(
        "--savename", type=str, default="nunubar_beam_background_NxN",
        help="Base name for output figures."
    )

    args = parser.parse_args()

    # Parse lists
    N_list = [int(x) for x in args.N_list.split(",") if x.strip()]
    alpha_list = [float(x) for x in args.alphas.split(",") if x.strip()]

    print("# ν–ν̄ microphysical beam-in-background test (N×N)")
    print(f"# N_ν = N_ν̄ values = {N_list}")
    print(f"# alphas (rad) = {alpha_list}")
    print(f"# mu = {args.mu}, tmax = {args.tmax}, steps = {args.steps}")

    for alpha in alpha_list:
        print("\n" + "=" * 70)
        print(f"# α = {alpha:.6f} rad  (~ {alpha/pi:.3f} π)")
        print("=" * 70)

        C_phys_list = []
        C_sigma_list = []

        last_t = None
        last_P_phys_beam = None
        last_P_sigma_beam = None
        last_N = None

        for N in N_list:
            N_nu = N
            N_nubar = N
            print(f"\n[alpha={alpha:.6f}] N_ν = N_ν̄ = {N}")

            # Prepare system (operators + initial rotated background)
            psi0_rot, S_list, ops = prepare_nunubar_system(N_nu, N_nubar, alpha)
            Jx0, Jy0, Jz0, Jx1, Jy1, Jz1 = ops

            # Build "physical" and "sigma" ν–ν̄ Hamiltonians
            H_phys = build_nunubar_cross_hamiltonian(args.mu, ops, model="physical")
            H_sigma = build_nunubar_cross_hamiltonian(args.mu, ops, model="sigma")

            # Evolve and get P_ee(t) for both bins
            t_grid, P_phys_beam, P_phys_bg = evolve_probabilities(
                H_phys, psi0_rot, S_list, Jz0, Jz1,
                t_max=args.tmax,
                steps=args.steps,
            )
            _, P_sigma_beam, P_sigma_bg = evolve_probabilities(
                H_sigma, psi0_rot, S_list, Jz0, Jz1,
                t_max=args.tmax,
                steps=args.steps,
            )

            # Estimate curvature near t=0 for the beam only
            C_phys, coefs_phys = estimate_curvature(t_grid, P_phys_beam)
            C_sigma, coefs_sigma = estimate_curvature(t_grid, P_sigma_beam)

            C_phys_list.append(C_phys)
            C_sigma_list.append(C_sigma)

            print(f"  physical (beam):  C ≈ {C_phys:.3e},  P_min = {P_phys_beam.min():.6f}")
            print(f"  sigma·sigma (beam): C ≈ {C_sigma:.3e},  P_min = {P_sigma_beam.min():.6f}")

            # Store the last one for plotting time series
            last_t = t_grid
            last_P_phys_beam = P_phys_beam
            last_P_sigma_beam = P_sigma_beam
            last_N = N

        # --- Plot for this α: time series (largest N) + curvature vs N ---

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8),
            sharex=False,
            height_ratios=[2, 1],
        )

        # Top: P_ee^{(beam)}(t) for largest N
        ax1.plot(last_t, last_P_phys_beam,
                 label=f"physical (N_ν=N_ν̄={last_N})", color="C0")
        ax1.plot(last_t, last_P_sigma_beam,
                 label=f"sigma·sigma (N_ν=N_ν̄={last_N})",
                 color="C1", linestyle="--")
        ax1.set_ylabel(r"$P_{ee}^{\rm beam}(t)$")
        ax1.set_title(
            rf"Beam survival $P_{{ee}}$ vs $t$  (α = {alpha:.3f}, N_ν=N_{{\bar\nu}}={last_N})"
        )
        ax1.grid(alpha=0.3)
        ax1.legend(loc="best")

        # Bottom: curvature vs N
        ax2.plot(N_list, C_phys_list, "o-", label="physical", color="C0")
        ax2.plot(N_list, C_sigma_list, "s--", label="sigma·sigma", color="C1")
        ax2.set_xlabel(r"$N_\nu = N_{\bar\nu}$")
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
