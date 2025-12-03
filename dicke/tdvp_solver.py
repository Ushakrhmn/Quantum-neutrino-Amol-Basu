
"""
TDVP solver for collective neutrino oscillations on the product-coherent manifold.

Design goals
------------
1) Accept bin initializations in the same way as the current Dicke-state (DC) scripts:
   - Bin a is specified by numbers of electron and muon neutrinos (e_a, mu_a).
   - Energies per bin omega_a = Δm^2 / (2 E_a) (the script takes "energy" then converts).
   - Uniform all-to-all coupling parameterized by `mu` (typically mu = j / n_total).

2) Output:
   - t_grid (same "baseline" grid used by emu_2bin_opt_with_s.py)
   - P_ee per bin along the grid
   - (optional) single-particle entropy per bin (identically ~0 by construction)

Mathematics
-----------
The TDVP on the separable spin-coherent manifold yields the precession law
    dP_a/dl = h_a × P_a
with
    h_a = omega_a * B + mu * sum_{b != a} J_b * P_b
and B = (sin 2θ_v, 0, -cos 2θ_v).  N_b (count in bin b).
We use "baseline" l as the time-like parameter (same as the scripts).

Conventions
-----------
- We exclude self-coupling in the sum (consistent with H = ω B·S + μ ∑_{a<b} S_a·S_b).
- P_a is a *unit* Bloch vector if all spins in bin a are aligned; with e_a, mu_a counts,
  the initial polarization is along +z/-z:
      P_a(0) = (0, 0, (e_a - mu_a) / N_a).
  This has |P_a(0)| <= 1.  Under TDVP the length is preserved step-to-step by normalization.
- Electron flavor probability in bin a is P_ee[a] = (1 + P_a · L) / 2 with L = (0,0,1).

API
---
- tdvp_evolve_bins(e_counts, mu_counts, omega_list, theta_v, mu, l_grid,
                   *, substeps=1, return_entropy=False)
    -> (t_grid, P_ee)  or  (t_grid, P_ee, S_entropy)

- convenience_from_two_bins(args-like) replicates the emu_2bin_opt_with_s.py arguments.

"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional


@dataclass
class TDVPInputs:
    e_counts: np.ndarray   # shape (B,), integers
    mu_counts: np.ndarray  # shape (B,), integers
    omega_list: np.ndarray # shape (B,)
    theta_v: float         # mixing angle (radians)
    mu: float              # ν-ν coupling coefficient (per pair)
    l_grid: np.ndarray     # baselines
    substeps: int = 1      # RK4 internal subdivisions per grid step
    return_entropy: bool = False


def _B_vector(theta_v: float) -> np.ndarray:
    """Vacuum 'magnetic field' direction in flavor space: B = (sin 2θ, 0, -cos 2θ)."""
    return np.array([np.sin(2.0*theta_v), 0.0, -np.cos(2.0*theta_v)], dtype=float)


def _init_P(e_counts: Iterable[int], mu_counts: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize bin polarizations P_a and weights J_a = N_a/2 from counts."""
    e_counts = np.asarray(e_counts, dtype=float)
    mu_counts = np.asarray(mu_counts, dtype=float)
    N = e_counts + mu_counts
    if np.any(N <= 0):
        raise ValueError("Each bin must have at least one neutrino (N_a = e_a + mu_a >= 1).")
    Pz = (e_counts - mu_counts) / N  # within [-1, 1]
    B = len(N)
    P = np.zeros((B, 3), dtype=float)
    P[:, 2] = Pz
    Nw = N
    return P, Nw


def _rk4_step(P: np.ndarray, omega: np.ndarray, Bvec: np.ndarray, mu: float, Nw: np.ndarray, dl: float) -> np.ndarray:
    """One RK4 step for dP_a/dl = (omega_a B + mu * sum_{b != a} J_b P_b) × P_a.
    P: (B,3), omega: (B,), Bvec: (3,), Nw = J (weights) shape (B,), dl: scalar
    """
    def rhs(Pcur):
        # effective fields h_a for each bin
        sumJP = (Nw[:, None] * Pcur).sum(axis=0)               # shape (3,)
        # exclude self by subtracting own contribution J_a * P_a later
        h = omega[:, None] * Bvec[None, :] + mu * (sumJP[None, :] - Nw[:, None] * Pcur)
        return np.cross(h, Pcur)
    k1 = rhs(P)
    k2 = rhs(P + 0.5*dl*k1)
    k3 = rhs(P + 0.5*dl*k2)
    k4 = rhs(P + dl*k3)
    Pn = P + (dl/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # renormalize each bin polarization length to prevent drift
    norms = np.linalg.norm(Pn, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    Pn = Pn / norms
    return Pn


def tdvp_evolve_bins(e_counts: Iterable[int],
                     mu_counts: Iterable[int],
                     omega_list: Iterable[float],
                     theta_v: float,
                     mu: float,
                     l_grid: Iterable[float],
                     *,
                     substeps: int = 1,
                     return_entropy: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evolve bin polarizations via TDVP and return P_ee per bin on the provided l_grid.

    Parameters
    ----------
    e_counts, mu_counts : lists/arrays of length B
        Electron and muon counts per bin (integers).
    omega_list : list/array of length B
        ω_a for each bin.
    theta_v : float
        Vacuum mixing angle (radians).
    mu : float
        ν–ν coupling per pair (for all-to-all uniform coupling typical choice mu=j/n_total).
    l_grid : list/array
        Monotonically increasing baseline values.
    substeps : int
        RK4 internal steps per l_grid interval (>=1).
    return_entropy : bool
        If True, also return single-particle entropies per bin (should be ≈ 0).

    Returns
    -------
    t : ndarray shape (T,)
    P_ee : ndarray shape (T, B)  [electron survival prob per bin]
    S_entropy (optional) : ndarray shape (T, B)  [should be ~0]
    """
    e_counts = np.asarray(e_counts, dtype=float)
    mu_counts = np.asarray(mu_counts, dtype=float)
    omega = np.asarray(omega_list, dtype=float)
    l_grid = np.asarray(l_grid, dtype=float)
    if l_grid.ndim != 1 or len(l_grid) < 2:
        raise ValueError("l_grid must be a 1D array with at least two points.")

    Bvec = _B_vector(theta_v)
    P, Nw = _init_P(e_counts, mu_counts)
    B = P.shape[0]

    T = len(l_grid)
    P_ee = np.empty((T, B), dtype=float)
    S_ent = np.empty((T, B), dtype=float) if return_entropy else None

    # L vector (electron flavor axis)
    Lvec = np.array([0.0, 0.0, 1.0], dtype=float)

    P_ee[0, :] = 0.5 * (1.0 + (P @ Lvec))
    if return_entropy:
        # For a single-particle RDM with Bloch length r = |P|, eigenvalues λ±=(1±r)/2
        # Here |P| ≡ 1 (coherent product), so entropy ~ 0.
        r = np.linalg.norm(P, axis=1)
        lam_plus = 0.5 * (1.0 + r)
        lam_minus = 0.5 * (1.0 - r)
        lam_plus = np.clip(lam_plus, 1e-15, 1.0); lam_minus = np.clip(lam_minus, 1e-15, 1.0)
        S_ent[0, :] = -(lam_plus*np.log(lam_plus) + lam_minus*np.log(lam_minus))

    for t in range(T-1):
        dl = (l_grid[t+1] - l_grid[t]) / max(1, substeps)
        Pcur = P.copy()
        for _ in range(max(1, substeps)):
            Pcur = _rk4_step(Pcur, omega, Bvec, mu, Nw, dl)
        P = Pcur
        P_ee[t+1, :] = 0.5 * (1.0 + (P @ Lvec))
        if return_entropy:
            r = np.linalg.norm(P, axis=1)
            lam_plus = 0.5 * (1.0 + r)
            lam_minus = 0.5 * (1.0 - r)
            lam_plus = np.clip(lam_plus, 1e-15, 1.0); lam_minus = np.clip(lam_minus, 1e-15, 1.0)
            S_ent[t+1, :] = -(lam_plus*np.log(lam_plus) + lam_minus*np.log(lam_minus))

    return (l_grid, P_ee, S_ent) if return_entropy else (l_grid, P_ee)


# ------------------------------------------------------------------
# Compatibility helpers that mirror the DC script's two-bin arguments
# ------------------------------------------------------------------

def tdvp_from_emu2bin_args(e1: int, e2: int, m1: int, m2: int,
                           energy1: float, energy2: float,
                           j: float, l: float, steps: int,
                           theta_v: float = 0.6, # same default as emu_2bin_opt_with_s.py
                           normalize: bool = False,
                           substeps: int = 1,
                           return_entropy: bool = True):
    """
    Convenience wrapper mirroring emu_2bin_opt_with_s.py arguments.
    Converts energies to ω_a = Δm^2 / (2 E_a) up to a common factor absorbed in units,
    then evolves TDVP and returns (t, Pee[, S]).

    Parameters
    ----------
    e1,e2,m1,m2 : counts per bin
    energy1,energy2 : energies for the bins
    j : interaction strength used in the scripts
    l : total baseline
    steps : number of steps
    theta_v : mixing angle used in the scripts (radians)
    normalize : ignored (for interface parity)
    substeps : RK4 subdivisions per step
    return_entropy : return S if True
    """
    # In the repo's emu script, omega1 ~ 1/energy1 (up to an overall scale).
    # We will follow the same proportionality so shapes match MFT/DC overlays.
    omega1 = 1.0 / float(energy1)
    omega2 = 1.0 / float(energy2)

    n1 = int(e1 + m1); n2 = int(e2 + m2)
    n_total = n1 + n2
    mu = float(j) / float(n_total)  # matches build_multi_bin_hamiltonian usage

    l_grid = np.linspace(0.0, float(l), int(steps))

    return tdvp_evolve_bins(
        e_counts=[e1, e2],
        mu_counts=[m1, m2],
        omega_list=[omega1, omega2],
        theta_v=float(theta_v),
        mu=mu,
        l_grid=l_grid,
        substeps=int(substeps),
        return_entropy=bool(return_entropy)
    )



# ---- Optional helpers for compatibility with Dicke-basis tooling ----

def angles_from_P(P: np.ndarray) -> Tuple[float, float]:
    """Return (theta, phi) from a single bin Bloch vector P (length 3)."""
    x, y, z = P
    r = np.linalg.norm(P)
    if r == 0.0:
        return np.pi/2.0, 0.0
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return float(theta), float(phi)


def coherent_dicke_vector(J: int, theta: float, phi: float) -> np.ndarray:
    """Return |J,theta,phi> components in the Dicke basis |J,m>, m=-J..+J (column vector)."""
    from math import comb
    m_vals = np.arange(-J, J+1, 1, dtype=int)
    coeffs = []
    c = np.cos(theta/2.0); s = np.sin(theta/2.0)
    for m in m_vals:
        amp = (comb(2*J, J+m)**0.5) * (c**(J+m)) * (s**(J-m)) * np.exp(-1j*(J+m)*phi)
        coeffs.append(amp)
    return np.array(coeffs, dtype=complex).reshape((-1,1))

# --------------
# Simple __main__
# --------------
if __name__ == "__main__":
    import argparse, matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description="TDVP solver (product-coherent) for collective neutrino oscillations")
    # Mirror emu_2bin_opt_with_s.py for familiarity
    parser.add_argument('--e1', type=int, default=1)
    parser.add_argument('--e2', type=int, default=0)
    parser.add_argument('--m1', type=int, default=0)
    parser.add_argument('--m2', type=int, default=1)
    parser.add_argument('--energy1', type=float, default=1.0)
    parser.add_argument('--energy2', type=float, default=1.2)
    parser.add_argument('--j', type=float, default=5.0, help="interaction strength (same meaning as emu script)")
    parser.add_argument('--l', type=float, default=10.0, help="total baseline")
    parser.add_argument('--s', type=int, default=100, help="number of steps")
    parser.add_argument('--theta', type=float, default=0.6, help="vacuum mixing angle (radians)")
    parser.add_argument('--substeps', type=int, default=1, help="RK4 subdivisions per step")
    parser.add_argument('--savename', type=str, default='emu_tdvp', help="base filename for the saved figure(s)")
    args = parser.parse_args()

    t, Pee, Sent = tdvp_from_emu2bin_args(
        args.e1, args.e2, args.m1, args.m2,
        args.energy1, args.energy2,
        args.j, args.l, args.s,
        theta_v=args.theta,
        normalize=False,
        substeps=args.substeps,
        return_entropy=True
    )

    # Plot results
    plt.figure()
    plt.plot(t, Pee[:,0], label=f"Bin 1 (TDVP)")
    plt.plot(t, Pee[:,1], label=f"Bin 2 (TDVP)")
    plt.xlabel("baseline")
    plt.ylabel("Pe (TDVP)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig1 = plt.gcf()
    fig1.savefig(args.savename + ".png", dpi=150)

    # Entropy (should be ~0)
    plt.figure()
    plt.plot(t, Sent[:,0], label="Bin 1 entropy")
    plt.plot(t, Sent[:,1], label="Bin 2 entropy")
    plt.xlabel("baseline")
    plt.ylabel("Entropy (nats)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig2 = plt.gcf()
    fig2.savefig(args.savename + "_entropy.png", dpi=150)
    print(f"Saved {args.savename}.png and {args.savename}_entropy.png")
