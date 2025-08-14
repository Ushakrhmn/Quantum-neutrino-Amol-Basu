#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dicke-state solver for collective neutrino oscillations (vacuum + ν–ν)
=====================================================================

Single-energy homogeneous gas ("single-J"): all pairs have identical coupling.
In the fully symmetric Dicke subspace.

Conventions
-----------
- Flavor isospin with ħ=1.
- Vacuum "magnetic field" B = (sin 2θ_v, 0, -cos 2θ_v).
- Hamiltonian (single-angle equal coupling):
    H = ω (B·J) + μ J^2
  If starting from Σ_{i<j} σ_i·σ_j, absorb factors of 4 into μ (since σ = 2J).

API
---
- spin_matrices(S): returns (Jx, Jy, Jz) for spin S in the Dicke |m> basis.
- build_single_bin_hamiltonian(N, omega, theta_v, mu): single-energy case.
- dicke_basis_vector(S, m): |S, m> basis vector (m=-S...S) in the Dicke basis.
- evolve_times(H, psi0, t_grid): exact evolution using spectral decomposition.
- bin_observables(states, Jz_list, S_list): returns ⟨Jz⟩ and P_ee(t).

Example usage is provided under __main__.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Spin algebra (Dicke basis) ----------

def spin_matrices(S: float):
    """Return (Jx, Jy, Jz) for spin-S in |m> basis, m=-S,-S+1,...,S (ħ=1)."""
    d = int(2 * S + 1) # dimension of the Dicke space
    m_vals = np.arange(-S, S + 1, 1, dtype=float) # m values given by -N / 2, ..., N / 2

    # Construct the raising and lowering operators Jp and Jm
    Jp = np.zeros((d, d), dtype=complex)
    Jm = np.zeros((d, d), dtype=complex)
    for i, m in enumerate(m_vals):
        jplus = S * (S + 1) - m * (m + 1) # jplus = S(S+1) - m(m+1)
        if i + 1 < d and jplus > 0:
            Jp[i + 1, i] = np.sqrt(jplus)
        jminus = S * (S + 1) - m * (m - 1) # jminus = S(S+1) - m(m-1)
        if i - 1 >= 0 and jminus > 0:
            Jm[i - 1, i] = np.sqrt(jminus)

    # Construct the x and y operators
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = np.diag(m_vals)
    return Jx, Jy, Jz

def dicke_basis_vector(S: float, m: float):
    """Return |S,m> in the Dicke basis (m integer/half-integer, -S <= m <= S)."""
    # In reality, this is just a vector with a 1 in the m-th position and 0s elsewhere
    d = int(2 * S + 1) # dimension of the Dicke space
    idx = int(m + S)  # m=-S maps to 0, m=S maps to 2S
    v = np.zeros((d,), dtype=complex)
    v[idx] = 1.0 # set the m-th element to 1
    return v

# ---------- Hamiltonians ----------

def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float):
    """
    Single-energy homogeneous gas (all-to-all equal coupling).
    In the fully symmetric S=N/2 Dicke subspace,
      H_int ∝ μ J^2 = μ S(S+1) * I  (a constant phase).
    So flavor probabilities are governed by the vacuum term only.
    We include the constant anyway (it does not affect P_ee).
    """
    S = N / 2.0 # N neutrino system, total spin S = N / 2
    Jx, Jy, Jz = spin_matrices(S)

    # define the vacuum oscillation values
    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)
    H_vac = omega * (Bx * Jx + Bz * Jz)

    # define the interaction term
    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
    H_int = mu * J2

    H = H_vac + H_int
    return H, (Jx, Jy, Jz), [S], [int(2*S+1)]

# ---------- Evolution & observables ----------

def evolve_times(H, psi0, t_grid):
    """Exact unitary evolution |ψ(t)> via spectral decomposition (Hermitian H)."""
    evals, evecs = np.linalg.eigh(H)
    coeff0 = evecs.conj().T @ psi0
    states = []
    for t in t_grid:
        psi_t = evecs @ (np.exp(-1j * evals * t) * coeff0)
        states.append(psi_t)
    return np.stack(states, axis=0)

def bin_observables(states, Jz_list, S_list):
    """
    For single bin: P_ee(t) = 1/2 * (1 + ⟨Jz⟩ / S).
    Jz_list should be a 1-tuple (Jz,).
    """
    if isinstance(Jz_list, tuple) and len(Jz_list) == 3:
        # Single bin case we passed (Jx, Jy, Jz)
        Jz_list = [Jz_list[2]]
    T = states.shape[0]
    K = len(S_list)
    Jz_t = np.zeros((T, K), dtype=float)
    Pee_t = np.zeros((T, K), dtype=float)
    for ti in range(T):
        psi = states[ti]
        for a in range(K):
            Jz = Jz_list[a]
            jz = np.vdot(psi, Jz @ psi).real
            Jz_t[ti, a] = jz
            Pee_t[ti, a] = 0.5 * (1.0 + jz / S_list[a])
    return Jz_t, Pee_t

# ---------- Helpers to build initial states from (n1, n2) ----------

def single_bin_initial_state(n1: int, n2: int):
    """
    Build the symmetric Dicke state for one bin with n1 ν_e (spin-up) and n2 ν_μ (spin-down).
    That's |S=N/2, m=(n1-n2)/2> in the symmetric subspace.
    """
    N = n1 + n2
    S = N / 2.0
    m = (n1 - n2) / 2.0
    v = dicke_basis_vector(S, m)
    return v, S

# ---------- Demos ----------

def demo_single_bin(n1=12, n2=4, omega=1.0, theta_v=0.15, mu=0.5,
                    t_max=40.0, T=400, make_plot=True):
    psi0, S = single_bin_initial_state(n1, n2)
    H, (Jx, Jy, Jz), S_list, dims = build_single_bin_hamiltonian(N=int(2*S),
                                                                 omega=omega,
                                                                 theta_v=theta_v,
                                                                 mu=mu)
    t_grid = np.linspace(0.0, t_max, T)
    states = evolve_times(H, psi0, t_grid)
    _, Pee_t = bin_observables(states, (Jx, Jy, Jz), S_list)
    if make_plot:
        plt.figure(figsize=(7,4))
        plt.plot(t_grid, Pee_t[:,0], label=f'single bin: N={int(2*S)}, n1={n1}, n2={n2}')
        plt.xlabel('time')
        plt.ylabel('Pee')
        plt.title('Single-energy (single-J): vacuum-dominated precession')
        plt.legend()
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.show()
    return t_grid, Pee_t

if __name__ == "__main__":
    # Example: single-energy homogeneous gas (one Dicke spin)
    # Expect vacuum-like precession; μ adds only a phase in symmetric subspace.
    demo_single_bin(n1=4, n2=10, omega=1.0, theta_v=np.pi/4, mu=5.0)
