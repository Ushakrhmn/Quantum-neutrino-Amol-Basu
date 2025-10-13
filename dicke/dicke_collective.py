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
from scipy.linalg import expm

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

def kron_on_slot(op, slot, dims):
    """Place op on tensor slot `slot` with identities elsewhere (Kronecker)."""
    out = None
    for a, d in enumerate(dims):
        A = op if a == slot else np.eye(d, dtype=complex)
        out = A if out is None else np.kron(out, A)
    return out

def product_dicke_state(S_list, m_list):
    """Return ⊗_a |S_a, m_a> as a vector in the tensor Dicke basis."""
    vec = np.array([1.0 + 0.0j])
    for S, m in zip(S_list, m_list):
        v = dicke_basis_vector(S, m)
        vec = np.kron(vec, v)
    return vec

# ---------- Hamiltonians ----------

def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float):
    """
    Single-energy homogeneous gas (all-to-all equal coupling).
    In the fully symmetric S=N/2 Dicke subspace,
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

def build_multi_bin_hamiltonian(N_list, omega_list, theta_v: float, mu: float):
    """
    Multi-energy, single-angle equal coupling μ for all inter-bin pairs:
      H = Σ_a ω_a (B·J_a) + μ Σ_{a<b} J_a · J_b
    """
    assert len(N_list) == len(omega_list)
    S_list = [n / 2.0 for n in N_list]

    # Local spin matrices per bin
    locals_ops = [spin_matrices(S) for S in S_list]
    dims = [ops[0].shape[0] for ops in locals_ops]

    # Lift to full space
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))

    dim = int(np.prod(dims))
    H = np.zeros((dim, dim), dtype=complex)

    # Vacuum field
    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)

    # Vacuum term
    for a, omega in enumerate(omega_list):
        H += omega * (Bx * Jx_list[a] + Bz * Jz_list[a])

    # ν–ν interaction: cross-bin only; intra-bin part is a constant in each S_a sector
    for a in range(len(N_list)):
        for b in range(a + 1, len(N_list)):
            H += mu * (
                Jx_list[a] @ Jx_list[b] +
                Jy_list[a] @ Jy_list[b] +
                Jz_list[a] @ Jz_list[b]
            )

    return H, (Jx_list, Jy_list, Jz_list), S_list, dims

def build_multi_bin_hamiltonian_rev_sign(N_list, omega_list, theta_v: float, mu: float):
    """
    Multi-energy, single-angle equal coupling μ for all inter-bin pairs:
      H = Σ_a ω_a (B·J_a) + μ Σ_{a<b} J_a · J_b
    """
    assert len(N_list) == len(omega_list)
    S_list = [n / 2.0 for n in N_list]

    # Local spin matrices per bin
    locals_ops = [spin_matrices(S) for S in S_list]
    dims = [ops[0].shape[0] for ops in locals_ops]

    # Lift to full space
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))

    dim = int(np.prod(dims))
    H = np.zeros((dim, dim), dtype=complex)

    # Vacuum field
    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)

    # Vacuum term
    for a, omega in enumerate(omega_list):
        H += omega * (Bx * Jx_list[a] + Bz * Jz_list[a])
    
    H += 2 * mu * (Jx_list[0] @ Jx_list[0] + Jy_list[0] @ Jy_list[0] + Jz_list[0] @ Jz_list[0])
    H += 2 * mu * (Jx_list[1] @ Jx_list[1] + Jy_list[1] @ Jy_list[1] + Jz_list[1] @ Jz_list[1])

    # ν–ν interaction: cross-bin only; intra-bin part is a constant in each S_a sector
    for a in range(len(N_list)):
        for b in range(a + 1, len(N_list)):
            H -= 2 * mu * (
                Jx_list[a] @ Jx_list[b] +
                Jy_list[a] @ Jy_list[b] +
                Jz_list[a] @ Jz_list[b]
            )

    return H, (Jx_list, Jy_list, Jz_list), S_list, dims

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

def multi_bin_initial_state(n1_list, n2_list):
    """
    Build ⊗_a |S_a, m_a> where S_a=(n1_a+n2_a)/2 and m_a=(n1_a-n2_a)/2.
    """
    assert len(n1_list) == len(n2_list)
    S_list = []
    m_list = []
    for n1, n2 in zip(n1_list, n2_list):
        N = n1 + n2
        S = N / 2.0
        m = (n1 - n2) / 2.0
        S_list.append(S)
        m_list.append(m)
    psi0 = product_dicke_state(S_list, m_list)
    return psi0, S_list

def spin_coherent_state_by_rotation(S: float, theta: float, phi: float = 0.0):
    """
    Construct |theta,phi> = e^{-i phi Jz} e^{-i theta Jy} |S,S> in the Dicke |m> basis.
    Basis ordering: m = -S, -S+1, ..., S (same as your spin_matrices).
    """
    Jx, Jy, Jz = spin_matrices(S)  # your existing function
    d = int(2*S + 1)

    # |S,S> corresponds to the last basis component (m=S -> index 2S)
    v_top = np.zeros((d,), dtype=complex)
    v_top[-1] = 1.0

    # Apply rotations: first around y, then around z
    Ry = expm(-1j * theta * Jy)
    Rz = expm(-1j * phi   * Jz)
    return Rz @ (Ry @ v_top)

def product_state(vecs):
    """Kronecker product of a list of state vectors."""
    out = np.array([1.0 + 0.0j])
    for v in vecs:
        out = np.kron(out, v)
    return out

def multi_bin_initial_state_coherent(n1_list, n2_list, phi_list=None):
    """
    Build ⊗_a |θ_a, φ_a>, where S_a=(n1_a+n2_a)/2 and cos θ_a = (n1_a - n2_a)/(n1_a + n2_a).
    If phi_list is None, all φ_a default to 0.

    Returns:
        psi0   : full tensor-product coherent state
        S_list : list of S per bin
        theta_list, phi_list : polar/azimuthal angles used per bin
    """
    assert len(n1_list) == len(n2_list)
    K = len(n1_list)
    if phi_list is None:
        phi_list = [0.0] * K

    S_list, theta_list, vecs = [], [], []
    for n1, n2, phi in zip(n1_list, n2_list, phi_list):
        N = n1 + n2
        if N <= 0:
            raise ValueError("Each bin must have N>0 for a coherent state.")
        S = N / 2.0

        # Match <Jz> to m = (n1-n2)/2  ->  cos(theta) = (n1-n2)/N
        cos_theta = (n1 - n2) / float(N)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numeric safety
        theta = float(np.arccos(cos_theta))

        v = spin_coherent_state_by_rotation(S, theta, phi)
        S_list.append(S)
        theta_list.append(theta)
        vecs.append(v)

    psi0 = product_state(vecs)
    return psi0, S_list, theta_list, phi_list

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

# ========= a-state helpers =========

def bloch_from_flavor(theta, phi=0.0):
    """
    Flavor state: |ψ> = cos(theta)|e> + e^{i phi} sin(theta)|μ>
    -> Bloch vector on flavor sphere: (x,y,z).
    """
    s2 = np.sin(2.0*theta)
    return np.array([s2*np.cos(phi), s2*np.sin(phi), np.cos(2.0*theta)], dtype=float)

def build_initial_state_ea(Ne, Na, alpha, phi=0.0, omega=1.0):
    """
    Build initial Bloch vectors for Ne electrons and Na copies of
    |a> = cos(alpha)|e> + e^{i phi} sin(alpha)|μ>.
    Returns:
        P0_list : list of shape (Ne+Na, 3)
        omegas  : np.ndarray of shape (Ne+Na,)
    """
    # basis Bloch vectors
    P_e  = np.array([0.0, 0.0,  1.0], dtype=float)     # |e>
    P_a  = bloch_from_flavor(alpha, phi)               # |a>

    P0_list = [P_e.copy() for _ in range(Ne)] + [P_a.copy() for _ in range(Na)]
    omegas  = np.full(Ne+Na, float(omega))
    return P0_list, omegas, P_e, P_a

if __name__ == "__main__":
    # Example: single-energy homogeneous gas (one Dicke spin)
    # Expect vacuum-like precession; μ adds only a phase in symmetric subspace.
    demo_single_bin(n1=1, n2=1, omega=1.0, theta_v=np.pi/2 - 0.2, mu=5.0)
