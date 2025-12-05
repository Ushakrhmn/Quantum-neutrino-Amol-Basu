#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dicke_rev_sign_test.py
========================================

Dicke-state solver for collective neutrino oscillations.

*** MODIFICATION NOTE ***
In this version, the cross-species (ν–ν̄) interaction is NO LONGER the
"physical" anisotropic Hamiltonian.
Instead, as requested, it is simply the **Heisenberg Hamiltonian with an overall negative sign**.

Interaction forms:
    - Same species (ν–ν or ν̄–ν̄): + J_a · J_b
    - Cross species (ν–ν̄):       - J_a · J_b  ( <--- This is the change )

Original functionality regarding Dicke basis and sparse matrices is preserved.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.sparse import (
    csr_matrix, lil_matrix, kron as sparse_kron, csr_array
)
from scipy.sparse.linalg import expm as sparse_expm, expm_multiply

# ---------- Spin algebra (Dicke basis) ----------

def spin_matrices(S: float):
    """Return (Jx, Jy, Jz) for spin-S in |m> basis, m=-S,-S+1,...,S (ħ=1).
    Each operator is returned as a CSR sparse matrix.
    """
    d = int(2 * S + 1)
    m_vals = np.arange(-S, S + 1, 1, dtype=float)

    Jp = lil_matrix((d, d), dtype=complex)
    Jm = lil_matrix((d, d), dtype=complex)
    for i, m in enumerate(m_vals):
        jp = S * (S + 1) - m * (m + 1)
        if i + 1 < d and jp > 0:
            Jp[i + 1, i] = np.sqrt(jp)
        jm = S * (S + 1) - m * (m - 1)
        if i - 1 >= 0 and jm > 0:
            Jm[i - 1, i] = np.sqrt(jm)

    Jp = Jp.tocsr()
    Jm = Jm.tocsr()

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = csr_matrix((m_vals, (np.arange(d), np.arange(d))), shape=(d, d), dtype=complex)
    return Jx, Jy, Jz

def dicke_basis_vector(S: float, m: float):
    """Return |S,m> as a sparse column vector in the Dicke basis."""
    d = int(2 * S + 1)
    idx = int(m + S)
    v = csr_array(([1.0 + 0.0j], ([idx], [0])), shape=(d, 1), dtype=complex)
    return v

def kron_on_slot(op, slot, dims):
    """Place `op` on tensor slot `slot` with identities elsewhere (Kronecker)."""
    out = None
    for a, d in enumerate(dims):
        A = op if a == slot else csr_matrix(np.eye(d, dtype=complex))
        out = A if out is None else sparse_kron(out, A)
    return out

def product_dicke_state(S_list, m_list):
    """Return ⊗_a |S_a, m_a> as a sparse column vector."""
    vec = csr_array([[1.0 + 0.0j]])
    for S, m in zip(S_list, m_list):
        vec = sparse_kron(vec, dicke_basis_vector(S, m))
    return vec

# ---------- Hamiltonians ----------

def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float):
    """
    Single-energy homogeneous gas (all-to-all equal coupling) in the
    fully symmetric S=N/2 Dicke subspace.
    """
    S = N / 2.0
    Jx, Jy, Jz = spin_matrices(S)

    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)
    H_vac = omega * (Bx * Jx + Bz * Jz)

    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
    H_int = mu * J2

    H = H_vac + H_int
    return H, (Jx, Jy, Jz), [S], [int(2*S+1)]

def _pair_term_same_species(Jx_a, Jy_a, Jz_a, Jx_b, Jy_b, Jz_b):
    """Heisenberg bilinear:  +JxJx + JyJy + JzJz."""
    return (
        (Jx_a @ Jx_b) +
        (Jy_a @ Jy_b) +
        (Jz_a @ Jz_b)
    )

def _pair_term_cross_species(Jx_a, Jy_a, Jz_a, Jx_b, Jy_b, Jz_b):
    """
    MODIFIED: Now returns NEGATIVE Heisenberg bilinear.
    Form: -(JxJx + JyJy + JzJz)
    """
    # (小声碎碎念：这里就是你要改的地方啦！)
    return -(
        (Jx_a @ Jx_b) +
        (Jy_a @ Jy_b) +
        (Jz_a @ Jz_b)
    )

def build_multi_bin_hamiltonian(
    N_list, omega_list, theta_v: float, mu: float, *,
    is_antineutrino=None,
    mu_cross: float | None = None,
):
    """
    Multi-energy, single-angle equal coupling.
      H = Σ_a ω_a (B·J_a)
        + Σ_{a<b} μ_{ab}  * PairTerm(J_a, J_b)

    PairTerm rules in this version:
      - same species (ν–ν or ν̄–ν̄) -> + J_a·J_b
      - cross species (ν–ν̄)       -> - J_a·J_b  (Negative Heisenberg)

    Parameters
    ----------
    N_list : list[int]
        Particle numbers per bin.
    omega_list : list[float]
        Vacuum frequencies for each bin.
    theta_v : float
        Vacuum mixing angle.
    mu : float
        Coupling for same-species pairs.
    is_antineutrino : None | list[bool]
        Boolean mask for antineutrinos.
        If None, all assumed same species -> all use +Heisenberg.
        If provided, mixed pairs use -Heisenberg.
    mu_cross : None | float
        Override coupling strength for cross pairs.

    Returns
    -------
    H, (JxL, JyL, JzL), S_list, dims
    """
    assert len(N_list) == len(omega_list), "N_list and omega_list must match."
    K = len(N_list)
    S_list = [n / 2.0 for n in N_list]

    # Local spin operators per bin
    locals_ops = [spin_matrices(S) for S in S_list]
    dims = [ops[0].shape[0] for ops in locals_ops]

    # Lifted operators
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))

    dim = int(np.prod(dims))
    H = csr_matrix((dim, dim), dtype=complex)

    # Vacuum term
    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)
    for a, omega in enumerate(omega_list):
        H += omega * (Bx * Jx_list[a] + Bz * Jz_list[a])

    # Interactions
    if is_antineutrino is None:
        is_antineutrino = [False] * K
    else:
        assert len(is_antineutrino) == K, "`is_antineutrino` must match bins."

    if mu_cross is None:
        mu_cross = mu

    for a in range(K):
        for b in range(a + 1, K):
            if bool(is_antineutrino[a]) == bool(is_antineutrino[b]):
                # same species: + Heisenberg
                H += mu * _pair_term_same_species(Jx_list[a], Jy_list[a], Jz_list[a],
                                                  Jx_list[b], Jy_list[b], Jz_list[b])
            else:
                # cross species: - Heisenberg (Modified per request)
                H += mu_cross * _pair_term_cross_species(Jx_list[a], Jy_list[a], Jz_list[a],
                                                         Jx_list[b], Jy_list[b], Jz_list[b])

    return H, (Jx_list, Jy_list, Jz_list), S_list, dims

# ---------- Evolution & observables ----------

def evolve_times(H, psi0, t_grid):
    """Evolve |psi0> under H for sample times t_grid -> array of states."""
    if not isinstance(H, csr_matrix):
        H = csr_matrix(H)

    if hasattr(psi0, 'toarray'):
        psi0 = psi0.toarray().flatten()
    else:
        psi0 = np.asarray(psi0).flatten()

    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    num = len(t_grid)
    Y = expm_multiply(-1j * H, psi0, start=t0, stop=t1, num=num, endpoint=True)
    return np.asarray(Y)

def bin_observables(states, Jz_list, S_list):
    """
    For single bin: P_ee(t) = 1/2 * (1 + ⟨Jz⟩ / S).
    """
    if isinstance(Jz_list, tuple) and len(Jz_list) == 3:
        Jz_list = [Jz_list[2]]
    T = states.shape[0]
    K = len(S_list)
    Jz_t = np.zeros((T, K), dtype=float)
    Pee_t = np.zeros((T, K), dtype=float)
    for ti in range(T):
        psi = states[ti]
        for a in range(K):
            Jz = Jz_list[a]
            if hasattr(Jz, 'dot'):
                jz = np.vdot(psi, (Jz.dot(psi.reshape(-1,1))).ravel()).real
            else:
                jz = np.vdot(psi, Jz @ psi).real
            Jz_t[ti, a] = jz
            Pee_t[ti, a] = 0.5 * (1.0 + jz / S_list[a])
    return Jz_t, Pee_t

# ---------- Helpers to build initial states ----------

def single_bin_initial_state(n1: int, n2: int):
    """One bin Dicke state with n1 ν_e (spin-up) and n2 ν_μ (spin-down)."""
    N = n1 + n2
    S = N / 2.0
    m = (n1 - n2) / 2.0
    v = dicke_basis_vector(S, m)
    return v, S

def multi_bin_initial_state(n1_list, n2_list):
    """⊗_a |S_a, m_a> where S_a=(n1_a+n2_a)/2 and m_a=(n1_a-n2_a)/2."""
    assert len(n1_list) == len(n2_list)
    S_list, m_list = [], []
    for n1, n2 in zip(n1_list, n2_list):
        N = n1 + n2
        S = N / 2.0
        m = (n1 - n2) / 2.0
        S_list.append(S)
        m_list.append(m)
    psi0 = product_dicke_state(S_list, m_list)
    return psi0, S_list

def product_state(vecs):
    """Kronecker product of a list of state vectors (sparse)."""
    out = csr_array([[1.0 + 0.0j]])
    for v in vecs:
        out = sparse_kron(out, v)
    return out

# ---------- Minimal demo ----------

if __name__ == "__main__":
    import numpy as _np
    import matplotlib.pyplot as _plt

    print("Running Demo with NEGATIVE Heisenberg cross-term...")

    Ne, Na = 4, 4           # numbers per bin
    S_list = [Ne/2, Na/2]

    theta = 0.01            # small vacuum mixing
    omega = 1.0             # set scale
    mu = 5.0                # interaction strength

    # Bin 1: pure ν_e  |m=S>
    # Bin 2: pure ν̄_e |m=S>
    psi0, _ = multi_bin_initial_state([Ne, Na], [0, 0])

    H, (JxL, JyL, JzL), S_list, dims = build_multi_bin_hamiltonian(
        N_list=[Ne, Na],
        omega_list=[+omega, -omega],
        theta_v=theta,
        mu=mu,
        is_antineutrino=[False, True],
    )

    t = _np.linspace(0.0, 20.0, 400)
    Y = evolve_times(H, psi0, t)
    _, Pee = bin_observables(Y, JzL, S_list)

    _plt.figure(figsize=(7,4))
    _plt.plot(t, Pee[:,0], label="bin 1 (ν)")
    _plt.plot(t, Pee[:,1], label="bin 2 (ν̄)")
    _plt.title("Dynamics with Cross Term = - J_a * J_b")
    _plt.xlabel("time")
    _plt.ylabel("Pee")
    _plt.legend()
    _plt.tight_layout()
    _plt.show()