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
from scipy.sparse import csr_matrix, lil_matrix, kron as sparse_kron, csr_array
from scipy.sparse.linalg import eigs, eigsh, expm as sparse_expm, expm_multiply

from tqdm import tqdm

# ---------- Spin algebra (Dicke basis) ----------

def spin_matrices(S: float):
    """Return (Jx, Jy, Jz) for spin-S in |m> basis, m=-S,-S+1,...,S (ħ=1).
    
    Returns sparse matrices to conserve memory for large spin systems.
    """
    d = int(2 * S + 1) # dimension of the Dicke space
    m_vals = np.arange(-S, S + 1, 1, dtype=float) # m values given by -N / 2, ..., N / 2

    # Use LIL format for efficient construction of sparse matrices
    Jp = lil_matrix((d, d), dtype=complex)
    Jm = lil_matrix((d, d), dtype=complex)
    
    for i, m in enumerate(m_vals):
        jplus = S * (S + 1) - m * (m + 1) # jplus = S(S+1) - m(m+1)
        if i + 1 < d and jplus > 0:
            Jp[i + 1, i] = np.sqrt(jplus)
        jminus = S * (S + 1) - m * (m - 1) # jminus = S(S+1) - m(m-1)
        if i - 1 >= 0 and jminus > 0:
            Jm[i - 1, i] = np.sqrt(jminus)

    # Convert to CSR format for efficient arithmetic operations
    Jp = Jp.tocsr()
    Jm = Jm.tocsr()

    # Construct the x and y operators using sparse arithmetic
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    
    # Jz is diagonal, so we can construct it directly as a sparse matrix
    Jz = csr_matrix((m_vals, (np.arange(d), np.arange(d))), shape=(d, d), dtype=complex)
    
    return Jx, Jy, Jz

def dicke_basis_vector(S: float, m: float):
    """Return |S,m> in the Dicke basis (m integer/half-integer, -S <= m <= S).
    
    Returns a sparse vector to conserve memory for large spin systems.
    """
    # In reality, this is just a vector with a 1 in the m-th position and 0s elsewhere
    d = int(2 * S + 1) # dimension of the Dicke space
    idx = int(m + S)  # m=-S maps to 0, m=S maps to 2S
    
    # Create a sparse vector with only one non-zero element (column vector)
    v = csr_array(([1.0 + 0.0j], ([idx], [0])), shape=(d, 1), dtype=complex)
    return v

def kron_on_slot(op, slot, dims):
    """Place op on tensor slot `slot` with identities elsewhere (Kronecker).
    
    Works with sparse matrices to conserve memory for large systems.
    """
    out = None
    for a, d in enumerate(dims):
        if a == slot:
            A = op
        else:
            # Create sparse identity matrix
            A = csr_matrix(np.eye(d, dtype=complex))
        out = A if out is None else sparse_kron(out, A)
    return out

def product_dicke_state(S_list, m_list):
    """Return ⊗_a |S_a, m_a> as a vector in the tensor Dicke basis.
    
    Returns a sparse vector to conserve memory for large systems.
    """
    vec = csr_array([[1.0 + 0.0j]])  # Start with a 1x1 sparse array
    for S, m in zip(S_list, m_list):
        v = dicke_basis_vector(S, m)
        # Use sparse Kronecker product
        vec = sparse_kron(vec, v)
    return vec

# ---------- Hamiltonians ----------

def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float):
    """
    Single-energy homogeneous gas (all-to-all equal coupling).
    In the fully symmetric S=N/2 Dicke subspace.
    
    Returns sparse matrices to conserve memory for large systems.
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
    
    Returns sparse matrices to conserve memory for large systems.
    """
    assert len(N_list) == len(omega_list)
    S_list = [n / 2.0 for n in N_list]

    # Local spin matrices per bin (already sparse from spin_matrices)
    locals_ops = [spin_matrices(S) for S in S_list]
    dims = [ops[0].shape[0] for ops in locals_ops]

    # Lift to full space using sparse Kronecker products
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))

    dim = int(np.prod(dims))
    # Initialize Hamiltonian as sparse matrix
    H = csr_matrix((dim, dim), dtype=complex)

    # Vacuum field
    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)

    # Vacuum term - accumulate in sparse format
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

# ---------- Evolution & observables ----------

def evolve_times(H, psi0, t_grid):
    """Evolve psi0 under Hamiltonian H for times in t_grid.

    Parameters
    ----------
    H : (N,N) array or sparse matrix
        Hamiltonian.
    psi0 : (N,) array
        Initial state.
    t_grid : array_like
        Monotonically increasing list/array of times.

    Returns
    -------
    Y : ndarray, shape (len(t_grid), N)
        State at each time in t_grid.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import expm_multiply

    # Ensure H is csr_matrix (not csr_array)
    if not isinstance(H, csr_matrix):
        H = csr_matrix(H)

    # Ensure psi0 is a dense array (expm_multiply requires this)
    if hasattr(psi0, 'toarray'):
        psi0 = psi0.toarray().flatten()
    else:
        psi0 = np.asarray(psi0).flatten()

    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    num = len(t_grid)

    # Let expm_multiply handle sampling on [t0, t1]
    Y = expm_multiply(-1j * H, psi0, start=t0, stop=t1, num=num, endpoint=True)

    # Y is usually of shape (num, D)
    return np.asarray(Y)

def bin_observables(states, Jz_list, S_list):
    """
    For single bin: P_ee(t) = 1/2 * (1 + ⟨Jz⟩ / S).
    Jz_list should be a 1-tuple (Jz,).
    
    Works with sparse Jz matrices to conserve memory.
    """
    if isinstance(Jz_list, tuple) and len(Jz_list) == 3:
        # Single bin case we passed (Jx, Jy, Jz)
        Jz_list = [Jz_list[2]]
    T = states.shape[0]
    K = len(S_list)
    Jz_t = np.zeros((T, K), dtype=float)
    Pee_t = np.zeros((T, K), dtype=float)
    for ti in range(T):
        psi = states[ti]  # This is now a flat array
        for a in range(K):
            Jz = Jz_list[a]
            # Handle both sparse and dense Jz matrices
            if hasattr(Jz, 'dot'):
                # Sparse matrix - convert psi to column vector for multiplication
                psi_col = psi.reshape(-1, 1)
                jz = np.vdot(psi, Jz.dot(psi_col).flatten()).real
            else:
                # Dense matrix
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

def product_state(vecs):
    """Kronecker product of a list of state vectors.
    
    Returns a sparse vector to conserve memory for large systems.
    """
    out = csr_array([[1.0 + 0.0j]])  # Start with a 1x1 sparse array
    for v in vecs:
        out = sparse_kron(out, v)
    return out

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
