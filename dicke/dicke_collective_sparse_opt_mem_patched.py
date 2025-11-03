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
from scipy.sparse import csr_matrix, lil_matrix, eye as sparse_eye, kron as sparse_kron, csr_array
from scipy.sparse.linalg import eigs, eigsh, expm as sparse_expm, expm_multiply

from tqdm import tqdm
import gc
import os
from scipy.sparse import eye as sparse_eye, kron as sparse_kron, csr_matrix, lil_matrix, csr_array
from scipy.sparse.linalg import LinearOperator, expm_multiply

# Default numeric type (memory-friendly)
DEFAULT_DTYPE = np.complex128


# ---------- Spin algebra (Dicke basis) ----------

def spin_matrices(S: float, dtype: np.dtype = DEFAULT_DTYPE):
    """Return (Jx, Jy, Jz) for spin-S in |m> basis, m=-S,-S+1,...,S (ħ=1).
    
    Returns sparse matrices to conserve memory for large spin systems.
    """
    d = int(2 * S + 1) # dimension of the Dicke space
    m_vals = np.arange(-S, S + 1, 1, dtype=float) # m values given by -N / 2, ..., N / 2

    # Use LIL format for efficient construction of sparse matrices
    Jp = lil_matrix((d, d), dtype=dtype)
    Jm = lil_matrix((d, d), dtype=dtype)
    
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
    Jx = (0.5 * (Jp + Jm)).astype(dtype)
    Jy = (-0.5j * (Jp - Jm)).astype(dtype)
    
    # Jz is diagonal, so we can construct it directly as a sparse matrix
    Jz = csr_matrix((m_vals.astype(float), (np.arange(d), np.arange(d))), shape=(d, d), dtype=dtype)
    
    return Jx, Jy, Jz

def dicke_basis_vector(S: float, m: float, dtype: np.dtype = DEFAULT_DTYPE):
    """Return |S,m> in the Dicke basis (m integer/half-integer, -S <= m <= S).
    
    Returns a sparse vector to conserve memory for large spin systems.
    """
    # In reality, this is just a vector with a 1 in the m-th position and 0s elsewhere
    d = int(2 * S + 1) # dimension of the Dicke space
    idx = int(m + S)  # m=-S maps to 0, m=S maps to 2S
    
    # Create a sparse vector with only one non-zero element (column vector)
    v = csr_array(([dtype(1.0 + 0.0j)], ([idx], [0])), shape=(d, 1), dtype=dtype)
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
            A = sparse_eye(d, dtype=(op.dtype if hasattr(op, 'dtype') else DEFAULT_DTYPE), format='csr')
        out = A if out is None else sparse_kron(out, A)
    return out

def product_dicke_state(S_list, m_list, dtype: np.dtype = DEFAULT_DTYPE):
    """Return ⊗_a |S_a, m_a> as a vector in the tensor Dicke basis.
    
    Returns a sparse vector to conserve memory for large systems.
    """
    vec = csr_array([[dtype(1.0 + 0.0j)]], dtype=dtype)  # Start with a 1x1 sparse array
    for S, m in zip(S_list, m_list):
        v = dicke_basis_vector(S, m, dtype=dtype)
        # Use sparse Kronecker product
        vec = sparse_kron(vec, v)
    return vec

# === Memory-lean linear-operator helpers (default path) ===

# Toggle with env var: set DICKE_USE_LINEAR_OPERATOR=0 to force explicit sparse matrices (legacy behavior)
USE_LINEAR_OPERATOR_DEFAULT = os.environ.get("DICKE_USE_LINEAR_OPERATOR", "1") != "0"

def _apply_on_slot(vec, op, slot, dims, *, dtype=None):
    """
    Apply a single-bin operator `op` (shape d_slot x d_slot) to the `slot`-th axis
    of a flattened tensor-product state `vec` with per-bin dimensions `dims`.
    Returns a 1D array with the same shape as `vec`.

    This avoids materializing the full Kronecker-lifted operator.
    """
    import numpy as _np
    v = _np.asarray(vec, dtype=dtype if dtype is not None else getattr(vec, "dtype", None)).reshape(dims, order="C")
    # bring target axis in front: (d_slot, ...rest...)
    v = _np.moveaxis(v, slot, 0)
    d_slot = dims[slot]
    rest = int(_np.prod(dims) // d_slot)
    v2 = v.reshape(d_slot, rest)
    # sparse or dense multiply along the leading axis
    if hasattr(op, "dot"):
        w2 = op.dot(v2)
    else:
        w2 = op @ v2
    # reshape back
    w = w2.reshape((d_slot,)+tuple(d for i, d in enumerate(dims) if i != slot))
    w = _np.moveaxis(w, 0, slot)
    return w.reshape(-1)

def _lift_linear_op(op_local, slot, dims, *, dtype):
    """
    Return a LinearOperator acting on the *full* tensor space corresponding to
    placing `op_local` on `slot` (Kronecker with identities elsewhere).
    """
    import numpy as _np
    D = int(_np.prod(dims))
    def _mv(x):
        return _apply_on_slot(x, op_local, slot, dims, dtype=dtype)
    # Hermitian single-site operators: rmatvec == matvec
    return LinearOperator((D, D), matvec=_mv, rmatvec=_mv, dtype=dtype)

# ---------- Hamiltonians ----------

def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float, *, dtype: np.dtype = DEFAULT_DTYPE):
    """
    Single-energy homogeneous gas (all-to-all equal coupling) in the Dicke subspace S=N/2.

    **Default behavior (changed)**: returns a *LinearOperator* for the Hamiltonian to reduce
    memory. Set environment variable ``DICKE_USE_LINEAR_OPERATOR=0`` to build an explicit
    sparse matrix (legacy behavior). API (function signature & return tuple) remains unchanged.

    Returns
    -------
    H : LinearOperator or scipy.sparse.spmatrix
        Many-body Hamiltonian on the Dicke subspace.
    (Jx_list, Jy_list, Jz_list) : tuple of sequences (length 1)
        Slot-lifted operators for observables (LinearOperator by default).
    S_list : list[float]
        [S]
    dims : list[int]
        [2S+1]
    """
    S = N / 2.0
    Jx, Jy, Jz = spin_matrices(S, dtype=dtype)
    d = int(2*S + 1)

    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)

    if USE_LINEAR_OPERATOR_DEFAULT:
        # Linear-operator Hamiltonian: H = ω(B·J) + μ(J·J) with J·J a constant phase in fixed-S,
        # but we implement it as Jx^2 + Jy^2 + Jz^2 for exact legacy parity.
        def _mv(v):
            v = np.asarray(v, dtype=dtype, order="C").reshape(d)
            out = omega * (Bx * (Jx.dot(v)) + Bz * (Jz.dot(v)))
            if mu != 0:
                out = out + mu * (Jx.dot(Jx.dot(v)) + Jy.dot(Jy.dot(v)) + Jz.dot(Jz.dot(v)))
            return out
        H = LinearOperator((d, d), matvec=_mv, rmatvec=_mv, dtype=dtype)
        # Slot-lifted Js (single slot)
        dims = [d]
        JxL = _lift_linear_op(Jx, 0, dims, dtype=dtype)
        JyL = _lift_linear_op(Jy, 0, dims, dtype=dtype)
        JzL = _lift_linear_op(Jz, 0, dims, dtype=dtype)
        return H, ([JxL], [JyL], [JzL]), [S], dims

    # ---- Legacy explicit sparse matrix path ----
    H_vac = omega * (Bx * Jx + Bz * Jz)
    H_int = mu * (Jx @ Jx + Jy @ Jy + Jz @ Jz)
    H = (H_vac + H_int).tocsr()
    return H, ([kron_on_slot(Jx, 0, [d])], [kron_on_slot(Jy, 0, [d])], [kron_on_slot(Jz, 0, [d])]), [S], [d]

def build_multi_bin_hamiltonian(N_list, omega_list, theta_v: float, mu: float, *, dtype: np.dtype = DEFAULT_DTYPE):
    """
    Multi-energy, single-angle equal coupling μ for all inter-bin pairs:
      H = Σ_a ω_a (B·J_a) + μ Σ_{a<b} J_a · J_b
    
    **Default behavior**: returns a *LinearOperator* for the Hamiltonian to reduce
    memory. Set environment variable ``DICKE_USE_LINEAR_OPERATOR=0`` to build an explicit
    sparse matrix (legacy behavior).
    
    Returns
    -------
    H : LinearOperator or scipy.sparse.spmatrix
        Many-body Hamiltonian on the multi-bin Dicke subspace.
    (Jx_list, Jy_list, Jz_list) : tuple of sequences
        Slot-lifted operators for observables (LinearOperator by default).
    S_list : list[float]
        Spin magnitudes for each bin
    dims : list[int]
        Dimensions [2S+1] for each bin
    """
    assert len(N_list) == len(omega_list)
    S_list = [n / 2.0 for n in N_list]

    # Local spin matrices per bin (already sparse from spin_matrices)
    locals_ops = [spin_matrices(S, dtype=dtype) for S in S_list]
    dims = [int(2*S + 1) for S in S_list]

    Bx = np.sin(2 * theta_v)
    Bz = -np.cos(2 * theta_v)

    if USE_LINEAR_OPERATOR_DEFAULT:
        # LinearOperator mode: build operators slot-by-slot
        Jx_list, Jy_list, Jz_list = [], [], []
        for a, (Jx_loc, Jy_loc, Jz_loc) in enumerate(locals_ops):
            Jx_list.append(_lift_linear_op(Jx_loc, a, dims, dtype=dtype))
            Jy_list.append(_lift_linear_op(Jy_loc, a, dims, dtype=dtype))
            Jz_list.append(_lift_linear_op(Jz_loc, a, dims, dtype=dtype))
        
        # Build Hamiltonian as a LinearOperator
        D = int(np.prod(dims))
        def _mv(v):
            v = np.asarray(v, dtype=dtype, order="C")
            out = np.zeros_like(v)
            # Vacuum term: Σ_a ω_a (Bx J_x^a + Bz J_z^a)
            for a, omega in enumerate(omega_list):
                out = out + omega * Bx * Jx_list[a].matvec(v)
                out = out + omega * Bz * Jz_list[a].matvec(v)
            # Interaction term: μ Σ_{a<b} J_a · J_b
            if mu != 0:
                for a in range(len(N_list)):
                    for b in range(a + 1, len(N_list)):
                        out = out + mu * Jx_list[a].matvec(Jx_list[b].matvec(v))
                        out = out + mu * Jy_list[a].matvec(Jy_list[b].matvec(v))
                        out = out + mu * Jz_list[a].matvec(Jz_list[b].matvec(v))
            return out
        
        H = LinearOperator((D, D), matvec=_mv, rmatvec=_mv, dtype=dtype)
        return H, (Jx_list, Jy_list, Jz_list), S_list, dims
    
    # ---- Legacy explicit sparse matrix path ----
    # Lift to full space using sparse Kronecker products
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))

    dim = int(np.prod(dims))
    # Initialize Hamiltonian as sparse matrix
    H = csr_matrix((dim, dim), dtype=dtype)

    # Vacuum term - accumulate in sparse format
    for a, omega in enumerate(omega_list):
        H = H + omega * (Bx * Jx_list[a] + Bz * Jz_list[a])

    # ν–ν interaction: cross-bin only; intra-bin part is a constant in each S_a sector
    for a in range(len(N_list)):
        for b in range(a + 1, len(N_list)):
            H = H + mu * (
                Jx_list[a] @ Jx_list[b] +
                Jy_list[a] @ Jy_list[b] +
                Jz_list[a] @ Jz_list[b]
            )

    return H, (Jx_list, Jy_list, Jz_list), S_list, dims

def evolve_times(H, psi0, t_grid, *, dtype: np.dtype = DEFAULT_DTYPE, gc_collect: bool = True):
    """Evolve psi0 under Hamiltonian H for times in t_grid.

    Parameters
    ----------
    H : (N,N) array, sparse matrix, or LinearOperator
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
    from scipy.sparse.linalg import expm_multiply, LinearOperator

    # Handle H based on type - expm_multiply works with LinearOperators directly
    if isinstance(H, LinearOperator):
        # LinearOperator: use as-is (expm_multiply supports this)
        pass
    elif not isinstance(H, csr_matrix):
        # Convert other types to csr_matrix
        H = csr_matrix(H)
        H = H.astype(dtype)
    else:
        # Already csr_matrix, just ensure dtype
        H = H.astype(dtype)

    # Ensure psi0 is a dense 1D array (expm_multiply requires this)
    if hasattr(psi0, 'toarray'):
        psi0 = np.asarray(psi0.toarray(), dtype=dtype).flatten()
    else:
        psi0 = np.asarray(psi0, dtype=dtype).flatten()

    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    num = len(t_grid)

    # Let expm_multiply handle sampling on [t0, t1]
    minus_i = np.array(-1j, dtype=dtype).item()
    Y = expm_multiply(minus_i * H, psi0, start=t0, stop=t1, num=num, endpoint=True)

    # Y is usually of shape (num, D)
    Y = np.asarray(Y, dtype=dtype)
    if gc_collect:
        del H
        gc.collect()
    return Y

def bin_observables(states, Jz_list, S_list):
    """
    For single bin: P_ee(t) = 1/2 * (1 + ⟨Jz⟩ / S).
    Jz_list should be a 1-tuple (Jz,).
    
    Works with sparse Jz matrices and LinearOperators to conserve memory.
    """
    from scipy.sparse.linalg import LinearOperator
    
    if isinstance(Jz_list, tuple) and len(Jz_list) == 3:
        # Single bin case we passed (Jx, Jy, Jz) - Jz is already a list
        Jz_list = Jz_list[2]
    T = states.shape[0]
    K = len(S_list)
    Jz_t = np.zeros((T, K), dtype=float)
    Pee_t = np.zeros((T, K), dtype=float)
    for ti in range(T):
        psi = states[ti]  # This is now a flat array
        for a in range(K):
            Jz = Jz_list[a]
            # Handle LinearOperator, sparse matrix, and dense matrix
            if isinstance(Jz, LinearOperator):
                # LinearOperator: use matvec (expects 1D vector, returns 1D)
                jz = np.vdot(psi, Jz.matvec(psi)).real
            elif hasattr(Jz, 'dot'):
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

def single_bin_initial_state(n1: int, n2: int, *, dtype: np.dtype = DEFAULT_DTYPE):
    """
    Build the symmetric Dicke state for one bin with n1 ν_e (spin-up) and n2 ν_μ (spin-down).
    That's |S=N/2, m=(n1-n2)/2> in the symmetric subspace.
    """
    N = n1 + n2
    S = N / 2.0
    m = (n1 - n2) / 2.0
    v = dicke_basis_vector(S, m, dtype=dtype)
    return v, S

def multi_bin_initial_state(n1_list, n2_list, *, dtype: np.dtype = DEFAULT_DTYPE):
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
    psi0 = product_dicke_state(S_list, m_list, dtype=dtype)
    return psi0, S_list

def product_state(vecs, *, dtype: np.dtype = DEFAULT_DTYPE):
    """Kronecker product of a list of state vectors.
    
    Returns a sparse vector to conserve memory for large systems.
    """
    init_val = dtype(1.0 + 0.0j)
    out = csr_array([[init_val]], dtype=dtype)  # Start with a 1x1 sparse array
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