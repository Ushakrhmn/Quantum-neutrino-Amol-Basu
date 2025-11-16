#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal, drop-in replacements for key functions using LinearOperator
====================================================================

This module provides LinearOperator-based implementations that avoid
building the full Hamiltonian matrix H explicitly. This is crucial for
large systems where storing H would be memory-prohibitive.

The functions here are drop-in replacements for:
- build_single_bin_hamiltonian
- build_multi_bin_hamiltonian
- evolve_times
- evolve_times_stream

These require the helper functions from dicke_collective_sparse_opt.py:
- spin_matrices(S)
- kron_on_slot(op, slot, dims)
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply

# Import helper functions from the main module
from dicke_collective_sparse_opt import spin_matrices, kron_on_slot


def build_single_bin_hamiltonian(N: int, omega: float, theta_v: float, mu: float):
    """
    Single-energy homogeneous gas (all-to-all equal coupling).
    In the fully symmetric S=N/2 Dicke subspace.
    
    Returns LinearOperator instead of explicit matrix to avoid memory issues.
    
    Parameters
    ----------
    N : int
        Number of neutrinos (total spin S = N/2)
    omega : float
        Vacuum oscillation frequency
    theta_v : float
        Vacuum mixing angle
    mu : float
        Neutrino-neutrino interaction strength
    
    Returns
    -------
    H : LinearOperator
        Hamiltonian as a LinearOperator (matrix-free)
    (Jx, Jy, Jz) : tuple
        Spin operators (sparse matrices)
    [S] : list
        List containing the spin magnitude
    [dim] : list
        List containing the dimension
    """
    S = N / 2.0
    Jx, Jy, Jz = spin_matrices(S)
    Bx = np.sin(2 * theta_v); Bz = -np.cos(2 * theta_v)
    dim = Jx.shape[0]

    def _dot(A, v): return A.dot(v) if hasattr(A, "dot") else (A @ v)

    def _mv(v):
        v = np.asarray(v, np.complex128).reshape(-1)
        out = omega * (Bx * _dot(Jx, v) + Bz * _dot(Jz, v))
        if mu != 0.0:
            out = out + mu * (_dot(Jx, _dot(Jx, v)) + _dot(Jy, _dot(Jy, v)) + _dot(Jz, _dot(Jz, v)))
        return out

    H = LinearOperator((dim, dim), matvec=_mv, dtype=np.complex128)
    # Cache trace(A) for A = -i H to avoid repeated trace estimation in expm_multiply.
    # Here H has trace mu * S * (S+1) * (2S+1) in the S=N/2 irreducible representation.
    trace_H = mu * S * (S + 1.0) * dim
    H.traceA = -1j * trace_H
    return H, (Jx, Jy, Jz), [S], [int(2*S+1)]


def build_multi_bin_hamiltonian(N_list, omega_list, theta_v: float, mu: float):
    """
    Multi-energy, single-angle equal coupling μ for all inter-bin pairs:
      H = Σ_a ω_a (B·J_a) + μ Σ_{a<b} J_a · J_b
    
    Returns LinearOperator instead of explicit matrix to avoid memory issues.
    
    Parameters
    ----------
    N_list : list of int
        Number of neutrinos per bin
    omega_list : list of float
        Vacuum oscillation frequencies per bin
    theta_v : float
        Vacuum mixing angle
    mu : float
        Neutrino-neutrino interaction strength
    
    Returns
    -------
    H : LinearOperator
        Hamiltonian as a LinearOperator (matrix-free)
    (Jx_list, Jy_list, Jz_list) : tuple
        Lists of spin operators for each bin
    S_list : list
        List of spin magnitudes for each bin
    dims : list
        List of dimensions for each bin
    """
    assert len(N_list) == len(omega_list)
    S_list = [n/2.0 for n in N_list]
    locals_ops = [spin_matrices(S) for S in S_list]
    dims = [ops[0].shape[0] for ops in locals_ops]
    Jx_list, Jy_list, Jz_list = [], [], []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(kron_on_slot(Jx, a, dims))
        Jy_list.append(kron_on_slot(Jy, a, dims))
        Jz_list.append(kron_on_slot(Jz, a, dims))
    dim = int(np.prod(dims))
    Bx = np.sin(2 * theta_v); Bz = -np.cos(2 * theta_v)

    def _dot(A, v): return A.dot(v) if hasattr(A, "dot") else (A @ v)

    def _mv(v):
        v = np.asarray(v, np.complex128).reshape(-1)
        out = np.zeros_like(v, dtype=np.complex128)
        for a, om in enumerate(omega_list):
            if om != 0.0:
                out = out + om * (Bx * _dot(Jx_list[a], v) + Bz * _dot(Jz_list[a], v))
        if mu != 0.0:
            K = len(Jx_list)
            for a in range(K):
                for b in range(a+1, K):
                    out = out + mu * _dot(Jx_list[a], _dot(Jx_list[b], v))
                    out = out + mu * _dot(Jy_list[a], _dot(Jy_list[b], v))
                    out = out + mu * _dot(Jz_list[a], _dot(Jz_list[b], v))
        return out

    H = LinearOperator((dim, dim), matvec=_mv, dtype=np.complex128)
    # Multi-bin Hamiltonian (with only cross-bin interactions) is traceless,
    # so trace(A) = trace(-i H) = 0. Cache this to skip trace estimation.
    H.traceA = 0.0
    return H, (Jx_list, Jy_list, Jz_list), S_list, dims


def evolve_times(H, psi0, t_grid):
    """
    Evolve psi0 under Hamiltonian H for times in t_grid.
    
    Works with both explicit matrices and LinearOperators.
    
    Parameters
    ----------
    H : array, sparse matrix, or LinearOperator
        Hamiltonian
    psi0 : (N,) array
        Initial state
    t_grid : array_like
        Monotonically increasing list/array of times
    
    Returns
    -------
    Y : ndarray, shape (len(t_grid), N)
        State at each time in t_grid
    """
    # Ensure psi0 is dense
    psi0 = np.asarray(psi0, dtype=np.complex128).flatten()
    t0, t1 = float(t_grid[0]), float(t_grid[-1]); num = len(t_grid)
    # Optional cached trace(A) for A = -i H to avoid repeated trace estimation
    traceA = getattr(H, "traceA", None)

    # If H is a LinearOperator, wrap -i*H as another LinearOperator
    # Otherwise, use -1j*H directly (works for arrays/sparse matrices)
    if isinstance(H, LinearOperator):
        # Provide both matvec and rmatvec (adjoint) for proper LinearOperator support
        # For -i*H where H is Hermitian, the adjoint is i*H, so rmatvec = i*H.matvec
        # Since H is a Hamiltonian (Hermitian), we always use H.matvec for the adjoint
        A = LinearOperator(
            H.shape, 
            matvec=lambda v: (-1j)*H.matvec(v),
            rmatvec=lambda v: (1j)*H.matvec(v),  # Adjoint of -i*H is i*H (since H is Hermitian)
            dtype=np.complex128
        )
    else:
        A = -1j * H
    Y = expm_multiply(A, psi0, start=t0, stop=t1, num=num, endpoint=True, traceA=traceA)
    return np.asarray(Y)


def evolve_times_stream(H, psi0, t_grid, *, chunk=64, normalize=False):
    """
    Generator: evolve |psi0> under Hamiltonian H and yield |psi(t_k)> one-by-one.
    
    Works with both explicit matrices and LinearOperators. Memory-efficient
    streaming version that doesn't store all states at once.
    
    Parameters
    ----------
    H : array, sparse matrix, or LinearOperator
        Hamiltonian
    psi0 : (D,) complex ndarray
        Initial state vector (dense)
    t_grid : 1D ndarray
        Sample times (monotonic, typically linspace)
    chunk : int, optional
        Number of points to compute in each internal block
    normalize : bool, optional
        If True, L2-normalize |psi> after every step
    
    Yields
    ------
    (t, psi) : tuple[float, ndarray]
        The time and the state at that time (copy)
    """
    import numpy as _np
    from scipy.sparse.linalg import expm_multiply as _expm_multiply, LinearOperator
    
    H_is_linop = isinstance(H, LinearOperator)
    # Optional cached trace(A) for A = -i H to avoid repeated trace estimation
    traceA = getattr(H, "traceA", None)
    
    # Prepare initial
    psi = _np.asarray(psi0, dtype=_np.complex128).reshape(-1)
    yield (float(t_grid[0]), psi.copy())
    i = 0
    
    def _is_uniform(arr):
        if arr.size < 3: return True
        d = _np.diff(arr); return _np.allclose(d, d[0])
    
    while i < len(t_grid) - 1:
        if _is_uniform(t_grid[i:i+min(chunk, len(t_grid)-i)]):
            n_block = min(chunk, len(t_grid)-i-1)
            t0 = float(t_grid[i]); t1 = float(t_grid[i + n_block]); rel_stop = t1 - t0
            if H_is_linop:
                # For -i*H where H is Hermitian, the adjoint is i*H
                # Since H is a Hamiltonian (Hermitian), we use H.matvec for the adjoint
                A = LinearOperator(
                    H.shape, 
                    matvec=lambda v: (-1j)*H.matvec(v),
                    rmatvec=lambda v: (1j)*H.matvec(v),  # Adjoint of -i*H is i*H
                    dtype=_np.complex128
                )
            else:
                A = -1j * H
            Y = _expm_multiply(A, psi, start=0.0, stop=rel_stop, num=n_block+1, endpoint=True, traceA=traceA)
            for k in range(1, n_block+1):
                psi = _np.asarray(Y[k]).reshape(-1)
                if normalize: psi = psi / _np.linalg.norm(psi)
                yield (float(t_grid[i+k]), psi.copy())
            i += n_block
        else:
            dt = float(t_grid[i+1] - t_grid[i])
            if H_is_linop:
                # For -i*H where H is Hermitian, the adjoint is i*H
                # Since H is a Hamiltonian (Hermitian), we use H.matvec for the adjoint
                A = LinearOperator(
                    H.shape, 
                    matvec=lambda v: (-1j)*H.matvec(v),
                    rmatvec=lambda v: (1j)*H.matvec(v),  # Adjoint of -i*H is i*H
                    dtype=_np.complex128
                )
            else:
                A = -1j * H
            Y = _expm_multiply(A, psi, start=0.0, stop=dt, num=2, endpoint=True, traceA=traceA)
            psi = _np.asarray(Y[-1]).reshape(-1)
            if normalize: psi = psi / _np.linalg.norm(psi)
            i += 1
            yield (float(t_grid[i]), psi.copy())

