#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dicke_linop.py
=================

LinearOperator-based evolution utilities for collective neutrino oscillations
in Dicke subspaces.

This is an optimized successor to ``dicke_collective_sparse_opt_linop.py``.
It keeps the same public API (build_* / evolve_*), and adds a few knobs that
are particularly valuable for checkpointed, distributed runs where you only
advance a small number of steps per job:

Key upgrades
------------
- **Reuse of A = -iH** inside ``evolve_times_stream`` (avoid repeated wrapper
  construction each block/step).
- **Optional no-copy streaming** via ``copy_state=False`` to avoid large
  ``psi.copy()`` bandwidth costs.
- **End-to-end dtype control** via ``dtype=...`` (complex128 by default; can
  use complex64 if your validation allows it).
- **Fast observables**: compute \langle Jz \rangle and Pe using the diagonal
  structure of Jz in the Dicke basis when ``dims`` is available (no sparse
  matvec).

Notes
-----
The Hamiltonian construction here still lifts local spin operators to the full
tensor product space via Kronecker products. This keeps the interface simple
and drop-in compatible, but it does not eliminate the memory footprint of the
lifted operators.

"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import identity, kron as sparse_kron
from scipy.sparse.linalg import LinearOperator, expm_multiply

# Local (single-bin) spin operators in the Dicke basis.
from dicke_collective_sparse_opt import spin_matrices


ComplexDType = np.dtype


def _normalize_complex_dtype(dtype: Optional[object]) -> np.dtype:
    """Normalize dtype inputs (np dtype, python type, or string)."""
    if dtype is None:
        return np.dtype(np.complex128)
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"dtype must be complex64 or complex128, got {dt!r}")
    return dt


def _kron_on_slot(op, slot: int, dims: Sequence[int], *, dtype: np.dtype):
    """Place ``op`` on tensor slot ``slot`` with identities elsewhere.

    This is a dtype-aware replacement for ``dicke_collective_sparse_opt.kron_on_slot``.
    Using ``identity(..., dtype=dtype)`` avoids accidental upcasting when you
    want to run in complex64.
    """
    out = None
    for a, d in enumerate(dims):
        A = op if a == slot else identity(d, format="csr", dtype=dtype)
        out = A if out is None else sparse_kron(out, A, format="csr")
    return out


def build_single_bin_hamiltonian(
    N: int,
    omega: float,
    theta_v: float,
    mu: float,
    *,
    dtype: Optional[object] = None,
):
    """Single-bin Hamiltonian in the fully symmetric Dicke subspace.

    Returns ``H`` as a SciPy ``LinearOperator``.
    """
    dt = _normalize_complex_dtype(dtype)
    S = N / 2.0

    Jx, Jy, Jz = spin_matrices(S)
    if Jx.dtype != dt:
        Jx = Jx.astype(dt, copy=False)
        Jy = Jy.astype(dt, copy=False)
        Jz = Jz.astype(dt, copy=False)

    Bx = float(np.sin(2.0 * theta_v))
    Bz = float(-np.cos(2.0 * theta_v))
    dim = int(Jx.shape[0])

    def _dot(A, v):
        return A.dot(v) if hasattr(A, "dot") else (A @ v)

    def _mv(v):
        v = np.asarray(v, dtype=dt).reshape(-1)

        # Vacuum term uses Jx v and Jz v.
        Jxv = _dot(Jx, v)
        Jzv = _dot(Jz, v)
        out = omega * (Bx * Jxv + Bz * Jzv)

        # Interaction term: mu * (Jx^2 + Jy^2 + Jz^2) v
        if mu != 0.0:
            Jyv = _dot(Jy, v)
            out += mu * (_dot(Jx, Jxv) + _dot(Jy, Jyv) + _dot(Jz, Jzv))
        return np.asarray(out, dtype=dt)

    H = LinearOperator((dim, dim), matvec=_mv, dtype=dt)

    # Cache trace(A) for A = -i H (helps expm_multiply avoid trace estimation).
    # Vacuum term is traceless; J^2 = S(S+1) I in the irreducible rep.
    trace_H = float(mu * S * (S + 1.0) * dim)
    H.traceA = (-1j) * trace_H
    return H, (Jx, Jy, Jz), [S], [int(2 * S + 1)]


def build_multi_bin_hamiltonian(
    N_list: Sequence[int],
    omega_list: Sequence[float],
    theta_v: float,
    mu: float,
    *,
    is_antineutrino: Optional[Sequence[bool]] = None,
    mu_cross: Optional[float] = None,
    dtype: Optional[object] = None,
):
    """Multi-bin Hamiltonian as a LinearOperator.

    Model
    -----
    H = Σ_a ω_a (B·J_a) + Σ_{a<b} μ_{ab} PairTerm(J_a, J_b)

    If ``is_antineutrino`` is provided, cross-species pairs (ν vs ν̄) use the
    *physical* anisotropic bilinear:

        - Jx_a Jx_b + Jy_a Jy_b - Jz_a Jz_b

    while same-species pairs use the Heisenberg form:

        + Jx_a Jx_b + Jy_a Jy_b + Jz_a Jz_b

    If ``is_antineutrino`` is None, the builder falls back to the original
    Heisenberg coupling for all pairs (backwards compatible).
    """
    if len(N_list) != len(omega_list):
        raise ValueError("N_list and omega_list must have the same length")

    dt = _normalize_complex_dtype(dtype)

    K = len(N_list)
    S_list = [n / 2.0 for n in N_list]
    locals_ops = [spin_matrices(S) for S in S_list]
    if locals_ops and locals_ops[0][0].dtype != dt:
        locals_ops = [(Jx.astype(dt, copy=False), Jy.astype(dt, copy=False), Jz.astype(dt, copy=False))
                     for (Jx, Jy, Jz) in locals_ops]

    dims = [int(ops[0].shape[0]) for ops in locals_ops]

    # Lifted operators (still sparse, but live in the full tensor space).
    Jx_list: List = []
    Jy_list: List = []
    Jz_list: List = []
    for a, (Jx, Jy, Jz) in enumerate(locals_ops):
        Jx_list.append(_kron_on_slot(Jx, a, dims, dtype=dt))
        Jy_list.append(_kron_on_slot(Jy, a, dims, dtype=dt))
        Jz_list.append(_kron_on_slot(Jz, a, dims, dtype=dt))

    dim = int(np.prod(dims))

    Bx = float(np.sin(2.0 * theta_v))
    Bz = float(-np.cos(2.0 * theta_v))

    if is_antineutrino is None:
        is_antineutrino = [False] * K
    else:
        if len(is_antineutrino) != K:
            raise ValueError("is_antineutrino must have the same length as N_list")

    if mu_cross is None:
        mu_cross = mu

    def _dot(A, v):
        return A.dot(v) if hasattr(A, "dot") else (A @ v)

    def _mv(v):
        v = np.asarray(v, dtype=dt).reshape(-1)
        out = np.zeros_like(v, dtype=dt)

        # Vacuum term.
        for a, om in enumerate(omega_list):
            if om != 0.0:
                out += om * (Bx * _dot(Jx_list[a], v) + Bz * _dot(Jz_list[a], v))

        # Interaction terms.
        if mu != 0.0 or (mu_cross is not None and mu_cross != 0.0):
            for a in range(K):
                for b in range(a + 1, K):
                    same = bool(is_antineutrino[a]) == bool(is_antineutrino[b])
                    mu_ab = mu if same else mu_cross
                    if mu_ab == 0.0:
                        continue

                    # Signs: same species -> (+,+,+); cross species -> (-,+,-)
                    sx = 1.0 if same else -1.0
                    sy = 1.0
                    sz = 1.0 if same else -1.0

                    out += (mu_ab * sx) * _dot(Jx_list[a], _dot(Jx_list[b], v))
                    out += (mu_ab * sy) * _dot(Jy_list[a], _dot(Jy_list[b], v))
                    out += (mu_ab * sz) * _dot(Jz_list[a], _dot(Jz_list[b], v))

        return out

    H = LinearOperator((dim, dim), matvec=_mv, dtype=dt)
    # Cross-bin bilinears are traceless; cache trace(A)=0 for A=-iH.
    H.traceA = 0.0
    return H, (Jx_list, Jy_list, Jz_list), S_list, dims


def _as_A(H, *, dtype: np.dtype):
    """Return A=-iH in a form accepted by expm_multiply, plus traceA."""
    traceA = getattr(H, "traceA", None)
    if isinstance(H, LinearOperator):
        # Construct once; expm_multiply will repeatedly call matvec.
        A = LinearOperator(
            H.shape,
            matvec=lambda v: (-1j) * H.matvec(v),
            rmatvec=lambda v: (1j) * H.matvec(v),
            dtype=dtype,
        )
        return A, traceA
    return (-1j) * H, traceA


def evolve_times(H, psi0, t_grid, *, dtype: Optional[object] = None):
    """Evolve ``psi0`` under ``H`` for sample times in ``t_grid``."""
    dt = _normalize_complex_dtype(dtype if dtype is not None else getattr(H, "dtype", None))
    psi0 = np.asarray(psi0, dtype=dt).reshape(-1)

    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size == 0:
        raise ValueError("t_grid must be non-empty")

    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    A, traceA = _as_A(H, dtype=dt)
    Y = expm_multiply(A, psi0, start=t0, stop=t1, num=int(t_grid.size), endpoint=True, traceA=traceA)
    return np.asarray(Y)


def evolve_times_stream(
    H,
    psi0,
    t_grid,
    *,
    chunk: int = 64,
    normalize: bool = False,
    copy_state: bool = True,
    dtype: Optional[object] = None,
):
    """Streaming evolution generator.

    Parameters
    ----------
    copy_state:
        If True (default), yield a ``psi.copy()`` at each time. This is safe
        but can be very expensive for large states.

        If False, yield the internal state array directly. This is faster,
        but callers must treat the returned array as *ephemeral* and must not
        store it for later use.
    """
    dt = _normalize_complex_dtype(dtype if dtype is not None else getattr(H, "dtype", None))

    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size == 0:
        raise ValueError("t_grid must be non-empty")

    psi = np.asarray(psi0, dtype=dt).reshape(-1)

    # Build A=-iH once.
    A, traceA = _as_A(H, dtype=dt)

    def _emit(t: float, vec: np.ndarray):
        return (t, vec.copy() if copy_state else vec)

    yield _emit(float(t_grid[0]), psi)

    # Decide uniform vs non-uniform once.
    if t_grid.size < 2:
        return
    d = np.diff(t_grid)
    uniform = bool(np.allclose(d, d[0]))

    i = 0
    while i < t_grid.size - 1:
        if uniform:
            n_block = int(min(chunk, t_grid.size - i - 1))
            rel_stop = float(t_grid[i + n_block] - t_grid[i])
            Y = expm_multiply(A, psi, start=0.0, stop=rel_stop, num=n_block + 1, endpoint=True, traceA=traceA)
            for k in range(1, n_block + 1):
                psi = np.asarray(Y[k]).reshape(-1)
                if normalize:
                    psi = psi / np.linalg.norm(psi)
                yield _emit(float(t_grid[i + k]), psi)
            i += n_block
        else:
            dt_step = float(t_grid[i + 1] - t_grid[i])
            Y = expm_multiply(A, psi, start=0.0, stop=dt_step, num=2, endpoint=True, traceA=traceA)
            psi = np.asarray(Y[-1]).reshape(-1)
            if normalize:
                psi = psi / np.linalg.norm(psi)
            i += 1
            yield _emit(float(t_grid[i]), psi)


def _m_vals_for_spin(S: float) -> np.ndarray:
    # m = -S, -S+1, ..., S
    # Use float64 regardless of complex dtype.
    return np.arange(-S, S + 1.0, 1.0, dtype=float)


def jz_expectations_diag(psi: np.ndarray, S_list: Sequence[float], dims: Sequence[int]) -> np.ndarray:
    """Compute <Jz_a> for each bin using only |psi|^2 and the Dicke m grid.

    This avoids sparse matvec with the lifted Jz matrices (which are diagonal
    but still incur index overhead).
    """
    psi = np.asarray(psi)
    K = len(S_list)
    if K != len(dims):
        raise ValueError("S_list and dims must have the same length")

    if K == 1:
        m0 = _m_vals_for_spin(S_list[0])
        # Use |psi|^2 without creating a full complex intermediate.
        p = np.abs(psi) ** 2
        return np.array([float(np.dot(m0, p))], dtype=float)

    # Reshape once.
    psi_t = psi.reshape(tuple(dims))

    if K == 2:
        d0, d1 = dims
        m0 = _m_vals_for_spin(S_list[0])
        m1 = _m_vals_for_spin(S_list[1])

        # Two-bin fast path: compute marginals without allocating a full |psi|^2 tensor.
        # p0[i] = Σ_j |psi[i,j]|^2,  p1[j] = Σ_i |psi[i,j]|^2
        psi_mat = psi_t.reshape((d0, d1))
        p0 = np.einsum('ij,ij->i', psi_mat.conj(), psi_mat).real
        p1 = np.einsum('ij,ij->j', psi_mat.conj(), psi_mat).real
        return np.array([float(np.dot(m0, p0)), float(np.dot(m1, p1))], dtype=float)

    # Generic K-bin path.
    abs2 = np.abs(psi_t) ** 2
    out = np.empty((K,), dtype=float)
    for a in range(K):
        m = _m_vals_for_spin(S_list[a])
        axes = tuple(i for i in range(K) if i != a)
        marg = abs2.sum(axis=axes)
        out[a] = float(np.dot(m, marg))
    return out


def observables_from_stream_diag(
    stream: Iterable[Tuple[float, np.ndarray]],
    S_list: Sequence[float],
    dims: Sequence[int],
    flip_bin: Optional[Sequence[bool]] = None,
):
    """Consume a (t, psi) stream and compute Jz and Pe using diagonal formulas."""
    times: List[float] = []
    Jz_vals: List[np.ndarray] = []
    Pe_vals: List[np.ndarray] = []

    S_arr = np.asarray(S_list, dtype=float)
    K = len(S_list)
    
    # Build flip mask: +1 for normal, -1 for flipped bins
    if flip_bin is not None:
        flip_mask = np.array([(-1.0 if flip_bin[i] else 1.0) for i in range(K)], dtype=float)
    else:
        flip_mask = np.ones(K, dtype=float)
    
    for (t, psi) in stream:
        times.append(float(t))
        jz = jz_expectations_diag(psi, S_list=S_list, dims=dims)
        Jz_vals.append(jz)
        Pe_vals.append(0.5 * (1.0 + flip_mask * jz / S_arr))

    return (
        np.asarray(times, dtype=float),
        np.asarray(Jz_vals, dtype=float),
        np.asarray(Pe_vals, dtype=float),
    )


def observables_from_stream(
    stream: Iterable[Tuple[float, np.ndarray]],
    Jz_list,
    S_list: Sequence[float],
    *,
    dims: Optional[Sequence[int]] = None,
    flip_bin: Optional[Sequence[bool]] = None,
):
    """Backwards-compatible observables helper.

    - If ``dims`` is provided, uses the faster diagonal formula.
    - Otherwise, falls back to sparse matvec with ``Jz_list``.
    """
    if dims is not None:
        return observables_from_stream_diag(stream, S_list=S_list, dims=dims, flip_bin=flip_bin)

    times: List[float] = []
    Jz_vals: List[List[float]] = []
    Pe_vals: List[List[float]] = []
    for (t, psi) in stream:
        psi = np.asarray(psi).ravel()
        times.append(float(t))
        Jz_row: List[float] = []
        Pe_row: List[float] = []
        for i, (Jz, S) in enumerate(zip(Jz_list, S_list)):
            vec = Jz.dot(psi)
            expJz = np.vdot(psi, vec).real
            Jz_row.append(float(expJz))
            if flip_bin is not None and flip_bin[i]:
                Pe_row.append(float(1.0 - 0.5 * (1.0 + expJz / S)))
            else:
                Pe_row.append(float(0.5 * (1.0 + expJz / S)))
        Jz_vals.append(Jz_row)
        Pe_vals.append(Pe_row)

    return (
        np.asarray(times, dtype=float),
        np.asarray(Jz_vals, dtype=float),
        np.asarray(Pe_vals, dtype=float),
    )
