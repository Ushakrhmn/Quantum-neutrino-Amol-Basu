#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local single-step validation: MFT vs Dicke (no entanglement propagation)
========================================================================

Flow per step k -> k+1:
  (A) Mean-field (MFT): integrate Raffelt–Sigl polarization-vector ODEs for Δt from the
      *product* bin states at step k to obtain P^{MFT}(k+1).
  (B) Quantum (Dicke): starting from the *same* step-k MFT product state, build per-bin
      spin-coherent states in the Dicke basis, form their tensor product, and evolve a
      *single* step Δt with either:
        - first-order Trotter, or
        - Strang (symmetric) Trotter, or
        - exact expm_multiply under the full multi-bin Hamiltonian (LinearOperator).
      Record P^{Dicke}(k+1).
  (C) Deviation: ΔP_e = P^{Dicke}(k+1) - P^{MFT}(k+1).

Important: The Dicke entanglement generated within the step is *not* propagated to step k+1
(i.e., we always reseed Dicke from the *MFT* product state at the beginning of each step),
so we are measuring the *local* single-step error of the MFT flow.

Requires local modules:
  - mft.py (Raffelt–Sigl ODEs; sigma_i, etc.)
  - dicke_linearop.py (Dicke operators with LinearOperator support)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime

from scipy.sparse.linalg import LinearOperator, expm_multiply

import mft
import dicke_linearop as dc

DEFAULT_DTYPE = np.complex128


# ---------- utilities ----------

def _ensure_dense1d(v, dtype=DEFAULT_DTYPE):
    if hasattr(v, "toarray"):
        v = v.toarray()
    v = np.asarray(v, dtype=dtype).reshape(-1)
    return v

def _unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


# ---------- MFT: one-step integrator with arbitrary Bloch initial data ----------

def mft_step(P_init: np.ndarray,
             theta_v: float,
             omegas: np.ndarray,
             lam: float,
             J_coupling: np.ndarray,
             t0: float,
             dt: float) -> np.ndarray:
    """
    One explicit step for Raffelt–Sigl polarization vectors:
      dP_i/dt = (ω_i B + λ L + Σ_j J_{ij} P_j) × P_i
    Integrate from t0 to t0+dt with solve_ivp.
    P_init: shape (N, 3). Returns same shape.
    """
    import scipy.integrate as integ

    N = P_init.shape[0]
    # build B, L in the same way as mft.P_osc_RS
    u = np.array([[np.cos(theta_v), np.sin(theta_v)],
                  [-np.sin(theta_v), np.cos(theta_v)]])
    b = 0.5 * np.diag([-1.0, 1.0])
    b = u @ b @ u.T
    l = np.diag([1.0, 0.0])

    B = np.real(np.array([np.trace(b @ m) for m in (mft.sigma_1, mft.sigma_2, mft.sigma_3)]))
    L = np.real(np.array([np.trace(l @ m) for m in (mft.sigma_1, mft.sigma_2, mft.sigma_3)]))

    def rhs(_t, P_flat):
        P = P_flat.reshape(N, 3).T  # (3, N)
        res = np.zeros((3, N), dtype=float)
        JP = P @ J_coupling.T       # (3, N)
        for k in range(N):
            Hk = omegas[k] * B + lam * L + JP[:, k]
            res[:, k] = np.cross(Hk, P[:, k])
        return res.T.reshape(3 * N)

    sol = integ.solve_ivp(rhs, (t0, t0 + dt), P_init.reshape(-1),
                          t_eval=[t0 + dt], rtol=1e-8, atol=1e-10)
    return sol.y.reshape(N, 3)


# ---------- Spin-coherent state from a Bloch vector (per bin) ----------

def spin_coherent_state_from_P(S: float, P_vec: np.ndarray, *, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """
    Build the Dicke spin-coherent state |θ,φ> = e^{-i φ Jz} e^{-i θ Jy} |S,S>
    whose mean polarization points along the given Bloch direction P_vec (|P_vec|≈1).
    Returns a dense 1D vector in the |m> Dicke basis of dimension 2S+1.
    """
    P = np.asarray(P_vec, dtype=float)
    nrm = np.linalg.norm(P)
    if nrm < 1e-12:
        # default to +z
        theta, phi = 0.0, 0.0
    else:
        Pn = P / nrm
        theta = float(np.arccos(np.clip(Pn[2], -1.0, 1.0)))
        phi = float(np.arctan2(Pn[1], Pn[0]))
    # local spin matrices
    Jx, Jy, Jz = dc.spin_matrices(S, dtype=dtype)
    # start from |S,S>
    v0 = dc.dicke_basis_vector(S, S, dtype=dtype)
    v = _ensure_dense1d(v0, dtype=dtype)
    # rotate by e^{-i θ Jy} then e^{-i φ Jz}
    v = _ensure_dense1d(expm_multiply((-1j*theta) * Jy, v), dtype=dtype)
    v = _ensure_dense1d(expm_multiply((-1j*phi)  * Jz, v), dtype=dtype)
    return _unit(v)


# ---------- Global generators for one-step Trotter/Strang ----------

def build_local_generators(Jx_list, Jz_list, omega_list, theta_v, *, dtype=DEFAULT_DTYPE):
    """
    G_a = ω_a (B·J_a), with B=(sin2θ, 0, -cos2θ). Operates on the global space as LinearOperators.
    """
    Bx = float(np.sin(2.0 * theta_v))
    Bz = float(-np.cos(2.0 * theta_v))
    D = Jx_list[0].shape[0]
    G_loc = []
    for a, omega in enumerate(omega_list):
        def make_mv(a=a, omega=omega):
            def _mv(x):
                return (omega * Bx) * Jx_list[a].matvec(x) + (omega * Bz) * Jz_list[a].matvec(x)
            return _mv
        mv = make_mv()
        G = LinearOperator((D, D), matvec=mv, rmatvec=mv, dtype=dtype)
        G_loc.append(G)
    return G_loc

def build_interaction_generator(Jx_list, Jy_list, Jz_list, mu, *, dtype=DEFAULT_DTYPE):
    """
    G_int = μ Σ_{a<b} J_a·J_b as a LinearOperator on the global space.
    """
    D = Jx_list[0].shape[0]
    A = len(Jx_list)
    def _mv(x):
        out = np.zeros_like(x, dtype=dtype)
        for a in range(A):
            for b in range(a + 1, A):
                out = out + Jx_list[a].matvec(Jx_list[b].matvec(x))
                out = out + Jy_list[a].matvec(Jy_list[b].matvec(x))
                out = out + Jz_list[a].matvec(Jz_list[b].matvec(x))
        return mu * out
    return LinearOperator((D, D), matvec=_mv, rmatvec=_mv, dtype=dtype)

def apply_exp_generator(G: LinearOperator, dt: float, psi, *, dtype=DEFAULT_DTYPE):
    psi = _ensure_dense1d(psi, dtype=dtype)
    A = LinearOperator(G.shape, matvec=lambda x: (-1j*dt)*G.matvec(x),
                       rmatvec=lambda x: (-1j*dt)*G.rmatvec(x), dtype=dtype)
    Y = expm_multiply(A, psi, start=0.0, stop=1.0, num=2, endpoint=True)
    return _ensure_dense1d(Y[-1], dtype=dtype)

def apply_exact_step(H: LinearOperator, dt: float, psi, *, dtype=DEFAULT_DTYPE):
    psi = _ensure_dense1d(psi, dtype=dtype)
    A = LinearOperator(H.shape, matvec=lambda x: (-1j*dt)*H.matvec(x),
                       rmatvec=lambda x: (-1j*dt)*H.rmatvec(x), dtype=dtype)
    Y = expm_multiply(A, psi, start=0.0, stop=1.0, num=2, endpoint=True)
    return _ensure_dense1d(Y[-1], dtype=dtype)


# ---------- Observables ----------

def pe_from_full_state(psi, Jz_list, S_list):
    states = np.asarray([_ensure_dense1d(psi)])
    _, Pee = dc.bin_observables(states, Jz_list, S_list)
    return Pee[0]  # shape (A,)


# ---------- Driver ----------

def run_localcheck(e1, m1, e2, m2,
                   energy1, energy2, theta_v, dmsq,
                   mu, baseline, steps,
                   evolver: str = "trotter",
                   lam: float = 0.0,
                   zero_self: bool = False,
                   dicke_steps: int = 1,
                   dtype=DEFAULT_DTYPE):
    """
    evolver: 'trotter' (first-order), 'strang' (symmetric), or 'exact' (full H).
    dicke_steps: number of Dicke evolution steps per MFT step (default 1).
    """
    # counts, frequencies
    n1, n2 = e1 + m1, e2 + m2
    N = n1 + n2
    omega1, omega2 = dmsq/(2.0*energy1), dmsq/(2.0*energy2)

    # coupling matrix for MFT
    J_cpl = mu * np.ones((N, N), dtype=float)
    if zero_self:
        np.fill_diagonal(J_cpl, 0.0)

    # Dicke operators (global) for the two-bin system
    H_full, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
        N_list=[n1, n2],
        omega_list=[omega1, omega2],
        theta_v=theta_v,
        mu=mu
    )
    # prebuild generators
    G_loc = build_local_generators(Jx_list, Jz_list, [omega1, omega2], theta_v, dtype=dtype)
    G_int = build_interaction_generator(Jx_list, Jy_list, Jz_list, mu, dtype=dtype)

    # initial MFT polarization per neutrino (product within each bin)
    P = np.zeros((N, 3), dtype=float)
    # bin 1
    P[:e1, :] = np.array([0.0, 0.0,  1.0])  # e
    P[e1:e1+m1, :] = np.array([0.0, 0.0, -1.0])  # μ
    # bin 2
    off = n1
    P[off:off+e2, :] = np.array([0.0, 0.0,  1.0])
    P[off+e2:off+e2+m2, :] = np.array([0.0, 0.0, -1.0])

    # time grid
    t_grid = np.linspace(0.0, baseline, steps + 1)
    dt = float(baseline) / float(steps)

    # storage
    mft_pe = np.zeros((steps + 1, 2), dtype=float)
    dicke_pe = np.zeros((steps + 1, 2), dtype=float)
    dev_pe = np.zeros((steps + 1, 2), dtype=float)

    # t=0 observables
    # MFT P_e from P
    P1_0 = P[:n1, :]
    P2_0 = P[n1:, :]
    mft_pe[0, 0] = 0.5 * (1.0 + np.mean(P1_0[:, 2]))
    mft_pe[0, 1] = 0.5 * (1.0 + np.mean(P2_0[:, 2]))

    # Dicke at t=0 from the same product: coherent states aligned with +z/-z aggregate
    phi1_0 = spin_coherent_state_from_P(S_list[0], np.array([0.0, 0.0, (e1 - m1)/float(n1) if n1>0 else 1.0]))
    phi2_0 = spin_coherent_state_from_P(S_list[1], np.array([0.0, 0.0, (e2 - m2)/float(n2) if n2>0 else 1.0]))
    psi0 = np.kron(phi1_0, phi2_0).astype(dtype)
    dicke_pe[0, :] = pe_from_full_state(psi0, Jz_list, S_list)
    dev_pe[0, :] = dicke_pe[0, :] - mft_pe[0, :]

    # main loop
    P_curr = P.copy()
    # Initialize persistent Dicke state from initial MFT product state
    psi_dicke = psi0.copy()

    for k in range(steps):
        t_k = t_grid[k]

        # (A) one MFT step from P_curr
        P_next = mft_step(P_curr, theta_v, np.array([omega1]*n1 + [omega2]*n2),
                          lam=lam, J_coupling=J_cpl, t0=t_k, dt=dt)
        # record MFT Pee at t_{k+1}
        P1 = P_next[:n1, :]
        P2 = P_next[n1:, :]
        mft_pe[k+1, 0] = 0.5 * (1.0 + np.mean(P1[:, 2]))
        mft_pe[k+1, 1] = 0.5 * (1.0 + np.mean(P2[:, 2]))

        # (B) Dicke evolution: restart from MFT result every dicke_steps steps
        #     Otherwise continue evolving the persistent Dicke state
        if k % dicke_steps == 0:
            # Restart Dicke state from current MFT product state
            P1_avg = P_curr[:n1, :].mean(axis=0) if n1>0 else np.array([0.0,0.0,1.0])
            P2_avg = P_curr[n1:, :].mean(axis=0) if n2>0 else np.array([0.0,0.0,1.0])
            phi1 = spin_coherent_state_from_P(S_list[0], P1_avg, dtype=dtype)
            phi2 = spin_coherent_state_from_P(S_list[1], P2_avg, dtype=dtype)
            psi_dicke = np.kron(phi1, phi2).astype(dtype)

        # Evolve Dicke one step forward
        if evolver == "trotter":
            # U ≈ (∏_a e^{-i dt ω_a B·J_a}) e^{-i dt μ Σ J·J}
            for Ga in G_loc:
                psi_dicke = apply_exp_generator(Ga, dt, psi_dicke, dtype=dtype)
            psi_dicke = apply_exp_generator(G_int, dt, psi_dicke, dtype=dtype)
        elif evolver == "strang":
            # U ≈ [e^{-i dt/2 G_loc} e^{-i dt G_int} e^{-i dt/2 G_loc}] with G_loc = Σ_a ω_a B·J_a (bins commute)
            for Ga in G_loc:
                psi_dicke = apply_exp_generator(Ga, 0.5*dt, psi_dicke, dtype=dtype)
            psi_dicke = apply_exp_generator(G_int, dt, psi_dicke, dtype=dtype)
            for Ga in G_loc:
                psi_dicke = apply_exp_generator(Ga, 0.5*dt, psi_dicke, dtype=dtype)
        elif evolver == "exact":
            psi_dicke = apply_exact_step(H_full, dt, psi_dicke, dtype=dtype)
        else:
            raise ValueError("Unknown evolver: choose from 'trotter', 'strang', 'exact'")

        # record Dicke Pee at t_{k+1}
        dicke_pe[k+1, :] = pe_from_full_state(psi_dicke, Jz_list, S_list)

        # (C) deviation
        dev_pe[k+1, :] = dicke_pe[k+1, :] - mft_pe[k+1, :]

        # advance MFT state
        P_curr = P_next

    return t_grid, mft_pe, dicke_pe, dev_pe


def plot_results(t, mft_pe, dicke_pe, dev_pe, energy1, energy2, theta_v, mu, savename, evolver, dicke_steps=1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.0, 1.0]})

    # top: Pee
    ax1.plot(t, dicke_pe[:, 0], label=f"Bin 1 Dicke (E={energy1:.2f})")
    ax1.plot(t, mft_pe[:,   0], "--", label="Bin 1 MFT")
    ax1.plot(t, dicke_pe[:, 1], label=f"Bin 2 Dicke (E={energy2:.2f})")
    ax1.plot(t, mft_pe[:,   1], "--", label="Bin 2 MFT")
    ax1.set_xlabel("time / baseline")
    ax1.set_ylabel(r"$P_{ee}$")
    ax1.legend()
    ax1.grid(alpha=0.3)
    info_text = f"θ={theta_v:.3f}\nμ={mu:.3f}\nstepper={evolver}"
    if dicke_steps > 1:
        info_text += f"\nDicke steps={dicke_steps}"
    ax1.text(0.01, 0.99, info_text,
             transform=ax1.transAxes, ha="left", va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # bottom: deviations
    ax2.plot(t, dev_pe[:, 0], label="ΔP_e Bin 1")
    ax2.plot(t, dev_pe[:, 1], label="ΔP_e Bin 2")
    ax2.set_xlabel("time / baseline")
    ax2.set_ylabel(r"$\Delta P_{ee}$ (Dicke - MFT)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # Generate timestamp including minutes (same format as emu_2bin_opt_lop.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_png = f"localcheck_{timestamp}_mft_vs_dicke.png"
    plt.savefig(out_png, dpi=150)
    print(f"[+] Figure saved -> {out_png}")
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Local single-step validation of MFT vs Dicke (no entanglement propagation)")
    ap.add_argument("--e1", type=int, default=1, help="number of electron neutrinos in bin 1")
    ap.add_argument("--m1", type=int, default=0, help="number of muon neutrinos in bin 1")
    ap.add_argument("--e2", type=int, default=0, help="number of electron neutrinos in bin 2")
    ap.add_argument("--m2", type=int, default=1, help="number of muon neutrinos in bin 2")

    ap.add_argument("--energy1", type=float, default=1.0, help="energy of bin 1")
    ap.add_argument("--energy2", type=float, default=1.2, help="energy of bin 2")

    ap.add_argument("--theta", type=float, default=np.pi/2 - 0.2, help="vacuum mixing angle")
    ap.add_argument("--dmsq",  type=float, default=1.0, help="Δm^2 (sets ω = Δm^2/(2E))")
    ap.add_argument("--mu",    type=float, default=5.0, help="interaction strength μ")

    ap.add_argument("--baseline", type=float, default=10.0, help="total evolution time (or baseline)")
    ap.add_argument("--steps",    type=int,   default=100, help="number of steps")

    ap.add_argument("--scheme", choices=["trotter","strang","exact"], default="trotter",
                    help="quantum single-step evolver")
    ap.add_argument("--zero-self", action="store_true", help="set J_ii=0 in MFT coupling")
    ap.add_argument("--dickesteps", type=int, default=1,
                    help="number of Dicke evolution steps per MFT step (default 1)")

    ap.add_argument("--savename", type=str, default="localcheck", help="basename for saved figure")

    args = ap.parse_args()

    t, mft_pe, dicke_pe, dev_pe = run_localcheck(
        e1=args.e1, m1=args.m1, e2=args.e2, m2=args.m2,
        energy1=args.energy1, energy2=args.energy2,
        theta_v=args.theta, dmsq=args.dmsq, mu=args.mu,
        baseline=args.baseline, steps=args.steps,
        evolver=args.scheme, zero_self=args.zero_self,
        dicke_steps=args.dickesteps
    )

    out_png = plot_results(t, mft_pe, dicke_pe, dev_pe,
                           args.energy1, args.energy2, args.theta, args.mu,
                           args.savename, args.scheme, args.dickesteps)
    print("[+] Done.")

if __name__ == "__main__":
    main()
