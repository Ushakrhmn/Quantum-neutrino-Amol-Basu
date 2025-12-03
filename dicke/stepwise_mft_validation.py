
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stepwise MFT validation against projected Dicke evolution
=========================================================

Idea (per colleague's suggestion):
----------------------------------
At each Trotter step Δℓ on the Dicke trajectory with *coherent-state projection*
(i.e. single-particle entropy ≈ 0 after each step), take the projected state
|ψ_proj(ℓ_i)⟩ and use its local Bloch directions to define the *mean-field*
polarization vectors. From these, perform **one** MFT step of size Δℓ to
estimate the state at ℓ_{i+1}. Compare P_e(ℓ_{i+1}) from Dicke (with projection)
to the one-step MFT prediction. Re-initialize from |ψ_proj(ℓ_{i+1})⟩ and repeat.

What this script produces
-------------------------
1) Top panel: P_e(ℓ) for each bin from Dicke (projected each step) and the
   corresponding stepwise-one-step MFT estimate.
2) Bottom panel: Per-step deviation ΔP_e = P_e^MFT(ℓ_i) - P_e^Dicke(ℓ_i).

Dependencies
------------
- dicke_collective_sparse_opt_proj.py : streaming Dicke evolution + projection
- mft.py (not directly called here, but kept for reference of conventions)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import dicke_collective_sparse_opt_proj as dc  # Dicke solver + coherent projection

# ----------------------------
# Helpers: MFT single RK4 step
# ----------------------------
def _mft_rhs(P, B, omega, J):
    """
    Vector-field for Raffelt–Sigl mean-field equations:
      dP_k/dℓ = (ω_k B + sum_j J_{k j} P_j) × P_k
    P : (N,3) array of polarization vectors
    B : (3,) vacuum field unit vector  (sin 2θ, 0, -cos 2θ)
    omega : (N,) array of vacuum frequencies
    J : (N,N) coupling matrix
    Returns dP/dℓ as (N,3).
    """
    # collective field per mode: H_k = ω_k B + Σ_j J_{k j} P_j
    H = omega[:, None] * B[None, :] + J @ P  # (N,3)
    # precession: dP_k = H_k × P_k  (row-wise cross product)
    return np.cross(H, P)

def _rk4_step(P, dt, B, omega, J):
    k1 = _mft_rhs(P, B, omega, J)
    k2 = _mft_rhs(P + 0.5*dt*k1, B, omega, J)
    k3 = _mft_rhs(P + 0.5*dt*k2, B, omega, J)
    k4 = _mft_rhs(P + dt*k3, B, omega, J)
    Pn = P + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # Numerical hygiene: keep |P_k| ≈ 1
    nrm = np.linalg.norm(Pn, axis=1, keepdims=True)
    nrm = np.where(nrm == 0.0, 1.0, nrm)
    return Pn / nrm

def _expectations(psi, Jx_list, Jy_list, Jz_list):
    """Compute ⟨Jx⟩,⟨Jy⟩,⟨Jz⟩ for each bin from a (dense) state psi."""
    if hasattr(psi, "toarray"):
        psi = psi.toarray().ravel()
    else:
        psi = np.asarray(psi).ravel()
    psi_col = psi.reshape(-1, 1)
    jx, jy, jz = [], [], []
    for Jx, Jy, Jz in zip(Jx_list, Jy_list, Jz_list):
        # Support sparse @ dense
        if hasattr(Jx, 'dot'):
            jx.append((np.vdot(psi, (Jx.dot(psi_col)).ravel())).real)
            jy.append((np.vdot(psi, (Jy.dot(psi_col)).ravel())).real)
            jz.append((np.vdot(psi, (Jz.dot(psi_col)).ravel())).real)
        else:
            jx.append((np.vdot(psi, Jx @ psi)).real)
            jy.append((np.vdot(psi, Jy @ psi)).real)
            jz.append((np.vdot(psi, Jz @ psi)).real)
    return np.array(jx), np.array(jy), np.array(jz)

def _pee_from_jz(jz, S_list):
    """Bin electron survival probability from Dicke expectations."""
    return 0.5 * (1.0 + jz / np.array(S_list))

def _pee_from_P(P, n1, n2):
    """Bin-averaged P_e from mean-field P vectors (|P_k|≈1)."""
    z1 = np.mean(P[:n1, 2]) if n1 > 0 else np.nan
    z2 = np.mean(P[n1:n1+n2, 2]) if n2 > 0 else np.nan
    return np.array([0.5*(1.0 + z1), 0.5*(1.0 + z2)])

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Stepwise MFT validation vs projected Dicke evolution")
parser.add_argument('--e1', type=int, default=1, help="number of ν_e in bin 1")
parser.add_argument('--m1', type=int, default=0, help="number of ν_μ in bin 1")
parser.add_argument('--e2', type=int, default=0, help="number of ν_e in bin 2")
parser.add_argument('--m2', type=int, default=1, help="number of ν_μ in bin 2")
parser.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1")
parser.add_argument('--energy2', type=float, default=1.2, help="energy of bin 2")
parser.add_argument('--j', type=float, default=5.0, help="uniform coupling strength (μ)")
parser.add_argument('--l', type=float, default=10.0, help="total baseline length")
parser.add_argument('--steps', type=int, default=200, help="number of Trotter steps (samples)")
parser.add_argument('--normalize', action='store_true', help="normalize |ψ| after each Dicke substep")
parser.add_argument('--savename', type=str, default='stepwise_mft_validation', help="base name for output figure")
args = parser.parse_args()

# --- derived
theta = np.pi/2 - 0.2
dmsq  = 1.0
omega1, omega2 = dmsq/(2*args.energy1), dmsq/(2*args.energy2)

n1 = args.e1 + args.m1
n2 = args.e2 + args.m2
n  = n1 + n2

# Vacuum "magnetic field" on flavor sphere
B = np.array([np.sin(2.0*theta), 0.0, -np.cos(2.0*theta)], float)

# Uniform all-to-all coupling matrix (same convention as emu_2bin_* script)
Jmat = (args.j / n) * np.ones((n, n), dtype=float)  # include diagonal as in the reference script

# Mode frequencies (duplicate by particle count within each bin)
omega = np.array([omega1]*n1 + [omega2]*n2, float)

# Time/baseline grid
t_grid = np.linspace(0.0, args.l, args.steps)
dt = np.diff(t_grid)  # (T-1,)

# ----------------------------
# Build Dicke Hamiltonian etc.
# ----------------------------
psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
m_list = [ (args.e1 - args.m1)/2.0, (args.e2 - args.m2)/2.0 ]
psi0 = dc.product_dicke_state(S_list, m_list)

H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=theta,
    mu=args.j / n
)

# ---------------------------------------------
# Stream Dicke with projection & do MFT 1-step
# ---------------------------------------------
stream = dc.evolve_times_stream_projected(H, psi0, t_grid, Jx_list, Jy_list, Jz_list, S_list, normalize=args.normalize)

times = []
pee_dicke = []   # shape (T, 2)
pee_mft    = []  # shape (T, 2), first entry mirrors Dicke at t0

# Pull first state
t_prev, psi_prev = next(stream)
times.append(t_prev)
jx, jy, jz = _expectations(psi_prev, Jx_list, Jy_list, Jz_list)
pee0 = _pee_from_jz(jz, S_list)
pee_dicke.append(pee0)
pee_mft.append(pee0)  # trivial at t0

# Build per-particle P from bin Bloch directions at t_prev
def _build_P_from_bin_bloch(jx, jy, jz, n1, n2, eps=1e-14):
    P = np.zeros((n1 + n2, 3), float)
    for a, cnt in enumerate((n1, n2)):
        j = np.array([jx[a], jy[a], jz[a]], float)
        norm = np.linalg.norm(j)
        if norm < eps:
            u = np.array([0.0, 0.0, 1.0])
        else:
            u = j / norm
        if a == 0 and cnt > 0:
            P[:n1, :] = u
        elif a == 1 and cnt > 0:
            P[n1:n1+n2, :] = u
    return P

P_prev = _build_P_from_bin_bloch(jx, jy, jz, n1, n2)

# Iterate remaining steps
for step_idx in range(1, len(t_grid)):
    # Pop next Dicke-projected state
    t_cur, psi_cur = next(stream)
    times.append(t_cur)

    # Dicke observables at t_cur
    jx, jy, jz = _expectations(psi_cur, Jx_list, Jy_list, Jz_list)
    pee_cur = _pee_from_jz(jz, S_list)
    pee_dicke.append(pee_cur)

    # One-step MFT from previous projected state
    dt_step = float(t_cur - t_prev)
    P_next = _rk4_step(P_prev, dt_step, B, omega, Jmat)
    pee_mft.append(_pee_from_P(P_next, n1, n2))

    # Re-initialize from current projected Dicke state for the next round
    P_prev = _build_P_from_bin_bloch(jx, jy, jz, n1, n2)
    t_prev = t_cur

times      = np.asarray(times, float)
pee_dicke  = np.asarray(pee_dicke, float)
pee_mft    = np.asarray(pee_mft, float)

# Deviations (MFT - Dicke); first point is identically 0 by construction
residuals = pee_mft - pee_dicke

# ----------------------------
# Plot
# ----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[2, 1])
# Top: P_e
ax1.plot(times, pee_dicke[:,0], label=f'Bin 1 Dicke (N={n1}, E={args.energy1:.2f})')
ax1.plot(times, pee_dicke[:,1], label=f'Bin 2 Dicke (N={n2}, E={args.energy2:.2f})')
ax1.plot(times, pee_mft[:,0],  '--', label='Bin 1 MFT (one-step)')
ax1.plot(times, pee_mft[:,1],  '--', label='Bin 2 MFT (one-step)')
ax1.set_xlabel('baseline ℓ')
ax1.set_ylabel(r'$P_e$')
ax1.grid(alpha=0.3)
ax1.legend(loc='best')
ax1.text(0.02, 0.98,
         f'θ = {theta:.3f}\nΔm² = {dmsq:.2f}\nμ = {args.j:.2f}\nsteps = {args.steps}',
         transform=ax1.transAxes, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Bottom: residuals
ax2.plot(times, residuals[:,0], label='ΔP_e (Bin 1)')
ax2.plot(times, residuals[:,1], label='ΔP_e (Bin 2)')
ax2.set_xlabel('baseline ℓ')
ax2.set_ylabel(r'Δ$P_e$ = MFT − Dicke')
ax2.grid(alpha=0.3)
ax2.legend(loc='best')

# Summary stats (skip first point which is trivially zero)
if len(times) > 1:
    abs_res = np.abs(residuals[1:,:])
    max_res = float(np.max(abs_res))
    mean_res = float(np.mean(abs_res))
else:
    max_res = mean_res = 0.0

ax2.text(0.02, 0.98,
         f'max |Δ| = {max_res:.2e}\nmean |Δ| = {mean_res:.2e}',
         transform=ax2.transAxes, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_png = f'{args.savename}_{timestamp}.png'
fig.savefig(out_png, dpi=150)
print(f'[Lilith] Figure saved to {out_png}')

