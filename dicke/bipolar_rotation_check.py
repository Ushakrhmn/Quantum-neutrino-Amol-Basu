#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import dicke_collective_sparse_opt as dc
from scipy.sparse.linalg import expm

# ----------
# Physical / bipolar-like parameters
# ----------

Ne = 8      # number of neutrinos in bin 1
Na = 8      # number of antineutrinos in bin 2

theta_v = 0.0001   # small vacuum mixing angle
omega    = 5.0   # |ω| sets the vacuum scale
mu       = 5.0   # self-interaction strength (μ >> ω → clear bipolar motion)

t = np.linspace(0.0, 40.0, 800)  # evolution time grid

# ----------
# 1) σ·σ picture with "same-sign" vacuum:
#    H_iso_same = H_vac(ν:+ω, ν̄:+ω) + μ * J·J
# ----------

H_iso, (Jx_iso, Jy_iso, Jz_iso), S_list_iso, dims_iso = dc.build_multi_bin_hamiltonian(
    N_list      = [Ne, Na],
    omega_list  = [ +omega, +omega ],   # ν and ν̄ both use +ω here
    theta_v     = theta_v,
    mu          = mu,
    is_antineutrino = None             # all pairs use Heisenberg J·J
)

# ----------
# 2) Physical ν–ν̄ Hamiltonian with "standard" ν:+ω, ν̄:-ω vacuum:
#    H_phys_std = H_vac(ν:+ω, ν̄:-ω) + μ * H_phys,int
# ----------

H_phys, (Jx_phys, Jy_phys, Jz_phys), S_list_phys, dims_phys = dc.build_multi_bin_hamiltonian(
    N_list      = [Ne, Na],
    omega_list  = [ +omega, -omega ],   # ν:+ω,  ν̄:-ω  (bipolar convention)
    theta_v     = theta_v,
    mu          = mu,
    is_antineutrino = [False, True]     # bin 1 = ν, bin 2 = ν̄ → physical ν–ν̄ term
)

# sanity check: Dicke structure must match
assert S_list_iso == S_list_phys
S_list = S_list_iso

# ----------
# 3) Bipolar initial state in σ·σ picture:
#    bin 1: pure ν_e, bin 2: pure ν̄_e
# ----------

psi0_iso, S_list0 = dc.multi_bin_initial_state(
    n1_list = [Ne, Na],   # number of e / ē
    n2_list = [0,  0 ]    # number of μ / μ̄
)
assert S_list0 == S_list

# ----------
# 4) Map this initial state to the physical picture via
#    U_flip = exp(-i π J_y^(bin 2))
# ----------

Jy_bg_phys = Jy_phys[1]                 # Jy operator for the antineutrino bin
U_flip     = expm(-1j * np.pi * Jy_bg_phys)

psi0_phys = U_flip @ psi0_iso           # initial state for H_phys

# ----------
# 5) Time evolution under both Hamiltonians
# ----------

Y_iso  = dc.evolve_times(H_iso,  psi0_iso,  t)
Y_phys = dc.evolve_times(H_phys, psi0_phys, t)

# ----------
# 6) Electron flavor survival probabilities for both bins
# ----------

_, Pee_iso  = dc.bin_observables(Y_iso,  Jz_iso,  S_list)
_, Pee_phys = dc.bin_observables(Y_phys, Jz_phys, S_list)

# ----------
# 7) Compare and plot
# ----------

diff_bin1 = np.max(np.abs(Pee_iso[:, 0] - Pee_phys[:, 0]))
diff_bin2 = np.max(np.abs(Pee_iso[:, 1] - Pee_phys[:, 1]))
print("max |ΔP_ee| (bin 1, ν)  =", diff_bin1)
print("max |ΔP_ee| (bin 2, ν̄) =", diff_bin2)

plt.figure(figsize=(10, 6))
plt.plot(t, Pee_iso[:, 0],       label="bin 1 (ν), σ·σ H (ν:+ω, ν̄:+ω)")
plt.plot(t, Pee_phys[:, 0], "--", label="bin 1 (ν), physical H (ν:+ω, ν̄:-ω) + U_flip")

plt.plot(t, Pee_iso[:, 1],       label="bin 2 (ν̄), σ·σ H (ν:+ω, ν̄:+ω)")
plt.plot(t, Pee_phys[:, 1], "--", label="bin 2 (ν̄), physical H (ν:+ω, ν̄:-ω) + U_flip")

plt.xlabel("t")
plt.ylabel("P_ee")
plt.legend()
plt.tight_layout()
plt.savefig("bipolar_rotation_check.png")
print("Figure saved to bipolar_rotation_check.png")