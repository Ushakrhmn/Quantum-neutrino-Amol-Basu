import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import dicke_collective_sparse_opt as dc  # :contentReference[oaicite:5]{index=5}

Ne   = 8
Nbar = 8
theta_v = 0.0    # irrelevant since omega=0
omega   = 0.0
mu      = 0.1

alpha = np.pi/2  # background angle in σ·σ picture

t = np.linspace(0.0, 20.0, 400)

# Bin 1: ν_e, bin 2: ν̄_e
psi_ee, S_list = dc.multi_bin_initial_state([Ne, Nbar], [0, 0])

# σ·σ Hamiltonian (all bins treated as neutrinos)
H_iso, (Jx_iso, Jy_iso, Jz_iso), S_iso, _ = dc.build_multi_bin_hamiltonian(
    N_list=[Ne, Nbar],
    omega_list=[omega, -omega],
    theta_v=theta_v,
    mu=mu,
    is_antineutrino=None
)

# Physical Hamiltonian (bin 2 is antineutrino)
H_phys, (Jx_phys, Jy_phys, Jz_phys), S_phys, _ = dc.build_multi_bin_hamiltonian(
    N_list=[Ne, Nbar],
    omega_list=[omega, -omega],
    theta_v=theta_v,
    mu=mu,
    is_antineutrino=[False, True]
)

Jy_bg = Jy_iso[1]  # background bin

# Put background in |ν̄(α)> = cosα|ν̄_e> + sinα|ν̄_μ> in the σ·σ picture
U_alpha   = expm(-1j * 2 * alpha * Jy_bg.toarray())
psi_iso0  = U_alpha @ psi_ee

# Flip the background by π around y to move to the physical basis
U_flip    = expm(-1j * np.pi * Jy_bg.toarray())
psi_phys0 = U_flip @ psi_iso0

# Evolve
Y_iso  = dc.evolve_times(H_iso,  psi_iso0,  t)
Y_phys = dc.evolve_times(H_phys, psi_phys0, t)

_, Pee_iso  = dc.bin_observables(Y_iso,  Jz_iso,  S_iso)
_, Pee_phys = dc.bin_observables(Y_phys, Jz_phys, S_phys)

print("max |ΔP_ee| =", np.max(np.abs(Pee_iso[:,0] - Pee_phys[:,0])))

plt.plot(t, Pee_iso[:,0],  label="beam, σ·σ H")
plt.plot(t, Pee_phys[:,0], "--", label="beam, physical H + rotated initial state")
plt.xlabel("t"); plt.ylabel("P_ee (beam)"); plt.legend(); plt.tight_layout()

plt.savefig("rotation_check.png")
plt.close()

print("Done")