import numpy as np
import matplotlib.pyplot as plt

import dicke_collective_sparse_opt as dc
from scipy.sparse.linalg import expm

# --- parameters ---
Ne   = 8           # beam ν_e count
Nbar = 8           # background ν̄ count

theta_v = 0.01     # vacuum mixing angle
omega   = 1.0      # vacuum frequency |ω|
mu      = 0.1      # self-interaction strength

# background mixing angle in the σ·σ picture:
# |ν̄(α)> = cos α |ν̄_e> + sin α |ν̄_μ>
alpha = np.pi/2

t = np.linspace(0.0, 20.0, 400)

# -------------------------------------------------------
# 1) Isotropic σ·σ many-body Hamiltonian (standard ν / ν̄ vacuum signs)
#    H_iso = H_vac(ν:+ω, ν̄:-ω) + μ * J·J
# -------------------------------------------------------

H_iso, (Jx_iso, Jy_iso, Jz_iso), S_list_iso, dims_iso = dc.build_multi_bin_hamiltonian(
    N_list      = [Ne, Nbar],
    omega_list  = [ +omega, -omega ],   # ν, ν̄: standard +ω / -ω
    theta_v     = theta_v,
    mu          = mu,
    is_antineutrino = None              # <-- all pairs use J·J (σ·σ)
)

# -------------------------------------------------------
# 2) "Physical" Hamiltonian with ν–ν̄ anisotropy AND flipped ν̄ vacuum sign
#    H_phys = H_vac(ν:+ω, ν̄:+ω) + μ * H_phys,int
# -------------------------------------------------------

H_phys, (Jx_phys, Jy_phys, Jz_phys), S_list_phys, dims_phys = dc.build_multi_bin_hamiltonian(
    N_list      = [Ne, Nbar],
    omega_list  = [ +omega, +omega ],   # NOTE: both ν and ν̄ use +ω here
    theta_v     = theta_v,
    mu          = mu,
    is_antineutrino = [False, True]     # <-- ν–ν̄ pairs use physical anisotropic kernel
)

# sanity check: Dicke basis should be the same
assert S_list_iso == S_list_phys
S_list = S_list_iso

# -------------------------------------------------------
# 3) Initial state: |ν_e> beam + |ν̄_e> background in Dicke space
# -------------------------------------------------------

psi_e_e_bar, S_list = dc.multi_bin_initial_state(
    n1_list = [Ne, Nbar],   # # of e / ē
    n2_list = [0,  0   ]    # # of μ / μ̄
)

# background bin index is 1
Jy_bg_iso  = Jy_iso[1]
Jy_bg_phys = Jy_phys[1]    # same operator in both constructions (same basis)

# -------------------------------------------------------
# 4) Prepare background state with angle α in σ·σ picture
#    single-particle: R_y(2α)|ē> = cos α |ē> + sin α |μ̄>
# -------------------------------------------------------

U_alpha = expm(-1j * 2.0 * alpha * Jy_bg_iso)
psi0_iso = U_alpha @ psi_e_e_bar       # initial state for H_iso (σ·σ picture)

# -------------------------------------------------------
# 5) Map this initial state into the "physical" basis via U_flip
#    U_flip = exp(-i π J_y^(bg)) implements the antineutrino y-flip
# -------------------------------------------------------

U_flip = expm(-1j * np.pi * Jy_bg_phys)
psi0_phys = U_flip @ psi0_iso          # initial state for H_phys

# -------------------------------------------------------
# 6) Time evolution
# -------------------------------------------------------

Y_iso  = dc.evolve_times(H_iso,  psi0_iso,  t)
Y_phys = dc.evolve_times(H_phys, psi0_phys, t)

# -------------------------------------------------------
# 7) Electron survival probability of the beam bin (bin 0)
# -------------------------------------------------------

_, Pee_iso  = dc.bin_observables(Y_iso,  Jz_iso,  S_list)
_, Pee_phys = dc.bin_observables(Y_phys, Jz_phys, S_list)

# -------------------------------------------------------
# 8) Compare and plot
# -------------------------------------------------------

max_diff = np.max(np.abs(Pee_iso[:, 0] - Pee_phys[:, 0]))
print("max |ΔP_ee (beam)| =", max_diff)

plt.figure(figsize=(7, 4))
plt.plot(t, Pee_iso[:, 0],  label="beam, σ·σ H (ν:+ω, ν̄:-ω)")
plt.plot(t, Pee_phys[:, 0], "--", label="beam, physical H (ν:+ω, ν̄:+ω) + U_flip")
plt.xlabel("t")
plt.ylabel("P_ee (beam)")
plt.legend()
plt.tight_layout()
plt.savefig("rotation_check_with_vacuum.png")
print("Figure saved to rotation_check_with_vacuum.png")