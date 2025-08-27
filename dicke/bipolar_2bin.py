import mft
import dicke_collective as dc

import numpy as np

from matplotlib import pyplot as plt

# -----
# Defining physical parameters
# -----
import argparse

parser = argparse.ArgumentParser(description="Physical parameters.")
parser.add_argument('--e', type=int, default=1, help="number of electron neutrinos in bin 1")
parser.add_argument('--b', type=int, default=1, help="number of electron antineutrinos in bin 2")
parser.add_argument('--energy', type=float, default=1.0, help="energy of all neutrinos and antineutrinos")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--l', type=float, default=10.0, help="baseline")
parser.add_argument('--s', type=int, default=100, help="number of steps")
args = parser.parse_args()

n1 = args.e
n2 = args.b
n = n1 + n2

print(f"Bin 1 has {n1} electron neutrinos, and bin 2 has {n2} electron antineutrinos.")
print(f"Bin 1 has energy {args.energy}, bin 2 has energy {args.energy}.")

print(f"Simulating with uniform interaction strength of {args.j}.")
print(f"Simulating with baseline {args.l} across {args.s} steps.")

theta = np.pi/2 - 0.2
dmsq = 1.0

# -----
# Plot options
# -----
import matplotlib

matplotlib.rcParams['font.family']    = 'serif'
matplotlib.rcParams['font.size']      = '16'
matplotlib.rcParams['figure.figsize'] = 16, 8

E_COLOR = "blue"
B_COLOR = "red"

# # -----
# # Evaluate mean field solution
# # -----

l_table = np.linspace(0, args.l, args.s)

# neutrinos have + omega, antineutrinos have - omega
omega1, omega2 = dmsq / (2 * args.energy), - dmsq / (2 * args.energy)

# uniform strength across bins
j = args.j * np.ones((n, n)) / n / 2

# np.fill_diagonal(j, 0) # no self-interaction

mft_omega = np.array([omega1] * n1 + [omega2] * n2)

mft_intial_flavours = ["e"] * n1 + ["ebar"] * n2

mft_sol = mft.P_osc_RS(l_table, theta, mft_omega, 0, j, initial_flavors=mft_intial_flavours)

mft_sol = np.reshape(mft_sol.y, (n,3,len(l_table)))

pz = mft_sol[:, 2, :]

# for the first n1 elements (neutrinos): Pe = (1 + Pz)/2
# for the last n2 elements (antineutrinos): Pebar = (1 - Pz)/2
p_e_nu   = 0.5*(1 + pz[:n1, :])
p_ebar   = 0.5*(1 + pz[n1:, :])

mft_p_e = [np.mean(p_e_nu, axis=0), np.mean(p_ebar, axis=0)]

print("Mean field solution evaluated.")

# -----
# Evaluate Dicke solution
# -----

psi0, S_list = dc.multi_bin_initial_state([n1, n2], [0, 0])
m_list = [ n1 / 2.0, n2 / 2.0 ]
psi0 = dc.product_dicke_state(S_list, m_list)

H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=theta,
    mu=args.j / n
)

states = dc.evolve_times(H, psi0, l_table)
_, dc_p_e = dc.bin_observables(states, Jz_list, S_list)

print("Dicke solution evaluated.")

# -----
# Plot results
# -----

# # Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# # Main plot (top subplot)
ax1.plot(l_table, dc_p_e[:, 0], label=f"Bin 1 nu_e", color=E_COLOR)
ax1.plot(l_table, 1 - dc_p_e[:, 1], label=f"Bin 2 nu_ebar", color=B_COLOR)
ax1.plot(l_table, mft_p_e[0], label=f'Bin 1 (MFT)', color=E_COLOR, ls="--")
ax1.plot(l_table, mft_p_e[1], label=f'Bin 2 (MFT)', color=B_COLOR, ls="--")
ax1.legend()
ax1.set_xlabel('baseline')
ax1.set_ylabel('1/2(1 + <Pz>/n)')
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# # Residuals plot (bottom subplot)
residual1 = mft_p_e[0] - dc_p_e[:,0]
ax2.plot(l_table, residual1, label="Residual (Bin 1)", color=E_COLOR)
residual2 = mft_p_e[1] - (1 - dc_p_e[:,1])
ax2.plot(l_table, residual2, label="Residual (Bin 2)", color=B_COLOR)
ax2.legend()
ax2.set_xlabel('baseline')
ax2.set_ylabel('Residuals (MFT - Dicke)')
ax2.grid(True, alpha=0.3)

# Add some statistics to show how close the solutions are
max_residual = max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
mean_residual = np.mean([np.mean(np.abs(residual1)), np.mean(np.abs(residual2))])

ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}', 
         transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('2bin_bipolar_.png')
plt.show()