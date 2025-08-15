import mft
import dicke_collective as dc

import numpy as np

from matplotlib import pyplot as plt

# -----
# Defining physical parameters
# -----
import argparse

parser = argparse.ArgumentParser(description="Physical parameters.")
parser.add_argument('--e', type=int, default=1, help="number of electron neutrinos")
parser.add_argument('--m', type=int, default=1, help="number of muon neutrinos")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--l', type=float, default=10.0, help="baseline")
parser.add_argument('--s', type=int, default=100, help="number of steps")
args = parser.parse_args()

theta = np.pi/2 - 0.2
dmsq = 1.0

n = args.e + args.m

# -----
# Plot options
# -----
import matplotlib

matplotlib.rcParams['font.family']    = 'serif'
matplotlib.rcParams['font.size']      = '16'
matplotlib.rcParams['figure.figsize'] = 16, 8

E_COLOR = "blue"
MU_COLOR = "red"

# -----
# Evaluate mean field solution
# -----

l_table = np.linspace(0, args.l, args.s)

omega = dmsq / (2*0.1) * np.array([1.] * n) # all neutrinos

j = args.j * np.ones((n, n)) / n

mft_sol = mft.P_osc_RS(l_table, theta, omega, 0, j, initial_flavors=["e"]*args.e + ["mu"]*args.m)

mft_sol = np.reshape(mft_sol.y, (n,3,len(l_table)))

mft_p_e = 0.5*(1+mft_sol[:,2,:])

mft_p_e = np.mean(mft_p_e, axis=0)

# -----
# Evaluate Dicke solution
# -----

psi0, S = dc.single_bin_initial_state(args.e, args.m)
H, (Jx, Jy, Jz), S_list, dims = dc.build_single_bin_hamiltonian(N=int(2*S),
                                                                 omega=dmsq / (2*0.1),
                                                                 theta_v=np.pi/2 - theta,
                                                                 mu=args.j)

states = dc.evolve_times(H, psi0, l_table)
_, dc_p_e = dc.bin_observables(states, (Jx, Jy, Jz), S_list)

# -----
# Plot results
# -----

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# Main plot (top subplot)
ax1.plot(l_table, mft_p_e, ls=':', lw=3, color=E_COLOR, label='MFT')
ax1.plot(l_table, 1-mft_p_e, ls=':', lw=3, color=MU_COLOR, label='MFT')
ax1.plot(l_table, dc_p_e[:,0], ls='-', lw=3, color=E_COLOR, label='Dicke')
ax1.plot(l_table, 1-dc_p_e[:,0], ls='-', lw=3, color=MU_COLOR, label='Dicke')
ax1.set_xlabel('baseline')
ax1.set_ylabel('Pe')
ax1.set_title('N_e = {}, N_mu = {}'.format(args.e, args.m))
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Residuals plot (bottom subplot)
residuals_e = mft_p_e - dc_p_e[:,0]
residuals_mu = (1-mft_p_e) - (1-dc_p_e[:,0])

ax2.plot(l_table, residuals_e, ls='-', lw=2, color=E_COLOR, label='Residuals (e)')
ax2.plot(l_table, residuals_mu, ls='-', lw=2, color=MU_COLOR, label='Residuals (μ)')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax2.set_xlabel('baseline')
ax2.set_ylabel('Residuals (MFT - Dicke)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add some statistics to show how close the solutions are
max_residual = max(np.max(np.abs(residuals_e)), np.max(np.abs(residuals_mu)))
mean_residual = np.mean([np.mean(np.abs(residuals_e)), np.mean(np.abs(residuals_mu))])
ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}', 
         transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('emu_{}_e_{}_j_{}_l_{}.png'.format(args.e, args.m, args.j, args.l))
plt.show()