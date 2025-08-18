import mft
import dicke_collective as dc

import numpy as np

from matplotlib import pyplot as plt

# -----
# Defining physical parameters
# -----
import argparse

parser = argparse.ArgumentParser(description="Physical parameters.")
parser.add_argument('--e1', type=int, default=1, help="number of electron neutrinos in bin 1")
parser.add_argument('--e2', type=int, default=1, help="number of electron neutrinos in bin 2")
parser.add_argument('--m1', type=int, default=1, help="number of muon neutrinos in bin 1")
parser.add_argument('--m2', type=int, default=1, help="number of muon neutrinos in bin 2")
parser.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1")
parser.add_argument('--energy2', type=float, default=1.2, help="energy of bin 2")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--l', type=float, default=10.0, help="baseline")
parser.add_argument('--s', type=int, default=100, help="number of steps")
args = parser.parse_args()

n1 = args.e1 + args.m1
n2 = args.e2 + args.m2

print("Bin 1 has {n1} neutrinos, with {e1} electrons and {m1} muons.".format(n1=n1, e1=args.e1, m1=args.m1))
print("Bin 2 has {n2} neutrinos, with {e2} electrons and {m2} muons.".format(n2=n2, e2=args.e2, m2=args.m2))
print("Bin 1 has energy {energy1}, bin 2 has energy {energy2}.".format(energy1=args.energy1, energy2=args.energy2))

print("Simulating with uniform interaction strength of {j}.".format(j=args.j))
print("Simulating with baseline {l} across {s} steps.".format(l=args.l, s=args.s))

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
MU_COLOR = "red"

# # -----
# # Evaluate mean field solution
# # -----

l_table = np.linspace(0, args.l, args.s)

omega1, omega2 = dmsq / (2*args.energy1), dmsq / (2*args.energy2)

# j = args.j * np.ones((n, n)) / n

# mft_sol = mft.P_osc_RS(l_table, theta, omega, 0, j, initial_flavors=["e"]*args.e + ["mu"]*args.m)

# mft_sol = np.reshape(mft_sol.y, (n,3,len(l_table)))

# mft_p_e = 0.5*(1+mft_sol[:,2,:])

# mft_p_e = np.mean(mft_p_e, axis=0)

# # -----
# # Evaluate Dicke solution
# # -----

def demo_two_bins(n1a=8, n2a=0, n1b=0, n2b=8, omega1=1.0, omega2=1.2,
                  theta_v=0.15, mu=0.5, t_max=40.0, T=400, make_plot=True):
    # Build initial product Dicke state for two bins
    psi0, S_list = multi_bin_initial_state([n1a+n2a, n1b+n2b], [0,0])  # we set m via signs below
    # But we want explicit n1 up / n2 down per bin; construct m_a = (n1 - n2)/2
    m_list = [ (n1a - n2a)/2.0, (n1b - n2b)/2.0 ]
    psi0 = product_dicke_state(S_list, m_list)

    H, (Jx_list, Jy_list, Jz_list), S_list, dims = build_multi_bin_hamiltonian(
        N_list=[int(2*S) for S in S_list],
        omega_list=[omega1, omega2],
        theta_v=theta_v,
        mu=mu
    )
    t_grid = np.linspace(0.0, t_max, T)
    states = evolve_times(H, psi0, t_grid)
    _, Pee_t = bin_observables(states, Jz_list, S_list)
    if make_plot:
        plt.figure(figsize=(7,4))
        plt.plot(t_grid, Pee_t[:,0], label=f'bin 1 (N={int(2*S_list[0])}, ω={omega1:.2f})')
        plt.plot(t_grid, Pee_t[:,1], label=f'bin 2 (N={int(2*S_list[1])}, ω={omega2:.2f})')
        plt.xlabel('time')
        plt.ylabel('Pee (bin-averaged)')
        plt.title('Two energy bins with equal μ (collective dynamics)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return t_grid, Pee_t

psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
m_list = [ (args.e1 - args.m1)/2.0, (args.e2 - args.m2)/2.0 ]
psi0 = dc.product_dicke_state(S_list, m_list)

H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=np.pi/2 - theta,
    mu=args.j
)

states = dc.evolve_times(H, psi0, l_table)
_, dc_p_e = dc.bin_observables(states, Jz_list, S_list)

# # -----
# # Plot results
# # -----

# # Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# # Main plot (top subplot)
ax1.plot(l_table, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})')
ax1.plot(l_table, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})')
ax1.legend()
plt.show()
# ax1.plot(l_table, mft_p_e, ls=':', lw=3, color=E_COLOR, label='MFT')
# ax1.plot(l_table, 1-mft_p_e, ls=':', lw=3, color=MU_COLOR, label='MFT')
# ax1.plot(l_table, dc_p_e[:,0], ls='-', lw=3, color=E_COLOR, label='Dicke')
# ax1.plot(l_table, 1-dc_p_e[:,0], ls='-', lw=3, color=MU_COLOR, label='Dicke')
# ax1.set_xlabel('baseline')
# ax1.set_ylabel('Pe')
# ax1.set_title('N_e = {}, N_mu = {}'.format(args.e, args.m))
# ax1.legend()
# ax1.set_ylim(0, 1)
# ax1.grid(True, alpha=0.3)

# # Residuals plot (bottom subplot)
# residuals_e = mft_p_e - dc_p_e[:,0]
# residuals_mu = (1-mft_p_e) - (1-dc_p_e[:,0])

# ax2.plot(l_table, residuals_e, ls='-', lw=2, color=E_COLOR, label='Residuals (e)')
# ax2.plot(l_table, residuals_mu, ls='-', lw=2, color=MU_COLOR, label='Residuals (μ)')
# ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
# ax2.set_xlabel('baseline')
# ax2.set_ylabel('Residuals (MFT - Dicke)')
# ax2.legend()
# ax2.grid(True, alpha=0.3)

# # Add some statistics to show how close the solutions are
# max_residual = max(np.max(np.abs(residuals_e)), np.max(np.abs(residuals_mu)))
# mean_residual = np.mean([np.mean(np.abs(residuals_e)), np.mean(np.abs(residuals_mu))])
# ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}', 
#          transform=ax2.transAxes, verticalalignment='top', 
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# plt.tight_layout()
# plt.savefig('emu_{}_e_{}_j_{}_l_{}.png'.format(args.e, args.m, args.j, args.l))
# plt.show()