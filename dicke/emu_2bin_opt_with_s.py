import mft
import dicke_collective_sparse_opt as dc  # optimized streaming API
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt

# -----
# Defining physical parameters
# -----
import argparse

parser = argparse.ArgumentParser(description="Physical parameters.")
parser.add_argument('--chunk', type=int, default=64, help='expm_multiply block size for streaming evolution')
parser.add_argument('--normalize', action='store_true', help='L2-normalize |psi| at each step (for numerical hygiene)')
parser.add_argument('--e1', type=int, default=1, help="number of electron neutrinos in bin 1")
parser.add_argument('--e2', type=int, default=0, help="number of electron neutrinos in bin 2")
parser.add_argument('--m1', type=int, default=0, help="number of muon neutrinos in bin 1")
parser.add_argument('--m2', type=int, default=1, help="number of muon neutrinos in bin 2")
parser.add_argument('--energy1', type=float, default=1.0, help="energy of bin 1")
parser.add_argument('--energy2', type=float, default=1.2, help="energy of bin 2")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--l', type=float, default=10.0, help="baseline")
parser.add_argument('--s', type=int, default=100, help="number of steps")
parser.add_argument('--savename', type=str, default='emu_2bin', help="name of the saved figure")
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

n = n1 + n2

# uniform strength across bins
j = args.j * np.ones((n, n))

# np.fill_diagonal(j, 0) # no self-interaction

mft_omega = np.array([omega1] * n1 + [omega2] * n2)

mft_intial_flavours = ["e"] * args.e1 + ["mu"] * args.m1 + ["e"] * args.e2 + ["mu"] * args.m2

mft_sol = mft.P_osc_RS(l_table, theta, mft_omega, 0, j, initial_flavors=mft_intial_flavours)

mft_sol = np.reshape(mft_sol.y, (n,3,len(l_table)))

# average for each bin
mft_p_e = 0.5*(1+mft_sol[:,2,:])

mft_p_e = [np.mean(mft_p_e[:n1, :], axis=0), np.mean(mft_p_e[n1:, :], axis=0)]

print("Mean field solution evaluated.")

# -----
# Evaluate Dicke solution
# -----

print("Evaluating Dicke solution ...")

print("Building Hamiltonian ...")

psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
m_list = [ (args.e1 - args.m1)/2.0, (args.e2 - args.m2)/2.0 ]
psi0 = dc.product_dicke_state(S_list, m_list)

H, (Jx_list, Jy_list, Jz_list), S_list, dims = dc.build_multi_bin_hamiltonian(
    N_list=[int(2*S) for S in S_list],
    omega_list=[omega1, omega2],
    theta_v=theta,
    mu=args.j
)

print("Evolving (streaming) ...")

print("Streaming evolution + on-the-fly observables ...")
t_stream, dc_p_e = dc.compute_pe_stream(H, psi0, l_table, Jz_list, S_list, chunk=args.chunk, normalize=args.normalize)
print("Calculated probabilities via streaming.")

print("Dicke solution evaluated.")

# -----
# Plot results
# -----

import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
plt.plot(t_stream, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
plt.plot(t_stream, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
plt.xlabel('baseline')
plt.ylabel('Pe')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(args.savename + '.png')
plt.show()


# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# # Main plot (top subplot)
ax1.plot(t_stream, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
ax1.plot(t_stream, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
ax1.plot(l_table, mft_p_e[0], label=f'Bin 1 (MFT)', color="blue", ls="--")
ax1.plot(l_table, mft_p_e[1], label=f'Bin 2 (MFT)', color="red", ls="--")
ax1.legend()
ax1.set_xlabel('baseline')
ax1.set_ylabel('Pe')
ax1.legend()
# ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

ax1.text(0.02, 0.98, f'theta = {theta:.2f}\ndmsq = {dmsq:.2f}\nJ = {args.j:.2f}', 
        transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# # Residuals plot (bottom subplot)
residual1 = mft_p_e[0] - dc_p_e[:,0]
ax2.plot(t_stream, residual1, label="Residual (Bin 1)", color="blue")
residual2 = mft_p_e[1] - dc_p_e[:,1]
ax2.plot(t_stream, residual2, label="Residual (Bin 2)", color="red")
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

# Generate timestamp including minutes
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f'emu_{timestamp}_n{n}.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.show()


# =====================================================================
# Add-on: Dicke single-particle entanglement entropy (per bin)
# Definition matches Patwardhan et al. (2021): S = -∑ λ log λ with
# λ± = (1 ± r)/2, r = 2 |<J>| / N_a ; natural logs (nats).
# This uses the Dicke wavefunction |psi(t)> evolved in the symmetric
# subspace and computes <Jx>,<Jy>,<Jz> for each bin along the same
# time grid used for the main plot. Saved as a separate PNG.
# =====================================================================

def _binary_entropy_nats(p, eps=1e-12):
    p = np.clip(p, 0.0, 1.0)
    q = 1.0 - p
    vals = 0.0
    if p > eps:
        vals -= p * np.log(p)
    if q > eps:
        vals -= q * np.log(q)
    return float(vals)

def _entropy_from_J_expectations(Jx, Jy, Jz, N):
    # r = 2 |<J>| / N  ; clip for numerical safety
    r = float(2.0 * np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz) / max(N, 1))
    r = float(np.clip(r, 0.0, 1.0))
    lam_p = 0.5*(1.0 + r)
    # H2 in nats
    return _binary_entropy_nats(lam_p)

def _compute_entropy_stream(H, psi0, t_grid, Jx_list, Jy_list, Jz_list, S_list, *, chunk=64, normalize=False):
    # Stream the Dicke state and return t_grid and S_single(t) for each bin (list of arrays).
    stream = dc.evolve_times_stream(H, psi0, t_grid, chunk=chunk, normalize=normalize)
    K = len(S_list)
    N_list = [int(2*S) for S in S_list]
    t_acc = []
    S_acc = [[] for _ in range(K)]
    for (t, psi) in stream:
        t_acc.append(t)
        psi_col = psi.reshape(-1, 1)
        for a in range(K):
            # Use sparse dot if available
            Jx = Jx_list[a]; Jy = Jy_list[a]; Jz = Jz_list[a]
            if hasattr(Jx, 'dot'):
                jx = (np.vdot(psi, (Jx.dot(psi_col)).ravel())).real
                jy = (np.vdot(psi, (Jy.dot(psi_col)).ravel())).real
                jz = (np.vdot(psi, (Jz.dot(psi_col)).ravel())).real
            else:
                jx = (np.vdot(psi, Jx @ psi)).real
                jy = (np.vdot(psi, Jy @ psi)).real
                jz = (np.vdot(psi, Jz @ psi)).real
            S_acc[a].append(_entropy_from_J_expectations(jx, jy, jz, N_list[a]))
    return np.asarray(t_acc, float), [np.asarray(S_acc[a], float) for a in range(K)]

try:
    # Time grid: try to reuse the existing l_table if present; otherwise rebuild from args
    try:
        t_grid_for_entropy = l_table  # should be a 1D array used on the main P_ee plot
    except NameError:
        # Fallback: if not present, try reconstructing a linear grid
        steps = getattr(args, "steps", 256)
        L = getattr(args, "l", 10.0)
        t_grid_for_entropy = np.linspace(0.0, L, int(steps))

    # Compute entanglement entropies for each bin (in nats)
    # Requires H, psi0, Jx_list, Jy_list, Jz_list, S_list to exist (already built above)
    t_S, S_bins = _compute_entropy_stream(
        H, psi0, t_grid_for_entropy,
        Jx_list, Jy_list, Jz_list, S_list,
        chunk=getattr(args, "chunk", 64),
        normalize=getattr(args, "normalize", False)
    )

    # Plot to a separate figure and save with matching timestamp-based name
    n_total = int(2*sum(S_list))
    fig_S, axS = plt.subplots(figsize=(10, 6))
    for a, S_curve in enumerate(S_bins, start=1):
        axS.plot(t_S, S_curve, label=f"Bin {a} (N={int(2*S_list[a-1])})")
    axS.set_xlabel("baseline")
    axS.set_ylabel("Single-particle entanglement entropy (nats)")
    axS.set_title("Dicke single-particle entanglement entropy per bin")
    axS.grid(True, alpha=0.3)
    axS.legend()

    # Save to a separate file next to the main figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    entropy_filename = f'emu_entropy_{timestamp}_n{n_total}.png'
    plt.tight_layout()
    fig_S.savefig(entropy_filename, dpi=150)
    print(f"Entropy plot saved to {entropy_filename}")
except Exception as _e:
    print(f"Skipped entropy computation due to: {_e}")
