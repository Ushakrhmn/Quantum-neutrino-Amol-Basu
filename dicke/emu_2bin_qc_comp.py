import mft
import dicke_collective as dc

import numpy as np

from matplotlib import pyplot as plt
import qiskit as qk

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
b = np.array([np.sin(2*theta), 0, -np.cos(2*theta)]) # the structure of the vacuum Hamiltonian in the Pauli basis
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

# -----
# Evaluate direct quantum simulation solution
# -----

from scipy.linalg import expm

def U_nunu(theta):
    """
    returns the interaction term for nunubar interactions
    """

    block = np.asarray([
        [2, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 2]
    ]) / 2

    return expm(-1j * theta * block)

l_table = np.linspace(0, args.l, args.s)

dt_table = np.diff(l_table)

omega1, omega2 = dmsq / (2*args.energy1), dmsq / (2*args.energy2)

n = n1 + n2

qc = qk.QuantumCircuit(n)

# uniform strength across bins
j = args.j * np.ones((n, n)) / n

# np.fill_diagonal(j, 0) # no self-interaction

qc_omega = np.array([omega1] * n1 + [omega2] * n2)

qc_initial_flavours = ["e"] * args.e1 + ["mu"] * args.m1 + ["e"] * args.e2 + ["mu"] * args.m2

for iq in range(n): # we use Duan and Fuller's convention here where \nu_e = |0> and \bar\nu_e = |0>
    if qc_initial_flavours[iq] == 'mu':
        qc.x(iq)

from tqdm import tqdm
from mft import sigma_1, sigma_2, sigma_3

for i in tqdm(range(len(dt_table))):
    dt = dt_table[i]

    # artificially remove entanglement by resetting each qubit to a non-entangled state
    qc_no_save = qc.remove_final_measurements(inplace=False)
    qc_no_save.data = [inst for inst in qc.data if inst.operation.name != 'save_density_matrix']
    sv = qk.quantum_info.Statevector(qc_no_save)
    rho = [ np.array(qk.quantum_info.partial_trace(sv, [j for j in range(n) if j != k])) for k in range(n) ]
    p   = [ np.real(np.array([ np.trace(sigma_1@rho[k]), np.trace(sigma_2@rho[k]), np.trace(sigma_3@rho[k]) ])) for k in range(n) ]
    
    # reset qubits to a non-entangled state to emulate the mean-field picture
    qc.reset(range(n))
    for j in range(n):
        qc.ry(np.arccos(p[j][2]), j)
        qc.rz(np.arctan2(p[j][1], p[j][0]), j)

    qc.save_density_matrix(label=str(i+1))

    for j in range(n):
        qc.rx(-dt*qc_omega[j]*b[0], j)
        qc.rz(-dt*qc_omega[j]*b[2], j)

    for iq1 in range(n):
        for iq2 in range(iq1+1, n):
            qc.unitary(U_nunu(-dt*j[iq1, iq2] / n), [iq1, iq2]) # rescale by number of neutrino

# save final state
qc.save_density_matrix(label=str(i+2))

import qiskit_aer as aer

# run simulation    
sim = aer.AerSimulator()
result = sim.run(qc).result()

# extract and plot results
rho_data = [result.data()[k] for k in result.data().keys()]  # assume keys are times

def reduce_qubit(j):
    tmp = { k : np.array(qk.quantum_info.partial_trace(result.data()[k], [q for q in range(n) if q != j])) for k in result.data().keys() }
    print("Finished calculation for qubit", j)
    return tmp

print("Obtaining partial traces ...")

# rho_reduced = Parallel(n_jobs=-1)(delayed(reduce_qubit)(j) for j in tqdm(range(n_qubits)))
rho_reduced = [reduce_qubit(j) for j in tqdm(range(n))]

print("Calculating probabilities ...")

# rho_reduced = [ { k : np.array(qk.quantum_info.partial_trace(result.data()[k], [q for q in range(n_qubits) if q != j])) for k in result.data().keys() } for j in tqdm(range(n_qubits)) ]
qc_p = np.array([ [[float(k), rho_reduced[j][k][0,0]] for k in rho_reduced[j].keys() ] for j in range(n) ])
qc_p_e = np.abs(np.take_along_axis(qc_p, qc_p[:,:,0].argsort(axis=1)[:,:,None], axis=1))

qc_p_e = np.mean(qc_p_e, axis=0)

print("Quantum solution evaluated.")

# -----
# Evaluate Dicke solution
# -----

psi0, S_list = dc.multi_bin_initial_state([args.e1, args.e2], [args.m1, args.m2])
m_list = [ (args.e1 - args.m1)/2.0, (args.e2 - args.m2)/2.0 ]
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
ax1.plot(l_table, dc_p_e[:,0], label=f'Bin 1 (N={int(2*S_list[0])}, E={args.energy1:.2f})', color="blue")
ax1.plot(l_table, dc_p_e[:,1], label=f'Bin 2 (N={int(2*S_list[1])}, E={args.energy2:.2f})', color="red")
ax1.plot(l_table, qc_p_e[0], label=f'Bin 1 (QC)', color="blue", ls="--")
ax1.plot(l_table, qc_p_e[1], label=f'Bin 2 (QC)', color="red", ls="--")
ax1.legend()
ax1.set_xlabel('baseline')
ax1.set_ylabel('Pe')
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# # Residuals plot (bottom subplot)
residual1 = qc_p_e[0] - dc_p_e[:,0]
ax2.plot(l_table, residual1, label="Residual (Bin 1)", color="blue")
residual2 = qc_p_e[1] - dc_p_e[:,1]
ax2.plot(l_table, residual2, label="Residual (Bin 2)", color="red")
ax2.legend()
ax2.set_xlabel('baseline')
ax2.set_ylabel('Residuals (QC - Dicke)')
ax2.grid(True, alpha=0.3)

# Add some statistics to show how close the solutions are
max_residual = max(np.max(np.abs(residual1)), np.max(np.abs(residual2)))
mean_residual = np.mean([np.mean(np.abs(residual1)), np.mean(np.abs(residual2))])

ax2.text(0.02, 0.98, f'Max residual: {max_residual:.2e}\nMean residual: {mean_residual:.2e}', 
         transform=ax2.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('2bin_emu_qc.png')
plt.show()