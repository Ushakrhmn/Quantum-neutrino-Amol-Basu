#!/usr/bin/env python
# coding: utf-8

"""
Simulate bipolar oscillation with neutrino-neutrino interactions
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
import qiskit_aer as aer
from tqdm import tqdm
from scipy.linalg import expm

import sys

sys.path.append("../")
sys.path.append("../joachim/")
sys.path.append("../dicke/")

from mft import P_osc_RS, P_osc_RS_bipolar
from mft import sigma_0, sigma_1, sigma_2, sigma_3
from ops import u as prepare_dicke

# Parse command line arguments
parser = argparse.ArgumentParser(description="Set Ne and Nmu from command line")
parser.add_argument('--e', type=int, default=1, help="N e")
parser.add_argument('--m', type=int, default=1, help="N mu")
parser.add_argument('--alpha', type=float, default=0.0, help="Initial superposition for neutrinos in flavor 'a'")
parser.add_argument('--method', type=str, default="statevector")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--s', type=int, default=64, help="# of type steps")
parser.add_argument('--l', type=float, default=0.5)
args = parser.parse_args()

print(f"Running with {args.e} electron neutrinos")
if args.alpha != 0.0:
    print(f"Running with {args.m} mixed initial neutrinos")
else:
    print(f"Running with {args.m} muon neutrinos")
print(f"Interaction strength is set to {args.j}.")
print(f"Evolving over L={args.l} over {args.s} timesteps.")
print("Using method: ", args.method)

# Physical parameters
L = args.l
t_steps = args.s
# theta = np.pi/2 - 0.2
theta = np.pi/2 - 0.8
dmsq = 1.0
if args.alpha != 0.0:
    initial_state = ['e']*args.e + ['a']*args.m
else:
    initial_state = ['e']*args.e + ['mu']*args.m

n_qubits = len(initial_state)
Delta = dmsq / (2*0.1) * np.array([1.] * n_qubits)
b = np.array([np.sin(2*theta), 0, -np.cos(2*theta)])  # vacuum Hamiltonian structure in Pauli basis
J = args.j * np.ones((n_qubits, n_qubits))

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '16'
matplotlib.rcParams['figure.figsize'] = 16, 8

# Pauli matrices
sigma_0 = np.array([[1, 0],  [0,  1]])
sigma_1 = np.array([[0, 1],  [1,  0]])
sigma_2 = np.array([[0,-1j], [1j, 0]])
sigma_3 = np.array([[1, 0],  [0, -1]])

# Mean-field solution
L_table = np.linspace(0, L, t_steps)
sol = P_osc_RS(L_table, theta, Delta, 0., abs(J), initial_flavors=initial_state, alpha=args.alpha)
MFT_P_table = np.reshape(sol.y, (n_qubits,3,len(L_table)))

def U_nunu(theta):
    """
    Returns the interaction term for nunubar interactions
    """
    block = np.asarray([
        [2, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 2]
    ]) / 2

    return expm(-1j * theta * block)

# Set up the discretization
dt_table = np.diff(L_table)

qc = qk.QuantumCircuit(n_qubits)

prepare_dicke(n_qubits, args.m, qc) # n_qubits in total, with arg.m "excited" mu-neutrinos

for i in tqdm(range(len(dt_table))):
    dt = dt_table[i]

    qc.save_density_matrix(label=str(i))

    for j in range(n_qubits):
        qc.rx(-dt*Delta[j]*b[0], j)
        qc.rz(-dt*Delta[j]*b[2], j)

    for iq1 in range(n_qubits):
        for iq2 in range(iq1+1, n_qubits):
            qc.unitary(U_nunu(-dt*J[iq1, iq2]), [iq1, iq2])

# Save final state
qc.save_density_matrix(label=str(i+1))

# Run simulation    
sim = aer.AerSimulator(method=args.method, precision="double", zero_threshold=1e-14, validation_threshold=1e-12)
result = sim.run(qc).result()

# Extract and plot results
n_qubits = qc.num_qubits
rho_data = [result.data()[k] for k in result.data().keys()]

def reduce_qubit(j):
    tmp = {k: np.array(qk.quantum_info.partial_trace(result.data()[k], [q for q in range(n_qubits) if q != j])) for k in result.data().keys()}
    print("Finished calculation for qubit", j)
    return tmp

print("Obtaining partial traces ...")
rho_reduced = [reduce_qubit(j) for j in tqdm(range(n_qubits))]

print("Calculating probabilities ...")
pp = np.array([[[float(k), rho_reduced[j][k][0,0]] for k in rho_reduced[j].keys()] for j in range(n_qubits)])
pp = np.abs(np.take_along_axis(pp, pp[:,:,0].argsort(axis=1)[:,:,None], axis=1))

plt.clf()
plt.text(0.9, 0.9, f"$\\theta = {(theta / np.pi * 180):.2f}$", transform=plt.gca().transAxes, fontsize=16)
plt.text(0.9, 0.8, f"$J = {J[0, 0]}$", transform=plt.gca().transAxes, fontsize=16)

# expected total polarization - mft

plt.plot(L_table, 0.5*(1+MFT_P_table[0,2,:]), ls=':', lw=3, color='blue', label="electron")
plt.plot(L_table, 0.5*(1+MFT_P_table[-1,2,:]), ls=':', lw=3, color='orange', label="muon")

# expected polarization - quantum simulation

qsim_expect = np.mean(pp[:, :, 1], axis=0)

plt.title("Dicke: expected polarization MFT: probability")

plt.plot(L_table, qsim_expect, color='blue')

qsim_expect = np.mean(1 - pp[:, :, 1], axis=0)

plt.plot(L_table, qsim_expect, color='orange')

plt.legend()
if args.alpha != 0.0:
    print(f"nunu_e_{args.e}_a_{args.m}_j_{args.j}_alpha_{args.alpha}_{args.method}.png")
    plt.savefig(f"nunu_e_{args.e}_a_{args.m}_j_{args.j}_alpha_{args.alpha}_{args.method}.png")
else:
    print(f"nunu_e_{args.e}_m_{args.m}_j_{args.j}_{args.method}.png")
    plt.savefig(f"nunu_e_{args.e}_m_{args.m}_j_{args.j}_{args.method}.png")