#!/usr/bin/env python
# coding: utf-8

# In[11]:


from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from matplotlib import pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator

import sys

sys.path.append("../utils")

from CircuitConvert import qiskit_to_quimb

from tqdm import tqdm

import cotengra as ctg
import quimb.tensor as qtn

import gc
import tracemalloc
import json

# In[12]:


# configs = {
#     "max_repeats": 256,
#     "shots": 8192,
#     "slicing_reconf_opts":{"target_size": 1e9},
#     "max_time":'rate:1e9',
#     "methods":"labels"
# }
# with open("config/tensor_config.json", 'w') as f:
#     json.dump(configs, f)

# In[13]:


configs = json.load(open("config/tensor_config.json"))

# In[14]:


tracemalloc.start()

# In[15]:


# simulator=AerSimulator(device='GPU')

# In[ ]:


#### Defining PMNS matrix and parameters for the Hamiltonian ####

def PMNS_matrix_4x4(theta_12, theta_13, theta_23, delta_CP, cp_sign=1):
    c12, s12 = np.cos(theta_12), np.sin(theta_12)
    c13, s13 = np.cos(theta_13), np.sin(theta_13)
    c23, s23 = np.cos(theta_23), np.sin(theta_23)
    e_idelta = np.exp(-1j * delta_CP * cp_sign)
    e_ideltap = np.exp(1j * delta_CP * cp_sign)

    U = np.array([
        [c12 * c13, s12 * c13, s13 * e_idelta],
        [-s12 * c23 - c12 * s23 * s13 * e_ideltap, c12 * c23 - s12 * s23 * s13 * e_ideltap, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e_ideltap, -c12 * s23 - s12 * c23 * s13 * e_ideltap, c23 * c13]
    ], dtype=complex)

    U_4x4 = np.eye(4, dtype=complex)
    U_4x4[np.ix_([0,1,2],[0,1,2])] = U
    return U_4x4


theta_12 = np.radians(33.67)
theta_13 = np.radians(8.58)
theta_23 = np.radians(42.3)
delta_CP = np.radians(232)

U_PMNS_4x4 = PMNS_matrix_4x4(theta_12, theta_13, theta_23, delta_CP)
assert np.allclose(U_PMNS_4x4.conj().T @ U_PMNS_4x4, np.eye(4)), "PMNS matrix is not unitary"

dm21 = 7.41e-5
dm31 = 2.505e-3
E = 1.0
N = 32

delta_m21 = dm21
delta_m31 = dm31
delta_m32 = delta_m31 - delta_m21
#mu=dm31*N/(2*E) #* 1e-8
mu = dm31 * N / (2 * E)
print('mu: ', mu)
#omega = np.sqrt(delta_m21**2 + ((delta_m31 + delta_m32)**2) / 3) / (4 * E)
omega=dm21 / (2 * E)  
Omega= dm31 / (2 * E)
print ('omega and Omega: ', omega, Omega)
#B3 = delta_m21 / np.sqrt(delta_m21**2 + ((delta_m31 + delta_m32)**2) / 3)
#B8 = (delta_m31 + delta_m32) / (np.sqrt(3) * np.sqrt(delta_m21**2 + ((delta_m31 + delta_m32)**2) / 3))
B3=0.025483
B8=0.999567
B = np.zeros(8)
B[2] = B3
B[7] = B8

def theta_ij(i, j):
    return np.arccos(0.9) * np.abs(i-j) / (N-1)

def J_ij(i, j, E):
    return  mu/(N) * (1 - np.cos(theta_ij(i, j))) * 1
print('theta_ij(0,1): ', theta_ij(0,1))
print ('J_ij(0, 1, E): ', J_ij(0,1,E))

# In[17]:


def U1_omega_Omega(t):
    A = -1 * omega * t
    B_val = -1 * Omega * t
    gate = QuantumCircuit(2)
    gate.rz(A, 0)
    gate.rz(B_val, 1)
    return gate.to_gate(label='U1_omega_Omega')


# In[18]:


def nunu_interaction(qc, q, alpha):
    a, b, c, d = q
    qc.cx(a, c)
    qc.cx(b, d)
    qc.ry(np.pi/4, a)
    qc.ry(np.pi/4, b)
    qc.cx(c, a)
    qc.cx(d, b)
    qc.ry(-np.pi/4, a)
    qc.ry(-np.pi/4, b)
    qc.rz(-alpha, c)
    qc.rz(-alpha, d)
    qc.cx(a, c)
    qc.cx(b, d)
    qc.cx(a, b)
    qc.rz(-2 * alpha, b)
    qc.rz(alpha, c)
    qc.rz(alpha, d)
    qc.cx(a, b)
    qc.cx(b, c)
    qc.rz(alpha, c)
    qc.cx(a, c)
    qc.rz(-alpha, c)
    qc.cx(a, d)
    qc.cx(b, c)
    qc.rz(alpha, d)
    qc.cx(b, d)
    qc.rz(-alpha, d)
    qc.cx(a, d)
    qc.ry(np.pi/4, a)
    qc.ry(np.pi/4, b)
    qc.cx(d, b)
    qc.cx(c, a)
    qc.ry(-np.pi/4, a)
    qc.ry(-np.pi/4, b)
    qc.cx(b, d)
    qc.cx(a, c)

# In[19]:


def apply_nunu_interaction(qc, pair_indices, alpha_list):
    for i, j, alpha_ij in alpha_list:
        q = [2*i, 2*i+1, 2*j, 2*j+1]
        nunu_interaction(qc, q, alpha_ij)

# In[20]:


# Time loop setup for N neutrinos

pair_indices = [(i, j) for i in range(N) for j in range(i+1, N)]
dt = 5 / mu
T_max = 20 / mu
steps = int(T_max / dt) + 1
times = np.linspace(0, T_max, steps)

probabilities = []
shots = configs['shots']
for t in times:

    print("------------------------------")
    print("Time: ", t)
    print("Time step: ", dt)
    qc = QuantumCircuit(2 * N, 2 * N)

    # Initial state: alternate nu_e, nu_mu, nu_e, nu_tau pattern
    for i in range(N):
        if i % 4 == 1:
            qc.x(2*i + 1)  # nu_mu = |01⟩
        elif i % 4 == 3:
            qc.x(2*i)      # nu_tau = |10⟩

    # Apply U†_PMNS
    Udag = Operator(U_PMNS_4x4.conj().T)
    for i in range(N):
        qc.unitary(Udag, [2*i, 2*i+1], label="U†_PMNS")

    # Trotterization
    Trotter = int(t / dt)
    for _ in range(Trotter):
        pair_alphas = [(i, j, J_ij(i, j, E) * dt) for i, j in pair_indices]
        for i in range(N):
            qc.append(U1_omega_Omega(dt), [2*i, 2*i+1])

        apply_nunu_interaction(qc, pair_indices, pair_alphas)

    # Apply U_PMNS
    U = Operator(U_PMNS_4x4)
    for i in range(N):
        qc.unitary(U, [2*i, 2*i+1], label="U_PMNS")

    # convert our circuit to quimb circuit

    basis = [
        'cz', 'x', 'rz', 'sx'
    ]
    tqc = transpile(qc, basis_gates = basis, optimization_level = 3)
    # fig = tqc.draw('mpl')
    # from IPython.display import display
    # display(fig)

    tqc = qiskit_to_quimb(tqc, backend='torch')

    tn = tqc.psi

    # optimizer = ctg.HyperOptimizer(
    #     methods = configs['methods'],
    #     max_repeats = configs['max_repeats'],
    #     slicing_reconf_opts = configs['slicing_reconf_opts'],
    #     max_time = configs['max_time'],
    #     progbar=True,
    #     parallel=True,
    # )

    # optimizer = ctg.ReusableHyperCompressedOptimizer(
    #     16,
    #     max_repeats = configs['max_repeats'],
    #     progbar=True,
    #     parallel=True,
    # )
    
    # tree = tn.contract_compressed(optimize=optimizer)

    # print(tree)[0]

    counts = {}

    gc.collect()
    
    print("Beginning sampling...")

    for s in tqdm(tqc.sample(shots, seed = 42, backend='torch'), total=shots):
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1

    # Target initial state pattern: 00 01 00 10 00 01 00 10 ...
    # Target initial state pattern based on same logic as initial state
    base_pattern = []
    for i in range(N):
        if i % 4 == 1:
            base_pattern.append('01')  # nu_mu
        elif i % 4 == 3:
            base_pattern.append('10')  # nu_tau
        else:
            base_pattern.append('00')  # nu_e
    # Reverse the *pairs*, not individual bits
    target_state = ''.join((base_pattern))

    #print("Target state:", target_state)

    prob = counts.get(target_state, 0) / shots
    print("Measurement counts:", counts)
    probabilities.append(prob)

    gc.collect()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")



# Plot
plt.plot(times * mu, probabilities, marker='o')
plt.xlabel(r"Time ($\mu^{-1}$)")
plt.ylabel("P(initial → initial)")
plt.title("Survival Probability of {0}-Neutrino State".format(N))
plt.grid(True)
plt.savefig("survi_prob_{0}_nu_dt=5.png".format(N))

tracemalloc.stop()


# In[ ]:



