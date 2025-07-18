#!/usr/bin/env python
# coding: utf-8

# In[134]:


"""
Try simulating bipolar oscillation, emu initial states
"""

# In[ ]:
import argparse

parser = argparse.ArgumentParser(description="Set Ne and Nebar from command line")
parser.add_argument('--e', type=int, default=1, help="N e")
parser.add_argument('--b', type=int, default=1, help="N ebar")
parser.add_argument('--method', type=str, default="statevector")
parser.add_argument('--j', type=float, default=5.0, help="interaction strength (default 5.0)")
parser.add_argument('--s', type=int, default=64, help="# of type steps")
parser.add_argument('--l', type=float, default=0.5)
args = parser.parse_args()

print(f"Running with {args.e} electron neutrinos")
print(f"Running with {args.b} electron anti-neutrinos")
print(f"Interaction strength is set to {args.j}.")
print(f"Evolving over L={args.l} over {args.s} timesteps.")
print("Using method: ", args.method)

"""
Set up to evaluate classical solution for nunu
"""
import numpy as np

L          = args.l
t_steps    = args.s
theta      = np.pi/2 - 0.2
# theta = np.pi/2
dmsq       = 1.0
initial_state = ['e']*args.e + ['ebar']*args.b

n_qubits   = len(initial_state)
Delta = dmsq / (2*0.1) * np.array([1.] * args.e + [-1.] * args.b)
b          = np.array([np.sin(2*theta), 0, -np.cos(2*theta)]) # the structure of the vacuum Hamiltonian in the Pauli basis
J          = args.j * np.ones((n_qubits, n_qubits))

# In[136]:


# %load_ext autoreload
# %autoreload 2
# %autosave 0
# %precision %g
import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qiskit as qk
import qiskit_aer as aer

# Launch Wolfram language interface
# from wolframclient.evaluation import WolframLanguageSession
# from wolframclient.language import wl, wlexpr
# wl_session = WolframLanguageSession()

from tqdm import tqdm

import sys

sys.path.append("../")

# set up matplotlib
# %matplotlib inline
# matplotlib.rcParams['text.usetex']    = True
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'
matplotlib.rcParams['font.family']    = 'serif'
matplotlib.rcParams['font.size']      = '16'
# matplotlib.rcParams['figure.figsize'] = 6.4, 4.8
matplotlib.rcParams['figure.figsize'] = 16, 8

# Unit conversion and physical constants
class my_units:
    # Energy and mass
    eV     = 1.
    keV    = 1.e3
    MeV    = 1.e6
    GeV    = 1.e9
    TeV    = 1.e12
    PeV    = 1.e15
    kg     = 5.62e35*eV
    grams  = 0.001*kg
    Kelvin = 8.617333262e-5       # PDG 2022
    Joule  = 1/1.602176634e-19*eV # PDG 2022 [exact]
    erg    = 1e-7*Joule
    
    # Length and time
    m     = 5.076e6
    meter = m
    km    = 1000*m
    cm    = 0.01*m
    nm    = 1.e-9*m
    fm    = 1.e-15*m
    AU    = 1.4960e11*m
    pc    = 30.857e15*m
    kpc   = 1.e3*pc
    Mpc   = 1.e6*pc
    Gpc   = 1.e9*pc
    ly    = 9460730472580800*m  # light year
    sec   = 1.523e15
    hours = 3600*sec
    days  = 24*hours
    yrs   = 365*days
    Hz    = 1./sec
    kHz   = 1.e3*Hz
    MHz   = 1.e6*Hz
    GHz   = 1.e9*Hz
    
    barn  = 1.e-24*cm**2
    pb    = 1.e-36*cm**2
    fb    = 1.e-39*cm**2
    
    # Various astrophysical constants
    GN    = 6.708e-39/1e18  # eV^-2, Newton's constant
    MPl   = 1.22093e19*GeV   # Planck mass, PDG 2013
    Msun  = 1.989e30*kg
    Rsun  = 6.9551e8*meter
    
    # cosmology
    h        = 0.674                         # according to Planck 2018
    H0       = h * 100. * km / sec/ Mpc      # Hubble parameter
    rho_c0   = 3. * H0**2/(8. * np.pi * GN)  # critical density today, Kolb Turner eq. (3.14)
    Omega_dm = 0.12 / h**2                   # dark matter density, Planck 2018
    
    # particle physics
    alpha_em = (1./137.035999139)            # electromagnetic fine structure constant (PDG 2018)
    GF       = 1.166378e-5 / GeV**2          # Fermi constant (PDG 2022)
    s2thW    = 1.03232 * 0.23121             # sin^2 \theta_W(M_Z) from PDG 2022; \kappa from https://arxiv.org/abs/1608.02671
    m_e      = 0.5109989461 * MeV            # electron mass (PDG 2018)
    m_mu     = 105.6583745 * MeV             # muon mass (PDG 2018)
    m_tau    = 1776.86 * MeV                 # tau mass (PDG 2018)
    m_p      = 938.2720813 * MeV             # proton mass (PDG 2018)
    m_n      = 939.565413 * MeV              # neutron mass (PDG 2018)
    m_u      = 931.4940954 * MeV             # atomic mass unit (PDG 2018)
    m_pi0    = 134.9768 * MeV                # neutral pion mass (PDG 2020)
    tau_mu   = 2.1969811e-6 * sec            # muon lifetime (PDG 2018)
    gA       = 1.26                          # axial coupling constant of the proton
    
    # PDG codes
    pdg_nu_e      = 12
    pdg_nu_mu     = 14
    pdg_nu_tau    = 16
    pdg_e         = 11
    pdg_mu        = 13
    pdg_tau       = 15
    pdg_gamma     = 22
    pdg_pip       = 211
    pdg_pi0       = 111
    pdg_KL        = 130
    pdg_Kp        = 321
    pdg_Dp        = 411
    pdg_D0        = 421
    pdg_Dp2s_2460 = 415
    pdg_D02s_2460 = 425
    pdg_Dsp       = 431
    pdg_Dsps      = 433
    pdg_Ds2s_2573 = 435
    
u = my_units()

# Pauli matrices
sigma_0 = np.array([[1, 0],  [0,  1]])
sigma_1 = np.array([[0, 1],  [1,  0]])
sigma_2 = np.array([[0,-1j], [1j, 0]])
sigma_3 = np.array([[1, 0],  [0, -1]])


# In[137]:


from mft import P_osc_RS, P_osc_RS_bipolar
from mft import sigma_0, sigma_1, sigma_2, sigma_3

# In[138]:


# mean-field solution
L_table = np.linspace(0, L, t_steps)
sol     = P_osc_RS(L_table, theta, Delta, 0., abs(J), initial_flavors=initial_state) # factor 2 because we compare to a system with 2 nu + 2 nubar
MFT_P_table = np.reshape(sol.y, (n_qubits,3,len(L_table)))

# P_table = np.reshape(sol.y, (4,3,len(L_table)))

plt.plot(L_table, 0.5*(1+MFT_P_table[0,2,:]), ls=':', lw=3)
plt.plot(L_table, 0.5*(1+MFT_P_table[-1,2,:]), ls=':', lw=3)

# In[139]:


adaptive_l_table = L_table

# In[140]:


print(adaptive_l_table)

# In[141]:


from scipy.linalg import expm

def U_nunu(theta):
    """the exponential of the neutrino-neutrino interaction Hamiltonian,
       exp(-1j*H_self_int(theta))), see Friedland & Lunardini,
       https://arxiv.org/abs/hep-ph/0304055
    """
    return np.array([[np.exp(-1j*theta), 0,                          0,                          0],
                     [0,                 0.5*( 1+np.exp(-1j*theta)), 0.5*(-1+np.exp(-1j*theta)), 0],
                     [0,                 0.5*(-1+np.exp(-1j*theta)), 0.5*( 1+np.exp(-1j*theta)), 0],
                     [0,                 0,                          0,                          np.exp(-1j*theta)]])

def U_nunubar(theta):
    """
    returns the interaction term for nunubar interactions
    """

    block = np.asarray([
        [-2, 0, 0, -1],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, -2]
    ]) / 2

    return expm(-1j * theta * block)
# In[142]:


# Set up the discretization
dt_table       = np.diff(adaptive_l_table)

qc = qk.QuantumCircuit(n_qubits)

REMOVE_ENTANGLE = True

for i in tqdm(range(len(dt_table))):
    dt = dt_table[i]

    if REMOVE_ENTANGLE:

        # artificially remove entanglement by resetting each qubit to a non-entangled state
        qc_no_save = qc.remove_final_measurements(inplace=False)
        qc_no_save.data = [inst for inst in qc.data if inst.operation.name != 'save_density_matrix']
        sv = qk.quantum_info.Statevector(qc_no_save)
        rho = [ np.array(qk.quantum_info.partial_trace(sv, [j for j in range(n_qubits) if j != k])) for k in range(n_qubits) ]
        p   = [ np.real(np.array([ np.trace(sigma_1@rho[k]), np.trace(sigma_2@rho[k]), np.trace(sigma_3@rho[k]) ])) for k in range(n_qubits) ]
        
        # reset qubits to a non-entangled state to emulate the mean-field picture
        qc.reset(range(n_qubits))
        for j in range(n_qubits):
            qc.ry(np.arccos(p[j][2]), j)
            qc.rz(np.arctan2(p[j][1], p[j][0]), j)

    qc.save_density_matrix(label=str(i+1))

    for j in range(n_qubits):
        qc.rx(-dt*Delta[j]*b[0], j)
        qc.rz(-dt*Delta[j]*b[2], j)

    for iq1 in range(n_qubits):
        for iq2 in range(iq1+1, n_qubits):
            if initial_state[iq1] == initial_state[iq2]: # if both e or both ebar
                qc.unitary(U_nunu(-dt*J[iq1, iq2]), [iq1, iq2])
            else:
                qc.unitary(U_nunubar(-dt*J[iq1, iq2]), [iq1, iq2])

# save final state
qc.save_density_matrix(label=str(i+2))

# run simulation    
sim = aer.AerSimulator(method = args.method,precision="double", zero_threshold=1e-14, validation_threshold=1e-12)
result = sim.run(qc).result()

# In[143]:


# extract and plot results
n_qubits = qc.num_qubits
rho_data = [result.data()[k] for k in result.data().keys()]  # assume keys are times

def reduce_qubit(j):
    tmp = { k : np.array(qk.quantum_info.partial_trace(result.data()[k], [q for q in range(n_qubits) if q != j])) for k in result.data().keys() }
    print("Finished calculation for qubit", j)
    return tmp

print("Obtaining partial traces ...")

# rho_reduced = Parallel(n_jobs=-1)(delayed(reduce_qubit)(j) for j in tqdm(range(n_qubits)))
rho_reduced = [reduce_qubit(j) for j in tqdm(range(n_qubits))]

print("Calculating probabilities ...")

# rho_reduced = [ { k : np.array(qk.quantum_info.partial_trace(result.data()[k], [q for q in range(n_qubits) if q != j])) for k in result.data().keys() } for j in tqdm(range(n_qubits)) ]
pp = np.array([ [[float(k), rho_reduced[j][k][0,0]] for k in rho_reduced[j].keys() ] for j in range(n_qubits) ])
pp = np.abs(np.take_along_axis(pp, pp[:,:,0].argsort(axis=1)[:,:,None], axis=1))

for j in [0, -1]:
    color = 'black' if j < n_qubits // 2 else 'red'
    plt.plot(adaptive_l_table, pp[j,:, 1], color=color)

plt.clf()

plt.text(0.9, 0.9, f"$\\theta = {(theta / np.pi * 180):.2f}$", transform=plt.gca().transAxes, fontsize=16)

# In[144]:

plt.plot(L_table, 0.5*(1+MFT_P_table[0,2,:]), ls=':', lw=3, color='blue')
plt.plot(L_table, 0.5*(1+MFT_P_table[-1,2,:]), ls=':', lw=3, color='orange')

plt.plot(adaptive_l_table, pp[0,:, 1], color='blue', label=initial_state[0])
plt.plot(adaptive_l_table, 1-pp[-1,:, 1], color='orange', label=initial_state[-1])

plt.legend()
plt.savefig(f"bipolar_e_{args.e}_ebar_{args.b}_j_{args.j}_{args.method}.png")

# In[ ]:



