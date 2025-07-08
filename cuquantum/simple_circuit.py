"""
Demonstration / Test for using cuquantum tensor network to simulate a simple qiskit circuit.
"""

import qiskit as qk

# to remove the +computecanada postfix in version number, which cause error
import importlib.metadata

# Save original version
_real_version = importlib.metadata.version

# Define patch
def patched_version(pkg_name):
    if pkg_name == "qiskit":
        return _real_version(pkg_name).split("+")[0]
    return _real_version(pkg_name)

# Apply patch
importlib.metadata.version = patched_version

import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Suppress the warning from importing cuquantum
from cuquantum import cutensornet as tn

warnings.resetwarnings()

# ------
# Define a simple circuit
qc = qk.QuantumCircuit(2)

qc.h(0)
qc.cx(0, 1)
# ------

# ------
# Convert the circuit to a cuquantum tensor network
converter = tn.CircuitToEinsum(qc)
# ------

# ------
# Calculate probabilities of measurement outcomes
qubits = qc.qubits

marginals = []

for i in range(len(qubits)):
    where = (qubits[i],)
    fixed = None # generate unbiased
    expr, operd = converter.reduced_density_matrix(where, fixed = fixed)

    # use simple contraction for now
    tmp_rdm = tn.contract(expr, *operd)

    sh = 2 ** len(where) # 2 for a single qubit
    prob = abs(tmp_rdm.reshape(sh, sh).diagonal()) ** 2 # take squared diagonal of rdm

    marginal = prob.reshape((2,) * len(where)) / prob.sum() # reshape and normalize

    marginals.append(marginal)

print("Marginals:")
for i, marginal in enumerate(marginals):
    print(f"Qubit {i}:")
    print(marginal)
# ------