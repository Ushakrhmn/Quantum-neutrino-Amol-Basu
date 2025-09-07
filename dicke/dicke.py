from qiskit import QuantumCircuit
from math import comb, sqrt
import numpy as np
import matplotlib.pyplot as plt

def dicke_statevector(n: int, k: int) -> np.ndarray:
    """
    Build the |D(n,k)> Dicke state's full statevector (length 2^n).
    Amplitudes are uniform over all computational basis states with Hamming weight = k.
    Basis ordering matches Qiskit's convention: |q_{n-1} ... q_1 q_0>.
    """
    assert 0 <= k <= n, "k must be between 0 and n"
    dim = 2 ** n
    vec = np.zeros(dim, dtype=complex)

    amp = 1.0 / sqrt(comb(n, k))
    # Enumerate all bitstrings 0..2^n-1; set amplitude if popcount == k
    for i in range(dim):
        if i.bit_count() == k:
            vec[i] = amp
    return vec


def prepare_dicke_simple(circ: QuantumCircuit, qubits, k: int):
    """
    The simplest, most straightforward Dicke state preparation:
    1) Construct the exact statevector of |D(n,k)>
    2) Use QuantumCircuit.initialize to set the register to that state

    Parameters
    ----------
    circ   : QuantumCircuit
    qubits : list of qubit indices (length n)
    k      : number of excitations (Hamming weight)
    """
    n = len(qubits)
    vec = dicke_statevector(n, k)
    # Initialize expects amplitudes in the |q_{n-1} ... q_0> ordering over this 'qubits' slice.
    circ.initialize(vec, qubits)


if __name__ == "__main__":
    n = 5
    k = 3
    qc = QuantumCircuit(n)
    prepare_dicke_simple(qc, list(range(n)), k)

    # Draw & save
    qc.draw('mpl')
    plt.tight_layout()
    plt.savefig("dicke_circuit.png", dpi=180)

    # (Optional) quick sanity check on normalization
    # from qiskit.quantum_info import Statevector
    # sv = Statevector.from_instruction(qc)
    # print("Norm:", np.vdot(sv.data, sv.data).real)