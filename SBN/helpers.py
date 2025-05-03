import qiskit as qk
import numpy as np

def swap_order_i_j(order, i, j):
    """
    Swap the elements at index i and j in the provided order
    """
    order[i], order[j] = order[j], order[i]
    return order

def swap_base_pattern_i_j(base_pattern, i, j):
    """
    Swap the elements at index i and j in the provided base pattern
    """
    base_pattern[i], base_pattern[j] = base_pattern[j], base_pattern[i]
    return base_pattern

def generate_noise_scaled_circuit(qc, lambd = 3):
    """
    Generate a circuit that in the noiseless limit is equivalent 
    to the original circuit, but artifically increase 
    the amount of two-qubit gates (assume that is dominant source of noise)
    This is done by adding pairs of two-qubits gates and inverse to each
    existing two-qubit gate in the circuit

    :raise NotImplementedError: if circuit contains >2 qubit gates.

    :param qc: qiskit QuantumCircuit
        The circuit to be scaled
    :param lambd: int
        The "noise level" factor, add (lambd - 1)/2 pairs of gates
    :param basis_2q_gate: str
        Currently only supports 'cx' and 'ecr'

    :return: qiskit QuantumCircuit
        The scaled circuit
    """
    # Create a new circuit
    noisy_circuit = qk.QuantumCircuit(qc.num_qubits)

    for instru in qc.data: # expand into each instruction
        op = instru.operation
        if op.num_qubits == 1: # this is a single qubit gate
            # simply need to copy this to the new circuit
            noisy_circuit.append(op, instru.qubits)
        elif op.num_qubits == 2: # this is a two qubit gate
            # we need to add scaling
            # first make copies in the new circuit
            noisy_circuit.append(op, instru.qubits)
            reps = (lambd - 1) // 2
            for _ in range(reps):
                noisy_circuit.append(op, instru.qubits)
                noisy_circuit.append(op.inverse(), instru.qubits)
        else:
            raise NotImplementedError("Currently only supports 1 and 2 qubit gates")
    
    return noisy_circuit