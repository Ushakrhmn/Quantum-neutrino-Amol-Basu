import qiskit as qk
from qiskit import QuantumCircuit
import quimb as qu
import quimb.tensor as qtn
import torch

def qiskit_to_quimb(qc, backend='numpy'):
    """
    Converts a qiskit QuantumCircuit object
    to a equivalent quimb circuit object

    Currently only supports 1 and 2 qubit gates of specific basis:

    [CZ, RZ, SX, X]

    :param qc: qiskit QuantumCircuit
        The circuit to be converted
    :return: quimb Circuit

    :raise NotImplementedError: if circuit contains >2 qubit gates.
    """

    def to_backend(x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return torch.tensor(x, dtype=torch.complex64)

    qu_circuit = qtn.CircuitMPS(qc.num_qubits, to_backend=to_backend)

    op1_dict = {
        'x': 'X',
        'rz': 'RZ',
        'sx': 'SX'
    }

    op2_dict = {
        'cz': 'CZ'
    }
    for instru in qc.data: # expand into each instruction
        op = instru.operation
        if op.num_qubits == 1: # this is a single qubit gate
            if op.name == 'x' or op.name == 'sx':
                qu_circuit.apply_gate(op1_dict[op.name], instru.qubits[0]._index)
            elif op.name == 'rz':
                qu_circuit.apply_gate(op1_dict[op.name], instru.params[0], instru.qubits[0]._index)
            else:
                raise NotImplementedError("Currently only supports X, RZ, SX gates")
        elif op.num_qubits == 2: # this is a two qubit gate
            if op.name == 'cz':
                qu_circuit.apply_gate(op2_dict[op.name], instru.qubits[0]._index, instru.qubits[1]._index)
            else:
                raise NotImplementedError("Currently only supports CZ gates")
        else:
            raise NotImplementedError("Currently only supports 1 and 2 qubit gates")
    
    return qu_circuit