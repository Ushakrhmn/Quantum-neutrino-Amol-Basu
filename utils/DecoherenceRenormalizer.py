"""
A module that implements decoherence renormalization error mitigation from https://arxiv.org/pdf/2103.08591
"""

import numpy as np
import qiskit as qk

from JobResult import JobResult

class DecoherenceRenormalizer(object):
    """
    A class that implements decoherence renormalization error mitigation from https://arxiv.org/pdf/2103.08591
    """
    def __init__(self, quantum_circuit):
        """
        :param quantum_circuit: If provided at construction, the identity circuit will be generated
        """
        self.identity_circuit = None
        self.rate_estimate = None

        if quantum_circuit is not None:
            self.identity_circuit = self.convert_to_cnot_identity(quantum_circuit)
    
    def set_identity_circuit(self, quantum_circuit):
        """
        Converts a quantum circuit to an identity circuit and bind it to this object
        :param quantum_circuit: The quantum circuit to convert
        """
        self.identity_circuit = self.convert_to_cnot_identity(quantum_circuit)
    
    def get_identity_circuit(self):
        """
        Returns the identity circuit
        :return: The identity circuit
        """
        return self.identity_circuit
    
    def get_rate_estimate(self):
        """
        Returns the rate estimate
        :return: The rate estimate
        """
        return self.rate_estimate

    def convert_to_cnot_identity(self, qc):
        """
        Simplest method to convert an identity circuit from Urbanek et al.
        Remove single qubit gates and replace all two qubits gates with CNOTs

        :param circuit: The quantum circuit to convert 
        """
        identity_circuit = qk.QuantumCircuit(qc.num_qubits, qc.num_clbits)

        for instru in qc.data: # expand into each instruction
            op = instru.operation
            if op.num_qubits == 1: # this is a single qubit gate
                # we only keep measurements
                if op.name == 'measure':
                    identity_circuit.append(op, instru.qubits, instru.clbits)
                else:
                    pass
            elif op.num_qubits == 2: # this is a two qubit gate
                identity_circuit.append(qk.circuit.library.CXGate(), instru.qubits)
            else:
                raise NotImplementedError("Currently only supports 1 and 2 qubit gates")
        
        return identity_circuit
    
    def estimate_error_rate(self, service, shots = 1024):
        """
        Estimate the error rate of the identity circuit by running it on the service
        :param service: The service to run the identity circuit on
        :param shots: The number of shots to run
        """
        if self.identity_circuit is None:
            raise ValueError("Identity circuit is not set")
        
        jr = JobResult(service = service)

        jr.run(self.identity_circuit, {"shots": shots})

        counts = jr.get_counts()

        # Since the circuit is intialized to all |0> state,
        # the circuit only has cnot gates
        # The circuit in ideal case should still have all |0> state

        ecount = counts.get('0'*self.identity_circuit.num_qubits, 0)

        self.rate_estimate = 1 - ecount / shots

        return self.rate_estimate
