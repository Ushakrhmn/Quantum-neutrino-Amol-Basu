"""
A module that implements decoherence renormalization error mitigation from https://arxiv.org/pdf/2103.08591
"""

import numpy as np
import qiskit as qk
import warnings

from JobResult import JobResult

class DecoherenceRenormalizer(object):
    """
    A class that implements decoherence renormalization error mitigation from https://arxiv.org/pdf/2103.08591
    """
    def __init__(self, quantum_circuit, verbose = False):
        """
        :param quantum_circuit: If provided at construction, the identity circuit will be generated
        """
        self.identity_circuit = None
        self.rate_estimate = None
        self.verbose = verbose

        if quantum_circuit is not None:
            # self.identity_circuit = self.convert_to_cnot_identity(quantum_circuit)
            self.identity_circuit = self.convert_to_cz_identity(quantum_circuit)
    
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
        Simplest method to convert to an identity circuit from Urbanek et al.
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
    
    def convert_to_cz_identity(self, qc):
        """
        This is relevant for IBM torino or other computers with Heron r1 process, native gate CZ
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
                identity_circuit.append(qk.circuit.library.CZGate(), instru.qubits)
            else:
                raise NotImplementedError("Currently only supports 1 and 2 qubit gates")
        
        return identity_circuit
    
    def estimate_error_rate(self, service, shots = 1024, transpile_options = None, plan = 3):
        """
        Estimate the error rate of the identity circuit by running it on the service
        :param service: The service to run the identity circuit on
        :param shots: The number of shots to run
        """
        if self.identity_circuit is None:
            raise ValueError("Identity circuit is not set")
        
        if self.verbose:
            print("Estimating error rate with {} shots".format(shots))
        
        jr = JobResult(service = service, verbose = self.verbose)

        # make sure there is not optimization, so cnot gates are not removed

        transpile_options = transpile_options.copy() if transpile_options is not None else None

        if 'optimization_level' in transpile_options:
            transpile_options['optimization_level'] = 0

        if 'initial_layout' in transpile_options:
            del transpile_options['initial_layout']

        tqc = qk.compiler.transpile(self.identity_circuit, **transpile_options)

        jr.run(tqc, {"shots": shots})

        counts = jr.get_counts()

        if self.verbose:
            print("Got counts for identity run.")

        # Since the circuit is intialized to all |0> state,
        # the circuit only has cnot gates
        # The circuit in ideal case should still have all |0> state

        ecount = 0

        for key in counts.keys():
            if int(key) != 0:
                ecount += counts[key]

        self.rate_estimate = ecount / shots

        return self.rate_estimate
    
    def estimate_error_rate_from_counts(self, counts):
        """
        Estimate the error rate from the counts of the identity circuit
        :param counts: The counts of the identity circuit
        :param shots: The number of shots used to get the counts

        :return: The estimated error rate
        """
        if self.verbose:
            print("Estimating error rate from counts")

        ecount = 0

        shots = 0

        for key in counts.keys():
            shots += counts[key]
            if int(key) != 0:
                ecount += counts[key]

        self.rate_estimate = ecount / shots

        return self.rate_estimate
    
    def estimate_error_rate_no_wait(self, service, shots = 1024, transpile_options = None):
        """
        Estimate the error rate of the identity circuit, submitting the job to the service
        Does not wait for the job to finish and does not return an estimated rate.

        :param service: The service to run the identity circuit on
        :param shots: The number of shots to run

        :return: the job id of the submitted job
        """
        if self.identity_circuit is None:
            raise ValueError("Identity circuit is not set")
        
        if self.verbose:
            print("Estimating error rate with {} shots".format(shots))
        
        jr = JobResult(service = service, verbose = self.verbose)

        # make sure there is not optimization, so cnot gates are not removed

        transpile_options = transpile_options.copy() if transpile_options is not None else None

        if 'optimization_level' in transpile_options:
            transpile_options['optimization_level'] = 0

        if 'initial_layout' in transpile_options:
            del transpile_options['initial_layout']

        tqc = qk.compiler.transpile(self.identity_circuit, **transpile_options)

        jr.run(tqc, {"shots": shots})

        return jr.get_job_id()
    
    def renormalize(self, expectation, c=0):
        """
        Given an expectation value, renormalize with the estimated error rate per formula

        <O> = (<O>_noisy - c) / (1 - p) + c

        p is the estimated error rate, c is a constant shift factor that by defualt is 0

        Note that for a certain final state |psi>, the probability of measuring the state is the same as
        the expectation of operator O = |psi><psi|

        :param expectation: The expectation value to renormalize
        :param c: The constant shift factor 
        """

        if self.rate_estimate is None:
            raise ValueError("Error rate is not estimated")
        
        try:
            corre_val = (expectation - c) / (1 - self.rate_estimate) + c
        except ZeroDivisionError:
            warnings.warn("Error rate is 1, returning expectation value as is.")
        return expectation
