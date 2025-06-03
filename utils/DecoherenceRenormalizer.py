"""
A module that implements decoherence renormalization error mitigation from https://arxiv.org/pdf/2103.08591
"""

import numpy as np
import qiskit as qk
import warnings
import mthree
import random

from sympy import N

from JobResult import JobResult

import sys

sys.path.append("../SBN")
from helpers import count_1q

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
        self.trace = None

        if quantum_circuit is not None:
            # self.identity_circuit = self.convert_to_cnot_identity(quantum_circuit)
            self.identity_circuit = self.convert_to_cnot_identity_with_some_1q(quantum_circuit)

    def set_trace(self, trace):
        """
        Sets the trace of the observable tr(O)
        """
        self.trace = trace

    def get_trace(self):
        """
        Returns the trace of the observable tr(O)
        :return: The trace of the observable
        """
        if self.trace is None:
            raise ValueError("Trace is not set")
        return self.trace
    
    def set_trace_by_statevector(self, statevector):
        """
        Set trace by computing the trace of the outer produce of a given statevector
        This is the trace of the projection operator onto the final state.

        For some statevector,

        tr(|x><x|) = sum_{i=1}^n (vv^dagger)_{ii} = |x|^2
        """

        self.trace = np.linalg.norm(statevector)**2

        return self.trace

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
    
    def _append_1q(self, qc, i, gate_name):
        if gate_name == 'I':
            pass
        elif gate_name == 'X':
            qc.x(i)
        elif gate_name == 'Y':
            qc.y(i)
        elif gate_name == 'Z':
            qc.z(i)

    def _count_1q_before(self, qc, op_idx, qubit_idx0, qubit_idx1):

        pos0check = False
        pos1check = False

        p0prev, p1prev = False, False

        count = 0

        i = op_idx - 1
        while pos0check is False or pos1check is False:
            try:
                op = qc.data[i].operation
                instru = qc.data[i]

                if op.num_qubits == 2:
                    q0, q1 = instru.qubits[0]._index, instru.qubits[1]._index
                    # we have reached the next two qubit gate
                    if q0 == qubit_idx0 or q1 == qubit_idx0:
                        pos0check = True
                    elif q0 == qubit_idx1 or q1 == qubit_idx1:
                        pos1check = True
                elif op.num_qubits == 1:
                    q0 = instru.qubits[0]._index
                    if q0 == qubit_idx0 and pos0check is False:
                        if p0prev is False:
                            p0prev = True
                        else:
                            pos0check = True
                            count += 1
                    elif q0 == qubit_idx1 and pos1check is False:
                        if p1prev is False:
                            p1prev = True
                        else:
                            pos1check = True
                            count += 1
            except IndexError:
                # we have reached the beginning of the circuit
                break

            i -= 1

        return count
    
    def _count_1q_after(self, qc, op_idx, qubit_idx0, qubit_idx1):
        
        pos0check = False
        pos1check = False
        p0prev, p1prev = False, False
        count = 0
        i = op_idx + 1
        while pos0check is False or pos1check is False:
            try:
                op = qc.data[i].operation
                instru = qc.data[i]

                if op.num_qubits == 2:
                    q0, q1 = instru.qubits[0]._index, instru.qubits[1]._index
                    # we have reached the next two qubit gate
                    if q0 == qubit_idx0 or q1 == qubit_idx0:
                        pos0check = True
                    elif q0 == qubit_idx1 or q1 == qubit_idx1:
                        pos1check = True
                elif op.num_qubits == 1:
                    q0 = instru.qubits[0]._index
                    if q0 == qubit_idx0 and pos0check is False:
                        if p0prev is False:
                            p0prev = True
                        else:
                            pos0check = True
                            count += 1
                    elif q0 == qubit_idx1 and pos1check is False:
                        if p1prev is False:
                            p1prev = True
                        else:
                            pos1check = True
                            count += 1
            except IndexError:
                # we have reached the end of the circuit
                break

            i += 1

        return count


    def convert_to_cnot_identity_with_some_1q(self, qc):
        """
        This is relevant for IBM torino or other computers with Heron r1 process, native gate CZ
        """

        libI = [
            ('I', 'I', 'I', 'I'),
        ]

        lib2 = [
            ('I', 'X', 'I', 'X'),
            ('I', 'Y', 'I', 'Y'),
            ('I', 'Z', 'Z', 'Z'),
            ('Z', 'I', 'Z', 'I'),
        ]

        lib3 = [
            ('Y', 'I', 'Y', 'X'),
            ('Y', 'X', 'Y', 'I'),
            ('X', 'I', 'X', 'X'),
            ('X', 'X', 'X', 'I'),
            ('Z', 'Y', 'I', 'Y'),
            ('Z', 'Z', 'I', 'Z'),

        ]

        lib4 = [
            ('Y', 'Y', 'X', 'Z'),
            ('Y', 'Z', 'X', 'Y'),
            ('X', 'Y', 'Y', 'Z'),
            ('X', 'Z', 'Y', 'Y'),
            ('Z', 'X', 'Z', 'X'),
        ]

        identity_circuit = qk.QuantumCircuit(qc.num_qubits, qc.num_clbits)

        mc = 0

        for op_idx, instru in enumerate(qc.data): # expand into each instruction
            op = instru.operation
            if op.num_qubits == 1: # this is a single qubit gate
                # we only keep measurements
                if op.name == 'measure':
                    identity_circuit.append(op, instru.qubits, instru.clbits)
                else:
                    pass
            elif op.num_qubits == 2: # this is a two qubit gate
                mc += 1
                adj_1q = self._count_1q_before(qc, op_idx, instru.qubits[0]._index, instru.qubits[1]._index) + self._count_1q_after(qc, op_idx, instru.qubits[0]._index, instru.qubits[1]._index)
                library = None
                if adj_1q == 0 or adj_1q == 1: # no 1q gates or only one 1q gate
                    identity_circuit.append(qk.circuit.library.CXGate(), instru.qubits)
                else: # we include some 1q gates with it, see method in 2103.08591
                    if adj_1q == 2:
                        library = lib2
                    elif adj_1q == 3:
                        library = libI
                    elif adj_1q == 4:
                        library = libI

                    pattern = random.choice(library)
                    self._append_1q(identity_circuit, instru.qubits[0]._index, pattern[0])
                    self._append_1q(identity_circuit, instru.qubits[1]._index, pattern[1])
                    identity_circuit.append(qk.circuit.library.CXGate(), instru.qubits)
                    self._append_1q(identity_circuit, instru.qubits[0]._index, pattern[2])
                    self._append_1q(identity_circuit, instru.qubits[1]._index, pattern[3])
            else:
                raise NotImplementedError("Currently only supports 1 and 2 qubit gates")
        print("Number of two qubit gates checked by identity maker: {}".format(mc))
        return identity_circuit
    
    def estimate_error_rate(self, service, shots = 65536, transpile_options = None, use_m3 = True, backend = None):
        """
        Estimate the error rate of the identity circuit by running it on the service
        :param service: The service to run the identity circuit on
        :param shots: The number of shots to run
        """
        if self.identity_circuit is None:
            raise ValueError("Identity circuit is not set")
        
        if self.verbose:
            print("Estimating error rate with {} shots".format(shots))

        counts = {}

        n_run = 8
        small_shots = shots // n_run

        random.seed(42)

        for ir in range(n_run):
            print("Running identity circuit run {}/{}".format(ir + 1, n_run))
            temp_count = self._run_identity_circuit(service, small_shots, transpile_options, use_m3, backend)
            for key in temp_count.keys():
                if key in counts:
                    counts[key] += temp_count[key]
                else:
                    counts[key] = temp_count[key]

        if self.verbose:
            print("Got counts for identity run.")

        # Since the circuit is intialized to all |0> state,
        # the circuit only has cnot gates
        # The circuit in ideal case should still have all |0> state

        ecount = 0

        for key in counts.keys():
            if int(key) != 0:
                ecount += counts[key]

        # @FIXME need to change shots in case shots % n_run != 0
        if not use_m3:
            self.rate_estimate = ecount / shots
        else: 
            self.rate_estimate = ecount / sum(counts.values())

        return self.rate_estimate

    def _run_identity_circuit(self, service, shots, transpile_options, use_m3, backend):
        jr = JobResult(service = service, verbose = self.verbose)

        # make sure there is not optimization, so cnot gates are not removed

        transpile_options = transpile_options.copy() if transpile_options is not None else None

        if 'optimization_level' in transpile_options:
            transpile_options['optimization_level'] = 0

        if 'initial_layout' in transpile_options:
            del transpile_options['initial_layout']

        tqc = qk.compiler.transpile(self.identity_circuit, **transpile_options)

        if use_m3:
            if self.verbose:
                print("DR is using mthree for error mitigation")
            mapping = mthree.utils.final_measurement_mapping(tqc)
    
            mit = mthree.M3Mitigation(backend)

            mit.cals_from_system(mapping)

        jr.run(tqc, {"shots": shots})

        counts = jr.get_counts()

        if use_m3:
            counts = mit.apply_correction(counts, mapping).nearest_probability_distribution()
        return counts
    
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
    
    def estimate_error_rate_from_counts_by_criteria(self, counts, criteria):
        """
        Estimate the error rate from the counts of the identity circuit
        :param counts: The counts of the identity circuit
        :param shots: The number of shots used to get the counts
        :param criteria: A callable function that takes a key and returns True if the key is expected, False if error occured

        :return: The estimated error rate
        """
        if self.verbose:
            print("Estimating error rate from counts by criteria")

        ecount = 0

        shots = 0

        for key in counts.keys():
            shots += counts[key]
            if not criteria(key):
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
        return corre_val
