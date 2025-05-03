import qiskit as qk
import numpy as np
import scipy as sp

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

    Note: this should be called after transpiling the circuit,
    as the transpiler can optimize the circuit and remove the identity pairs

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

def extrapolate_to_zero(meas, lambd, method = 'exponential'):
    """
    Extrapolate the data to zero noise level using the provided method
    :param meas: list of measurements
    :param lambd: list of noise levels (0 is noiseless, 1 is unscaled circuit)
    :param method: str method to use for extrapolation
        'exponential', 'linear', or 'quadratic'

    :return: float extrapolated value at lambda = 0
    :raise ValueError: if method is not supported

    """    

    if method not in ['exponential', 'linear', 'quadratic']:
        raise ValueError("Method not supported, use 'exponential', 'linear', or 'quadratic'")
    
    def linear(x, a, b):
        return a * x + b
    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c
    def exponential(x, a, b):
        return a * np.exp(b*x)
    
    f = None
    if method == 'linear':
        f = linear
    elif method == 'quadratic':
        f = quadratic
    elif method == 'exponential':
        f = exponential

    assert f is not None

    params, _, _, _, flag = sp.optimize.curve_fit(f, lambd, meas, full_output=True)

    if flag not in [1, 2, 3, 4]:
        raise ValueError("Curve fit failed, check your data and method")
    
    model = lambda x: f(x, *params)

    return model(0)
    
def count_to_probability(count, target_state, shots):
    """
    Given a dictionary of counts (from circuit run) and a target state,
    calculate the probability of measuring the said state
    This is equivalent to the expectation value of |x><x| for the state x
    :param count: dict
        The counts from the circuit run
    :param target_state: str
        The target state to measure
    :param shots: int
        The number of shots used in the circuit run
    :return: float
        The probability of measuring the target state
    """
    

