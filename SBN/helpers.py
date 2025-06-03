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
    noisy_circuit = qk.QuantumCircuit(qc.num_qubits, qc.num_clbits)

    for instru in qc.data: # expand into each instruction
        op = instru.operation
        if op.num_qubits == 1: # this is a single qubit gate
            # simply need to copy this to the new circuit
            noisy_circuit.append(op, instru.qubits, instru.clbits)
        elif op.num_qubits == 2: # this is a two qubit gate
            # we need to add scaling
            # first make copies in the new circuit
            noisy_circuit.append(op, instru.qubits)
            reps = (lambd - 1) // 2
            for _ in range(reps):
                noisy_circuit.append(op, instru.qubits, instru.clbits)
                noisy_circuit.append(op.inverse(), instru.qubits, instru.clbits)
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
    def exponential(x, a, b, c):
        return a * np.exp(b*x) + c
    
    f = None
    if method == 'linear':
        f = linear
    elif method == 'quadratic':
        f = quadratic
    elif method == 'exponential':
        f = exponential

    assert f is not None

    params, _, _, _, flag = sp.optimize.curve_fit(f, lambd, meas, full_output=True, maxfev = int(1e5))

    if flag not in [1, 2, 3, 4]:
        raise ValueError("Curve fit failed, check your data and method")
    
    model = lambda x: f(x, *params)

    fit0 = model(0)

    if fit0 > 3: # we have obviously overfitted
        raise ValueError("Extrapolation resulted in a value > 3")

    return fit0

def count_2q(qc):
    """
    Count the number of 2q

    :param qc: The quantum circuit to count CNOT gates in.
    :return: The number of 2q gates in the circuit.
    """
    count = 0

    for instru in qc.data: # expand into each instruction
        op = instru.operation
        if op.num_qubits == 2: # this is a two qubit gate
            count += 1

    return count

def count_1q(qc):
    """
    Count the number of 1q gates

    :param qc: The quantum circuit to count CNOT gates in.
    :return: The number of 1q gates in the circuit.
    """
    count = 0

    for instru in qc.data: # expand into each instruction
        op = instru.operation
        if op.num_qubits == 1: # this is a two qubit gate
            count += 1

    return count

def get_target_state(base_pattern):
    """
    Generates target state string from base pattern represented
    by 'M', 'T', and 'E' characters.
    """

    target_state = ''.join((base_pattern))
    # if order[-1] > order[0]:
        # we are in original order
    target_state = target_state.replace('M', '01')
    target_state = target_state.replace('T', '10')
    target_state = target_state.replace('E', '00')
    return target_state

def swap_correction_order_bp(order, base_pattern, i, j):
    """
    Swap the elements at index i and j in the provided order and base pattern
    """
    order = swap_order_i_j(order, i, j)
    base_pattern = swap_base_pattern_i_j(base_pattern, i, j)
    return order, base_pattern

def get_interaction_pair_index(N, method = 'default'):
    """
    Returns a list of tuples to interact pairs of neutrinos

    :param N: int, number of neutrinos
    :param method: str
        The method to use for pairing, can be 'default', 'full', or 'bubble'
    """
    if method == 'default':
        return [(i, j) for i in range(N) for j in range(i+1, N)]
    elif method == 'bubble':
        pair_indices = []
        for j in range(N):
            nmj = N - j - 1
            for i in range(nmj):
                pair_indices.append((i, i+1))
        return pair_indices