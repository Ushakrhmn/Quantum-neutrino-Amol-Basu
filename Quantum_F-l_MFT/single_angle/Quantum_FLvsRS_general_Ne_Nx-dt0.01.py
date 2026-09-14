#!/usr/bin/env python
# coding: utf-8

# # Quantum_FL_RS_general_Ne_Nx
# 
# Clean generalized notebook for comparing the many-body Qiskit evolution with the Raffelt--Sigl mean-field evolution for
# 
# \[
# N_e\;\nu_e + N_x\;\nu_x(\alpha),
# \qquad
# \nu_x(\alpha)=\cos\alpha\,\nu_e+\sin\alpha\,\nu_\mu .
# \]
# 
# This is the generalized version of the older fixed \(1\nu_e+15\nu_x\) notebook.
# 
# Default choice:
# 
# \[
# N_e=1,\qquad N_x=15.
# \]
# 
# To check \(N_e=2\) later, change only `Ne` in Cell 1.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile
)
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator


# In[2]:


import argparse

parser = argparse.ArgumentParser(
    description="Many-body vs Random-Phase neutrino oscillation simulation"
)

parser.add_argument(
    "--Ne",
    type=int,
    help="Number of initially electron-flavor neutrinos",
)

parser.add_argument(
    "--Nx",
    type=int,
    help="Number of initially x-flavor neutrinos",
)

args, _ = parser.parse_known_args()

if args.Ne is None:
    Ne = int(input("Enter the number of electron-flavor neutrinos (Ne): "))
else:
    Ne = args.Ne

if args.Nx is None:
    Nx = int(input("Enter the number of x-flavor neutrinos (Nx): "))
else:
    Nx = args.Nx

N = Ne + Nx

if Ne < 1 or Nx < 1:
    raise ValueError("Ne and Nx must be positive integers.")


# In[3]:


# =============================
# CELL 1: Imports and parameters
# =============================

# -----------------------------
# Particle numbers
# -----------------------------


# -----------------------------
# Physics parameters
# -----------------------------
E = 1.0
dm2 = 1.0
theta_vac = 0.195

sin_2th = np.sin(2.0 * theta_vac)
cos_2th = np.cos(2.0 * theta_vac)
omega1 = dm2 / (4.0 * E)

bx = omega1 * np.sin(2.0 * theta_vac)
by = 0.0
bz = -omega1 * np.cos(2.0 * theta_vac)

# This is the scale used for the dimensionless time axis t*mu.
mu = 1

# -----------------------------
# Evolution choices
# -----------------------------
# Keep these False to match the later cells of the old e1_x15 notebook:
# self-interaction only, with no vacuum and no matter term.
use_vacuum = False
use_matter = False

# -----------------------------
# Time / Trotter / sampling parameters
# -----------------------------
dt = 0.01
T_max = 10 #200.0
sample_every = 200

n_steps_max = int(round(T_max / dt))
times = np.arange(0, n_steps_max + 1, sample_every) * dt

# -----------------------------
# Qiskit simulation parameters
# -----------------------------
shots = 4096
seed_simulator = 12345
backend = AerSimulator(seed_simulator=seed_simulator)

# -----------------------------
# Alpha cases
# -----------------------------
alpha_cases = [
    (np.pi/2, r"\pi/2"),
    (np.pi/3, r"\pi/3"),
    (np.pi/4, r"\pi/4"),
    (np.pi/6, r"\pi/6"),
    (0.0,     r"0"),
]

print(f"Using Ne = {Ne}, Nx = {Nx}, total N = {N}")
print(f"omega1 = {omega1}")
print(f"mu = omega1 * N = {mu}")
print(f"Number of sampled times = {len(times)}, T_max = {T_max}, dt = {dt}")
print(f"Evolution mode: self-interaction only = {not use_vacuum and not use_matter}")


# In[4]:


# =======================================
# CELL 2: Interaction matrix J_ij
# =======================================



#J = interaction_matrix(N, dm2, E)
J=1
#print("J shape =", J.shape)
#print("J min/max =", J.min(), J.max())


# In[5]:


# =========================================
# CELL 3: Qiskit evolution building blocks
# =========================================

def add_vacuum_evolution(qc, n, bx, by, bz, N):
    """
    Add n first-order vacuum steps to the circuit.

    This function is retained for completeness, but use_vacuum=False by default
    to match the old e1_x15 comparison.
    """
    for _ in range(n):
        for q in range(N):
            qc.rx(2.0 * bx, q)
            qc.ry(2.0 * by, q)
            qc.rz(2.0 * bz, q)


def add_interaction_evolution(qc, n, J, N, dt):
    """
    Add n first-order interaction steps to the circuit.

    For each pair (i,j), apply approximately

        exp[-i Jij dt (XX + YY + ZZ)]

    using

        RXX(2 Jij dt) RYY(2 Jij dt) RZZ(2 Jij dt).
    """
    for _ in range(n):
        for i in range(N):
            for j in range(i + 1, N):
                phi = 2.0 * J* dt
                qc.rxx(phi, i, j)
                qc.ryy(phi, i, j)
                qc.rzz(phi, i, j)


# In[6]:


# Dicke State Block

# ============================================================
# Dicke-state preparation
# ============================================================

def prepare_dicke_state(qc, qubits, N, m):
    S = N / 2

    j = int(round(m + S))

    if j < 0 or j > N:
        raise ValueError(
            f"Invalid m={m} for N={N}: j={j} is outside [0,N]."
        )

    for p, q in enumerate(qubits):
        if (j >> p) & 1:
            qc.x(q)


# ============================================================
# Initial state of a symmetric neutrino mode in the
# Dicke-state basis
# ============================================================

def prepare_dicke_superposition(qc, qubits, N, alpha):
    """
    Prepare

        (sin(alpha)|nu_mu> + cos(alpha)|nu_e>)^(⊗N)

    in the Dicke basis.

    Convention:

        |nu_e>  = |0>  <-> j = N
        |nu_mu> = |1>  <-> j = 0

    Therefore j counts the number of nu_e states.

    The Dicke-basis state is

        sum_{j=0}^N sqrt(C(N,j))
            cos(alpha)^j
            sin(alpha)^(N-j)
            |j>.

    Hence:

        alpha = 0:
            |j=N> = |nu_e>^(⊗N)

        alpha = pi/2:
            |j=0> = |nu_mu>^(⊗N)
    """

    from math import comb

    n_qubits = len(qubits)

    required_qubits = int(np.ceil(np.log2(N + 1)))

    if n_qubits < required_qubits:
        raise ValueError(
            f"Register has {n_qubits} qubits, "
            f"but N={N} requires at least {required_qubits}."
        )

    amplitudes = np.zeros(2**n_qubits, dtype=complex)

    for j in range(N + 1):

        amplitudes[j] = (
            np.sqrt(comb(N, j))
            * np.cos(alpha)**j
            * np.sin(alpha)**(N - j)
        )

    amplitudes /= np.linalg.norm(amplitudes)

    qc.initialize(amplitudes, qubits)
# ============================================================
# Equality test:
#
# ancilla = 1 iff register == target
# ============================================================

def equality_test(qc, reg, anc, target):
    """
    Compute anc = 1 iff the binary register equals `target`.

    The register is returned to its original state.
    """

    n = len(reg)

    # Flip register bits where target has a 0.
    # This converts |target> -> |11...1>.
    for p, q in enumerate(reg):
        if ((target >> p) & 1) == 0:
            qc.x(q)

    # Multi-controlled X:
    # ancilla flips iff all register qubits are 1.
    qc.mcx(reg, anc)

    # Undo the bit flips.
    for p, q in enumerate(reg):
        if ((target >> p) & 1) == 0:
            qc.x(q)

# ============================================================
# Ancilla-controlled decrement
# ============================================================

def controlled_decrement(qc, reg, anc):
    """
    Decrement the binary register by one if anc == 1.

    Qubit ordering:
        reg[0] = least significant bit
        reg[-1] = most significant bit
    """

    n = len(reg)

    # Starting from LSB:
    # bit p flips if anc=1 and all lower bits are 0.
    #
    # We implement the condition by temporarily flipping
    # the lower bits, applying MCX, and undoing the flips.

    for p in reversed(range(n)):

        # Lower bits must originally be 0.
        for r in range(p):
            qc.x(reg[r])

        controls = [anc] + reg[:p]

        if p == 0:
            qc.cx(anc, reg[p])
        else:
            qc.mcx(controls, reg[p])

        for r in range(p):
            qc.x(reg[r])

# ============================================================
# Ancilla-controlled increment
# ============================================================

def controlled_increment(qc, reg, anc):
    """
    Increment the binary register by one if anc == 1.

    Qubit ordering:
        reg[0] = least significant bit
        reg[-1] = most significant bit

    This is the exact inverse of controlled_decrement().
    """

    n = len(reg)

    # Work from MSB to LSB.
    #
    # Bit p flips when anc = 1 AND all lower bits are 1.
    #
    # Processing from MSB -> LSB ensures that the lower-bit
    # conditions are evaluated before those lower bits are
    # themselves modified.

    for p in reversed(range(n)):

        controls = [anc] + reg[:p]

        if p == 0:
            qc.cx(anc, reg[p])
        else:
            qc.mcx(controls, reg[p])
# ============================================================
# Conditional register-bit flips for R_X^(1)
# ============================================================

# ============================================================
# Flip zero-bits of k
# ============================================================

def flip_zero_bits(qc, reg, k):
    """
    Flip every register qubit corresponding to a 0 bit in k.

    Qubit ordering:
        reg[0] = least significant bit
        reg[-1] = most significant bit
    """

    n = len(reg)

    for p in range(n):
        if ((k >> p) & 1) == 0:
            qc.x(reg[p])

# ============================================================
# Dicke-state R_X^(1)(k, k+1; theta)
# ============================================================

def dicke_rx(qc, reg, anc, k, theta):
    """
    Implement the ancilla-based

        R_X^(1)(k, k+1; theta)

    acting on the two-dimensional subspace

        {|k>, |k+1>}.

    The operation is

        exp[-i theta/2 ( |k+1><k| + |k><k+1| )]

    Qubit ordering:
        reg[0] = least significant bit
        reg[-1] = most significant bit

    anc:
        single ancilla qubit, initially |0>.
    """

    # --------------------------------------------------------
    # 1. Extract |k+1> into the ancilla
    # --------------------------------------------------------
    equality_test(qc, reg, k + 1, anc)

    # --------------------------------------------------------
    # 2. If anc = 1, decrement |k+1> -> |k>
    # --------------------------------------------------------
    controlled_decrement(qc, reg, anc)

    # --------------------------------------------------------
    # 3. Map |k> to |11...1>
    #
    #    This allows the subsequent multi-controlled RX
    #    to act on exactly the desired register state.
    # --------------------------------------------------------
    flip_zero_bits(qc, reg, k)

    # --------------------------------------------------------
    # 4. Rotate the ancilla, controlled by the register
    #
    #    Register = |11...1>  -->  RX(theta) on ancilla
    # --------------------------------------------------------
    qc.mcrx(theta, reg, anc)

    # --------------------------------------------------------
    # 5. Undo the register bit flips
    # --------------------------------------------------------
    flip_zero_bits(qc, reg, k)

    # --------------------------------------------------------
    # 6. If anc = 1, increment |k> -> |k+1>
    # --------------------------------------------------------
    controlled_increment(qc, reg, anc)

    # --------------------------------------------------------
    # 7. Uncompute the equality test
    # --------------------------------------------------------
    equality_test(qc, reg, k + 1, anc)

# ============================================================
# Two-register equality test
#
# ancilla = 1 iff
#
#     reg1 == target1
#     AND
#     reg2 == target2
#
# ============================================================

def equality_test_two(qc, reg1, reg2, anc, target1, target2):
    """
    Compute anc = 1 iff both registers equal their respective
    target values.

    The registers are returned to their original states.
    """

    # Flip zero-bits of target1
    for p, q in enumerate(reg1):
        if ((target1 >> p) & 1) == 0:
            qc.x(q)

    # Flip zero-bits of target2
    for p, q in enumerate(reg2):
        if ((target2 >> p) & 1) == 0:
            qc.x(q)

    # Ancilla flips iff every register qubit is 1
    controls = list(reg1) + list(reg2)
    qc.mcx(controls, anc)

    # Undo target1 flips
    for p, q in enumerate(reg1):
        if ((target1 >> p) & 1) == 0:
            qc.x(q)

    # Undo target2 flips
    for p, q in enumerate(reg2):
        if ((target2 >> p) & 1) == 0:
            qc.x(q)

# ============================================================
# Dicke-state R_X^(2)
#
# |j, k>  <-->  |j+1, k-1>
#
# ============================================================

def dicke_rx2(qc, reg1, reg2, anc, j, k, theta):
    """
    Implement the ancilla-based two-register rotation

        R_X^(2)(j,k; theta)

    coupling

        |j, k> <--> |j+1, k-1>.

    The operation is

        exp[-i theta/2 (
            |j+1,k-1><j,k|
          + |j,k><j+1,k-1|
        )].

    Qubit ordering:
        reg[0] = least significant bit
        reg[-1] = most significant bit

    anc:
        single ancilla qubit, initially |0>.
    """

    if k <= 0:
        raise ValueError(
            f"Invalid k={k}: need k >= 1 for |j+1,k-1>."
        )

    # --------------------------------------------------------
    # State to extract:
    #
    #     |j+1, k-1>
    #
    # After extraction, anc = 1 only for this state.
    # --------------------------------------------------------
    equality_test_two(
        qc,
        reg1,
        reg2,
        anc,
        j + 1,
        k - 1
    )

    # --------------------------------------------------------
    # Transform
    #
    # |j+1, k-1> --> |j, k>
    #
    # using the ancilla as the control.
    # --------------------------------------------------------
    controlled_decrement(qc, reg1, anc)
    controlled_increment(qc, reg2, anc)

    # --------------------------------------------------------
    # Map |j,k> to |11...1> in BOTH registers.
    # --------------------------------------------------------
    flip_zero_bits(qc, reg1, j)
    flip_zero_bits(qc, reg2, k)

    # --------------------------------------------------------
    # Rotate ancilla conditioned on BOTH registers.
    #
    # The only register state satisfying all controls is
    #
    #     |j,k>.
    # --------------------------------------------------------
    controls = list(reg1) + list(reg2)
    qc.mcrx(theta, controls, anc)

    # --------------------------------------------------------
    # Undo register bit flips.
    # --------------------------------------------------------
    flip_zero_bits(qc, reg1, j)
    flip_zero_bits(qc, reg2, k)

    # --------------------------------------------------------
    # Reverse the conditional transformation:
    #
    # |j,k> --> |j+1,k-1>
    #
    # --------------------------------------------------------
    controlled_increment(qc, reg1, anc)
    controlled_decrement(qc, reg2, anc)

    # --------------------------------------------------------
    # Uncompute equality test.
    # --------------------------------------------------------
    equality_test_two(
        qc,
        reg1,
        reg2,
        anc,
        j + 1,
        k - 1
    )

# ============================================================
# Dicke-state S_z S_z evolution
# ============================================================

# ============================================================
# Dicke-state S_z S_z evolution
# ============================================================

def add_dicke_szsz_evolution(
    qc,
    reg_e,
    reg_x,
    Ne,
    Nx,
    J,
    dt
):
    """
    Implement

        exp[-i 4 J dt S_e,z S_x,z]

    in the binary Dicke-state encoding.

    Dicke convention:

        |j> <-> |S=N/2, m=j-N/2>

    with

        j = sum_p 2^p b_p,

        b_p = (1-Z_p)/2.

    Therefore, for a register with n qubits,

        S_z
        = 1/2 [ (2^n - 1 - N)
                - sum_p 2^p Z_p ].

    The resulting S_z S_z evolution is decomposed into
    single-qubit RZ gates and cross-register RZZ gates.

    Qubit ordering:

        reg[0] = LSB
        reg[-1] = MSB
    """

    # --------------------------------------------------------
    # Register sizes
    # --------------------------------------------------------

    ne_bits = len(reg_e)
    nx_bits = len(reg_x)

    # --------------------------------------------------------
    # Binary-weight sums
    # --------------------------------------------------------

    We = 2**ne_bits - 1
    Wx = 2**nx_bits - 1

    # Constants appearing in S_z
    Ce = We - Ne
    Cx = Wx - Nx

    # --------------------------------------------------------
    # 4 J S_e,z S_x,z
    #
    # = J (Ce - Ae)(Cx - Ax)
    #
    # where
    #
    # Ae = sum_p 2^p Z_e,p
    # Ax = sum_q 2^q Z_x,q
    #
    # = J Ce Cx
    #   - J Ce Ax
    #   - J Cx Ae
    #   + J Ae Ax
    #
    # The first term is a global phase and is omitted.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Single-qubit Z terms
    #
    # -J Cx * Ae
    # -J Ce * Ax
    # --------------------------------------------------------

    for p, qe in enumerate(reg_e):

        theta = 2.0 * J * Cx * (2**p) * dt

        qc.rz(theta, qe)

    for q, qx in enumerate(reg_x):

        theta = 2.0 * J * Ce * (2**q) * dt

        qc.rz(theta, qx)

    # --------------------------------------------------------
    # Cross-register ZZ terms
    #
    # +J Ae Ax
    # --------------------------------------------------------

    for p, qe in enumerate(reg_e):

        for q, qx in enumerate(reg_x):

            theta = (
                2.0
                * J
                * (2**p)
                * (2**q)
                * dt
            )

            qc.rzz(theta, qe, qx)
# ============================================================
# Dicke-state interaction evolution
# ============================================================

def add_dicke_interaction_evolution(
    qc,
    reg_e,
    reg_x,
    anc,
    Ne,
    Nx,
    J,
    dt,
    n=1
):
    """
    Add n first-order Trotter steps for the neutrino-neutrino
    interaction in the Dicke-state encoding.

    Hamiltonian:

        H_nunu = 4 J S_e . S_x

    up to the constant term

        2 J (S_e^2 + S_x^2),

    which contributes only a global phase and is therefore
    omitted.

    Using

        S_e . S_x
        =
        S_e,z S_x,z
        + 1/2 (S_e,+ S_x,- + S_e,- S_x,+),

    we implement

        H_nunu
        =
        4 J S_e,z S_x,z
        +
        2 J (S_e,+ S_x,- + S_e,- S_x,+).

    The diagonal S_z S_z part is implemented with RZZ gates.

    The off-diagonal part is decomposed into the Dicke-state
    rotations

        R_X^(2)(j_e, j_x; theta),

    coupling

        |j_e, j_x>
            <-->
        |j_e+1, j_x-1>.

    Register convention:

        reg_e[0] = LSB
        reg_e[-1] = MSB

        reg_x[0] = LSB
        reg_x[-1] = MSB

    Dicke-state convention:

        |j_e> <-> |S_e, m_e = j_e - N_e/2>

        |j_x> <-> |S_x, m_x = j_x - N_x/2>

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to which the evolution is appended.

    reg_e : QuantumRegister/list
        Electron-neutrino Dicke register.

    reg_x : QuantumRegister/list
        x-neutrino Dicke register.

    anc : Qubit
        Ancilla used by the R_X^(2) construction.

    Ne, Nx : int
        Number of electron and x neutrinos.

    J : float
        Neutrino-neutrino coupling.

    dt : float
        Trotter time step.

    n : int
        Number of first-order Trotter steps.
    """

    # --------------------------------------------------------
    # Required Dicke-register sizes
    # --------------------------------------------------------

    n_e_required = int(np.ceil(np.log2(Ne + 1)))
    n_x_required = int(np.ceil(np.log2(Nx + 1)))

    if len(reg_e) < n_e_required:
        raise ValueError(
            f"Electron register has {len(reg_e)} qubits, "
            f"but Ne={Ne} requires at least {n_e_required}."
        )

    if len(reg_x) < n_x_required:
        raise ValueError(
            f"X-neutrino register has {len(reg_x)} qubits, "
            f"but Nx={Nx} requires at least {n_x_required}."
        )

    if Ne < 1 or Nx < 1:
        raise ValueError(
            "Ne and Nx must both be positive."
        )

    if n < 1:
        raise ValueError(
            "Number of Trotter steps n must be at least 1."
        )

    # --------------------------------------------------------
    # First-order Trotter evolution
    # --------------------------------------------------------

    for _ in range(n):

        # ====================================================
        # 1. S_e,z S_x,z
        #
        #     exp[-i 4 J dt S_e,z S_x,z]
        #
        # Implemented directly by add_dicke_szsz_evolution().
        # ====================================================

        add_dicke_szsz_evolution(
            qc,
            reg_e,
            reg_x,
            Ne,
            Nx,
            J,
            dt
        )

        # ====================================================
        # 2. Off-diagonal interaction
        #
        #     2 J (S_e,+ S_x,- + S_e,- S_x,+)
        #
        # We use
        #
        #     |j_e, j_x>
        #          <-->
        #     |j_e+1, j_x-1>
        #
        # for
        #
        #     j_e = 0,...,Ne-1
        #     j_x = 1,...,Nx.
        #
        # ====================================================

        for j_e in range(Ne):

            for j_x in range(1, Nx + 1):

                # ------------------------------------------------
                # Dicke ladder coefficients
                #
                # S_e,+ |j_e>
                #     = sqrt[(Ne-j_e)(j_e+1)] |j_e+1>
                #
                # S_x,- |j_x>
                #     = sqrt[j_x(Nx-j_x+1)] |j_x-1>
                # ------------------------------------------------

                coeff = np.sqrt(
                    (Ne - j_e)
                    * (j_e + 1)
                    * j_x
                    * (Nx - j_x + 1)
                )

                # ------------------------------------------------
                # Hamiltonian matrix element:
                #
                #     2 J * coeff
                #
                # R_X^(2) is defined as
                #
                # exp[-i theta/2 (|a><b| + |b><a|)]
                #
                # Therefore
                #
                #     theta/2 = 2 J dt * coeff
                #
                # giving
                #
                #     theta = 4 J dt * coeff
                # ------------------------------------------------

                theta = 4.0 * J * dt * coeff

                dicke_rx2(
                    qc,
                    reg_e,
                    reg_x,
                    anc,
                    j_e,
                    j_x,
                    theta
                )


# In[7]:


# ============================================
# CELL 4: Raffelt--Sigl mean-field solver
# ============================================

# Pauli matrices
sigma_0 = np.array([[1, 0],  [0,  1]], dtype=complex)
sigma_1 = np.array([[0, 1],  [1,  0]], dtype=complex)
sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_3 = np.array([[1, 0],  [0, -1]], dtype=complex)


def P_osc_RS(t_table, theta, omega, lam, J, initial_flavors=None, alpha=None):
    """
    Raffelt--Sigl two-flavor polarization-vector evolution.

    Parameters
    ----------
    t_table : array
        Sampled times.
    theta : float
        Vacuum mixing angle.
    omega : array
        Vacuum frequencies for all modes.
    lam : float
        Matter strength.
    J : array
        Coupling matrix.
    initial_flavors : array
        Entries may be:
            'e'  : |nu_e>
            'x'  : |nu_mu>
            'mu' : |nu_mu>
            'a'  : cos(alpha)|nu_e> + sin(alpha)|nu_mu>
    alpha : float
        Superposition angle for every mode labeled 'a'.
    """
    n_modes = omega.size

    u = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    b = 0.5 * np.diag([-1, 1])
    b = u @ b @ u.T
    l = np.diag([1, 0])

    Ee = np.array([0.0, 0.0,  1.0])
    Emu = np.array([0.0, 0.0, -1.0])

    B = np.real(np.array([np.trace(b @ sigma_1),
                          np.trace(b @ sigma_2),
                          np.trace(b @ sigma_3)]))
    L = np.real(np.array([np.trace(l @ sigma_1),
                          np.trace(l @ sigma_2),
                          np.trace(l @ sigma_3)]))

    ini_state = np.where(omega > 0, Ee[:, None], Emu[:, None])

    if initial_flavors is not None:
        initial_flavors = np.asarray(initial_flavors)

        for f in initial_flavors:
            if f not in ["e", "x", "mu", "a"]:
                raise ValueError(f"Unknown flavor label: {f}")
            if f == "a" and alpha is None:
                raise ValueError("Flavor label 'a' requires alpha.")

        ini_state[:, initial_flavors == "e"] = Ee[:, None]
        ini_state[:, initial_flavors == "x"] = Emu[:, None]
        ini_state[:, initial_flavors == "mu"] = Emu[:, None]
        ini_state[:, initial_flavors == "a"] = np.array(
            [np.sin(2.0 * alpha), 0.0, np.cos(2.0 * alpha)]
        )[:, None]

    def rhs(t, P_flat):
        """
        dP_k/dt = H_k x P_k,
        H_k = omega_k B + lambda L + sum_j J_kj P_j.
        """
        res = np.zeros(3 * n_modes)
        PP = np.reshape(P_flat, (n_modes, 3)).T

        for k in range(n_modes):
            H_k = omega[k] * B + lam * L + J * np.sum(PP, axis=1)
            res[3*k:3*k+3] = np.cross(H_k, PP[:, k])

        return res

    return integ.solve_ivp(
        rhs,
        (t_table[0], t_table[-1]),
        ini_state.T.flatten(),
        t_eval=t_table,
    )


# In[8]:


# =====================================================
# CELL 5: Many-body Qiskit solver for Ne + Nx(alpha)
# =====================================================

def prepare_initial_circuit(Ne, Nx, alpha):
    """
    Prepare the initial many-body state:

        first Ne qubits: |nu_e> = |0>
        next  Nx qubits: |nu_x(alpha)> = cos(alpha)|0> + sin(alpha)|1>

    RY(2 alpha)|0> = cos(alpha)|0> + sin(alpha)|1>.

    For alpha = pi/2, the Nx group is exactly |1>, reproducing
    the old 1 nu_e + 15 nu_mu setup when Ne=1, Nx=15.
    """
    N = Ne + Nx
    qc0 = QuantumCircuit(N, N)

    # First Ne qubits are |nu_e> = |0>, so no gate is needed.
    for q in range(Ne, N):
        qc0.ry(2.0 * alpha, q)

    return qc0


def run_many_body_qiskit(alpha, Ne, Nx, J, times, dt, shots, backend,
                         use_vacuum=False, bx=0.0, by=0.0, bz=0.0,
                         verbose=True):
    """
    Run the many-body Qiskit simulation for a given alpha.

    Output
    ------
    P_bit1[q, k] :
        Probability that qubit q is measured as |1> at time times[k].

    For the first Ne qubits, which start as |nu_e>=|0>,
        P_bit1 = P(nu_e -> nu_mu).

    The plotted observable is the Ne-average:
        <P(nu_e -> nu_mu)> over the initially electron-flavor group.
    """
    N = Ne + Nx
    qc0 = prepare_initial_circuit(Ne, Nx, alpha)

    P_bit1 = [[] for _ in range(N)]

    for t in times:
        n = int(round(t / dt))
        qc = qc0.copy()

        #if use_vacuum:
            #add_vacuum_evolution(qc, n=n, bx=bx, by=by, bz=bz, N=N)

        add_interaction_evolution(qc, n=n, J=J, N=N, dt=dt)

        qc.measure(range(N), range(N))

        tqc = transpile(qc, backend, optimization_level=0)
        counts = backend.run(tqc, shots=shots).result().get_counts()

        p1_t = [0.0] * N

        for bitstring, c in counts.items():
            prob = c / shots

            for q in range(N):
                # Qiskit displays bitstrings with the highest classical bit first.
                bit_q = int(bitstring[-1 - q])
                if bit_q == 1:
                    p1_t[q] += prob

        for q in range(N):
            P_bit1[q].append(p1_t[q])

        if verbose:
            print(
                f"alpha={alpha:.6f}, t={t:8.3f}: "
                f"<P_e_to_mu>_MB = {np.mean(p1_t[:Ne]):.5f}"
            )

    P_bit1 = np.array(P_bit1)

    return {
        "P_bit1": P_bit1,
        "P_e_to_mu_avg": np.mean(P_bit1[:Ne, :], axis=0),
    }


# In[9]:


# =====================================================
# CELL 6: RS solver and combined alpha-case runner
# =====================================================

def run_rs_mean_field(alpha, Ne, Nx, J, times,
                      use_vacuum=False, use_matter=False):
    """
    Run the Raffelt--Sigl mean-field evolution for the same initial state.

    Default:
        omega = 0,
        lambda = 0,

    so this solves the self-interaction-only mean-field problem, matching
    the old e1_x15 comparison.
    """
    N = Ne + Nx

    if use_vacuum:
        omega = np.ones(N) * omega1
        theta_rs = theta_vac
    else:
        omega = np.zeros(N)
        theta_rs = 0.0

    lam = 1.0 if use_matter else 0.0

    initial_flavors = np.array(["e"] * Ne + ["a"] * Nx)

    sol = P_osc_RS(
        times,
        theta_rs,
        omega,
        lam,
        J,
        initial_flavors=initial_flavors,
        alpha=alpha,
    )

    P_all = sol.y
    P = P_all.reshape(N, 3, -1)
    Pz = P[:, 2, :]

    # P(nu_mu) = (1 - Pz)/2 for each mode.
    P_mu = 0.5 * (1.0 - Pz)

    return {
        "sol": sol,
        "P": P,
        "Pz": Pz,
        "P_mu": P_mu,
        "P_e_to_mu_avg": np.mean(P_mu[:Ne, :], axis=0),
    }


def run_alpha_case(alpha, verbose=True):
    """
    Run both many-body Qiskit and RS mean-field for one alpha.
    """
    N = Ne + Nx
    J = 1

    mb = run_many_body_qiskit(
        alpha=alpha,
        Ne=Ne,
        Nx=Nx,
        J=J,
        times=times,
        dt=dt,
        shots=shots,
        backend=backend,
        use_vacuum=use_vacuum,
        bx=bx,
        by=by,
        bz=bz,
        verbose=verbose,
    )

    rs = run_rs_mean_field(
        alpha=alpha,
        Ne=Ne,
        Nx=Nx,
        J=J,
        times=times,
        use_vacuum=use_vacuum,
        use_matter=use_matter,
    )

    return {
        "alpha": alpha,
        "Ne": Ne,
        "Nx": Nx,
        "many_body": mb,
        "rs": rs,
    }


# In[10]:


# =====================================================
# CELL 7: Plotting helpers
# =====================================================

results = {}

# Choose the x-axis used in all plots.
# Options:
#     "t"    : physical notebook time
#     "t_mu" : dimensionless t*mu
plot_time_axis = "t"


def get_plot_time():
    if plot_time_axis == "t":
        return times, r"Total time"
    elif plot_time_axis == "t_mu":
        return times , r"$t\mu$"
    else:
        raise ValueError("plot_time_axis must be either 't' or 't_mu'.")


def run_and_plot_alpha(alpha, alpha_label, fig_no=None, verbose=True, show_plot=True):
    """
    Run one alpha case and optionally plot the average conversion probability
    of the initially electron-flavor group.
    """
    key = alpha_label
    result = run_alpha_case(alpha, verbose=verbose)
    results[key] = result

    if show_plot:
        P_mb = result["many_body"]["P_e_to_mu_avg"]
        P_rs = result["rs"]["P_e_to_mu_avg"]

        x, xlabel = get_plot_time()

        plt.figure(figsize=(9, 5.5))

        plt.plot(
            x,
            P_mb,
            marker="o",
            linestyle="dashed",
            label=rf"Many-body Qiskit, $\alpha={alpha_label}$",
        )

        plt.plot(
            x,
            P_rs,
            linestyle="-",
            linewidth=2,
            label=rf"RS mean field, $\alpha={alpha_label}$",
        )

        plt.xlabel(xlabel)
        plt.ylabel(r"$\langle P(\nu_e\to\nu_\mu)\rangle_{N_e}$")

        if fig_no is None:
            plt.title(
                rf"$N_e={Ne}$, $N_x={Nx}$, "
                rf"$\nu_x=\cos\alpha\,\nu_e+\sin\alpha\,\nu_\mu$"
            )
        else:
            plt.title(
                rf"Fig. {fig_no}: $N_e={Ne}$, $N_x={Nx}$, "
                rf"$\nu_x=\cos\alpha\,\nu_e+\sin\alpha\,\nu_\mu$"
            )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result


# In[11]:


# =====================================================
# CELL 5: Dicke-state Qiskit solver for Ne + Nx(alpha)
# =====================================================

def run_dicke_qiskit(alpha, Ne, Nx, J, times, dt, shots, backend,
                     verbose=True):
    """
    Run the Dicke-state Qiskit simulation for a given alpha.

    Initial state:

        electron register:
            |nu_e>^(⊗Ne)
            <-> |j_e = Ne>
            <-> |S_e = Ne/2, m_e = +Ne/2>

        x-neutrino register:
            (cos(alpha)|nu_e> + sin(alpha)|nu_x>)^(⊗Nx)

            represented in the Dicke basis.

    Dicke convention:

        |j> <-> |S=N/2, m=j-N/2>

    Therefore:

        j = N      -> all nu_e
        j = 0      -> all nu_x

    Observable:

        P(nu_e -> nu_x)
            = 1 - <j_e>/Ne

    where j_e is the measured electron-register Dicke
    occupation number.
    """

    # -----------------------------------------------------
    # Dicke-register sizes
    # -----------------------------------------------------

    n_e = int(np.ceil(np.log2(Ne + 1)))
    n_x = int(np.ceil(np.log2(Nx + 1)))

    # -----------------------------------------------------
    # Quantum registers
    # -----------------------------------------------------

    reg_e = QuantumRegister(n_e, "e")
    reg_x = QuantumRegister(n_x, "x")
    anc = QuantumRegister(1, "anc")

    # -----------------------------------------------------
    # Classical register
    #
    # c[0 : n_e]       -> electron Dicke register
    # c[n_e : n_e+n_x] -> x-neutrino Dicke register
    # -----------------------------------------------------

    creg = ClassicalRegister(n_e + n_x, "c")

    qc0 = QuantumCircuit(
        reg_e,
        reg_x,
        anc,
        creg
    )

    # -----------------------------------------------------
    # Electron group:
    #
    # All neutrinos initially in |nu_e> = |0>.
    #
    # With our Dicke convention:
    #
    #     |nu_e>^(⊗Ne)
    #         <-> |j_e = Ne>
    #         <-> m_e = +Ne/2
    #
    # -----------------------------------------------------

    prepare_dicke_state(
        qc0,
        reg_e,
        Ne,
        m=Ne / 2
    )

    # -----------------------------------------------------
    # X-neutrino group:
    #
    # (cos(alpha)|nu_e> + sin(alpha)|nu_x>)^(⊗Nx)
    #
    # Since
    #
    #     |nu_e> = |0>
    #     |nu_x> = |1>,
    #
    # the Dicke-state amplitudes are
    #
    #     sqrt[C(Nx,j)]
    #     cos(alpha)^(Nx-j)
    #     sin(alpha)^j
    #
    # -----------------------------------------------------

    prepare_dicke_superposition(
        qc0,
        reg_x,
        Nx,
        alpha
    )

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    j_e_avg = []
    P_e_to_x_avg = []
    counts_all = []

    # -----------------------------------------------------
    # Time evolution
    # -----------------------------------------------------

    for t in times:

        n = int(round(t / dt))

        qc = qc0.copy()

        # -------------------------------------------------
        # Add n first-order Dicke Trotter steps
        # -------------------------------------------------

        if n > 0:

            add_dicke_interaction_evolution(
                qc,
                reg_e,
                reg_x,
                anc[0],
                Ne,
                Nx,
                J,
                dt,
                n=n
            )

        # -------------------------------------------------
        # Measure the Dicke registers
        # -------------------------------------------------

        qc.measure(reg_e, range(n_e))
        qc.measure(reg_x, range(n_e, n_e + n_x))

        # -------------------------------------------------
        # Transpile and execute
        # -------------------------------------------------

        tqc = transpile(
            qc,
            backend,
            optimization_level=0
        )

        counts = backend.run(
            tqc,
            shots=shots
        ).result().get_counts()

        counts_all.append(counts)

        # -------------------------------------------------
        # Calculate <j_e>
        #
        # Electron register is mapped to the LOWEST
        # classical bits:
        #
        #     c[0], ..., c[n_e-1]
        #
        # Qiskit displays the highest classical bit first,
        # so the electron register is the RIGHTMOST n_e
        # bits of the displayed bitstring.
        # -------------------------------------------------

        j_e_t = 0.0

        for bitstring, count in counts.items():

            prob = count / shots

            electron_bits = bitstring[-n_e:]

            j_e = int(electron_bits, 2)

            j_e_t += prob * j_e

        # -------------------------------------------------
        # Conversion probability
        #
        # j_e counts the number of remaining nu_e states
        # in the initially electron-flavor group.
        #
        # Therefore:
        #
        #     P(nu_e -> nu_x)
        #       = 1 - <j_e>/Ne
        # -------------------------------------------------

        P_t = 1.0 - j_e_t / Ne

        j_e_avg.append(j_e_t)
        P_e_to_x_avg.append(P_t)

        if verbose:

            print(
                f"alpha={alpha:.6f}, "
                f"t={t:8.3f}: "
                f"<P_e_to_mu>_Dicke = {P_t:.5f}"
            )

    # -----------------------------------------------------
    # Convert to numpy arrays
    # -----------------------------------------------------

    j_e_avg = np.array(j_e_avg)
    P_e_to_x_avg = np.array(P_e_to_x_avg)

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {
        "j_e_avg": j_e_avg,
        "P_e_to_x_avg": P_e_to_x_avg,
        "counts": counts_all,
    }


# In[ ]:





# ## Figures for different \(\alpha\)
# 
# Each cell below runs one \(\alpha\) case and produces one figure.
# 
# The plotted observable is the average conversion probability of the initially electron-flavor group:
# 
# \[
# \left\langle P(\nu_e\rightarrow \nu_\mu)\right\rangle_{N_e}.
# \]

# In[ ]:





# In[12]:


import numpy as np
from scipy.special import gammaln
from sympy.physics.wigner import clebsch_gordan
from sympy import S as sympy_S


def cg_coefficient(j1, m1, j2, m2, J, M):
    """
    Clebsch-Gordan coefficient

        <j1,m1; j2,m2 | J,M>

    using the SymPy convention.
    """
    return float(
        clebsch_gordan(
            sympy_S(j1),
            sympy_S(j2),
            sympy_S(J),
            sympy_S(m1),
            sympy_S(m2),
            sympy_S(M)
        )
    )


def collective_oscillation_probability(N1, N2, alpha, t, lam=1.0):
    """
    Collective nu_e -> nu_mu oscillation probability for a two-beam system.

    Parameters
    ----------
    N1 : int
        Number of initial nu_e neutrinos in beam 1.

    N2 : int
        Number of neutrinos in beam 2.

    alpha : float
        Mixing angle appearing in
            |x> = cos(alpha)|e> + sin(alpha)|mu>.

    t : float or array-like
        Evolution time.

    lam : float, optional
        Interaction strength lambda.

    Returns
    -------
    P_emu : float or ndarray
        Collective nu_e -> nu_mu conversion probability
        for a neutrino in beam 1.
    """

    # ------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------

    if N1 < 1 or N2 < 1:
        raise ValueError("N1 and N2 must be positive integers.")

    if not isinstance(N1, (int, np.integer)):
        raise TypeError("N1 must be an integer.")

    if not isinstance(N2, (int, np.integer)):
        raise TypeError("N2 must be an integer.")

    # ------------------------------------------------------------
    # Spin quantum numbers of the two beams
    # ------------------------------------------------------------

    j1 = N1 / 2
    j2 = N2 / 2

    # Maximum total spin
    Jmax = j1 + j2

    # Make t an array so that the same code works for
    # scalar or array-valued t
    t = np.asarray(t, dtype=float)

    # Total expected number of muon neutrinos in beam 1
    Nmu = np.zeros_like(t, dtype=float)

    c = np.cos(alpha)
    s = np.sin(alpha)

    # ============================================================
    # SPECIAL CASE: alpha = pi/2
    #
    # |x> = |mu>
    #
    # Therefore only i = N2 contributes.
    # ============================================================

    if np.isclose(alpha, np.pi / 2):

        i = N2

        # Total magnetic quantum number
        M = Jmax - i

        # Allowed minimum total spin
        Smin = max(M, abs(j1 - j2))

        # --------------------------------------------------------
        # j = number of nu_e -> nu_mu conversions in beam 1
        # --------------------------------------------------------

        for j in range(1, min(i, N1) + 1):

            # Initial product state:
            #
            # |j1, j1> |j2, j2-i>
            #
            m1_initial = j1
            m2_initial = j2 - i

            # Final product state:
            #
            # |j1, j1-j> |j2, j2-i+j>
            #
            m1_final = j1 - j
            m2_final = j2 - i + j

            # Real and imaginary parts of transition amplitude
            real_part = np.zeros_like(t, dtype=float)
            imag_part = np.zeros_like(t, dtype=float)

            # ----------------------------------------------------
            # Sum over total spin S
            # ----------------------------------------------------

            S_values = np.arange(Smin, Jmax + 1, 1)

            for Stotal in S_values:

                # CG coefficient of the initial state
                C_initial = cg_coefficient(
                    j1, m1_initial,
                    j2, m2_initial,
                    Stotal, M
                )

                # CG coefficient of the final state
                C_final = cg_coefficient(
                    j1, m1_final,
                    j2, m2_final,
                    Stotal, M
                )

                coefficient = C_initial * C_final

                # Energy eigenvalue
                E_S = 2 * lam * Stotal * (Stotal + 1)

                # Evolution e^{-i E_S t}
                real_part += coefficient * np.cos(E_S * t)
                imag_part -= coefficient * np.sin(E_S * t)

            # Transition probability
            P_ij = real_part**2 + imag_part**2

            # Since the weight of the i=N2 sector is exactly 1,
            # directly add j * P_ij.
            Nmu += j * P_ij

        # Probability for one neutrino in beam 1
        return Nmu / N1

    # ============================================================
    # GENERAL CASE: alpha != pi/2
    # ============================================================

    for i in range(1, N2 + 1):

        # --------------------------------------------------------
        # Total magnetic quantum number
        # --------------------------------------------------------

        M = Jmax - i

        # --------------------------------------------------------
        # Weight of the i-th sector
        #
        # C(N2, i)
        # * (cos^2 alpha)^(N2-i)
        # * (sin^2 alpha)^i
        #
        # Calculate using logarithms for numerical stability.
        # --------------------------------------------------------

        if c != 0:
            log_weight = (
                gammaln(N2 + 1)
                - gammaln(i + 1)
                - gammaln(N2 - i + 1)
                + 2 * (N2 - i) * np.log(abs(c))
            )
        else:
            # If cos(alpha)=0, this sector has zero weight
            # for every i < N2.
            log_weight = -np.inf

        if s != 0:
            log_weight += 2 * i * np.log(abs(s))
        elif i > 0:
            log_weight = -np.inf

        weight = np.exp(log_weight)

        # --------------------------------------------------------
        # Allowed total-spin range
        #
        # S = Jmax, Jmax-1, ..., Smin
        #
        # Smin = max(Jmax-i, |j1-j2|)
        # --------------------------------------------------------

        Smin = max(M, abs(j1 - j2))

        # --------------------------------------------------------
        # j = number of nu_e -> nu_mu conversions
        # --------------------------------------------------------

        for j in range(1, min(i, N1) + 1):

            # Initial product state
            m1_initial = j1
            m2_initial = j2 - i

            # Final product state
            m1_final = j1 - j
            m2_final = j2 - i + j

            # Real and imaginary parts of transition amplitude
            real_part = np.zeros_like(t, dtype=float)
            imag_part = np.zeros_like(t, dtype=float)

            # ----------------------------------------------------
            # Sum over total spin S
            # ----------------------------------------------------

            S_values = np.arange(Smin, Jmax + 1, 1)

            for Stotal in S_values:

                # CG coefficient for initial product state
                C_initial = cg_coefficient(
                    j1, m1_initial,
                    j2, m2_initial,
                    Stotal, M
                )

                # CG coefficient for final product state
                C_final = cg_coefficient(
                    j1, m1_final,
                    j2, m2_final,
                    Stotal, M
                )

                coefficient = C_initial * C_final

                # Energy eigenvalue
                E_S = 2 * lam * Stotal * (Stotal + 1)

                # Evolution e^{-i E_S t}
                real_part += coefficient * np.cos(E_S * t)
                imag_part -= coefficient * np.sin(E_S * t)

            # Transition probability
            P_ij = real_part**2 + imag_part**2

            # Add weighted contribution to expected number
            # of muon neutrinos in beam 1
            Nmu += weight * j * P_ij

    # ------------------------------------------------------------
    # Convert expected number of muon neutrinos to the
    # conversion probability for one neutrino in beam 1
    # ------------------------------------------------------------

    P_emu = Nmu / N1

    return P_emu


# In[ ]:


alpha_cases = [
    (np.pi/2, r"\pi/2"),
    (np.pi/3, r"\pi/3"),
    (np.pi/4, r"\pi/4"),
    (np.pi/6, r"\pi/6"),
]

P_mb = {}
P_dicke = {}
P_emu = {}

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for ax, (alpha, alpha_label), color in zip(
    axes, alpha_cases, colors
):

    # =====================================================
    # Conventional many-body quantum simulation
    # =====================================================

    result_mb = run_and_plot_alpha(
        alpha,
        alpha_label,
        show_plot=False,
    )

    P_mb[alpha_label] = (
        result_mb["many_body"]["P_e_to_mu_avg"]
    )

    # =====================================================
    # Dicke-state quantum simulation
    # =====================================================

    result_dicke = run_dicke_qiskit(
        alpha=alpha,
        Ne=Ne,
        Nx=Nx,
        J=J,
        times=times,
        dt=dt,
        shots=shots,
        backend=backend,
        verbose=True,
    )

    P_dicke[alpha_label] = (
        result_dicke["P_e_to_x_avg"]
    )

    # =====================================================
    # Analytical result
    # =====================================================
    t = np.linspace(0, 10, 100)
    P = collective_oscillation_probability(
        N1=Ne,
        N2=Nx,
        alpha=alpha,
        t=t,
        lam=J
    )

    P_emu[alpha_label] = P

    # =====================================================
    # Plot
    # =====================================================

    x, xlabel = get_plot_time()

    # Conventional QS: points only
    ax.plot(
        x,
        P_mb[alpha_label],
        "o",
        color=color,
        linestyle="None",
        label="Conventional QS"
    )

    # Dicke QS: points only
    ax.plot(
        x,
        P_dicke[alpha_label],
        "s",
        color=color,
        linestyle="None",
        label="Dicke QS"
    )

    # Analytical: continuous line
    ax.plot(
        t,
        P_emu[alpha_label],
        "-",
        lw=2,
        color=color,
        label="Analytical"
    )

fig.supxlabel(xlabel)
fig.supylabel(
r"$\langle P(\nu_e\to\nu_\mu)\rangle_{N_e}$"
)
fig.suptitle(
rf"$N_e={Ne}$, $N_x={Nx}$",
fontsize=15
)
plt.tight_layout(
rect=[0, 0, 1, 0.96]
)
plt.show()


# ## 
