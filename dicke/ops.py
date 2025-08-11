"""
Operators from https://arxiv.org/pdf/1904.07358
"""
import qiskit as qk
import numpy as np

def two_qubit(circuit: qk.QuantumCircuit, n: int) -> None:
  control = n - 2
  target = n - 1
  circuit.cx(control, target)
  theta = 2 * np.arccos(np.sqrt(1 / n))
  circuit.cry(theta, target, control)
  circuit.cx(control, target)

def three_qubit(circuit: qk.QuantumCircuit, n: int, l: int) -> None:
  circuit.cx(n - l - 1, n - 1)
  theta = 2 * np.arccos(np.sqrt(l / n))
  circuit.mcry(theta, [n - 1, n - l], n - l - 1, None)
  circuit.cx(n - l - 1, n - 1)

def scs(n: int, k: int, circuit: qk.QuantumCircuit) -> None:

  if n == 2 and k == 1:
    two_qubit(circuit, n)
    return
  
  two_qubit(circuit, n)
  for l in range(2, k+1):
    three_qubit(circuit, n, l)
  
  return

def u(n: int, k: int, circuit: qk.QuantumCircuit) -> None:

  if n < k:
    raise ValueError(f"n must be greater than or equal to k, but n={n} and k={k}")

  # base case
  if n == 1 and k == 1:
    # identity
    return

  if n == k:
    scs(k, k-1, circuit) # SCS_{k, k-1}
    u(k-1, k-1, circuit) # U_{k-1, k-1} tensor I
    return

  scs(n, k, circuit) # SCS_{n, k}
  u(n-1, k, circuit) # U_{n-1, k}
  return

if __name__ == "__main__":
  n = 5
  circuit = qk.QuantumCircuit(n)
  u(5, 3, circuit)

  import matplotlib.pyplot as plt

  fig = circuit.draw(output='mpl')
  fig.savefig("scs_circuit.png")
  plt.close(fig)
