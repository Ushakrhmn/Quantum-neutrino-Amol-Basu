from qiskit_ibm_runtime import QiskitRuntimeService

for j in QiskitRuntimeService().jobs():
    print(j)
    try:
        j.cancel()
    except Exception as e:
        pass