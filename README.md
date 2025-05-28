Hello! Please make sure you do not publish this repository publicly, because it contains our IBM tokens.

# Misc. notes

## Conversion into .py scripts

First install nbconvert (activate virtual environment first)

    pip install nbconvert

Use the following command to convert notebooks to scripts

    jupyter nbconvert --to script [path/to/script.ipynb]

## Running AerSimulator with GPU

See the documents below for reference:

https://qiskit.github.io/qiskit-aer/getting_started.html

https://docs.quantum.ibm.com/api/qiskit/0.38/qiskit_aer.AerSimulator


The qiskit-aer-gpu package is funtionally the same as the normal qiskit-aer package. It replaces the said package and adds gpu support. Therefore, it is only necessary (and possible) to install one of the two versions.

Run

    pip install qiskit-aer-gpu

Specifying running the AerSimulator on GPU can be done by adding an extra kwarg at initialization:

    simulator = AerSimulator(device="GPU")

Methods that run on the GPU:

- statevector
- density_matrix
- unitary

It is possible to check the list of available devices by using

    AerSimulator.available_devices()

### cuStateVec

accelerating by cuStateVec library of NVIDIA can be enabled, the flag is ignored if support does not exist (thus should be safe to use)

    cuStateVec_enable = True

### Using Qiskit on Compute Canada

Refer to https://docs.alliancecan.ca/wiki/Qiskit/fr
