Hello! Please make sure you do not publish this repository publicly, because it contains our IBM tokens.

# Environment Setup

The instructions and environment scripts are prepared for CCDB Fir server. It should in theory be generalizable to other environments, but some modules available on CCDB servers may need to be installed manually.

## Setting up a virtual environment

First load python
    
    module load python/3.11

After cloning the repository, create a virtual environment in the project root level.

Follow https://docs.alliancecan.ca/wiki/Python

or you can directly use the `setup.sh` in the included scripts.

If you are not planning to run the tensor network scripts, you can install the packages in requirements.txt. Otherwise, use requirements_tn.txt, and follow the instructions below for KaHyPar.

## Activating environment

The project is designed in such a way that the different tasks (for example, dicke method, tensor network, real ibm computer) shares the same environment to reduce complexity.

You can find the environment activation script env.sh in the project root level.

When trying to activate the environment, first do

    cd [subfolder]

For example, if you are working with Dicke state simulations,

    cd dicke

Then, run the environment script

    source ../env.sh

You should be all set. You can run scripts with

    python [script_name.py] --[flags as defined in script]


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

### Installing KaHyPar (on Compute Canada)

KaHyPar is a package that is heavily used by quimb + cotengra for optimizing contraction path on tensor networks, the package is found here https://github.com/kahypar/kahypar.

Unfortunately, pip install for kahypar does not work on Graham (I don't know why), therefore, here is a guide to installation.

First load the modules

    module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

This was from the qiskit guide, just to be safe. Then load boost

    module load boost

This is a necessary dependency for KaHyPar. Now, source into the virtual environment, for example

    source ~/ENV/bin/activate

Now follow the full instruction for installing KaHyPar python interfact on https://github.com/kahypar/kahypar#the-python-interface, stop before the last step 5 of copying

Run on terminal

    python -c "import site; print(site.getsitepackages()[0])"

This should give you a path like

    /home/<username>/ENV/lib/python3.11/site-packages

Now make sure you are still in the correct directory,

    <parent dir>/kahypar/build/python

where the kahypar.<version>.so is present. Finally put

    cp kahypar.<version>.so ~/ENV/lib/python3.11/site-packages/kahypar.so

This should complete installation. To verify install, type

    python -c "import kahypar; print(kahypar)"

If this does not give you a ModuleNotFoundError, then KaHyPar should be successfully installed.


