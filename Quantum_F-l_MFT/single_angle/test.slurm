#!/bin/bash
#SBATCH --account=def-nilic
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

source /home/urahaman/scratch/Quantum-neutrino-Amol-Basu/env.sh

echo "===== PYTHON ====="
which python
python --version

echo "===== BEFORE PYTHON ====="

python -c 'print("Plain Python works")'

echo "===== QISKIT ====="

python -c 'import qiskit; print("Qiskit import works")'

echo "===== AER ====="

python -c 'from qiskit_aer import AerSimulator; print("Aer import works")'

echo "===== DONE ====="
