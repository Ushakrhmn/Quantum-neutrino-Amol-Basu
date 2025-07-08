#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=8G               # memory per node
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=1

source env.sh

nvidia-smi

python simple_circuit.py