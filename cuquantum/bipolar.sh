#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=64G               # memory per node
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=1

source ../env.sh

nvidia-smi

python cuquantum_bipolar.py "$@"