#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-03:00
./program                         # you can use 'nvidia-smi' for a test