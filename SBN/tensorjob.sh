#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=64G               # memory per node
#SBATCH --time=0-24:00:00
#SBATCH --mail-user=xyz.yu@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1

module load StdEnv/2023 gcc python/3.11 symengine/0.11.2 boost

nvidia-smi

source ~/ENV/bin/activate

python baseline_tensor.py
