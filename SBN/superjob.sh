#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=64G               # memory per node
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=xyz.yu@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2

module load StdEnv/2023 gcc python/3.11 symengine/0.11.2 boost arch/avx512

nvidia-smi

source ~/ENV/bin/activate

python baseline_tensor.py
