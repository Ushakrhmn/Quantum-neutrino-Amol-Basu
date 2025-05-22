#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-03:00:00
#SBATCH --mail-user=xyz.yu@mail.utoronto.ca
#SBATCH --mail-type=ALL
source ~/ENV/bin/activate

python SBN/SNB_3flavour_20-neutrino_2407_tensor.py