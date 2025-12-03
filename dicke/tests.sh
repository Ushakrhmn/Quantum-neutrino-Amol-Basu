#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=128G               # memory per node
#SBATCH --time=1-00:00:00

source ../env.sh

python stepwise_mft_validation.py --e1 2000 --m2 2000 --j 5.0 --steps 2048