#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=128G               # memory per node
#SBATCH --time=3-00:00:00

source ../env.sh

python nonbipolar.py --e 1500 --b 1500