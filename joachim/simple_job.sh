#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=100G               # memory per node
#SBATCH --time=3-00:00:00

source env.sh

python nunu.py
