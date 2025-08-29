#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=100G               # memory per node
#SBATCH --time=0-2:00:00

python emu_2bin_qc_comp.py --j 0.25 --l 15 --s 128 --e1 4 --e2 8 --m1 8 --m2 4 --f 8