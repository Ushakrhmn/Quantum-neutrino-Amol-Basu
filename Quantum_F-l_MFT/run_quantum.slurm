#!/bin/bash
#SBATCH --account=def-nilic
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

Ne=$1
Nx=$2

scontrol update JobId=$SLURM_JOB_ID JobName=FLRS_Ne${Ne}_Nx${Nx}

source /home/urahaman/scratch/Quantum-neutrino-Amol-Basu/env.sh
cd /home/urahaman/scratch/Quantum-neutrino-Amol-Basu/Quantum_F-l_MFT

echo "Running on $(hostname)"
echo "Started at $(date)"

# Monitor memory usage every 30 seconds
(
    while true; do
        echo "===== $(date) ====="
        sstat -j ${SLURM_JOB_ID}.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveVMSize
        sleep 30
    done
) > logs/memory_${SLURM_JOB_ID}.log &

MONITOR_PID=$!

# Run the simulation and report peak memory at the end
/usr/bin/time -v python Quantum_FLvsRS_general_Ne_Nx.py --Ne ${Ne} --Nx ${Nx}

# Stop the monitor
kill ${MONITOR_PID}

echo "Finished at $(date)"
