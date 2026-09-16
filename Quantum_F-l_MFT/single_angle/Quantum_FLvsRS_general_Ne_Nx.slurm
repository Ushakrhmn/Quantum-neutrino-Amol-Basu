#!/bin/bash
#SBATCH --account=def-nilic
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

Ne=$1
Nx=$2

scontrol update JobId=$SLURM_JOB_ID JobName=FLRS_Ne${Ne}_Nx${Nx}

source /scratch/urahaman/Quantum-neutrino-Amol-Basu/env.sh
echo "=== MODULES ==="
module list 2>&1

echo "=== MPI ENVIRONMENT ==="
env | grep -Ei '^(MPI|PMI|PMIX|OMPI|OPAL|ORTE|SLURM)' | sort
cd /scratch/urahaman/Quantum-neutrino-Amol-Basu/Quantum_F-l_MFT/single_angle

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Ne = ${Ne}"
echo "Nx = ${Nx}"
echo "CPUs = ${SLURM_CPUS_PER_TASK}"

# Monitor memory usage every 30 seconds
(
    while true; do
        echo "===== $(date) ====="
        sstat -j ${SLURM_JOB_ID}.batch \
            --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveVMSize
        sleep 30
    done
) > logs/memory_${SLURM_JOB_ID}.log &

MONITOR_PID=$!

# Run the simulation
#/usr/bin/time -v \
#    python -u Quantum_FLvsRS_general_Ne_Nx.py \
#    --Ne "${Ne}" \
#    --Nx "${Nx}"
srun --ntasks=1 --cpus-per-task=4 python -u Quantum_FLvsRS_general_Ne_Nx.py --Ne "${Ne}" --Nx "${Nx}"
echo "=== BEFORE PYTHON ==="
which python
python --version

python -u -c 'print("PYTHON STARTED SUCCESSFULLY", flush=True)'

STATUS=$?

echo "=== AFTER PYTHON ==="
echo "Python exit status = ${STATUS}"

exit ${STATUS}
STATUS=$?

# Stop the monitor
kill ${MONITOR_PID} 2>/dev/null

echo "Finished at $(date)"
echo "Exit status = ${STATUS}"

exit ${STATUS}
