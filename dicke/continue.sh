#!/bin/bash
#SBATCH --account=def-nilic 
#SBATCH --mem=128G               # memory per node
#SBATCH --time=3-00:00:00

# Continue checkpointed bipolar simulation series
# Usage: sbatch continue.sh <run_folder>

set -e

# Get run folder from command line argument
RUN_FOLDER=${1:-${RUN_FOLDER}}

if [ -z "$RUN_FOLDER" ]; then
    echo "Error: Run folder must be specified"
    echo "Usage: sbatch continue.sh <run_folder>"
    echo "   or: RUN_FOLDER=<run_folder> sbatch continue.sh"
    exit 1
fi

# Source environment
source ../env.sh

# Change to script directory
cd "$(dirname "$0")"

echo "Continuing simulation: $RUN_FOLDER"
echo "Started at: $(date)"

# Run driver script
python3 series_bipolar_driver.py "$RUN_FOLDER"

echo "Completed at: $(date)"

