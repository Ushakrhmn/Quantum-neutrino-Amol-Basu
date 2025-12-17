#!/bin/bash
# Setup script for new checkpointed bipolar simulation series
# Run this interactively to initialize a new simulation run

set -e

# ============================================================================
# PARAMETER CONFIGURATION SECTION
# ============================================================================
# Edit the values below to configure your simulation parameters

# Run folder name (required - will be created)
RUN_FOLDER="run_001"

# Physical parameters
E=100                    # Number of electron neutrinos in bin 1
B=100                    # Number of electron antineutrinos in bin 2
ENERGY=1.0               # Energy of all neutrinos and antineutrinos
J=5.0                    # Interaction strength
L=16.0                   # Baseline (total simulation length)
S=500                    # Total number of time steps
THETA=0.001              # Mixing angle
DMSQ=-1.0                # Mass squared difference

# Evolution parameters
CHUNK=64                 # expm_multiply block size
STEPS_PER_RUN=100        # Steps per driver invocation (checkpoint frequency)
NORMALIZE=0              # Set to 1 to enable L2-normalization at each step

# ============================================================================
# END OF PARAMETER CONFIGURATION SECTION
# ============================================================================

# Source environment
source ../env.sh

# Normalize flag
NORMALIZE_FLAG=""
if [ "$NORMALIZE" = "1" ]; then
    NORMALIZE_FLAG="--normalize"
fi

echo "Setting up new bipolar simulation series..."
echo "Run folder: $RUN_FOLDER"
echo "Parameters:"
echo "  e=$E, b=$B, energy=$ENERGY"
echo "  j=$J, l=$L, s=$S"
echo "  theta=$THETA, dmsq=$DMSQ"
echo "  chunk=$CHUNK, steps_per_run=$STEPS_PER_RUN"
echo ""

# Run configuration script
python3 series_bipolar_config.py \
    --output "$RUN_FOLDER" \
    --e "$E" \
    --b "$B" \
    --energy "$ENERGY" \
    --j "$J" \
    --l "$L" \
    --s "$S" \
    --theta "$THETA" \
    --dmsq "$DMSQ" \
    --chunk "$CHUNK" \
    --steps-per-run "$STEPS_PER_RUN" \
    $NORMALIZE_FLAG

echo ""
echo "Setup complete! Run folder created: $RUN_FOLDER"
echo ""
echo "To continue the simulation, submit the batch job:"
echo "  sbatch continue.sh $RUN_FOLDER"
echo ""
echo "Or run interactively:"
echo "  source ../env.sh && python3 series_bipolar_driver.py $RUN_FOLDER"

