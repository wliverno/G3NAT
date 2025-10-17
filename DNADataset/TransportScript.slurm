#!/bin/bash
#SBATCH --job-name=dna_setup
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=willll@uw.edu
##

# Master script to set up DNA transmission calculations
# Creates 4 run directories with different parameters and submits child jobs

# Configuration
PDB_FILE="ggccgg.pdb"  # Change this to your PDB file
PYTHON_SCRIPT="./TransportSetup.py"  # Your Python script name

# Check if required files exist
if [ ! -f "$PDB_FILE" ]; then
    echo "Error: PDB file $PDB_FILE not found!"
    exit 20
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!"
    exit 21
fi

# Check if Hamiltonian .mat file exists
BASE_NAME=$(basename "$PDB_FILE" .pdb)
if [ ! -f "${BASE_NAME}.mat" ]; then
    echo "Warning: Hamiltonian file ${BASE_NAME}.mat not found in current directory!"
    echo "Make sure this file exists before running the transmission calculations."
fi

echo "Setting up DNA transmission calculations..."
echo "PDB file: $PDB_FILE"
echo "Base name: $BASE_NAME"
echo ""

# Define the 4 cases: gamma value, mode (same/cross), description
declare -a CASES=(
    "0.1:same:gamma0.1_same"
    "0.1:cross:gamma0.1_cross"
    "0.6:same:gamma0.6_same"
    "0.6:cross:gamma0.6_cross"
)

# Create run directories and generate Parameters.txt files
for i in {1..4}; do
    RUN_DIR="run${i}"
    
    # Parse case parameters
    IFS=':' read -r GAMMA MODE DESC <<< "${CASES[$((i-1))]}"
    
    echo "Creating ${RUN_DIR}/ (gamma=${GAMMA} eV, mode=${MODE})"
    
    # Create directory
    mkdir -p "$RUN_DIR"
    
    # Generate Parameters.txt using Python script
    python3 "$PYTHON_SCRIPT" "$PDB_FILE" --mode "$MODE" --gamma "$GAMMA"
    
    # Move generated Parameters file to run directory
    PARAM_FILE="Parameters_${BASE_NAME}.txt"
    if [ -f "$PARAM_FILE" ]; then
        mv "$PARAM_FILE" "${RUN_DIR}/Parameters.txt"
        echo "  ✓ Generated ${RUN_DIR}/Parameters.txt"
    else
        echo "  ✗ Failed to generate ${PARAM_FILE}"
        continue
    fi
    
    # Copy Hamiltonian file to parent directory if not already there
    # (MATLAB script looks for it in parent directory of run folders)
    
    echo ""
done

echo "Setup complete!"
echo ""
echo "Submitting child jobs for transmission calculations..."
echo ""

# Submit child jobs for each run directory
for i in {1..4}; do
    RUN_DIR="run${i}"
    IFS=':' read -r GAMMA MODE DESC <<< "${CASES[$((i-1))]}"
    
    if [ -d "$RUN_DIR" ] && [ -f "${RUN_DIR}/Parameters.txt" ]; then
        # Submit transmission calculation job
        JOB_ID=$(sbatch --parsable run_transmission.slurm "$i" "$DESC")
        echo "Submitted transmission job for ${RUN_DIR}: Job ID ${JOB_ID}"
    else
        echo "Skipping ${RUN_DIR} - directory or Parameters.txt not found"
    fi
done

echo ""
echo "All jobs submitted!"
echo "Monitor jobs with: squeue -u \$USER"
