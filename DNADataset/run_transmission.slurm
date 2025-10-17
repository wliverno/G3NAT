#!/bin/bash
#SBATCH --job-name=dna_trans
#SBATCH --partition=ckpt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=20GB
#SBATCH --output=run%a/transmission_%j.out
#SBATCH --error=run%a/transmission_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=willll@uw.edu
##

# Child script to run DNA transmission calculations
# Usage: sbatch run_transmission.slurm <run_number> <description>

RUN_NUM=$1
DESC=$2

if [ -z "$RUN_NUM" ]; then
    echo "Error: Run number not provided!"
    exit 1
fi

RUN_DIR="run${RUN_NUM}"

echo "=========================================="
echo "DNA Transmission Calculation"
echo "=========================================="
echo "Run directory: ${RUN_DIR}"
echo "Description: ${DESC}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory ${RUN_DIR} does not exist!"
    exit 10
fi

# Check if Parameters.txt exists
if [ ! -f "${RUN_DIR}/Parameters.txt" ]; then
    echo "Error: ${RUN_DIR}/Parameters.txt not found!"
    exit 11
fi

# Load MATLAB module
module load matlab

echo "Loaded modules:"
module list
echo ""

# Run ballistic transmission calculation
echo "Running DNATransmission_Ballistic(${RUN_NUM})..."
matlab -nodisplay -nosplash -nodesktop -r "try; DNATransmission_Ballistic(${RUN_NUM}); catch ME; disp(ME.message); exit(1); end; exit(0);"

if [ $? -eq 0 ]; then
    echo ""
    echo "DNATransmission_Ballistic completed successfully"
else
    echo ""
    echo "DNATransmission_Ballistic failed!"
    exit 12
fi

# Run DOS calculation
echo ""
echo "Running DOS_calc(${RUN_NUM})..."
matlab -nodisplay -nosplash -nodesktop -r "try; DOS_calc(${RUN_NUM}); catch ME; disp(ME.message); exit(1); end; exit(0);"

if [ $? -eq 0 ]; then
    echo ""
    echo "DOS_calc completed successfully"
else
    echo ""
    echo "DOS_calc failed!"
    exit 13
fi

echo ""
echo "=========================================="
echo "All calculations completed for ${RUN_DIR}"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Generated files in ${RUN_DIR}:"
ls -lh "${RUN_DIR}/"

exit 0
