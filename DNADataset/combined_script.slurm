#!/bin/bash
#SBATCH --job-name=job
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
## Nodes
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --mem=200GB
#SBATCH --mail-type=END
#SBATCH --mail-user=asyed4@uw.edu
##

set -e

# --- Generate random DNA sequence ---
bases=("a" "t" "c" "g")

# pick a random length between 4 and 8
length=$((RANDOM % 5 + 4))

while true; do
    sequence=""
    for ((i=0; i<length; i++)); do
        base=${bases[$((RANDOM % 4))]}
        sequence+=$base
    done

    file="$sequence"

    if [ ! -d "$file" ]; then
        break
    fi

    echo "Folder $file already exists, generating a new sequence..."
done

echo "Random DNA sequence: $file"
echo "Sequence length: $length"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    NEW_NAME="${file}_job"
    scontrol update JobId=$SLURM_JOB_ID JobName=$NEW_NAME
    echo "Job name updated to: $NEW_NAME"
fi

WORKDIR="/gscratch/anantram/asyed4/DNADataSet/$file"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Working directory: $(pwd)"
echo "Starting..."

echo "Generating PDB and GJF files for $file ..."
/gscratch/anantram/asyed4/DNADataGen/dnabuilder -s "$file" -t "$DNA_TYPE"

# Make a first-pass gjf for SCF (no matrix dump flags, no trailer)
cp "$file.gjf" "${file}_scf.gjf"

# Remove the matrix dump flags from the route line in the first-pass version
# This deletes "guess=(read,only) pop=minimal output=(matrix,i4lab)" from that copy
sed -i 's/guess=(read,only)//I; s/pop=minimal//I; s/output=(matrix,i4lab)//I' "${file}_scf.gjf"

# Remove the trailer line ("$file.mat") from the first-pass version
sed -i "/${file}\.mat/Id" "${file}_scf.gjf"

echo "PDB and GJF generation complete."

# --- Gaussian step ---
module load chem/g16

export GAUSS_SCRDIR='/tmp'
export GAUSS_PDEF=$SLURM_NTASKS
export GAUSS_MEMDEF=${SLURM_MEM_PER_NODE}MB

echo "Running SCF / checkpoint build (first pass) on ${file}_scf.gjf ..."
g16 "${file}_scf.gjf" && echo "First Gaussian run complete."

echo "Running matrix dump for $file.gjf ..."
g16 "$file.gjf" && echo "$file.mat should now exist."

# g16 $file.gjf && echo "Finished DFT Calc, writing to LOG file..."

if [ ! -f "${file}.mat" ]; then
    echo "ERROR: ${file}.mat not found. Gaussian second pass failed."
    exit 1
fi

echo ".mat file successfully generated."

# --- dftMatProcess section ---

echo $file
echo "Converting outputs..."

# COPY LOG FILE AND RUN MAT GENERATION JOB
# mv $file.log $file.init.log
# sed -i '/^#.*b3lyp/I s/$/ guess=(read,only) pop=minimal output=(matrix,i4lab)/' $file.gjf
# sed -i 's/rob3lyp/ub3lyp/' $file.gjf
# sed -i 's/ guess=read / /' $file.gjf
# echo "$file.mat" >> $file.gjf
# echo "" >> $file.gjf
# echo "" >> $file.gjf

# g16 $file.gjf && echo "Log file written, converting MAT file (fortran)..."

# CONVERT GAUSSIAN MAT TO PLAINTEXT
module load gcc
../scripts/readmat $file.mat > $file.txt && echo "TXT file written, converting to matlab matrix..."

cp "$file.mat" "$file"_gaussian.mat

# CONVERT PLAINTEXT TO GAUSSIAN MAT
ulimit -s unlimited
module load matlab

matlab -nojvm -r "addpath('../scripts'); readMAT('$file'); quit" > output.out && echo "Hamiltonian written to $file.mat!"

rm $file.txt
# Master script to set up DNA transmission calculations
# Creates 4 run directories with different parameters and submits child jobs

module load cesg/python/3.8.10

# Configuration
PDB_FILE="${file}.pdb"
PYTHON_SCRIPT="../scripts/TransportSetup.py"  # Your Python script name

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
        echo "   ^|^s Generated ${RUN_DIR}/Parameters.txt"
    else
        echo "   ^|^w Failed to generate ${PARAM_FILE}"
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
JOB_IDS=()

for i in {1..4}; do
    RUN_DIR="run${i}"
    IFS=':' read -r GAMMA MODE DESC <<< "${CASES[$((i-1))]}"

    if [ -d "$RUN_DIR" ] && [ -f "${RUN_DIR}/Parameters.txt" ]; then
        JOB_ID=$(sbatch --parsable ../run_transmission.slurm "$i" "$DESC")
        echo "Submitted transmission job for ${RUN_DIR}: Job ID ${JOB_ID}"
        JOB_IDS+=("$JOB_ID")
    else
        echo "Skipping ${RUN_DIR} - directory or Parameters.txt not found"
    fi
done

# Guard: abort if no transmission jobs were submitted
if [ "${#JOB_IDS[@]}" -eq 0 ]; then
    echo "ERROR: No transmission jobs were submitted (all run dirs missing or incomplete)."
    exit 3
fi

# Build colon-separated dependency string for afterok
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
echo "Dependency string: afterok:${DEP_STR}"

# Submit pickle job — SLURM will release it only when all transmission jobs complete successfully
PICKLE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${DEP_STR}" \
    --export=ALL,WORKDIR="$WORKDIR" \
    ../convert_to_pickle.slurm)
echo "Submitted pickle job: Job ID ${PICKLE_JOB_ID}"
echo "  will run after: ${JOB_IDS[*]}"

echo ""
echo "All jobs submitted!"
echo "Transmission jobs: ${JOB_IDS[*]}"
echo "Pickle job:        ${PICKLE_JOB_ID} (runs after all transmission jobs complete)"
echo "Monitor jobs with: squeue -u \$USER"
