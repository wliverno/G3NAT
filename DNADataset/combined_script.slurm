#!/bin/bash
#SBATCH --job-name=tagca_job
#SBATCH --account=stf
#SBATCH --partition=cpu-g2
## Nodes
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --mem=200GB
#SBATCH --mail-type=END
#SBATCH --mail-user=asyed4@uw.edu
##

set -e

file="tagca"

echo "Starting..."

echo "Generating PDB and GJF files for $file ..."
../willll/DNADataGen/dnabuilder -s "$file" -t "$DNA_TYPE"

echo "PDB and GJF generation complete."

# --- Gaussian step ---
module load chem/g16

export GAUSS_SCRDIR='/tmp'
export GAUSS_PDEF=$SLURM_NTASKS
export GAUSS_MEMDEF=${SLURM_MEM_PER_NODE}MB

g16 $file.gjf && echo "Finished DFT Calc, writing to LOG file..."

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
scripts/readmat $file.mat > $file.txt && echo "TXT file written, converting to matlab matrix..."

cp "$file.mat" "$file"_gaussian.mat

# CONVERT PLAINTEXT TO GAUSSIAN MAT
ulimit -s unlimited
module load matlab

matlab -nojvm -r "addpath('scripts'); readMAT('$file'); quit" > output.out && echo "Hamiltonian written to $file.mat!"

# --- collect all outputs into a folder named after the file (keep .gjf/.pdb here) ---
OUTDIR="$PWD/$file"
mkdir -p "$OUTDIR"

# move everything that starts with the file name
mv -f ${file}* "$OUTDIR"/ 2>/dev/null || true

# copy the slurm log as well
if [[ -n "${SLURM_JOB_ID:-}" && -f "slurm-${SLURM_JOB_ID}.out" ]]; then
  cp "slurm-${SLURM_JOB_ID}.out" "$OUTDIR/"
fi

echo "Saved outputs to: $OUTDIR"

exit 0
