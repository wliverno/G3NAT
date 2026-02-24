#!/bin/bash
#SBATCH --job-name=dna_pickle
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=pickle_%j.out
#SBATCH --error=pickle_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asyed4@uw.edu

set -e

# WORKDIR is injected by combined_script.slurm via --export=ALL,WORKDIR=...
if [ -z "${WORKDIR:-}" ]; then
    echo "ERROR: WORKDIR is not set. Submit with --export=ALL,WORKDIR=<path>"
    exit 1
fi

echo "Pickle job starting."
echo "WORKDIR: $WORKDIR"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo ""

module load cesg/python/3.8.10

# Derive path to convert_to_pickle.py — one directory above WORKDIR
# e.g. WORKDIR=/gscratch/anantram/asyed4/DNADataSet/atcg
#      SCRIPT_DIR=/gscratch/anantram/asyed4/DNADataSet
SCRIPT_DIR="$(dirname "$WORKDIR")"
PICKLE_SCRIPT="${SCRIPT_DIR}/convert_to_pickle.py"

if [ ! -f "$PICKLE_SCRIPT" ]; then
    echo "ERROR: Pickle script not found at $PICKLE_SCRIPT"
    exit 1
fi

echo "Running: python $PICKLE_SCRIPT $WORKDIR"
python "$PICKLE_SCRIPT" "$WORKDIR"

echo ""
echo "Pickle generation complete."
echo "End: $(date)"
