#!/bin/bash
#SBATCH --job-name=g3nat-regen
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-25
##
# Regenerate the dataset. Pure parsing -- upstream DFT/transport are NOT re-run.
# Each array task takes a contiguous slice of the sequence directories.
#
# Full two-step pipeline (step 1 is this SLURM array job, step 2 is a single
# process run after step 1 finishes):
#
#   sbatch DNADataset/run_regeneration.sh                  # -> pickle_files_v2/
#   python DNADataset/export_hdf5.py pickle_files_v2 g3nat_dna_transport.h5

source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT

MIN_FREE_GB=${MIN_FREE_GB:-50}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "ABORT: only ${FREE_GB}GB free, need ${MIN_FREE_GB}GB. Writes would fail mid-run."
  exit 1
fi

SRC=${SRC:-/mmfs1/gscratch/anantram/asyed4/DNADataSet}
OUT=${OUT:-pickle_files_v2}
mkdir -p "$OUT"

mapfile -t DIRS < <(find "$SRC" -maxdepth 1 -mindepth 1 -type d \
                    -regextype posix-extended -regex '.*/[acgt]+$' | sort)
N=${#DIRS[@]}
: "${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is unset -- run this via sbatch, not directly}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset -- run this via sbatch, not directly}"
CHUNK=$(( (N + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
echo "task $SLURM_ARRAY_TASK_ID: dirs $START..$((START+CHUNK-1)) of $N"

for (( i=START; i<START+CHUNK && i<N; i++ )); do
  python DNADataset/convert_to_pickle.py "${DIRS[$i]}" --out-dir "$OUT"
done
echo "task $SLURM_ARRAY_TASK_ID done"
