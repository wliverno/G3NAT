#!/bin/bash
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --exclude=g3070
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-ldos-%A_%a.out
set -euo pipefail

# Four-phase LDOS experiment. Phase is chosen by PHASE=... in the sbatch env.
# This script covers phases A, B and C only (see Phase D note below).
#
#   A  3 runs   b=0 re-baseline on v2. Establishes cross-seed scatter on both
#               DOS+T and the UNTRAINED LDOS agreement.
#   B  15 runs  b sweep with ldos_residue.
#   C  6 runs   ldos_base_only at Phase B's best b and one neighbour.
#
# Phase D (composition holdout) is NOT here. It runs only if Phase B is
# positive, needs a --holdout_composition flag and a split change that do not
# exist yet, and building them before knowing Phase B's outcome is speculative.
# It gets its own plan at that point.
#
# Submit one phase at a time. Phase B's b values are fixed here because b is a
# convex weight on [0,1] and a uniform grid needs no calibration; Phase C reads
# values that only exist after Phase B, so they are passed in.
#
#   PHASE=A sbatch --array=0-2  scripts/run_ldos_phases.sh
#   PHASE=B sbatch --array=0-14 scripts/run_ldos_phases.sh
#   PHASE=C B_BEST=<b> B_NEIGHBOUR=<b> sbatch --array=0-5 scripts/run_ldos_phases.sh

: "${PHASE:?set PHASE=A|B|C in the sbatch environment}"
: "${SLURM_ARRAY_TASK_ID:?run this via sbatch --array, not directly}"

DATA_DIR="${DATA_DIR:-pickle_files_v2}"
EPOCHS="${EPOCHS:-15000}"
SEEDS=(42 43 44)

# Held fixed across every phase: the 3-seed replication config whose cross-seed
# scatter is on record (docs/model-results.md replication section).
#
# NOTE: conv_type is lowercase "gat". scripts/train.py declares
# choices=['gat', 'transformer'] and argparse choice matching is case
# sensitive, so "GAT" would fail with an invalid-choice error at argparse
# time on every task in the array before any training starts.
COMMON=(
  --data_source pickle
  --data_dir "${DATA_DIR}"
  --model_type hamiltonian
  --conv_type gat
  --hidden_dim 256
  --num_layers 4
  --num_heads 4
  --n_orb 1
  --learning_rate 1e-3
  --batch_size 32
  --num_epochs "${EPOCHS}"
)

i="${SLURM_ARRAY_TASK_ID}"

case "${PHASE}" in
  A)
    SEED="${SEEDS[$(( i % 3 ))]}"
    B_VAL=0.0
    TARGET=residue
    TAG="A_b0.0_s${SEED}"
    ;;
  B)
    B_GRID=(0.1 0.25 0.5 0.75 0.9)
    B_VAL="${B_GRID[$(( i / 3 ))]}"
    SEED="${SEEDS[$(( i % 3 ))]}"
    TARGET=residue
    TAG="B_b${B_VAL}_s${SEED}"
    ;;
  C)
    : "${B_BEST:?Phase C needs B_BEST=<b> from Phase B}"
    : "${B_NEIGHBOUR:?Phase C needs B_NEIGHBOUR=<b> from Phase B}"
    C_GRID=("${B_BEST}" "${B_NEIGHBOUR}")
    B_VAL="${C_GRID[$(( i / 3 ))]}"
    SEED="${SEEDS[$(( i % 3 ))]}"
    TARGET=base_only
    TAG="C_baseonly_b${B_VAL}_s${SEED}"
    ;;
  *)
    echo "unknown PHASE=${PHASE}; this runner covers A, B and C." >&2
    echo "Phase D (composition holdout) gets its own plan once Phase B lands." >&2
    exit 1
    ;;
esac

# Per-run directory, always under the repo tree on /gscratch -- never /tmp,
# which is node-local on this cluster and disappears once the job ends.
#
# Used as BOTH --checkpoint_dir and --output_dir so this run's
# checkpoint_latest.pth / checkpoint_best.pth and its final saved model file
# never collide with another run's files. scripts/train.py auto-resumes from
# <checkpoint_dir>/checkpoint_latest.pth when present, so two phase runs
# sharing a directory would silently resume each other's weights; leaving
# --output_dir at its shared default would let concurrent array tasks
# clobber the same final-model file instead.
OUT="outputs_ldos/${PHASE}/${TAG}"
mkdir -p "${OUT}"

echo "PHASE=${PHASE} task=${i} tag=${TAG} b=${B_VAL} target=${TARGET} seed=${SEED}"

srun conda run -n g3nat python scripts/train.py \
  "${COMMON[@]}" \
  --loss_a 1.0 \
  --loss_b "${B_VAL}" \
  --ldos_target "${TARGET}" \
  --split_seed "${SEED}" \
  --checkpoint_dir "${OUT}" \
  --output_dir "${OUT}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
