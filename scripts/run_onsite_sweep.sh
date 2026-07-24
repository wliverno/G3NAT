#!/bin/bash
#SBATCH --job-name=g3nat-onsite-sweep
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-17
##
# Structured-onsite alpha sweep (plan Task 7, Steps 2-3).
#
# Fixed GLOBAL alpha sweep on DFT (pickle) data:
#     onsite = alpha*baseline[base] + (1-alpha)*context_head.
# One array task per (alpha, seed) cell. alpha=0 is the current free-onsite model under the
# clean grouped split (H byte-identical to non-structured; the baseline params exist but get
# zero gradient, so this IS the honest baseline that replaces the leaky 0.547). alpha=1 is the
# pure per-base baseline. The val_loss(alpha) curve is the discriminator: how much context the
# DFT data actually needs in the onsite.
#
# Grid (override via --export for the smoke):
#   ALPHAS      default "0 0.25 0.5 0.75 0.9 1.0"  (6)
#   SEEDS       default "42 43 44"                 (3)  -> 18 cells = --array=0-17
#   NUM_EPOCHS  default 5000
#   PREFIX      default onsite_sweep               -> dirs outputs_<PREFIX>_a<alpha>_s<seed>
#
# Resume-safe: train.py checkpoints every 10 epochs to --checkpoint_dir and resumes from
# checkpoint_latest.pth, so ckpt-all preemption + --requeue continues instead of restarting.
#
# SMOKE (1 cell, 50 epochs, pure-baseline path -- verifies plumbing + gives per-epoch timing):
#   sbatch --array=0-0 --time=00:30:00 --job-name=g3nat-onsite-smoke \
#     --export=ALL,ALPHAS=1.0,SEEDS=42,NUM_EPOCHS=50,PREFIX=onsite_smoke scripts/run_onsite_sweep.sh
# FULL SWEEP (after sign-off):
#   sbatch scripts/run_onsite_sweep.sh          # optionally throttle concurrency: --array=0-17%6

module load cuda
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT

IFS=' ' read -ra ALPHA_ARR <<< "${ALPHAS:-0 0.25 0.5 0.75 0.9 1.0}"
IFS=' ' read -ra SEED_ARR  <<< "${SEEDS:-42 43 44}"
NA=${#ALPHA_ARR[@]}
NS=${#SEED_ARR[@]}
NUM_EPOCHS=${NUM_EPOCHS:-5000}
PREFIX=${PREFIX:-onsite_sweep}

TID=${SLURM_ARRAY_TASK_ID:?must run via sbatch --array}
if [ "$TID" -ge $(( NA * NS )) ]; then
  echo "ERROR: array task $TID out of range for ${NA}x${NS}=$(( NA * NS )) cells"
  exit 1
fi
AI=$(( TID / NS ))
SI=$(( TID % NS ))
ALPHA=${ALPHA_ARR[$AI]}
SEED=${SEED_ARR[$SI]}
TAG="a${ALPHA}_s${SEED}"
OUT="outputs_${PREFIX}_${TAG}"
CKPT="ckpt_${PREFIX}_${TAG}"

echo "=== onsite cell: task=$TID alpha=$ALPHA seed=$SEED epochs=$NUM_EPOCHS -> $OUT ==="
SECONDS=0
python -u scripts/train.py \
  --data_source pickle --data_dir pickle_files \
  --model_type hamiltonian --conv_type gat \
  --hidden_dim 256 --num_layers 4 --num_heads 4 --n_orb 1 --num_energy_points 100 \
  --batch_size 32 --num_epochs "$NUM_EPOCHS" --learning_rate 1e-3 \
  --structured_onsite --alpha_granularity global --alpha_mode fixed --alpha_value "$ALPHA" \
  --split_seed "$SEED" \
  --output_dir "$OUT" --checkpoint_dir "$CKPT"
RC=$?
echo "=== cell done: alpha=$ALPHA seed=$SEED rc=$RC wall=${SECONDS}s ($(( SECONDS / 60 ))m) model=$OUT/hamiltonian_pickle_model.pth ==="
exit $RC
