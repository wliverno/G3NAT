#!/bin/bash
#SBATCH --job-name=g3nat-opt-sweep
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-14
##
# Optimizer / weight-decay sweep. Task 17 item 2 -- the EVIDENCED first move against the
# overfitting, per docs/references.md.
#
# WHY. Loshchilov & Hutter, ICLR 2019 (arXiv:1711.05101): Adam's `weight_decay` is folded into
# the gradient and then rescaled by Adam's per-parameter adaptive rates, so it is NOT true
# weight decay. We have been running torch.optim.Adam(weight_decay=1e-5), i.e. essentially no
# regularization, against a model that overfits hard (train ~0.40 vs val ~0.63, best val at
# epoch ~550-2900 of 5000). AdamW decouples the decay so the nominal value means what it says.
#
# ARMS (5) x SEEDS (3) = 15 cells:
#   adam  1e-5   <- CONTROL, reproduces every historical run exactly
#   adamw 1e-5   <- isolates the DECOUPLING alone, decay magnitude held fixed
#   adamw 1e-3 / 1e-2 / 1e-1  <- the decay sweep
# The adam/adamw pair at the same 1e-5 is the point: it separates "AdamW is different" from
# "more decay helps", which a naive sweep would confound.
#
# READ THE PHYSICS TOO, not just val loss. The hypothesis (willll) is that an
# under-regularized, over-parameterized model has spare capacity to place somewhere
# unphysical, so regularization may improve generalization AND interpretability together.
# Check eta2, per-base resolution, and the A-T separation alongside the train-val gap.
#
# 5000 epochs: this is a screening sweep and all arms share the budget, so the comparison is
# internally fair. Note stronger decay may converge on a different timescale -- check
# best_val_epoch for cap-hugging before trusting any winner (see docs/metrics.md).
#
#   sbatch scripts/run_optimizer_sweep.sh

module load cuda
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT

# --- disk guard -------------------------------------------------------------
# 2026-07-25: 15 of 22 cells in the 15000-epoch run died within 60s across 11
# different nodes, with logs truncated mid-traceback, because /mmfs1 was 100%
# full. Failed writes look like a crash, not a disk error, and burn GPU time
# before dying. Fail loudly and early instead.
MIN_FREE_GB=${MIN_FREE_GB:-50}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "ABORT: only ${FREE_GB}GB free on $(df -h . | tail -1 | awk '{print $6}'), need ${MIN_FREE_GB}GB."
  echo "       Checkpoint writes will fail mid-run and the job will die looking like a crash."
  exit 1
fi
echo "disk check: ${FREE_GB}GB free (min ${MIN_FREE_GB}GB) -- ok"

OPTS=(adam  adamw adamw adamw adamw)
WDS=( 1e-5  1e-5  1e-3  1e-2  1e-1)
IFS=' ' read -ra SEED_ARR <<< "${SEEDS:-42 43 44}"
NS=${#SEED_ARR[@]}
NUM_EPOCHS=${NUM_EPOCHS:-5000}
PREFIX=${PREFIX:-optsweep}

TID=${SLURM_ARRAY_TASK_ID:?must run via sbatch --array}
AI=$(( TID / NS ))
SI=$(( TID % NS ))
if [ "$AI" -ge ${#OPTS[@]} ]; then echo "ERROR: task $TID out of range"; exit 1; fi
OPT=${OPTS[$AI]}
WD=${WDS[$AI]}
SEED=${SEED_ARR[$SI]}
TAG="${OPT}_wd${WD}_s${SEED}"
OUT="outputs_${PREFIX}_${TAG}"
CKPT="ckpt_${PREFIX}_${TAG}"

echo "=== opt cell: task=$TID optimizer=$OPT weight_decay=$WD seed=$SEED epochs=$NUM_EPOCHS -> $OUT ==="
SECONDS=0
python -u scripts/train.py \
  --data_source pickle --data_dir pickle_files \
  --model_type hamiltonian --conv_type gat \
  --hidden_dim 256 --num_layers 4 --num_heads 4 --n_orb 1 --num_energy_points 100 \
  --batch_size 32 --num_epochs "$NUM_EPOCHS" --learning_rate 1e-3 \
  --optimizer "$OPT" --weight_decay "$WD" \
  --split_seed "$SEED" \
  --output_dir "$OUT" --checkpoint_dir "$CKPT"
RC=$?
echo "=== cell done: opt=$OPT wd=$WD seed=$SEED rc=$RC wall=${SECONDS}s ($(( SECONDS / 60 ))m) ==="
exit $RC
