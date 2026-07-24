#!/bin/bash
#SBATCH --job-name=g3nat-layers-sweep
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-7
##
# Receptive-field sweep: num_layers in {1,2,3,4} x split_seed in {42,43}.
#
# WHY. DNA sequences in this dataset are 4-8 bases, i.e. 8-16 DNA nodes on a ladder graph.
# Measured DNA-subgraph diameter: length 4 -> 4, length 8 -> 8 (roughly 2*len(seq)). So with
# num_layers=4 the convolutions cover the whole molecule only for length-4 sequences (26% of
# the dataset) and about half of it at length 8. Either way 4 hops reaches ~4 bases away,
# well beyond the 1-2 hop neighbourhood -- immediate stacking neighbours plus the H-bond
# partner -- that sets onsite energy physically. This sweep settles empirically whether the
# extra reach buys fit, and what it costs in how strongly onsite is tied to base identity.
#
# Read the result as a trade-off curve, NOT a single winner:
#   val_loss(L)  -- does reach buy fit?
#   eta2(L)      -- fraction of onsite variance explained by base identity (probe_onsite_dilution)
# If a small L matches the fit of L=4 at higher eta2, the extra layers were only smearing.
#
# CONTROL: the L=4, seed=42 cell is configured identically to the alpha=0 cell of
# run_onsite_sweep.sh (val 0.6054, outputs_onsite_sweep_a0_s42). It should reproduce it. Any
# gap between them is run-to-run noise and is the error bar for reading this sweep.
#
# Everything except num_layers is held identical to run_onsite_sweep.sh on purpose.
# alpha=0 (free onsite) so this measures the architecture, not the onsite parameterization.
#
# Grid (override via --export):
#   LAYERS      default "1 2 3 4"     (4)
#   SEEDS       default "42 43"       (2)  -> 8 cells = --array=0-7
#   NUM_EPOCHS  default 5000
#   PREFIX      default layers_sweep  -> dirs outputs_<PREFIX>_L<layers>_s<seed>
#
# SMOKE (1 cell, 50 epochs):
#   sbatch --array=0-0 --time=00:30:00 --job-name=g3nat-layers-smoke \
#     --export=ALL,LAYERS=2,SEEDS=42,NUM_EPOCHS=50,PREFIX=layers_smoke scripts/run_layers_sweep.sh
# FULL SWEEP:
#   sbatch scripts/run_layers_sweep.sh

module load cuda
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT

IFS=' ' read -ra LAYER_ARR <<< "${LAYERS:-1 2 3 4}"
IFS=' ' read -ra SEED_ARR  <<< "${SEEDS:-42 43}"
NL=${#LAYER_ARR[@]}
NS=${#SEED_ARR[@]}
NUM_EPOCHS=${NUM_EPOCHS:-5000}
PREFIX=${PREFIX:-layers_sweep}

TID=${SLURM_ARRAY_TASK_ID:?must run via sbatch --array}
if [ "$TID" -ge $(( NL * NS )) ]; then
  echo "ERROR: array task $TID out of range for ${NL}x${NS}=$(( NL * NS )) cells"
  exit 1
fi
LI=$(( TID / NS ))
SI=$(( TID % NS ))
NLAYERS=${LAYER_ARR[$LI]}
SEED=${SEED_ARR[$SI]}
TAG="L${NLAYERS}_s${SEED}"
OUT="outputs_${PREFIX}_${TAG}"
CKPT="ckpt_${PREFIX}_${TAG}"

echo "=== layers cell: task=$TID num_layers=$NLAYERS seed=$SEED epochs=$NUM_EPOCHS -> $OUT ==="
SECONDS=0
python -u scripts/train.py \
  --data_source pickle --data_dir pickle_files \
  --model_type hamiltonian --conv_type gat \
  --hidden_dim 256 --num_layers "$NLAYERS" --num_heads 4 --n_orb 1 --num_energy_points 100 \
  --batch_size 32 --num_epochs "$NUM_EPOCHS" --learning_rate 1e-3 \
  --split_seed "$SEED" \
  --output_dir "$OUT" --checkpoint_dir "$CKPT"
RC=$?
echo "=== cell done: layers=$NLAYERS seed=$SEED rc=$RC wall=${SECONDS}s ($(( SECONDS / 60 ))m) model=$OUT/hamiltonian_pickle_model.pth ==="
exit $RC
