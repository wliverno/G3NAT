#!/bin/bash
#SBATCH --job-name=g3nat-baseline
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3070
#SBATCH --requeue
#SBATCH --output=slurm-base-%A_%a.out
#
# NON-SELF BASELINE: the physics-blind direct-regression GNN.
#
# G3NAT's claim is that routing the prediction through a tight-binding
# Hamiltonian and a NEGF solver is worth doing. The control that answers it is
# the SAME graph and the SAME encoder regressing DOS and transmission directly,
# with no Hamiltonian and no NEGF. Every previous standard-model number was
# measured under the old flat-index split, which put the same sequence in train
# and val across its ~4 contact variants, so all of them are inflated and none
# are usable. This re-runs it under the grouped split.
#
# WHAT IS CONTROLLED. The encoder is identical between the two models --
# node_proj (4->hidden), edge_proj (5->hidden), num_layers x {GATConv or
# TransformerConv}(hidden, hidden/heads, heads, edge_dim=hidden), LayerNorm,
# conv->norm->relu. Verified: 533,248 encoder parameters in both at
# hidden=256/layers=4/heads=4/gat.
#
# WHAT DIFFERS, deliberately, because it IS the experimental variable:
#   blind  global_mean_pool -> two INDEPENDENT MLP heads (hidden -> hidden/2 ->
#          n_energy), one for DOS and one for T. Readout 117,650 params.
#   G3NAT  no pooling (H is per-site) -> onsite_proj per node + coupling_proj
#          per edge -> H -> NEGF -> both spectra from ONE G^r. Readout 264,712.
# Totals 650,898 (blind, gat) vs 797,960 (G3NAT, gat, n_orb=2). G3NAT carries
# 23% MORE parameters, all of it in the readout. We do not parameter-match,
# because matching would mean shrinking the encoder and breaking the control;
# report both counts instead.
#
# Note the structural consequence to test for: the blind model's DOS and T heads
# share nothing but the pooled vector, so nothing can make a transmission
# resonance land on a DOS peak. In G3NAT that is automatic.
#
# Dropout is forced to 0.0. The standard model otherwise passes dropout into its
# conv AND applies F.dropout after each ReLU, while the hamiltonian model does
# neither -- at 0.0 the two encoder stacks are equivalent.
#
# Both convolutions are run. If the claim is that the physics layer costs
# accuracy, the baseline gets its strongest configuration, not a hobbled one.
#
#   sbatch --array=0-5 scripts/run_baseline.sh
#
# 6 cells = 2 conv types x 3 seeds.

module load cuda
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT

# --- disk guard (see run_ldos_phases.sh: a full /mmfs1 killed 15 of 22 cells
# on 2026-07-25 with logs truncated mid-traceback, looking like a crash) ------
MIN_FREE_GB=${MIN_FREE_GB:-50}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "ABORT: only ${FREE_GB}GB free, need ${MIN_FREE_GB}GB. Writes would fail mid-run."
  exit 1
fi

DATA_DIR="${DATA_DIR:-pickle_files_v2}"
EPOCHS="${EPOCHS:-15000}"
IFS=' ' read -ra SEEDS <<< "${SEEDS:-42 43 44}"
IFS=' ' read -ra CONVS <<< "${CONVS:-gat transformer}"

i="${SLURM_ARRAY_TASK_ID}"
: "${i:?run this via sbatch, not directly}"
N=$(( ${#CONVS[@]} * ${#SEEDS[@]} ))
if [ "$i" -ge "$N" ]; then
  echo "ERROR: array task $i out of range (valid 0-$(( N - 1 )), $N cells = ${#CONVS[@]} convs x ${#SEEDS[@]} seeds)" >&2
  exit 1
fi
CONV="${CONVS[$(( i / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"

TAG="baseline_${CONV}_s${SEED}"
OUT="outputs_${TAG}"
CKPT="ckpt_${TAG}"
echo "=== cell start: tag=${TAG} conv=${CONV} seed=${SEED} epochs=${EPOCHS} ==="

python -u scripts/train.py \
  --data_source pickle \
  --data_dir "${DATA_DIR}" \
  --model_type standard \
  --conv_type "${CONV}" \
  --hidden_dim 256 \
  --num_layers 4 \
  --num_heads 4 \
  --dropout 0.0 \
  --learning_rate 1e-3 \
  --batch_size 32 \
  --num_epochs "${EPOCHS}" \
  --split_seed "${SEED}" \
  --init_seed "${SEED}" \
  --output_dir "${OUT}" \
  --checkpoint_dir "${CKPT}"
RC=$?
echo "=== cell done: tag=${TAG} rc=${RC} wall=${SECONDS}s ($(( SECONDS / 60 ))m) ==="
exit $RC
