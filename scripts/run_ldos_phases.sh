#!/bin/bash
#SBATCH --job-name=g3nat-ldos-phases
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3070
#SBATCH --requeue
#SBATCH --output=slurm-ldos-%A_%a.out

# Four-phase LDOS experiment. Phase is chosen by PHASE=... in the sbatch env.
# This script covers phases A, B and C only (see Phase D note below).
#
# Header, environment setup, disk guard, error-handling posture and python
# invocation deliberately match the other sweep runners in this repo
# (scripts/run_layers_sweep.sh, run_onsite_sweep.sh, run_optimizer_sweep.sh):
# --gpus=1, --nodes=1, --ntasks-per-node=8, --time=24:00:00, module load +
# conda activate, the disk-space guard, and a direct `python -u` invocation
# (no srun/conda-run wrapper, no `set -e`). Training at hidden_dim=256/
# num_layers=4 for 15000 epochs needs a GPU and far more than 6 hours; those
# precedent scripts are the ones known to actually work here. None of them
# use `set -e`: this script does not either, so a failed training cell still
# falls through to log its own exit code and tag (see "cell done" below)
# instead of the script dying silently mid-command with no record of which
# cell failed. --exclude=g3070 (uncorrectable ECC) and --requeue (preemptible
# partition) are ours -- none of the precedent scripts have the g3070
# exclusion.
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
#   PHASE=O sbatch --array=0-8  scripts/run_ldos_phases.sh      # 3 b-values x 3 seeds
#   PHASE=O O_B="0.0 0.1" sbatch --array=0-5 scripts/run_ldos_phases.sh

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

: "${PHASE:?set PHASE=A|B|C in the sbatch environment}"
: "${SLURM_ARRAY_TASK_ID:?run this via sbatch --array, not directly}"

DATA_DIR="${DATA_DIR:-pickle_files_v2}"
EPOCHS="${EPOCHS:-15000}"
# n_orb is overridable for the frontier-level experiment. The HOMO+/-1eV window
# holds 1.25-2.62 DFT levels per base (counted from the Gaussian logs) while
# n_orb=1 supplies one, and the measured DOS offset of 0.2199 decades matches
# that shortfall (level-counting predicts 0.2005). Prediction: at n_orb=2 the
# offset collapses toward zero, most strongly for AT-rich sequences where the
# level count is highest. docs/model-results.md section 7.
N_ORB="${N_ORB:-1}"
# Seeds 42/43/44 by default. SEEDS is overridable so the pre-declared backup
# seed 45 can be run for a cell flagged anomalous, without editing this file:
#   PHASE=O O_B=0.1 SEEDS=45 sbatch --array=0-0 scripts/run_ldos_phases.sh
IFS=' ' read -ra SEEDS <<< "${SEEDS:-42 43 44}"

# Held fixed across every phase: the 3-seed replication config whose cross-seed
# scatter is on record (docs/model-results.md replication section).
#
# SEEDS sets --split_seed, and --init_seed follows it unless INIT_SEED overrides.
# Before 2026-07-31 --split_seed was the ONLY seed, so a "3 seed" sweep varied the
# held-out set with initialization left uncontrolled, and no cell was reproducible.
# Tying them by default keeps a seed meaning one (split, init) pair. To vary
# initialization at a FIXED split -- which is what a reproducibility sweep over H
# requires -- pin SEEDS to one value and set INIT_SEED per cell, e.g.
#   for k in 1 2 3 4 5; do SEEDS=42 INIT_SEED=$k N_ORB=2 \
#     sbatch --array=0-0 scripts/run_ldos_phases.sh; done
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
  --n_orb "${N_ORB}"
  --learning_rate 1e-3
  --batch_size 32
  --num_epochs "${EPOCHS}"
)

# GEOM=1 fuses the SE(3)-invariant X3DNA edge geometry (7 channels per edge).
#
# Off by default only for continuity with the runs already on record -- NOT
# because geometry has been shown to be unhelpful. The one comparison ever made
# (0.538 vs 0.547) was a single final-epoch pair under the since-retired leaking
# split and is not usable evidence. Whether the channel helps, hurts or is
# neutral is an open empirical question, and GEOM=1 is how it gets answered.
#
# It is also a capability requirement independent of val loss: a model trained
# without the channel can never respond to conformational change, so it could
# never be run over an MD trajectory to watch onsite energies fluctuate. That
# use is a goal of the project, and it needs the weights to exist and be trained.
#
# Cache: geometry_v2.pkl covers all 520 pickle_files_v2 sequences. The older
# geometry.pkl has 515 and silently omits the five v2 additions.
GEOM_TAG=""
if [ "${GEOM:-0}" = "1" ]; then
  COMMON+=(--use_geometry --geom_cache "${GEOM_CACHE:-geom_cache/geometry_v2.pkl}")
  GEOM_TAG="_geom"
fi

# Suffix only when non-default, so existing tag names are unchanged.
NORB_TAG=""
if [ "${N_ORB}" != "1" ]; then NORB_TAG="_n${N_ORB}"; fi

i="${SLURM_ARRAY_TASK_ID}"

case "${PHASE}" in
  A)
    N=${#SEEDS[@]}
    if [ "$i" -ge "$N" ]; then
      echo "ERROR: array task $i out of range for phase A (valid range 0-$(( N - 1 )), $N cells = ${#SEEDS[@]} seeds)" >&2
      exit 1
    fi
    SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"
    B_VAL=0.0
    TARGET=residue
    TAG="A_b0.0_s${SEED}${NORB_TAG}${GEOM_TAG}"
    ;;
  B)
    # B_B overrides the grid, mirroring O_B in phase O, so a single b value can
    # be added to the factorial without editing this list.
    #
    # b = 1.0 was missing from the original grid and is the symmetric endpoint to
    # phase A's b = 0.0: b weights b*LDOS + (1-b)*DOS, so b=0 is DOS-only and
    # b=1 is LDOS-only. It is safe at b=1 because shape_loss defaults False (and
    # is retracted, config.py:35): the shape path derives its alignment offset
    # from dos_pred - dos_target and applies it to LDOS, which at b=1 would anchor
    # on a channel receiving no gradient. On the absolute loss that path is not
    # taken and the objective is simply a*T + LDOS.
    #
    #   PHASE=B B_B=1.0 N_ORB=2 sbatch --array=0-2 scripts/run_ldos_phases.sh
    IFS=' ' read -ra B_GRID <<< "${B_B:-0.1 0.25 0.5 0.75 0.9}"
    N=$(( ${#B_GRID[@]} * ${#SEEDS[@]} ))
    if [ "$i" -ge "$N" ]; then
      echo "ERROR: array task $i out of range for phase B (valid range 0-$(( N - 1 )), $N cells = ${#B_GRID[@]} b-values x ${#SEEDS[@]} seeds)" >&2
      exit 1
    fi
    B_VAL="${B_GRID[$(( i / ${#SEEDS[@]} ))]}"
    SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"
    TARGET=residue
    TAG="B_b${B_VAL}_s${SEED}${NORB_TAG}${GEOM_TAG}"
    ;;
  C)
    : "${B_BEST:?Phase C needs B_BEST=<b> from Phase B}"
    : "${B_NEIGHBOUR:?Phase C needs B_NEIGHBOUR=<b> from Phase B}"
    if [ "${B_BEST}" = "${B_NEIGHBOUR}" ]; then
      echo "ERROR: B_BEST and B_NEIGHBOUR are both ${B_BEST}; phase C needs two" >&2
      echo "       distinct b values, or cells i and i+3 (same seed) would share" >&2
      echo "       the same TAG/OUT/CKPT and silently resume each other's weights." >&2
      exit 1
    fi
    C_GRID=("${B_BEST}" "${B_NEIGHBOUR}")
    N=$(( ${#C_GRID[@]} * ${#SEEDS[@]} ))
    if [ "$i" -ge "$N" ]; then
      echo "ERROR: array task $i out of range for phase C (valid range 0-$(( N - 1 )), $N cells = ${#C_GRID[@]} b-values x ${#SEEDS[@]} seeds)" >&2
      exit 1
    fi
    B_VAL="${C_GRID[$(( i / ${#SEEDS[@]} ))]}"
    SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"
    TARGET=base_only
    TAG="C_baseonly_b${B_VAL}_s${SEED}${NORB_TAG}${GEOM_TAG}"
    ;;
  O)
    # STRUCTURED ONSITE, alpha = 1.0. This is the phase that tests the actual
    # deliverable: a toy Hamiltonian that fits DNA with only N onsite terms.
    # At alpha=1.0 the context head is switched off and `onsite_baseline` is the
    # real learned 4-value per-base table -- scripts/extract_tb_params.py reads
    # exactly that tensor.
    #
    # CORRECTED 2026-08-03. The header of that script says the table is
    # "meaningless for any alpha < 1, where the mixing is a vacuous
    # reparametrisation". That is overstated, and it is why alpha was only ever
    # run at 0 and 1 -- a continuous parameter treated as a switch.
    #   - It is true that for alpha < 1 the free context head can absorb the
    #     baseline, so the FUNCTION SPACE is the same at every alpha < 1.
    #   - It does not follow that alpha is inert. Function space is not
    #     optimisation: different parametrisations have different gradient
    #     geometry and implicit bias. Untested here.
    #   - Nor is the table meaningless below 1. It is PARTIAL:
    #     onsite = alpha*baseline + (1-alpha)*context still learns a real
    #     4-value table explaining an alpha-fraction of the onsite energy.
    #     "Base identity alone accounts for 50% of the onsite term" is a
    #     reportable result, not a degenerate one.
    #   - alpha_mode='learned' exists and has never been run at the current
    #     configuration. It answers the question directly: how much of the
    #     onsite energy does the data want explained by base identity?
    # A fractional sweep plus a learned-alpha arm is outstanding work.
    #
    # Phases A and B ran the DEFAULT free-onsite model, where onsite is a
    # continuous function of context and there is no N-value table at all
    # (verified: structured_onsite=False, onsite_baseline absent from the
    # state_dict). They therefore cannot speak to the N-onsite-terms question.
    #
    # b = 0.0 is included as a MATCHED baseline and is NOT optional. The recorded
    # cross-seed scatter for this table -- C std 0.52 with the rank order
    # changing between seeds, docs/model-results.md section 4b -- was measured on
    # v1 data through the old two-term loss path, so it is not a valid comparison
    # point for these runs. The measurement here is whether LDOS supervision
    # shrinks that scatter, which needs both arms from the same pipeline.
    IFS=' ' read -ra O_GRID <<< "${O_B:-0.0 0.1 0.5}"
    N=$(( ${#O_GRID[@]} * ${#SEEDS[@]} ))
    if [ "$i" -ge "$N" ]; then
      echo "ERROR: array task $i out of range for phase O (valid range 0-$(( N - 1 )), $N cells = ${#O_GRID[@]} b-values x ${#SEEDS[@]} seeds)" >&2
      exit 1
    fi
    B_VAL="${O_GRID[$(( i / ${#SEEDS[@]} ))]}"
    SEED="${SEEDS[$(( i % ${#SEEDS[@]} ))]}"
    TARGET=residue
    TAG="O_a1.0_b${B_VAL}_s${SEED}${NORB_TAG}${GEOM_TAG}"
    EXTRA=(--structured_onsite --alpha_granularity global --alpha_mode fixed --alpha_value 1.0)
    ;;
  *)
    echo "unknown PHASE=${PHASE}; this runner covers A, B, C and O." >&2
    echo "Phase D (composition holdout) gets its own plan once Phase B lands." >&2
    exit 1
    ;;
esac

# Per-cell directories, always under the repo tree on /gscratch -- never
# /tmp, which is node-local on this cluster and disappears once the job ends.
#
# Separate OUT (final model) and CKPT (resume checkpoints), each unique to
# this (phase, b, seed) cell, matching outputs_<prefix>_<tag> / ckpt_<prefix>
# _<tag> in the other repo sweep scripts. scripts/train.py auto-resumes from
# <checkpoint_dir>/checkpoint_latest.pth when present, so two cells sharing a
# checkpoint dir would silently resume each other's weights; a shared output
# dir would let concurrent array tasks clobber the same final-model file.
# The outputs_*/ and ckpt_*/ gitignore patterns already cover this "outputs_
# ldos_..." / "ckpt_ldos_..." naming, so these directories never enter the
# repo, same as the other sweep scripts' run artifacts.
# Suffix only when INIT_SEED is decoupled from SEED, so existing tag names are
# unchanged. Without this, an init sweep at a fixed split would write every cell
# into the SAME output directory and silently overwrite itself.
if [ -n "${INIT_SEED:-}" ] && [ "${INIT_SEED}" != "${SEED}" ]; then
  TAG="${TAG}_i${INIT_SEED}"
fi

OUT="outputs_ldos_${TAG}"
CKPT="ckpt_ldos_${TAG}"
mkdir -p "${OUT}" "${CKPT}"

echo "=== ldos cell: phase=${PHASE} task=${i} tag=${TAG} b=${B_VAL} target=${TARGET} seed=${SEED} -> ${OUT} ==="
SECONDS=0
python -u scripts/train.py \
  "${COMMON[@]}" \
  --loss_a 1.0 \
  --loss_b "${B_VAL}" \
  --ldos_target "${TARGET}" \
  --split_seed "${SEED}" \
  --init_seed "${INIT_SEED:-${SEED}}" \
  --output_dir "${OUT}" \
  --checkpoint_dir "${CKPT}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
RC=$?
echo "=== cell done: phase=${PHASE} tag=${TAG} rc=${RC} wall=${SECONDS}s ($(( SECONDS / 60 ))m) model=${OUT}/hamiltonian_pickle_model.pth ==="
exit $RC
