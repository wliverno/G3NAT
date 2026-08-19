#!/bin/bash
#SBATCH --job-name=g3nat-supmatrix
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3070
#SBATCH --requeue
#SBATCH --output=slurm-supmatrix-%A_%a.out
# Default open-mode TRUNCATES the log when a preempted job is requeued, throwing
# away everything stdout produced before the preemption -- including the
# best-publication warning and the nan-skip counts, which are the artifacts that
# make a silently-failed run visible. On a preemptible partition that is the
# common case, not the rare one.
#SBATCH --open-mode=append
#
# 72-run supervision x n_orb x num_layers x geometry matrix (24 cells x 3 init seeds).
#
# THIS FILE IS BOTH THE DRIVER AND THE JOB BODY.
#   - With SLURM_ARRAY_TASK_ID unset it is the DRIVER: it builds the design table,
#     prints or submits. It runs no python and loads no modules, so it is safe on a
#     login node.
#   - With SLURM_ARRAY_TASK_ID set it is the JOB: module load, conda, train one cell.
# The mode switch happens BEFORE any module/conda/python line, deliberately.
#
# USAGE
#   bash scripts/run_supervision_matrix.sh                  # dry run (DEFAULT)
#   bash scripts/run_supervision_matrix.sh --dry-run        # same thing, explicit
#   bash scripts/run_supervision_matrix.sh --submit         # actually submit
#   bash scripts/run_supervision_matrix.sh --submit --force # submit even completed cells
#   MAXPAR=12 bash scripts/run_supervision_matrix.sh --submit   # throttle concurrency
#
# WHAT WAS REUSED from scripts/run_ldos_phases.sh (known-good, do not re-litigate):
#   the whole sbatch preamble (account anantram-ckpt, partition ckpt-all, --nodes=1,
#   --gpus=1, --ntasks-per-node=8, --mem=32G, --exclude=g3070, --requeue), the
#   `module load cuda` + conda.sh + `conda activate g3nat` + `cd` sequence, the
#   free-disk guard, the no-`set -e` posture (a failed cell must still log its own
#   rc and tag), the direct `python -u scripts/train.py` invocation with no srun or
#   conda-run wrapper, and per-cell OUT/CKPT directories.
#
# WHAT CHANGED, and why:
#   --time 08:00:00 -> 12:00:00. MEASURED: the completed cells in this repo's
#   slurm-ldos-*.out logs top out at 427 min (7.1 h), with several at 399-407 min --
#   that is 89% of an 8 h wall. A TIMEOUT is NOT requeued by slurm (only preemption
#   and node failure are), so an 8 h wall strands the slowest cells at ~14000/15000
#   epochs with no published _best.pth. This matrix adds num_layers=4 x n_orb=2 x
#   geometry cells that are at or beyond the slowest measured configuration, so the
#   margin at 8 h is not there. 12 h costs nothing on a preemptible partition.
#
# ---------------------------------------------------------------------------

set -u

REPO=/mmfs1/gscratch/anantram/willll/G3NAT

# Stash the command line NOW. The table-construction and uniqueness-check loops
# below use `set --` to split each packed cell record, which overwrites "$@";
# without this the driver would parse an empty argument list and --submit would
# be silently ignored (or, worse, silently honoured).
SCRIPT_ARGS=("$@")

# ===========================================================================
# DESIGN TABLE -- generated, never hand-listed.
# 3 supervision x 2 n_orb x 2 num_layers x 2 geometry x 3 init seeds = 72.
# ===========================================================================
SUPERVISION_LEVELS=(dos ldos tonly)
NORB_LEVELS=(1 2)
LAYER_LEVELS=(2 4)
GEOM_LEVELS=(0 1)
SEEDS=(42 43 44)

CELLS=()
build_table() {
  CELLS=()
  local sup norb lay geom seed
  for sup in "${SUPERVISION_LEVELS[@]}"; do
    for norb in "${NORB_LEVELS[@]}"; do
      for lay in "${LAYER_LEVELS[@]}"; do
        for geom in "${GEOM_LEVELS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            CELLS+=("${sup} ${norb} ${lay} ${geom} ${seed}")
          done
        done
      done
    done
  done
}

# Loss-weight encoding of SUPERVISION. loss_a (transmission) is always 1.0.
#   dos   : DOS family at full weight, no LDOS component
#   ldos  : DOS family at full weight, half of it is LDOS
#   tonly : DOS family off entirely; loss_b is then irrelevant but is pinned to
#           0.0 so the recorded args are unambiguous rather than incidental.
loss_weights_for() {
  case "$1" in
    dos)   LOSS_A=1.0; LOSS_B=0.0; LOSS_C=1.0 ;;
    ldos)  LOSS_A=1.0; LOSS_B=0.5; LOSS_C=1.0 ;;
    tonly) LOSS_A=1.0; LOSS_B=0.0; LOSS_C=0.0 ;;
    *) echo "ERROR: unknown supervision level '$1'" >&2; return 1 ;;
  esac
}

# TAG / OUT / CKPT for a cell. The cell is readable straight off the path.
cell_paths() {
  local sup="$1" norb="$2" lay="$3" geom="$4" seed="$5"
  local gtag="nogeom"
  [ "${geom}" = "1" ] && gtag="geom"
  TAG="${sup}_n${norb}_L${lay}_${gtag}_s${seed}"
  OUT="outputs_v2/${TAG}"
  CKPT="ckpt_v2/${TAG}"
  # scripts/train.py publishes <output_dir>/<model_type>_<data_source>_model_best.pth
  # at the END of a successful run. That file is the completion sentinel; see the
  # long note at the skip gate below for why it, and not checkpoint_latest.pth, is
  # the thing to test.
  SENTINEL="${OUT}/hamiltonian_pickle_model_best.pth"
}

# Full argv for a cell. Everything defining is passed explicitly; nothing that the
# design varies is left to a default.
CMD=()
build_cmd() {
  local sup="$1" norb="$2" lay="$3" geom="$4" seed="$5"
  loss_weights_for "${sup}" || return 1
  CMD=(
    python -u scripts/train.py
    --data_source pickle
    --data_dir pickle_files_v2
    --model_type hamiltonian
    --hidden_dim 256
    --num_heads 4
    --conv_type gat
    --solver_type complex
    --log_floor 1e-38
    --floor_mode smooth
    --complex_eta 1e-12
    --use_log_outputs True
    --enforce_hermiticity True
    --ldos_target residue
    --optimizer adam
    --weight_decay 1e-5
    --learning_rate 1e-3
    --batch_size 32
    --num_epochs 15000
    --split_seed 42
    --device auto
    --n_orb "${norb}"
    --num_layers "${lay}"
    --loss_a "${LOSS_A}"
    --loss_b "${LOSS_B}"
    --loss_c "${LOSS_C}"
    --init_seed "${seed}"
    --output_dir "${OUT}"
    --checkpoint_dir "${CKPT}"
  )
  # geometry_v2.pkl is already the argparse default, and it is passed anyway:
  # a default that silently changes would repoint the geometry arm at
  # geometry.pkl, which covers 515 of the 520 v2 sequences.
  if [ "${geom}" = "1" ]; then
    CMD+=(--use_geometry --geom_cache geom_cache/geometry_v2.pkl)
  fi
  # per_base_onsite is OFF everywhere in this design: the flag is never added.
}

build_table
N_CELLS=${#CELLS[@]}

# ---------------------------------------------------------------------------
# Uniqueness self-check. Runs in BOTH modes, before anything else happens.
# A duplicate TAG means two cells share a checkpoint dir and silently resume each
# other's weights (scripts/train.py auto-resumes from checkpoint_latest.pth), which
# is a data-corruption bug that looks like a successful run.
# ---------------------------------------------------------------------------
_all_tags=""
for _c in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- ${_c}
  cell_paths "$1" "$2" "$3" "$4" "$5"
  _all_tags="${_all_tags}${TAG}"$'\n'
done
_n_uniq=$(printf '%s' "${_all_tags}" | sort -u | wc -l)
if [ "${_n_uniq}" -ne "${N_CELLS}" ]; then
  echo "FATAL: ${N_CELLS} cells produce only ${_n_uniq} unique tags. Duplicates:" >&2
  printf '%s' "${_all_tags}" | sort | uniq -d >&2
  exit 1
fi

# ===========================================================================
# JOB MODE
# ===========================================================================
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then

  i="${SLURM_ARRAY_TASK_ID}"
  if [ "$i" -ge "${N_CELLS}" ] || [ "$i" -lt 0 ]; then
    echo "ERROR: array task $i out of range (valid 0-$(( N_CELLS - 1 )), ${N_CELLS} cells)" >&2
    exit 1
  fi

  # shellcheck disable=SC2086
  set -- ${CELLS[$i]}
  SUP="$1"; NORB="$2"; LAY="$3"; GEOM="$4"; SEED="$5"
  cell_paths "${SUP}" "${NORB}" "${LAY}" "${GEOM}" "${SEED}"

  module load cuda
  source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
  conda activate g3nat
  cd "${REPO}" || { echo "ABORT: cannot cd to ${REPO}" >&2; exit 1; }

  echo "=== supmatrix task=${i} tag=${TAG} sup=${SUP} n_orb=${NORB} layers=${LAY} geom=${GEOM} seed=${SEED} ==="
  echo "    node=$(hostname) job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0}"

  # --- disk guard --------------------------------------------------------
  # 2026-07-25: 15 of 22 cells died within 60s across 11 nodes because /mmfs1 was
  # 100% full. Failed writes look like a crash, not a disk error, and burn GPU time
  # before dying. Fail loudly and early instead.
  MIN_FREE_GB=${MIN_FREE_GB:-50}
  FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "ABORT: only ${FREE_GB}GB free on $(df -h . | tail -1 | awk '{print $6}'), need ${MIN_FREE_GB}GB."
    exit 1
  fi
  echo "disk check: ${FREE_GB}GB free (min ${MIN_FREE_GB}GB) -- ok"

  # --- completion gate ---------------------------------------------------
  # Re-checked here, not only in the driver: the driver's snapshot can be minutes
  # or hours stale by the time a queued array task starts.
  if [ -e "${SENTINEL}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP: ${TAG} is already complete (${SENTINEL} exists). Set FORCE=1 to override."
    exit 0
  fi

  # --- orphan tmp reaper -------------------------------------------------
  # g3nat/training/callbacks.py writes checkpoint_best.pth.tmp and then renames it.
  # A SIGKILL (preemption, OOM, node fault) between the two leaves a zero-byte .tmp
  # that nothing ever cleans up. At job start nothing is writing it, so any .tmp
  # present is by definition an orphan.
  for _tmp in "${CKPT}"/checkpoint_best.pth.tmp "${CKPT}"/checkpoint_latest.pth.tmp; do
    if [ -e "${_tmp}" ]; then
      echo "reaping orphan partial checkpoint: ${_tmp} ($(stat -c %s "${_tmp}" 2>/dev/null) bytes)"
      rm -f "${_tmp}"
    fi
  done

  mkdir -p "${OUT}" "${CKPT}"

  # --- log-floor / flush-to-zero probe, ON THE ALLOCATED DEVICE ----------
  # These jobs land on whatever GPU the ckpt partition has free. 1e-38 is a
  # SUBNORMAL float32 (smallest normal is ~1.175e-38), so any flush-to-zero path in
  # the allocated device or in the build's kernels turns it into exactly 0.0 and
  # log10 into -inf. That does not crash: it makes every deep-tail energy point a
  # non-finite loss term and a silently skipped optimizer step, and the run finishes
  # looking normal. ABORT rather than warn.
  #
  # `if not ...: raise` on purpose, NOT `assert` -- assert is stripped under
  # python -O, which would delete the check without deleting the confidence in it.
  python -u - <<'PYEOF'
import math
import sys
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
print("log-floor probe: device=%s (%s) torch=%s" % (dev.type, name, torch.__version__))

# A GPU must actually be visible. Without this the probe merely REPORTS the
# device: if `module load cuda` faults (a recorded post-maintenance failure mode
# on this cluster, 2026-08-12), torch falls back to CPU, train.py's
# setup_device('auto') falls back the same way, and the cell burns its whole
# wall clock training on CPU, times out, is NOT requeued (timeouts never are),
# and leaves no sentinel. Fail in the first seconds instead of the twelfth hour.
if dev.type != 'cuda':
    raise RuntimeError(
        "no CUDA device visible; refusing to train on CPU. This job requested "
        "--gpus=1, so a CPU-only torch means the CUDA module or driver failed to "
        "load rather than that no GPU was allocated. Training would silently run "
        "~100x slower, hit the wall clock, and produce no published weights.")

val = float(torch.log10(torch.zeros(1, device=dev) + 1e-38).item())
print("log-floor probe: log10(0 + 1e-38) = %r" % (val,))
if not (math.isfinite(val) and abs(val - (-38.0)) < 1e-3):
    raise RuntimeError(
        "FLUSH-TO-ZERO / LOG-FLOOR CHECK FAILED on %s (%s): "
        "log10(0 + 1e-38) = %r, expected a finite value within 1e-3 of -38.0. "
        "1e-38 is subnormal in float32; this device or build is flushing it to "
        "zero, which would turn every deep-tail point into a non-finite loss term "
        "and a silently skipped optimizer step. Aborting instead of training "
        "something that only looks finished." % (name, dev.type, val))
print("log-floor probe: OK")
sys.exit(0)
PYEOF
  PROBE_RC=$?
  if [ "${PROBE_RC}" -ne 0 ]; then
    echo "=== ABORT: log-floor probe failed (rc=${PROBE_RC}) on $(hostname); tag=${TAG} not trained ==="
    exit "${PROBE_RC}"
  fi

  build_cmd "${SUP}" "${NORB}" "${LAY}" "${GEOM}" "${SEED}" || exit 1

  echo "+ ${CMD[*]}"
  SECONDS=0
  "${CMD[@]}"
  RC=$?
  echo "=== cell done: tag=${TAG} rc=${RC} wall=${SECONDS}s ($(( SECONDS / 60 ))m) best=${SENTINEL} ==="
  exit $RC
fi

# ===========================================================================
# DRIVER MODE (login-node safe: no python, no module load, no conda)
# ===========================================================================
MODE="dry-run"
FORCE=0
for arg in ${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --submit)  MODE="submit" ;;
    --force)   FORCE=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0 ;;
    *)
      echo "unknown option '${arg}'. Valid: --dry-run (default), --submit, --force" >&2
      exit 1 ;;
  esac
done

cd "${REPO}" || { echo "ABORT: cannot cd to ${REPO}" >&2; exit 1; }

echo "==========================================================================="
echo "G3NAT supervision matrix -- ${N_CELLS} runs (${#SUPERVISION_LEVELS[@]} supervision x ${#NORB_LEVELS[@]} n_orb x ${#LAYER_LEVELS[@]} num_layers x ${#GEOM_LEVELS[@]} geometry x ${#SEEDS[@]} seeds)"
echo "mode=${MODE}  force=${FORCE}  repo=${REPO}"
echo "tag uniqueness: ${_n_uniq}/${N_CELLS} unique -- OK"
echo "==========================================================================="

SUBMIT_IDX=()
SKIPPED=()
for (( i=0; i<N_CELLS; i++ )); do
  # shellcheck disable=SC2086
  set -- ${CELLS[$i]}
  cell_paths "$1" "$2" "$3" "$4" "$5"
  build_cmd "$1" "$2" "$3" "$4" "$5" || exit 1

  if [ -e "${SENTINEL}" ] && [ "${FORCE}" != "1" ]; then
    SKIPPED+=("${i}:${TAG}")
    echo ""
    echo "[${i}] ${TAG}  -- SKIP (already complete)"
    echo "      sentinel exists: ${SENTINEL}"
    echo "      re-running here would delete this cell's checkpoint_best.pth"
    echo "      (a finished run removes checkpoint_latest.pth, so train.py's"
    echo "       maybe_clear_stale_best() treats the dir as a fresh run). Pass"
    echo "      --force only if you intend to destroy this result."
    continue
  fi

  SUBMIT_IDX+=("${i}")
  echo ""
  echo "[${i}] ${TAG}"
  echo "      mkdir -p ${OUT}"
  echo "      mkdir -p ${CKPT}"
  if [ -e "${SENTINEL}" ]; then
    echo "      WARNING: sentinel EXISTS and --force was given; this run will"
    echo "               overwrite a completed cell's published best weights."
  fi
  echo "      ${CMD[*]}"
done

echo ""
echo "==========================================================================="
echo "SUMMARY"
echo "  total cells      : ${N_CELLS}"
echo "  unique tags      : ${_n_uniq}"
echo "  already complete : ${#SKIPPED[@]}"
echo "  to submit        : ${#SUBMIT_IDX[@]}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo "  skipped cells    :"
  for s in "${SKIPPED[@]}"; do echo "      ${s}"; done
fi
echo "==========================================================================="

if [ "${#SUBMIT_IDX[@]}" -eq 0 ]; then
  echo "nothing to submit."
  exit 0
fi

# Compact the index list into a slurm array spec (0-5,9,12-14 form).
ARRAY_SPEC=""
run_start=""
run_prev=""
flush_run() {
  local piece
  if [ -z "${run_start}" ]; then return; fi
  if [ "${run_start}" = "${run_prev}" ]; then piece="${run_start}"
  else piece="${run_start}-${run_prev}"; fi
  if [ -z "${ARRAY_SPEC}" ]; then ARRAY_SPEC="${piece}"
  else ARRAY_SPEC="${ARRAY_SPEC},${piece}"; fi
}
for idx in "${SUBMIT_IDX[@]}"; do
  if [ -z "${run_start}" ]; then
    run_start="${idx}"; run_prev="${idx}"
  elif [ "${idx}" -eq $(( run_prev + 1 )) ]; then
    run_prev="${idx}"
  else
    flush_run; run_start="${idx}"; run_prev="${idx}"
  fi
done
flush_run

# Concurrency throttle. DEFAULT 24, not unlimited: this account also carries
# gauNEGF array chains for another project, and 72 simultaneous GPU jobs would
# starve them for days. Three waves of 24 at ~7h each still lands the campaign
# inside a day. MAXPAR=0 disables the throttle.
MAXPAR="${MAXPAR:-24}"
if [ "${MAXPAR}" != "0" ]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAXPAR}"
fi

SBATCH_ARGS=(--array="${ARRAY_SPEC}")
if [ "${FORCE}" = "1" ]; then
  SBATCH_ARGS+=(--export=ALL,FORCE=1)
fi

if [ "${MODE}" = "dry-run" ]; then
  echo ""
  echo "WOULD SUBMIT (and did not -- dry run is the default):"
  echo "  sbatch ${SBATCH_ARGS[*]} scripts/run_supervision_matrix.sh"
  echo ""
  echo "No directory was created and no job was submitted."
  echo "To actually submit:  bash scripts/run_supervision_matrix.sh --submit"
  exit 0
fi

echo ""
echo "SUBMITTING: sbatch ${SBATCH_ARGS[*]} scripts/run_supervision_matrix.sh"
sbatch "${SBATCH_ARGS[@]}" scripts/run_supervision_matrix.sh
SRC=$?
echo "sbatch rc=${SRC}"
exit $SRC
