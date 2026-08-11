#!/bin/bash
#SBATCH --job-name=air_pilot
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=sherlock/logs/pilot_%a.out
#SBATCH --error=sherlock/logs/pilot_%a.err

# 8G, NOT the full run's 32G. This is a scheduling decision, not a guess about need.
# Sherlock's `normal` nodes carry 20-64 cores against 128-384 GB, i.e. ~2-19 GB per
# core, and the cluster default is 6 GB/core. A 32 GB request against 1 CPU therefore
# reserves the memory of roughly five cores on every node model in the partition, so
# the scheduler must find 18 such slots before anything starts -- and the Sherlock
# docs are explicit that over-requesting costs queue time. The full run needs 32G
# because RSF at n=20000 OOM'd at 8G; this run stops at n=2000, a tenth of that.
# The wider n>=5000 invocation below must therefore put 32G back.

# Fast confirmation run, ~10 minutes wall clock:
#
#   mkdir -p sherlock/logs
#   sbatch --array=0-17 sherlock/submit_pilot.sh
#
# 18 tasks = 3 DGPs x 6 methods, one seed, small n only. Cost scales with n, and
# 500+1000+2000 is ~9% of the full 500..20000 ladder, so each task is ~1-3 min
# against the archive's ~8 min per (cell, seed). All 18 run concurrently, and a
# 20-minute wall request backfills quickly, so results land in roughly 10 minutes.
#
# WHAT THIS VALIDATES
#   - every (DGP, method) pair runs to completion on main's package
#   - the conda env resolves and the IST-3 super_population.npz artifact is present
#   - the array index mapping, the output paths, and collect/merge/plot round-trip
#   - iMSE decreases in n, and IST-3 AIR has not collapsed to ~0
#
# WHAT IT DOES NOT VALIDATE -- do not extrapolate from it
#   - the 32 GB ceiling. That binds on RSF at n=20000, which is skipped here.
#   - the full-run cost, or the --time setting in submit_array.sh. The open question
#     is whether main's longer oracle training (clean pseudo-outcomes keep improving,
#     so patience=30 trips later) eats the 5x saving from n_restarts=1. Answering it
#     needs a task that actually reaches n >= 5000; see the larger pilot in STATUS.md.
#
# Overridable, so this same script serves a wider run without editing it.
#
# Pass the overrides as an ENVIRONMENT PREFIX, never through --export. SLURM splits
# --export on commas and gives no way to escape them, so `--export=ALL,N_VALUES=500,1000`
# parses as N_VALUES=500 plus two bare names -- the job would silently run n=500 only and
# finish suspiciously fast. A `VAR=... sbatch` prefix puts the variable in sbatch's own
# environment, and sbatch defaults to --export=ALL, so the job inherits it intact.
#
# Split by memory rather than submitting all 18 at 32G. Only RSF needs 32G (it OOM'd at
# 8G at n=20000); the four AIR/TP methods peaked at 1.5 GiB at n=2000 and are dominated
# by fixed XLA overhead, so 16G is already generous for them. Asking for 32G on 18 slots
# is what left the first confirmation run stuck in PD -- see the transfer log in
# RUNBOOK.md. METHOD_IDX = TASK_ID % 6, and RSF-T/RSF-S are indices 4 and 5, hence the
# two array lists below.
#
#   N_VALUES=500,1000,2000,5000,10000,20000 N_SEEDS=5 \
#       sbatch --array=0-3,6-9,12-15 --time=03:00:00 --mem-per-cpu=16G \
#       sherlock/submit_pilot.sh
#
#   N_VALUES=500,1000,2000,5000,10000,20000 N_SEEDS=5 \
#       sbatch --array=4,5,10,11,16,17 --time=03:00:00 --mem-per-cpu=32G \
#       sherlock/submit_pilot.sh

N_VALUES="${N_VALUES:-500,1000,2000}"
SEED_START="${SEED_START:-0}"
N_SEEDS="${N_SEEDS:-1}"

DGPS=("Weibull" "Mixture" "IST-3")
METHODS=("AIR" "Tensor-prod" "AIR-oracle" "TP-oracle" "RSF-T" "RSF-S")

# 18 tasks, method varying fastest. Deliberately a different mapping from
# submit_array.sh, which has a 200-wide block dimension this run does not need.
TASK_ID=${SLURM_ARRAY_TASK_ID}
METHOD_IDX=$((TASK_ID % 6))
DGP_IDX=$((TASK_ID / 6))

DGP="${DGPS[$DGP_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"

echo "Pilot task ${TASK_ID}: DGP=${DGP}, Method=${METHOD}," \
     "seeds ${SEED_START}..$((SEED_START + N_SEEDS - 1)), n_values=${N_VALUES}"
echo "Node: $(hostname), Start: $(date)"

# Sherlock's glibc 2.17 is too old for modern jaxlib wheels, so this uses conda
# (which bundles its own glibc) rather than uv.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

# Run from wherever sbatch was invoked, so a tree at ~/air_cate_full works without
# editing this script. SLURM sets SLURM_SUBMIT_DIR to the submitting directory; the
# $HOME/air_cate fallback preserves the old behaviour, and AIR_CATE_DIR overrides both.
cd "${AIR_CATE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/air_cate}}" || exit 1
echo "Running from: $(pwd)"

# A separate directory from the full run's $SCRATCH/air_cate_results. The in-task
# resume logic skips (n, seed) cells already present in the output file, so pilot
# partials sitting in the real results directory would silently poison the full run.
RESULTS_DIR=$SCRATCH/air_cate_pilot
mkdir -p $RESULTS_DIR

DGP_SAFE=$(echo "$DGP" | tr -d '-')
OUTFILE="${RESULTS_DIR}/pilot_${DGP_SAFE}_${METHOD// /_}.json"

python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods "$METHOD" \
    --seed-start "$SEED_START" \
    --extra-seeds "$N_SEEDS" \
    --n-values "$N_VALUES" \
    --output "$OUTFILE"

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
