#!/bin/bash
#SBATCH --job-name=air_confirm
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sherlock/logs/confirm_%a.out
#SBATCH --error=sherlock/logs/confirm_%a.err

# ============================================================================
# THE SCIENTIFIC GATE before the 999-seed run: does AIR still beat RSF-T at
# n=20000 on IST-3?
#
# The archive (100 seeds) says yes, by ~1.45x:
#
#         n        AIR   AIR-oracle      RSF-T     RSF-T/AIR
#       500     0.3271       0.1793     0.1716         0.52x
#      2000     0.3034       0.1461     0.1482         0.49x
#     10000     0.1333       0.0855     0.1287         0.97x
#     20000     0.0854       0.0664     0.1239         1.45x   <-- the claim
#
# The crossover sits between n=10000 and n=20000 and is driven by the shrinking
# nuisance-estimation penalty (AIR/AIR-oracle falls from 1.82x at n=500 to 1.29x
# at n=20000). So NO run that stops below n=10000 can test the claim at all --
# which is why this script goes to the top of the ladder while the 10-minute
# pilot in submit_pilot.sh deliberately does not.
#
# WHY THIS RUN EXISTS RATHER THAN JUST TRUSTING THE PILOT
# Local paired runs on the ported package reproduce the archive exactly at n=500
# (cross-fitted AIR 0.3300 vs 0.3271) but come in 1.35x worse at n=2000 (0.4106
# vs 0.3034). The deficit GROWS with n, and it is confined to the cross-fitted
# arms -- both oracle arms converge normally. Extrapolating that trend to n=20000
# is exactly what cannot be done from below, so it gets measured.
#
# WHAT WOULD FALSIFY THE CLAIM
#   - RSF-T/AIR at n=20000 comes in near or below 1.0 rather than ~1.45
#   - AIR's iMSE stops falling in n (the archive drops 0.303 -> 0.085 over
#     2000 -> 20000; a flat tail is the failure mode the deficit predicts)
#   - AIR/AIR-oracle stays near 2.2x at n=20000 instead of tightening to ~1.3x
# ============================================================================
#
# ---------------------------------------------------------------------------
# HOW TO RUN IT -- two sbatch calls, because the memory needs differ by method
# ---------------------------------------------------------------------------
#
#   mkdir -p sherlock/logs
#
#   # AIR + AIR-oracle: 20 tasks at 16G
#   sbatch --array=0-19 sherlock/submit_confirm.sh
#
#   # RSF-T: 10 tasks at 32G. RSF OOM'd at 8G at n=20000; 16G is untested for it,
#   # and one OOM here costs a resubmission, so this arm pays for the headroom.
#   sbatch --array=20-29 --mem-per-cpu=32G sherlock/submit_confirm.sh
#
# Splitting by memory is deliberate. Asking 32G on all 30 slots is what left an
# earlier run stuck in PD (see the transfer log in RUNBOOK.md): a 32G request
# against 1 CPU reserves roughly five cores' worth of memory on every node model
# in `normal`, so the scheduler must find 30 such slots before anything starts.
# 20 tasks at 16G backfill almost immediately.
#
# WALL CLOCK: ~15 minutes. Every task runs ONE seed of the full ladder, and all
# 30 run concurrently, so wall clock is the cost of a single slowest task, not a
# sum. Sized from the pilot's own recorded runtime_seconds on Sherlock hardware,
# fitted as runtime ~ a + b*n per method and extrapolated up the ladder:
#
#     method          500  1000  2000  5000* 10000* 20000*   ladder
#     AIR              29    44    45    73    118    207     8.6 min  <-- binds
#     AIR-oracle        9    10    13    20     34     60     2.4 min
#     RSF-T             0     1     1     2      3      6     0.2 min  (see below)
#     (* extrapolated; linear because the nuisance MLPs run a fixed epoch count,
#      so cost per epoch grows with rows while the epoch count does not)
#
# The 30-minute --time is 3.5x the 8.6 min estimate. That margin is not padding:
# the RSF figures above are fitted from sub-second measurements and are the least
# trustworthy numbers here -- RSF is superlinear in n and is the arm that OOM'd,
# so treat its 0.2 min as a lower bound, not a prediction. A task that exceeds
# --time is killed with partial output, which is the one failure mode worth
# over-insuring against.
#
# ---------------------------------------------------------------------------
# ENVIRONMENT OVERRIDES -- pass as a PREFIX, never via --export
# ---------------------------------------------------------------------------
# SLURM splits --export on commas with no way to escape them, so
# `--export=ALL,N_VALUES=500,1000` parses as N_VALUES=500 plus two bare names:
# the job would silently run n=500 only and finish suspiciously fast. A
# `VAR=... sbatch` prefix puts the variable in sbatch's own environment, and
# sbatch defaults to --export=ALL, so the job inherits it intact.
#
#   # 20 seeds instead of 10. The --array MUST be widened to match: the mapping
#   # below derives method from TASK_ID / N_SEEDS_TOTAL, so changing one without
#   # the other silently reassigns which method each task runs.
#   N_SEEDS_TOTAL=20 sbatch --array=0-39 sherlock/submit_confirm.sh
#   N_SEEDS_TOTAL=20 sbatch --array=40-59 --mem-per-cpu=32G sherlock/submit_confirm.sh

N_VALUES="${N_VALUES:-500,1000,2000,5000,10000,20000}"
SEED_START="${SEED_START:-0}"
N_SEEDS_TOTAL="${N_SEEDS_TOTAL:-10}"

DGP="IST-3"
METHODS=("AIR" "AIR-oracle" "RSF-T")

# Method varies SLOWEST, seed fastest, so each method owns a CONTIGUOUS block of
# task IDs: 0-9 AIR, 10-19 AIR-oracle, 20-29 RSF-T at the default 10 seeds. That
# is what lets the two sbatch calls above split memory with a simple range.
# (submit_pilot.sh uses the opposite convention -- method fastest -- because it
# has no seed dimension. Different mapping, deliberately.)
TASK_ID=${SLURM_ARRAY_TASK_ID}
METHOD_IDX=$((TASK_ID / N_SEEDS_TOTAL))
SEED=$((SEED_START + TASK_ID % N_SEEDS_TOTAL))

METHOD="${METHODS[$METHOD_IDX]}"
if [ -z "$METHOD" ]; then
    echo "ERROR: task ${TASK_ID} maps to method index ${METHOD_IDX}, but only" \
         "${#METHODS[@]} methods exist. Your --array is wider than" \
         "3 x N_SEEDS_TOTAL (=${N_SEEDS_TOTAL}). Fix the --array or N_SEEDS_TOTAL."
    exit 1
fi

echo "Confirm task ${TASK_ID}: DGP=${DGP}, Method=${METHOD}, seed ${SEED}," \
     "n_values=${N_VALUES}"
echo "Node: $(hostname), Start: $(date)"

# Sherlock's glibc 2.17 is too old for modern jaxlib wheels, so this uses conda
# (which bundles its own glibc) rather than uv.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

# Run from wherever sbatch was invoked, so a tree at ~/air_cate_full works with no
# edit here. AIR_CATE_DIR overrides; $HOME/air_cate is the legacy fallback.
cd "${AIR_CATE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/air_cate}}" || exit 1
echo "Running from: $(pwd)"

# Its own directory. The in-task resume logic SKIPS any (n, seed) cell already
# present in the output file, so results from a different run sitting here would
# be silently adopted as this run's answer -- the single most expensive mistake
# available on this cluster. One file per (method, seed) also means no two tasks
# ever write the same path, so there is no interleaving to reason about.
RESULTS_DIR=$SCRATCH/air_cate_confirm
mkdir -p $RESULTS_DIR

OUTFILE="${RESULTS_DIR}/confirm_IST3_${METHOD// /_}_s${SEED}.json"

# --extra-seeds 1 with --seed-start SEED runs exactly that one seed. --large
# supplies n_test=1000 and the cate_val_frac=0.2 stopping rule, matching every
# local measurement quoted in the header above.
python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods "$METHOD" \
    --seed-start "$SEED" \
    --extra-seeds 1 \
    --n-values "$N_VALUES" \
    --output "$OUTFILE"

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
