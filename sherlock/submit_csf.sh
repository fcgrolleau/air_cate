#!/bin/bash
#SBATCH --job-name=air_cate_csf
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sherlock/logs/csf_%a.out
#SBATCH --error=sherlock/logs/csf_%a.err

# The CSF shard of the trajectory-metrics run: grf's causal survival forest, which is
# the pointwise-horizon comparator the new metrics exist to characterize.
#
# WHY A SEPARATE SCRIPT rather than a 7th method in submit_array.sh. CSF refits one
# forest per horizon, so a replication is M+1 grf fits. Measured single-core locally:
# ~11 / 31 / 82 min at n = 5000 / 10000 / 20000, i.e. ~2.1h for ONE seed across the
# full 6-point ladder -- against 0.71h for an entire 5-seed AIR block. So this shard
# wants BLOCK=1 and a 6h wall. That is a different task shape, and folding it into the
# other script's index arithmetic would either give every method the 6h wall (costing
# scheduling priority on 3600 tasks) or give CSF a ragged block size inside a mapping
# that assumes a uniform one.
#
# THE 200-SEED RUN (RUNBOOK section 5b). Same RUN_TAG as the six-method shard, so
# collect_results.sh merges both in one pass -- it globs by prefix and is
# method-agnostic:
#
#   mkdir -p sherlock/logs
#   CENSORING_FIX=1 RUN_TAG=metrics200 N_SEEDS=200 sbatch --array=0-599 \
#       sherlock/submit_csf.sh
#
# N_SEEDS is not optional. It sets N_BLOCKS and hence the whole index mapping; a call
# that omits it decodes its task ids under a different layout and writes the wrong
# filenames without erroring.
#
# Stage-2 shakedown first (10 seeds, 3 x 10 = 30 tasks). Its own tag, so it cannot
# land on the 200-seed partials and be resume-skipped:
#
#   CENSORING_FIX=1 RUN_TAG=metrics10 N_SEEDS=10 sbatch --array=0-29 \
#       sherlock/submit_csf.sh
#
# Overrides must be an ENVIRONMENT PREFIX, never --export: SLURM splits --export on
# commas with no escape, so any comma-bearing value is silently truncated.

# --- Array index mapping ---
# 3 DGPs x N_SEEDS blocks of 1 seed. task_id -> (dgp_idx, block_idx = seed), with the
# seed varying fastest. At N_SEEDS=200 that is 600 tasks; at 10 it is 30.
#
# Wall limit: 6h, ~2.9x the measured 2.1h worst case, which is a wider margin than the
# 2.8x on the other shard for a thinner reason -- the 2.1h is a LOCAL single-core
# measurement on the Weibull DGP, and IST-3 (p=15, embeddings) has never been timed
# under CSF. Tightening is cheap to be wrong about either way: resume means a
# wall-clock kill keeps every (n, seed) cell already finished, so resubmitting the same
# task id continues rather than restarts. Stage 2 measures IST-3 and this comes down.
#
# Memory: 16G, unlike the other shard's 32G default. grf holds one forest at a time and
# the RSF arms are what needed 32G at n=20000. Confirm against
# `sacct -j <id> --format=JobID,MaxRSS,Elapsed` after stage 2 before trusting it for
# 600 tasks.

DGPS=("Weibull" "Mixture" "IST-3")

N_SEEDS="${N_SEEDS:-200}"
BLOCK=1                                        # one seed per task; see the header
N_BLOCKS=$((N_SEEDS / BLOCK))
N_TASKS=$((${#DGPS[@]} * N_BLOCKS))

# CSF fits no nuisance model of ours -- grf estimates its own -- so --censoring-fix
# cannot change a CSF number. The flag is still plumbed through so this shard's results
# carry the same recorded config as the arms it is compared against, and so a run
# recipe can pass one environment prefix to both scripts without a special case.
CENSORING_FIX="${CENSORING_FIX:-0}"
if [ "$CENSORING_FIX" = "1" ]; then
    CFIX_FLAG="--censoring-fix"
else
    CFIX_FLAG=""
fi

# RUN_TAG moves the scratch directory AND the filename prefix together. The in-task
# resume logic SKIPS any (n, seed) cell already present in OUTFILE, so a run writing
# another arm's filenames would load that arm's answers, find every cell present, and
# exit successful having computed nothing.
RUN_TAG="${RUN_TAG:-metrics200}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
BLOCK_IDX=$((TASK_ID % N_BLOCKS))
DGP_IDX=$((TASK_ID / N_BLOCKS))

DGP="${DGPS[$DGP_IDX]}"

if [ -z "$DGP" ]; then
    echo "ERROR: task ${TASK_ID} maps to dgp index ${DGP_IDX}, which is out of range." \
         "Valid ids are 0-$((N_TASKS - 1)) (${#DGPS[@]} DGPs x ${N_BLOCKS} blocks at" \
         "N_SEEDS=${N_SEEDS}). Fix your --array range."
    exit 1
fi

SEED_START=$((BLOCK_IDX * BLOCK))
N_THIS=$((N_SEEDS - SEED_START))
if [ "$N_THIS" -gt "$BLOCK" ]; then
    N_THIS=$BLOCK
fi

echo "Task ${TASK_ID}: DGP=${DGP}, Method=CSF, block=${BLOCK_IDX}," \
     "seeds ${SEED_START}..$((SEED_START + N_THIS - 1)) (${N_THIS} seeds)," \
     "censoring_fix=${CENSORING_FIX}, tag=${RUN_TAG}"
echo "Node: $(hostname), Start: $(date)"

# --- Activate conda environment ---
# Sherlock's glibc 2.17 is too old for modern jaxlib wheels, so this uses conda
# (which bundles its own glibc) rather than uv.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

# --- R for grf ---
# AFTER `mamba activate`, deliberately. Activating an env prepends its bin to PATH, so
# loading the module first lets conda's own R (if the env ever grows one) shadow the
# module's -- and grf is installed against the module's R, not conda's. Loading second
# puts the module ahead of the env.
#
# R_LIBS_USER is where `install.packages("grf")` put the package on the login node.
# Compute nodes do not inherit the login shell's environment, so it must be set here or
# grf will not be found even though it is installed.
module load R/4.2.0 2>/dev/null || module load R
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/x86_64-pc-linux-gnu-library/4.2}"
RSCRIPT="$(command -v Rscript)"
echo "Rscript: ${RSCRIPT:-NOT FOUND}, R_LIBS_USER: ${R_LIBS_USER}"

# Fail before the 6h wall rather than after it. The harness has its own grf_available
# check that exits with an install hint, but this reports the module state too, which
# is what actually goes wrong on a cluster.
if [ -z "$RSCRIPT" ]; then
    echo "ERROR: no Rscript on PATH after 'module load R'. Check 'module avail R'."
    exit 1
fi

cd "${AIR_CATE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/air_cate}}" || exit 1
echo "Running from: $(pwd)"

# --- Output file ---
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/air_cate_${RUN_TAG}}"
mkdir -p $RESULTS_DIR

DGP_SAFE=$(echo "$DGP" | tr -d '-')
OUTFILE="${RESULTS_DIR}/${RUN_TAG}_${DGP_SAFE}_CSF_b${BLOCK_IDX}.json"

# --- Run simulation ---
# --csf-rscript is passed explicitly rather than relying on PATH: the harness spawns
# Rscript as a subprocess, and naming the binary the module actually provided keeps the
# recorded config honest about which R produced these numbers.
#
# --csf-threads is left at its default of 1. The runtimes recorded in the results table
# are single-core and raising it would make this shard's numbers incomparable with
# them, for a speedup SLURM has already been asked for as --cpus-per-task=1 anyway.
python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods CSF \
    --csf-rscript "$RSCRIPT" \
    --seed-start "$SEED_START" \
    --extra-seeds "$N_THIS" \
    --output "$OUTFILE" \
    $CFIX_FLAG
    # CFIX_FLAG is deliberately UNQUOTED and last. Unquoted, an empty value vanishes;
    # quoted, it would hand argparse an empty-string argument.

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
