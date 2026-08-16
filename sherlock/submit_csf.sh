#!/bin/bash
#SBATCH --job-name=air_cate_csf
#SBATCH --partition=normal
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=sherlock/logs/csf_%a.out
#SBATCH --error=sherlock/logs/csf_%a.err

# The CSF shard of the trajectory-metrics run: grf's causal survival forest, which is
# the pointwise-horizon comparator the new metrics exist to characterize.
#
# WHY A SEPARATE SCRIPT rather than a 7th method in submit_array.sh. CSF refits one
# forest per horizon, so a replication is M+1 grf fits. Measured on Sherlock at 10 seeds,
# single-core, max over seeds: 76 / 186 / 490 / 1986 / 6279 s at n = 500 / 1000 / 2000 /
# 5000 / 10000, i.e. ~8h for ONE seed across the full 6-point ladder -- against 0.71h for
# an entire 5-seed AIR block, a factor of ~56. So this shard wants BLOCK=1 and a 14h
# wall. That is a different task shape, and folding it into the other script's index
# arithmetic would either give every method the 14h wall (costing scheduling priority on
# 180 tasks) or give CSF a ragged block size inside a mapping that assumes a uniform one.
#
# THE 50-SEED RUN (RUNBOOK section 5b). Same RUN_TAG as the six-method shard, so
# collect_results.sh merges both in one pass -- it globs by prefix and is method-agnostic.
# 3 x 50 = 150 tasks, comfortably under SLURM's 1000-per-submission cap, so each pass is
# ONE submission of 0-149. DGP_IDX = TASK_ID / N_BLOCKS makes one DGP exactly one 50-task
# range: Weibull 0-49, Mixture 50-99, IST-3 100-149.
#
# Submitted in TWO PASSES, cheap rungs first. Both write the same tag, and resume skips
# by (n, seed), so pass 2 fills in the top rung rather than redoing anything:
#
#   mkdir -p sherlock/logs
#   # Pass 1 -- the five rungs to n=10000. Measured on Sherlock at 10 seeds: 2.5h
#   # (Weibull), 2.6h (Mixture), 1.3h (IST-3) for one seed across those five. So the
#   # wall is overridden DOWN to 5h, ~1.9x the worst, which keeps 150 short tasks out
#   # of a 14h reservation they do not need.
#   CENSORING_FIX=1 RUN_TAG=metrics50 N_SEEDS=50 \
#       N_VALUES=500,1000,2000,5000,10000 \
#       sbatch --array=0-149 --time=05:00:00 sherlock/submit_csf.sh
#
#   # Pass 2 -- the n=20000 rung alone, at this script's default 14h wall. Extrapolating
#   # the measured 3.2x-per-doubling cost puts this rung at 5.5-7h (Weibull, Mixture)
#   # and 2.5-3.2h (IST-3). 64G, not the default 16G: see "Memory" below -- and note that
#   # a failed pass 2 leaves NaN cells that resume COUNTS AS DONE, so they must be purged
#   # from the CSF files before resubmitting (RUNBOOK section 5b).
#   #
#   # --dependency is NOT optional. Task k of both passes computes the SAME OUTFILE --
#   # RUN_TAG, DGP and BLOCK_IDX are all identical, and only N_VALUES differs, which is
#   # not in the filename. Run concurrently, each task loads the file at startup, writes
#   # its own rungs, and the later writer clobbers the other pass's. `afterany` releases
#   # pass 2 once pass 1 has finished in ANY state; `afterok` would demand all 150 pass-1
#   # tasks exit 0 and deadlock the whole pass on one hiccup. Submitting pass 2 by hand
#   # after pass 1 drains is equally safe -- what is unsafe is submitting it un-gated.
#   CENSORING_FIX=1 RUN_TAG=metrics50 N_SEEDS=50 N_VALUES=20000 \
#       sbatch --dependency=afterany:<pass1_jobid> --array=0-149 \
#       --mem-per-cpu=64G sherlock/submit_csf.sh
#
# WHY TWO PASSES rather than one 14h submission of the whole ladder. The n=20000 rung is
# over half this shard's cost, so a single pass makes all 150 tasks reserve 14h to
# accommodate a rung that only the last few hours of each task spend. Splitting lets 150
# tasks queue at 5h and only the expensive rung pay the long-wall priority penalty. It
# also gives the top rung its own submission, which is where --mem-per-cpu would be
# raised if it turns out 16G does not survive n=20000 (see "Memory" below).
#
# N_SEEDS is not optional. It sets N_BLOCKS and hence the whole index mapping; a call
# that omits it decodes its task ids under a different layout and writes the wrong
# filenames without erroring -- and this script's default is 200, which is NOT this run.
#
# To EXTEND to more seeds later, do not raise N_SEEDS: that renumbers every task and
# writes a second, incompatible set of b<idx> filenames into the same directory. Submit
# under a new tag, or accept the renumbering and re-run the whole shard.
#
# Overrides must be an ENVIRONMENT PREFIX, never --export: SLURM splits --export on
# commas with no escape, so any comma-bearing value is silently truncated.
#
# Stage-2 shakedown, for reference (10 seeds, ladder capped, 3 x 10 = 30 tasks). Its own
# tag, so it cannot land on the 50-seed partials and be resume-skipped:
#
#   CENSORING_FIX=1 RUN_TAG=metrics10 N_SEEDS=10 N_VALUES=500,1000,2000,5000,10000 \
#       sbatch --array=0-29 --time=06:00:00 sherlock/submit_csf.sh

# --- Array index mapping ---
# 3 DGPs x N_SEEDS blocks of 1 seed. task_id -> (dgp_idx, block_idx = seed), with the
# seed varying fastest. At N_SEEDS=50 that is 150 tasks, one 50-task range per DGP;
# at 10 it is 30.
#
# Wall limit: 14h, RAISED from 6h on stage-2 measurement. The original 6h was 2.9x a
# LOCAL single-core projection of 2.1h for the full ladder; the cluster turned out to be
# ~3x slower than that projection (n=10000 measured 6279s against a ~31min local
# estimate), and the per-doubling cost is 3.2-4.0x, so the full six-rung ladder
# extrapolates to 8.0h (Weibull), 8.4h (Mixture), 3.9h (IST-3). 6h would have timed out
# two thirds of the shard. 14h is ~1.7x the worst extrapolation.
#
# The two-pass recipe above overrides this DOWN to 5h for the five cheap rungs, which is
# the same idiom as the other shard's memory split: the DEFAULT is the safe value, so
# forgetting an override costs queue priority rather than losing 8h of work at hour 6.
#
# Tightening is cheap to be wrong about in one direction only: resume means a wall-clock
# kill keeps every (n, seed) cell already finished, so resubmitting the same task id
# continues rather than restarts -- but a task killed INSIDE the n=20000 rung loses that
# rung entirely, which at 150 tasks is recoverable by hand but still a wasted day.
#
# Memory: 16G, which is MEASURED SUFFICIENT for the ladder capped at n=10000 and
# MEASURED INSUFFICIENT for n=20000. Stage 2 (job 39073799, five rungs to 10000) peaked at
# MaxRSS 11.6-12.4G across tasks -- 76% of the 16G request. Pass 2 therefore overrides UP
# while pass 1 keeps 16G.
#
# The scaling is QUADRATIC in n, not linear -- a 32G pass 2 (job 39190716) was OOM-killed
# on all 150 tasks after 624s, i.e. inside the FIRST of its 50 forests. baselines_csf.R
# calls causal_survival_forest without `failure.times`, so grf defaults to every unique
# event time for its internal survival and censoring curves; that grid grows with n, and
# grf itself warns about it ("consider supplying a coarser grid with the `failure.times`
# argument") once events exceed 10% of the sample. Footprint therefore goes as n x
# |failure.times| ~ n^2: 12G at n=10000 projects to ~46-50G at n=20000, hence 64G.
#
# Supplying failure.times = the shared M-grid would make this linear and much faster, and
# would also remove a confound -- as written, CSF's internal time resolution silently
# grows with n across the convergence ladder. It is not done here because it changes the
# baseline's specification and would invalidate the five completed rungs; note it in the
# paper instead.
#
# Because the peak is reached in the first forest, a 64G probe is cheap to verify: submit
# --array=0-0, wait ~15 min, and read the live peak with
#   sstat -j <jobid> --format=JobID,MaxRSS --units=G
# (sstat reports a running step; sacct's MaxRSS only appears once the step has ended).
#
# To re-measure after any change to num_trees, n_test or M (MaxRSS is reported only on
# the .batch step rows, so they cannot be filtered out -- and sacct hides jobs that
# started before today unless -S widens the window):
#   sacct -S <yyyy-mm-dd> -j <csf_jobid> --format=JobID,Elapsed,MaxRSS --units=G \
#       | grep -E 'JobID|\.batch'

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

# Sample-size ladder; see the same block in submit_array.sh. It matters more here than
# there: a CSF replication is M+1 grf fits, and the n=20000 rung alone measured ~82 min
# against ~42 min for the whole rest of the ladder combined. Capping at 10000 roughly
# halves this shard, which is what makes the stage-2 shakedown cheap.
N_VALUES="${N_VALUES:-}"
if [ -n "$N_VALUES" ]; then
    NVAL_FLAG="--n-values $N_VALUES"
else
    NVAL_FLAG=""
fi

# RUN_TAG moves the scratch directory AND the filename prefix together. The in-task
# resume logic SKIPS any (n, seed) cell already present in OUTFILE, so a run writing
# another arm's filenames would load that arm's answers, find every cell present, and
# exit successful having computed nothing.
RUN_TAG="${RUN_TAG:-metrics50}"

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
     "censoring_fix=${CENSORING_FIX}, n_values=${N_VALUES:-full ladder}," \
     "tag=${RUN_TAG}"
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
    $NVAL_FLAG $CFIX_FLAG
    # Both flags are deliberately UNQUOTED and last. Unquoted, an empty value vanishes;
    # quoted, it would hand argparse an empty-string argument. NVAL_FLAG relies on the
    # same splitting to become two words; commas are not in IFS, so the list survives.

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
