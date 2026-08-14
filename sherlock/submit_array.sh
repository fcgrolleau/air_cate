#!/bin/bash
#SBATCH --job-name=air_cate
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=sherlock/logs/task_%a.out
#SBATCH --error=sherlock/logs/task_%a.err

# NOTE: --array is deliberately NOT set here. SLURM caps an array at 1000 tasks
# per submission and this run needs 3600, so pass the range on the command line.
#
# THE HEADLINE 999-SEED RUN, with the censoring fix (RUNBOOK section 4d/5):
#
#   mkdir -p sherlock/logs
#   # Six calls, not four. Splitting on the RSF blocks lets 2400 of the 3600 tasks
#   # ask for 16G instead of 32G, which is the main lever on queue time. Ranges are
#   # derived from the mapping below; see "Memory split" for why they are what they are.
#   for r in 0-799 1200-1999 2400-3199; do        # AIR, Tensor-prod, and both oracles
#       CENSORING_FIX=1 RUN_TAG=full999 sbatch --array=$r --mem-per-cpu=16G \
#           sherlock/submit_array.sh
#   done
#   for r in 800-1199 2000-2399 3200-3599; do     # RSF-T, RSF-S -- these need 32G
#       CENSORING_FIX=1 RUN_TAG=full999 sbatch --array=$r sherlock/submit_array.sh
#   done
#
# Overrides must be an ENVIRONMENT PREFIX, never --export: SLURM splits --export on
# commas with no escape, so any comma-bearing value is silently truncated.
#
# THE 200-SEED TRAJECTORY-METRICS RUN (RUNBOOK section 5b). 999 seeds is overkill --
# the full999 CIs are already very tight -- and the new metrics (derivative MSE, H1,
# excess reversals) come from the same fits, so 200 seeds buys the same figures for a
# fifth of the queue. N_SEEDS=200, BLOCK=5 -> 40 blocks -> 3 x 6 x 40 = 720 tasks, and
# CSF's 600 tasks go through submit_csf.sh with the SAME RUN_TAG so one collect merges
# both shards. Memory ranges are re-derived for 40 blocks, not copied from above:
#
#   for r in 0-159 240-399 480-639; do            # AIR, Tensor-prod, both oracles
#       CENSORING_FIX=1 RUN_TAG=metrics200 N_SEEDS=200 sbatch --array=$r \
#           --mem-per-cpu=16G sherlock/submit_array.sh
#   done
#   for r in 160-239 400-479 640-719; do          # RSF-T, RSF-S -- these need 32G
#       CENSORING_FIX=1 RUN_TAG=metrics200 N_SEEDS=200 sbatch --array=$r \
#           sherlock/submit_array.sh
#   done
#   CENSORING_FIX=1 RUN_TAG=metrics200 N_SEEDS=200 sbatch --array=0-599 \
#       sherlock/submit_csf.sh
#
# N_SEEDS must be passed to EVERY call in the run, including the CSF one. It changes
# N_BLOCKS, hence the whole index mapping: omitting it on one call makes those tasks
# decode their ids under the 200-block layout and write another arm's filenames.
#
# Pilot (seeds 0-4 for all 18 DGP x method pairs = the 18 tasks with BLOCK_IDX 0).
# GIVE IT ITS OWN TAG. Without RUN_TAG it writes full999_* files into the full run's
# directory, and if it also omits CENSORING_FIX those cells hold fix-OFF results that
# the real submission will then resume-SKIP -- one result set, two specifications,
# no error message:
#   CENSORING_FIX=1 RUN_TAG=pilot999 \
#       sbatch --array=0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400 \
#       sherlock/submit_array.sh
#
# OPTIONAL paired control arm, if you want the censoring-fix ablation on all three
# DGPs rather than only the 10-seed IST-3 A/B already in hand. Only the two
# CROSS-FITTED methods change under the flag (AIR = block 0, Tensor-prod = block 1
# within each DGP), so the control costs 1200 tasks, not another 3600 -- the oracle
# and RSF arms never call cross_fit_nuisances and are shared between the arms:
#
#   for r in 0-399 1200-1599 2400-2799; do
#       CENSORING_FIX=0 RUN_TAG=full999_cbase sbatch --array=$r --mem-per-cpu=16G \
#           sherlock/submit_array.sh
#   done

# --- Array index mapping ---
# 3 DGPs x 6 methods x ceil(N_SEEDS / BLOCK) seed-blocks. At the default 999 seeds and
# BLOCK=5 that is 200 blocks and 3600 tasks; at N_SEEDS=200 it is 40 blocks and 720.
# task_id -> (dgp_idx, method_idx, block_idx); block_idx varies fastest.
#
# EVERY range below is stated for the 3600-task layout. Change N_SEEDS and the ranges
# move with N_BLOCKS -- the 200-seed recipe above has its own, re-derived.
#
# Each task runs one (DGP, method) over 5 consecutive seeds x all 6 sample sizes.
#
# Wall limit: 2h, tightened from 3h on measurement rather than on expectation. The
# 10-seed IST-3 confirmation run (RUNBOOK 4d) recorded runtime_seconds per (method,
# n) cell. Worst observed single seed across all 6 sample sizes, x5 for a 5-seed
# block, on IST-3 -- the most expensive of the three DGPs (p=15, embeddings):
#
#   AIR 0.71h    AIR-oracle 0.25h    RSF-S 0.16h    RSF-T 0.12h
#
# So 2h is ~2.8x the worst measured task. Two caveats, both survivable:
#
#   - Tensor-prod and TP-oracle were not in that run, so they are projected from
#     nothing. They are cheaper models than MGD (fixed tensor-product basis, no
#     mixture), and TP-oracle is bounded by the same oracle-nuisance effect that
#     put AIR-oracle at 0.25h, so neither is plausibly 3x AIR.
#   - The archived 100-seed run's worst block was 86 min, but at n_restarts=5.
#     This run uses 1, and 0.71h is consistent with that 5x.
#
# Tightening is cheap to be wrong about because RESUME makes a wall-clock kill
# recoverable: a killed task's partial file keeps every (n, seed) cell it finished,
# and resubmitting the SAME task id skips them and continues. So the failure mode is
# "resubmit a handful of tasks", not "lose a seed block". Over-requesting, by
# contrast, costs scheduling priority on every one of 3600 tasks.
#
# Memory split. The default here is 32G, which is what the RSF arms need at n=20000;
# the AIR/oracle arms ran fine at 16G in the confirmation run. The default is
# deliberately the SAFE one, so the recipe above overrides DOWN to 16G for the
# non-RSF ranges: forgetting an override then costs queue time, not an OOM at hour
# two. Methods are contiguous within each DGP's 1200-task block (METHOD_IDX =
# (TASK_ID / 200) % 6), so RSF-T and RSF-S occupy blocks 4-5, i.e. the last 400 task
# ids of each DGP: 800-1199, 2000-2399, 3200-3599.

DGPS=("Weibull" "Mixture" "IST-3")
# Six methods, deliberately not seven. CSF (grf's causal survival forest) has its own
# script, sherlock/submit_csf.sh: it refits one forest per horizon, so a single seed
# across the ladder costs ~2.1h against 0.71h for a whole 5-seed AIR block. That wants
# BLOCK=1 and a much longer wall -- a different task shape, not another index here.
METHODS=("AIR" "Tensor-prod" "AIR-oracle" "TP-oracle" "RSF-T" "RSF-S")

# Overridable so the 200-seed run does not need a second copy of this script, with
# the 999-seed headline recipe as the default so it stays byte-reproducible.
N_SEEDS="${N_SEEDS:-999}"
BLOCK="${BLOCK:-5}"
N_BLOCKS=$(((N_SEEDS + BLOCK - 1) / BLOCK))   # ceil
N_TASKS=$((${#DGPS[@]} * ${#METHODS[@]} * N_BLOCKS))

# Respecify the CENSORING nuisance: drop its hidden layers, so the Weibull shape and
# scale are each a softplus of a single linear index in x (32 weights on IST-3)
# rather than of a (64, 32) MLP (3170). Set to 1 to enable. RUNBOOK section 4d.
#
# Confirmed on IST-3 at n=20000, 10 seeds, paired: cross-fitted AIR 0.1498 -> 0.0990
# against RSF-T 0.1239 and RSF-S 0.1214, lower on 10/10 seeds. Untested on the
# Weibull and Mixture DGPs, which is why the headline run turns it on for ALL THREE
# -- one specification everywhere is defensible; picking it per DGP is not. The
# optional control arm above measures what it costs on the other two.
#
# Affects the CROSS-FITTED arms only (AIR, Tensor-prod). AIR-oracle and TP-oracle
# build nuisances from the DGP and never call cross_fit_nuisances; RSF fits no
# nuisance model at all. Those four arms are therefore identical under both settings,
# which is what makes them valid unchanged controls in an A/B on this flag.
CENSORING_FIX="${CENSORING_FIX:-0}"
if [ "$CENSORING_FIX" = "1" ]; then
    CFIX_FLAG="--censoring-fix"
else
    CFIX_FLAG=""
fi

# Sample-size ladder. Unset (the default) means the harness's own LARGE_CONFIG ladder,
# six rungs to n=20000, which is what every headline run uses -- so leaving this alone
# reproduces those runs exactly. Set it to cap the ladder for a shakedown:
# N_VALUES=500,1000,2000,5000,10000 drops the top rung, which is over half the cost of
# the whole ladder and proves nothing about the pipeline that 10000 does not.
#
# NOT part of RUN_TAG, unlike CENSORING_FIX, and that asymmetry is deliberate but sharp:
# resume skips by (n, seed), so a capped run and a full run under the SAME tag compose
# correctly -- the full one fills in the missing top rung rather than being skipped. Two
# runs with DIFFERENT ladders and different specifications, however, would silently
# merge. Vary this alone, or vary the tag too.
N_VALUES="${N_VALUES:-}"
if [ -n "$N_VALUES" ]; then
    NVAL_FLAG="--n-values $N_VALUES"
else
    NVAL_FLAG=""
fi

# RUN_TAG moves the scratch directory AND the filename prefix together, which is the
# only safe way to vary a setting. The in-task resume logic SKIPS any (n, seed) cell
# already present in OUTFILE, so a censoring-fix run writing the old run's filenames
# would load the old answers, find every cell present, and exit successful having
# computed nothing -- reporting the previous numbers as the new result. One tag, both
# paths, no way to change one and forget the other.
#
# The default changed from the bare `partial_` / air_cate_results layout precisely so
# that a fresh submission CANNOT land on the archived run's partials. Whatever you
# pass here must also be passed to collect_results.sh as RESULTS_DIR/PATTERN/
# MERGED_NAME -- its defaults no longer match this script's.
RUN_TAG="${RUN_TAG:-full999}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
BLOCK_IDX=$((TASK_ID % N_BLOCKS))
METHOD_IDX=$(((TASK_ID / N_BLOCKS) % ${#METHODS[@]}))
DGP_IDX=$((TASK_ID / (N_BLOCKS * ${#METHODS[@]})))

DGP="${DGPS[$DGP_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"

# Fail loudly on an --array wider than the mapping. Without this, an out-of-range task
# gets an empty DGP, passes --dgp "" to the harness, and fails in a way that looks like
# a code bug rather than a typo in a range. Six ranges get typed by hand now, so this
# earns its keep -- and the bound moves with N_SEEDS, so it must be computed.
if [ -z "$DGP" ] || [ -z "$METHOD" ]; then
    echo "ERROR: task ${TASK_ID} maps to dgp index ${DGP_IDX} / method index" \
         "${METHOD_IDX}, which is out of range. Valid ids are 0-$((N_TASKS - 1))" \
         "(${#DGPS[@]} DGPs x ${#METHODS[@]} methods x ${N_BLOCKS} blocks" \
         "at N_SEEDS=${N_SEEDS}, BLOCK=${BLOCK}). Fix your --array range."
    exit 1
fi

# Seeds [SEED_START, SEED_START + N_THIS). The last block is short: 999 = 199*5 + 4.
SEED_START=$((BLOCK_IDX * BLOCK))
N_THIS=$((N_SEEDS - SEED_START))
if [ "$N_THIS" -gt "$BLOCK" ]; then
    N_THIS=$BLOCK
fi

echo "Task ${TASK_ID}: DGP=${DGP}, Method=${METHOD}, block=${BLOCK_IDX}," \
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

# Run from wherever sbatch was invoked, so a tree at ~/air_cate_full works without
# editing this script. SLURM sets SLURM_SUBMIT_DIR to the submitting directory; the
# $HOME/air_cate fallback preserves the old behaviour, and AIR_CATE_DIR overrides both.
cd "${AIR_CATE_DIR:-${SLURM_SUBMIT_DIR:-$HOME/air_cate}}" || exit 1
echo "Running from: $(pwd)"

# --- Output file ---
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/air_cate_${RUN_TAG}}"
mkdir -p $RESULTS_DIR

# Sanitize DGP name for filename (IST-3 -> IST3)
DGP_SAFE=$(echo "$DGP" | tr -d '-')
OUTFILE="${RESULTS_DIR}/${RUN_TAG}_${DGP_SAFE}_${METHOD// /_}_b${BLOCK_IDX}.json"

# --- Run simulation ---
# Resume is on by default: a requeued task reloads whatever OUTFILE already holds
# and skips completed (n, seed) cells. That is also why stale partials from an
# earlier run must be deleted before a fresh submission.
python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods "$METHOD" \
    --seed-start "$SEED_START" \
    --extra-seeds "$N_THIS" \
    --output "$OUTFILE" \
    $NVAL_FLAG $CFIX_FLAG
    # Both flags are deliberately UNQUOTED and last. Unquoted, an empty value vanishes;
    # quoted, it would hand argparse an empty-string argument and every task in the
    # arm that leaves the flag unset would die at parse time. NVAL_FLAG relies on the
    # same splitting to become two words ("--n-values" and the list); commas are not in
    # IFS, so the list itself survives as one argument.

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
