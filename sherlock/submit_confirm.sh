#!/bin/bash
#SBATCH --job-name=air_confirm
#SBATCH --partition=normal
#SBATCH --time=01:30:00
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
# vs 0.3034). The deficit GROWS with n. Extrapolating that trend to n=20000 is
# exactly what cannot be done from below, so it gets measured.
#
# WHAT WOULD FALSIFY THE CLAIM
#   - RSF-T/AIR at n=20000 comes in near or below 1.0 rather than ~1.45
#   - AIR's iMSE stops falling in n (the archive drops 0.303 -> 0.085 over
#     2000 -> 20000; a flat tail is the failure mode the deficit predicts)
#   - AIR/AIR-oracle stays near 2.2x at n=20000 instead of tightening to ~1.3x
#
# ---------------------------------------------------------------------------
# RESULT, 2026-08-10, 10 seeds/method: THE GATE FAILED. Do not scale up.
# ---------------------------------------------------------------------------
#         n        AIR   AIR-oracle      RSF-T     RSF-T/AIR
#       500     0.3374       0.1882     0.1749         0.52x
#      1000     0.3827       0.1548     0.1559         0.41x
#      2000     0.4600       0.1512     0.1482         0.32x
#      5000     0.3969       0.1309     0.1344         0.34x
#     10000     0.2682       0.1102     0.1278         0.48x
#     20000     0.1559       0.0882     0.1239         0.79x   <-- target 1.45x
#
# Falsifier 1 fired (0.79x, not 1.45x) and falsifier 3 fired (AIR/AIR-oracle
# 1.77x, not ~1.3x). AIR is also non-monotone: it RISES to n=2000 before falling,
# and ends 1.83x above the archive's 0.0854.
#
# The load-bearing finding is that AIR-ORACLE DEGRADES TOO. It tracks the archive
# at n=500 (0.188 vs 0.179) and n=2000 (0.151 vs 0.146) but is 1.29x worse at
# n=10000 (0.110 vs 0.0855) and 1.33x worse at n=20000 (0.0882 vs 0.0664). With
# oracle nuisances there is no nuisance-estimation error to blame, so the defect
# lives in the CATE FIT, not the nuisance path. An earlier version of this header
# claimed the opposite ("confined to the cross-fitted arms"); that was inferred
# from local data topping out at n=2000 and is wrong.
#
# Suspects, all in the CATE stage and all untested at n >= 10000:
#   - n_restarts=1 vs the archive's 5. The 0.7%-spread measurement that justified
#     dropping restarts was taken on MGD at SMALL n only, and restarts demonstrably
#     mattered for TP (SE fell 3.2x with 5 restarts).
#   - Fixed capacity: K=2, text_K=4, encoder (16,) do not grow with n, so the model
#     may simply saturate while RSF keeps refining.
#   - Optimizer steps per epoch grow with n at fixed cate_epochs, so the effective
#     training budget is not held constant across the ladder.
#
# ---------------------------------------------------------------------------
# ALL THREE SUSPECTS ARE NOW MEASURED, AND NONE OF THEM WAS THE GAP
# ---------------------------------------------------------------------------
# n_restarts=5 made it WORSE (0.1559 -> 0.1670). The initializer split (helps the
# oracle arms, hurts the cross-fitted ones). Capacity is flat: K=2/4/8/16 gives
# 0.1714/0.1660/0.1649/0.1642, so 8x the components buys 4%.
#
# So "the defect lives in the CATE fit" was the wrong inference -- it followed from
# AIR-oracle degrading too, which is real, but the cross-fitted arms have a SECOND
# and much larger defect that the oracle arms cannot show. Splicing oracle
# components in one at a time (examples/diagnose_ist3_nuisance.py) puts essentially
# the whole cross-fitted gap in the CENSORING survival: at n=20000 seed 0, oracle
# censoring alone takes iMSE 0.2418 -> 0.0966 and the signal-free noise floor
# r_loss_at_zero 22.05 -> 1.007, while oracle propensity makes it slightly worse.
# The fix is LESS censoring-model flexibility, not more: see CENSORING_FIX below
# and section 4d of RUNBOOK.md.
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
# THE RESTART RERUN -- the diagnostic the failed gate below calls for. Identical in
# every respect except n_restarts, on the same seeds, into its own directory. The
# --output/--error overrides matter: the #SBATCH directives above are static, so
# without them this run would overwrite the failed gate's logs, which are evidence.
#
#   N_RESTARTS=5 RUN_TAG=confirm_r5 sbatch --array=0-19 \
#       --output=sherlock/logs/confirm_r5_%a.out \
#       --error=sherlock/logs/confirm_r5_%a.err \
#       sherlock/submit_confirm.sh
#
#   N_RESTARTS=5 RUN_TAG=confirm_r5 sbatch --array=20-29 --mem-per-cpu=32G \
#       --output=sherlock/logs/confirm_r5_%a.out \
#       --error=sherlock/logs/confirm_r5_%a.err \
#       sherlock/submit_confirm.sh
#
# then collect it under the same tag:
#
#   RESULTS_DIR=$SCRATCH/air_cate_confirm_r5 PATTERN='confirm_r5_*.json' \
#   MERGED_NAME=confirm_r5_merged.json FIGURES_DIR=examples/figures/confirm_r5 \
#       bash sherlock/collect_results.sh
#
# THE CENSORING-FIX CONFIRMATION -- the run the failed gate above actually calls for,
# now that the gap has been attributed. ~15 min, three sbatch calls, 50 tasks. Arm A
# is the new censoring spec with BOTH forest baselines; arm B is the paired control,
# same seeds and same HEAD with the flag off. Full recipe and how to read it: section
# 4d of RUNBOOK.md.
#
#   # Arm A, 16G: AIR (0-9) + AIR-oracle (10-19)
#   CENSORING_FIX=1 RUN_TAG=cfix METHODS_LIST="AIR,AIR-oracle,RSF-T,RSF-S" \
#       sbatch --array=0-19 \
#       --output=sherlock/logs/cfix_%a.out --error=sherlock/logs/cfix_%a.err \
#       sherlock/submit_confirm.sh
#
#   # Arm A, 32G: RSF-T (20-29) + RSF-S (30-39)
#   CENSORING_FIX=1 RUN_TAG=cfix METHODS_LIST="AIR,AIR-oracle,RSF-T,RSF-S" \
#       sbatch --array=20-39 --mem-per-cpu=32G \
#       --output=sherlock/logs/cfix_%a.out --error=sherlock/logs/cfix_%a.err \
#       sherlock/submit_confirm.sh
#
#   # Arm B, the control: flag OFF, AIR only, same 10 seeds
#   RUN_TAG=cbase METHODS_LIST="AIR" sbatch --array=0-9 \
#       --output=sherlock/logs/cbase_%a.out --error=sherlock/logs/cbase_%a.err \
#       sherlock/submit_confirm.sh
#
# METHODS_LIST must be IDENTICAL in both arm-A calls -- the mapping below derives the
# method from TASK_ID / N_SEEDS_TOTAL, so a shorter list in the second call would
# reassign tasks 20-39 to the wrong methods. The control arm is NOT optional: commit
# 9d336d6 changed the CATE fit after the gate above ran, so its 0.1559 is no longer a
# valid comparator. Tags are cfix/cbase rather than cfix/cfix_off because a
# 'cfix_*.json' glob would also match 'cfix_off_*.json' at collection time.
#
# Splitting by memory is deliberate. Asking 32G on all 30 slots is what left an
# earlier run stuck in PD (see the transfer log in RUNBOOK.md): a 32G request
# against 1 CPU reserves roughly five cores' worth of memory on every node model
# in `normal`, so the scheduler must find 30 such slots before anything starts.
# 20 tasks at 16G backfill almost immediately.
#
# WALL CLOCK: ~15 minutes at N_RESTARTS=1, ~20 at N_RESTARTS=5. Every task runs
# ONE seed of the full ladder, and all 30 run concurrently, so wall clock is the
# cost of a single slowest task, not a sum. Sized from the pilot's own recorded
# runtime_seconds on Sherlock hardware, fitted as runtime ~ a + b*n per method and
# extrapolated up the ladder:
#
#     method          500  1000  2000  5000* 10000* 20000*   ladder    at r5**
#     AIR              29    44    45    73    118    207     8.6 min   ~19 min
#     AIR-oracle        9    10    13    20     34     60     2.4 min   ~10 min
#     RSF-T             0     1     1     2      3      6     0.2 min   0.2 min
#     (* extrapolated; linear because the nuisance MLPs run a fixed epoch count,
#      so cost per epoch grows with rows while the epoch count does not)
#
# ** N_RESTARTS multiplies the CATE STAGE ONLY. Comparing the two arms at r1 puts
#    the CATE stage at roughly 30% of AIR (9-13s of AIR-oracle against 29-45s of
#    AIR), so r5 costs AIR about 0.7 + 0.3*5 = 2.2x and AIR-oracle, which is
#    almost all CATE, closer to 4x. RSF-T fits no CATE model and is unaffected --
#    it is rerun only so one merged file holds one config throughout.
#
# The 90-minute --time is 4.7x the 19 min r5 estimate. That margin is not padding.
# Two of the numbers above are untrustworthy: the ladder figures are a 3-point
# linear fit extrapolated 10x beyond its data, and the RSF row is fitted from
# sub-second measurements for the arm that is superlinear in n and the one that
# OOM'd. A task that exceeds --time is killed with partial output, which is the
# one failure mode worth over-insuring against; --time costs nothing but queue
# priority, since SLURM bills the time actually used.
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

# Independent CATE fits per cell, best TRAINING loss wins. The archived 100-seed run
# used 5; the ported package defaults to 1. See the RESULT block above for why this
# is now the variable under test.
N_RESTARTS="${N_RESTARTS:-1}"

# RUN_TAG moves the scratch directory AND the filename prefix together, which is the
# only safe way to vary a setting. The in-task resume logic SKIPS any (n, seed) cell
# already present in OUTFILE, so an r5 run writing r1's filenames would load r1's
# answers, find every cell present, and exit successful having computed nothing --
# reporting the old numbers as the new result. One tag, both paths, no way to change
# one and forget the other. Whatever you pass here must also be passed to
# collect_results.sh as RESULTS_DIR/PATTERN/MERGED_NAME.
RUN_TAG="${RUN_TAG:-confirm}"

# Draw CATEMixture's lambda_raw from N(0, 1), as the archived 100-seed run did,
# instead of the current N(1.0, 0.5). Set to 1 to enable. Affects AIR and
# AIR-oracle only -- the tensor-product model has no lambda, and RSF has no CATE
# model at all, so those three methods are identical under both settings and can
# be run once and shared between the arms.
LEGACY_INIT="${LEGACY_INIT:-0}"
if [ "$LEGACY_INIT" = "1" ]; then
    INIT_FLAG="--mgd-legacy-lambda-init"
else
    INIT_FLAG=""
fi

# MGD mixture components. Empty = the config default (text_K=4 on IST-3). Set it to
# sweep capacity, the one remaining suspect for the plateau that neither the restart
# nor the initializer diagnostic explained.
MGD_K="${MGD_K:-}"
if [ -n "$MGD_K" ]; then
    K_FLAG="--mgd-k $MGD_K"
else
    K_FLAG=""
fi

# Respecify the CENSORING nuisance: drop its hidden layers, so the Weibull shape
# and scale are each a softplus of a single linear index in x (32 weights) rather
# than of a (64, 32) MLP (3170 weights). Set to 1 to enable. See section 4d of
# RUNBOOK.md; local measurement at n=20000, 4 seeds, is iMSE 0.183 -> 0.110
# against an oracle-censoring ceiling of 0.098 and RSF-T's 0.1239.
#
# Affects the cross-fitted arms only. AIR-oracle and TP-oracle build their
# nuisances from the DGP and never call cross_fit_nuisances; RSF fits no nuisance
# model at all. So those arms are identical under both settings, which is what
# makes them valid unchanged controls in an A/B on this flag.
CENSORING_FIX="${CENSORING_FIX:-0}"
if [ "$CENSORING_FIX" = "1" ]; then
    CFIX_FLAG="--censoring-fix"
else
    CFIX_FLAG=""
fi

DGP="IST-3"

# Comma-separated, overridable, because the useful method set changes between
# runs: the gate needed AIR/AIR-oracle/RSF-T, an initializer A/B needs only the
# two MGD arms plus RSF-T as the benchmark to beat. Order is load-bearing -- it
# fixes which block of task IDs each method owns, so the memory split below has
# to be recomputed whenever this changes.
METHODS_LIST="${METHODS_LIST:-AIR,AIR-oracle,RSF-T}"
IFS=',' read -r -a METHODS <<< "$METHODS_LIST"

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
     "n_values=${N_VALUES}, n_restarts=${N_RESTARTS}, legacy_init=${LEGACY_INIT}," \
     "mgd_k=${MGD_K:-default}, censoring_fix=${CENSORING_FIX}, tag=${RUN_TAG}"
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

# Its own directory, keyed to RUN_TAG. The in-task resume logic SKIPS any (n, seed)
# cell already present in the output file, so results from a different run sitting
# here would be silently adopted as this run's answer -- the single most expensive
# mistake available on this cluster. One file per (method, seed) also means no two
# tasks ever write the same path, so there is no interleaving to reason about.
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/air_cate_${RUN_TAG}}"
mkdir -p $RESULTS_DIR

OUTFILE="${RESULTS_DIR}/${RUN_TAG}_IST3_${METHOD// /_}_s${SEED}.json"

# --extra-seeds 1 with --seed-start SEED runs exactly that one seed. --large
# supplies n_test=1000 and the cate_val_frac=0.2 stopping rule, matching every
# local measurement quoted in the header above. The seeds are the SAME as the r1
# run's, so the two are paired cell by cell: any difference is the restarts, not
# the draws.
python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods "$METHOD" \
    --seed-start "$SEED" \
    --extra-seeds 1 \
    --n-values "$N_VALUES" \
    --n-restarts "$N_RESTARTS" \
    --output "$OUTFILE" \
    $INIT_FLAG $K_FLAG $CFIX_FLAG
    # All three flags are deliberately UNQUOTED and last. Unquoted, an empty value
    # vanishes; quoted, it would pass argparse an empty-string argument and the
    # job would die at parse time on every task in the arm that leaves it unset.
    # K_FLAG also relies on word-splitting to become two arguments ("--mgd-k" "8").

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
