#!/bin/bash
#SBATCH --job-name=air_cate
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=sherlock/logs/task_%a.out
#SBATCH --error=sherlock/logs/task_%a.err

# NOTE: --array is deliberately NOT set here. SLURM caps an array at 1000 tasks
# per submission and this run needs 3600, so pass the range on the command line:
#
#   mkdir -p sherlock/logs
#   rm -f $SCRATCH/air_cate_results/partial_*.json   # stale seeds break resume
#   for r in 0-999 1000-1999 2000-2999 3000-3599; do
#       sbatch --array=$r sherlock/submit_array.sh
#   done
#
# Pilot (seeds 0-4 for all 18 DGP x method pairs = the 18 tasks with BLOCK_IDX 0):
#   sbatch --array=0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400 \
#       sherlock/submit_array.sh

# --- Array index mapping ---
# 3600 tasks = 3 DGPs x 6 methods x 200 seed-blocks of 5 seeds (999 seeds total).
# task_id -> (dgp_idx, method_idx, block_idx); block_idx varies fastest.
#
# Each task runs one (DGP, method) over 5 consecutive seeds x all 6 sample sizes.
# Worst per-seed cost on the 100-seed run was ~12.6 min (IST-3, cross-fitted MGD),
# so a 5-seed block is ~63 min worst case -- hence the 2h wall limit.

DGPS=("Weibull" "Mixture" "IST-3")
METHODS=("AIR" "Tensor-prod" "AIR-oracle" "TP-oracle" "RSF-T" "RSF-S")
N_SEEDS=999
BLOCK=5
N_BLOCKS=200   # ceil(999 / 5)

TASK_ID=${SLURM_ARRAY_TASK_ID}
BLOCK_IDX=$((TASK_ID % N_BLOCKS))
METHOD_IDX=$(((TASK_ID / N_BLOCKS) % 6))
DGP_IDX=$((TASK_ID / (N_BLOCKS * 6)))

DGP="${DGPS[$DGP_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"

# Seeds [SEED_START, SEED_START + N_THIS). The last block is short: 999 = 199*5 + 4.
SEED_START=$((BLOCK_IDX * BLOCK))
N_THIS=$((N_SEEDS - SEED_START))
if [ "$N_THIS" -gt "$BLOCK" ]; then
    N_THIS=$BLOCK
fi

echo "Task ${TASK_ID}: DGP=${DGP}, Method=${METHOD}, block=${BLOCK_IDX}," \
     "seeds ${SEED_START}..$((SEED_START + N_THIS - 1)) (${N_THIS} seeds)"
echo "Node: $(hostname), Start: $(date)"

# --- Activate conda environment ---
# Sherlock's glibc 2.17 is too old for modern jaxlib wheels, so this uses conda
# (which bundles its own glibc) rather than uv.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

cd $HOME/air_cate

# --- Output file ---
RESULTS_DIR=$SCRATCH/air_cate_results
mkdir -p $RESULTS_DIR

# Sanitize DGP name for filename (IST-3 -> IST3)
DGP_SAFE=$(echo "$DGP" | tr -d '-')
OUTFILE="${RESULTS_DIR}/partial_${DGP_SAFE}_${METHOD// /_}_b${BLOCK_IDX}.json"

# --- Run simulation ---
# Resume is on by default: a requeued task reloads whatever OUTFILE already holds
# and skips completed (n, seed) cells. That is also why stale partials from an
# earlier run must be deleted before a fresh submission.
python examples/plot_simulation.py --large \
    --dgp "$DGP" \
    --methods "$METHOD" \
    --seed-start "$SEED_START" \
    --extra-seeds "$N_THIS" \
    --output "$OUTFILE"

STATUS=$?
echo "Done: $(date), Exit code: ${STATUS}"
exit $STATUS
