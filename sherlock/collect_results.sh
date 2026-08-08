#!/bin/bash
# Merge all partial result files and generate plots.
#
# Defaults collect the full run. Override via the environment to collect the fast
# confirmation run instead, which writes elsewhere under a different filename prefix
# and must not overwrite the real figures:
#
#   RESULTS_DIR=$SCRATCH/air_cate_pilot PATTERN='pilot_*.json' \
#   MERGED_NAME=pilot_merged.json FIGURES_DIR=examples/figures/pilot \
#       bash sherlock/collect_results.sh

source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

# Locate the tree from this script's own path, so it works from any working directory
# and from a tree at ~/air_cate_full without edits. Unlike the sbatch scripts, this one
# is invoked directly, so BASH_SOURCE is reliable here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AIR_CATE_DIR:-$(dirname "$SCRIPT_DIR")}" || exit 1
echo "Running from: $(pwd)"

RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/air_cate_results}"
PATTERN="${PATTERN:-partial_*.json}"
MERGED_NAME="${MERGED_NAME:-simulation_results_merged.json}"
MERGED="${RESULTS_DIR}/${MERGED_NAME}"

# Unquoted on purpose: this must glob into a list of paths for --merge, which takes
# nargs="+". Quoting it would pass the literal pattern as a single filename.
PARTIALS=$(ls ${RESULTS_DIR}/${PATTERN} 2>/dev/null)
N_FILES=$(echo "$PARTIALS" | wc -w)

echo "Found ${N_FILES} files matching ${PATTERN} in ${RESULTS_DIR}"

if [ "$N_FILES" -eq 0 ]; then
    echo "No matching files found. Are the jobs done?"
    exit 1
fi

# Merge
python examples/plot_simulation.py \
    --merge ${PARTIALS} \
    --output "$MERGED"

echo "Merged results saved to: $MERGED"

# Plot. FIGURES_DIR keeps a pilot's figures away from the publication ones.
FIG_ARGS=()
if [ -n "$FIGURES_DIR" ]; then
    FIG_ARGS=(--figures-dir "$FIGURES_DIR")
fi

python examples/plot_simulation.py \
    --plot-only \
    --output "$MERGED" \
    "${FIG_ARGS[@]}"

echo "Plots saved to: ${FIGURES_DIR:-examples/figures/}"
