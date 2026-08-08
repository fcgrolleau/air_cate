#!/bin/bash
# Merge all partial result files and generate plots

source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate

cd $HOME/air_cate

RESULTS_DIR=$SCRATCH/air_cate_results
MERGED="${RESULTS_DIR}/simulation_results_merged.json"

# List partial files
PARTIALS=$(ls ${RESULTS_DIR}/partial_*.json 2>/dev/null)
N_FILES=$(echo "$PARTIALS" | wc -w)

echo "Found ${N_FILES} partial result files in ${RESULTS_DIR}"

if [ "$N_FILES" -eq 0 ]; then
    echo "No partial files found. Are the jobs done?"
    exit 1
fi

# Merge
python examples/plot_simulation.py \
    --merge ${PARTIALS} \
    --output "$MERGED"

echo "Merged results saved to: $MERGED"

# Plot
python examples/plot_simulation.py \
    --plot-only \
    --output "$MERGED"

echo "Plots saved to examples/figures/"
