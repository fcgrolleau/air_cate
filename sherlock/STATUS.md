# Sherlock Simulation — Status & Resume Guide

## Planned run: 999 seeds (3600 tasks) on `full-sherlock-sims`

Branch `full-sherlock-sims` reruns the same design as the archived 100-seed run, but
against **main's updated `air_cate` package**. Results are therefore **not numerically
comparable** to the 100-seed run (main's `CATEMixture` uses the raw time grid and a
symmetry-breaking initializer, and the best-of-5 restart search is gone). Do not merge
old partials into this run.

### Array mapping

`3 DGPs × 6 methods × 200 seed-blocks = 3600 tasks`, where each block is 5 consecutive
seeds (999 = 199×5 + 4, so block 199 runs only 4). Block index varies fastest:

```
BLOCK_IDX  = TASK_ID % 200
METHOD_IDX = (TASK_ID / 200) % 6
DGP_IDX    = TASK_ID / (200 * 6)
SEED_START = BLOCK_IDX * 5
```

One task = one (DGP, method) × 5 seeds × all 6 sample sizes = 30 fits.

### Submission

`--array` is **not** in the script (SLURM caps arrays at 1000/submission). Pass the range:

```bash
cd ~/air_cate
mkdir -p sherlock/logs
rm -f $SCRATCH/air_cate_results/partial_*.json     # see Lessons learned
for r in 0-999 1000-1999 2000-2999 3000-3599; do
    sbatch --array=$r sherlock/submit_array.sh
done
```

### Pilot first (Phase 2)

Before the full run, submit only the 18 tasks with `BLOCK_IDX=0` — seeds 0–4 for every
(DGP, method) pair, 90 fits, ~12 CPU-hours:

```bash
sbatch --array=0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400 \
    sherlock/submit_array.sh
```

Gate before Phase 3: 18 partials written, no `Traceback` in `sherlock/logs/`, iMSE
decreasing in n for every (DGP, method), IST-3 AIR not collapsed to ~0, and
`runtime_seconds` recorded so the full-run cost can be re-estimated.

Then `rm` the pilot partials before the full submission (block 0 would otherwise
resume from them, which is harmless, but any *other* stale file is not).

### Per-task resources

- Partition: `normal` (CPU only, AMD EPYC 7502)
- 1 CPU, 32 GB RAM, **2 h** wall limit
- Worst per-seed cost measured on the 100-seed run: ~12.6 min (IST-3, cross-fitted MGD)
  → ~63 min for a 5-seed block. Average block is much cheaper (~40 min).
- Budget: ≤ ~2310 CPU-hours (999 × 2.31 CPU-h/seed from the published table); likely
  well under, since dropping the 5-restart search cuts the AIR/TP CATE stage.

### Result format (enriched)

Each entry is a dict with: `seed`, `imse`, `mse_per_timepoint` (31 values), `time_grid`,
`runtime_seconds`.

### Not bit-reproducible

`run_single` seeds python, numpy and Keras per (n, seed) cell, so the data draw and the
network initialization no longer depend on how many fits preceded them in the process —
a requeued task starts where a fresh one does. **Exact iMSE values are still not
reproducible.** Measured locally: two byte-identical invocations of the same cell
(Weibull / AIR-oracle / n=5000 / seed 0) returned 0.016127 and 0.016253 — ~1% apart.
XLA's CPU reductions are not run-to-run deterministic, and early stopping turns a tiny
numeric difference into a different stopping epoch. Across four measurements of that
cell the spread was 0.0152–0.0163 (~7%), against a between-seed SD of ~20–25%, so
averages over 999 seeds are unaffected. Do not treat a per-cell mismatch on rerun as
a bug.

## Archived run: 100 seeds (1800 tasks)

Submitted 2026-04-30 from the `sherlock-sims` branch as two batches:
- Job 23335757: tasks 0–999
- Job 23335780: tasks 1000–1799

One task = one (DGP, method, seed); ~18 min max, ~8 min average; ~231 CPU-hours total.
Produced the numbers currently in `latex/simulation_table.tex`.

## Environment

- **Miniforge** at `$GROUP_HOME/grolleau/miniforge3`
- **Conda env** `air_cate`: python 3.12, jax, keras, numpy, scikit-learn, scikit-survival, matplotlib, scipy, pytest, ruff
- **Package** installed editable (`pip install -e . --no-deps`)
- **Why not uv?** Sherlock glibc 2.17 is too old for modern jaxlib wheels. Conda bundles its own glibc.
- **IST-3 artifact**: `src/air_cate/dgp/artifacts_text/super_population.npz` (5 MB) is
  gitignored and must be present on Sherlock — it was transferred for the 100-seed run.
  `IST3TextDGP.load()` needs only that plus the tracked `artifacts_text/config.json`;
  `embeddings.npz` and the Bio_ClinicalBERT checkpoint are generator-only.

## Check progress

```bash
ssh grolleau@login.sherlock.stanford.edu

# Are jobs still running?
squeue -u $USER

# How many tasks completed?
ls $SCRATCH/air_cate_results/partial_*.json | wc -l
# Target: 18 (pilot) / 3600 (full run)

# Any errors?
grep -l "Error\|Traceback" sherlock/logs/task_*.err 2>/dev/null | head -10

# Queue limits the block sizing assumes
scontrol show config | grep -i maxarray
sacctmgr show qos format=Name,MaxSubmitJobsPerUser
```

## Merge & collect results

```bash
sh_dev -t 00:30:00 -m 8G
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate
cd ~/air_cate
bash sherlock/collect_results.sh
```

Pull to Mac:
```bash
rsync -avz grolleau@login.sherlock.stanford.edu:/scratch/users/grolleau/air_cate_results/simulation_results_merged.json \
  ~/Desktop/github_repos/git_repo_2026/air_cate/examples/figures/simulation_results.json
```

Then regenerate the table:
```bash
uv run python examples/make_table.py
```

## Key Sherlock paths

| Path | What | Quota |
|------|------|-------|
| `$HOME` = `/home/users/grolleau` | Code, scripts | 15GB |
| `$SCRATCH` = `/scratch/users/grolleau` | Job output, results | 100TB (90-day purge) |
| `$GROUP_HOME` = `/home/groups/nigam` | Miniforge install, shared env | 1TB |

## Lessons learned

- **Always `rm $SCRATCH/air_cate_results/partial_*.json` before resubmitting** — the
  in-task resume logic loads whatever the output file already holds and skips those
  (n, seed) cells, so stale partials silently poison a fresh run.
- **32GB memory** needed — RSF at n=20000 OOMs at 8GB
- **SLURM array max**: 1000 per submission — split into multiple `sbatch` calls
- **`merge_partial_results`** had a bug (assignment instead of extend) — fixed
- **`--merge` and `--plot-only`** respect the `--output` flag

## Simulation CLI reference

```bash
# Available DGPs: Weibull, Mixture, IST-3
# Available methods: AIR, Tensor-prod, AIR-oracle, TP-oracle, RSF-T, RSF-S

# One (DGP, method) over a 5-seed block — all n_values
python examples/plot_simulation.py --large \
  --dgp "Weibull" --methods "AIR" \
  --seed-start 0 --extra-seeds 5 \
  --output results_partial.json

# Merge partial results
python examples/plot_simulation.py --merge file1.json file2.json ... --output merged.json

# Plot from saved results
python examples/plot_simulation.py --plot-only --output merged.json
```
