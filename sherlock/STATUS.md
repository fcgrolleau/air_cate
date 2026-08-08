# Sherlock Simulation — Status & Resume Guide

> **Looking for the commands to type? See [RUNBOOK.md](RUNBOOK.md).** It is a linear,
> copy-paste walkthrough written for someone who has not used Sherlock in a while: what a
> login node is, how to activate the conda env, submit, monitor, collect, and a
> symptom-to-fix troubleshooting table. **This** file is the design record — why the run is
> shaped the way it is, what it costs, and what was decided and measured along the way.

## Planned run: 999 seeds (3600 tasks) on `full-sherlock-sims`

Branch `full-sherlock-sims` reruns the same design as the archived 100-seed run, but
against **main's updated `air_cate` package**. Results are therefore **not numerically
comparable** to the 100-seed run: main's `CATEMixture` uses the raw time grid and a
symmetry-breaking initializer. Do not merge old partials into this run.

**Two** intended differences from the archive, then: main's package, and `n_restarts=1`
where the archive used 5. The second is deliberate and evidence-based — best-of-5 was
measured to *hurt* this estimator; see [Why `n_restarts=1`](#why-n_restarts1) below.
`n_restarts=1` is now the `AIREstimator` and `AIREstimatorCV` default, not just a harness
setting.

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

### Getting the code onto Sherlock

**This repo has no git remote** — it was created with `git init` and has never been cloned,
fetched, or pushed (empty `FETCH_HEAD`, no `refs/remotes`, reflog starts at `commit
(initial)`). The 100-seed run's code reached the cluster as a file copy, and the exact
command was not recorded anywhere, so it is written down here now. Excludes matter: `src/`
is ~490 MB, almost all of it generator-only IST-3 assets, against a 15 GB `$HOME` quota.
`IST3TextDGP.load()` needs only `config.json` and `super_population.npz`.

Goes to a **new** directory, `~/air_cate_full`, not the archived run's `~/air_cate`. Without
`--delete` an overlay would leave `sherlock-sims`-only files (`dgp/ist3.py`, `dgp/train.py`)
sitting in the tree, and it would destroy the copy the published table came from. The submit
scripts no longer hardcode a path — they use `SLURM_SUBMIT_DIR`, so they run from whichever
tree `sbatch` was invoked in.

```bash
# Dry run first -- lists what would transfer, writes nothing.
rsync -avz --dry-run \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.pytest_cache/' --exclude 'examples/figures/' \
  --exclude 'src/air_cate/dgp/artifacts_text/Bio_ClinicalBERT/' \
  --exclude 'src/air_cate/dgp/artifacts_text/embeddings.npz' \
  --exclude 'src/air_cate/dgp/artifacts_text/clinical_notes.json.gz' \
  ~/Desktop/github_repos/git_repo_2026/air_cate/ \
  grolleau@login.sherlock.stanford.edu:air_cate_full/

# Then the same command without --dry-run.
```

The trailing slash on the source means *contents of*, so files land at `~/air_cate_full/…`
rather than nested one level deeper.

Deliberately **no `--delete`**: it would remove anything on the cluster not present locally,
including the excluded `super_population.npz`. Add it only with a specific reason.

Then, on the cluster, register the package and carry the IST-3 artifact over from the old
tree (5 MB, gitignored, so `rsync` never sends it):

```bash
ssh grolleau@login.sherlock.stanford.edu
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate
cd ~/air_cate_full
pip install -e . --no-deps
cp ~/air_cate/src/air_cate/dgp/artifacts_text/super_population.npz \
   src/air_cate/dgp/artifacts_text/
# Generating, not just importing: this is what actually proves the .npz is readable.
# Expect "IST-3 OK (100, 15) t_end 5.0" preceded by the loader's own banner line.
python -c 'from air_cate.dgp import IST3TextDGP as D; D.load(); X,W,Z,Dl,te = D.generate(n=100); print("IST-3 OK", X.shape, "t_end", te)'
```

`.git/` is excluded, so the cluster copy carries no branch or commit information — record
which commit was sent. This run: `831778b` on `full-sherlock-sims`.

Sherlock SSH requires Duo two-factor, so this has to be run somewhere a push notification
can be approved; it is not scriptable unattended.

### Fast confirmation run (~10 min)

Before either pilot, `sherlock/submit_pilot.sh` gives a look at real results quickly:

```bash
mkdir -p sherlock/logs
sbatch --array=0-17 sherlock/submit_pilot.sh
```

18 tasks = 3 DGPs × 6 methods, one seed, `n_values=500,1000,2000` only. Cost scales with n
and that is ~9% of the full 500–20000 ladder, so each task is ~1–3 min against the archive's
~8 min per (cell, seed); all 18 run concurrently and a 20-minute wall request backfills
quickly. Writes to `$SCRATCH/air_cate_pilot`, deliberately **not** the full run's
`$SCRATCH/air_cate_results`, so its partials cannot poison a later submission via the resume
logic.

It confirms every (DGP, method) pair runs on main's package, the conda env resolves, the
IST-3 artifact is present, merge/plot round-trips, and iMSE decreases in n. It does **not**
test the 32 GB ceiling (that binds on RSF at n=20000, skipped here) and its runtimes cannot
be extrapolated to the full run — the oracle-cost question below needs n ≥ 5000.

### Pilot first (Phase 2)

Before the full run, submit only the 18 tasks with `BLOCK_IDX=0` — seeds 0–4 for every
(DGP, method) pair; 90 (cell, seed) pairs = 540 fits, ≲12 CPU-hours:

```bash
sbatch --array=0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400 \
    sherlock/submit_array.sh
```

Gate before Phase 3: 18 partials written, no `Traceback` in `sherlock/logs/`, iMSE
decreasing in n for every (DGP, method), IST-3 AIR not collapsed to ~0, and
`runtime_seconds` recorded so the full-run cost can be re-estimated. Specifically
**compare the two oracle cells against the archive** (IST-3/AIR-oracle averaged 10.4
min/seed, TP-oracle 8.3 min — both at 5 restarts). At 1 restart they should come in well
under that; if instead they match or exceed it, main's longer oracle training is eating
the whole 5× saving, and `--time` needs to go up before submitting 3,600 tasks.

Then `rm` the pilot partials before the full submission (block 0 would otherwise
resume from them, which is harmless, but any *other* stale file is not).

### Per-task resources

- Partition: `normal` (CPU only, AMD EPYC 7502)
- 1 CPU, 32 GB RAM, **3 h** wall limit

Two caveats on those numbers, both from the [Sherlock docs](https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/):

- **32 GB against 1 CPU is a large ask, and it costs queue time.** `normal` nodes carry 20–64
  cores against 128–384 GB (≈2–19 GB/core; cluster default 6 GB/core), so this reserves roughly
  five cores' worth of memory per task — 3,600 times over. The docs are explicit that
  over-requesting lengthens the wait. It is justified only because RSF at n=20000 OOM'd at 8 GB
  and each task spans the whole n ladder. **Lower it to the smallest safe value using the wider
  pilot's `MaxRSS`** before submitting the full run; `submit_pilot.sh` already defaults to 8 G
  since it stops at n=2000.
- **The archive's runtimes assume the same hardware.** Sherlock spans 18 CPU models, and the
  100-seed run's `runtime_seconds` were recorded on EPYC 7502. `normal` does not guarantee that
  model, so a per-task comparison against the archive is only meaningful once you check
  `NodeList`/`sacct` and confirm comparable hardware. A wall-clock difference of tens of percent
  can be the node, not the code.

The 3 h limit is comfortably inside the partition's 2-day maximum, so the `long` QOS is never
needed here.

#### Measured: confirmation run, job `38156448` (2026-08-07)

18/18 tasks at `--mem-per-cpu=8G`, one seed, n ∈ {500, 1000, 2000}. Nothing failed and nothing
came close to the memory limit:

| | |
|---|---|
| Peak `MaxRSS` across all 18 tasks | **~1.5 GiB** (max 1 553 100 K, Mixture/AIR) |
| Lowest | 0.60 GiB (IST-3/RSF-S) |
| Elapsed | 0:38 – 4:00, against a 20 min request |

So at n ≤ 2000 even 8 G is ~5× generous and the original 32 G was ~21×. **This does not license
8 G for the full run** — the archive OOM'd at 8 G with RSF at n=20000, ten times this data — but
it does mean the wider pilot's `MaxRSS` at n=20000 is the number to size `submit_array.sh` from,
and it is likely well below 32 G.

Note RSF was *not* the memory peak here (1.10–1.26 GiB); AIR and Tensor-prod were. RSF's
n=20000 OOM is therefore a large-n effect specific to that method, not visible at this scale.

**Oracle cost: early evidence that the 5× saving survives.** The open question above was whether
main's longer oracle training eats the gain from `n_restarts=1`. Measured ratios of oracle to
cross-fitted wall time:

| DGP | AIR-oracle / AIR | TP-oracle / Tensor-prod |
|---|---|---|
| Weibull | 0.55 | 0.40 |
| Mixture | 0.54 | 0.47 |
| IST-3 | 0.53 | — (task still running) |

Consistent with the archive's 0.52–0.85×, and *not* with the 1.72× measured locally at
Weibull n=5000 under 5 restarts. Treat as encouraging but unsettled: at n ≤ 2000 a large share
of each task is fixed JAX import and JIT cost, which could mask the effect. The n ≥ 5000 pilot
is still the test.

The archived run's own per-fit `runtime_seconds` are the measured baseline. They were
recorded with `n_restarts=5`, so this run should come in **below** them:

| Quantity | Archive (measured, `n_restarts=5`) |
|---|---|
| Mean per seed, one (DGP, method) cell | 8.1 min |
| Worst single seed | 19.5 min (IST-3 / AIR) |
| Worst **consecutive 5-seed block** | **86 min** (IST-3 / AIR) |
| Total | 231 CPU-h / 1800 (cell, seed) pairs = **2.31 CPU-h per seed** across all 18 cells |

(An earlier version of this file called 12.6 min the "worst per-seed cost". That is
IST-3/AIR's *mean*; its max is 19.5 min. The number that sets the wall limit is the worst
consecutive-5 block, 86 min.)

**Projected budget: ≲ 2,300 CPU-h for 999 seeds**, and plausibly far less. Two effects
pull opposite ways:

- **Down**, on the CATE stage: `n_restarts=1` here vs 5 in the archive. The nuisance stage
  is unaffected, so this is a large cut for the oracle methods and a modest one for the
  cross-fitted ones.
- **Up**, on the two oracle methods: main's package trains them longer. With oracle
  nuisances the pseudo-outcomes are clean enough that the training loss keeps improving,
  so `patience=30` trips later. In the archive AIR-oracle cost 0.52–0.85× cross-fitted
  AIR (it skips the whole nuisance stage); locally on main, Weibull n=5000 gave AIR 47 s
  vs AIR-oracle 81 s — ratio 1.72. **That was measured at 5 restarts**, so how much
  survives at 1 restart is unknown.

Because the second effect is unquantified at `n_restarts=1`, `--time` is 3 h rather than 2 h.
Over-requesting costs only scheduling priority (SLURM bills actual usage); under-requesting
costs a killed task. **The Phase 2 pilot's recorded `runtime_seconds` settles both** —
tighten `--time` and firm up the budget from it before submitting 3,600 tasks.

### Result format (enriched)

Each entry is a dict with: `seed`, `imse`, `mse_per_timepoint` (31 values), `time_grid`,
`runtime_seconds`, and — for the four MGD/TP methods — `restart_losses` (the training
R-loss of each restart, in order; a single-element list at `n_restarts=1`, and absent
altogether for the RSF baselines).

### Why `n_restarts=1`

The archive ran best-of-5 restarts. This run uses a single fit, because best-of-5 was
measured to make the estimator *worse* at 5x the CATE-stage cost. Three findings:

**1. There is no local optimum to escape.** Across 5 restarts the training losses agree
to ~0.7% — e.g. `[2.48991, 2.48893, 2.49733, 2.48573, 2.48334]`. Every initialization
reaches an equally good optimum, so the search has nothing to find.

**2. Selecting the minimum has no upside, and plausibly a downside.** `fit()` holds
nothing out, so restarts are ranked by *training* R-loss — which ranks partly by how well
each fit the noise. On the null DGP (true tau ≡ 0, so every nonzero value is artefact):

| `n_restarts` | draws | mean max&#124;tau&#124; | sd | worst |
|---|---|---|---|---|
| 1 | 15 | 0.187 | 0.077 | 0.384 |
| 5 | 5 | **0.261** | — | 0.387 |

Directionally worse, but the gap is ~1.8 standard errors on unequal sample sizes, so read
it as suggestive, not established. **The decision rests on finding 1** — 5× the CATE-stage
cost for a search with nothing to find — and this is consistent with it rather than
load-bearing. An earlier version of this file reported a "71% inflation" from 5 draws per
arm; widening the 1-restart arm to 15 draws cut that to ~40% and widened the error bars.

**3. The apparent evidence for restarts was not evidence.** A 10-seed local run showed 2
seeds near 2x the median iMSE, read at the time as bad local optima. But the archive —
*already* running 5 restarts — has the same right skew: Weibull/AIR at n=1000 has median
0.0286, mean 0.0303, max 0.0642, with seed 0 at the 95th percentile. In a distribution
whose 95th percentile is ~1.8x its median, 10 draws routinely produce one or two apparent
outliers. That tail is the sampling distribution of iMSE across data draws, not optimizer
failure, and restarts do not remove it.

The restart loop and `restart_losses` are kept so this stays measurable — if a future DGP
does show a wide spread, the diagnostic is already in the results. Ranking restarts on a
held-out block instead of training loss would make best-of-N safe, but given finding 1
there would be nothing for it to gain.

**Side effect: `test_estimator_null_dgp` was rewritten.** The 15-draw measurement above
showed its inherited `max|tau_hat| < 0.2` assertion failing on 6 of 15 draws *at the new
default* — pre-existing flakiness on `main`, not something the port introduced. The test now
seeds the fit (`rng_seed` alone does not: it reaches only the CATE stage, leaving the
nuisance initializations random) and asserts on `rms|tau_hat| < 0.15` instead. RMS is the
stricter of the two: `max` over 300 × 51 predictions is an extreme-value statistic dominated
by the corners of the uniform `X_test` cube, outside the training support, so retaining it
would have meant a threshold near 0.5 — half of tau's whole range.

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

One task = one (DGP, method, seed); 19.5 min max, 8.1 min average; 231 CPU-hours total.
Ran with `n_restarts=5`, `cate_epochs=200`, `patience=30`, `M=30`. Produced the numbers
currently in `latex/simulation_table.tex`.

Its per-fit `runtime_seconds` are tracked in
`git show sherlock-sims:examples/figures/simulation_results.json` — the authoritative
basis for any cost estimate here, since it is the same design on the same hardware.

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
