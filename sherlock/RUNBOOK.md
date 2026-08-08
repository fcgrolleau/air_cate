# Sherlock Runbook — copy-paste guide to running the simulations

**Read this one if you have forgotten how Sherlock works.** It is a linear walkthrough:
start at the top, run each block, check the stated expected output before moving on.

Companion document: [STATUS.md](STATUS.md) explains *why* the run is designed the way it is
— the array mapping, the `n_restarts=1` decision, the cost estimates, lessons learned. This
file only tells you what to type. When the two disagree about a command, this file is the one
that was last actually run.

---

## 0. Sherlock in ten lines

You need this much mental model and no more:

- **Login node** (`sh02-ln01`, what you land on via `ssh`) — shared by everyone. Edit files,
  submit jobs, run tiny checks. **Do not run simulations here**; it slows the machine for
  every other user on it.
- **Compute node** — where jobs actually run. You never ssh to one. You either submit a batch
  job (`sbatch`) or grab an interactive one (`sh_dev`).
- **SLURM** is the scheduler. `sbatch` hands it a script; it queues the work and runs it when a
  node is free. Your job waits in a queue — pending is normal, not an error.
- **Job array** — one script submitted N times with a different index each time
  (`$SLURM_ARRAY_TASK_ID`). Our scripts read that index and decode it into "which DGP, which
  method, which seeds". This is how 3,600 independent simulation cells run in parallel.
- **Three filesystems, three purposes:**

| Variable | Path | Use for | Quota | Caution |
|---|---|---|---|---|
| `$HOME` | `/home/users/grolleau` | code, scripts | **15 GB** | small — never put data here |
| `$SCRATCH` | `/scratch/users/grolleau` | job output, results | 100 TB | **purged after 90 days** |
| `$GROUP_HOME` | `/home/groups/nigam` | the Miniforge install / conda env | 1 TB | shared with the lab |

- **Duo two-factor** is required for ssh and rsync. Every connection triggers a push to your
  phone, so none of this is scriptable unattended — be at a device that can approve.
- **`uv` does not work here.** Sherlock's glibc is 2.17, too old for modern jaxlib wheels.
  Everything runs in a **conda** env (`air_cate`) under `$GROUP_HOME`, because conda ships its
  own glibc. This is the single most common source of confusion when coming from local dev.

---

## 1. Send the code from your Mac

Run this **on your Mac**, not on Sherlock.

This repo has no git remote (it was created with `git init` and never pushed anywhere), so the
code travels as a file copy. `rsync` is the right tool: it only sends what changed, so re-running
it after a local edit is cheap.

The excludes are not optional. `src/` is ~490 MB, almost all of it generator-only IST-3 assets,
against a **15 GB `$HOME` quota**. Pruned, the upload is ~10 MB. `IST3TextDGP.load()` reads only
`config.json` (tracked) and `super_population.npz` (5 MB, handled in step 2).

```bash
# --dry-run FIRST. It prints what would be copied and writes nothing.
# Drop that one flag to do it for real; everything else is identical.
#
#   -a  archive: recurse, preserve timestamps/permissions/symlinks
#   -v  verbose: name each file considered
#   -z  compress in transit
#
# The trailing slash on the SOURCE means "the contents of this directory".
# Without it, everything would land nested at ~/air_cate_full/air_cate/.
#
# Quotes around each pattern stop your local zsh from expanding e.g. *.pyc
# against the current directory before rsync ever sees it.
rsync -avz --dry-run \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.pytest_cache/' --exclude 'examples/figures/' \
  --exclude 'src/air_cate/dgp/artifacts_text/Bio_ClinicalBERT/' \
  --exclude 'src/air_cate/dgp/artifacts_text/embeddings.npz' \
  --exclude 'src/air_cate/dgp/artifacts_text/clinical_notes.json.gz' \
  ~/Desktop/github_repos/git_repo_2026/air_cate/ \
  grolleau@login.sherlock.stanford.edu:air_cate_full/
```

**Expected:** a file list ending in a `total size` on the order of **megabytes**. If you see
`Bio_ClinicalBERT` or `embeddings.npz` in the list, an `--exclude` is mistyped — fix it before
running for real, or you will push ~490 MB into a 15 GB quota.

Then run the exact same command **without `--dry-run`**.

Two deliberate choices worth knowing:

- **Target is `~/air_cate_full`, a new directory** — not the archived 100-seed run's
  `~/air_cate`. That copy produced the numbers currently in `latex/simulation_table.tex`;
  overlaying it would mix two package versions and leave `sherlock-sims`-only files
  (`dgp/ist3.py`, `dgp/train.py`) lying around.
- **No `--delete`.** It would erase anything present on the cluster but absent locally —
  including the deliberately-excluded `super_population.npz`. Add it only for a specific reason.

Because `.git/` is excluded, the cluster copy carries no commit information. Record by hand
which commit you sent; see the log at the bottom of this file.

---

## 2. One-time setup on the cluster

```bash
ssh grolleau@login.sherlock.stanford.edu    # approve the Duo push on your phone
```

Everything from here runs **on Sherlock**.

```bash
# --- Activate the conda environment ---------------------------------------
# Do this in EVERY new shell. Nothing below works without it, and the error you
# get if you forget is a confusing "No module named air_cate".
#
# `source` runs the script in the CURRENT shell rather than a subshell, which is
# required for a PATH change to survive the command.
source $GROUP_HOME/$USER/miniforge3/bin/activate

# `mamba shell hook` PRINTS the shell functions that make `mamba activate` work.
# $( ) captures that printed text; `eval` then executes it. The double quotes keep
# the multi-line output as a single argument.
eval "$(mamba shell hook --shell bash)"

mamba activate air_cate     # python 3.12, jax, keras, scikit-survival, ...
```

Your prompt should now start with `(air_cate)`. That is the check.

```bash
# --- Register the package -------------------------------------------------
cd ~/air_cate_full

# -e = editable: `import air_cate` resolves to THESE files, so a later rsync of
#      changed source takes effect with no reinstall.
# --no-deps = do not let pip resolve dependencies. Conda already provides them,
#      and pip re-resolving could silently downgrade jax and break the backend.
pip install -e . --no-deps
```

**Expected:** `Successfully installed air-cate-0.1.0`.

> Note this **repoints the `air_cate` env away from `~/air_cate`** to the new tree. That is
> intended. The archived results are already saved as JSON so nothing is at risk, but don't
> resubmit the old scripts expecting the old package.

```bash
# --- Carry over the IST-3 artifact ---------------------------------------
# super_population.npz (5 MB) is gitignored AND excluded from the rsync, so this
# copy from the archived tree is the only way it arrives. It is byte-identical to
# what the 100-seed run used, so the DGP is unchanged.
cp ~/air_cate/src/air_cate/dgp/artifacts_text/super_population.npz \
   src/air_cate/dgp/artifacts_text/

# Prove it actually loads. Generating 100 subjects tests the .npz is readable --
# a bare `import` would pass even with the file missing.
python -c 'from air_cate.dgp import IST3TextDGP as D; D.load(); X,W,Z,Dl,te = D.generate(n=100); print("IST-3 OK", X.shape, "t_end", te)'
```

**Expected, exactly:**

```
IST3TextDGP loaded: 100000 subjects, p_struct=10, p_embed=5, p_total=15
IST-3 OK (100, 15) t_end 5.0
```

**If this fails, stop.** Every IST-3 task would fail and a third of your results would come
back empty.

---

## 3. The ~10 minute confirmation run

Always run this after any transfer, before committing real compute. 18 tasks = 3 DGPs × 6
methods, one seed, at n = 500, 1000, 2000 only.

```bash
cd ~/air_cate_full

# SLURM does NOT create the directory for its --output/--error paths. Without this,
# every task fails immediately with nothing written to explain why.
# -p = create parents as needed, and do not error if it already exists.
mkdir -p sherlock/logs

# --array=0-17 runs the script 18 times with SLURM_ARRAY_TASK_ID = 0..17.
# The script decodes that index into (DGP, method) internally.
# It is passed on the command line rather than baked into the script, matching how
# submit_array.sh has to be driven (SLURM caps an array at 1000 per submission).
sbatch --array=0-17 sherlock/submit_pilot.sh
```

**Expected:** `Submitted batch job <JOBID>`. Write the job ID down — you need it for `sacct`.

The script requests 20 min, 1 CPU, 32 GB per task, and writes to `$SCRATCH/air_cate_pilot`
— deliberately **not** the full run's `$SCRATCH/air_cate_results`. The harness resumes by
skipping `(n, seed)` cells already present in an output file, so pilot partials sitting in the
real results directory would silently poison the 999-seed run.

### Watching it

```bash
# Your live jobs. Watch the ST (state) column.
#   PD = pending (queued, normal)     R = running
#   The parenthesised REASON while pending is informational, not an error:
#     (None)      just submitted, scheduler has not evaluated it yet
#     (Priority)  waiting behind higher-priority work
#     (Resources) waiting for a free node
squeue -u $USER
```

`squeue` shows only **live** jobs, so finished tasks disappear from it — and an empty listing
looks the same whether they succeeded or died. For actual outcomes use `sacct`:

```bash
# Per-task accounting, including finished and failed tasks.
#   State:   COMPLETED / FAILED / OUT_OF_MEMORY / TIMEOUT / CANCELLED
#   Elapsed: wall time
#   MaxRSS:  peak memory -- tracked per job STEP, so it shows on the `.batch`
#            rows rather than the parent rows. That is expected, not missing data.
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,NodeList
```

```bash
# How many of the 18 output files exist. `ls` expands the glob to matching paths,
# one per line; the pipe feeds them to `wc -l`, which counts lines.
ls $SCRATCH/air_cate_pilot/pilot_*.json | wc -l   # target: 18
```

> **`ls: cannot access .../air_cate_pilot: No such file or directory` is normal while
> everything is still pending.** The results directory is created by `mkdir -p` *inside* the
> job script, so it does not exist until the first task starts running.

Each task takes ~1–3 min once started, but whether all 18 start together depends on how busy
the `normal` partition is, so total wall time can exceed 10 minutes.

```bash
# Cancel if you need to.
scancel <JOBID>        # the whole array
scancel <JOBID>_3      # just task 3
```

### Collecting the results

```bash
# Failure check FIRST. Empty output = pass.
#   -l  print only the NAMES of matching files, not the matching lines
#   "Error\|Traceback" is a basic-regex alternation; \| means "or", and the double
#     quotes stop the shell from eating the backslash
#   2>/dev/null discards "no such file" noise from tasks that have not logged yet
grep -l "Error\|Traceback" sherlock/logs/pilot_*.err 2>/dev/null | head -10
```

```bash
# Merge the 18 partials into one JSON and render the convergence figure.
#
# `VAR=value VAR2=value cmd` sets variables for THAT COMMAND ONLY -- they do not
# leak into your shell. collect_results.sh reads exactly these four:
#   RESULTS_DIR  where to look            (default: the full run's directory)
#   PATTERN      which files to merge     (default: partial_*.json)
#   MERGED_NAME  output filename
#   FIGURES_DIR  where plots go  <-- KEEP THIS. Without it the pilot's 3-point
#                curves overwrite the tracked publication figures.
RESULTS_DIR=$SCRATCH/air_cate_pilot PATTERN='pilot_*.json' \
MERGED_NAME=pilot_merged.json FIGURES_DIR=examples/figures/pilot \
    bash sherlock/collect_results.sh
```

Merging 18 files on a login node is fine. Merging 3,600 is not — see step 5.

### Gates before spending more compute

Read these off the `print_summary` output:

1. **18 files, no tracebacks.** A missing IST-3 file → the step-2 artifact copy. A missing RSF
   file → scikit-survival in the env.
2. **iMSE decreasing in n** for all 18 (DGP, method) pairs. Three points is enough to see it.
3. **IST-3 / AIR not collapsed to ~0.** This specific failure mode is what main's initializer
   rework targets, and the whole reason a pilot exists.
4. **`runtime_seconds` present** on every entry.

**What this run does NOT tell you.** It stops at n=2000, so `MaxRSS` will look comfortable —
that is *not* clearance for the full run. The 32 GB request exists because RSF at n=20000 OOM'd
at 8 GB, and nothing here goes near that. Runtimes likewise do not extrapolate.

---

## 4. Wider pilot — before the full run

The open cost question is whether main's longer oracle training (with oracle nuisances the
pseudo-outcomes are clean, so the training loss keeps improving and `patience=30` trips later)
eats the 5× saving from `n_restarts=1`. It only shows up at n ≥ 5000, so the same script is
driven wider — no edits needed:

```bash
# --export=ALL,VAR=value passes environment variables into the job. ALL first means
# "keep the rest of my environment too"; omit it and the job starts nearly bare.
# Note N_VALUES contains commas -- that is fine, the script forwards the whole string
# to --n-values, which parses comma-separated ints.
sbatch --array=0-17 --time=03:00:00 \
  --export=ALL,N_VALUES=500,1000,2000,5000,10000,20000,N_SEEDS=5 \
  sherlock/submit_pilot.sh
```

This one *does* exercise the 32 GB ceiling. Compare its recorded `runtime_seconds` against the
archive's **10.4 min** (IST-3/AIR-oracle) and **8.3 min** (TP-oracle) per seed — both measured
at 5 restarts, so at 1 restart these should come in well under. If they match or exceed, raise
`--time` in `submit_array.sh` before submitting 3,600 tasks.

---

## 5. The full 999-seed run

**Only after the gates above pass.** 3,600 tasks, budget ≲2,300 CPU-hours.

```bash
cd ~/air_cate_full
mkdir -p sherlock/logs

# MANDATORY. The in-task resume logic loads whatever the output file already holds
# and SKIPS those (n, seed) cells -- so stale partials from any earlier run make a
# fresh submission silently incomplete. This has bitten this project before.
rm -f $SCRATCH/air_cate_results/partial_*.json

# SLURM caps an array at 1000 tasks per submission, so 3600 goes in four batches.
# The `for r in ...; do ... done` loop just issues four sbatch calls; $r expands to
# each range in turn.
for r in 0-999 1000-1999 2000-2999 3000-3599; do
    sbatch --array=$r sherlock/submit_array.sh
done
```

Each task = one (DGP, method) × 5 consecutive seeds × all 6 sample sizes = 30 fits. The index
decoding lives in the script; the mapping is documented in [STATUS.md](STATUS.md#array-mapping).

Then, once `squeue -u $USER` is empty and `sacct` shows 3,600 `COMPLETED`:

```bash
# 3,600 files is real work -- grab an interactive compute node instead of merging
# on the shared login node. sh_dev gives you a 30-minute, 8 GB interactive session.
sh_dev -t 00:30:00 -m 8G

# A compute node starts with a bare environment, so re-activate.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate
cd ~/air_cate_full

# Defaults are the full run, so no environment overrides here.
bash sherlock/collect_results.sh
```

```bash
# Sanity check before pulling: every (DGP, method, n) cell should hold 999 entries.
ls $SCRATCH/air_cate_results/partial_*.json | wc -l   # target: 3600
```

---

## 6. Bring the results home

Run this **on your Mac**.

```bash
# Note this overwrites the tracked publication results file. Commit or stash the
# current one first if you want to be able to go back.
rsync -avz \
  grolleau@login.sherlock.stanford.edu:/scratch/users/grolleau/air_cate_results/simulation_results_merged.json \
  ~/Desktop/github_repos/git_repo_2026/air_cate/examples/figures/simulation_results.json

# Regenerate the LaTeX table. Locally uv works fine -- it is only Sherlock that cannot use it.
cd ~/Desktop/github_repos/git_repo_2026/air_cate
uv run python examples/make_table.py
```

Then regenerate the figure, and update the `latex/simulation_study.tex` prose — it currently
still describes 2 DGPs and 20 seeds, stale in either direction.

**`$SCRATCH` is purged after 90 days.** Pull anything you care about promptly.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PD` with `(None)` / `(Priority)` / `(Resources)` | normal queueing | wait |
| `PD` with `(QOSMaxJobsPerUserLimit)` | too many jobs submitted at once | wait; they release as others finish |
| `ls: cannot access $SCRATCH/air_cate_pilot` | directory is created by the first task that *runs* | wait for `ST=R` |
| `ModuleNotFoundError: No module named 'air_cate'` | conda env not activated in this shell, or `pip install -e .` never run in it | redo step 2's activate block |
| `ModuleNotFoundError: No module named 'tensorflow'` | something imported `keras` before `air_cate`, so `KERAS_BACKEND=jax` was never set | import `air_cate` first |
| IST-3 tasks fail, others fine | `super_population.npz` missing | redo the `cp` in step 2 |
| `State = OUT_OF_MEMORY` | exceeded `--mem-per-cpu` | raise it; RSF at n=20000 needs 32 GB |
| `State = TIMEOUT` | exceeded `--time` | raise it and resubmit those tasks — the resume logic continues from the partial file rather than restarting |
| `.out`/`.err` files never appear | `sherlock/logs/` does not exist | `mkdir -p sherlock/logs`, resubmit |
| `Invalid job array specification` | array larger than 1000 | split into batches |
| Run finishes suspiciously fast, results incomplete | stale partials, skipped via resume | `rm -f $SCRATCH/air_cate_results/partial_*.json`, resubmit |
| Figures look wrong after a pilot | `FIGURES_DIR` override omitted, so pilot plots overwrote the real ones | `git checkout examples/figures/` |
| Per-cell iMSE differs slightly on rerun | XLA CPU reductions are not run-to-run deterministic; early stopping amplifies it (~1–7%) | not a bug — see [STATUS.md](STATUS.md#not-bit-reproducible) |

### Command cheat sheet

```bash
sbatch --array=0-17 script.sh    # submit an array job
squeue -u $USER                  # my live jobs
squeue -u $USER -t RUNNING       # only the running ones
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS   # outcomes, incl. finished
scancel <JOBID>                  # cancel a whole array
scancel <JOBID>_3                # cancel one task
scontrol show job <JOBID>        # everything SLURM knows about a job
sh_dev -t 00:30:00 -m 8G         # interactive compute node
scontrol show config | grep -i maxarray          # the array cap (assumed 1000)
sacctmgr show qos format=Name,MaxSubmitJobsPerUser
```

---

## Transfer log

`.git/` is excluded from the rsync, so the cluster copy has no commit information. Record each
transfer here.

| Date | Local commit | Branch | Target | Note |
|---|---|---|---|---|
| 2026-04-30 | (unrecorded) | `sherlock-sims` | `~/air_cate` | the archived 100-seed run; exact command was never written down |
| 2026-08-07 | `9c6e628` + uncommitted script edits | `full-sherlock-sims` | `~/air_cate_full` | pilot job `38154411` (`--array=0-17`) |
