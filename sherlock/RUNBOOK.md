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

### Sizing your request — the main lever on queue time

Per the [Sherlock docs](https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/), the
`normal` partition:

| | |
|---|---|
| Cores per node | 20–64 |
| Memory per node | 128–384 GB |
| **Default memory per core** | **6 GB** |
| Default wall time | 2 hours |
| Max wall time | **2 days** (7 with the `long` QOS — we never need it) |

The consequence, and the thing to internalize: **memory, not CPU, is what makes our tasks hard
to place.** Nodes carry roughly 2–19 GB per core, so a `--mem-per-cpu=32G` request against
1 CPU reserves the memory of about *five* cores on every node model in the partition. The
scheduler has to find that much contiguous free memory, and for an 18- or 3600-task array it
has to find it that many times. The docs say it directly: *"To minimize queue time, request only
the resources necessary for your workload. Overestimating resource needs can result in longer
wait times."*

So the rule here is: **request 8G for anything that stops below n=5000, and 32G only for runs
that reach n=20000** (where RSF genuinely OOM'd at 8G). `submit_pilot.sh` defaults to 8G for
this reason; `submit_array.sh` *defaults* to 32G because each of its tasks spans the whole n
ladder and must be sized for its largest fit.

At 3,600 tasks the difference between a 16G and a 32G request is hours of queueing, so section
5's recipe splits the submission and overrides **down** to 16G for the 2,400 non-RSF tasks —
the AIR and oracle arms ran fine at 16G through n=20000 in the section 4d confirmation run.
Only the RSF arms take the 32G default. The default is deliberately the safe one: forgetting an
override costs queue time, whereas defaulting to 16G and forgetting would cost an OOM at hour
two. Read `MaxRSS` off the wider pilot (`sacct`, section 4) if you want to tighten further.

```bash
# Submission limits per partition -- the documented way to see them.
# Relevant because the full run goes in batches of ~1000.
sh_part
```

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

### Verify the transfer landed, before trusting any submission

**`Submitted batch job` proves nothing about the code.** `sbatch` validates the directives it can
see; it cannot know that the script is an older revision. And the environment overrides these
scripts are driven by fail *silently* when absent — an old script simply ignores `MGD_K=8` and
runs its default, so a five-arm experiment becomes five identical reruns, in five correctly-named
directories, with nothing anywhere to say why. This cost a submit-cancel-resubmit cycle on
2026-08-11.

So after every rsync, grep the cluster's copy for whatever variable the run depends on. `^`
anchors to line start, so the many mentions in the header comments do not count:

```bash
grep -c '^MGD_K=' sherlock/submit_confirm.sh          # want 1
grep -c '^LEGACY_INIT=' sherlock/submit_confirm.sh    # want 1
grep -c '^METHODS_LIST=' sherlock/submit_confirm.sh   # want 1
```

Do not use `#SBATCH --time` as the tell — it changes rarely, so it matches across several
revisions and reads as reassurance when the script is in fact stale.

Once tasks start, each one echoes its own resolved settings, which is the check that an
experiment was actually an experiment:

```bash
grep -h "Confirm task 0:" sherlock/logs/night_*_0.out
```

Five identical lines there means the arms did not differ and no number in the output is worth
reading. If a run did go out on a stale script, `scancel` it and then **delete the partials**
(`rm -rf $SCRATCH/air_cate_<tag>*`) before resubmitting: the in-task resume logic skips any
(n, seed) cell already present in the output file, so one stale file is silently adopted as the
new run's answer for that cell.

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

**Expected:** `Submitted batch job 38156448` (your number will differ). Immediately store it in a
shell variable, so every monitoring command below can be pasted verbatim:

```bash
# Replace the number with whatever sbatch just printed.
# Do this in each new shell -- shell variables do not survive logout.
JOBID=38156448
```

> Commands below use `$JOBID`, never a `<PLACEHOLDER>`. If you ever paste something like
> `sacct -j <JOBID>` literally, bash reads `<` as an input redirect and fails with
> `-bash: JOBID: No such file or directory`. Harmless, but confusing.

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

**How to read the array notation.** While every task is pending, the whole array collapses to
one row, `38156448_[0-17]`. As tasks start, the bracketed range shrinks from the left: a row
reading `38156448_[2-17]` means **tasks 0 and 1 have already left the pending state** — this is
your first sign of progress, and it appears well before any output file does. Narrow terminal
columns truncate the range with a `+`, so `38156448_[2+` is the same thing. Started tasks then
appear as their own rows, `38156448_0`, `38156448_1`, …

`squeue` shows only **live** jobs, so finished tasks disappear from it — and an empty listing
looks the same whether they succeeded or died. For actual outcomes use `sacct`:

```bash
# Per-task accounting, including finished and failed tasks.
#   State:   COMPLETED / FAILED / OUT_OF_MEMORY / TIMEOUT / CANCELLED
#   Elapsed: wall time
#   MaxRSS:  peak memory -- tracked per job STEP, so it shows on the `.batch`
#            rows rather than the parent rows. That is expected, not missing data.
#
# Note the accounting database lags a few seconds behind squeue, so a task that
# squeue already shows as running may not have a row here yet.
sacct -j $JOBID --format=JobID,State,Elapsed,MaxRSS,NodeList
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
scancel $JOBID         # the whole array
scancel ${JOBID}_3     # just task 3 -- braces so the shell does not look for
                       # a variable named JOBID_3
```

Cancelling a fully-pending array costs nothing (`TIME 0:00` = no work done), so if the queue
wait looks wrong, diagnose and resubmit rather than waiting it out.

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

## 4. The ~15 minute scientific gate — IST-3 to n=20000

**This is the run that decides whether the full 999-seed run is worth submitting.** Section 3
proves the pipeline executes; this one asks whether the *result* still holds.

The claim under test: on IST-3, AIR beats RSF-T at n=20000 by ~1.45× (archive, 100 seeds:
AIR 0.0854 vs RSF-T 0.1239). The crossover sits between n=10000 and n=20000, so **no run that
stops below n=10000 can test it** — which is why section 3's pilot cannot substitute.

Why it needs measuring rather than assuming: local paired runs on the ported package reproduce
the archive *exactly* at n=500 (cross-fitted AIR 0.3300 vs 0.3271) but come in 1.35× worse at
n=2000 (0.4106 vs 0.3034). The deficit grows with n, and extrapolating that trend upward is
precisely what cannot be done from below.

> **This gate has been run, and it FAILED.** See section 4b for the numbers and the follow-up
> diagnostic. Section 5's full run is blocked until 4b resolves. An earlier version of this
> section added that the deficit "is confined to the cross-fitted arms — both oracle arms
> converge normally"; the gate falsified that, and the sentence is gone.

```bash
mkdir -p sherlock/logs

# AIR + AIR-oracle -- 20 tasks at the script's 16G default.
sbatch --array=0-19 sherlock/submit_confirm.sh

# RSF-T -- 10 tasks at 32G. RSF OOM'd at 8G at n=20000 and 16G is untested for it.
sbatch --array=20-29 --mem-per-cpu=32G sherlock/submit_confirm.sh
```

Two calls, not one, because asking 32G on all 30 slots is what left an earlier run stuck in
`PD` (see the transfer log). 20 tasks at 16G backfill almost immediately.

Each task runs **one seed of the full ladder** and all 30 run concurrently, so wall clock is
one slowest task (~8.6 min, from the pilot's own recorded `runtime_seconds` extrapolated up
the ladder) rather than a sum. Ten seeds per method is the statistical power; the `--time` is
set to 30 min because the RSF timings are fitted from sub-second measurements and are the
least trustworthy numbers in the estimate.

If you want more seeds, **widen `--array` and `N_SEEDS_TOTAL` together** — the script derives
the method from `TASK_ID / N_SEEDS_TOTAL`, so changing one alone silently reassigns which
method each task runs:

```bash
# Pass overrides as an ENVIRONMENT PREFIX, never through --export.
#
# SLURM splits --export on commas with no way to escape them, so
# `--export=ALL,N_VALUES=500,1000` parses as N_VALUES=500 plus two bare names --
# the job would silently run n=500 only and finish suspiciously fast. A `VAR=... sbatch`
# prefix puts the variable in sbatch's own environment, and sbatch defaults to
# --export=ALL, so the job inherits it intact.
N_SEEDS_TOTAL=20 sbatch --array=0-39 sherlock/submit_confirm.sh
N_SEEDS_TOTAL=20 sbatch --array=40-59 --mem-per-cpu=32G sherlock/submit_confirm.sh
```

Collect it into its own directory and figures folder, so nothing touches the publication
outputs:

```bash
RESULTS_DIR=$SCRATCH/air_cate_confirm PATTERN='confirm_*.json' \
MERGED_NAME=confirm_merged.json FIGURES_DIR=examples/figures/confirm \
  bash sherlock/collect_results.sh
```

**Read it against these three numbers.** The run has failed the gate if any hold:

| check | archive says | failure looks like |
|---|---|---|
| RSF-T / AIR at n=20000 | ~1.45× | at or below 1.0 |
| AIR falls with n | 0.303 (n=2000) → 0.085 (n=20000) | a flat tail |
| AIR / AIR-oracle at n=20000 | ~1.29× | still ~2.2×, as at n=2000 |

Also record `MaxRSS` (`sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State`) — it is what
should set the full run's memory request, and this run is the only one that reaches n=20000
before the full submission.

---

## 4b. Gate result (2026-08-10) and the restart diagnostic

**The gate failed.** 10 seeds per method, full ladder:

| n | AIR | AIR-oracle | RSF-T | RSF-T / AIR |
|---|---|---|---|---|
| 500 | 0.3374 | 0.1882 | 0.1749 | 0.52× |
| 1000 | 0.3827 | 0.1548 | 0.1559 | 0.41× |
| 2000 | 0.4600 | 0.1512 | 0.1482 | 0.32× |
| 5000 | 0.3969 | 0.1309 | 0.1344 | 0.34× |
| 10000 | 0.2682 | 0.1102 | 0.1278 | 0.48× |
| 20000 | 0.1559 | 0.0882 | 0.1239 | **0.79×** ← target 1.45× |

Checks 1 and 3 both fired: RSF-T/AIR is 0.79× rather than ~1.45×, and AIR/AIR-oracle is 1.77×
rather than ~1.29×. AIR is also non-monotone — it *rises* to n=2000 before falling — and ends
1.83× above the archive's 0.0854.

**The finding that redirects the search: AIR-oracle degrades too.** It tracks the archive at
n=500 (0.188 vs 0.179) and n=2000 (0.151 vs 0.146), but is 1.29× worse at n=10000 (0.110 vs
0.0855) and 1.33× worse at n=20000 (0.0882 vs 0.0664). The oracle arm is handed the true
nuisances, so there is no nuisance-estimation error available to blame. That exonerates
`weibull_aft.py`, `cross_fit.py` and the propensity model at large n and puts the defect in the
**CATE fit**.

The first suspect is `n_restarts`, because it is a *known* difference from the archive rather
than a hypothesis: the archive used 5, the ported package defaults to 1, and the measurement
that justified dropping it (restarts landing within ~0.7% of each other in training loss) was
taken on MGD at n ≤ 2000 — exactly the regime where the two runs still agree. Restarts also
demonstrably mattered for the tensor-product model, whose standard error fell 3.2× with 5 of
them.

Same seeds, same ladder, one variable changed, into its own directory:

```bash
# ~20 min. N_RESTARTS multiplies the CATE stage only, so AIR costs ~2.2x and
# AIR-oracle ~4x; RSF-T fits no CATE model and is unaffected.
#
# The --output/--error overrides are not optional bookkeeping: the #SBATCH
# directives in the script are static, so without them this run overwrites the
# failed gate's logs in sherlock/logs/confirm_*.{out,err}.
N_RESTARTS=5 RUN_TAG=confirm_r5 sbatch --array=0-19 \
  --output=sherlock/logs/confirm_r5_%a.out \
  --error=sherlock/logs/confirm_r5_%a.err \
  sherlock/submit_confirm.sh

N_RESTARTS=5 RUN_TAG=confirm_r5 sbatch --array=20-29 --mem-per-cpu=32G \
  --output=sherlock/logs/confirm_r5_%a.out \
  --error=sherlock/logs/confirm_r5_%a.err \
  sherlock/submit_confirm.sh
```

`RUN_TAG` moves the scratch directory *and* the filename prefix together. Do not point a
restart run at `confirm_*` filenames: the in-task resume logic skips any (n, seed) cell already
present in the output file, so it would load the `n_restarts=1` answers, find every cell
present, exit successful having computed nothing, and report the old numbers as the new result.

```bash
RESULTS_DIR=$SCRATCH/air_cate_confirm_r5 PATTERN='confirm_r5_*.json' \
MERGED_NAME=confirm_r5_merged.json FIGURES_DIR=examples/figures/confirm_r5 \
  bash sherlock/collect_results.sh
```

**Read it against the r1 column above, at n=20000 specifically.** Restarts explain the gap if
AIR lands near 0.085 and RSF-T/AIR climbs back above 1.0. If AIR stays near 0.156, restarts are
not the cause and the next suspect is model capacity — `K=2` and encoder `(16,)` are fixed while
n grows 40×, and a saturating model looks exactly like this. That one needs a small sweep over
`K` rather than a single rerun, so it is worth ruling restarts out first.

---

## 4c. Restart + initializer results, and the overnight capacity sweep

### What the two diagnostics settled

**Restarts are not the cause.** `n_restarts=5` on the same 10 seeds moved AIR the wrong way at
the only n that matters: 0.1559 → 0.1670 at n=20000. RSF-T reproduced 0.123899 to six digits
across both runs, which is the control that proves only the intended variable moved. AIR-oracle
improved 8.5% but remained 1.22× the archive.

**The initializer explains the oracle arm, and only the oracle arm.** `CATEMixture`'s
`lambda_raw` was `RandomNormal(stddev=1.0)` in the archive; `fb6c345` changed it to
`RandomNormal(mean=1.0, stddev=0.5)` while fixing an unrelated `initializer="ones"` bug — so the
current default was measured against a *constant* initializer, never against the archive's. It
is not a neutral change, because `lambda = softplus(lambda_raw)` and softplus compresses:

| | λ mean | λ sd | cv | λ p5–p95 |
|---|---|---|---|---|
| `N(0, 1.0)` (archive) | 0.804 | 0.515 | **0.641** | [0.18, 1.82] |
| `N(1.0, 0.5)` (current) | 1.336 | 0.361 | **0.270** | [0.79, 1.97] |

The current default starts the K components with 2.4× less relative spread in their decay rates
and cannot reach the slow-decay half of the range at all. Paired local A/B, IST-3, 6 seeds,
identical config otherwise (`--mgd-legacy-lambda-init`):

| method | n | current | archive init | paired t | vs archive result |
|---|---|---|---|---|---|
| AIR-oracle | 2000 | 0.1497 | **0.1401** | −2.42 | 1.02× → **0.96×** |
| AIR-oracle | 5000 | 0.1329 | **0.1240** | −2.54 | 1.16× → **1.08×** |
| AIR | 2000 | **0.5722** | 0.6112 | +0.33 | 1.89× → 2.01× |
| AIR | 5000 | **0.3876** | 0.4643 | +3.83 | 1.52× → 1.83× |

Read the sign carefully — the verdict is **split, and consistently so**. The archive's wider
init is better with oracle nuisances (5/6 seeds at both n) and worse with cross-fitted ones (0/6
seeds at n=5000, t=+3.83). That is one coherent story rather than two contradictory ones: the
wider init buys reach across temporal shapes, which helps when the pseudo-outcomes are clean and
hurts when they are noisy, because the extra reach is spent fitting the noise. **So the
initializer is not the fix for the gate**, which is a cross-fitted result. Keep the current
default; `--mgd-legacy-lambda-init` stays as a diagnostic.

The same table shows where the real problem is. Cross-fitted AIR's per-seed iMSE at n=2000 was
0.244, 0.356, 0.394, 0.416, 0.865, 1.157 — a 2.9× spread with two clear divergences, against an
oracle arm sitting in 0.140–0.169. The cross-fitted CATE fit is *unstable*, not merely biased,
and no initializer choice addresses that.

### What runs overnight

Capacity is the last untested suspect, and it is a good one: `text_K=4` and encoder `(16,)` are
fixed while n grows 40× along the ladder, and a model that saturates looks exactly like the
plateau we are chasing. `--mgd-k` (new) overrides **both** `K` and `text_K`, because IST-3 reads
`text_K` and setting only `K` is a silent no-op on this DGP.

Five arms — four capacities at the current initializer, plus K=4 at the archive initializer to
re-test the split verdict at n=10000 and 20000, which the local run could not reach. 25 seeds
per cell, full ladder, `n_restarts=1`, AIR and AIR-oracle. 250 tasks, all at the 16G default, so
there is no 32G request to queue behind (that is what stalled an earlier run — see the transfer
log).

RSF-T is deliberately **not** rerun. It has already reproduced **0.1239** at n=20000 to six
digits in two independent runs, so it is a settled benchmark, and skipping it keeps every task in
this submission at the same memory footprint.

```bash
mkdir -p sherlock/logs

# One sbatch per arm. "cur:8" means current initializer, K=8.
#   ${ARM%%:*} strips the longest trailing match of ":*"  -> the init name
#   ${ARM##*:} strips the longest leading match of "*:"   -> the K value
# --array=0-49 is 2 methods x 25 seeds; the script maps task/25 to the method,
# so widening the seed count REQUIRES widening the array to match.
for ARM in "cur:2" "cur:4" "cur:8" "cur:16" "leg:4"; do
  INIT="${ARM%%:*}"; K="${ARM##*:}"
  LEG=0; if [ "$INIT" = "leg" ]; then LEG=1; fi
  TAG="night_${INIT}_k${K}"
  LEGACY_INIT=$LEG MGD_K=$K RUN_TAG=$TAG N_SEEDS_TOTAL=25 \
  METHODS_LIST="AIR,AIR-oracle" \
    sbatch --array=0-49 \
      --output=sherlock/logs/${TAG}_%a.out \
      --error=sherlock/logs/${TAG}_%a.err \
      sherlock/submit_confirm.sh
done
```

Each arm writes to `$SCRATCH/air_cate_night_<init>_k<K>/` with a matching filename prefix.
`RUN_TAG` moves the directory *and* the prefix together, which is the only safe way to vary a
setting here: the in-task resume logic skips any (n, seed) cell already present in the output
file, so two arms sharing a path would make the second one load the first's answers, find every
cell present, exit successful having computed nothing, and report the old numbers as the new
result.

Confirm all five landed, then collect in the morning:

```bash
squeue -u $USER | wc -l                      # 251 while all are queued/running
ls $SCRATCH/air_cate_night_*/ | grep -c json # 250 when finished

grep -l "Error\|Traceback" sherlock/logs/night_*.err 2>/dev/null   # empty = pass

for ARM in cur_k2 cur_k4 cur_k8 cur_k16 leg_k4; do
  echo "=== $ARM ==="
  RESULTS_DIR=$SCRATCH/air_cate_night_$ARM PATTERN="night_${ARM}_*.json" \
  MERGED_NAME=night_${ARM}_merged.json FIGURES_DIR=examples/figures/night_$ARM \
    bash sherlock/collect_results.sh
done
```

**How to read it.** The number to look at is cross-fitted AIR at n=20000 against RSF-T's 0.1239.
Beating 0.1239 is the bar; `cur_k4` is the 0.1559 we already have. **Do not treat the archive's
0.0854 as the target** — an earlier version of this line did, and it is not safe to: the
archive's IST-3 fits died of NaN at epoch 60–120, which acted as accidental early stopping, so
0.0854 may be an artifact of a broken run rather than a level to reproduce. If a larger K moves
AIR toward 0.1239, capacity was the constraint and the fix is to scale K with n. If all four
capacities land near 0.156, capacity is ruled out too, and the next place to look is the
pseudo-outcome noise rather than the model.

**Result: all four capacities land flat** — 0.1714 / 0.1660 / 0.1649 / 0.1642 for K =
2 / 4 / 8 / 16. So capacity is ruled out, and 8× the components buys 4%. Following the noise
instead is what found the actual cause; see §4d.

---

## 4d. The censoring nuisance was the gap — and the ~15 minute run that confirms it

### What the capacity sweep settled, and what it did not

**Capacity is ruled out.** Cross-fitted AIR at n=20000 is flat in K: 0.1714 (K=2), 0.1660 (K=4),
0.1649 (K=8), 0.1642 (K=16). A 8× capacity increase buys 4%. So all three CATE-stage suspects
listed in `submit_confirm.sh`'s header — restarts, initializer, capacity — are now measured and
none of them is the gap. Section 4c's "the remaining explanation is instability in the CATE fit"
was the wrong inference; the nuisance path had simply never been measured component by component.

### Attribution: it is the censoring survival, and almost nothing else

`examples/diagnose_ist3_nuisance.py` fits one set of nuisances per seed, then trains the CATE
model repeatedly while splicing oracle components in one at a time. Same data, same CATE
initialization, so every difference is the spliced component. n=20000, seed 0:

| spliced in | iMSE | `r_loss_at_zero` |
|---|---|---|
| nothing (all estimated) | 0.2418 | 22.05 |
| oracle propensity | 0.2577 | 22.05 |
| oracle event survival | 0.1985 | 14.55 |
| **oracle censoring survival** | **0.0966** | **1.007** |
| oracle event + censoring | 0.0906 | 0.917 |
| everything | 0.0949 | 0.917 |

The censoring survival accounts for essentially the whole gap. The propensity is *exonerated by
measurement* — rmse 0.046 against truth, nothing clipped, and splicing the oracle in makes things
very slightly **worse**. The event survival is a distant second and worth ~0.006 once censoring
is fixed. `r_loss_at_zero` is the R-loss paid by tau == 0, i.e. the signal-free part of the
objective, and it is the diagnostic that localizes this: 22.05 -> 1.007 on the censoring splice
alone.

### The mechanism: bias in `S_cens` is cheap, variance is not

The AIPCW pseudo-outcome divides by `S_cens` in **all three** of its terms, so it is a delicate
cancellation. A smooth systematic error in `S_cens` largely cancels; an erratic per-subject error
breaks the cancellation and lands undamped in the tail of `Y_tilde`, where the quadratic R-loss
is most sensitive to it. This is why RMSE and bias against the true `S_cens` are the *wrong*
metrics to tune — the left tail is what matters.

The fix is therefore not a better censoring model but a **less flexible** one. `--censoring-fix`
drops the censoring Weibull's hidden layers, so shape and scale are each a softplus of a single
linear index in x: **32 weights on IST-3's p=15, against 3170** for the `(64, 32)` MLP. Event
models, propensity and CATE model untouched. n=20000, 4 seeds, paired:

| | baseline | restricted | oracle `S_cens` |
|---|---|---|---|
| `S_cens` bias | −0.033 | **−0.129** (4× worse) | 0 |
| `S_cens` rmse | 0.168 | **0.139** (better) | 0 |
| frac at clip floor | 0.014 | **0.003** | 0 |
| max abs `Y_tilde` | 136–190 | **5.6–7.7** | 3.1–4.3 |
| frac abs `Y_tilde` > 10 | 0.17% | **0** | 0 |
| `r_loss_at_zero` | 17–26 | **1.52–1.57** | 1.00 |
| **iMSE** | **0.1830** | **0.1094** | **0.0980** |

Read the first two rows together — that is the whole mechanism. The restricted model is four
times more biased and still has lower RMSE, because it trades a lot of variance for that bias.

### Two respecifications that do NOT pay, recorded so they are not retried

Both were argued from theory, implemented, measured at zero, and left available but **off**:

| arm | admin fix | shared shape | linear λ | iMSE |
|---|---|---|---|---|
| `lin_Sce` | yes | yes | yes | 0.1133 |
| `linonly_Sce` | — | yes | yes | 0.1128 |
| `linfree_Sce` | yes | — | yes | 0.1102 |
| **`linplain_Sce`** (what the flag does) | — | — | yes | **0.1094** |
| `shfix_Sce` | yes | yes | — | **0.3298** |

- **`admin_as_censored`.** `1 - Delta` treats end-of-follow-up as an *observed* censoring event
  (14.7% of rows, 27% of all censoring events). A Weibull has no atom to offer, and term2 divides
  by `S_cens(Z)` with `Z = t_end` exactly. Both arguments are sound. Worth 0.0008.
- **`shared_shape`.** A per-subject `k(x)` is heteroscedastic Weibull regression, not AFT. Worth
  −0.003, i.e. very slightly harmful.
- **Together without the flexibility restriction (`shfix_Sce`): 0.3298, far worse than the 0.1830
  baseline** — while having the *best* `S_cens` bias of any arm (−0.011). Same lesson again.

The flag is therefore the **minimal one-knob change**, which also happens to score best.

### The honest caveat, which the run below is partly designed to expose

This is a **regularization choice, not a correctness fix**. A single-index censoring scale happens
to be nearly the right *shape* here — the DGP's true scale is
`lambda_c(x) = lambda_c_base * exp(-b * z_prog)` with `z_prog` a linear index. It is still
misspecified (`_get_params` uses softplus, not exp, which is what the −0.129 bias is), so the gain
is variance reduction, not accuracy.

Two consequences to keep in view:

1. **It may hurt at small n.** A single local seed at n=500 went 0.247 -> 0.645. At n=500 each
   censoring model sees ~167 rows, so this is not the overfitting regime the fix targets. One
   seed proves nothing, which is exactly why the run below does the **full ladder with a paired
   control** rather than n=20000 alone. A method that wins at n=20000 only via a tuning choice
   that is harmful at n=500 is a weaker result, and we should know it before writing it up.
2. **"We beat RSF after hand-picking the censoring model's capacity" is weaker than "we beat
   RSF."** The open question is whether that capacity can be selected data-adaptively. Encouraging
   sign: the restricted model also wins on RMSE, so a cross-validated censoring likelihood might
   pick it without oracle knowledge. Untested.

### The run

Full ladder, 10 seeds, three sbatch calls, ~15 minutes wall clock. Arm A is the new spec with
both forest baselines; **arm B is the paired control** — the same seeds and the same HEAD with
the flag off. The control is not optional: `9d336d6` changed the CATE fit after section 4b's gate
ran, so 4b's 0.1559 is no longer a valid comparator for anything.

```bash
mkdir -p sherlock/logs

# Arm A, 16G: AIR (tasks 0-9) + AIR-oracle (10-19).
CENSORING_FIX=1 RUN_TAG=cfix METHODS_LIST="AIR,AIR-oracle,RSF-T,RSF-S" \
  sbatch --array=0-19 \
    --output=sherlock/logs/cfix_%a.out --error=sherlock/logs/cfix_%a.err \
    sherlock/submit_confirm.sh

# Arm A, 32G: RSF-T (20-29) + RSF-S (30-39). RSF OOM'd at 8G at n=20000.
CENSORING_FIX=1 RUN_TAG=cfix METHODS_LIST="AIR,AIR-oracle,RSF-T,RSF-S" \
  sbatch --array=20-39 --mem-per-cpu=32G \
    --output=sherlock/logs/cfix_%a.out --error=sherlock/logs/cfix_%a.err \
    sherlock/submit_confirm.sh

# Arm B, the control: flag OFF, AIR only, same 10 seeds.
RUN_TAG=cbase METHODS_LIST="AIR" \
  sbatch --array=0-9 \
    --output=sherlock/logs/cbase_%a.out --error=sherlock/logs/cbase_%a.err \
    sherlock/submit_confirm.sh
```

`METHODS_LIST` must be **identical in both arm-A calls** — the script derives the method from
`TASK_ID / N_SEEDS_TOTAL`, so a shorter list in the second call would silently reassign tasks
20-39 to the wrong methods (or exit 1 if the index overruns, which is the good case). The tags
are `cfix` and `cbase` rather than `cfix`/`cfix_off` on purpose: a `cfix_*.json` glob would also
match `cfix_off_*.json`, and the resume logic adopting the other arm's cells as its own answer is
the most expensive mistake available here.

`AIR-oracle`, `RSF-T` and `RSF-S` never call `cross_fit_nuisances`, so they are bit-identical
under both settings and are run once, in arm A only.

Collect the two arms separately:

```bash
RESULTS_DIR=$SCRATCH/air_cate_cfix PATTERN='cfix_*.json' \
MERGED_NAME=cfix_merged.json FIGURES_DIR=examples/figures/cfix \
  bash sherlock/collect_results.sh

RESULTS_DIR=$SCRATCH/air_cate_cbase PATTERN='cbase_*.json' \
MERGED_NAME=cbase_merged.json FIGURES_DIR=examples/figures/cbase \
  bash sherlock/collect_results.sh
```

### How to read it

Local 4-seed expectations at n=20000, to compare against: AIR-with-fix **0.109**, AIR-control
**0.183**, oracle-censoring ceiling **0.098**, RSF-T **0.1239** (settled — reproduced to six
digits in two independent runs).

| check | passes if | fails if |
|---|---|---|
| AIR vs RSF-T at n=20000 | AIR below 0.1239, paired per seed | at or above |
| AIR vs RSF-S at n=20000 | AIR below RSF-S | at or above |
| fix vs control at n=20000 | `cfix` below `cbase`, paired | no separation |
| AIR monotone in n | falls across the whole ladder | rises at small n (see caveat 1) |
| AIR / AIR-oracle at n=20000 | tightens toward ~1.1 | stays near 1.8 |

**Pair per seed and report the paired difference, not two means.** The margin at n=20000 is thin
— locally 0.1094 against 0.1239, about 0.011, with 3 of 4 seeds below and seed 1 at 0.1263 just
above. Ten unpaired seeds may well not separate that; ten paired ones should, since AIR and RSF-T
see identical draws seed by seed. If the paired test does not clear it, the honest conclusion is
"comparable to RSF-T at n=20000", not a win, and the next lever is the event survival
(`orc_Sev_Sce` 0.0906 vs `orc_Sce` 0.0966) rather than more seeds.

---

## 5. The full 999-seed run

**The gate passed.** Section 4d's confirmation run put cross-fitted AIR at iMSE 0.0990 against
RSF-T's 0.1239 and RSF-S's 0.1214 at n=20000 on IST-3, lower on 10/10 paired seeds
(p<0.0001), so the full run goes out **with `--censoring-fix` on**.

3,600 tasks, budget ≲2,300 CPU-hours.

### Send the code first, then verify it landed

The run depends on two things that did not exist in the archived run: the
`--censoring-fix` flag and `submit_array.sh`'s `CENSORING_FIX`/`RUN_TAG` handling. Do the
section 1 rsync, then — on the cluster — confirm the copy is the new one. A stale
`submit_array.sh` fails *silently* by ignoring `CENSORING_FIX=1`; a stale `src/` fails loudly
with `TypeError` on every AIR task.

```bash
cd ~/air_cate_full
grep -c '^CENSORING_FIX=' sherlock/submit_array.sh   # want 1
grep -c '^RUN_TAG=' sherlock/submit_array.sh         # want 1

# Definitive, unlike a grep count: does the installed package accept the parameters?
python -c 'import inspect; from air_cate import AIREstimator; p = inspect.signature(AIREstimator.__init__).parameters; print("censoring_config" in p, "admin_as_censored" in p)'
# want: True True
```

### Submit

```bash
cd ~/air_cate_full
mkdir -p sherlock/logs

# The in-task resume logic loads whatever the output file already holds and SKIPS
# those (n, seed) cells, so stale partials make a fresh submission silently
# incomplete. RUN_TAG=full999 is a directory no previous run has written, which is
# the real protection; this is belt-and-braces for a SECOND attempt at full999.
rm -f $SCRATCH/air_cate_full999/full999_*.json

# Six calls, not four. SLURM caps an array at 1000 tasks per submission, and
# splitting on the RSF blocks lets 2400 of the 3600 tasks ask for 16G rather than
# 32G -- the main lever on queue time. RSF-T and RSF-S are blocks 4-5 within each
# DGP's 1200 ids, i.e. the last 400 of each: 800-1199, 2000-2399, 3200-3599.
# Verified: the three ranges below contain zero RSF tasks, the three above contain
# only RSF tasks, and all 3600 ids are covered exactly once.
for r in 0-799 1200-1999 2400-3199; do
    CENSORING_FIX=1 RUN_TAG=full999 sbatch --array=$r --mem-per-cpu=16G \
        sherlock/submit_array.sh
done
for r in 800-1199 2000-2399 3200-3599; do
    CENSORING_FIX=1 RUN_TAG=full999 sbatch --array=$r sherlock/submit_array.sh
done
```

Each task = one (DGP, method) × 5 consecutive seeds × all 6 sample sizes = 30 fits. The index
decoding lives in the script; the mapping is documented in [STATUS.md](STATUS.md#array-mapping).

**Go/no-go, once the first tasks start.** SLURM creates a task's `.out` when it *starts*, not
when it is queued, so wait for the file before grepping it. Every task must echo
`censoring_fix=1, tag=full999`; if any says `censoring_fix=0` the environment prefix did not
take and the run is measuring the old specification.

```bash
until [ -f sherlock/logs/task_0.out ]; do sleep 15; done
grep -h "^Task " sherlock/logs/task_*.out | grep -c 'censoring_fix=1, tag=full999'
grep -h "^Task " sherlock/logs/task_*.out | grep -c 'censoring_fix=0'   # want 0
```

Note the space in `sleep 15` — a typo like `dosleep 15` spins with no delay.

### Wall limit

2h, down from 3h, sized on section 4d's recorded `runtime_seconds`: worst single seed × 5 was
**0.71h** for AIR, 0.25h for AIR-oracle, ≤0.16h for the forests, all on IST-3 (the most
expensive DGP). Tensor-prod and TP-oracle were not in that run. If a task is killed at the
wall, resume makes it cheap: the partial keeps every finished `(n, seed)` cell, so
resubmitting the same task id continues rather than restarting.

```bash
# After everything finishes, catch any task that hit the wall rather than converging.
sacct -j <jobid> --format=JobID,State,Elapsed | grep -c TIMEOUT   # want 0
```

### Collect

Once `squeue -u $USER` is empty and `sacct` shows 3,600 `COMPLETED`:

```bash
# 3,600 files is real work -- grab an interactive compute node instead of merging
# on the shared login node. sh_dev gives you a 30-minute, 8 GB interactive session.
sh_dev -t 00:30:00 -m 8G

# A compute node starts with a bare environment, so re-activate.
source $GROUP_HOME/$USER/miniforge3/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate air_cate
cd ~/air_cate_full

# collect_results.sh's DEFAULTS NO LONGER MATCH submit_array.sh. The script's defaults
# are still the archived run's air_cate_results/partial_*.json layout; this run writes
# air_cate_full999/full999_*.json. Pass all four overrides.
RESULTS_DIR=$SCRATCH/air_cate_full999 PATTERN='full999_*.json' \
MERGED_NAME=full999_merged.json FIGURES_DIR=examples/figures/full999 \
  bash sherlock/collect_results.sh
```

```bash
# Sanity check before pulling: every (DGP, method, n) cell should hold 999 entries.
ls $SCRATCH/air_cate_full999/full999_*.json | wc -l   # target: 3601 (3600 + merged)
```

### Optional: the paired control arm

The fix is confirmed on IST-3 only. If you want the ablation on all three DGPs, run the
control — but note it costs **1200 tasks, not another 3600**: only the two cross-fitted
methods change under the flag (AIR is block 0, Tensor-prod block 1 within each DGP), and the
oracle and RSF arms never call `cross_fit_nuisances`, so they are shared between the arms.

```bash
for r in 0-399 1200-1599 2400-2799; do
    CENSORING_FIX=0 RUN_TAG=full999_cbase sbatch --array=$r --mem-per-cpu=16G \
        sherlock/submit_array.sh
done

RESULTS_DIR=$SCRATCH/air_cate_full999_cbase PATTERN='full999_cbase_*.json' \
MERGED_NAME=full999_cbase_merged.json FIGURES_DIR=examples/figures/full999_cbase \
  bash sherlock/collect_results.sh
```

**What to check in it.** At n=20000 the fix should reproduce IST-3's win. The rung to watch is
**n=500**, where the 10-seed run had the fix *directionally worse* (+0.093, SE 0.072, better on
only 4/10 seeds) — ten seeds could not separate that from zero, 999 will. If it is real, the
honest claim is "wins at large n, harmful at small n", not a clean win.

---

## 6. Bring the results home

Run this **on your Mac**.

```bash
# Note this overwrites the tracked publication results file. Commit or stash the
# current one first if you want to be able to go back.
#
# The remote path must be the LITERAL /scratch/users/grolleau/... -- $SCRATCH is
# undefined on your Mac and would expand to an empty string.
rsync -avz \
  grolleau@login.sherlock.stanford.edu:/scratch/users/grolleau/air_cate_full999/full999_merged.json \
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
| `PD` with `(None)` / `(Priority)` / `(Resources)` | normal queueing | wait — `(None)` means not yet evaluated, `(Priority)` means queued behind others, `(Resources)` means waiting for a free node |
| `PD` for a long time, `TIME 0:00` | usually an oversized `--mem-per-cpu`: 32G against 1 CPU is ~5 cores' worth of memory on every node model | check `squeue --start` and `scontrol show job <id>`; if the estimate is far out, `scancel` and resubmit with `--mem-per-cpu=8G` (safe below n=5000) |
| `PD` with `(QOSMaxJobsPerUserLimit)` | too many jobs submitted at once | wait; they release as others finish. `sh_part` shows the limits |
| `-bash: JOBID: No such file or directory` | a `<PLACEHOLDER>` was pasted literally; the shell read `<` as an input redirect | set `JOBID=<number>` once, then use `$JOBID` |
| `sacct` shows only a `_[2-17]` row and no per-task rows | the array's *pending* range now starts at 2, i.e. tasks 0–1 have started; the accounting DB also lags `squeue` by seconds | progress, not a problem — cross-check with `squeue -u $USER` |
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
| `SyntaxError: Missing parentheses in call to 'print'` | bare `python` on a login node is **Python 2**; the conda env was not activated | redo step 2's activate block. Inside the env, `python -V` must print 3.x |
| A job sees none of the variables you set | `--export=VAR=value` **replaces** the environment instead of adding to it | always `--export=ALL,VAR=value` — `ALL` first keeps the rest. Beware that a value containing commas is parsed as further `VAR=value` pairs, so quote it and keep comma-valued settings (like `N_VALUES=500,1000`) as a single quoted argument |
| Runtime per fit looks 10× too slow | the harness prints **cumulative** elapsed seconds per line, not per-fit | subtract consecutive lines; e.g. `(31s)` after `(15s)` is a 16 s fit |
| `rsync` reports `received 151 bytes` and no files appear | it was run **on Sherlock**, so it pulled the cluster's own files onto the cluster | every `rsync` in this runbook runs **on your Mac**. Check your prompt: a Sherlock shell shows `sh0x-lognn` |
| Locally, a multi-flag variable arrives as one giant argument | zsh (the Mac default) does **not** word-split unquoted expansions the way bash does, so `$ARGS` stays a single word | use an array: `ARGS=(--large --dgp IST-3)` then `cmd "${ARGS[@]}"` |

### Command cheat sheet

```bash
JOBID=38156448                   # set this first; everything below reuses it
sbatch --array=0-17 script.sh    # submit an array job
squeue -u $USER                  # my live jobs
squeue -u $USER -t RUNNING       # only the running ones
squeue -u $USER --start          # estimated start times (N/A = not yet computed)
sacct -j $JOBID --format=JobID,State,Elapsed,MaxRSS,NodeList  # outcomes, incl. finished
sprio -j $JOBID                  # why my priority is what it is
sinfo -p normal -o "%P %a %D %t" # is the partition full?
scancel $JOBID                   # cancel a whole array
scancel ${JOBID}_3               # cancel one task
scontrol show job $JOBID         # everything SLURM knows about a job
sh_part                          # submission limits per partition
sh_dev -t 00:30:00 -m 8G         # interactive compute node
scontrol show config | grep -i maxarray          # the array cap (assumed 1000)
```

---

## Transfer log

`.git/` is excluded from the rsync, so the cluster copy has no commit information. Record each
transfer here.

| Date | Local commit | Branch | Target | Note |
|---|---|---|---|---|
| 2026-04-30 | (unrecorded) | `sherlock-sims` | `~/air_cate` | the archived 100-seed run; exact command was never written down |
| 2026-08-07 | `9c6e628` + uncommitted script edits | `full-sherlock-sims` | `~/air_cate_full` | pilot `38154411` at 32G never left `PD`; cancelled and resubmitted as `38156448` with `--mem-per-cpu=8G`, which placed within minutes. `submit_pilot.sh` now defaults to 8G, so the flag is no longer needed once `sherlock/` is re-synced. |
| 2026-08-10 | (uncommitted) | `full-sherlock-sims` | `~/air_cate_full` | the two gate runs: `confirm` at `n_restarts=1` and `confirm_r5` at 5. Both failed the gate — see section 4b. |
| 2026-08-11 | `ac36c11` | `full-sherlock-sims` | `~/air_cate_full` | the overnight capacity sweep, section 4c. Carries `--mgd-k`, `--mgd-legacy-lambda-init`, the `LEGACY_INIT`/`MGD_K`/`METHODS_LIST` overrides, and the `plot_convergence` legend fix (without which `--plot-only` on any method subset raises `IndexError`). |
