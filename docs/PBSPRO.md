# Running scIntegrate under PBS Pro

A working recipe. Nothing here is specific to one site: every number you must supply is named, and
the way to read it off your own scheduler is given beside it.

## The shape of the job

```bash
#!/bin/bash
#PBS -N scint
#PBS -q <queue>
#PBS -l select=1:ncpus=16:mem=120gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail
cd "$PBS_O_WORKDIR"

TOOL=/path/to/scIntegrate                 # the tool, installed ONCE, outside any project
PY=/path/to/project/env/bin/python        # the PROJECT'S interpreter
RUNDIR="${RUNDIR:?pass with qsub -v RUNDIR=...}"   # everything written goes inside the project

mkdir -p "$RUNDIR"/{logs,cache}

# Thread counts derived from the allocation, so the #PBS line is the only place the core count is
# written down. Getting these from $NCPUS rather than hardcoding them is what stops a job that was
# resized in one place and not the other from oversubscribing its node.
export OMP_NUM_THREADS="${NCPUS:-1}" MKL_NUM_THREADS="${NCPUS:-1}" \
       OPENBLAS_NUM_THREADS="${NCPUS:-1}"

# Keep every library's scratch inside the run directory. Left at their defaults these land in
# $HOME, where they are invisible, shared between concurrent jobs, and outlive the run.
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME="$RUNDIR/cache" MPLCONFIGDIR="$RUNDIR/cache/mpl" \
       NUMBA_CACHE_DIR="$RUNDIR/cache/numba" TORCH_HOME="$RUNDIR/cache/torch"

# Fail early and visibly if the interpreter is not the one you think, rather than discovering it
# after the models have trained.
"$PY" -c 'import sys; print("python", sys.version.split()[0])'
PYTHONPATH="$TOOL" "$PY" -m scintegrate.cli doctor

PYTHONPATH="$TOOL" "$PY" -m scintegrate.cli integrate \
    --h5ad  "$IN" \
    --out   "$RUNDIR" \
    --batch-key sample --label-key cell_type --l1-key cell_compartment \
    --design "$DESIGN" --bio-factor age --bio-factor diet \
    --methods none,harmony,bbknn,scvi,scanvi \
    --n-cores "${NCPUS:-1}" --seed 0 \
  2>&1 | tee "$RUNDIR/logs/run.log"
```

## Run `doctor` inside the job, before the work

The interpreter a scheduler gives you is not always the one you tested. `doctor` is stdlib-only and
returns in under a second, and it prints which capabilities are present and what each absence costs.
Two failures it catches immediately, both of which otherwise surface hours in:

- **`scib` absent** — the run completes and no default embedding is chosen, because the choice is
  defined as the scIB total. You get a report with no recommendation and no error.
- **`torch` present but no CUDA device** — scVI and scANVI run on CPU. Nothing fails; it just takes
  far longer than the walltime you sized for a GPU.

## Sizing

Estimate, add 25%, then pick the queue that fits — never the reverse. Under-requesting costs the
whole run and you find out at the end.

**Memory** is set by the object, not by the methods. Budget roughly:

| what | how much |
|---|---|
| the input object | `X` + both count layers, resident |
| one copy during the write | the deliverable is assembled before it is written |
| the embeddings | cells × (n_latent + 2) × 4 bytes per method — small, tens of MB |
| scIB clustering | a neighbour graph over the cells carrying a real label |

A cohort of ~10⁵ cells × ~3·10⁴ genes with two count layers sits in the low tens of GB. Read your
input's size off disk and start from three times it.

**Walltime is set by the METRICS, not the models** — which is the opposite of what you would guess,
and was measured rather than estimated.

On a ~10⁵-cell cohort: training all five methods took **~50 minutes**; scoring took **~3 hours per
method**. scIB clusters *twice* per method — once in `cluster_optimal_resolution` for NMI/ARI, once
inside `isolated_labels_f1` — and each sweep runs one Leiden per resolution.

Three levers, each a stated trade rather than a free lunch:

```bash
--scib-resolutions 7      # a coarser SEARCH for the best-agreeing resolution, not a new metric
--lisi-subsample 50       # a noisier LISI estimate, not a biased one
                          # (the igraph Leiden flavour is forced by default: same algorithm,
                          #  same graph, same resolution, faster implementation)
```

Omit all three to reproduce a published scIB benchmark exactly.

**A walltime kill no longer costs you the models.** The object is written as soon as the embeddings
exist, before scoring starts, and rewritten once the scores are there. If the job dies during
scoring, run `scintegrate score --h5ad <that object>` and finish the benchmark in minutes.

Find your queue's ceilings rather than guessing:

```bash
qstat -Qf | grep -E '^Queue|resources_max|max_queued'
```

## Asking for a GPU

Request **exactly** the number you will use:

```
#PBS -l select=1:ncpus=16:mem=120gb:ngpus=1
```

On some configurations a queue advertises no per-job GPU ceiling, and an over-request is **accepted
by `qsub` and then never scheduled** — the job sits queued indefinitely with no error to read. Check
what is actually available:

```bash
pbsnodes -a | grep -B5 'resources_available.ngpus'
```

scvi-tools is told `accelerator="auto"`, so it uses the GPU when it can see one and CPU when it
cannot. That is deliberate: a job that silently falls back is better than one that dies, but it is
only safe because `doctor` reports which happened.

If your GPU queue caps concurrently queued jobs at one, run the methods as **one job that loops**,
not as an array.

## Logs must land on shared storage

`#PBS -o` names a path on the *submitting* host. If the submitter is itself a job, the copy fails
with `Exit_status = 0` and no file — a job that looks successful and left nothing to read. Have the
job write its own log to shared storage, as the `tee` above does, and point `-o` inside the run
directory:

```bash
qsub -o "$RUNDIR/logs/" -v RUNDIR="$RUNDIR",IN="$IN",DESIGN="$DESIGN" job.pbs
```

Never `/tmp`. It is node-local: a report written on one node is invisible from every other.

## Two things that will waste a run

**Line endings.** A `.pbs` transferred from a Windows machine arrives with CRLF and dies at the
shebang — *after* PBS has granted the allocation. The job vanishes having produced nothing, and in a
driver log that is indistinguishable from a job that ran and wrote no output. Anything that *reads*
a transferred file is exposed too, not only what executes it: a trailing `\r` prints identically and
never compares equal.

```bash
find . -name '*.pbs' -o -name '*.py' | xargs sed -i 's/\r$//'
```

**Job history.** Many sites keep finished-job records for only 24 hours. Record the job id the same
day or the run cannot be reconstructed:

```bash
qstat -xf <jobid> > "$RUNDIR/logs/pbs_record.txt"
```

## Where output goes

`--out` inside the project that asked for the run. Not the tool's directory: a tool that writes into
its own install accumulates other people's runs, and absolute paths from one project end up recorded
inside another's outputs.

The tool writes its own layout — `objects/`, `tables/`, `figures/`, `reports/`, `report.json`,
`README.md`. Read `README.md`: it is generated by inspecting the directory, so it describes what is
actually there rather than what the run intended to produce.
