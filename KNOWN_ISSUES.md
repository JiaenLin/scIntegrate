# Known issues

Measured, not suspected. Each entry says what it costs you and what to do about it.

## A command line written for 0.4.x stops at argparse: `--species` is required from 0.5.0

`assess`, `integrate` and `score` require `--species` from 0.5.0 (README, *Versions*). A job script
written for 0.4.x stops before it reads anything, with exit 2:

```
scintegrate assess: error: the following arguments are required: --species
```

**What it costs:** the run, and nothing else. argparse refuses before the run directory is touched,
so nothing is written to `--out`.

**What to do:** add `--species` with the species the cells came from, as Ensembl writes it
(`homo_sapiens`, `mus_musculus`; `human` and `mouse` are read the same way). Nothing else in the
command needs to change, and no number depends on it. The species is checked against the object's
gene identifiers and recorded, and no parameter, method or metric reads it.

## kBET is normally absent, and the total is taken over one fewer metric

`kBET` is the one scIB batch metric that needs R: `rpy2` plus the kBET R package. On a plain Python
install it cannot run, and the report says so rather than leaving the cell blank.

**What it costs:** the batch score is the mean of four metrics instead of five. That is recorded
beside every aggregate as an `n/of` count, and `scintegrate integrate` prints the absence per
method as it goes.

**Why it is not pinned into the lock:** pulling an R toolchain into a Python environment to recover
one metric costs every user of that file more than the metric is worth. Install `rpy2` and kBET
yourself if you want it; nothing else changes.

## A CUDA wheel can be newer than the driver, and scVI falls back silently

`pip install scvi-tools` resolves the default `torch` wheel for your platform, which on
linux/x86_64 carries a recent CUDA. If the host's NVIDIA driver is older than that build requires,
torch imports fine, reports no usable device, and scVI and scANVI train **on CPU** — correctly, and
far more slowly than a walltime sized for a GPU.

**Measured example:** `torch 2.13.0+cu130` against driver `12040` — CUDA initialisation refused,
training fell back to CPU.

**What to do:** run `scintegrate doctor`, which reports what torch can actually see, *inside the
job* and not only on the machine you submitted from. A login node and a compute node need not carry
the same driver. If you need the GPU, install a torch build matching the driver's CUDA version.

## scIB's LISI helper is prebuilt, and often will not start

scIB ships `knn_graph/knn_graph.o` compiled on its build host. Where the running host's glibc or
libstdc++ is older, it fails to start — and scIB invokes it with `subprocess.run(...)` **without
checking the return code**, so the failure is silent and surfaces much later as

```
FileNotFoundError: .../lisi_xxxx/graph_lisi_indices_3.txt
```

which names nothing about the cause. **Both iLISI and cLISI vanish, one from each side of the
scIB ledger**, and the aggregate is quietly taken over two fewer metrics.

`scintegrate doctor` **executes** the binary rather than checking it exists — those are different
claims — and prints the one-line rebuild from the source scIB already ships:

```bash
cd $(python -c 'import scib,pathlib; print(pathlib.Path(scib.__file__).parent/"knn_graph")')
g++ -std=c++11 -O3 knn_graph.cpp -o knn_graph.o
```

Measured example: `GLIBC_2.38 not found` before, usage text after, and `doctor` flipping to
`lisi helper: ok`.

## scIB calls a pandas API removed in 2.0

`scib` 1.1.7 uses the module-level `pd.value_counts(...)`, gone since pandas 2. That is an
`AttributeError` inside `graph_connectivity` and inside kBET's component sizing, so a batch metric
disappears for a reason unrelated to your data.

scIntegrate restores it in a **scoped** shim for the duration of the scoring call and removes it
again afterwards. It is not a reimplementation — `pd.Series(x).value_counts()` is exactly what the
removed function did.

## Scoring costs more than training, and the cost is Leiden

On a 10⁵-cell cohort, training all five methods took ~50 minutes and **scoring took ~3 hours per
method**. scIB clusters **twice** per method — once in `cluster_optimal_resolution` for NMI/ARI,
once inside `isolated_labels_f1` — and each sweep runs one Leiden per resolution.

Two levers, each a stated trade:

| lever | what it costs you |
|---|---|
| igraph Leiden flavour (default on) | nothing — same algorithm, same graph, same resolution, faster implementation. `--no-fast-leiden` restores scIB's literal behaviour |
| `--scib-resolutions N` | a coarser **search** for the best-agreeing resolution, not a different metric. Leave unset to reproduce a published scIB number exactly |

A third, `--lisi-subsample`, is gone in 0.5.0 (below): it saved seconds, and cost every total its
reproducibility.

## iLISI and cLISI are computed on every cell; `--lisi-subsample` is gone (0.5.0)

Before 0.5.0, `--lisi-subsample PCT` had scIB estimate iLISI and cLISI on a random PCT% of the
cells. scIB's compiled LISI helper draws that subsample from `std::random_device`
(`knn_graph.cpp`: `random_device rd; mt19937 gen(rd())`) and takes no seed, so no seed this tool
passes reaches it. Every execution scored a different set of cells, and the two metrics, one on
each side of the scIB total, moved between two runs of the same command.

**Measured between two executions at `--lisi-subsample 50`**, on one cohort of about 10⁵
observations, for the methods that train nothing (2026-11-09): iLISI moved by as much as
**1.9e-3** and cLISI by as much as **6.2e-4**, while every other scIB metric of those methods
agreed to 1e-6. The first of three comparisons had given 1.5e-4 and 2.6e-4, and a draw can move
them further. The value is `benchmark.LISI_SUBSAMPLE_SPREAD`.

**Why every cell, and not a seed.** The helper skips a cell only when a draw from 1 to 100
exceeds the percentage. At 100, which is what scIB's LISI functions use when they are given no
subsample, no draw can skip a cell, so the value does not depend on the draw. A seed is not on
offer: the helper takes exactly five arguments, and none is a seed. A tool could still draw its
own half, but only by running the helper on every cell anyway (it cannot be told which) and
replacing scIB's own call path to pick the half before the Python step. Measured on 10⁵ synthetic
cells, per metric: about 16 s for half the cells and 25 to 27 s for all of them, on one to four
cores. The seeded route would have saved 4 to 6 of those seconds. A run makes two LISI calls per
method per label column, so every cell costs minutes against a run of hours.

**What changed in 0.5.0.** `integrate` and `score` score every cell. `--lisi-subsample` stops at
argparse with exit 2 and says why; drop it, and nothing else in the command changes.
`report.json` records `lisi_subsample: null`, and the report says, under the totals, that the two
were computed on all the cells and are identical in every execution on the same embedding.
`tests/test_lisi_deterministic.py` runs each command twice and compares them to the last bit.

**A record written with a subsample** (`lisi_subsample` 50 in its `report.json`, or a
`report.json` from before that key) carries the movement above, and the report says so under the
totals. Each metric enters a total through a mean, so a method's total moves by at most
`w_bio × ΔcLISI / n_bio + (1 − w_bio) × ΔiLISI / n_batch`, where `n_bio` and `n_batch` are the
metrics its two means were taken over (the `n/of` counts printed beside every total). For six and
four at `--w-bio 0.6` that is about 2.5e-4. This is an inference from the weighting, not a
measurement. A gap between two such totals smaller than that is no gap.

## scANVI is semi-supervised and ranks against unsupervised methods on equal terms

scANVI trains on the cell-type labels that the biological-conservation metrics then score. That
tends to flatter it relative to scVI, Harmony and BBKNN, which never see the labels.

The ranking does not adjust for this — every method is scored identically on the scIB total, which
is how scIB benchmarks are normally published. The `kind` and method notes state that scANVI reads
the annotation, so the fact is on the page; deciding what to make of it is yours.

## `graph` methods are not scored on the same operation as `embed` methods

BBKNN corrects the neighbour graph and leaves the coordinates alone, so there is no corrected space
to return. It is scored with scIB's `type_="knn"`, on connectivities, and its kNN-retention figure
is not measuring the same transformation as Harmony's or scVI's.

Both appear in one table, because the alternative is two tables nobody compares. The `kind` column
is how they are kept distinct. Read it.

## The default embedding is chosen on cell-type structure, not on your question

The scIB total weighs cell-type conservation against batch mixing. It knows nothing about the
factor your study is actually about.

Where a biological factor varies only *between* batches — each library carrying one value of it —
a correction on the batch key necessarily reduces that contrast too, and where each value is a single
library (the factor is aliased with the batch key) it can remove it entirely. The report computes
this from your design and prints it as **Constraint on use**: the chosen embedding is for
visualisation, clustering and cell-type identification, and not for a composition claim across the
nested factor. Differential expression is unaffected, because it reads `layers['counts']`.

**This is not a bug and there is no setting that fixes it.** It is a property of the design, and
the only reason it is listed here is that a named default invites being used for everything.

## Nothing is bit-reproducible across versions

Neither UMAP nor Leiden is bit-identical across releases of their own packages, and a variational
model's latent space is not reproducible across a change of `torch`. The seed fixes what it can.

Use `setup/environment.lock.yml` when you intend to compare against a published number.
`pyproject.toml` declares what scIntegrate needs to *import*; the lock is what a result needs to
*reproduce*.

## `assess` will not measure a small cell type, and that is deliberate

Below `--min-per-k × k` cells, a `k`-neighbour ratio approaches 1.0 from smallness rather than from
mixing, so a rare population would be reported as perfectly mixed *because it is rare*. Those rows
carry their `n` and no ratio.

Lower `--k` if you need them measured, and understand that you are then measuring a different
neighbourhood size from the other rows.

## The report is one page and grows with the method count

Every method adds a column to each panel figure, all at one shared scale. Five methods produce wide
figures. They scroll rather than shrink, because rescaling per panel is the single easiest way to
mislead with this figure — a dispersed method drawn to fit looks compact.

## Harmony's embedding varied between executions — CLOSED in 0.5.0 (GAP-C54)

**What it was.** Up to 0.4.x, two runs of one command over a bit-identical input did not give
harmony the same `X_harmony`, and harmony's clustering metrics moved with it. On one cohort of
about 10⁵ observations, three runs at commits that did not touch harmony moved its ARI by more
than a tenth, and its NMI and isolated-label F1 with it, while its three silhouettes agreed to
1e-6 and `none` and `bbknn` agreed on every metric but LISI. Its rank among the methods moved too.

**The cause, measured.** sklearn's KMeans, which harmonypy runs to place its first centroids, adds
each OpenMP thread's partial sums under a lock, in the order the threads arrive
(`sklearn/cluster/_k_means_lloyd.pyx`). With more than one thread the centroids differ in their
last bits from one execution to the next, and harmony's rounds of soft clustering and correction
carry the difference into the embedding. NMI, ARI and the isolated-label F1 are computed from a kNN
graph and a Leiden sweep on that embedding, and those discrete steps can turn a last-bit difference
into a different graph and partition; how far depends on the data. Each candidate was tested on its
own, on 40,000 synthetic cells:

| candidate | measured |
|---|---|
| the fit, pools free (OpenMP at 12 threads) | three fits differ, max \|diff\| 1.9e-5 and 1.7e-5 |
| the fit, every pool held at one thread | three fits bit-identical, and identical across processes |
| the neighbour graph, built twice from one embedding | identical; also with numba's pool at 1 or 12 |
| the Leiden sweep, NMI, ARI and isolated-label F1 on one graph | identical twice, every partition |
| cells withheld with `--drop-labels` | the same NaN rows in every run: dropped by label before any method |
| two threaded fits' graphs and sweeps (how far the difference travels) | 4 of 640,000 kNN entries and iLISI (1.2e-5) differ; no Leiden partition, so ARI and NMI agree here. On the cohort they did not |

The 2026-09-06 measurements on a cluster node agree: two harmonypy calls in one process differed
at 24 threads (1.6e-12) and did not at one thread (0.0e+00). They also put a 0.214 difference on
two nodes reporting the same CPU model. That was one threaded fit per node, so it cannot separate
the node from the threads' arrival order, and it is no longer read as a node effect.

**The fix.** The fit runs inside `threadpoolctl.threadpool_limits(limits=1)`, opened after
harmonypy is imported (threadpoolctl limits only the libraries loaded when its context opens).
Every pool is held, OpenMP and the BLAS, because the cluster measurement held them together. The
run records the fit's thread count, read inside the limit (`fit_threads` in harmony's settings)
and in the method's note. `tests/test_harmony_reproducible.py` holds the cause, the controls and
two runs of the real CLI.

**What remains, named.**
- **Another machine.** One thread gives the same bits on one machine. It is not established that a
  different CPU, or a different build of the numerical libraries, gives the same bits. If it does
  not, the clustering metrics can move again for the reason above. This has not been measured.
- **Earlier numbers.** A harmony number made before 0.5.0 is one draw of the threaded fit. A run
  at 0.5.0 is a fixed draw, which need not be close to it. Compare a re-run with a re-run.
- **The cost.** One thread for the fit and no other step. Every other method, and all scoring,
  keeps the threads it had. Measured on the workstation, 40,000 synthetic cells, free and held
  fits interleaved: the same CPU time (17.0 s against 16.9 s, median of two) and 21 s against
  25 s of wall time in the quieter round. There the BLAS could not be held (Accelerate), so a
  node whose BLAS is threaded may lose more; that has not been measured.
