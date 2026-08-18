# Known issues

Measured, not suspected. Each entry says what it costs you and what to do about it.

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

Three levers, each a stated trade:

| lever | what it costs you |
|---|---|
| igraph Leiden flavour (default on) | nothing — same algorithm, same graph, same resolution, faster implementation. `--no-fast-leiden` restores scIB's literal behaviour |
| `--scib-resolutions N` | a coarser **search** for the best-agreeing resolution, not a different metric. Leave unset to reproduce a published scIB number exactly |
| `--lisi-subsample PCT` | a noisier LISI estimate, not a biased one |

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

Where a biological factor varies only *between* batches — one library per animal, so one age per
library — a correction on the batch key necessarily reduces that contrast too. The report computes
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
