# How each method is set, and why

Every parameter scIntegrate passes to an integration method, what the package's own default is,
and where the two differ. A benchmark whose settings live only in source code is one nobody can
reproduce or argue with.

The effective settings of a run are also written into `report.json` under `method_settings` and
rendered in the report under **How each method was actually run** — recorded by each method *as it
ran*, not copied from this page. If this document and a run disagree, the run is right.

---

## The comparison is only as good as what all five share

Four things are held constant across every method, deliberately.

| | what | why it is shared |
|---|---|---|
| cells | every cell in the input object | a method compared on a subset is not compared |
| genes | one highly-variable set, reused | see below |
| PCs | `--n-pcs` from one `X_pca` | `none`, Harmony and BBKNN all read it |
| labels | one `--label-key` | the bio metrics are computed against it |

**The gene set is a trade-off, and it is worth stating which way it was taken.** scVI's own
documentation recommends selecting HVGs with `flavor="seurat_v3"` on raw counts, and with
`batch_key` set so genes variable in only one batch do not dominate. scIntegrate does neither by
default: it **reuses the mask already on the object** when there is one, whatever produced it.

That is the deliberate choice. Giving scVI a gene set chosen by scVI's preferred method, while
Harmony and BBKNN work from a PCA built on a different one, compares methods *and* gene sets at
once and cannot separate them. One gene set for everyone costs scVI a little of its recommended
setup and buys a comparison where the only thing varying is the correction.

When the object carries no mask, one is computed here over **all** genes with `seurat_v3` on the
counts layer, with no gene class excluded, and the run says so. No gene class — mitochondrial,
ribosomal, haemoglobin — is ever dropped: HVG selection changes what a model *sees* and every gene
stays in the object and in the deliverable.

---

## Species and assay: declared, and what each one changes

**The species changes no setting, and is checked.** `--species` is required: the run reads the
Ensembl gene identifiers the object carries (the `var` column holding most of them, prefixes matched
longest first, 90% agreement decisive) and refuses a declared species they contradict, naming both,
before anything is computed. Without identifiers, only the case the symbols are written in can be
read — a convention, reported as a note. The rule is the harness's `axes` probe, restated in
`scintegrate/axes.py` so that this package runs without the harness.

**The assay changes nothing here, and that is measured, not assumed.** No parameter on this page,
no threshold (`--indicates-below`, `--min-per-k`, `--k`) and no scIB metric reads it: cells and
nuclei are integrated and scored by the same rules. `axes.ASSAY_RULES` — the declaration of every
rule that differs by assay — is empty. `tests/test_cohort_axes.py` integrates one object as
`scrna`, as `snrna` and undeclared and finds every embedding, metric and table identical, and fails
on any branch taken on the assay outside `axes.py`. So `--assay` is optional: recorded in the run's
provenance and wording when given, recorded as `not declared` when not, and never defaulted. A rule
that comes to depend on it is declared per assay in `ASSAY_RULES`, and `--assay` then becomes
required — the same test holds the two together.

---

## `none` — the baseline, and it is a method

```python
emb = adata.obsm["X_pca"][:, :n_pcs]
```

Runs first, always, and cannot be turned off. The stage's question is whether integration is
*needed*; a comparison without the uncorrected case can only say which correction is strongest,
never whether any was warranted.

**It reuses `X_pca` from the input object rather than recomputing, unless cells are withheld.**
That PCA is an input to the comparison and scIntegrate does not audit how it was made — whether on
lognormalised values, whether scaled, over which genes. Three of the five methods read it, so if
it is wrong they are all wrong together. If the object carries no `X_pca`, one is computed here
with `sc.pp.pca(n_comps=--n-pcs, svd_solver="arpack", random_state=--seed)` on `X`, over the
object's `highly_variable` mask when it carries one, and the run says so.

When `--drop-labels` withholds cells and any of them has a row in the input's `X_pca`, they were
in that PCA's fit, so it is **refit on the retained cells** by the same call (below, *Withholding
cells from the integration*). The record of which PCA the three methods read, and how it was
made, is `report.json` `pca` (also in the object's provenance and on the page): `source` is
`input`, `refit` or `computed`, with the cells fitted, the withheld cells not fitted and in the
fit, the components, the genes, the call and its thread count. A PCA made here is fitted on one
thread, as Harmony is, because an arpack PCA's last bits depend on the BLAS's thread count and
Harmony and every metric are fitted on those bits (KNOWN_ISSUES).

If `X_pca` has fewer components than `--n-pcs`, slicing returns what exists and raises nothing.
The run now prints the shortfall and every method uses the same reduced number.

---

## Harmony

```python
meta = obs[[batch_key]]                               # unused categories removed, order kept
with threadpoolctl.threadpool_limits(limits=1):      # opened after harmonypy is imported
    harmonypy.run_harmony(X_pca[:, :n_pcs], meta, [batch_key],
                          max_iter_harmony=20, random_state=seed)
```

**A category no cell carries is removed first.** harmonypy counts the levels it observes for
theta and builds its design matrix from every category, so an unused one (what an object filtered
upstream keeps of what it lost) failed the fit with `operands could not be broadcast together`
before 0.6.0 (GAP-C65). Removing it keeps the order of the rest: a column without one gives the
same fit, to the last bit (`tests/test_method_failure_status.py`).

| parameter | here | harmonypy default | why |
|---|---|---|---|
| `max_iter_harmony` | **20** | 10 | **deviation.** A cohort with many libraries can still be moving at 10; stopping there scores a correction that had not converged |
| `random_state` | `--seed` | 0 | **fixed defect.** It was not passed, so `--seed` changed scVI, scANVI and every UMAP while Harmony stayed on a fixed internal seed. Two runs at different seeds gave one identical embedding and four different ones, with nothing saying so |
| threads for the fit | **1** (every pool threadpoolctl finds: OpenMP and the BLAS) | the process's | **fixed defect (GAP-C54).** On more than one thread the fit did not return the same bits twice, and scIB's clustering metrics turned the difference into a different ranking. Below |
| `theta` | default (1) | 1 | diversity penalty; raising it corrects harder. **The record said 2.0**, which is R harmony's default; harmonypy 0.0.10 fits with 1 when none is passed (measured: unset and `theta=1` give the same bits). The run now records the value read from the fitted model |
| `sigma` | default (0.1) | 0.1 | soft-cluster width; recorded as read from the fitted model |
| `nclust` | default (auto) | auto | |

**Why one thread.** sklearn's KMeans, which harmonypy runs to place its first centroids, adds each
OpenMP thread's partial sums under a lock, in the order the threads arrive
(`sklearn/cluster/_k_means_lloyd.pyx`). With more than one thread the centroids differ in their
last bits between executions, and harmony's rounds carry that into `X_harmony`. NMI, ARI and the
isolated-label F1 are computed from a kNN graph and a Leiden sweep on the embedding, and those
discrete steps can turn a last-bit difference into a different graph and partition. How far depends
on the data: on one cohort harmony's ARI moved by more than a tenth between runs; on 40,000
synthetic cells two threaded fits changed 4 of 640,000 kNN entries and iLISI by 1.2e-5, and no partition. On one
embedding's bits the steps are exact: the graph is identical built twice, and so are the sweep and
its metrics. So the
variation is removed at the fit and nowhere downstream (`tests/test_harmony_reproducible.py`).
Measured on 40,000 synthetic cells: three fits with the pools free differ by up to 1.9e-5, and held
at one thread they are bit-identical within a process and across processes. Every pool is held,
not only OpenMP, because the one measurement on a cluster node held OpenMP and the BLAS together.
What one thread does **not** establish: identical bits on a different CPU or a different build of
the numerical libraries. That has not been measured, and on a different machine the clustering
metrics can move again for the reason above.

Harmony corrects **in PC space**, so its output has `--n-pcs` dimensions and is directly
comparable with `none`.

---

## BBKNN

```python
bbknn.bbknn(B, batch_key=batch_key, n_pcs=n_pcs, neighbors_within_batch=3)
```

| parameter | here | bbknn default | why |
|---|---|---|---|
| `neighbors_within_batch` | 3 | 3 | left at default and **recorded**. The graph holds `n_batches x this` edges per cell, so ten libraries already give 30 |
| `n_pcs` | `--n-pcs` | 50 | shared with every other method |
| `trim`, `metric` | defaults | | |

**BBKNN is scored differently from the others, and this is the important part.** It corrects the
**neighbour graph** and leaves coordinates alone — there is no corrected space to hand back. It
therefore declares `kind: graph` and is scored on its connectivities via scIB's `type_="knn"`,
while the rest are scored on coordinates with `type_="embed"`.

The metrics that need a coordinate space — `asw_label`, `asw_batch`, `isolated_labels_asw`, `pcr`
— are **absent** for BBKNN, not zero and not borrowed. An earlier version returned BBKNN's UMAP as
if it were an embedding, which meant those metrics described a two-dimensional nonlinear layout
rather than the space the correction happened in.

A run also passes BBKNN a **light** AnnData holding only the PCA and the batch column, not a copy
of the input: copying a cohort-sized object to read one obs column is how a comparison dies at the
last method rather than the first.

---

## scVI

```python
scvi.settings.seed = seed
SCVI.setup_anndata(sub, layer=counts_layer, batch_key=batch_key)
SCVI(sub, n_latent=n_latent).train(max_epochs=None, accelerator="auto")
```

| parameter | here | scvi-tools default | why |
|---|---|---|---|
| `layer` | `--counts-layer` | — | **raw counts, checked**. The layer is tested for integrality before any count model runs, and the run refuses if it is not. A count model handed normalised data trains happily and returns a plausible embedding |
| `batch_key` | `--batch-key` | — | the only covariate. No continuous covariates are passed |
| `n_latent` | `--n-latent` (30) | 10 | **deviation.** 30 to sit nearer the PC counts the other methods work in |
| `max_epochs` | unset | auto from `n_obs` | scvi-tools' own heuristic |
| `n_layers`, `dropout_rate`, `dispersion`, `gene_likelihood` | defaults | | |

scVI is **unsupervised**: it is never shown the cell-type labels. Its biological-conservation
metrics are therefore measuring generalisation, in a way scANVI's are not.

---

## scANVI

```python
s = SCANVI.from_scvi_model(m, labels_key="_scanvi_labels", unlabeled_category=UNL, adata=sub)
s.train(max_epochs=scanvi_max_epochs, n_samples_per_label=100, accelerator="auto")
```

Initialised from the **already-trained scVI model** above, not from scratch — the supported path,
and it means scANVI's result depends on scVI's.

| parameter | here | note |
|---|---|---|
| `n_samples_per_label` | 100 | matches scvi-tools' scANVI tutorial |
| `max_epochs` | `--scanvi-max-epochs`, unset by default | **read this row.** The tutorial uses `20`. Unset, scANVI inherits the same length formula scVI uses, which on a large cohort is several times that — a longer fit against the labels. Exposed as its own flag; the default is unchanged so no existing comparison moves underneath anyone |
| `unlabeled_category` | annotator sentinels | see below |

**Sentinels are passed as genuinely unlabelled.** Every cell whose label is one of
`--label-sentinel` (default `EXCLUDED`, `UNRESOLVED`) is mapped to the unlabelled category. That is
not a workaround: a cell the annotation declined to call *is* unlabelled, and passing it as a class
of its own would ask the model to learn "unresolved" as a cell type.

### The thing to know before reading scANVI's score

> scANVI is **trained on the label column it is later scored against**.

scIB's biological-conservation half is NMI, ARI, ASW-label and the isolated-label scores — every
one computed against the cell-type column. scANVI is given that column. When it wins the bio half,
part of what is being measured is that it was told the answer, and it is ranked against methods
that were not.

This is not a defect in scANVI, which is doing exactly what it is designed to do, and not a reason
to drop it — a supervised method may genuinely be the right choice, especially when the labels are
trusted and the goal is to preserve them. It is a reason the totals are **not a like-for-like
ranking**.

The report states this under the winner whenever a label-supervised method appears in the ranking,
because a reader who does not already know it will read the totals as comparable and nothing else
on the page would say otherwise. The **batch** half is unaffected and can be read directly.

---

## The 2-D views

Computed by scanpy, exactly as a standard workflow would:

```python
A = ad.AnnData(X=emb)                 # the method's own coordinates AS X
sc.pp.neighbors(A, use_rep="X", n_neighbors=15, random_state=seed)
sc.tl.umap(A, min_dist=0.2, random_state=seed)
```

One departure from scanpy's defaults: `min_dist=0.2` rather than 0.5, so these panels match the
embedding an annotation typically ships and the two can be read against each other. `n_neighbors`
is scanpy's own 15, stated rather than inherited — it is the parameter a layout is most sensitive
to after `min_dist`, and both are printed in the report.

A graph method's view comes from **its own corrected graph** (`sc.tl.umap` on the BBKNN
connectivities), which is the only honest 2-D view of a method that produced no coordinates.

**Layout parameters change nothing but the picture.** No metric, count or label depends on
`min_dist` or `n_neighbors`.

### Why the panels are drawn by hand rather than by `sc.pl.umap`

One requirement scanpy's plotting will not meet: **every method must be drawn at the same scale.**
Per-panel autoscaling makes a dispersed method look compact, and it is the single easiest way to
mislead with this figure. The panels also colour by several label columns in turn — one row each —
because label columns disagree exactly where the annotation was least certain, which is where a
correction is most likely to have moved something.

---

## Reproducibility

`--seed` reaches `scvi.settings.seed`, `harmonypy`'s `random_state`, `sc.pp.pca`, `sc.pp.neighbors`
and `sc.tl.umap`. BBKNN is deterministic. Harmony's fit runs on one thread, so on one machine it
gives the same bits every time (above, *Why one thread*).

The neighbour graph scIB scores from is exact on fixed bits only while scanpy's `settings.n_jobs`
is 1, its default, which this tool never changes. scanpy hands that number to pynndescent, which
draws its candidates per thread (`rng_state + n` in `new_build_candidates`), so at any other value
the graph, and every clustering metric on it, depends on the thread count (measured: 40,000
synthetic cells, `n_jobs` 12 against 1: 30,040 of 640,000 kNN entries differ).

What `--seed` does **not** pin: GPU non-determinism in scVI/scANVI training. Two runs on the same
seed and different hardware can differ slightly. Where that matters, compare the ranking rather
than the fourth decimal place.

---

## What none of this decides

The default embedding is the highest scIB total under `--w-bio`, and **that weight is a value
judgement** — it asserts how much residual batch structure is worth trading for retained biology.
It is on the command line for exactly that reason. A different downstream question wants a
different answer, every method's embedding is kept in the delivered object, and the figures exist
because no integration decision should be made on metrics alone.

---

## Withholding cells from the integration

```bash
--drop-labels EXCLUDED
```

Cells carrying a named label value are withheld from **the fit**: they are not in the PCA, not in
Harmony's clustering, not in BBKNN's graph, and not in scVI or scANVI's training. They remain in
the delivered object, with **`NaN` in every embedding**, `obsm['X_pca']` included.

**The PCA is refit for this.** An `X_pca` made upstream was fitted on every cell the upstream
object held, the withheld ones included; reused as delivered, they would still shape the space
`none`, Harmony and BBKNN are fitted in. So when a withheld cell has a row in the input's `X_pca`,
the PCA is refit on the retained cells (the call in *`none`* above, the input's `highly_variable`
mask reused verbatim) and the input's is not used. Before 0.6.0 it was reused, and the delivered
`X_pca` kept the withheld cells' rows (GAP-128). Two cases keep the input's PCA: it is NaN on every
withheld cell, so none of them can have been in its fit; or the run declares `--keep-input-pca`,
the old behaviour, and `report.json` `pca` then records how many withheld cells are in its fit
and the delivered `X_pca` keeps their rows. What the refit cannot undo: the `highly_variable`
mask itself was chosen upstream, possibly with the withheld cells present, and is reused as
delivered, as every method's gene set is.

**Why it can matter.** Sentinel cells left in the fit shape the space the retained cells sit in,
and they do so unevenly. BBKNN is the clearest case: it forces batch-balanced edges, so a
population present in one library and absent from four gets neighbours across batches that
correspond to nothing. Harmony is next — a population at 9% in one library and 0% in four is
exactly the signal it is built to remove.

**When it is legitimate, and when it is not.** Use it to propagate a removal an earlier stage
already made and approved. That is not a new decision; it is declining to build the manifold out
of cells nobody will analyse. Do **not** use it for a label meaning *the annotator was uncertain* —
those are cells with real data and unknown identity, and integration is plausibly what resolves
them.

**What is recorded.** The per-arm rate is measured *before* anything is withheld and printed
whether or not it is alarming, into `tables/integration_withheld_by_arm.csv`, with a note when the
rate differs more than 3× across arms. The constraint on use carries the count, and the withheld
cells are `NaN` rather than absent so a reader can tell a cell that was withheld from one that was
never delivered — and so nothing can average them in, which a zero would.

## Scoring against more than one label column

```bash
--score-against cell_type_forced,cell_compartment
```

Runs the whole benchmark again against each column and prints the rankings side by side, writing
`tables/scib_by_label_column.csv`.

One ranking is a ranking **under one label set**. Where an annotation ships several — a fine level,
a coarse one, a forced variant in which uncertain calls have been pushed to a leaf — they disagree
exactly where the annotation was least certain. A label-supervised method is trained on one of
them and scored against it.

If a method's advantage moves between label sets, the advantage was partly about the labels. That
turns an argument about the metric into evidence, and the run says so explicitly when the winner
changes.

**A forced label is not a better label.** Forced calls are made at margins the walk did not
support. Scoring against them is a diagnostic, not an upgrade, and training a supervised method on
them would make the circularity above strictly worse — the model would be reproducing a guess and
being scored on how faithfully it did so.
