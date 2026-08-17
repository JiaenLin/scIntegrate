# What scIntegrate writes, and which file is the answer

```
<--out>/
├── objects/cohort_integrated.h5ad   THE DELIVERABLE (integrate only)
├── reports/
│   ├── assessment.html              from `assess` — is integration needed
│   └── integration.html             from `integrate` — START HERE
├── tables/*.csv                     every number the reports show
├── figures/*.png                    every method at one shared scale
├── report.json                      every number again, machine-readable
└── README.md                        written by INSPECTING this directory
```

`README.md` is generated from what is on disk, not from what the run intended, so where the two
disagree the README is right. Read it before the HTML if you have arrived at this directory cold.

## The object

One file. Every method's embedding lives in it, beside the counts they were derived from.

| slot | holds | scale |
|---|---|---|
| `X` | expression | **log1p of library-size-normalised counts**, normalised over all samples together |
| `layers['counts']` | the same cells and genes | **raw integer counts** — what the count models read, and what DE reads |
| `layers['lognorm']` | identical values to `X` | named explicitly, so `X`'s scale is never implicit |
| `obsm['X_pca']` | the **uncorrected** baseline | kept permanently, so `none` stays comparable forever |
| `obsm['X_<method>']` | each corrected space | latent coordinates, one per `embed`-kind method |
| `obsm['X_umap_<method>']` | one 2-D view per method | **figures only** — never an input to a metric or a model |
| `obsm['X_umap']` | a copy of the **chosen** method's view | so a plotting call with no argument uses it |
| `obs` | the batch key, the label column(s), every design column | nothing else |
| `uns['scintegrate']` | the benchmark, the choice and its reason, the constraint on use | plain types only |

**Why the counts are carried at all**, when embeddings are the point: the next stage's differential
expression reads counts per sample, and it must read them from an object whose cell set is exactly
the one the embedding describes. Handing it the embedding and letting it fetch counts from upstream
is how a cell filtered in one place and not the other becomes a silent mismatch.

**`X` and `layers['lognorm']` deliberately hold the same values.** Leaving the identity of `X`
implicit is how an object ends up with nobody able to say whether it holds counts, CPM or log1p.

### Reading the choice out of the object

```python
A.uns["scintegrate"]["default_embedding"]   # 'X_scanvi', or 'NONE CHOSEN'
A.uns["scintegrate"]["default_method"]
A.uns["scintegrate"]["constraint_on_use"]   # computed from your design
A.uns["scintegrate"]["benchmark"]           # every metric, per method
A.uns["scintegrate"]["assessment"]          # the per-cell-type measurement
```

The choice is recorded **in** the object rather than in a filename, so it travels with the data.

### It opens in a browser viewer

Every write holds `classic_string_encoding`, so labels and indices land as HDF5 string **datasets**
rather than the nullable-string groups newer anndata writes by default and most other readers cannot
open. The failure that guards against is invisible in Python — anndata reads both encodings into the
same pandas object — and surfaces in a viewer as a property access on `undefined`, which points
nowhere near the cause.

## The tables

| file | what |
|---|---|
| `celltype_batch_mixing.csv` | per cell type: foreign share, its chance level, the ratio, or why it was not measured |
| `celltype_factor_mixing.csv` | the same per declared biological factor — **read beside the one above** |
| `celltype_batch_dominance.csv` | the largest share any single batch holds in each cell type |
| `label_sentinels_by_arm.csv` | where the annotator's no-call labels fall, per arm of the design |
| `scib_metrics.csv` | every scIB metric, per method |
| `scib_aggregate.csv` | bio / batch / total per method, **with the count of metrics behind each** |
| `scib_absent.csv` | every metric that could not be computed, and why |
| `knn_metrics.csv` | the tool's own measurements, which need no external package |

`scib_absent.csv` is not an error log. It is part of the result: a total from eight metrics is not
comparable with one from nine, and the counts in `scib_aggregate.csv` are how you tell.

## The figures

Every panel shares one set of axis limits. Per-panel autoscaling makes a dispersed method look
compact and is the easiest way to mislead with this figure.

| file | what to read in it |
|---|---|
| `F1_celltype_*.png` | has a method moved cells away from their own kind? |
| `F2_batch_*.png` | is the structure cell type, or library? |
| `F3_<n>_group_<factor>_*.png` | has the correction flattened the factor the study is about? |
| `F4_highlight_*.png` | is one library **aligned with its counterparts**, or dispersed through everything? |
| `F5_celltype_mixing.png` | batch against each factor, per cell type, with the declared threshold |

`F4` is the one no metric substitutes for. A number can say a population was mixed; only this
separates aligned from dispersed.

## Which file must not be used

- **Not `obsm['X_umap_*']` as an input.** Two dimensions of a nonlinear layout, not the space the
  correction happened in. For clustering or a metric, use `X_<method>`.
- **Not a `figures/*.png` as evidence of a number.** The numbers are in `tables/` and `report.json`.
- **Not the default embedding for a composition claim across a nested factor.** The report's
  *Constraint on use* section says which factors those are for your design, and why.

## What is absent, and why that is stated rather than filled

A method whose package is not installed is **named** in the report as not compared. A metric that
could not run is a cell containing the word `absent` and its reason.

An empty cell reads as a zero. A batch score of zero and a batch score that does not exist lead to
opposite decisions, so they must not look alike.
