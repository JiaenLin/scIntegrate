# Quickstart

Five commands, in the order the questions arrive.

## 0. Check the environment before trusting it

```bash
pip install -e '.[run]'
scintegrate doctor
```

`doctor` is stdlib-only, so it runs in the interpreter you are worried about — which is by
definition the one where something is missing. It prints every capability, what each absence costs
you, and what torch can actually see.

```
  ok    core     numpy 2.4.6  sklearn 1.9.0
  ok    read     anndata 0.13.2  scanpy 1.12.3
  ok    figures  matplotlib 3.11.1
  MISS  scib
        missing: scib
        without it: the scIB benchmark is not computed and NO DEFAULT EMBEDDING IS CHOSEN;
                    the tool's own kNN metrics are still reported
        fix: pip install -e '.[scib]'

  gpu: torch present, no CUDA device visible (scVI/scANVI will run on CPU)
```

Install the methods you want compared, one extra each:

```bash
pip install -e '.[harmony]'    # harmonypy
pip install -e '.[bbknn]'      # bbknn
pip install -e '.[scib]'       # the benchmark, and therefore the default choice
pip install -e '.[scvi]'       # scVI AND scANVI - one package. Pulls torch, ~2.5 GB
pip install -e '.[all]'        # everything
```

A method whose package is absent is **named in the report** as not compared. It is not silently
dropped, because a method missing from a comparison changes what the comparison means.

## 1. Write the design table

The factors your study is about are usually not in an annotated object — it carries the annotation,
not the animal metadata. A CSV keyed on the batch:

```csv
sample,age,diet,chemistry,batch
s1,young,chow,v3,A
s2,aged,chow,v3,A
s3,young,HFD,v3.1,B
```

Every column travels into the delivered object. Only the ones you name with `--bio-factor` drive the
assessment and the constraint on use — those two are statements about what the study is *for*, and a
technical covariate is not one.

A batch present in the object with no row here is **refused by name**. A design applied to some
batches and not others is worse than none, because the factor silently becomes "missing" for exactly
the cells it omits.

## 2. Ask whether integration is needed — before spending anything

```bash
scintegrate assess --h5ad joint.h5ad --out results/03_integrate \
    --batch-key sample --label-key cell_type --l1-key cell_compartment \
    --design design.csv --bio-factor condition --bio-factor timepoint
```

Nothing is trained and nothing is corrected. It measures mixing **inside each cell type** on the
uncorrected embedding, and again for each declared factor:

```
  cell type                       n    batch  factor_a  factor_b
  Cell type A                40,000    0.680     0.576     0.620
  Cell type B                16,000    0.734     0.654     0.728
  Cell type C                    47       --   (below 3x k)

  threshold: a ratio below 0.8 is called batch structure
  N of M measured cell types fall below the threshold ...
```

*Illustrative output with placeholder names and counts — the shape of the table, not a result.*

**1.00 is fully mixed** against chance for that population's own batch composition, so every row has
its own denominator. Read the two columns together — where **both** are low, the library structure
and the biology are the same structure, and correcting the batch removes the contrast with it.

Read `reports/assessment.html` and the figures before going on. This step is cheap precisely so the
next one is a decision rather than a default.

## 3. Integrate, benchmark, and get one object

```bash
scintegrate integrate --h5ad joint.h5ad --out results/03_integrate \
    --batch-key sample --label-key cell_type --l1-key cell_compartment \
    --design design.csv --bio-factor condition --bio-factor timepoint \
    --methods none,harmony,bbknn,scvi,scanvi
```

`none` is always included and always first — it is the baseline every cost is measured against, and
without it you can only learn which correction is strongest, never whether any was warranted.

```
  DEFAULT EMBEDDING: X_scanvi   (scIB total 0.7123, margin 0.0141 over scvi)
```

Out comes one object:

```
results/03_integrate/
  objects/cohort_integrated.h5ad    every embedding, plus the counts they came from
  reports/integration.html          START HERE
  tables/*.csv                      every number the report shows
  figures/*.png                     every method at one shared scale
  report.json                       so nothing has to be scraped from HTML
  README.md                         written by inspecting this directory
```

## 4. Use it

```python
import scanpy as sc
A = sc.read_h5ad("results/03_integrate/objects/cohort_integrated.h5ad")

A.uns["scintegrate"]["default_embedding"]      # 'X_scanvi'
A.uns["scintegrate"]["constraint_on_use"]      # read this before any claim

sc.pl.umap(A, color="cell_type")               # obsm['X_umap'] is the chosen view
A.layers["counts"]                             # raw integers, untouched by any correction
A.obsm["X_pca"]                                # the uncorrected baseline, kept permanently
```

Every method's embedding is in the same object under `X_<method>`, so a comparison never means
re-running anything.

## Re-scoring without retraining

The metrics depend on things the models do not: a pandas version, a compiled LISI helper, whether
`rpy2` is installed. When one of those is fixed, you should not have to retrain scVI.

```bash
scintegrate score --h5ad results/03_integrate/objects/cohort_integrated.h5ad \
    --out results/03_integrate --batch-key sample --label-key cell_type
```

It reads the embeddings out of `obsm`, re-scores, and rewrites `tables/scib_*.csv`. Then rebuild
the document with `scintegrate report --out`.

This works because **the object is written before scoring starts**, so even a run killed at the
walltime leaves its embeddings on disk.

## Rebuilding the document without recomputing

```bash
scintegrate report --out results/03_integrate
```

Reads `report.json` and rewrites the HTML. Useful after editing prose or when a figure has been
redrawn.
