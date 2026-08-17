# scIntegrate

**Is integration needed? Which method? And one object carrying the answer.**

Three commands, in the order the questions actually arrive.

```bash
scintegrate doctor                       # what is installed, and what each absence costs

scintegrate assess    --h5ad joint.h5ad --out results/ \
    --batch-key sample --label-key cell_type --design design.csv

scintegrate integrate --h5ad joint.h5ad --out results/ \
    --batch-key sample --label-key cell_type --design design.csv \
    --methods none,harmony,bbknn,scvi,scanvi
```

`assess` answers *is integration needed* without training anything, because that question should be
settled before a GPU is spent on methods you may not want. `integrate` runs the methods, scores
them with **scIB**, names a default embedding, and writes **one** object holding every embedding
next to the counts they came from.

## The unit of measurement is the cell type

A global mixing number cannot answer the question. A cohort looks badly mixed when one abundant
population is split by library and every other population is already interleaved — and correcting
the whole manifold for that is a large intervention justified by one cell type. It looks well mixed
when the abundant populations dominate the average while a rare one sits in ten separate islands.

So `assess` rebuilds the neighbourhood graph **inside each cell type** and asks what share of each
cell's neighbours come from another library, against the share random mixing would give for **that
population's own** library composition. For two equal libraries that is 0.50; for a 90/10 split,
0.18. A raw "72% foreign" means nothing without the number beside it, so the report gives a ratio
where **1.00 is fully mixed**.

Then it does the whole thing again with your biological factor in place of the library. **This is
the comparison that matters.** If a cell type's libraries do not mix, that is batch structure. If
its age groups also do not mix, in the same populations and to a similar degree, then the library
structure and the biology are the same structure — and a correction on the library key removes the
contrast the study exists to measure. Nothing measured after the correction can recover that
distinction, because by then it is gone.

## It names a default embedding, and states separately what that embedding may be used for

The default is the **highest scIB total**: the mean of the biological-conservation metrics weighted
`--w-bio` (0.6, scIB's convention), the mean of the batch-correction metrics taking the rest. It is
written into the object, so the choice travels with the data instead of living in a filename.

What that choice does **not** settle is what the embedding may then carry. A factor that varies only
between libraries — one animal per library, so one age per library — cannot be separated from the
library by any correction on the library key. That is the ordinary shape of a multi-animal design
and not a defect, and it does not stop a default being named. It decides what comes next, and the
report computes it from your design rather than reciting it:

> …may carry visualisation, clustering and cell-type identification. It must **not** carry a
> composition or abundance claim across `age` — for that, use the uncorrected `X_pca` and say so,
> or a test that models the factor rather than removing it. Differential expression on per-sample
> counts is unaffected: it reads `layers['counts']`, which no correction here touches.

Two questions, answered separately, because the answer to one is not the answer to the other.

## An absence is named, never a blank cell

Every scIB metric is computed on its own and returns either a number or the reason it could not.
`kBET` needs `rpy2` and the kBET R package and is usually absent on a plain Python install; it is
reported as absent. **An empty cell reads as a zero**, and a batch score of zero and a batch score
that does not exist lead to opposite decisions.

The count of metrics behind each aggregate travels with it, so a total from eight metrics is never
quietly compared against one from nine — and when two methods' totals rest on different counts, the
report says so instead of ranking them silently.

## Four things it will not do

**It will not treat a sentinel as a cell type.** An annotator that declines to guess emits values
like `EXCLUDED` or `UNRESOLVED`. Those are statements about the annotation, not populations in the
tissue, and NMI, ARI, cLISI and ASW would all score them as though they were. They are found,
counted, **kept in every embedding and in the deliverable**, and excluded from label metrics only —
with a per-arm table, because an exclusion inherited from an upstream filter is rarely even.

**It will not reconstruct a column that ships measured.** A hierarchical label invites
`path.split("/")[0]` as a coarse level. That is a truncation of one walk, not an independent
annotation of the compartment. Pass `--l1-key` and the measured column is used; the truncation is
available behind `--coarse-from-path` and is labelled as one wherever it appears.

**It will not accept normalised values as counts.** scVI and scANVI model counts. Handed log1p data
they train anyway and return a plausible embedding, which is the worst available outcome. The counts
layer is checked for integrality and refused by name.

**It will not let a gene class leave the study.** Restricting a model to highly-variable genes
changes what it *sees* and is reversible — every gene stays in the object. Where the input carries
an HVG mask it is reused verbatim rather than recomputed with a filter of this tool's devising, and
no gene class is ever excluded.

## `none` is a method, and it is never optional

A comparison that omits the un-integrated case can only answer which correction is strongest, never
whether any was warranted. Leave it out of `--methods` and it is put back, and put first — it is the
baseline every retention figure is measured against.

BBKNN is kept honest the same way. It corrects the **graph**, not the coordinates, so there is no
corrected space to return; it declares itself a `graph` method and is scored on connectivities
(scIB's `type_="knn"`). Returning its UMAP as though it were an embedding would make its retention
figure answer a different question from everyone else's, and the `kind` column exists so those two
are never read as one.

## Every panel at the same scale

No integration decision should be made on metrics alone. Each method is drawn coloured by cell type,
by library, by every declared factor, and with one library highlighted against everything else in
grey — all sharing one set of axis limits, because per-panel autoscaling makes a dispersed method
look compact.

A number can say a population was mixed. Only the picture separates *aligned with its counterparts*
from *dispersed uniformly through everything*.

## One object out

```
objects/cohort_integrated.h5ad
    X                     log1p of library-size-normalised counts, over all samples together
    layers['counts']      raw integer counts — what the count models read, and what DE reads
    layers['lognorm']     the same values as X, named so X's scale is never implicit
    obsm['X_pca']         the UNCORRECTED baseline, kept permanently
    obsm['X_harmony']     …one per method compared
    obsm['X_umap_<m>']    one 2-D view per method
    obsm['X_umap']        a copy of the CHOSEN view, so plotting with no argument uses it
    obs                   the batch key, the label column(s), the declared design factors. No more.
    uns['scintegrate']    the benchmark, which method was chosen and why, the constraint on use
```

`obs` is deliberately narrow. An object carrying fifty label columns from an upstream sweep makes a
reader guess which one an embedding was scored against.

**The output opens in scRNA-seq Lab and visualises in scRNA-seq Studio.** Every write holds
`classic_string_encoding`, so labels and indices land as HDF5 string *datasets* rather than the
nullable-string groups that newer anndata writes by default and most other readers cannot open.

A `README.md` is written beside the object **by inspecting the directory**, so it describes what is
actually there rather than what the run intended.

## The design table

The factors a study is about are usually not in an annotated object — it carries the annotation, not
the animal metadata. They arrive as a CSV keyed on the batch:

```csv
sample,age,diet,chemistry
s1,young,chow,v3
s2,aged,chow,v3
s3,young,HFD,v3.1
```

A batch present in the object with no row in the table is refused by name. Deriving `age` by
pattern-matching a sample name would bake one project's naming into a tool every other project has
to work around.

## Install

```bash
git clone https://github.com/JiaenLin/scIntegrate.git && cd scIntegrate
pip install -e '.[run]'                 # anndata + scanpy + matplotlib
scintegrate doctor                      # measure the env before trusting it
```

Then the methods you want compared, each its own extra:

```bash
pip install -e '.[harmony]'      # harmonypy
pip install -e '.[bbknn]'        # bbknn
pip install -e '.[scib]'         # the benchmark — without it, no default is chosen
pip install -e '.[scvi]'         # scVI AND scANVI (one package), pulls torch, ~2.5 GB
pip install -e '.[all]'          # everything
```

The measurement layer is **numpy + scikit-learn**, and `doctor` is stdlib-only so it runs in the
interpreter you are worried about. A capability that is absent is named in the report, with what it
costs you — a method missing from a comparison changes what the comparison means.

`setup/install_env.sh` creates a known-good environment or audits the one you have:

```bash
setup/install_env.sh --check                       # audit this interpreter, change nothing
setup/install_env.sh --prefix ~/envs/scintegrate   # create one from the lock
```

`setup/environment.lock.yml` is captured from a **working install**, not composed from bounds:
`pyproject.toml` declares what scIntegrate needs to *import*, and the lock is what a result needs to
*reproduce*.

```bash
python tests/test_design.py     # the design locks; no data and no pytest needed
python tests/test_version.py
```

## Running on a cluster

`docs/PBSPRO.md` is a working PBS Pro recipe: queue sizing, why the GPU request must be exact, and
where the logs have to go so they survive the job. Nothing in it is specific to one site.

## What it cannot tell you

Nothing here establishes that a batch effect is technical. Where a biological factor is confounded
with a batch factor, a method that removes "batch" may be removing the biology, and no mixing
statistic can separate them. The figures show what was moved; they cannot tell you whether it
should have been.

Nor can it show that your labels are correct. They are inherited from the annotation, and every
limit recorded there applies to every number here. A metric scored against a wrong label set is
precise about the wrong thing.
