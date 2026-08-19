# scIntegrate

**Decide whether integration is needed, compare the methods, and get one object carrying the
answer.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

📖 **[Quickstart](docs/QUICKSTART.md)** · **[Methods](docs/METHODS.md)** ·
**[Metrics](docs/METRICS.md)** · **[Known issues](KNOWN_ISSUES.md)**

---

## Install

```bash
git clone https://github.com/JiaenLin/scIntegrate.git && cd scIntegrate
pip install -e '.[run]'      # anndata + scanpy + matplotlib
scintegrate doctor           # measure the environment before trusting it
```

Methods are separate extras:

```bash
pip install -e '.[harmony]'   # harmonypy
pip install -e '.[bbknn]'     # bbknn
pip install -e '.[scib]'      # the benchmark — without it, no default is chosen
pip install -e '.[scvi]'      # scVI and scANVI, pulls torch, ~2.5 GB
pip install -e '.[all]'
```

The measurement layer is numpy and scikit-learn; `doctor` is stdlib-only, so it runs in the
interpreter you are worried about. A capability that is absent is named in the report with what
its absence costs.

`setup/install_env.sh --check` audits the current interpreter and changes nothing;
`--prefix <dir>` builds one from `setup/environment.lock.yml`, which is captured from a working
install rather than composed from bounds.

## Run

```bash
scintegrate assess    --h5ad joint.h5ad --out results/ \
    --batch-key sample --label-key cell_type --design design.csv

scintegrate integrate --h5ad joint.h5ad --out results/ \
    --batch-key sample --label-key cell_type --design design.csv \
    --colour-by cell_type,cell_type_forced,cell_compartment \
    --methods none,harmony,bbknn,scvi,scanvi

scintegrate score     --h5ad results/objects/cohort_integrated.h5ad --out results/ \
    --batch-key sample --label-key cell_type
```

`assess` answers whether integration is needed without training anything. `integrate` runs the
methods, scores them with scIB, names a default embedding, and writes one object holding every
embedding beside the counts they came from. `score` re-runs the benchmark against an object that
already holds the embeddings — nothing is retrained.

The object is written **as soon as the embeddings exist**, complete but for the benchmark, then
rewritten in place once the scores are there. A scoring failure costs the metrics, not the models.

## Methods

| method | corrects | scored on |
|---|---|---|
| `none` | nothing — the baseline | coordinates |
| `harmony` | PC space, iteratively | coordinates |
| `bbknn` | the neighbour graph | connectivities |
| `scvi` | latent space, unsupervised | coordinates |
| `scanvi` | latent space, semi-supervised on the labels | coordinates |

`none` always runs and always runs first; without it a comparison can say which correction is
strongest but never whether any was warranted.

BBKNN produces no corrected coordinate space, so it is scored with scIB's `type_="knn"` and the
metrics needing coordinates are reported **absent** for it — not zero, and not borrowed from
another method's embedding.

📄 **[Every parameter each method is run at](docs/METHODS.md)**, with the package default beside
it and the reason where they differ.

## Reading the result

The default embedding is the highest scIB total under `--w-bio`, which weights biological
conservation against batch correction. That weight is a value judgement and is exposed on the
command line for exactly that reason.

A method trained on the label column it is later scored against is declared in the report wherever
the ranking appears — the biological half of the scIB total is computed against that same column,
so such a ranking is not like-for-like.

Every method is drawn at the same scale, coloured by each label column in turn, by batch, and by
each declared design factor, plus one batch highlighted against everything else in grey. Mixing is
measured **within each cell type**, against chance for that population's own batch composition,
because a cohort-level statistic averages over the populations that differ.

## Output

```
objects/cohort_integrated.h5ad
    X                     log1p of library-size-normalised counts
    layers['counts']      raw integer counts — what the count models and DE read
    layers['lognorm']     the same values as X, named so X's scale is never implicit
    obsm['X_pca']         the uncorrected baseline, kept permanently
    obsm['X_harmony']     one per method compared
    obsm['X_umap_<m>']    one 2-D view per method
    obsm['X_umap']        a copy of the chosen view
    obs                   the batch key, the label columns, the declared design factors
    uns['scintegrate']    the benchmark, the chosen method, the constraint on use
```

`obs` is deliberately narrow: an object carrying fifty label columns from an upstream sweep makes
a reader guess which one an embedding was scored against.

Every write holds `classic_string_encoding`, so labels and indices land as HDF5 string datasets
rather than the nullable-string groups newer anndata writes by default and most other readers
cannot open. A `README.md` is written beside the object by inspecting the directory, so it
describes what is there rather than what the run intended.

## The design table

Study factors arrive as a CSV keyed on the batch:

```csv
sample,age,diet,chemistry
s1,young,chow,v3
s2,aged,chow,v3
s3,young,HFD,v3.1
```

A batch present in the object with no row in the table is refused by name. Nothing is derived by
pattern-matching a sample name.

## Tests

```bash
python tests/test_design.py             # the design locks; no data, no pytest needed
python tests/test_methods_settings.py   # every documented setting matches the source
python tests/test_colour_columns.py     # needs numpy and pandas
```

## Documentation

| | |
|---|---|
| [QUICKSTART](docs/QUICKSTART.md) | the three commands, in the order the questions arrive |
| [METHODS](docs/METHODS.md) | every parameter each method is run at, and where it departs from the package default |
| [METRICS](docs/METRICS.md) | what every number means, and what it cannot mean |
| [OUTPUTS](docs/OUTPUTS.md) | which file is the answer, and which must not be used |
| [DESIGN](docs/DESIGN.md) | why the tool behaves the way it does |
| [PBSPRO](docs/PBSPRO.md) | a working PBS Pro recipe: sizing, GPU request, log placement |
| [KNOWN_ISSUES](KNOWN_ISSUES.md) | measured, not suspected |

## Limits

Nothing here establishes that a batch effect is technical. Where a biological factor is confounded
with a batch factor, a method removing "batch" may be removing the biology, and no mixing
statistic separates them. The figures show what was moved, not whether it should have been.

Nor does anything here show the labels are correct. They are inherited from the annotation, and
every limit recorded there applies to every number here.

## License

MIT — see [LICENSE](LICENSE).
