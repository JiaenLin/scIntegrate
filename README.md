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
scintegrate assess    --h5ad joint.h5ad --out results/assess \
    --species homo_sapiens --assay scrna \
    --batch-key sample --label-key cell_type --design design.csv

scintegrate integrate --h5ad joint.h5ad --out results/integrate \
    --species homo_sapiens --assay scrna \
    --batch-key sample --label-key cell_type --design design.csv \
    --colour-by cell_type,cell_type_forced,cell_compartment \
    --methods none,harmony,bbknn,scvi,scanvi

scintegrate score     --h5ad results/integrate/objects/cohort_integrated.h5ad \
    --out results/integrate --species homo_sapiens --batch-key sample --label-key cell_type
```

`assess` answers whether integration is needed without training anything. `integrate` runs the
methods, scores them with scIB, names a default embedding, and writes one object holding every
embedding beside the counts they came from. `score` re-runs the benchmark against an object that
already holds the embeddings and graphs — nothing is retrained, and the object is not rewritten:
it rewrites the tables and the `report.json` that `integrate` wrote, and rebuilds the report from
them.

Each command writes its own run directory: `report.json`, `STATUS.json` and the seal. So `assess`
and `integrate` take different `--out`, and `score` is pointed at the directory `integrate` wrote.
`score` leaves a `report.json` that another command wrote exactly as it found it.

### Species and assay

`--species` is **required** and has no default: a default species is the first cohort's species
carried onto every other. It is written as Ensembl writes it (`homo_sapiens`, `mus_musculus`; the
common names `human`, `mouse` are read the same way) and it is **checked against the object**:
where `var` carries Ensembl gene identifiers and they name another species, the run is refused
before anything is computed, naming both. Without identifiers only the symbol case can be read —
a convention, reported as a note, never a refusal.

`--assay` (`scrna` or `snrna`) is **optional**, because no parameter, threshold or method here
depends on it: cells and nuclei are integrated and scored by the same rules, and a count matrix
cannot show which it holds. It is recorded in the run's provenance and wording when given, and
recorded as `not declared` when it is not — never guessed and never defaulted.

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
    obsm['X_pca']         the uncorrected baseline none, Harmony and BBKNN read: the input's,
                          or refit on the retained cells when withheld cells were in its fit;
                          NaN on withheld cells (report.json pca says which)
    obsm['X_harmony']     one per method compared
    obsm['X_umap_<m>']    one 2-D view per method
    obsm['X_umap']        a copy of the chosen view
    obs                   the batch key, the label columns, the declared design factors
    uns['scintegrate']    the benchmark, the chosen method, the constraint on use, and the
                          run's provenance - the species as declared and as checked, the assay
```

`obs` is deliberately narrow: an object carrying fifty label columns from an upstream sweep makes
a reader guess which one an embedding was scored against.

Every write holds `classic_string_encoding`, so labels and indices land as HDF5 string datasets
rather than the nullable-string groups newer anndata writes by default and most other readers
cannot open. A `README.md` is written beside the object by inspecting the directory, so it
describes what is there rather than what the run intended.

## What a run leaves behind

Four states are distinguishable from the filesystem alone, by a reader who did not watch the run:

| file | written | says |
|---|---|---|
| `STATUS.json` | first as `partial`, last with the outcome | `ok` · `partial` (it died, with `RUNNING.txt` standing and no seal; or it completed without a method it was asked for, named in the headline and in `absent[]`, and sealed) · `refused` with a `fix` · `failed`; the commit read from `.git` by file, the job, the products on disk, `sees` per method, the wrapped package versions |
| `RUNNING.txt` | at start | replaced at exit by one of the two below |
| `SEALED.txt` | exit 0 **and** every expected product present, outcome `ok` or `partial` | the object, the tables and `report.json` |
| `FAILED.txt` | anything else | names what is missing |

`scintegrate describe` prints the declaration — needs, provides, what each method was shown
(`sees`: only scANVI sees the labels), the gates that refuse a run, the constraints that only
classify, the species and assay axes, `cannot_show`, `state_version` — as JSON. Every
ranking table and `report.json` carry `sees` beside each method, because a method trained on the
label column is not sitting the same exam as one that was not.

## The design table

Study factors arrive as a CSV keyed on the batch:

```csv
sample,treatment,timepoint,chemistry
s1,control,d0,v3
s2,treated,d0,v3
s3,control,d7,v3.1
```

A batch present in the object with no row in the table is refused by name. Nothing is derived by
pattern-matching a sample name.

## Tests

```bash
python tests/run_all.py --jobs 4        # every suite, one process each; the exit code is the gate
python tests/test_design.py             # the design locks; no data, no pytest needed
python tests/test_methods_settings.py   # every documented setting matches the source
python tests/test_cohort_axes.py        # a human scRNA cohort through the real CLI
```

The leak guard, `tests/test_portability.py`, reads the cohort's forbidden words from the file
named by `$SCINTEGRATE_FORBIDDEN_TERMS`, so this repository never spells them.

## Part of a family

Four tools share one contract. Each works on its own, and they fit together:

| tool | does |
|---|---|
| [scQC](https://github.com/JiaenLin/scQC) | quality control, and who set each threshold |
| [scAnno](https://github.com/JiaenLin/scAnno) | cell-type labels, only as deep as the evidence goes |
| [scIntegrate](https://github.com/JiaenLin/scIntegrate) | whether you need batch integration, and which method |
| [scProfile](https://github.com/JiaenLin/scProfile) | communication, velocity, pseudotime, differential expression |

[single-cell-harness](https://github.com/JiaenLin/single-cell-harness) runs them as one stack, so a
decision made in one can be taken back later and every tool downstream is told what that costs.

## Changing this tool

Using scIntegrate and changing it are different jobs, and the second one has its own suite.
`DEVPOINTS.yaml` in the root says what can be added here — methods — and what each one has to
declare. The suite lives in the harness:

```bash
git clone https://github.com/JiaenLin/single-cell-harness.git ../single-cell-harness
alias sch='PYTHONPATH=../single-cell-harness python3 -m sch'

sch dev map --root .                            # what plugs in here, and what it must declare
sch dev new method my_method       # a skeleton, a spec written before the code, and a test
sch dev check --point method --name my_method   # checks, cheapest first
```

Every check prints what it does **not** prove, so a green run is not read as more than it is. Two
of them run your addition against the same synthetic cohort written twice, with every column
renamed — code that asks for a role passes both, code that knows a column name passes one.

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

## Versions

**0.6.0.** When `--drop-labels` withholds cells that have rows in the input's `X_pca`, the PCA is
refit on the retained cells, and `none`, Harmony and BBKNN are fitted on that: before, the input's
PCA was reused with the withheld cells in its fit, and the delivered `X_pca` kept their rows. The
same command line therefore gives another `X_pca` (NaN on the withheld cells) and other `none`,
Harmony and BBKNN numbers than 0.5.0 on such an object; scVI and scANVI read no PCA. `report.json`
`pca` says which PCA was read and how it was made; `--keep-input-pca` keeps the old behaviour, as
declared (docs/METHODS.md, KNOWN_ISSUES). No command line 0.5.0 accepted stops. Also in 0.6.0:
Harmony runs on a batch column holding a category no cell carries (it failed), and a requested
method that fails leaves `STATUS.json` `partial`, naming it, where it said `ok`.

**0.5.0.** `--species` is required by `assess`, `integrate` and `score`, and is checked against
the object. `--assay` is new, optional, and recorded. A command line written for 0.4.x stops at
argparse with exit 2 (`the following arguments are required: --species`). Add `--species` with the
species the cells came from, and nothing else in the command changes: the species is checked and
recorded, and no parameter, method or metric reads it. Also in 0.5.0: `score` rewrites only a
`report.json` that `integrate` wrote. iLISI and cLISI are computed on every cell, so on the same
embedding they are the same in every execution: `--lisi-subsample` stops at argparse with exit 2,
because scIB draws that subsample from an unseeded source; drop it and nothing else in the command
changes (KNOWN_ISSUES).
Harmony's fit runs on one thread, so on one machine it returns the same embedding, and the same
scIB numbers, every time. A harmony number from 0.4.x was one draw of a threaded fit and need not
match a 0.5.0 run (KNOWN_ISSUES). Harmony's recorded theta is now the value the fit used, 1, where
the record said 2.0.

## License

MIT — see [LICENSE](LICENSE).
