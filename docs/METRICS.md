# What every number means, and what it cannot mean

Two families of number, computed for different reasons. The tool's own kNN measurements need
nothing but numpy and are therefore never absent. The scIB benchmark is the published convention
and is what names a default embedding.

## Family one — the tool's own, and why they exist at all

### Foreign-neighbour share, and its ratio to chance

For each cell, the share of its `k` nearest neighbours that come from a different batch. Averaged.

Reported **as a ratio to the share random mixing would give**, and that denominator is not 1.0 and
is not the same for every measurement. For two equal batches, chance is 0.50. For a 90/10 split it
is 0.18. For one cell type whose libraries happen to be balanced and another where one library
dominates, it differs between the two rows of the same table.

```
chance = 1 - sum(p_i^2)      p_i = share of level i among the cells being measured
ratio  = observed_foreign_share / chance
```

**1.00 is fully mixed.** Above 1.00 is not "better integrated" — it is more dispersed than random,
which is over-correction.

A raw percentage without its chance level is uninterpretable, and this is the single most common
way a mixing figure misleads: "72% foreign neighbours" is excellent for two equal batches and
impossible to achieve for a 90/10 split.

### kNN retention

The share of each cell's neighbourhood that survives the correction, measured against `none`. This
is the **cost** side, and it is the reason mixing is never reported alone.

Any embedding can score perfectly on mixing: destroy the structure and every neighbourhood becomes
a random sample of libraries. Retention is what distinguishes a method that aligned the batches
from one that shuffled the cells. **Higher mixing with lower retention is a trade, not a victory.**

It is not defined for `none`, which is the baseline it is measured against, and shows as `—`.

### Label coherence

The share of each cell's neighbourhood carrying its own label. **A property of the graph, not a
quality score** — a cell type that genuinely sits between two others scores low without being
wrong. It is here because a method that raises mixing while lowering this has moved cells away from
their own kind.

### Per-cell-type mixing — the measurement `assess` is built around

The three above, computed **inside one cell type at a time**, with the graph rebuilt among just
those cells.

Rebuilding matters. Inherit the whole manifold's neighbours and a rare type's neighbours are mostly
*other types*, so the number describes the cohort rather than the type. Subsetting first asks the
question that was meant: within this population, do the libraries sit together?

Then the same computation with a **declared biological factor** in place of the library. The
comparison of those two columns is the finding:

| batch | factor | reading |
|---|---|---|
| high | — | libraries already mix in this population; nothing to correct here |
| low | high | batch structure that is **separable** from the declared factor |
| low | low | the library structure and the biology are **the same structure**. A correction on the batch key removes the contrast with the batch, and nothing measured afterwards can separate them |

**Small populations are not measured.** Below `--min-per-k × k` cells the neighbourhood is most of
the population, so the ratio approaches 1.0 from smallness rather than from mixing — a rare type
would be reported as perfectly mixed *because it is rare*. Those rows carry their `n` and no ratio,
which is an absence a reader can see.

## Family two — scIB

Each metric is called on its own and returns either a number or the reason it could not run. The
aggregate is scIB's convention:

```
bio    = mean of the biological-conservation metrics
batch  = mean of the batch-correction metrics
total  = w_bio * bio + (1 - w_bio) * batch        w_bio defaults to 0.6
```

### Biological conservation

| metric | what it asks |
|---|---|
| `nmi` | agreement between clustering the corrected space and the annotation |
| `ari` | the same, adjusted for chance |
| `asw_label` | how separated the cell types are in the corrected space |
| `isolated_labels_f1` | whether the cell types present in the fewest batches survive |
| `isolated_labels_asw` | the same, by silhouette rather than clustering |
| `clisi` | local diversity of cell TYPES per neighbourhood — lower is better, rescaled so higher is |

NMI and ARI need a clustering, and the resolution is **optimised against the label set** rather
than fixed. A single resolution flatters whichever method happens to produce clusters at that
granularity.

### Batch correction

| metric | what it asks |
|---|---|
| `graph_connectivity` | whether each cell type stays one connected component |
| `asw_batch` | how little the batches separate within each cell type |
| `ilisi` | local diversity of BATCHES per neighbourhood — higher is better |
| `kbet` | whether each neighbourhood's batch composition matches the global one |
| `pcr` | how much batch-explained variance the correction removed |

`pcr` compares before with after, so it cannot be computed from the result alone — `none` is the
`pre` object, which is one more reason it is always run and always first.

### Absences are cells with words in them

`kBET` needs `rpy2` and the kBET R package and is normally absent on a plain Python install. It is
reported as absent, with its reason.

**An empty cell reads as a zero, and a batch score of zero and a batch score that does not exist
lead to opposite decisions.** So nothing is ever blank, `NaN` is treated as an absence at every
layer rather than averaged in, and **the number of metrics behind each aggregate travels with it**.
A total from eight metrics is not comparable with one from nine, and when two methods' totals rest
on different counts the report says so instead of ranking them silently.

### `embed` and `graph` are different kinds and are not one column

Harmony, scVI and scANVI return corrected coordinates and are scored with scIB's `type_="embed"`.
BBKNN corrects the neighbour **graph** and leaves coordinates alone; it is scored with
`type_="knn"`, on connectivities.

Passing BBKNN's UMAP off as an embedding would make its retention figure answer a different
question from everyone else's — a UMAP is two dimensions of a nonlinear layout, not the space the
correction happened in. The `kind` column exists so those two are never read as one.

## What none of it can tell you

**Nothing here establishes that a batch effect is technical.** Every metric locates structure. None
can say whether removing it was right, because that depends on the question asked next, and no
metric knows what that question is.

**A metric scored against a wrong label set is precise about the wrong thing.** The labels are
inherited, and every limit recorded by whatever produced them still applies here.

**The scores are comparable across methods on one dataset and nowhere else.** The chance level, the
label set, the neighbourhood size and the batch composition all differ between datasets, so a total
of 0.71 here and 0.71 elsewhere are not the same claim.

## Two parameters that are judgements, not constants

`--w-bio` (default 0.6) is the weight between conservation and correction. It asserts how much
residual batch structure is worth trading for retained biology. A different downstream question
wants a different answer, which is why it is a flag and not a literal.

`--indicates-below` (default 0.80) is where the line falls between "mixes acceptably" and "batch
structure". It is **declared, not discovered**. Every ratio is in the tables, so a different line
can be drawn without re-running anything.
