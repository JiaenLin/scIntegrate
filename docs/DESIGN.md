# Design

Why scIntegrate behaves the way it does. Moved here from the README so that
document can describe the tool; the reasoning is unchanged.

See also [METHODS.md](METHODS.md) for every parameter each method is run at, and
[METRICS.md](METRICS.md) for what each number means.

---

## The expensive half is written before the fragile half

Training is what costs hours. Scoring is what breaks: it depends on a pandas version, a compiled
LISI helper, whether `rpy2` is installed, a walltime.

So **the object is written as soon as the embeddings exist**, complete but for the benchmark, and
rewritten in place once the scores are there. A scoring failure now costs the metrics, not the
models — and `score` recomputes those in minutes from the file that survived.

This is not hypothetical. A twelve-hour run once completed every method, produced a default
embedding, and then died serialising a list of dicts into `uns`; what landed was 10 MB of `obs`
and `var` with no `X`, no layers and no embeddings.

---

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

---

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

---

## An absence is named, never a blank cell

Every scIB metric is computed on its own and returns either a number or the reason it could not.
`kBET` needs `rpy2` and the kBET R package and is usually absent on a plain Python install; it is
reported as absent. **An empty cell reads as a zero**, and a batch score of zero and a batch score
that does not exist lead to opposite decisions.

The count of metrics behind each aggregate travels with it, so a total from eight metrics is never
quietly compared against one from nine — and when two methods' totals rest on different counts, the
report says so instead of ranking them silently.

---

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

---

## `none` is a method, and it is never optional

A comparison that omits the un-integrated case can only answer which correction is strongest, never
whether any was warranted. Leave it out of `--methods` and it is put back, and put first — it is the
baseline every retention figure is measured against.

BBKNN is kept honest the same way. It corrects the **graph**, not the coordinates, so there is no
corrected space to return; it declares itself a `graph` method and is scored on connectivities
(scIB's `type_="knn"`). Returning its UMAP as though it were an embedding would make its retention
figure answer a different question from everyone else's, and the `kind` column exists so those two
are never read as one.

---

## Every panel at the same scale

No integration decision should be made on metrics alone. Each method is drawn coloured by cell type,
by library, by every declared factor, and with one library highlighted against everything else in
grey — all sharing one set of axis limits, because per-panel autoscaling makes a dispersed method
look compact.

A number can say a population was mixed. Only the picture separates *aligned with its counterparts*
from *dispersed uniformly through everything*.
