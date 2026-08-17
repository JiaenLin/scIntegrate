"""Is integration needed? Measured per cell type, and per group, before any method is run.

WHY THIS IS A SEPARATE STEP FROM INTEGRATING

The global mixing number cannot answer the question. A cohort can look badly mixed because one
abundant population is split by library while every other population is already interleaved -
and correcting the whole manifold for that is a large intervention justified by one cell type.
It can equally look well mixed because the abundant populations dominate the average while a
rare one sits in ten separate islands. The question "is integration needed" is really "needed
FOR WHAT, and WHERE", so the unit of measurement is the cell type.

THE MEASUREMENT, AND THE TRAP IT IS BUILT AROUND

Within one cell type, rebuild the neighbourhood graph among just those cells and ask what share
of each cell's neighbours come from a different library. Compare it to the share random mixing
would give FOR THAT SUBSET'S library composition - which is not 0.5, and is not the same for two
cell types with different sample balance.

Then do it again with the biological factor in place of the library. This is the part that
matters and the part usually skipped. If a cell type's libraries do not mix, that is batch
structure. If its AGE GROUPS also do not mix, in the same populations and to a similar degree,
then the library structure and the biology are the same structure, and a correction on the
library key removes the contrast the study exists to measure. No mixing statistic on a corrected
embedding can recover that distinction, because by then it is gone.

Nothing here returns a verdict. It returns two columns whose comparison is the finding, and the
standing instruction on this project is that the decision is made with the figures in view.
"""
from __future__ import annotations
import numpy as np

from .metrics import _knn


def _chance(values):
    """Share of neighbours you would expect from a different level, under random mixing.

    1 - sum(p^2) for the composition actually present. For two equal levels 0.50; for a 90/10
    split 0.18. A raw "72% foreign" is uninterpretable without this number beside it.
    """
    _, c = np.unique(np.asarray(values), return_counts=True)
    p = c / c.sum()
    return float(1.0 - (p ** 2).sum())


def mixing_within(emb, factor, k=30):
    """Foreign-neighbour share and its chance level, among the cells handed in.

    The caller subsets to one cell type first, so the graph is rebuilt inside that population
    rather than inherited from the whole manifold - otherwise a rare type's neighbours are mostly
    other types and the number measures the cohort, not the type.
    """
    factor = np.asarray(factor)
    n_lvl = len(np.unique(factor))
    if n_lvl < 2:
        return {"n": int(len(factor)), "levels": n_lvl, "foreign": None, "expected": None,
                "ratio": None, "why": "only one level present, nothing to mix"}
    idx = _knn(np.asarray(emb), k)
    foreign = 1.0 - (factor[:, None] == factor[idx]).mean(axis=1)
    exp = _chance(factor)
    return {"n": int(len(factor)), "levels": n_lvl, "foreign": float(foreign.mean()),
            "expected": exp, "ratio": float(foreign.mean() / exp) if exp else None,
            "why": None}


def per_celltype(emb, labels, batch, factors=None, k=30, min_per_k=3, is_real=None):
    """One row per cell type: how well the libraries mix inside it, and how well each factor does.

    `min_per_k` guards a degenerate measurement. With k=30 inside a population of 47 cells the
    neighbourhood is most of the population, so the foreign share approaches the composition
    itself and the ratio approaches 1.0 whatever the geometry - a rare type would be reported as
    perfectly mixed because it is small. Types below min_per_k * k cells are reported with their n
    and no ratio, which is an absence a reader can see.

    `is_real` masks annotator sentinels (EXCLUDED / UNRESOLVED). They are not cell types and are
    reported separately rather than scored as populations.
    """
    labels = np.asarray(labels)
    batch = np.asarray(batch)
    emb = np.asarray(emb)
    keep = np.ones(len(labels), dtype=bool) if is_real is None else np.asarray(is_real, bool)
    rows = []
    floor = min_per_k * k
    for lab in sorted(set(labels[keep])):
        m = keep & (labels == lab)
        n = int(m.sum())
        row = {"label": lab, "n": n, "min_cells": floor}
        if n < floor:
            row.update({"measured": False,
                        "why": f"{n:,} cells is below {min_per_k}x k={k}: inside a population "
                               f"this small the neighbourhood is most of it, and the ratio "
                               f"approaches 1.0 from smallness rather than from mixing"})
            rows.append(row)
            continue
        row["measured"] = True
        row["batch"] = mixing_within(emb[m], batch[m], k)
        row["factors"] = {f: mixing_within(emb[m], np.asarray(v)[m], k)
                          for f, v in (factors or {}).items()}
        rows.append(row)
    return rows


def dominance(labels, batch, is_real=None):
    """For each cell type, the largest share held by any single library.

    A population that is 90% one animal is a different object from one drawn evenly from ten,
    and integration cannot fix it: there is nothing to align it with. This is here so that a
    cell type reported as 'badly mixed' can be checked for whether it is a population at all.
    """
    labels, batch = np.asarray(labels), np.asarray(batch)
    keep = np.ones(len(labels), dtype=bool) if is_real is None else np.asarray(is_real, bool)
    out = []
    for lab in sorted(set(labels[keep])):
        m = keep & (labels == lab)
        b, c = np.unique(batch[m], return_counts=True)
        j = int(np.argmax(c))
        out.append({"label": lab, "n": int(m.sum()), "n_batches": int(len(b)),
                    "top_batch": str(b[j]), "top_share": float(c[j] / c.sum())})
    return out


def summarise(rows, indicates_below=0.80):
    """Count how many measured cell types show batch structure, and how many of those are
    ALSO unmixed on a biological factor - which is the pair that decides whether a correction
    on the batch key can be read as removing a technical effect.

    `indicates_below` is a DECLARED threshold, not a discovered one: a within-type mixing ratio
    below this share of chance is called batch structure. It is exposed on the command line
    because where the line falls is a judgement about how much residual structure the next stage
    can tolerate, and that belongs to whoever is answering the next question.
    """
    meas = [r for r in rows if r.get("measured")]
    unmixed, confounded, clean = [], [], []
    for r in rows:
        if not r.get("measured"):
            continue
        rb = r["batch"].get("ratio")
        if rb is None:
            continue
        if rb < indicates_below:
            unmixed.append(r["label"])
            # is a declared biological factor ALSO unmixed here, to a similar degree?
            hit = {f: v["ratio"] for f, v in (r.get("factors") or {}).items()
                   if v.get("ratio") is not None and v["ratio"] < indicates_below}
            (confounded if hit else clean).append(r["label"])
            r["also_unmixed_factors"] = hit
    return {
        "threshold": indicates_below,
        "n_types_measured": len(meas),
        "n_types_below": len(unmixed),
        "types_below": unmixed,
        "types_below_and_factor_also_unmixed": confounded,
        "types_below_batch_only": clean,
        "indication": _indication(len(meas), unmixed, confounded),
    }


def _indication(n_meas, unmixed, confounded):
    """The sentence the report prints. An INDICATION, labelled as one - never a verdict.

    Whether integration is needed is a question about what the study is for, and nothing measured
    here knows which question is being asked.
    """
    if not n_meas:
        return ("No cell type had enough cells to measure at this k. Nothing here indicates "
                "anything about integration; lower k or pool labels first.")
    if not unmixed:
        return (f"None of the {n_meas} measured cell types falls below the threshold: within "
                f"every population large enough to measure, the libraries already mix at or near "
                f"chance. On this evidence integration is not indicated, and `none` is a "
                f"defensible choice. Read the figures before accepting that.")
    frac = len(unmixed) / n_meas
    s = (f"{len(unmixed)} of {n_meas} measured cell types fall below the threshold, so batch "
         f"structure is present" + (" in most populations" if frac > 0.5 else
                                    " in a minority of populations") + ". ")
    if confounded:
        s += (f"In {len(confounded)} of those, a declared biological factor is ALSO unmixed: "
              f"{', '.join(confounded)}. There the library structure and the biology are the "
              f"same structure, and a correction on the batch key removes the contrast with the "
              f"batch. That cannot be recovered afterwards from a corrected embedding.")
    else:
        s += ("In none of them is a declared biological factor also unmixed, so on this evidence "
              "the structure is separable from the factors declared.")
    return s
