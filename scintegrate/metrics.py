"""What integration did, measured two ways that pull against each other.

WHY TWO AND NOT ONE

Every integration method can be made to score perfectly on mixing alone: destroy the structure
and every neighbourhood becomes a random sample of libraries. So mixing is reported ONLY beside
what it cost — how much of the biological structure survived. A method that mixes better and
retains less has not won; it has made a trade, and the trade is the finding.

This module computes numbers. It does not rank methods and it does not choose one. The choice
needs the figures beside it, which is a standing instruction on this project and the reason the
report shows every method at the same scale.
"""
from __future__ import annotations
import numpy as np


def _knn(X, k):
    from sklearn.neighbors import NearestNeighbors
    n = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=n).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def mixing(emb, batch, k=30):
    """Per-cell share of the neighbourhood NOT from the cell's own batch, and its mean.

    Reported against the share you would expect if batches were mixed at random — which is not
    1.0, and is not the same for every cell, because the batches differ in size. A raw
    "72% foreign" means nothing without the number it is being compared to.
    """
    batch = np.asarray(batch)
    idx = _knn(np.asarray(emb), k)
    own = batch[:, None] == batch[idx]
    foreign = 1.0 - own.mean(axis=1)
    _, counts = np.unique(batch, return_counts=True)
    p = counts / counts.sum()
    expected = 1.0 - float((p ** 2).sum())          # chance level for THIS batch composition
    return {"foreign_mean": float(foreign.mean()), "foreign_expected": expected,
            "ratio_to_chance": float(foreign.mean() / expected) if expected else float("nan"),
            "per_cell": foreign}


def structure_retained(emb_before, emb_after, k=30):
    """Share of each cell's kNN that survives the transformation.

    The cost side of the trade. 1.0 means the embedding did not move anything; 0.0 means every
    neighbourhood was rebuilt from scratch, which no biological claim survives.
    """
    a, b = _knn(np.asarray(emb_before), k), _knn(np.asarray(emb_after), k)
    keep = np.array([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)])
    return {"knn_retained_mean": float(keep.mean()), "per_cell": keep}


def label_coherence(emb, labels, k=30):
    """Share of each cell's neighbourhood carrying its own label.

    A property of the graph, not a quality score: a cell type that genuinely sits between two
    others scores low without being wrong. It is here because a method that raises mixing while
    lowering this has moved cells away from their own kind.
    """
    labels = np.asarray(labels, dtype=object)
    idx = _knn(np.asarray(emb), k)
    same = labels[:, None] == labels[idx]
    return {"label_coherence_mean": float(same.mean(axis=1).mean()),
            "per_cell": same.mean(axis=1)}


def assess(emb_before, emb_after, batch, labels, k=30):
    """All three, for one method. Returns plain floats plus the per-cell arrays for the figures."""
    m = mixing(emb_after, batch, k)
    s = structure_retained(emb_before, emb_after, k)
    c = label_coherence(emb_after, labels, k)
    return {
        "k": k,
        "foreign_mean": m["foreign_mean"], "foreign_expected": m["foreign_expected"],
        "ratio_to_chance": m["ratio_to_chance"],
        "knn_retained_mean": s["knn_retained_mean"],
        "label_coherence_mean": c["label_coherence_mean"],
        "_per_cell": {"foreign": m["per_cell"], "retained": s["per_cell"],
                      "coherence": c["per_cell"]},
    }


# ============================================================ the composite, and when it lies

#: scIB's convention: biological conservation weighted above batch removal, because a method
#: that mixes perfectly and destroys biology has not integrated anything. Exposed, not hidden -
#: this weight IS the value judgement, and a different downstream question wants a different one.
DEFAULT_W_BIO = 0.6


def composite(rows, w_bio=DEFAULT_W_BIO):
    """One score per method: w_bio * biological conservation + (1-w_bio) * batch removal.

    **Batch removal** is the mixing ratio to chance, clipped at 1.0 - past chance is not "more
    integrated", it is over-correction, and rewarding it would rank a method higher for
    scattering cells further than random.
    **Biological conservation** is the mean of kNN retention and label coherence.

    Both terms are in [0, 1] and the score is comparable ACROSS METHODS ON ONE DATASET only.
    It is not comparable between datasets: the chance level, the label set and the neighbourhood
    size all differ.
    """
    out = []
    for r in rows:
        batch_removal = min(float(r["ratio_to_chance"]), 1.0)
        ret = 1.0 if r["method"] == "none" else float(r["knn_retained_mean"])
        bio = 0.5 * (ret + float(r["label_coherence_mean"]))
        out.append({**r, "batch_removal": batch_removal, "bio_conservation": bio,
                    "score": w_bio * bio + (1.0 - w_bio) * batch_removal, "w_bio": w_bio})
    return out


def confounding(obs, batch_key, bio_factors):
    """Is a biological factor NESTED inside the batch key?

    This is the question that decides whether a recommendation is safe to make, and no mixing
    statistic can answer it. If every cell in a batch shares one value of `age`, then `age` is
    constant within batch, and a method that removes between-batch variation removes the
    age difference WITH it. The score will happily rank that method first: it mixed the batches,
    which is exactly what it was asked to do and exactly what destroys the comparison.

    Returns one entry per factor: nested (and therefore unrecoverable by any correction on this
    key), aliased (the factor and the batch key partition the cells identically), or separable.
    """
    import numpy as np
    b = np.asarray(obs[batch_key].astype(str))
    out = {}
    for f in bio_factors:
        if f not in obs:
            out[f] = {"status": "absent", "detail": f"no obs column {f!r}"}
            continue
        v = np.asarray(obs[f].astype(str))
        per_batch = {lvl: set(v[b == lvl]) for lvl in set(b)}
        nested = all(len(s) == 1 for s in per_batch.values())
        aliased = nested and len(set(b)) == len(set(v))
        out[f] = {
            "status": "aliased" if aliased else ("nested" if nested else "separable"),
            "levels": sorted(set(v))[:6],
            "detail": (f"every {batch_key} carries exactly one {f}"
                       if nested else
                       f"{f} varies within at least one {batch_key}"),
        }
    return out


def recommend(scored, confounds, w_bio=DEFAULT_W_BIO):
    """The ranking, and a recommendation ONLY when one is safe to make.

    It refuses - and says why - when a declared biological factor is nested inside the batch
    key. That is not caution for its own sake: on such a design the highest-scoring method is
    the one that most thoroughly removed the contrast you intend to measure, and the score
    cannot tell the difference. A number that ranks confidently in exactly the case where it is
    wrong is worse than no number.
    """
    ranked = sorted(scored, key=lambda r: -r["score"])
    blocking = {f: c for f, c in confounds.items() if c["status"] in ("nested", "aliased")}
    if blocking:
        return {"ranked": ranked, "recommended": None, "w_bio": w_bio,
                "refused_because": blocking,
                "reason": ("a declared biological factor is nested inside the batch key, so any "
                           "correction on that key removes the biology with the batch. The "
                           "ranking below is still computed and is still readable as 'what each "
                           "method did'; it must not be read as 'which to use'.")}
    return {"ranked": ranked, "recommended": ranked[0]["method"] if ranked else None,
            "w_bio": w_bio, "refused_because": {},
            "runner_up": ranked[1]["method"] if len(ranked) > 1 else None,
            "margin": round(ranked[0]["score"] - ranked[1]["score"], 4) if len(ranked) > 1 else None}
