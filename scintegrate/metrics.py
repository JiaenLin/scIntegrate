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
