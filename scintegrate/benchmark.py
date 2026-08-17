"""The scIB benchmark, computed metric by metric so that an absence is a named cell.

WHY EACH METRIC IS CALLED SEPARATELY RATHER THAN THROUGH scib.metrics.metrics()

The aggregate entry point takes twenty flags and returns a DataFrame in which a metric that could
not be computed is indistinguishable from a metric that scored zero. Both appear as an empty or
NaN cell, and NaN silently propagates into any mean taken over the column. A batch score of 0.0
and a batch score that does not exist lead to opposite decisions, so they must not look alike.

Every metric here is therefore called in isolation and returns either a float or the reason it
could not run. `kBET` needs rpy2 and the kBET R package and is expected to be absent on a plain
python install: it is reported as absent, and the aggregate says how many metrics it was taken
over, so a total computed from eight metrics is never quietly compared with one computed from
nine.

THE AGGREGATION IS scIB's OWN CONVENTION and nothing cleverer: the batch-correction score is the
mean of the batch metrics, the biological-conservation score is the mean of the bio metrics, and
the total is 0.6 * bio + 0.4 * batch. That 0.6 is a value judgement, not a constant of nature -
it says a method that mixes perfectly and destroys biology has not integrated anything - and it
is exposed on the command line for the same reason.
"""
from __future__ import annotations
import warnings

#: metric -> side of the ledger it counts toward. scIB's grouping.
BIO = ("nmi", "ari", "asw_label", "isolated_labels_f1", "isolated_labels_asw", "clisi")
BATCH = ("graph_connectivity", "asw_batch", "ilisi", "kbet", "pcr")

#: what each metric answers, for the report's column notes
MEANING = {
    "nmi": "agreement between clustering the corrected space and the annotation (bio)",
    "ari": "the same agreement, adjusted for chance (bio)",
    "asw_label": "how separated the cell types are in the corrected space (bio)",
    "isolated_labels_f1": "whether the cell types present in fewest batches survive (bio)",
    "isolated_labels_asw": "the same, measured by silhouette rather than clustering (bio)",
    "clisi": "local diversity of cell TYPES in each neighbourhood - lower is better, "
             "rescaled so higher is better (bio)",
    "graph_connectivity": "whether each cell type stays one connected component (batch)",
    "asw_batch": "how little the batches separate within each cell type (batch)",
    "ilisi": "local diversity of BATCHES in each neighbourhood - higher is better (batch)",
    "kbet": "whether each neighbourhood's batch composition matches the global one (batch)",
    "pcr": "how much batch-explained variance the correction removed (batch)",
}


def _guard(name, fn, out):
    """Run one metric. A failure is recorded with its reason and never becomes a number."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = fn()
        if v is None:
            out[name] = {"value": None, "why": "the metric returned None"}
            return
        v = float(v)
        if v != v:                                        # NaN is an absence, not a score
            out[name] = {"value": None, "why": "the metric returned NaN"}
            return
        out[name] = {"value": v, "why": None}
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        out[name] = {"value": None, "why": f"{type(e).__name__}: {msg[:200]}"}


def prepare(emb, adata_graph, obs, batch_key, label_key, *, kind, seed=0, n_neighbors=15):
    """The minimal object scIB wants: the labels, the batch, an embedding and a graph.

    Built fresh rather than by copying the input, so nothing else on the object can influence a
    metric. `X_emb` is the name scIB's LISI functions default to.
    """
    import anndata as ad, numpy as np, scanpy as sc
    if kind == "graph":
        A = adata_graph.copy()
        # BBKNN's own graph IS the result. Keep it; do not recompute one.
        if "X_pca" in A.obsm:
            A.obsm["X_emb"] = np.asarray(A.obsm["X_pca"])
    else:
        A = ad.AnnData(X=np.zeros((len(obs), 1), dtype="float32"))
        A.obs_names = list(obs.index.astype(str))
        A.obsm["X_emb"] = np.asarray(emb)
        sc.pp.neighbors(A, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)
    A.obs[batch_key] = list(obs[batch_key].astype(str).values)
    A.obs[label_key] = list(obs[label_key].astype(str).values)
    A.obs[batch_key] = A.obs[batch_key].astype("category")
    A.obs[label_key] = A.obs[label_key].astype("category")
    return A


def score(A_post, A_pre, batch_key, label_key, *, kind, n_cores=1, lisi_subsample=None,
          cluster_seed=0):
    """Every scIB metric, each either a float or a stated absence.

    `A_pre` is the uncorrected object, needed only by PCR, which asks how much batch-explained
    variance the correction removed and therefore cannot be computed from the result alone.
    """
    import scib.metrics as M
    type_ = "knn" if kind == "graph" else "embed"
    out = {}

    # --- clustering, for NMI and ARI. scIB's convention is to optimise the resolution against
    # the label set rather than fix one, because a single resolution flatters whichever method
    # happens to produce clusters at that granularity.
    cl = "scib_cluster"
    # `cluster_why` is bound BEFORE the try, not inside the except. Binding it only on the error
    # path leaves it undefined in the case where the call succeeds and writes no column - a
    # NameError in the one branch that exists to explain a failure.
    cluster_why = "the call wrote no cluster column"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M.cluster_optimal_resolution(A_post, cluster_key=cl, label_key=label_key)
        have_cluster = cl in A_post.obs
    except Exception as e:
        have_cluster = False
        cluster_why = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"

    if have_cluster:
        _guard("nmi", lambda: M.nmi(A_post, cl, label_key), out)
        _guard("ari", lambda: M.ari(A_post, cl, label_key), out)
    else:
        for m in ("nmi", "ari"):
            out[m] = {"value": None, "why": f"no clustering: {cluster_why}"}

    _guard("asw_label", lambda: M.silhouette(A_post, label_key, "X_emb"), out)
    _guard("isolated_labels_f1",
           lambda: M.isolated_labels_f1(A_post, label_key, batch_key, "X_emb", verbose=False), out)
    _guard("isolated_labels_asw",
           lambda: M.isolated_labels_asw(A_post, label_key, batch_key, "X_emb", verbose=False), out)
    _guard("clisi", lambda: M.clisi_graph(A_post, label_key, type_, use_rep="X_emb",
                                         subsample=lisi_subsample, n_cores=n_cores), out)

    _guard("graph_connectivity", lambda: M.graph_connectivity(A_post, label_key), out)
    _guard("asw_batch", lambda: M.silhouette_batch(A_post, batch_key, label_key, "X_emb",
                                                  verbose=False), out)
    _guard("ilisi", lambda: M.ilisi_graph(A_post, batch_key, type_, use_rep="X_emb",
                                         subsample=lisi_subsample, n_cores=n_cores), out)
    _guard("kbet", lambda: M.kBET(A_post, batch_key, label_key, type_, embed="X_emb"), out)
    _guard("pcr", lambda: M.pcr_comparison(A_pre, A_post, batch_key, embed="X_emb"), out)
    return out


def aggregate(metrics, w_bio=0.6):
    """scIB's own weighting, and an explicit count of what each mean was taken over.

    A total from eight metrics must never be silently comparable with a total from nine, so the
    counts travel with the numbers.
    """
    def _real(v):
        # NaN is an absence wearing a number's clothes, and it propagates through any mean it
        # enters. `_guard` already converts it, but this must hold whatever built the dict -
        # a metric dict assembled anywhere else must not be able to poison an aggregate.
        return v is not None and v == v

    def mean_of(keys):
        vals = [metrics[k]["value"] for k in keys
                if k in metrics and _real(metrics[k]["value"])]
        return (sum(vals) / len(vals) if vals else None), len(vals), len(keys)

    bio, n_bio, t_bio = mean_of(BIO)
    bat, n_bat, t_bat = mean_of(BATCH)
    total = None
    if bio is not None and bat is not None:
        total = w_bio * bio + (1.0 - w_bio) * bat
    return {"bio": bio, "batch": bat, "total": total, "w_bio": w_bio,
            "n_bio": n_bio, "of_bio": t_bio, "n_batch": n_bat, "of_batch": t_bat,
            "absent": {k: (v["why"] or "the metric returned NaN")
                       for k, v in metrics.items() if not _real(v["value"])}}


def choose_default(rows):
    """The highest scIB total, and why it is a choice about the EMBEDDING and nothing else.

    Comparability is checked rather than assumed: two methods whose totals rest on different
    numbers of metrics are not ranked against each other silently. If they differ, the fact is
    returned and the report states it.
    """
    def _real(v):
        return v is not None and v == v

    scored = [r for r in rows if _real(r.get("aggregate", {}).get("total"))]
    if not scored:
        return {"default": None, "reason": "no method produced a complete scIB total",
                "ranked": [], "comparable": True}
    counts = {(r["aggregate"]["n_bio"], r["aggregate"]["n_batch"]) for r in scored}
    ranked = sorted(scored, key=lambda r: -r["aggregate"]["total"])
    top = ranked[0]
    return {
        "default": top["method"],
        "total": top["aggregate"]["total"],
        "runner_up": ranked[1]["method"] if len(ranked) > 1 else None,
        "margin": (round(ranked[0]["aggregate"]["total"] - ranked[1]["aggregate"]["total"], 4)
                   if len(ranked) > 1 else None),
        "ranked": [r["method"] for r in ranked],
        "comparable": len(counts) == 1,
        "reason": ("chosen on the scIB total: the mean of the biological-conservation metrics "
                   f"weighted {top['aggregate']['w_bio']}, the mean of the batch-correction "
                   f"metrics weighted {round(1 - top['aggregate']['w_bio'], 3)}"),
    }
