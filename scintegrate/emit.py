"""The one object the stage delivers, and the string encoding that makes it readable elsewhere.

ONE FILE, NOT A DIRECTORY OF THEM. Every method's embedding lives in the same object, under its
own `obsm` key, beside the counts they were derived from. A directory holding one h5ad per method
invites a downstream step to read whichever it opened first, and nothing in the filename says
which one the benchmark chose. Here the choice is recorded IN the object, so it travels with it.

WHAT THE OBJECT CARRIES

    X                     log-normalised expression, normalised over ALL samples together
    layers['counts']      raw integer counts - what the count models were trained on
    layers['lognorm']     the same values as X, named explicitly
    obsm['X_pca']         the UNCORRECTED baseline, kept so `none` stays comparable forever
    obsm['X_<method>']    each corrected space
    obsm['X_umap_<m>']    each method's 2-D view
    obsm['X_umap']        a copy of the CHOSEN method's view, so plotting works with no argument
    uns['scintegrate']    the benchmark table, which method was chosen and why, and the
                          constraint on what the chosen embedding may be used for

`X` and `layers['lognorm']` deliberately hold the same values. Leaving the identity of `X`
implicit is how an object ends up with nobody able to say whether it is counts, CPM or log1p, and
the six-question README rule on this project exists because that happened.

WHY THE COUNTS ARE CARRIED AT ALL when the embeddings are the point: the next stage's
differential expression reads counts per sample, and it must read them from an object whose cell
set is exactly the one the embedding describes. Handing it the embedding and letting it fetch
counts from upstream is how a cell filtered in one place and not the other becomes a silent
mismatch.
"""
from __future__ import annotations

import json

import numpy as np


class classic_string_encoding:
    """Write string columns as HDF5 string DATASETS, not as nullable-string groups.

    On pandas >= 3 a string column cannot be held as `object` - the new `str` dtype is the default
    and pandas coerces back to it - and anndata >= 0.11 writes that dtype as a NULLABLE STRING: a
    group of `values` + `mask`. The result is valid AnnData, round-trips through anndata perfectly,
    and is unreadable by anything else, because `obs/_index` is no longer a dataset. A viewer
    reports a property access on undefined, which points nowhere near the cause.

    `allow_write_nullable_strings = False` is the only thing that fixes it, and it is the
    COMPATIBLE direction: the classic encoding is what every other reader expects, so the file is
    readable by more things, not fewer.

    Scoped rather than set once at import, because it is a global on the anndata module and this
    library has no business changing how objects written elsewhere in the caller's process are
    stored. Restored even if the write raises.

    Duplicated from its sibling tool rather than imported from it, deliberately: this package must
    install and write a readable file without the annotator present. A guarantee about the
    deliverable cannot be held in a dependency the deliverable does not otherwise need.
    """

    def __init__(self):
        self._prev = None
        self._had = False

    def __enter__(self):
        try:
            import anndata as ad
            self._prev = ad.settings.allow_write_nullable_strings
            ad.settings.allow_write_nullable_strings = False
            self._had = True
        except Exception:                                                  # noqa: BLE001
            self._had = False        # older anndata has no such setting and does not need one
        return self

    def __exit__(self, *exc):
        if self._had:
            import anndata as ad
            ad.settings.allow_write_nullable_strings = self._prev
        return False


def write_h5ad(adata, path, **kw):
    """The only write path in this package. Holds the encoding guarantee."""
    with classic_string_encoding():
        adata.write_h5ad(str(path), **kw)
    return path


def build(source, results, obs_keep, *, chosen=None, benchmark=None, assessment=None,
          constraint=None, provenance=None):
    """Assemble the deliverable from the input object and the method results.

    `obs_keep` is the whole of obs that survives: the batch key, the label columns and the design
    factors. Everything else is dropped, because an object carrying fifty label columns from an
    upstream sweep makes a reader guess which one an embedding was scored against - and the answer
    is recorded in uns here, against the column that is actually present.
    """
    A = source
    keep = [c for c in obs_keep if c in A.obs.columns]
    missing = [c for c in obs_keep if c not in A.obs.columns]
    if missing:
        raise KeyError(f"cannot keep obs column(s) that do not exist: {missing}. "
                       f"obs has: {list(A.obs.columns)}")

    import anndata as ad
    out = ad.AnnData(X=A.X, obs=A.obs[keep].copy(), var=A.var.copy())
    out.obs_names = A.obs_names
    out.var_names = A.var_names

    for name in ("counts", "lognorm"):
        if name in A.layers:
            out.layers[name] = A.layers[name]
    if "lognorm" not in out.layers and A.X is not None:
        # X's identity must never be implicit. If the input did not name it, name it here.
        out.layers["lognorm"] = A.X

    if "X_pca" in A.obsm:
        out.obsm["X_pca"] = np.asarray(A.obsm["X_pca"])

    for r in results:
        m = r["method"]
        if r.get("emb") is not None:
            out.obsm[f"X_{m}"] = np.asarray(r["emb"], dtype="float32")
        if r.get("umap") is not None:
            out.obsm[f"X_umap_{m}"] = np.asarray(r["umap"], dtype="float32")
        # A GRAPH METHOD'S RESULT IS ITS GRAPH. Without this the object carried only a UMAP for
        # BBKNN - two dimensions of a nonlinear layout, which is exactly what this tool refuses to
        # score it on, and which no downstream clustering can use. A reader who wanted the
        # correction could not have it.
        g = r.get("graph")
        if g is not None:
            for slot, mat in g.items():
                if mat is not None:
                    out.obsp[f"{m}_{slot}"] = mat

    # the chosen view under the name every plotting call reaches for by default
    if chosen and f"X_umap_{chosen}" in out.obsm:
        out.obsm["X_umap"] = np.array(out.obsm[f"X_umap_{chosen}"], dtype="float32")

    out.uns["scintegrate"] = _uns(results, chosen, benchmark, assessment, constraint, provenance)
    return out


def _uns(results, chosen, benchmark, assessment, constraint, provenance):
    """Everything a reader needs to know what this object is, without opening the report.

    Kept to plain types - str, float, int, list, dict - because uns is serialised into HDF5 and a
    numpy scalar or a pandas object in here becomes a read error for somebody else's reader.
    """
    return {
        "default_embedding": (f"X_{chosen}" if chosen else "NONE CHOSEN"),
        "default_method": chosen or "none chosen",
        "embeddings": {f"X_{r['method']}": r.get("note", "") for r in results
                       if r.get("emb") is not None},
        "graphs": {f"{r['method']}_connectivities": r.get("note", "") for r in results
                   if r.get("graph")},
        "uncorrected_baseline": "X_pca",
        "X_is": "log1p of library-size-normalised counts, normalised over all samples together",
        "counts_layer": "layers['counts'] - raw integer counts, what the count models read",
        "benchmark": _plain(benchmark or {}),
        "assessment": _plain(assessment or {}),
        "constraint_on_use": constraint or "",
        "provenance": _plain(provenance or {}),
    }


def _plain(x):
    """Recursively convert to types HDF5 and every other reader can hold.

    A LIST OF DICTS IS THE ONE SHAPE THAT LOOKS FINE AND IS NOT. Every element converts cleanly,
    so a recursive walk returns a list of dicts unchanged - and anndata then tries to write it as
    a ragged string array and raises `Can't implicitly convert non-string objects to strings`,
    naming a uns key and nothing about the cause. It happens at the very END of a run, after every
    expensive step has completed, and takes the whole object with it.

    Such a list becomes a JSON string: always writable, readable by anything, and losslessly
    recoverable with `json.loads`. The canonical form of that data is the CSV in `tables/`
    anyway - uns carries it so the object is self-describing, not so it can be computed from.
    """
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if any(isinstance(v, dict) for v in x):
            return json.dumps([_plain(v) for v in x], default=str)
        return [_plain(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    return str(x)
