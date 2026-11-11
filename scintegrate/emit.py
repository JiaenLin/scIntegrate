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


#: The suffix a write in progress carries. Not one the run's product list reads (status.products_of),
#: so a file a crash or a kill left behind is never listed as the object.
WRITING = ".writing"


def write_h5ad(adata, path, **kw):
    """The only write path in this package. Holds the encoding guarantee, and writes ALL OR NOTHING.

    The file is written beside its name, under WRITING, and renamed into place only once the writer
    has finished. A write the writer gives up on part-way used to leave a half-written file under
    the object's own name (502 cells, no genes, no embeddings: W046's verifier), which the run's
    status then listed as its object; and the rewrite after scoring opens the file it replaces, so
    a failure there destroyed the embeddings written first precisely so that scoring could not lose
    them. Now a failed write removes its own partial file and leaves the earlier object as it was.
    """
    import os
    from pathlib import Path
    path = Path(path)
    part = path.with_name(path.name + WRITING)
    try:
        with classic_string_encoding():
            adata.write_h5ad(str(part), **kw)
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return path


def _words(e):
    """The writer's own words for a refusal, on one line."""
    return f"{type(e).__name__}: " + " ".join(str(e).split())[:240]


def _refusal_of(obs=None, var=None):
    """Write an object holding only these tables through write_h5ad, into a temporary directory, and
    return the writer's words if it refuses them, else None."""
    import tempfile
    from pathlib import Path
    import anndata as ad
    with tempfile.TemporaryDirectory(prefix="scintegrate-names-") as d:
        try:
            write_h5ad(ad.AnnData(obs=obs, var=var), Path(d) / "names.h5ad")
        except Exception as e:                                             # noqa: BLE001
            return _words(e)
    return None


def unwritable(obs, var=None):
    """What the object's writer refuses in the obs and var this run will write, ASKED OF THE WRITER.

    Not a copy of its rules: a copy is right only for the version it was copied from, and a name it
    misses is found again where W046's verifier found it, at the write, after every method has
    trained. The tables are written through `write_h5ad` itself, into a temporary directory, before
    anything is computed. Measured on anndata 0.13.2 over h5py 3.16 (the lock's pins), the writer
    refuses as a column name a '/' anywhere, exactly '_index', the empty name, '.', and a NUL, and
    refuses a name repeated or the index's name on a column whose values differ; it takes spaces,
    dots, a leading underscore, non-ASCII, newlines and quotes. Another version's rules are its own,
    and this asks them.

    Returns [(table, column, the writer's words)], one per column it refuses on its own; a refusal
    no single column explains is returned once, with column None. [] when it takes them.
    """
    whole = _refusal_of(obs, var)
    if whole is None:
        return []
    out = []
    for axis, df in (("obs", obs), ("var", var)):
        if df is None:
            continue
        for i, c in enumerate(df.columns):
            w = _refusal_of(**{axis: df.iloc[:, [i]]})
            if w is not None:
                out.append((axis, c, w))
    return out or [("obs", None, whole)]


def _uns_key_refused(key, _seen={}):
    """Whether the writer refuses `key` as a key of uns, asked of the writer (an in-memory file)."""
    if key not in _seen:
        import io
        import h5py
        try:
            from anndata.io import write_elem
        except ImportError:                                                # anndata < 0.11
            from anndata.experimental import write_elem
        try:
            with h5py.File(io.BytesIO(), "w") as f:
                write_elem(f, "uns", {key: 0})
            _seen[key] = False
        except Exception:                                                  # noqa: BLE001
            _seen[key] = True
    return _seen[key]


def build(source, results, obs_keep, *, chosen=None, benchmark=None, assessment=None,
          constraint=None, provenance=None):
    """Assemble the deliverable from the input object and the method results.

    `obs_keep` is the whole of obs that survives: the batch key, the label columns, the design
    factors and the design's key column. Everything else is dropped, because an object carrying
    fifty label columns from an upstream sweep makes a reader guess which one an embedding was scored against - and the answer
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

    A DICT KEYED ON A NAME THE WRITER REFUSES becomes a JSON string the same way. Such keys are the
    run's own values, not this tool's words: a sentinel label ('no/call') counted in the provenance,
    a method name asked for and absent. Both are correct inputs; refusing them would block a run the
    object can carry, and writing them raised at the rewrite after scoring (W046 fix 1). Every other
    dict is kept as it was.
    """
    if isinstance(x, dict):
        d = {str(k): _plain(v) for k, v in x.items()}
        if any(_uns_key_refused(k) for k in d):
            return json.dumps(d, default=str)
        return d
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
