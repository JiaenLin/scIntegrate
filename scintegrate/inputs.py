"""Reading the object honestly: which column is the answer, what is not a cell type, and
whether the counts are really counts.

THREE THINGS THIS MODULE REFUSES TO DO, each because doing it silently changes a result.

**It does not reconstruct a column that ships measured.** A hierarchical annotation like
`Immune/Myeloid/Macrophage` invites `path.split("/")[0]` as a coarse level. That is a
TRUNCATION of one walk, not an independent annotation of the compartment, and the two are
different objects even when they agree. If you have a measured coarse column, pass `--l1-key`
and it is used. If you do not, the truncation is available but only behind an explicit flag that
names it as one.

**It does not treat a sentinel as a cell type.** An annotator that declines to guess emits
labels like `EXCLUDED` or `UNRESOLVED` for cells it withheld or could not resolve. Those are
statements about the annotation, not populations in the tissue, and every label-based metric -
coherence here, ARI/NMI/cLISI/ASW under scIB - scores them as though they were. They are
identified, counted, reported, and excluded from label metrics ONLY; they stay in the object and
in every embedding, because integration must embed every cell it was given.

**It does not accept normalised values as counts.** scVI and scANVI model counts. Handed log1p
data they train anyway and return a plausible embedding, which is the worst available outcome.
The counts layer is checked for integrality and named if it fails.
"""
from __future__ import annotations
import numpy as np

#: Labels an annotator uses to say "no call", which are NOT cell types.
#:
#: These two are scAnno's declared sentinels, and scIntegrate is the stage that reads scAnno's
#: output - so this is a default that matches its documented upstream, not an assumption about
#: any particular dataset. Override with --label-sentinel, or clear it with --label-sentinel ''
#: for an annotation that uses none.
DEFAULT_SENTINELS = ("EXCLUDED", "UNRESOLVED")


class Refuse(Exception):
    """Raised when continuing would produce a number that looks right and is not."""


def resolve_keys(obs, batch_key, label_key, l1_key=None):
    """Confirm the columns exist, naming what IS there when one does not.

    A KeyError deep inside a metric tells you a column is missing. It does not tell you what the
    object actually offers, which is the only thing that helps.
    """
    cols = list(obs.columns)
    for role, k in (("--batch-key", batch_key), ("--label-key", label_key),
                    ("--l1-key", l1_key)):
        if k is None:
            continue
        if k not in obs:
            raise Refuse(f"{role} {k!r} is not in obs. obs carries {len(cols)} column(s): "
                         + ", ".join(repr(c) for c in cols))
    return {"batch": batch_key, "label": label_key, "l1": l1_key}


def label_view(obs, label_key, sentinels=DEFAULT_SENTINELS):
    """The label column, plus the mask of cells carrying a real call.

    Returns (labels, is_real, found) where `found` maps each sentinel present to its count, so
    the report can state the exclusion rather than absorb it. A sentinel that is declared but
    absent from this object is simply not reported - it is not an error to be careful.
    """
    lab = np.asarray(obs[label_key].astype(str))
    sent = tuple(s for s in (sentinels or ()) if s)
    is_sent = np.isin(lab, sent) if sent else np.zeros(len(lab), dtype=bool)
    found = {s: int((lab == s).sum()) for s in sent if (lab == s).any()}
    return lab, ~is_sent, found


def coarse_labels(obs, label_key, l1_key=None, from_path=False, sep="/"):
    """A coarse colouring for the figures. Measured if you have one, truncated only on request.

    Returns (values, provenance) where provenance is the sentence the report prints beside the
    figure. The figure legend has to say which of the two it is: 'level 1' and 'the first
    component of the path' are not interchangeable claims even when they are the same strings.
    """
    if l1_key:
        return np.asarray(obs[l1_key].astype(str)), f"measured column {l1_key!r}"
    lab = np.asarray(obs[label_key].astype(str))
    if from_path:
        return (np.array([x.split(sep)[0] for x in lab]),
                f"FIRST COMPONENT of {label_key!r}, a truncation of the path and not an "
                f"independent annotation of the compartment")
    return lab, f"the full label {label_key!r} (no coarse column was supplied)"


def check_counts(A, layer="counts"):
    """Is there a raw-count matrix, and is it actually integral?

    Returns (ok, detail). Never raises: a run that only compares harmony and bbknn does not need
    counts, and refusing it for a layer it will not read would be a gate firing on correct
    behaviour. The caller refuses only when a model that needs counts is requested.
    """
    if layer not in A.layers:
        return False, (f"no layers[{layer!r}]. Present: "
                       + (", ".join(map(repr, A.layers.keys())) or "none"))
    X = A.layers[layer]
    sub = X[:2000] if X.shape[0] > 2000 else X
    # `hasattr(sub, "data")` is TRUE for a dense numpy array - ndarray.data is a memoryview, which
    # has no .size and is not the stored values. Only a scipy sparse matrix means by `.data` what
    # is meant here, so dense is tested for FIRST rather than inferred from the absence of an
    # attribute that both types have.
    if isinstance(sub, np.ndarray):
        d = sub.ravel()
    elif hasattr(sub, "data") and not isinstance(sub.data, memoryview):
        d = np.asarray(sub.data)                                  # scipy sparse: stored values
    else:
        d = np.asarray(sub).ravel()
    if d.size == 0:
        return False, f"layers[{layer!r}] has no stored values"
    frac = float(np.mean(d == np.rint(d)))
    if frac < 1.0:
        return False, (f"layers[{layer!r}] is not integral - {100*(1-frac):.2f}% of the first "
                       f"{min(2000, X.shape[0]):,} rows' stored values are not whole numbers. "
                       f"This looks like normalised data, and a count model handed normalised "
                       f"data trains anyway and returns a plausible embedding.")
    return True, (f"layers[{layer!r}] {X.shape[0]:,} x {X.shape[1]:,}, integral on the first "
                  f"{min(2000, X.shape[0]):,} rows, max {float(d.max()):.0f}")


def hvg_mask(A, key="highly_variable"):
    """The highly-variable mask already on the object, or None.

    NOT RECOMPUTED, and no gene class is ever excluded from it. Selecting HVGs restricts what a
    model SEES; it must never be the route by which a gene class disappears from a study. If the
    object carries a mask, that selection was made and recorded upstream and is reused verbatim.
    If it does not, the caller computes one over ALL genes with no class excluded, and says so.
    """
    if key in A.var:
        m = np.asarray(A.var[key]).astype(bool)
        return m if m.any() else None
    return None


def read(path, batch_key, label_key, l1_key=None, sentinels=DEFAULT_SENTINELS,
         coarse_from_path=False):
    """Open the object and return it with everything the stage needs, or refuse by name."""
    import anndata as ad
    A = ad.read_h5ad(path)
    if A.X is None and "lognorm" not in A.layers:
        raise Refuse("the object has neither X nor layers['lognorm']: nothing to embed")
    keys = resolve_keys(A.obs, batch_key, label_key, l1_key)
    lab, is_real, sent = label_view(A.obs, label_key, sentinels)
    coarse, coarse_note = coarse_labels(A.obs, label_key, l1_key, coarse_from_path)
    counts_ok, counts_note = check_counts(A)
    return {
        "adata": A, "keys": keys,
        "batch": np.asarray(A.obs[batch_key].astype(str)),
        "label": lab, "is_real": is_real, "sentinels": sent,
        "coarse": coarse, "coarse_note": coarse_note,
        "counts_ok": counts_ok, "counts_note": counts_note,
        "hvg": hvg_mask(A),
    }


# ------------------------------------------------------------------------------- the design

def read_design(path, samples, sample_col=None):
    """Join a design table onto the batches present, or refuse and say which are unmatched.

    The factors a study is ABOUT are usually not in the annotated object - it carries the
    annotation, not the animal metadata. They arrive here as a CSV keyed on the batch, which
    keeps the tool general: deriving `age` by pattern-matching a sample name would bake one
    project's naming into a tool every other project has to work around.

    Returns (DataFrame indexed by sample, [factor names]).
    """
    import csv
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise Refuse(f"design table {path} has no rows")
    cols = list(rows[0].keys())
    key = sample_col or cols[0]
    if key not in cols:
        raise Refuse(f"--design-sample-col {key!r} not in {path}: columns are "
                     + ", ".join(map(repr, cols)))
    table = {r[key]: r for r in rows}
    want = sorted(set(map(str, samples)))
    unmatched = [s for s in want if s not in table]
    if unmatched:
        raise Refuse(
            f"{len(unmatched)} batch(es) in the object have no row in {path}: "
            + ", ".join(map(repr, unmatched))
            + f".  The table's {key!r} column holds: " + ", ".join(map(repr, sorted(table)[:12]))
            + ".  A design applied to some batches and not others is worse than none, because "
              "the factor silently becomes 'missing' for exactly the cells it omits.")
    factors = [c for c in cols if c != key]
    return table, key, factors
