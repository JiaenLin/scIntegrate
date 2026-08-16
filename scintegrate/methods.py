"""The methods compared, including the one that does nothing.

`none` IS A METHOD and is always run first. Stage 3's question is whether integration is NEEDED,
and a comparison that omits the un-integrated case can only answer which correction is strongest
- never whether any was warranted. It is also the baseline every retention metric is measured
against.

Nothing here chooses. Each function returns an embedding; the report shows them side by side.
"""
from __future__ import annotations
import numpy as np

#: name -> (needs_extra_package, one-line description shown in the report)
METHODS = {
    "none":    (None,         "no correction: PCA on the pooled cells, the baseline"),
    "harmony": ("harmonypy",  "Harmony: iterative clustering + linear correction in PC space"),
    "bbknn":   ("bbknn",      "BBKNN: batch-balanced neighbours; corrects the GRAPH, not the PCs"),
}


def available(names):
    """Which requested methods can actually run here, and why the others cannot.

    Reported rather than silently skipped: a method missing from a comparison changes what the
    comparison means, and a reader cannot see the absence of a panel that was never drawn.
    """
    import importlib
    ok, missing = [], {}
    for n in names:
        pkg = METHODS.get(n, (None, ""))[0]
        if n not in METHODS:
            missing[n] = "not a known method"
        elif pkg is None:
            ok.append(n)
        else:
            try:
                importlib.import_module(pkg)
                ok.append(n)
            except ImportError:
                missing[n] = f"needs `pip install {pkg}`"
    return ok, missing


def run(adata, method, batch_key, n_pcs=50, seed=0):
    """Return an embedding for `method`. `adata` must already carry X_pca."""
    import scanpy as sc
    if method == "none":
        return np.asarray(adata.obsm["X_pca"])[:, :n_pcs]
    if method == "harmony":
        import harmonypy
        h = harmonypy.run_harmony(np.asarray(adata.obsm["X_pca"])[:, :n_pcs],
                                  adata.obs, [batch_key], max_iter_harmony=20)
        return np.asarray(h.Z_corr).T
    if method == "bbknn":
        # BBKNN corrects the GRAPH, so there is no corrected PC space to return. The embedding
        # is recomputed from its graph, and that difference is stated in the report rather than
        # hidden: its retention score is not measuring the same operation as Harmony's.
        import bbknn
        B = adata.copy()
        bbknn.bbknn(B, batch_key=batch_key, n_pcs=n_pcs)
        sc.tl.umap(B, random_state=seed)
        return np.asarray(B.obsm["X_umap"])
    raise ValueError(f"unknown method: {method}")
