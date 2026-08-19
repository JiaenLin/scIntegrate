"""The methods compared, including the one that does nothing.

`none` IS A METHOD and is always run first. The stage's question is whether integration is
NEEDED, and a comparison that omits the un-integrated case can only answer which correction is
strongest - never whether any was warranted. It is also the baseline every retention metric is
measured against.

WHAT A METHOD RETURNS, AND WHY IT IS NOT ALWAYS AN EMBEDDING

Harmony, scVI and scANVI return a corrected coordinate space. BBKNN does not: it corrects the
NEIGHBOUR GRAPH and leaves the coordinates alone, so there is no corrected space to hand back.
Earlier this was papered over by returning BBKNN's UMAP as though it were an embedding, which
makes its retention score answer a different question from everyone else's - a UMAP is two
dimensions of a nonlinear layout, not the space the correction happened in.

So a method now declares its KIND. `embed` methods are scored on their coordinates; `graph`
methods are scored on their connectivities, which is what scIB's `type_="knn"` exists for. The
distinction is carried through to the report rather than flattened, because a single column
holding both is a column whose rows are not comparable.

Nothing here chooses. Each entry produces a result; the benchmark ranks and the figures decide.
"""
from __future__ import annotations
import numpy as np

#: name -> (import name of the extra package or None, kind, one-line description for the report)
METHODS = {
    "none": (None, "embed",
             "no correction: PCA over the pooled cells, the baseline every cost is measured "
             "against"),
    "harmony": ("harmonypy", "embed",
                "Harmony: iterative clustering and linear correction in PC space"),
    "bbknn": ("bbknn", "graph",
              "BBKNN: batch-balanced neighbours. Corrects the GRAPH, not the coordinates, so it "
              "is scored on connectivities rather than on a corrected space"),
    "scvi": ("scvi", "embed",
             "scVI: variational autoencoder on raw counts, batch as a covariate. Unsupervised - "
             "it is not shown the cell-type labels"),
    "scanvi": ("scvi", "embed",
               "scANVI: scVI extended with the cell-type labels as semi-supervision. Cells the "
               "annotator declined are passed as genuinely unlabelled, which is what the model "
               "is designed for"),
}

#: methods that model counts and must NOT be handed normalised values
NEEDS_COUNTS = ("scvi", "scanvi")
#: methods that read the annotation
NEEDS_LABELS = ("scanvi",)


def available(names):
    """Which requested methods can run here, and why the others cannot.

    Reported rather than silently skipped: a method missing from a comparison changes what the
    comparison means, and a reader cannot see the absence of a panel that was never drawn.
    """
    import importlib
    ok, missing = [], {}
    for n in names:
        if n not in METHODS:
            missing[n] = f"not a known method (known: {', '.join(METHODS)})"
            continue
        pkg = METHODS[n][0]
        if pkg is None:
            ok.append(n)
            continue
        try:
            importlib.util.find_spec(pkg)
            if importlib.util.find_spec(pkg) is None:
                raise ImportError(pkg)
            ok.append(n)
        except ImportError:
            hint = "scvi-tools" if pkg == "scvi" else pkg
            missing[n] = f"needs `pip install {hint}`"
    return ok, missing


def kind(method):
    return METHODS[method][1]


def _hvg_subset(adata, hvg, n_top_fallback=2000):
    """The view a count model is trained on, and a sentence describing it.

    NO GENE CLASS IS EVER EXCLUDED HERE. Restricting a model to highly-variable genes changes
    what it SEES and is reversible - every gene stays in the object and in the deliverable. It
    must never become the route by which a class of genes quietly leaves a study, so the mask is
    reused from upstream when it exists rather than recomputed with a filter of this tool's own
    devising.
    """
    if hvg is not None:
        return adata[:, hvg], (f"{int(hvg.sum()):,} highly-variable genes, mask reused verbatim "
                               f"from the object's var (not recomputed here)")
    import scanpy as sc, anndata as ad
    # A light object over the counts only. `adata.copy()` here would clone X and every layer to
    # read one of them.
    A = ad.AnnData(X=adata.layers["counts"], var=adata.var[[]].copy())
    sc.pp.highly_variable_genes(A, n_top_genes=n_top_fallback, flavor="seurat_v3",
                                batch_key=None)
    m = np.asarray(A.var["highly_variable"]).astype(bool)
    return adata[:, m], (f"{int(m.sum()):,} highly-variable genes computed here over ALL "
                         f"{adata.n_vars:,} genes with no class excluded (the object carried no "
                         f"mask)")


def run(adata, method, batch_key, *, label_key=None, unlabeled=(), hvg=None,
        n_latent=30, n_pcs=50, seed=0, max_epochs=None, scanvi_max_epochs=None,
        counts_layer="counts", **kwargs):
    """Run one method. Returns {"kind", "emb", "adata", "note"}.

    `emb` is the corrected coordinate space for `embed` methods and None for `graph` methods,
    whose corrected object is returned instead so the benchmark can read its connectivities.
    """
    import scanpy as sc

    have_pcs = int(np.asarray(adata.obsm["X_pca"]).shape[1])
    used_pcs = min(n_pcs, have_pcs)
    if used_pcs < n_pcs:
        # `emb[:, :50]` on a 30-component PCA returns 30 columns and raises nothing. Three of
        # the five methods read this array, so a silent shortfall changes the comparison for all
        # of them and appears nowhere.
        print(f"    NOTE: --n-pcs {n_pcs} but the object's X_pca has {have_pcs} components; "
              f"using {used_pcs}. Every method reading X_pca uses the same {used_pcs}.")

    if method == "none":
        return {"kind": "embed", "emb": np.asarray(adata.obsm["X_pca"])[:, :used_pcs],
                "adata": None,
                "note": f"the object's own X_pca, first {used_pcs} of {have_pcs} components",
                "settings": {"n_pcs": used_pcs, "pcs_available": have_pcs,
                             "source": "obsm['X_pca'], computed upstream and reused"}}

    if method == "harmony":
        import harmonypy
        # `random_state` is a real parameter of run_harmony with its own default of 0. Not
        # passing it meant --seed changed scVI, scANVI and every UMAP while leaving Harmony on a
        # fixed internal seed - so two runs at different seeds produced a Harmony embedding that
        # was identical and four others that were not, with nothing saying so.
        #
        # max_iter_harmony is 20 against harmonypy's default of 10: a deliberate deviation, so
        # that a cohort with many libraries is not scored on a correction that was still moving
        # when it stopped. It is recorded here and in docs/METHODS.md rather than left in code.
        kw = dict(max_iter_harmony=20, random_state=seed)
        h = harmonypy.run_harmony(np.asarray(adata.obsm["X_pca"])[:, :used_pcs],
                                  adata.obs, [batch_key], **kw)
        return {"kind": "embed", "emb": np.asarray(h.Z_corr).T, "adata": None,
                "note": f"Harmony on {used_pcs} PCs, max_iter_harmony=20, random_state={seed}",
                "settings": {"n_pcs": used_pcs, "max_iter_harmony": 20, "random_state": seed,
                             "theta": "harmonypy default (2.0)",
                             "sigma": "harmonypy default (0.1)"}}

    if method == "bbknn":
        import bbknn, anndata as ad
        # A LIGHT object, not adata.copy(). BBKNN reads obsm['X_pca'] and one obs column; copying
        # the input would duplicate X and both count layers - on a cohort-sized object that is
        # several gigabytes cloned to be ignored, and it is how a run dies at the last method
        # rather than the first.
        B = ad.AnnData(X=np.zeros((adata.n_obs, 1), dtype="float32"))
        B.obs_names = list(adata.obs_names.astype(str))
        B.obs[batch_key] = adata.obs[batch_key].astype(str).values
        B.obs[batch_key] = B.obs[batch_key].astype("category")
        B.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])[:, :used_pcs]
        # neighbors_within_batch is BBKNN's own default of 3. Its documentation suggests raising
        # it when there are few batches and lowering it when there are many, since the graph
        # holds n_batches x neighbors_within_batch edges per cell; at 3 a ten-library cohort
        # already gets 30. Left at the default and RECORDED, rather than tuned silently.
        nwb = int(kwargs.get("neighbors_within_batch", 3)) if kwargs else 3
        n_batches = int(adata.obs[batch_key].astype(str).nunique())
        bbknn.bbknn(B, batch_key=batch_key, n_pcs=used_pcs, neighbors_within_batch=nwb)
        return {"kind": "graph", "emb": None, "adata": B,
                "note": (f"BBKNN on {used_pcs} PCs, neighbors_within_batch={nwb} over "
                         f"{n_batches} batches ({nwb * n_batches} edges per cell). The GRAPH is "
                         f"the result; there is no corrected coordinate space, so it is scored "
                         f"on connectivities"),
                "settings": {"n_pcs": used_pcs, "neighbors_within_batch": nwb,
                             "n_batches": n_batches, "trim": "bbknn default (None)",
                             "metric": "bbknn default (euclidean, annoy)"}}

    if method in ("scvi", "scanvi"):
        import scvi
        scvi.settings.seed = seed
        sub, hvg_note = _hvg_subset(adata, hvg)
        sub = sub.copy()
        scvi.model.SCVI.setup_anndata(sub, layer=counts_layer, batch_key=batch_key)
        m = scvi.model.SCVI(sub, n_latent=n_latent)
        m.train(max_epochs=max_epochs, accelerator="auto")
        eff = int(getattr(m, "history", {}).get("elbo_train", []).shape[0]) \
            if hasattr(getattr(m, "history", {}).get("elbo_train", None), "shape") else None
        base = {"n_latent": n_latent, "n_genes": int(sub.n_vars), "seed": seed,
                "counts_layer": counts_layer, "batch_key": batch_key,
                "max_epochs_requested": max_epochs if max_epochs else "scvi-tools default",
                "max_epochs_run": eff,
                "gene_selection": hvg_note}
        if method == "scvi":
            return {"kind": "embed", "emb": np.asarray(m.get_latent_representation()),
                    "adata": None,
                    "note": (f"scVI, {n_latent} latent dimensions, trained on {hvg_note}"
                             + (f", {eff} epochs" if eff else "")),
                    "settings": base}

        # scANVI: the annotator's sentinels are the unlabelled category. That is not a
        # workaround - a cell the annotation declined to call IS unlabelled, and passing it as a
        # class of its own would ask the model to learn "unresolved" as a cell type.
        if not label_key:
            raise ValueError("scanvi needs --label-key")
        lab = sub.obs[label_key].astype(str).copy()
        UNL = "scintegrate_unlabelled"
        n_unl = int(lab.isin(list(unlabeled)).sum())
        lab[lab.isin(list(unlabeled))] = UNL
        sub.obs["_scanvi_labels"] = lab.values
        s = scvi.model.SCANVI.from_scvi_model(m, labels_key="_scanvi_labels",
                                             unlabeled_category=UNL, adata=sub)
        # scANVI is a FINE-TUNE of an already-trained scVI model, and scvi-tools' own scANVI
        # tutorial uses max_epochs=20 with n_samples_per_label=100. Left at None it inherits the
        # same length formula scVI uses, which on a cohort-sized object is several times the
        # tutorial value - a longer fit against the labels, scored afterwards on label-based
        # metrics. Exposed as its own flag, defaulting to the current behaviour so that no
        # existing comparison changes underneath anyone, and printed either way.
        sm = scanvi_max_epochs if scanvi_max_epochs is not None else max_epochs
        s.train(max_epochs=sm, n_samples_per_label=100, accelerator="auto")
        seff = int(getattr(s, "history", {}).get("elbo_train", []).shape[0]) \
            if hasattr(getattr(s, "history", {}).get("elbo_train", None), "shape") else None
        return {"kind": "embed", "emb": np.asarray(s.get_latent_representation()),
                "adata": None,
                "note": (f"scANVI fine-tuned from the scVI model, {n_latent} latent dimensions, "
                         f"{hvg_note}. {n_unl:,} cells carried an annotator sentinel and were "
                         f"passed as unlabelled"
                         + (f", {seff} fine-tuning epochs" if seff else "")),
                "settings": dict(base, label_key=label_key, n_samples_per_label=100,
                                 unlabelled_cells=n_unl,
                                 scanvi_max_epochs_requested=(
                                     sm if sm else "scvi-tools default"),
                                 scanvi_max_epochs_run=seff,
                                 USES_LABELS=("this method is TRAINED on the label column it is "
                                              "later scored against on label-based metrics"))}

    raise ValueError(f"unknown method: {method}")
