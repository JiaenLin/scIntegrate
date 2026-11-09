"""A small synthetic HUMAN SINGLE-CELL cohort, shared by the suites that run the real CLI on it.

Not a suite: run_all.py runs `test_*.py` only. Nothing here is biology and nothing may be quoted.
Its shape is what a human scRNA study looks like and the first cohort did not: Ensembl ENSG
identifiers in var, upper-case symbols, six libraries from six donors under two conditions, a
population found in one library only, cells an earlier stage withheld, and annotator sentinels.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

LIBRARIES = [f"L{i}" for i in range(1, 7)]
DONOR = {lib: f"D{i}" for i, lib in enumerate(LIBRARIES, 1)}             # one library per donor
CONDITION = {lib: ("unstimulated" if i <= 3 else "stimulated") for i, lib in enumerate(LIBRARIES, 1)}
CHEMISTRY = {lib: ("v3" if i % 2 else "v3.1") for i, lib in enumerate(LIBRARIES, 1)}


def cohort(path, seed=0):
    """Human single cells, synthetic. Nothing here is biology and nothing here may be quoted."""
    rng = np.random.default_rng(seed)
    types = ["T_cell", "B_cell", "Monocyte", "NK_cell"]
    rows = []
    for lib in LIBRARIES:
        for t in types:
            rows += [(lib, t)] * 18
        rows += [(lib, "EXCLUDED")] * 3            # withheld from the fit, kept in the object
        rows += [(lib, "UNRESOLVED")] * 2          # an annotator sentinel, kept everywhere
    rows += [("L1", "Platelet")] * 40              # a population found in ONE library only
    lib = np.array([r[0] for r in rows])
    lab = np.array([r[1] for r in rows])
    n, g = len(rows), 160
    mu = np.full((n, g), 0.3)
    for k, t in enumerate(types + ["Platelet", "EXCLUDED", "UNRESOLVED"]):
        mu[lab == t, k * 20:(k + 1) * 20] += 2.2
    for k, b in enumerate(LIBRARIES):                       # a library effect worth correcting
        mu[lib == b, 140 + (k % 4) * 5:140 + (k % 4) * 5 + 5] += 1.5
    X = rng.poisson(np.expm1(mu)).astype("float32")
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs_names = [f"cell{i:05d}" for i in range(n)]
    a.var_names = [f"HSG{i:04d}" for i in range(g - 6)] + ["ACTB", "GAPDH", "MT-CO1", "MT-ND1",
                                                            "RPS4X", "RPL13"]
    a.var["gene_ids"] = [f"ENSG{100000 + i:011d}" for i in range(g)]
    a.var["feature_types"] = "Gene Expression"
    a.obs["library"] = lib
    a.obs["cell_type"] = lab
    sc.pp.pca(a, n_comps=15, svd_solver="arpack", random_state=0)
    a.write_h5ad(path)
    return a


def design(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("library,donor,condition,chemistry\n")
        for lib in LIBRARIES:
            fh.write(f"{lib},{DONOR[lib]},{CONDITION[lib]},{CHEMISTRY[lib]}\n")
