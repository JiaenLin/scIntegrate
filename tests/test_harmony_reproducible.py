#!/usr/bin/env python3
"""Harmony gives back the same embedding, and the same scIB numbers, every time it runs (GAP-C54).

WHY THIS EXISTS. On one cohort, three runs at commits that did not touch harmony moved harmony's
ARI by more than a tenth, and its NMI and isolated-label F1 with it, while its silhouettes agreed
to 1e-6 and the two methods that train nothing agreed on every metric but LISI.
The variation was in what is computed FROM harmony's embedding by discrete steps, and it began in
the fit: harmony did not return the same bits twice.

THE CAUSE, measured on the synthetic object below and on a larger one (40,000 cells): sklearn's
KMeans, which harmonypy runs to place its first centroids, adds each OpenMP thread's partial sums
under a lock in the order the threads arrive (sklearn/cluster/_k_means_lloyd.pyx). With more than
one thread the centroids differ in their last bits from one execution to the next, and harmony's
rounds carry the difference into X_harmony. On one embedding's bits the downstream steps are exact:
the neighbour graph is identical built twice, and so are the Leiden sweep and its metrics (the
controls below, which pass on any commit). So the fix is at the fit: every thread pool is held at
one thread while harmonypy runs (scintegrate/methods.py).

HOW FAR THE STEPS CARRY IT depends on the data, and is why the fix cannot be downstream. On 40,000
synthetic cells, two threaded fits 1.9e-5 apart changed 4 of 640,000 kNN entries and iLISI by 1.2e-5,
and no Leiden partition. On the cohort, a difference of the same kind moved ARI by more than a tenth.

What is held, in order:
  1. THE CAUSE. Three fits through `methods.run`, two on one OpenMP pool of several threads and
     one on a smaller pool, give a bit-identical X_harmony. Fails on 69a54b2: a threaded fit's
     last bits follow the threads' arrival order (which can agree by chance) and the pool size
     (which regroups the sums every time).
  2. THE CONTROLS (pass on any commit). One embedding's neighbour graph, built twice, is
     identical; the scIB clustering sweep, NMI, ARI and isolated-label F1 on it are identical twice.
  3. THE CONSEQUENCE, through the real CLI. Two `integrate --methods none,harmony --drop-labels
     EXCLUDED` runs, one command on two pool sizes, give X_harmony bit-identical, the same NaN rows
     (exactly the withheld cells), and every scIB metric of harmony equal within 1e-6, LISI
     included (no subsample is drawn). `none` is equal in both: the control. Fails on 69a54b2.
  4. THE RECORD. The fit sits inside the limit, the limit opens after harmonypy is imported, the
     run's settings say how many threads the fit had, and the theta they name is the one used.

Synthetic throughout: nothing here is biology and nothing may be quoted. Needs harmonypy, scanpy,
scib and threadpoolctl; prints `SKIP:` without them. The thread checks need two or more cores.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:900]}" if detail and not cond
                                                      else ""))
    if not cond:
        FAILED.append(name)


def _code(path):
    """Source with comment-only lines stripped, so a check cannot pass on its own explanation."""
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.strip().startswith("#"))


# ---------------------------------------------------------------- 4. the record, read from source
print("\nthe fit is held at one thread, and the limit opens after the import")
src = _code(ROOT / "scintegrate" / "methods.py")
h = src[src.find('if method == "harmony":'):src.find('if method == "bbknn":')]
lines = h.splitlines()


def _at(text):
    return next((i for i, ln in enumerate(lines) if text in ln), None)


def _indent(ln):
    return len(ln) - len(ln.lstrip())


w, f, i = _at("with threadpool_limits(limits=1)"), _at("harmonypy.run_harmony("), _at("import harmonypy")
inside = (w is not None and f is not None and f > w
          and all(_indent(ln) > _indent(lines[w]) for ln in lines[w + 1:f + 1] if ln.strip()))
check("harmonypy.run_harmony is called inside `with threadpool_limits(limits=1)`", inside, h[:1200])
check("harmonypy is imported before the limit opens (threadpoolctl limits only loaded pools)",
      i is not None and w is not None and i < w)
check("the run's settings record the fit's threads", '"fit_threads"' in h)

try:
    import numpy as np
    import pandas as pd
    import anndata as ad
    import harmonypy  # noqa: F401  - loads sklearn's OpenMP runtime before any limit below
    import scanpy  # noqa: F401
    import scib  # noqa: F401
    from threadpoolctl import threadpool_limits
except ImportError as e:
    print(f"SKIP: needs numpy, anndata, harmonypy, scanpy, scib and threadpoolctl ({e})")
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
    sys.exit(1 if FAILED else 0)

import scintegrate.methods as ME  # noqa: E402
import scintegrate.benchmark as BM  # noqa: E402

N_CELLS = 9000          # above scanpy's 8,192: the approximate neighbour search the cohort scale uses
THREADS = min(4, os.cpu_count() or 0)      # unknown is treated as one core: the thread checks skip
RES = [round(x, 3) for x in np.linspace(0.1, 2.0, 5)]


def synthetic(n=N_CELLS, d=30, n_lib=6, seed=11):
    """Libraries with a shift, ten populations (two barely apart, two in one or two libraries),
    2% of labels swapped within a compartment. A float32 PCA, as an object carries it."""
    rng = np.random.default_rng(seed)
    latent, k = 8, 10
    comp = rng.normal(0, 6.0, size=(3, latent))
    comp_of = np.arange(k) % 3
    centres = comp[comp_of] + rng.normal(0, 2.2, size=(k, latent))
    centres[1] = centres[0] + rng.normal(0, 0.9, latent)
    w = rng.dirichlet(np.full(k, 1.5)) + 0.02
    w[[8, 9]] = 0.01
    lab = rng.choice(k, size=n, p=w / w.sum())
    lib = rng.integers(0, n_lib, size=n)
    lib[lab == 8] = 0
    lib[lab == 9] = rng.integers(1, 3, size=int((lab == 9).sum()))
    Z = centres[lab] + rng.normal(0, 1.0, size=(n, latent))
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    X = Z @ (Q[:, :latent] * np.linspace(1.5, 0.6, latent)).T
    X += rng.normal(0, 1.0, size=(n, d)) * np.linspace(0.9, 0.3, d)
    X += (rng.normal(0, 2.0, size=(n_lib, d)) * np.linspace(1.0, 0.3, d))[lib]
    X += (rng.normal(0, 0.7, size=(n_lib, k, d)) * np.linspace(1.0, 0.3, d))[lib, lab]
    seen = lab.copy()
    for i in np.where(rng.random(n) < 0.02)[0]:       # never TO a rare one: it stays isolated
        seen[i] = rng.choice(np.where((comp_of == comp_of[lab[i]]) & (np.arange(k) < 8))[0])
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    obs = pd.DataFrame({"library": [f"L{b + 1}" for b in lib],
                        "cell_type": [f"type_{t:02d}" for t in seen]},
                       index=[f"cell{i:06d}" for i in range(n)])
    return (U * S).astype("float32"), obs


pcs, obs = synthetic()
A = ad.AnnData(X=np.zeros((len(obs), 1), dtype="float32"), obs=obs)
A.obsm["X_pca"] = pcs

# ---------------------------------------------------------------- 1. the cause
# Two fits on the same pool, then one on a smaller pool. Arrival order varies from run to run but
# not on demand: two fits on one pool can agree by chance (measured: two runs of the CLI at four
# threads did, on 69a54b2). A different pool size regroups the partial sums every time, so on a
# threaded fit the third differs for certain, and on a fit held at one thread none can.
POOLS = [THREADS, THREADS, max(2, THREADS // 2)]
print(f"\n1. three harmony fits, {N_CELLS:,} synthetic cells, the OpenMP pool at {POOLS} threads")
if THREADS < 2:
    print("  SKIP: one core; neither the arrival order nor the pool size can vary here")
    fits = []
else:
    import logging
    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    fits = []
    for t in POOLS:
        with threadpool_limits(limits=t, user_api="openmp"):
            fits.append(np.asarray(ME.run(A, "harmony", "library", n_pcs=30, seed=0)["emb"]))
    diffs = {f"fit {i} ({POOLS[i]} threads)": float(np.max(np.abs(fits[i].astype("float64")
                                                                  - fits[0].astype("float64"))))
             for i in (1, 2)}
    check("X_harmony is bit-identical in all three fits, whatever the pool holds",
          all(np.array_equal(f, fits[0]) for f in fits[1:]),
          f"max |diff| from fit 0 ({POOLS[0]} threads): {diffs} - the fit's threads decide its last bits")

# ---------------------------------------------------------------- 2. the controls
print("\n2. on ONE embedding's bits, the steps scIB takes from it are exact (any commit)")
emb = fits[0] if fits else np.asarray(ME.run(A, "harmony", "library", n_pcs=30, seed=0)["emb"])


def graph():
    return BM.prepare(emb, None, obs, "library", "cell_type", kind="embed", seed=0)


def same_csr(a, b):
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices()
    b.sort_indices()
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data))


G1, G2 = graph(), graph()
check("the neighbour graph, built twice from one embedding, is identical",
      same_csr(G1.obsp["connectivities"], G2.obsp["connectivities"])
      and same_csr(G1.obsp["distances"], G2.obsp["distances"]))


def sweep(G):
    import scib.metrics as M
    with BM.scib_pandas_compat(), BM.scanpy_fast_leiden(True):
        M.cluster_optimal_resolution(G, cluster_key="scib_cluster", label_key="cell_type",
                                     resolutions=RES, verbose=False)
        return {"nmi": M.nmi(G, "scib_cluster", "cell_type"),
                "ari": M.ari(G, "scib_cluster", "cell_type"),
                "isolated_labels_f1": M.isolated_labels_f1(G, "cell_type", "library", "X_emb",
                                                           resolutions=RES, verbose=False),
                "partition": G.obs["scib_cluster"].astype(str).tolist()}


s1, s2 = sweep(G1), sweep(G2)
check("the Leiden sweep on that graph gives the same partition twice", s1["partition"] == s2["partition"])
for k in ("nmi", "ari", "isolated_labels_f1"):
    check(f"{k} is identical twice on that graph", s1[k] == s2[k], (s1[k], s2[k]))

# ---------------------------------------------------------------- 3. the consequence, real CLI
print("\n3. two `integrate --methods none,harmony` runs through the real CLI, on two pool sizes")
if THREADS < 2:
    print("  SKIP: one core")
else:
    import json
    import scipy.sparse as sp

    def cli(*args, omp):
        env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(omp)}
        r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, args)],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
        return r.returncode, r.stdout + r.stderr

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n = len(obs)
        rng = np.random.default_rng(3)
        counts = rng.poisson(1.0, size=(n, 40)).astype("float32")
        B = ad.AnnData(X=sp.csr_matrix(np.log1p(counts)), obs=obs.copy())
        B.layers["counts"] = sp.csr_matrix(counts)
        B.var_names = [f"HSG{i:04d}" for i in range(40)]
        B.var["gene_ids"] = [f"ENSG{100000 + i:011d}" for i in range(40)]
        lab = B.obs["cell_type"].astype(str).to_numpy().copy()
        withheld = rng.random(n) < 0.01
        lab[withheld] = "EXCLUDED"             # withheld from the fit, NaN in every embedding
        B.obs["cell_type"] = lab
        B.obsm["X_pca"] = pcs
        src = td / "synthetic.h5ad"
        B.write_h5ad(src)
        runs = []
        # One command, two allocations, as two jobs on differently sized nodes would see it.
        for i, omp in ((1, POOLS[0]), (2, POOLS[2])):
            out = td / f"run{i}"
            rc, log = cli("integrate", "--h5ad", src, "--out", out, "--batch-key", "library",
                          "--label-key", "cell_type", "--species", "homo_sapiens",
                          "--methods", "none,harmony", "--drop-labels", "EXCLUDED",
                          "--n-pcs", "30", "--k", "15", "--scib-resolutions", "3",
                          "--n-cores", str(THREADS), omp=omp)
            check(f"run {i} (OMP_NUM_THREADS={omp}) exits 0", rc == 0, log[-2000:])
            runs.append(out)
        if all((r / "report.json").exists() for r in runs):
            rep = [json.loads((r / "report.json").read_text()) for r in runs]
            obj = [ad.read_h5ad(r / "objects" / "cohort_integrated.h5ad") for r in runs]
            X1, X2 = (np.asarray(o.obsm["X_harmony"]) for o in obj)
            nan1, nan2 = np.isnan(X1).any(1), np.isnan(X2).any(1)
            check("the same NaN rows in both runs, and they are exactly the withheld cells",
                  np.array_equal(nan1, nan2) and np.array_equal(nan1, withheld),
                  (int(nan1.sum()), int(nan2.sum()), int(withheld.sum())))
            check("X_harmony is bit-identical between the two runs",
                  np.array_equal(X1[~nan1], X2[~nan2]),
                  f"max |diff| {float(np.nanmax(np.abs(X1.astype('float64') - X2.astype('float64')))):.3e}")
            m = [{r["method"]: r["metrics"] for r in p["methods"]} for p in rep]
            for meth in ("none", "harmony"):
                a, b = m[0].get(meth) or {}, m[1].get(meth) or {}
                moved = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
                         if not (a.get(k) is None and b.get(k) is None)
                         and (a.get(k) is None or b.get(k) is None or abs(a[k] - b[k]) > 1e-6)}
                tag = "(the control)" if meth == "none" else ""
                check(f"{meth}: every scIB metric within 1e-6 between the two runs {tag}".rstrip(),
                      bool(a) and not moved, moved)
            fs = [(p.get("method_settings") or {}).get("harmony") or {} for p in rep]
            check("both runs record one thread for harmony's fit",
                  all(f.get("fit_threads") == 1 for f in fs), [f.get("fit_threads") for f in fs])
            # The theta the record names is the theta the fit used: harmonypy's own default, read
            # here from a fit, not remembered. It once said 2.0 (R harmony's) while harmonypy fitted 1.
            with threadpool_limits(limits=1):
                hh = harmonypy.run_harmony(pcs[:2000], obs.iloc[:2000], ["library"],
                                           max_iter_harmony=1, random_state=0)
            used = np.unique(np.asarray(hh.theta, dtype="float64"))
            check("the recorded theta is the theta harmonypy fitted with",
                  used.size == 1 and all(str(f.get("theta", "")).split(" ")[0] == f"{used[0]:g}"
                                         for f in fs), ([f.get("theta") for f in fs], used.tolist()))

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
