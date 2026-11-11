#!/usr/bin/env python3
"""Cells withheld from the integration are not in the PCA the methods read (GAP-128).

WHY THIS EXISTS. `--drop-labels` withholds cells from the fit, and docs/METHODS.md said they were
"not in the PCA ... and NaN in every embedding". Both were false whenever the input carried an
`X_pca`, which is every object an upstream stage embeds. The PCA was reused as delivered, fitted
with the withheld cells in it, so they still shaped the space `none`, Harmony and BBKNN were
fitted in; and the delivered object copied that `X_pca` whole, so the withheld cells kept their
coordinates in it. Measured on this file's cohort at 0.5.0: 18 of 502 cells withheld, `X_none` on
the other 484 equal to the input's PCA to the last bit, the delivered `X_pca` without one NaN row.

WHAT IT HOLDS THE TOOL TO, through `python -m scintegrate.cli`, on tests/_cohort.py's cohort,
whose `X_pca` was fitted on every cell, the withheld ones included:
  1. with cells withheld, the PCA is refit on the retained cells: the delivered `X_pca` is NaN on
     exactly the withheld rows; over the retained cells its columns are centred and orthogonal,
     which a PCA of exactly those cells is and the input's is not; it equals an independent
     `sc.pp.pca` of them over the input's `highly_variable` mask; `X_none` is its first --n-pcs
     columns; report.json and the object's provenance record the refit with its counts
  2. `--keep-input-pca` keeps the input's PCA whole, the withheld cells in its fit, and says so
  3. nothing withheld: the input's PCA is reused, and the record says so
  4. no `X_pca` in the input: the PCA is computed on the retained cells and delivered, NaN on
     the withheld rows (0.5.0 delivered no `X_pca` at all in that case)
  5. two executions give the same `X_pca` to the last bit

Needs anndata, scanpy, scipy, harmonypy, scib and threadpoolctl; prints `SKIP:` without them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import harmonypy  # noqa: F401
    import scib  # noqa: F401
    from threadpoolctl import threadpool_limits
except ImportError as e:
    print(f"SKIP: needs anndata, scanpy, scipy, harmonypy, scib and threadpoolctl ({e})")
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from _cohort import cohort  # noqa: E402

N_PCS = 15


def integrate(src, out, *extra, methods="none"):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", "integrate", "--h5ad", str(src),
                        "--out", str(out), "--batch-key", "library", "--label-key", "cell_type",
                        "--species", "homo_sapiens", "--methods", methods, "--k", "10",
                        "--n-pcs", str(N_PCS), "--scib-resolutions", "3", *map(str, extra)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def read(out):
    """What the run left: report.json and the delivered object, or empty stand-ins."""
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    p = out / "objects" / "cohort_integrated.h5ad"
    return rep, (ad.read_h5ad(p) if p.exists() else None)


def obsm(obj, key):
    return np.asarray(obj.obsm[key], dtype="float64") if obj is not None and key in obj.obsm else None


def pca_of(cells, mask):
    """An independent PCA of exactly these cells, by the recipe the run must record."""
    K = cells.copy()
    K.obsm.pop("X_pca", None)
    with threadpool_limits(limits=1):
        sc.pp.pca(K, n_comps=N_PCS, mask_var="highly_variable" if mask else None,
                  svd_solver="arpack", random_state=0)
    return np.asarray(K.obsm["X_pca"], dtype="float64")


def centred_orthogonal(P):
    """Largest |column mean| in units of the column's sd, and largest |off-diagonal correlation|."""
    sd = P.std(0)
    c = np.corrcoef(P.T)
    return float(np.max(np.abs(P.mean(0)) / sd)), float(np.max(np.abs(c - np.eye(len(c)))))


def pca_record(rep, obj):
    prov = ((obj.uns.get("scintegrate") or {}).get("provenance") or {}) if obj is not None else {}
    return rep.get("pca") or {}, prov.get("pca") or {}


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    A = cohort(td / "base.h5ad")
    withheld = np.asarray(A.obs["cell_type"].astype(str)) == "EXCLUDED"
    kept = ~withheld
    n_w, n_k = int(withheld.sum()), int(kept.sum())
    # The upstream mask, as an annotation stage leaves it: most genes, not all.
    mask_genes = np.arange(A.n_vars) % 4 != 3
    A.var["highly_variable"] = mask_genes
    src = td / "with_pca.h5ad"
    A.write_h5ad(src)
    inp = np.asarray(A.obsm["X_pca"], dtype="float64")

    # ------------------------------------------------------------------------------------- 1
    print(f"\n1. {n_w} of {A.n_obs} cells withheld: the PCA is refit on the other {n_k}")
    out1 = td / "refit"
    rc, log = integrate(src, out1, "--drop-labels", "EXCLUDED", methods="none,harmony")
    check("integrate exits 0", rc == 0, log[-2000:])
    rep, obj = read(out1)
    P, N = obsm(obj, "X_pca"), obsm(obj, "X_none")
    check("the delivered X_pca exists", P is not None, sorted(obj.obsm) if obj is not None else "no object")
    if P is not None and N is not None:
        nan = np.isnan(P).any(1)
        check(f"X_pca is NaN on exactly the {n_w} withheld cells (0.5.0: on none of them)",
              np.array_equal(nan, withheld), f"NaN rows {int(nan.sum())}, withheld {n_w}")
        Pk = P[kept]
        mean_sd, corr = centred_orthogonal(Pk)
        check("over the retained cells X_pca's columns are centred (|mean| <= 1e-4 sd): a PCA of them",
              mean_sd <= 1e-4, f"largest |mean|/sd {mean_sd:.2e}")
        check("over the retained cells X_pca's columns are orthogonal (|corr| <= 1e-4)",
              corr <= 1e-4, f"largest |corr| {corr:.2e}")
        ref = pca_of(A[kept], mask=True)
        d = float(np.max(np.abs(Pk - ref))) if Pk.shape == ref.shape else float("inf")
        check("X_pca on the retained cells equals an independent sc.pp.pca of them over the mask "
              "(within 1e-4)", d <= 1e-4, f"max |diff| {d:.3e}, shapes {Pk.shape} {ref.shape}")
        mean_in, corr_in = centred_orthogonal(inp[kept])
        check("the control: the input's PCA, fitted with the withheld cells, is not centred over "
              "the retained ones", mean_in > 1e-3, f"largest |mean|/sd {mean_in:.2e}")
        check("X_none is X_pca's first --n-pcs columns on the retained cells, bit for bit, and NaN "
              "on the withheld", np.array_equal(N[kept], Pk[:, :N_PCS]) and np.isnan(N[withheld]).all())
        check("X_none is not the input's PCA any more",
              float(np.max(np.abs(N[kept] - inp[kept][:, :N_PCS]))) > 1e-3)
    rec, prov = pca_record(rep, obj)
    check("report.json records the refit: source, cells fitted, withheld, components, genes",
          rec.get("source") == "refit" and rec.get("n_cells_fitted") == n_k
          and rec.get("n_withheld_not_fitted") == n_w and rec.get("n_comps") == N_PCS
          and rec.get("n_genes") == int(mask_genes.sum()) and "highly_variable" in str(rec.get("genes")),
          rec)
    check("the refit was held at one thread, and says so", rec.get("fit_threads") == 1, rec)
    check("the object's provenance carries the same record", prov == rec and bool(rec), prov)
    ms = (rep.get("method_settings") or {}).get("none") or {}
    note = next((m.get("note", "") for m in rep.get("methods") or [] if m.get("method") == "none"), "")
    check("none's source and note name the refit", "refit" in str(ms.get("source")) and "refit" in note,
          f"{ms.get('source')!r} / {note!r}")
    page = (out1 / "reports" / "integration.html")
    check("the page says the PCA was refit on the retained cells",
          page.exists() and "refit on the" in page.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------------------------- 2
    print("\n2. --keep-input-pca: the input's PCA, declared, the withheld cells in its fit")
    out2 = td / "keep"
    rc, log = integrate(src, out2, "--drop-labels", "EXCLUDED", "--keep-input-pca")
    check("integrate --keep-input-pca exits 0", rc == 0, log[-1500:])
    rep, obj = read(out2)
    P, N = obsm(obj, "X_pca"), obsm(obj, "X_none")
    check("X_none is the input's PCA on the retained cells, bit for bit",
          N is not None and np.array_equal(N[kept], inp[kept][:, :N_PCS]))
    check("X_pca is the input's, whole: the withheld cells keep their rows",
          P is not None and np.array_equal(P, inp))
    rec, prov = pca_record(rep, obj)
    check("the record says the input's PCA was kept as declared, with the withheld cells in its fit",
          rec.get("source") == "input" and rec.get("declared") == "--keep-input-pca"
          and rec.get("n_withheld_in_fit") == n_w and prov == rec, rec)

    # ------------------------------------------------------------------------------------- 3
    print("\n3. nothing withheld: the input's PCA is reused, and recorded")
    out3 = td / "all"
    rc, log = integrate(src, out3)
    check("integrate without --drop-labels exits 0", rc == 0, log[-1500:])
    rep, obj = read(out3)
    N = obsm(obj, "X_none")
    check("X_none is the input's PCA", N is not None and np.array_equal(N, inp[:, :N_PCS]))
    rec, _ = pca_record(rep, obj)
    check("the record says the input's PCA, no cell withheld",
          rec.get("source") == "input" and rec.get("n_withheld_not_fitted") == 0
          and rec.get("declared") is None, rec)

    # ------------------------------------------------------------------------------------- 4
    print("\n4. no X_pca in the input: computed on the retained cells, and delivered")
    B = A.copy()
    del B.obsm["X_pca"]
    del B.var["highly_variable"]
    src4 = td / "no_pca.h5ad"
    B.write_h5ad(src4)
    out4 = td / "computed"
    rc, log = integrate(src4, out4, "--drop-labels", "EXCLUDED")
    check("integrate on an object without X_pca exits 0", rc == 0, log[-1500:])
    rep, obj = read(out4)
    P = obsm(obj, "X_pca")
    check("the delivered object carries X_pca (0.5.0: none), NaN on exactly the withheld",
          P is not None and np.array_equal(np.isnan(P).any(1), withheld),
          sorted(obj.obsm) if obj is not None else "no object")
    if P is not None:
        ref = pca_of(B[kept], mask=False)
        d = float(np.max(np.abs(P[kept] - ref))) if P[kept].shape == ref.shape else float("inf")
        check("it is a PCA of the retained cells over every gene (no mask on the input)", d <= 1e-4,
              f"max |diff| {d:.3e}")
    rec, _ = pca_record(rep, obj)
    check("the record says computed, on the retained cells, over every gene",
          rec.get("source") == "computed" and rec.get("n_cells_fitted") == n_k
          and rec.get("n_genes") == A.n_vars, rec)

    # ------------------------------------------------------------------------------------- 5
    print("\n5. the refit is the same in a second execution")
    out5 = td / "again"
    rc, log = integrate(src, out5, "--drop-labels", "EXCLUDED")
    _, o1 = read(out1)
    _, o5 = read(out5)
    a1, a5 = obsm(o1, "X_pca"), obsm(o5, "X_pca")
    check("X_pca bit-identical between two executions (the methods compared do not enter it)",
          rc == 0 and a1 is not None and a5 is not None and np.array_equal(a1, a5, equal_nan=True))

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
