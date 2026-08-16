"""scintegrate assess — the whole tool.

    scintegrate assess --h5ad joint.h5ad --out DIR \
        --batch-key sample --label-key scanno_path_r1p0 --methods none,harmony,bbknn
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REFUSE = 2


def _assess(a):
    try:
        import anndata as ad, scanpy as sc, numpy as np
    except ImportError:
        print("scintegrate: needs anndata + scanpy.  pip install -e '.[run]'", file=sys.stderr)
        return 1
    from . import __version__
    from .methods import available, run, METHODS
    from .metrics import assess
    from .figures import panel, highlight, _umap
    from .report import write

    A = ad.read_h5ad(a.h5ad)
    for k in (a.batch_key, a.label_key):
        if k not in A.obs:
            print(f"scintegrate: REFUSE - obs has no column {k!r}", file=sys.stderr)
            return REFUSE
    if A.X is None:
        print("scintegrate: REFUSE - the object has no expression matrix", file=sys.stderr)
        return REFUSE

    want = [m.strip() for m in a.methods.split(",") if m.strip()]
    ok, missing = available(want)
    if "none" not in ok:
        ok = ["none"] + ok          # the baseline is not optional: see methods.py
    print(f"{A.n_obs:,} cells · {A.obs[a.batch_key].nunique()} batches · methods: {', '.join(ok)}")
    for k, v in missing.items():
        print(f"  NOT compared - {k}: {v}")

    if "X_pca" not in A.obsm:
        print("  computing PCA (no X_pca in the object)")
        sc.pp.pca(A, n_comps=a.n_pcs, svd_solver="arpack", random_state=a.seed)

    embs, rows = {}, []
    base = None
    for m in ok:
        print(f"  {m} ...", flush=True)
        e = run(A, m, a.batch_key, n_pcs=a.n_pcs, seed=a.seed)
        embs[m] = e
        if base is None:
            base = e
        rows.append({"method": m, **assess(base, e, A.obs[a.batch_key].astype(str).values,
                                           A.obs[a.label_key].astype(str).values, k=a.k)})

    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    print("  drawing (every method at one scale)")
    views = {m: _umap(e, seed=a.seed) for m, e in embs.items()}
    import matplotlib.pyplot as plt
    lab = A.obs[a.label_key].astype(str).values
    l1 = np.array([x.split("/")[0] for x in lab])
    figs = []
    cats = sorted(set(l1))
    cols = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}
    figs.append(("coloured by cell type",
                 panel(views, l1, cats, cols, "Cell type", out / "figures" / "F1_by_label.png"),
                 "If a method has moved cells away from their own kind, it shows here first."))
    bat = A.obs[a.batch_key].astype(str).values
    bcats = sorted(set(bat))
    bcols = {c: plt.cm.tab20(i % 20) for i, c in enumerate(bcats)}
    figs.append(("coloured by batch",
                 panel(views, bat, bcats, bcols, f"Batch ({a.batch_key})",
                       out / "figures" / "F2_by_batch.png"),
                 "The question the stage exists for: is the structure cell type, or library?"))
    big = max(bcats, key=lambda c: (bat == c).sum())
    figs.append((f"one batch highlighted: {big}",
                 highlight(views, bat == big, big, out / "figures" / "F3_highlight.png"),
                 "Aligned with its counterparts, or dispersed through everything? "
                 "This is the reading no metric substitutes for."))

    meta = {"version": __version__, "input": str(a.h5ad), "batch_key": a.batch_key,
            "label_key": a.label_key, "seed": a.seed, "n_cells": int(A.n_obs),
            "n_batches": int(A.obs[a.batch_key].nunique())}
    p, payload = write(out, rows, figs, meta, absent=missing)
    print("")
    print(f"wrote {p}")
    print(f"      {out}/report.json")
    print("")
    print("This tool does not choose a method. Read the figures, then decide.")
    return 0


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="scintegrate", description=__doc__)
    ap.add_argument("--version", action="version", version=f"scintegrate {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    s = sub.add_parser("assess", help="is integration needed? compare methods including none")
    s.add_argument("--h5ad", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--batch-key", default="sample")
    s.add_argument("--label-key", required=True,
                   help="the annotation, used for the cell-type panel and label coherence")
    s.add_argument("--methods", default="none,harmony,bbknn",
                   help="`none` is always included: the stage asks whether integration is NEEDED")
    s.add_argument("--k", type=int, default=30)
    s.add_argument("--n-pcs", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_assess)
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 0
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
