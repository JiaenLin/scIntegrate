"""scintegrate — is integration needed, which method, and one object carrying the answer.

    scintegrate doctor                        # what is installed, and what each absence costs
    scintegrate assess    --h5ad J.h5ad --out DIR --label-key L --design d.csv
    scintegrate integrate --h5ad J.h5ad --out DIR --label-key L --design d.csv \
                          --methods none,harmony,bbknn,scvi,scanvi
    scintegrate report    --out DIR           # rebuild the document from what is on disk

`assess` answers "is integration needed" WITHOUT training anything, because that question should
be settled before spending a GPU on methods you may not want. `integrate` runs the methods, scores
them with scIB, names a default embedding and writes the single deliverable.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REFUSE = 2


# --------------------------------------------------------------------------------- shared load

def _load(a):
    """Open the object, resolve the keys, join the design. Refusals are printed and re-raised."""
    from . import inputs
    sent = tuple(s for s in (a.label_sentinel or []) if s)
    D = inputs.read(a.h5ad, a.batch_key, a.label_key, l1_key=a.l1_key,
                    sentinels=sent, coarse_from_path=a.coarse_from_path)
    A = D["adata"]
    print(f"{A.n_obs:,} cells x {A.n_vars:,} genes")
    print(f"  batch  {a.batch_key!r}: {len(set(D['batch']))} level(s)")
    print(f"  label  {a.label_key!r}: {len(set(D['label']))} value(s)")
    print(f"  coarse : {D['coarse_note']}")
    print(f"  counts : {D['counts_note']}")
    if D["sentinels"]:
        tot = sum(D["sentinels"].values())
        print(f"  SENTINELS - not cell types, excluded from LABEL metrics only, kept in every "
              f"embedding: {tot:,} cells")
        for k, v in sorted(D["sentinels"].items(), key=lambda kv: -kv[1]):
            print(f"      {k:24s} {v:>8,}")
    if D["hvg"] is not None:
        print(f"  hvg    : {int(D['hvg'].sum()):,} flagged in var, reused verbatim")

    D["factors"], D["covariates"], D["design_note"] = {}, {}, "no --design given"
    if a.design:
        table, key, cols = inputs.read_design(a.design, set(D["batch"]), a.design_sample_col)
        want = list(a.bio_factor or cols)
        bad = [f for f in want if f not in cols]
        if bad:
            raise inputs.Refuse(f"--bio-factor {bad} not in {a.design}: it offers {cols}")
        import numpy as np
        # EVERY design column is joined onto the cells and travels in the deliverable - a
        # technical covariate is exactly what a later stage needs in order to model rather than
        # remove it, and a covariate joined on afterwards by position is one nobody can check.
        # Only the DECLARED bio factors drive the assessment and the constraint, because those
        # two are statements about what the study is for.
        for f in cols:
            D["covariates"][f] = np.array([table[s][f] for s in D["batch"]])
        D["factors"] = {f: D["covariates"][f] for f in want}
        other = [f for f in cols if f not in want]
        D["design_note"] = (
            f"{a.design} keyed on {key!r}; biological factor(s) "
            + ", ".join(f"{f} ({len(set(D['factors'][f]))} levels)" for f in want)
            + (f"; carried but not declared biological: {', '.join(other)}" if other else ""))
        print(f"  design : {D['design_note']}")
    return D


def _views(results, seed=0):
    """A 2-D view per method, computed once and shared by every figure."""
    from .figures import _umap
    out = {}
    for r in results:
        if r.get("umap") is not None:
            out[r["method"]] = r["umap"]
        elif r.get("emb") is not None:
            out[r["method"]] = _umap(r["emb"], seed=seed)
    return out


def _csv(path, header, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _sentinel_tables(out, D, a):
    """What the label metrics did not see, per design arm.

    Rule one, question three, and the reason it is a TABLE rather than a total: a sentinel class
    inherited from an upstream filter is usually not evenly distributed, and an exclusion that
    removes a tenth of one arm and a fiftieth of another has turned a technical property into an
    apparent biological difference. Reported per arm whether or not it looks even.
    """
    import numpy as np
    rows = []
    lab = D["label"]
    for s in sorted(D["sentinels"]):
        m = lab == s
        rows.append(["ALL", s, int(m.sum()), len(lab), f"{100*m.mean():.3f}"])
        for f, v in D["factors"].items():
            for lvl in sorted(set(v)):
                k = np.asarray(v) == lvl
                rows.append([f"{f}={lvl}", s, int((m & k).sum()), int(k.sum()),
                             f"{100*(m & k).sum()/max(1, k.sum()):.3f}"])
        for b in sorted(set(D["batch"])):
            k = D["batch"] == b
            rows.append([f"{a.batch_key}={b}", s, int((m & k).sum()), int(k.sum()),
                         f"{100*(m & k).sum()/max(1, k.sum()):.3f}"])
    return _csv(out / "tables" / "label_sentinels_by_arm.csv",
                ["arm", "sentinel", "n_sentinel", "n_in_arm", "pct_of_arm"], rows)


def _assessment_tables(out, rows, dom, a):
    tr = []
    for r in rows:
        if not r.get("measured"):
            tr.append([r["label"], r["n"], "not measured", "", "", "", r.get("why", "")])
            continue
        b = r["batch"]
        tr.append([r["label"], r["n"], "measured", f"{b['foreign']:.4f}",
                   f"{b['expected']:.4f}", f"{b['ratio']:.4f}" if b["ratio"] else "", ""])
    _csv(out / "tables" / "celltype_batch_mixing.csv",
         ["label", "n", "status", "foreign_mean", "chance", "ratio_to_chance", "why_not"], tr)

    fr = []
    for r in rows:
        for f, v in (r.get("factors") or {}).items():
            if v.get("ratio") is None:
                fr.append([r["label"], r["n"], f, "", "", "", v.get("why", "")])
            else:
                fr.append([r["label"], r["n"], f, f"{v['foreign']:.4f}", f"{v['expected']:.4f}",
                           f"{v['ratio']:.4f}", ""])
    if fr:
        _csv(out / "tables" / "celltype_factor_mixing.csv",
             ["label", "n", "factor", "foreign_mean", "chance", "ratio_to_chance", "why_not"], fr)

    _csv(out / "tables" / "celltype_batch_dominance.csv",
         ["label", "n", "n_batches", "top_batch", "top_share"],
         [[d["label"], d["n"], d["n_batches"], d["top_batch"], f"{d['top_share']:.4f}"]
          for d in dom])


# ------------------------------------------------------------------------------------- assess

def _assess(a):
    from . import inputs, assess as AS
    try:
        D = _load(a)
    except inputs.Refuse as e:
        print(f"scintegrate: REFUSE - {e}", file=sys.stderr)
        return REFUSE
    A = D["adata"]
    out = Path(a.out)
    if "X_pca" not in A.obsm:
        print("  computing PCA (the object carries no X_pca)")
        import scanpy as sc
        sc.pp.pca(A, n_comps=a.n_pcs, svd_solver="arpack", random_state=a.seed)
    emb = A.obsm["X_pca"][:, :a.n_pcs]

    print("\nmeasuring, per cell type, on the UNCORRECTED embedding")
    rows = AS.per_celltype(emb, D["label"], D["batch"], D["factors"], k=a.k,
                           min_per_k=a.min_per_k, is_real=D["is_real"])
    dom = AS.dominance(D["label"], D["batch"], is_real=D["is_real"])
    summ = AS.summarise(rows, indicates_below=a.indicates_below)

    width = max((len(r["label"]) for r in rows), default=10)
    fnames = list(D["factors"])
    print(f"  {'cell type':<{width}}  {'n':>8}  {'batch':>7}  "
          + "  ".join(f"{f:>7}" for f in fnames))
    for r in rows:
        if not r.get("measured"):
            print(f"  {r['label']:<{width}}  {r['n']:>8,}  {'--':>7}   (below "
                  f"{a.min_per_k}x k)")
            continue
        cells = [f"{r['batch']['ratio']:.3f}" if r["batch"]["ratio"] else "-"]
        for f in fnames:
            v = (r.get("factors") or {}).get(f, {})
            cells.append(f"{v['ratio']:.3f}" if v.get("ratio") else "-")
        print(f"  {r['label']:<{width}}  {r['n']:>8,}  " + "  ".join(f"{c:>7}" for c in cells))
    print(f"\n  threshold: a ratio below {a.indicates_below} is called batch structure")
    print("  " + summ["indication"])

    _assessment_tables(out, rows, dom, a)
    if D["sentinels"]:
        _sentinel_tables(out, D, a)

    figs = _draw(out, {"none": _views([{"method": "none", "emb": emb}], a.seed)["none"]},
                 D, a, tag="uncorrected")
    figs += _draw_assessment(out, rows, fnames, a)

    payload = {"command": "assess", "n_cells": int(A.n_obs), "batch_key": a.batch_key,
               "label_key": a.label_key, "k": a.k, "n_pcs": a.n_pcs, "seed": a.seed,
               "design": D["design_note"], "coarse": D["coarse_note"],
               "sentinels": D["sentinels"], "celltypes": _plainrows(rows),
               "dominance": dom, "summary": summ,
               "figures": [{"name": n, "path": str(Path(p).name), "caption": c}
                           for n, p, c in figs]}
    _write_json(out, payload)
    from .report import write_assess
    p = write_assess(out, payload, figs)
    print(f"\nwrote {p}")
    print(f"      {out}/report.json")
    print(f"      {out}/tables/  {len(list((out / 'tables').glob('*.csv')))} table(s)")
    print("\nNothing was integrated. This step measures the question; the decision is yours,")
    print("and the figures are the half of the evidence the table cannot carry.")
    return 0


def _plainrows(rows):
    return [{k: v for k, v in r.items() if k != "_per_cell"} for r in rows]


def _write_json(out, payload):
    from . import __version__
    from datetime import datetime, timezone
    out.mkdir(parents=True, exist_ok=True)
    payload = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
               "version": __version__, **payload}
    (out / "report.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------------- integrate

def _integrate(a):
    from . import inputs, assess as AS, benchmark as BM, methods as ME, emit
    import numpy as np
    try:
        D = _load(a)
    except inputs.Refuse as e:
        print(f"scintegrate: REFUSE - {e}", file=sys.stderr)
        return REFUSE
    A = D["adata"]
    out = Path(a.out)

    want = [m.strip() for m in a.methods.split(",") if m.strip()]
    ok, missing = ME.available(want)
    if "none" not in ok:
        ok = ["none"] + ok
    else:
        ok = ["none"] + [m for m in ok if m != "none"]
    for m in ok:
        if m in ME.NEEDS_COUNTS and not D["counts_ok"]:
            print(f"scintegrate: REFUSE - {m} models raw counts and {D['counts_note']}",
                  file=sys.stderr)
            return REFUSE
    print(f"\nmethods: {', '.join(ok)}")
    for k, v in missing.items():
        print(f"  NOT compared - {k}: {v}")

    if "X_pca" not in A.obsm:
        print("  computing PCA (the object carries no X_pca)")
        import scanpy as sc
        sc.pp.pca(A, n_comps=a.n_pcs, svd_solver="arpack", random_state=a.seed)

    sent = tuple(D["sentinels"])
    results = []
    for m in ok:
        print(f"\n=== {m} ===", flush=True)
        r = ME.run(A, m, a.batch_key, label_key=a.label_key, unlabeled=sent, hvg=D["hvg"],
                   n_latent=a.n_latent, n_pcs=a.n_pcs, seed=a.seed, max_epochs=a.max_epochs)
        r["method"] = m
        print(f"    {r['note']}")
        results.append(r)

    print("\ndrawing a 2-D view per method", flush=True)
    import scanpy as sc
    from .figures import _umap
    for r in results:
        if r["kind"] == "graph":
            sc.tl.umap(r["adata"], random_state=a.seed)
            r["umap"] = np.asarray(r["adata"].obsm["X_umap"])
        else:
            r["umap"] = _umap(r["emb"], seed=a.seed)

    # ---- the tool's own kNN metrics, which do not need scIB and are never absent
    from .metrics import assess as knn_assess
    base = results[0]["emb"]
    for r in results:
        e = r["emb"] if r["emb"] is not None else r["umap"]
        r["knn"] = knn_assess(base, e, D["batch"], D["label"], k=a.k)

    # ---- scIB
    print("\nscIB benchmark", flush=True)
    real = D["is_real"]
    n_drop = int((~real).sum())
    if n_drop:
        print(f"  computed on the {int(real.sum()):,} cells carrying a real label; "
              f"{n_drop:,} sentinel cells are excluded from the METRICS and remain in every "
              f"embedding and in the deliverable")
    obs_sub = A.obs.loc[real, [a.batch_key, a.label_key]]
    A_pre = None
    for r in results:
        print(f"  {r['method']} ...", flush=True)
        e = r["emb"][real] if r["emb"] is not None else None
        g = r["adata"][real].copy() if r["adata"] is not None else None
        Ai = BM.prepare(e, g, obs_sub, a.batch_key, a.label_key, kind=r["kind"], seed=a.seed)
        if A_pre is None:
            A_pre = Ai                       # `none` is first, and is the pre object for PCR
        r["metrics"] = BM.score(Ai, A_pre, a.batch_key, a.label_key, kind=r["kind"],
                                n_cores=a.n_cores, lisi_subsample=a.lisi_subsample)
        r["aggregate"] = BM.aggregate(r["metrics"], w_bio=a.w_bio)
        ag = r["aggregate"]
        print("      bio {} · batch {} · total {}".format(
            f"{ag['bio']:.4f}" if ag["bio"] is not None else "-",
            f"{ag['batch']:.4f}" if ag["batch"] is not None else "-",
            f"{ag['total']:.4f}" if ag["total"] is not None else "-"))
        for k, why in ag["absent"].items():
            print(f"      absent: {k} - {why}")

    chosen = BM.choose_default(results)
    print("")
    if chosen["default"]:
        print(f"  DEFAULT EMBEDDING: X_{chosen['default']}   (scIB total "
              f"{chosen['total']:.4f}, margin {chosen['margin']} over {chosen['runner_up']})")
        if not chosen["comparable"]:
            print("  NOTE: the totals do not all rest on the same number of metrics - see the "
                  "report before comparing them")
    else:
        print(f"  NO DEFAULT CHOSEN - {chosen['reason']}")

    # ---- the assessment, on the baseline, so the deliverable carries the question too
    print("\nassessment, on the uncorrected baseline", flush=True)
    arows = AS.per_celltype(base, D["label"], D["batch"], D["factors"], k=a.k,
                            min_per_k=a.min_per_k, is_real=real)
    dom = AS.dominance(D["label"], D["batch"], is_real=real)
    summ = AS.summarise(arows, indicates_below=a.indicates_below)
    print("  " + summ["indication"])

    constraint = _constraint(D, a, chosen)

    # ---- figures
    views = _views(results, a.seed)
    figs = _draw(out, views, D, a, tag="methods")
    figs += _draw_assessment(out, arows, list(D["factors"]), a)

    # ---- tables
    _assessment_tables(out, arows, dom, a)
    if D["sentinels"]:
        _sentinel_tables(out, D, a)
    allm = sorted({k for r in results for k in r["metrics"]})
    _csv(out / "tables" / "scib_metrics.csv",
         ["method", "kind"] + allm,
         [[r["method"], r["kind"]] + [
             ("" if r["metrics"].get(k, {}).get("value") is None
              else f"{r['metrics'][k]['value']:.6f}") for k in allm] for r in results])
    _csv(out / "tables" / "scib_absent.csv", ["method", "metric", "why"],
         [[r["method"], k, why] for r in results for k, why in r["aggregate"]["absent"].items()])
    _csv(out / "tables" / "scib_aggregate.csv",
         ["method", "kind", "bio", "batch", "total", "w_bio", "n_bio", "of_bio",
          "n_batch", "of_batch", "is_default"],
         [[r["method"], r["kind"],
           "" if r["aggregate"]["bio"] is None else f"{r['aggregate']['bio']:.6f}",
           "" if r["aggregate"]["batch"] is None else f"{r['aggregate']['batch']:.6f}",
           "" if r["aggregate"]["total"] is None else f"{r['aggregate']['total']:.6f}",
           r["aggregate"]["w_bio"], r["aggregate"]["n_bio"], r["aggregate"]["of_bio"],
           r["aggregate"]["n_batch"], r["aggregate"]["of_batch"],
           "YES" if r["method"] == chosen["default"] else ""] for r in results])
    _csv(out / "tables" / "knn_metrics.csv",
         ["method", "kind", "foreign_mean", "chance", "ratio_to_chance", "knn_retained",
          "label_coherence"],
         [[r["method"], r["kind"], f"{r['knn']['foreign_mean']:.6f}",
           f"{r['knn']['foreign_expected']:.6f}", f"{r['knn']['ratio_to_chance']:.6f}",
           "" if r["method"] == "none" else f"{r['knn']['knn_retained_mean']:.6f}",
           f"{r['knn']['label_coherence_mean']:.6f}"] for r in results])

    # ---- the object
    print("\nwriting the deliverable", flush=True)
    obs_keep = [a.batch_key, a.label_key] + ([a.l1_key] if a.l1_key else [])
    for f, v in D["covariates"].items():
        if f not in A.obs:
            A.obs[f] = v
        obs_keep.append(f)
    prov = {"tool": "scintegrate", "input": str(a.h5ad), "batch_key": a.batch_key,
            "label_key": a.label_key, "l1_key": a.l1_key or "", "seed": a.seed,
            "k": a.k, "n_pcs": a.n_pcs, "n_latent": a.n_latent, "w_bio": a.w_bio,
            "methods_compared": ok, "methods_absent": missing,
            "design": D["design_note"], "sentinels": D["sentinels"],
            "scib_metrics_computed_on": int(real.sum()),
            "scib_metrics_excluded_sentinels": n_drop}
    bench = {"aggregate": {r["method"]: r["aggregate"] for r in results},
             "metrics": {r["method"]: {k: v["value"] for k, v in r["metrics"].items()}
                         for r in results},
             "chosen": chosen}
    obj = emit.build(A, results, obs_keep, chosen=chosen["default"], benchmark=bench,
                     assessment={"summary": summ, "celltypes": _plainrows(arows)},
                     constraint=constraint, provenance=prov)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    op = out / "objects" / a.object_name
    emit.write_h5ad(obj, op)
    print(f"  {op}  ({op.stat().st_size / 1e9:.2f} GB)")
    print(f"  obs: {', '.join(obj.obs.columns)}")
    print(f"  layers: {', '.join(obj.layers)}")
    print(f"  obsm: {', '.join(obj.obsm)}")

    payload = {"command": "integrate", "n_cells": int(A.n_obs), "n_genes": int(A.n_vars),
               "batch_key": a.batch_key, "label_key": a.label_key, "k": a.k, "seed": a.seed,
               "n_pcs": a.n_pcs, "n_latent": a.n_latent, "w_bio": a.w_bio,
               "design": D["design_note"], "coarse": D["coarse_note"],
               "sentinels": D["sentinels"], "counts": D["counts_note"],
               "methods": [{"method": r["method"], "kind": r["kind"], "note": r["note"],
                            "metrics": {k: v["value"] for k, v in r["metrics"].items()},
                            "absent": r["aggregate"]["absent"],
                            "aggregate": {k: v for k, v in r["aggregate"].items()
                                          if k != "absent"},
                            "knn": {k: v for k, v in r["knn"].items() if k != "_per_cell"}}
                           for r in results],
               "not_compared": missing, "chosen": chosen, "constraint_on_use": constraint,
               "celltypes": _plainrows(arows), "dominance": dom, "summary": summ,
               "object": str(op), "metric_meaning": BM.MEANING,
               "figures": [{"name": n, "path": str(Path(p).name), "caption": c}
                           for n, p, c in figs]}
    _write_json(out, payload)
    from .report import write_integrate
    p = write_integrate(out, payload, figs)
    print(f"\nwrote {p}")
    print(f"      {out}/report.json")
    from .readme import write_readme
    print(f"      {write_readme(out, payload)}")
    print("")
    print("A default embedding is named above on what scIB measures: cell-type conservation")
    print("against batch mixing. Read the figures before taking it, and read the constraint")
    print("on use in the report before any claim crosses a declared factor.")
    return 0


def _constraint(D, a, chosen):
    """What the chosen embedding may and may not carry. Computed, not boilerplate.

    A factor that varies only BETWEEN batches cannot be separated from the batch by any
    correction on the batch key. That is a property of the design, present before any method
    runs, and it does not stop a default embedding being named - it decides what the embedding
    is allowed to be used for afterwards.
    """
    from .metrics import confounding
    if not D["factors"]:
        return ("No biological factor was declared (--design / --bio-factor), so nothing here "
                "constrains use. Declaring the factors the study is about is what lets this "
                "section say something.")
    conf = confounding(D["adata"].obs.assign(**{f: v for f, v in D["factors"].items()}),
                       a.batch_key, list(D["factors"]))
    nested = [f for f, c in conf.items() if c["status"] in ("nested", "aliased")]
    sep = [f for f, c in conf.items() if c["status"] == "separable"]
    if not nested:
        return (f"Every declared factor ({', '.join(sep)}) varies WITHIN at least one "
                f"{a.batch_key}, so it is separable from the batch key and a correction on that "
                f"key does not remove it by construction.")
    return (
        f"{', '.join(nested)} vary only BETWEEN levels of {a.batch_key}, never within one. That "
        f"is the ordinary structure of a design with one library per animal, and it is not a "
        f"defect - but it means a correction on {a.batch_key} necessarily reduces the contrast "
        f"in {', '.join(nested)} along with the batch effect, and no measurement taken after the "
        f"correction can separate them again.\n\n"
        f"So: the chosen embedding"
        + (f" (X_{chosen['default']})" if chosen.get("default") else "")
        + " may carry visualisation, clustering and cell-type identification. It must NOT carry "
          f"a composition or abundance claim across {', '.join(nested)} - for that, use the "
          f"uncorrected X_pca and say so, or a test that models the factor rather than removing "
          f"it. Differential expression on per-sample counts is unaffected: it reads "
          f"layers['counts'], which no correction here touches."
        + (f"  Separable, and therefore unconstrained: {', '.join(sep)}." if sep else ""))


# -------------------------------------------------------------------------------------- draw

def _draw(out, views, D, a, tag=""):
    """Every method at one scale: by cell type, by batch, by each declared factor, and one
    batch highlighted. The standing instruction this serves is that no integration decision is
    presented on metrics alone."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib absent - every figure is a NAMED ABSENCE and the report is still "
              "written")
        return []
    import numpy as np
    from .figures import panel, highlight
    fd = out / "figures"
    figs = []

    def cols(cats):
        return {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}

    co = D["coarse"]
    cats = sorted(set(co))
    figs.append(("coloured by cell type",
                 panel(views, co, cats, cols(cats), "Cell type", fd / f"F1_celltype_{tag}.png"),
                 f"Colour is {D['coarse_note']}. If a method has moved cells away from their own "
                 f"kind, it shows here before it shows in any table."))
    bat = D["batch"]
    bcats = sorted(set(bat))
    figs.append(("coloured by batch",
                 panel(views, bat, bcats, cols(bcats), f"Batch ({a.batch_key})",
                       fd / f"F2_batch_{tag}.png"),
                 "The question the stage exists for: is the structure cell type, or library?"))
    for i, (f, v) in enumerate(D["factors"].items(), start=1):
        fc = sorted(set(v))
        figs.append((f"coloured by {f}",
                     panel(views, np.asarray(v), fc, cols(fc), f"Group ({f})",
                           fd / f"F3_{i}_group_{f}_{tag}.png"),
                     f"The declared biological factor. If the correction has flattened this, it "
                     f"has removed what the study is about - and this figure is where that is "
                     f"visible, not the mixing column."))
    big = max(bcats, key=lambda c: (bat == c).sum())
    figs.append((f"one batch highlighted: {big}",
                 highlight(views, bat == big, big, fd / f"F4_highlight_{tag}.png"),
                 "Aligned with its counterparts, or dispersed through everything? A number can "
                 "say a population was mixed; only this distinguishes the two."))
    return figs


def _draw_assessment(out, rows, fnames, a):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return []
    from .figures import mixing_bars
    p = mixing_bars(rows, fnames, a.indicates_below,
                    out / "figures" / "F5_celltype_mixing.png")
    if p is None:
        return []
    return [("mixing within each cell type",
             p, f"Per cell type, how well the libraries mix inside that population, against "
                f"chance for its own composition. 1.0 is fully mixed. Bars for a declared factor "
                f"sit beside it: where BOTH are low the library structure and the biology are "
                f"the same structure. The dashed line is the declared threshold "
                f"({a.indicates_below}).")]


# ------------------------------------------------------------------------------------ report

def _report(a):
    out = Path(a.out)
    p = out / "report.json"
    if not p.exists():
        print(f"scintegrate: REFUSE - no {p}. Run assess or integrate first.", file=sys.stderr)
        return REFUSE
    payload = json.loads(p.read_text())
    figs = [(f["name"], out / "figures" / f["path"], f["caption"])
            for f in payload.get("figures", [])]
    from .report import write_assess, write_integrate
    w = write_integrate if payload.get("command") == "integrate" else write_assess
    print(f"wrote {w(out, payload, figs)}")
    if payload.get("command") == "integrate":
        from .readme import write_readme
        print(f"      {write_readme(out, payload)}")
    return 0


# --------------------------------------------------------------------------------------- main

def _common(s):
    s.add_argument("--h5ad", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--batch-key", default="sample")
    s.add_argument("--label-key", required=True,
                   help="the annotation column scored and coloured by")
    s.add_argument("--l1-key", default=None,
                   help="a MEASURED coarse-level column, if the annotation ships one. Used for "
                        "the cell-type figure. Without it the full label is used; the path is "
                        "never truncated unless --coarse-from-path says so")
    s.add_argument("--coarse-from-path", action="store_true",
                   help="derive the coarse colouring as the first component of --label-key. This "
                        "is a TRUNCATION of one walk, not an independent annotation, and is "
                        "labelled as such wherever it appears")
    s.add_argument("--label-sentinel", action="append", default=None, metavar="VALUE",
                   help="a label that means 'no call' rather than a cell type, repeatable. "
                        "Defaults to EXCLUDED and UNRESOLVED. Sentinel cells stay in every "
                        "embedding and in the deliverable; they are excluded from LABEL metrics "
                        "only, and counted per design arm in a table")
    s.add_argument("--design", default=None, type=Path,
                   help="CSV keyed on the batch, carrying the factors the study is about")
    s.add_argument("--design-sample-col", default=None,
                   help="the column in --design holding the batch name (default: the first)")
    s.add_argument("--bio-factor", action="append", default=None, metavar="COLUMN",
                   help="which --design columns are the biology, repeatable (default: all)")
    s.add_argument("--k", type=int, default=30)
    s.add_argument("--n-pcs", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-per-k", type=int, default=3, metavar="M",
                   help="a cell type is measured only if it has at least M x k cells; below that "
                        "the neighbourhood is most of the population and the ratio approaches "
                        "1.0 from smallness rather than from mixing (default 3)")
    s.add_argument("--indicates-below", type=float, default=0.80, metavar="R",
                   help="a within-type mixing ratio below R x chance is called batch structure. "
                        "A DECLARED threshold, not a discovered one (default 0.80)")


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="scintegrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"scintegrate {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")

    d = sub.add_parser("doctor", help="what is installed, and what each absence costs you")
    d.set_defaults(fn=lambda a: __import__("scintegrate.env", fromlist=["env"]).doctor())

    s = sub.add_parser("assess", help="is integration needed? measured per cell type and group, "
                                      "without training anything")
    _common(s)
    s.set_defaults(fn=_assess)

    g = sub.add_parser("integrate", help="run the methods, score with scIB, name a default "
                                         "embedding, write one object")
    _common(g)
    g.add_argument("--methods", default="none,harmony,scvi,scanvi",
                   help="`none` is always included and always first: the stage asks whether "
                        "integration is NEEDED, and it is the baseline every cost is measured "
                        "against")
    g.add_argument("--n-latent", type=int, default=30, help="latent dimensions for scVI/scANVI")
    g.add_argument("--max-epochs", type=int, default=None,
                   help="passed to scVI/scANVI; the default lets scvi-tools scale it to n_obs")
    g.add_argument("--w-bio", type=float, default=0.6, metavar="W",
                   help="weight on biological conservation in the scIB total, the rest on batch "
                        "correction (default 0.6, scIB's convention). THIS IS THE VALUE "
                        "JUDGEMENT: a different downstream question wants a different weight")
    g.add_argument("--n-cores", type=int, default=1)
    g.add_argument("--lisi-subsample", type=int, default=None, metavar="PCT",
                   help="subsample percentage for the LISI metrics; the default uses every cell")
    g.add_argument("--object-name", default="cohort_integrated.h5ad")
    g.set_defaults(fn=_integrate)

    r = sub.add_parser("report", help="rebuild the document from report.json")
    r.add_argument("--out", required=True, type=Path)
    r.set_defaults(fn=_report)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 0
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
