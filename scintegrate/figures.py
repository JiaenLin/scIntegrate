"""The figures the decision is actually made on.

STANDING INSTRUCTION, and the reason this module exists: no integration decision is presented on
metrics alone. Every method is drawn, coloured by biological group and by library, AT THE SAME
SCALE, before anyone is asked to choose.

The scale matters more than it sounds. "Sample-dominated clusters: 0" and "group structure
removed: 69.4%" described one method accurately and gave no sense that its cells from one animal
had been scattered uniformly across the whole manifold - visible instantly in a figure. A number
can say a population was mixed; only the picture distinguishes ALIGNED WITH ITS COUNTERPARTS
from DISPERSED EVERYWHERE.
"""
from __future__ import annotations
import numpy as np

GREY = "#D9D9D9"; INK = "#1a1a1a"

#: Okabe-Ito, extended by golden-angle hue generation past the eighth entry. Distinguishable with
#: any common form of colour vision and in greyscale; tab20 is neither, and a cohort with fifteen
#: populations exhausts any hand-picked list.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def palette(labels, sentinels=()):
    """A colour per label, with annotator sentinels forced to GREY.

    A sentinel is the annotator declining to make a call, not a cell type. Giving it a hue of its
    own puts it in the legend beside real populations as though it were one, and nothing in the
    figure tells a reader otherwise. Held out of the palette entirely, so removing or adding a
    sentinel never shifts the colour of a real population.
    """
    import colorsys
    sent = {str(s) for s in (sentinels or ())}
    real = [l for l in sorted(map(str, set(labels))) if l not in sent]
    out = {}
    for i, l in enumerate(real):
        if i < len(OKABE_ITO):
            out[l] = OKABE_ITO[i]
        else:
            h = ((i - len(OKABE_ITO)) * 0.381966) % 1.0
            out[l] = "#%02x%02x%02x" % tuple(
                int(255 * c) for c in colorsys.hls_to_rgb(h, 0.52, 0.62))
    for l in map(str, set(labels)):
        if l in sent:
            out[l] = GREY
    return out


def _legend_label(lab, colours):
    return f"{lab} (not a cell type)" if colours.get(lab) == GREY else str(lab)


def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as p
    # Type 42 embeds TrueType rather than converting glyphs to paths. Matplotlib's default, type 3,
    # lands in Illustrator with text that cannot be selected or corrected, and journals reject it.
    p.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    return p


#: UMAP min_dist for every view this tool draws. 0.2 rather than scanpy's 0.5, so the panels match
#: the joint embedding the annotation ships and the two can be read against each other. It is a
#: LAYOUT parameter: no metric, count or label depends on it. Recorded in the provenance.
MIN_DIST = 0.2


#: Neighbourhood size for the layout. scanpy's own default, stated rather than inherited: it is
#: the parameter a UMAP's appearance is most sensitive to after min_dist, and a figure whose
#: layout parameters are implicit cannot be reproduced from the report.
N_NEIGHBORS = 15


def _umap(emb, seed=0, min_dist=MIN_DIST, n_neighbors=N_NEIGHBORS):
    """A 2-D view, computed by scanpy exactly as a standard workflow would.

    This is `sc.pp.neighbors` then `sc.tl.umap` on the method's own coordinates - no custom
    layout, no hand-rolled projection. The only departure from scanpy's defaults is min_dist,
    which is 0.2 rather than 0.5 so these panels match the embedding the annotation ships and
    the two can be read against each other; every parameter is recorded in the report.

    The AnnData built here holds the embedding AS ITS X and the graph is built on `use_rep="X"`.
    An earlier version created a dummy one-column X and hid the real coordinates in
    `obsm["X_pca"]`, which worked but made the object a puzzle: anything inspecting it saw a
    matrix of zeros, and `use_rep` was the only thing pointing at the data.
    """
    emb = np.asarray(emb, dtype="float32")
    if emb.shape[1] == 2:
        return emb                      # a graph method's layout, already computed by scanpy
    import scanpy as sc, anndata as ad
    A = ad.AnnData(X=emb)
    sc.pp.neighbors(A, use_rep="X", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(A, min_dist=min_dist, random_state=seed)
    return np.asarray(A.obsm["X_umap"])


def _same_scale(views):
    """One set of limits for every panel. Per-panel autoscaling makes a dispersed method look
    compact and is the single easiest way to mislead with this figure."""
    allpts = np.vstack(list(views.values()))
    pad = 0.03 * (allpts.max(axis=0) - allpts.min(axis=0))
    return (allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0],
            allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])


def _point_size(n):
    """Points sized for the number of cells. A fixed size that reads well at 5,000 cells is a
    solid block at 100,000, and the panel then shows the outline of the manifold rather than
    where the populations are."""
    if n > 200_000:
        return 0.35
    if n > 80_000:
        return 0.6
    if n > 20_000:
        return 1.2
    return 3.0


def panel(views, colour_by, order, colours, title, out, s=None):
    """One row per colouring, one column per method, all sharing one scale."""
    p = _plt()
    names = list(views)
    xlo, xhi, ylo, yhi = _same_scale(views)
    if s is None:
        s = _point_size(len(np.asarray(colour_by)))
    fig, axs = p.subplots(1, len(names), figsize=(5.0 * len(names), 5.2), squeeze=False)
    for ax, m in zip(axs.ravel(), names):
        xy = views[m]
        for lab in order:
            k = np.asarray(colour_by) == lab
            if k.any():
                ax.scatter(xy[k, 0], xy[k, 1], s=s, c=[colours.get(lab, GREY)],
                           linewidths=0, rasterized=True)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(m, fontsize=11, loc="left")
    # A publication panel names its axes even where the values carry no units. An unlabelled
    # embedding is the commonest reason a reader asks what they are looking at.
    axs.ravel()[0].set_xlabel("UMAP 1", loc="left", fontsize=9)
    axs.ravel()[0].set_ylabel("UMAP 2", loc="bottom", fontsize=9)

    import matplotlib.lines as ml
    fig.legend(handles=[ml.Line2D([], [], marker="o", ls="", ms=5,
                                  color=colours.get(l, GREY), label=_legend_label(l, colours))
                        for l in order],
               loc="lower center", ncol=min(len(order), 6), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "   —   every panel at the SAME scale", fontsize=12, x=.01, ha="left")
    # subplots_adjust sets ONLY what it is given. Passing bottom and top left matplotlib's
    # defaults of left=0.125 and right=0.9 in place, so on a 25-inch figure 3.1 inches of blank
    # sat to the left of the first panel and 2.5 to the right of the last - 22.5% of the width -
    # and every panel was drawn smaller for it.
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.15, top=0.88, wspace=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")   # vector, for a manuscript
    p.close(fig)
    return out


def mixing_bars(rows, factor_names, threshold, out):
    """Per cell type: how well the libraries mix inside it, beside each declared factor.

    The figure exists because the comparison is the finding and a table hides it. A cell type
    whose libraries do not mix is batch structure. A cell type where the AGE GROUPS also do not
    mix is the same structure wearing two names, and a correction on the library key removes the
    contrast the study is for. Side-by-side bars make that pairing visible; two columns of
    numbers do not.

    Returns None when nothing was measurable, so the caller reports a named absence rather than
    an empty panel.
    """
    meas = [r for r in rows if r.get("measured") and r["batch"].get("ratio") is not None]
    if not meas:
        return None
    p = _plt()
    import numpy as np
    labs = [r["label"] for r in meas]
    series = [("batch", [r["batch"]["ratio"] for r in meas], "#c0504d")]
    palette = ["#4f81bd", "#9bbb59", "#8064a2", "#f79646"]
    for i, f in enumerate(factor_names):
        vals = [(r.get("factors") or {}).get(f, {}).get("ratio") for r in meas]
        if any(v is not None for v in vals):
            series.append((f, [np.nan if v is None else v for v in vals],
                           palette[i % len(palette)]))

    n, m = len(labs), len(series)
    h = max(2.6, 0.42 * n * max(1, m) / 2 + 1.4)
    fig, ax = p.subplots(figsize=(8.4, h))
    y = np.arange(n)
    bh = 0.8 / m
    for j, (name, vals, c) in enumerate(series):
        ax.barh(y + (j - (m - 1) / 2) * bh, vals, height=bh, color=c, label=name,
                edgecolor="none")
    ax.axvline(1.0, color=INK, lw=1.0)
    ax.axvline(threshold, color=INK, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("foreign-neighbour share / chance for that population   "
                  "(1.0 = fully mixed)", fontsize=9)
    ax.set_xlim(0, max(1.08, float(np.nanmax([v for _, vs, _ in series for v in vs])) * 1.05))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=m, loc="lower right")
    ax.set_title(f"Mixing within each cell type   —   solid line 1.0 = chance, "
                 f"dashed = declared threshold {threshold}", fontsize=10, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    p.close(fig)
    return out


def highlight(views, mask, label, out, s_bg=0.8, s_hi=3.0):
    """One library (or group) against everything else in grey, in every method's embedding.

    This is the panel that separates aligned from dispersed, and it is the one the metrics
    cannot substitute for.
    """
    p = _plt()
    names = list(views)
    xlo, xhi, ylo, yhi = _same_scale(views)
    fig, axs = p.subplots(1, len(names), figsize=(5.0 * len(names), 5.2), squeeze=False)
    m = np.asarray(mask, dtype=bool)
    for ax, name in zip(axs.ravel(), names):
        xy = views[name]
        ax.scatter(xy[~m, 0], xy[~m, 1], s=s_bg, c=[GREY], linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], s=s_hi, c=["#c0504d"], linewidths=0, rasterized=True)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=11, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{label} (red) against every other cell (grey) — same scale.\n"
                 f"Read whether it sits WITH its counterparts or is dispersed through everything.",
                 fontsize=11, x=.01, ha="left")
    fig.subplots_adjust(left=0.012, right=0.995, top=0.84, bottom=0.02, wspace=0.04)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    p.close(fig)
    return out
