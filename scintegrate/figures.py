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


def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as p; return p


def _umap(emb, seed=0):
    """A 2-D view. Methods returning a PC space get one computed; BBKNN already returns 2-D."""
    emb = np.asarray(emb)
    if emb.shape[1] == 2:
        return emb
    import scanpy as sc, anndata as ad
    A = ad.AnnData(X=np.zeros((emb.shape[0], 1), dtype="float32"))
    A.obsm["X_pca"] = emb
    sc.pp.neighbors(A, use_rep="X_pca", random_state=seed)
    sc.tl.umap(A, random_state=seed)
    return np.asarray(A.obsm["X_umap"])


def _same_scale(views):
    """One set of limits for every panel. Per-panel autoscaling makes a dispersed method look
    compact and is the single easiest way to mislead with this figure."""
    allpts = np.vstack(list(views.values()))
    pad = 0.03 * (allpts.max(axis=0) - allpts.min(axis=0))
    return (allpts[:, 0].min() - pad[0], allpts[:, 0].max() + pad[0],
            allpts[:, 1].min() - pad[1], allpts[:, 1].max() + pad[1])


def panel(views, colour_by, order, colours, title, out, s=1.0):
    """One row per colouring, one column per method, all sharing one scale."""
    p = _plt()
    names = list(views)
    xlo, xhi, ylo, yhi = _same_scale(views)
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
        ax.set_title(m, fontsize=11, loc="left")
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color=colours.get(l, GREY), label=str(l)) for l in order],
               loc="lower center", ncol=min(len(order), 8), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title + "   —   every panel at the SAME scale", fontsize=12, x=.01, ha="left")
    fig.subplots_adjust(bottom=0.14, top=0.88)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); p.close(fig)
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
    fig.suptitle(f"{label} (red) against every other cell (grey) — same scale.\n"
                 f"Read whether it sits WITH its counterparts or is dispersed through everything.",
                 fontsize=11, x=.01, ha="left")
    fig.subplots_adjust(top=0.84)
    fig.savefig(out, dpi=130, bbox_inches="tight"); p.close(fig)
    return out
