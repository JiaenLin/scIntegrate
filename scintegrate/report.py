"""The stage-3 documents: is integration needed, and what did each method do.

TWO THINGS THIS REPORT WILL NOT DO.

**It will not print a blank cell for a metric that could not be computed.** An empty cell reads as
a zero, and a batch score of zero and a batch score that does not exist lead to opposite
decisions. Every absence is named, with its reason, and the count of metrics each mean was taken
over travels beside the mean.

**It will not present a ranking as a verdict.** A default embedding is named, because a downstream
stage needs one and choosing by eye across five panels is not more rigorous for being manual. What
the name means is stated exactly: highest scIB total under a weighting written on the page, chosen
on cell-type conservation against batch mixing, and nothing else. What the embedding may then be
used for is a separate section, computed from the design rather than assumed.
"""
from __future__ import annotations
import html
import json
from datetime import datetime, timezone
from pathlib import Path

CSS = """
:root{--bg:#fff;--fg:#191919;--mut:#5b5b5b;--line:#e6e4e0;--card:#faf9f7;--warn:#fff8ec;
--warnl:#b06d12;--bad:#fdeeed;--badl:#a8403c;--good:#eef7ee;--goodl:#3f7d43;--acc:#2f5c8a}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e8e6e3;
--mut:#a3a09b;--line:#2d3138;--card:#1d2025;--warn:#2a2115;--warnl:#e0a44a;--bad:#2a1717;
--badl:#e07b76;--good:#16241a;--goodl:#7fbf85;--acc:#8ab4e8}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.5rem 1.5rem 5rem;
font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto}
h1{font-size:1.9rem;margin:0 0 .35rem} h2{font-size:1.2rem;margin:2.8rem 0 .8rem;
padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.6rem 0 .5rem}
.sub{color:var(--mut);font-size:.9rem} .lede{font-size:1.03rem}
.warn,.bad,.good{padding:1rem 1.2rem;margin:1.4rem 0;border-radius:0 5px 5px 0;font-size:.93rem}
.warn{background:var(--warn);border-left:3px solid var(--warnl)}
.bad{background:var(--bad);border-left:3px solid var(--badl)}
.good{background:var(--good);border-left:3px solid var(--goodl)}
.wrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.45rem .55rem;text-align:right;
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-size:.7rem;text-transform:uppercase;color:var(--mut)}
tr.win td{background:var(--good);font-weight:600}
td.absent{color:var(--badl);font-style:italic;font-size:.78rem}
figure{margin:1.8rem 0;padding:1rem;background:var(--card);border:1px solid var(--line);
border-radius:8px} img{max-width:100%;height:auto;display:block;border-radius:4px;background:#fff}
figcaption{color:var(--mut);font-size:.86rem;margin-top:.7rem}
code{font-family:ui-monospace,Consolas,monospace;font-size:.85em}
dl{margin:.6rem 0} dt{font-weight:600;font-size:.85rem;margin-top:.6rem}
dd{margin:.1rem 0 0 0;color:var(--mut);font-size:.85rem}
.pill{display:inline-block;padding:.1rem .45rem;border-radius:3px;background:var(--card);
border:1px solid var(--line);font-size:.75rem;color:var(--mut)}
"""


def _esc(v):
    return html.escape("" if v is None else str(v), quote=True)


def _num(v, fmt="{:.4f}", dash="—"):
    return dash if v is None else fmt.format(v)


def _figs(figs):
    return "".join(
        f"<figure><h3 class='sub'>{_esc(name)}</h3>"
        f'<img src="../figures/{_esc(Path(p).name)}" alt="{_esc(name)}">'
        f"<figcaption>{_esc(cap)}</figcaption></figure>"
        for name, p, cap in figs)


def _celltype_table(payload):
    rows = payload.get("celltypes", [])
    if not rows:
        return ""
    fnames = sorted({f for r in rows for f in (r.get("factors") or {})})
    head = ("<tr><th>cell type</th><th>n</th><th>batch mixing</th>"
            + "".join(f"<th>{_esc(f)}</th>" for f in fnames)
            + "<th></th></tr>")
    thr = payload.get("summary", {}).get("threshold", 0.8)
    body = ""
    for r in rows:
        if not r.get("measured"):
            body += (f"<tr><td>{_esc(r['label'])}</td><td>{r['n']:,}</td>"
                     f"<td class='absent' colspan='{2 + len(fnames)}'>not measured — "
                     f"{_esc(r.get('why', ''))}</td></tr>")
            continue
        rb = r["batch"].get("ratio")
        flag = "" if rb is None or rb >= thr else " class='win'"
        cells = f"<td><b>{_num(rb, '{:.3f}')}</b></td>"
        for f in fnames:
            cells += f"<td>{_num((r.get('factors') or {}).get(f, {}).get('ratio'), '{:.3f}')}</td>"
        also = r.get("also_unmixed_factors") or {}
        note = ("<b>batch AND " + ", ".join(map(_esc, also)) + " both below threshold</b>"
                if also else ("below threshold on batch only" if flag else ""))
        body += (f"<tr{flag}><td>{_esc(r['label'])}</td><td>{r['n']:,}</td>{cells}"
                 f"<td class='sub'>{note}</td></tr>")
    return (f"<div class='wrap'><table>{head}{body}</table></div>"
            f"<p class='sub'>Each number is the share of a cell's neighbours from a different "
            f"level, divided by the share random mixing would give <b>for that population's own "
            f"composition</b> — so <b>1.00 is fully mixed</b> and the chance level is different "
            f"for every row. Highlighted rows fall below the declared threshold "
            f"{_esc(thr)}. Source: <code>tables/celltype_batch_mixing.csv</code> and "
            f"<code>tables/celltype_factor_mixing.csv</code>.</p>")


def _sentinels(payload):
    s = payload.get("sentinels") or {}
    if not s:
        return ""
    tot = sum(s.values())
    items = ", ".join(f"<b>{_esc(k)}</b> {v:,}" for k, v in sorted(s.items(), key=lambda kv: -kv[1]))
    return (f"<div class='warn'><b>{tot:,} cells carry an annotator sentinel, not a cell type</b> "
            f"— {items}. These are statements about the annotation, not populations in the "
            f"tissue, and every label-based metric would otherwise score them as though they "
            f"were. They are excluded from the LABEL metrics only: they remain in every "
            f"embedding, in every figure, and in the delivered object.<br><br>"
            f"How they fall across each arm of the design is "
            f"<code>tables/label_sentinels_by_arm.csv</code> — read it rather than assuming they "
            f"are even, because an exclusion inherited from an upstream filter usually is "
            f"not.</div>")


def _provenance(payload, extra=()):
    rows = [("input", f"<code>{_esc(payload.get('input') or payload.get('h5ad', ''))}</code>"),
            ("batch key", f"<code>{_esc(payload.get('batch_key'))}</code>"),
            ("label key", f"<code>{_esc(payload.get('label_key'))}</code>"),
            ("coarse colouring", _esc(payload.get("coarse"))),
            ("design", _esc(payload.get("design"))),
            ("k", _esc(payload.get("k"))),
            ("n_pcs", _esc(payload.get("n_pcs"))),
            ("seed", _esc(payload.get("seed"))),
            ("scintegrate", _esc(payload.get("version"))),
            ("generated", _esc(payload.get("generated")))]
    rows += list(extra)
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f"<h2>Provenance</h2><div class='wrap'><table>"
            f"<tr><th>item</th><th>value</th></tr>{body}</table></div>")


CANNOT = """<h2>What this cannot show</h2><div class='warn'>
<b>Nothing here establishes that a batch effect is technical.</b> The measurements say where
structure follows the library and where it follows a declared factor; they cannot say which of
those a correction <i>should</i> remove, because that depends on the question being asked next and
nothing measured here knows what it is.<br><br>
<b>It cannot show that the labels are correct.</b> They are inherited from the annotation, and
every limit recorded there still applies. A metric scored against a wrong label set is precise
about the wrong thing.<br><br>
<b>A figure is evidence the table cannot carry.</b> Mixing statistics summarise, and the summary
hides the distinction that matters: a number can say a population was mixed, and only the picture
separates <i>aligned with its counterparts</i> from <i>dispersed uniformly through everything</i>.
</div>"""


# ------------------------------------------------------------------------------------- assess

def write_assess(out_dir, payload, figs):
    out = Path(out_dir)
    summ = payload.get("summary", {})
    n_below = summ.get("n_types_below", 0)
    conf = summ.get("types_below_and_factor_also_unmixed") or []
    cls = "good" if not n_below else ("bad" if conf else "warn")
    doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Is integration needed?</title><style>{CSS}</style><main>
<h1>Is integration needed?</h1>
<p class="sub">{payload.get('n_cells', 0):,} cells · batch <code>{_esc(payload.get('batch_key'))}</code>
· k={_esc(payload.get('k'))} · scintegrate {_esc(payload.get('version'))}
· generated {_esc(payload.get('generated'))}</p>

<p class="lede">Measured on the <b>uncorrected</b> embedding, per cell type, before any method was
run. Nothing was integrated to produce this page.</p>

<div class="{cls}">{_esc(summ.get('indication', ''))}</div>

{_sentinels(payload)}

<h2>Mixing within each cell type</h2>
<p class="lede">The unit of measurement is the cell type, not the cohort. A global number cannot
distinguish one abundant population split by library from every population being split, and those
two call for different decisions.</p>
{_celltype_table(payload)}

<h2>Is it a population at all?</h2>
<p class="lede">A cell type drawn almost entirely from one library is a different object from one
drawn evenly from ten, and integration cannot repair it — there is nothing to align it with. Read
this beside any row above that mixes badly.</p>
<div class="wrap"><table><tr><th>cell type</th><th>n</th><th>batches</th><th>largest batch</th>
<th>its share</th></tr>{''.join(
    f"<tr><td>{_esc(d['label'])}</td><td>{d['n']:,}</td><td>{d['n_batches']}</td>"
    f"<td>{_esc(d['top_batch'])}</td><td>{100 * d['top_share']:.1f}%</td></tr>"
    for d in payload.get('dominance', []))}</table></div>
<p class="sub">Source: <code>tables/celltype_batch_dominance.csv</code>.</p>

<h2>The figures the decision is made on</h2>
<p class="lede">Every panel at the <b>same scale</b>.</p>
{_figs(figs)}

{CANNOT}
{_provenance(payload)}
</main>"""
    (out / "reports").mkdir(parents=True, exist_ok=True)
    p = out / "reports" / "assessment.html"
    p.write_text(doc, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------- integrate

def _bench_table(payload):
    ms = payload.get("methods", [])
    if not ms:
        return ""
    keys = [k for k in ("nmi", "ari", "asw_label", "isolated_labels_f1", "isolated_labels_asw",
                        "clisi", "graph_connectivity", "asw_batch", "ilisi", "kbet", "pcr")
            if any(k in m.get("metrics", {}) for m in ms)]
    chosen = (payload.get("chosen") or {}).get("default")
    head = ("<tr><th>method</th><th>kind</th>"
            + "".join(f"<th>{_esc(k)}</th>" for k in keys)
            + "<th>bio</th><th>batch</th><th>total</th></tr>")
    body = ""
    for m in ms:
        ag = m.get("aggregate", {})
        cells = ""
        for k in keys:
            v = m.get("metrics", {}).get(k)
            if v is None:
                cells += "<td class='absent'>absent</td>"
            else:
                cells += f"<td>{v:.3f}</td>"
        cls = " class='win'" if m["method"] == chosen else ""
        body += (f"<tr{cls}><td><b>{_esc(m['method'])}</b></td>"
                 f"<td class='sub'>{_esc(m.get('kind'))}</td>{cells}"
                 f"<td>{_num(ag.get('bio'), '{:.3f}')} "
                 f"<span class='pill'>{ag.get('n_bio')}/{ag.get('of_bio')}</span></td>"
                 f"<td>{_num(ag.get('batch'), '{:.3f}')} "
                 f"<span class='pill'>{ag.get('n_batch')}/{ag.get('of_batch')}</span></td>"
                 f"<td><b>{_num(ag.get('total'), '{:.3f}')}</b></td></tr>")
    absent = {}
    for m in ms:
        for k, why in (m.get("absent") or {}).items():
            absent.setdefault(f"{k} — {why}", []).append(m["method"])
    ab = ""
    if absent:
        ab = ("<div class='warn'><b>Metrics not computed, and why.</b> Named rather than left "
              "blank: an empty cell reads as a zero.<dl>"
              + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(', '.join(v))}</dd>"
                        for k, v in sorted(absent.items()))
              + "</dl></div>")
    meaning = payload.get("metric_meaning", {})
    dl = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(meaning[k])}</dd>" for k in keys if k in meaning)
    return (f"<div class='wrap'><table>{head}{body}</table></div>"
            f"<p class='sub'>The <span class='pill'>n/of</span> pill is how many metrics each "
            f"mean was taken over. <b>A total from eight metrics is not comparable with a total "
            f"from nine</b>, so the counts travel with the numbers. Source: "
            f"<code>tables/scib_metrics.csv</code>, <code>tables/scib_aggregate.csv</code>, "
            f"<code>tables/scib_absent.csv</code>.</p>{ab}"
            f"<h3>What each metric answers</h3><dl>{dl}</dl>")


def _knn_table(payload):
    ms = payload.get("methods", [])
    body = ""
    for m in ms:
        k = m.get("knn", {})
        ret = "—" if m["method"] == "none" else _num(k.get("knn_retained_mean"), "{:.1%}")
        body += (f"<tr><td><b>{_esc(m['method'])}</b></td>"
                 f"<td>{_num(k.get('foreign_mean'), '{:.1%}')}</td>"
                 f"<td>{_num(k.get('ratio_to_chance'), '{:.2f}×')}</td>"
                 f"<td>{ret}</td>"
                 f"<td>{_num(k.get('label_coherence_mean'), '{:.1%}')}</td></tr>")
    return (f"<div class='wrap'><table><tr><th>method</th><th>foreign neighbours</th>"
            f"<th>vs chance</th><th>kNN retained</th><th>label coherence</th></tr>"
            f"{body}</table></div>"
            f"<p class='sub'><b>vs chance</b> — the foreign share divided by what random mixing "
            f"would give for these batch sizes; <b>1.00× is fully mixed, not 100%</b>. "
            f"<b>kNN retained</b> — the share of each cell's neighbourhood that survived the "
            f"correction, measured against <code>none</code>. This is the cost, and it is why "
            f"higher mixing with lower retention is a trade rather than a victory. Source: "
            f"<code>tables/knn_metrics.csv</code>.</p>")


def write_integrate(out_dir, payload, figs):
    out = Path(out_dir)
    ch = payload.get("chosen") or {}
    summ = payload.get("summary", {})
    obj = payload.get("object", "")

    if ch.get("default"):
        head = (f"<div class='good'><b>Default embedding: "
                f"<code>X_{_esc(ch['default'])}</code></b> — scIB total "
                f"{_num(ch.get('total'), '{:.4f}')}"
                + (f", margin {_esc(ch.get('margin'))} over <code>{_esc(ch.get('runner_up'))}</code>"
                   if ch.get("runner_up") else "") + ".<br><br>"
                f"{_esc(ch.get('reason', ''))}. It is written into the delivered object as "
                f"<code>uns['scintegrate']['default_embedding']</code>, and its 2-D view is "
                f"copied to <code>obsm['X_umap']</code> so a plotting call with no argument uses "
                f"it. Every other method's embedding is kept in the same object.<br><br>"
                f"<b>This is a ranking under one weighting, not a verdict.</b> The weight between "
                f"biological conservation and batch correction is "
                f"<code>--w-bio {_esc(payload.get('w_bio'))}</code>. <b>That weight is the value "
                f"judgement</b>, not a constant of nature — it asserts how much residual batch "
                f"structure is worth trading for retained biology, and a different downstream "
                f"question wants a different answer. It is exposed on the command line for "
                f"exactly that reason.</div>")
        if not ch.get("comparable", True):
            head += ("<div class='bad'><b>The totals do not all rest on the same number of "
                     "metrics.</b> Compare them only with the per-method counts in the benchmark "
                     "table in view; a method whose mean was taken over fewer metrics is not "
                     "ranked on the same basis.</div>")
    else:
        head = (f"<div class='bad'><b>No default embedding was chosen.</b> "
                f"{_esc(ch.get('reason', ''))}</div>")

    cls = "good" if not summ.get("n_types_below") else (
        "bad" if summ.get("types_below_and_factor_also_unmixed") else "warn")

    doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Integration</title><style>{CSS}</style><main>
<h1>Integration</h1>
<p class="sub">{payload.get('n_cells', 0):,} cells × {payload.get('n_genes', 0):,} genes ·
batch <code>{_esc(payload.get('batch_key'))}</code> · k={_esc(payload.get('k'))} ·
scintegrate {_esc(payload.get('version'))} · generated {_esc(payload.get('generated'))}</p>

{head}

<h2>Constraint on use</h2>
<div class="warn">{_esc(payload.get('constraint_on_use', '')).replace('&#10;&#10;', '<br><br>').replace(chr(10) + chr(10), '<br><br>')}</div>

<h2>Was integration needed in the first place?</h2>
<p class="lede">Measured on the <b>uncorrected</b> baseline, per cell type — the same measurement
<code>scintegrate assess</code> makes, carried here so the deliverable answers the question it was
built to settle.</p>
<div class="{cls}">{_esc(summ.get('indication', ''))}</div>
{_celltype_table(payload)}

{_sentinels(payload)}

<h2>The scIB benchmark</h2>
<p class="lede">Every metric computed separately, so an absence is a named cell rather than a
blank one. The total is scIB's own convention: <b>{_esc(payload.get('w_bio'))}</b> on the mean of
the biological-conservation metrics, the remainder on the mean of the batch-correction metrics.</p>
{_bench_table(payload)}

<h2>What each method did, and what it cost</h2>
<p class="lede">This tool's own kNN measurements, which need no external package and are therefore
never absent. Read them as a check on the benchmark rather than a replacement for it.</p>
{_knn_table(payload)}
<div class="warn"><b>A method can win on mixing by destroying structure.</b> Any embedding scores
perfectly on mixing once every neighbourhood is a random sample of libraries. And a
<code>graph</code>-kind method corrects the neighbour graph rather than the coordinates, so its
retention figure is not measuring the same operation as an <code>embed</code>-kind method's — the
<b>kind</b> column is there to stop those two being read as one.</div>

<h2>Before and after, at one scale</h2>
<p class="lede">The leftmost panel is always <code>none</code> — the uncorrected baseline. Every
panel shares one set of axis limits, because per-panel autoscaling makes a dispersed method look
compact and is the easiest way to mislead with this figure.</p>
{_figs(figs)}

<h2>The delivered object</h2>
<div class="wrap"><table><tr><th>item</th><th>value</th></tr>
<tr><td>path</td><td><code>{_esc(obj)}</code></td></tr>
<tr><td>X</td><td>log-normalised, over all samples together</td></tr>
<tr><td>layers</td><td><code>counts</code> (raw integers) · <code>lognorm</code></td></tr>
<tr><td>default embedding</td><td><code>{_esc('X_' + ch['default'] if ch.get('default') else 'none chosen')}</code></td></tr>
<tr><td>every embedding</td><td>{_esc(', '.join('X_' + m['method'] for m in payload.get('methods', []) if m.get('kind') == 'embed'))}</td></tr>
<tr><td>uncorrected baseline</td><td><code>X_pca</code></td></tr>
<tr><td>obs</td><td>the batch key, the label column(s) and the declared design factors — nothing else</td></tr>
</table></div>
<p class="sub">Read <code>README.md</code> beside the object: it is written by inspecting the
directory, so it describes what is actually there.</p>

{''.join(f"<h2>Methods not compared</h2><div class='warn'>" + '; '.join(f'<b>{_esc(k)}</b>: {_esc(v)}' for k, v in payload['not_compared'].items()) + ".<br>A method missing from a comparison changes what the comparison means, so it is named here rather than left as a panel nobody drew.</div>" for _ in [1] if payload.get('not_compared'))}

{CANNOT}
{_provenance(payload, extra=[
    ("n_latent", _esc(payload.get("n_latent"))),
    ("w_bio", _esc(payload.get("w_bio"))),
    ("counts", _esc(payload.get("counts"))),
    ("scIB metrics computed on",
     _esc(f"{payload.get('n_cells', 0) - sum((payload.get('sentinels') or {}).values()):,} cells "
          f"carrying a real label")),
])}
</main>"""
    (out / "reports").mkdir(parents=True, exist_ok=True)
    p = out / "reports" / "integration.html"
    p.write_text(doc, encoding="utf-8")
    return p
