"""The stage-3 document. It presents evidence and REFUSES TO CHOOSE.

That refusal is the design, not modesty. Whether integration is needed is a question about what
the study is for - a batch effect that must be removed for a composition claim may be the very
signal a different question depends on - and no metric in this tool knows which question is
being asked. It reports what each method did and what it cost, shows every method at one scale,
and stops.
"""
from __future__ import annotations
import html, json
from datetime import datetime, timezone
from pathlib import Path

CSS = """
:root{--bg:#fff;--fg:#191919;--mut:#5b5b5b;--line:#e6e4e0;--card:#faf9f7;--warn:#fff8ec;
--warnl:#b06d12;--bad:#fdeeed;--badl:#a8403c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e8e6e3;
--mut:#a3a09b;--line:#2d3138;--card:#1d2025;--warn:#2a2115;--warnl:#e0a44a;--bad:#2a1717;
--badl:#e07b76}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.5rem 1.5rem 5rem;
font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1150px;margin:0 auto}
h1{font-size:1.9rem;margin:0 0 .35rem} h2{font-size:1.2rem;margin:2.6rem 0 .8rem;
padding-bottom:.35rem;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:.9rem} .lede{font-size:1.03rem}
.warn,.bad{padding:1rem 1.2rem;margin:1.4rem 0;border-radius:0 5px 5px 0;font-size:.93rem}
.warn{background:var(--warn);border-left:3px solid var(--warnl)}
.bad{background:var(--bad);border-left:3px solid var(--badl)}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.88rem;
font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.5rem .6rem;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-size:.72rem;text-transform:uppercase;color:var(--mut)}
figure{margin:1.8rem 0;padding:1rem;background:var(--card);border:1px solid var(--line);
border-radius:8px} img{max-width:100%;height:auto;display:block;border-radius:4px;background:#fff}
figcaption{color:var(--mut);font-size:.86rem;margin-top:.7rem}
code{font-family:ui-monospace,Consolas,monospace;font-size:.85em}
"""

REFUSAL = """<b>This report does not choose a method, and no future version of it will.</b>
Whether integration is needed is a question about what the study is for. A batch effect that must
be removed before a composition claim is the same signal another question depends on, and nothing
measured here knows which question is being asked. What follows is what each method did and what
it cost. The decision is yours, and the standing instruction on this project is that it is made
with the FIGURES in front of you, not the table."""


def _esc(v): return html.escape("" if v is None else str(v), quote=True)


def write(out_dir, rows, figs, meta, absent=()):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone()
    base = rows[0] if rows else {}

    hdr = ("<tr><th>method</th><th>foreign neighbours</th><th>vs chance</th>"
           "<th>kNN retained</th><th>label coherence</th></tr>")
    body = ""
    for r in rows:
        ret = "—" if r["method"] == "none" else f"{100*r['knn_retained_mean']:.1f}%"
        body += (f"<tr><td><b>{_esc(r['method'])}</b></td>"
                 f"<td>{100*r['foreign_mean']:.1f}%</td>"
                 f"<td>{r['ratio_to_chance']:.2f}x</td>"
                 f"<td>{ret}</td>"
                 f"<td>{100*r['label_coherence_mean']:.1f}%</td></tr>")

    figs_html = "".join(
        f"<figure><h3 class='sub'>{_esc(name)}</h3>"
        f'<img src="../figures/{Path(p).name}" alt="{_esc(name)}">'
        f"<figcaption>{_esc(cap)}</figcaption></figure>"
        for name, p, cap in figs)

    miss = ""
    if absent:
        miss = ("<h2>Methods not compared</h2><div class='warn'>" + "; ".join(
            f"<b>{_esc(k)}</b>: {_esc(v)}" for k, v in dict(absent).items()) +
            ".<br>A method missing from a comparison changes what the comparison means, so it is "
            "named here rather than left as a panel nobody drew.</div>")

    html_doc = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Integration assessment</title><style>{CSS}</style><main>
<h1>Is integration needed?</h1>
<p class="sub">{meta.get('n_cells',0):,} cells · {meta.get('n_batches',0)} batches
({_esc(meta.get('batch_key'))}) · k={base.get('k','?')} ·
scIntegrate {_esc(meta.get('version'))} · generated {now:%Y-%m-%d %H:%M %Z}</p>

<div class="bad">{REFUSAL}</div>

<h2>What each method did, and what it cost</h2>
<table>{hdr}{body}</table>
<p class="sub"><b>foreign neighbours</b> — share of each cell's k nearest neighbours from a
different batch, averaged. <b>vs chance</b> — that share divided by what random mixing would give
for these batch sizes; <b>1.00x is fully mixed, not 100%</b>. <b>kNN retained</b> — share of each
cell's neighbourhood that survived the correction, measured against <code>none</code>; this is
the cost. <b>label coherence</b> — share of the neighbourhood carrying the same cell-type label.</p>

<div class="warn"><b>A method can win on mixing by destroying structure.</b> Read the three
columns together: higher mixing with lower retention is a trade, not a victory. And
<code>bbknn</code> corrects the GRAPH rather than the PC space, so its retention score is not
measuring the same operation as the others' — compare it to them with that in mind.</div>

<h2>The figures the decision is made on</h2>
<p class="lede">Every panel is at the <b>same scale</b>. A number can say a population was mixed;
only the picture distinguishes <i>aligned with its counterparts</i> from <i>dispersed
everywhere</i>.</p>
{figs_html}
{miss}

<h2>What this cannot show</h2>
<div class="warn">Nothing here establishes that a batch effect is technical. On this design
<b>age is confounded with library chemistry and sequencing run</b>, so a method that removes
"batch" may be removing age. Mixing statistics cannot separate those, and neither can the
figures — they can only show you what was moved. It also cannot show that the labels are correct;
they are inherited from the annotation and every limit recorded there still applies.</div>

<h2>Provenance</h2>
<table><tr><th>item</th><th>value</th></tr>
<tr><td>input</td><td><code>{_esc(meta.get('input'))}</code></td></tr>
<tr><td>batch key</td><td><code>{_esc(meta.get('batch_key'))}</code></td></tr>
<tr><td>label key</td><td><code>{_esc(meta.get('label_key'))}</code></td></tr>
<tr><td>seed</td><td>{_esc(meta.get('seed'))}</td></tr>
</table></main>"""
    (out / "reports").mkdir(exist_ok=True)
    p = out / "reports" / "integration.html"
    p.write_text(html_doc, encoding="utf-8")
    payload = {"generated": now.isoformat(timespec="seconds"), **meta,
               "methods": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
               "not_compared": dict(absent), "chooses_a_method": False}
    (out / "report.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return p, payload
