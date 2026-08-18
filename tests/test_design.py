"""Every check here is a property the tool must keep, not an implementation detail.

Run it directly. No pytest: a test suite that needs a package installed to tell you the package
layer is broken is a test suite you cannot run when you need it most.

    python tests/test_design.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                              # noqa: E402

fails = []


def check(n, c, d=""):
    print(f"  {'ok  ' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


def section(t):
    print(f"\n{t}")


# ============================================================================================
section("A. mixing is measured against CHANCE, not against 100%")
from scintegrate.metrics import mixing, structure_retained, label_coherence                # noqa
from scintegrate.assess import _chance, mixing_within, per_celltype, summarise, dominance  # noqa

rng = np.random.default_rng(0)
n = 200
sep_emb = np.vstack([rng.normal(0, .3, (n, 2)), rng.normal(20, .3, (n, 2))])
bat = np.array(["a"] * n + ["b"] * n)
sep = mixing(sep_emb, bat, k=15)
check("separated batches score near zero", sep["foreign_mean"] < 0.05, f"{sep['foreign_mean']:.3f}")
check("chance for two equal batches is 0.50, not 1.0",
      abs(sep["foreign_expected"] - 0.5) < 1e-9, f"{sep['foreign_expected']:.3f}")
mix = mixing(rng.normal(0, 1, (2 * n, 2)), bat, k=15)
check("intermixed lands near chance", abs(mix["ratio_to_chance"] - 1.0) < 0.15,
      f"{mix['ratio_to_chance']:.2f}x")
check("unequal batches get a DIFFERENT chance level",
      abs(_chance(np.array(["a"] * 360 + ["b"] * 40)) - 0.18) < 0.01,
      f"{_chance(np.array(['a'] * 360 + ['b'] * 40)):.3f}")
check("chance is computed from the composition handed in, not from the cohort",
      abs(_chance(np.array(["a"] * 5 + ["b"] * 5)) - 0.5) < 1e-9)

section("B. retention is the COST side, measured against the baseline")
check("an unchanged embedding retains everything",
      structure_retained(sep_emb, sep_emb, k=15)["knn_retained_mean"] > 0.99)
check("a destroyed embedding retains almost nothing",
      structure_retained(sep_emb, rng.normal(0, 1, sep_emb.shape), k=15)["knn_retained_mean"] < 0.15)
check("mixing alone cannot tell those apart - which is why both are reported",
      mixing(rng.normal(0, 1, sep_emb.shape), bat, k=15)["ratio_to_chance"] > 0.85)

section("C. `none` is a method, is never optional, and is always FIRST")
from scintegrate.methods import METHODS, available, kind, NEEDS_COUNTS, NEEDS_LABELS       # noqa
check("none is declared", "none" in METHODS)
check("none is an embed kind", kind("none") == "embed")
check("bbknn is declared a GRAPH method, not an embedding",
      kind("bbknn") == "graph")
check("the count models are named", set(NEEDS_COUNTS) == {"scvi", "scanvi"})
check("the label-reading model is named", set(NEEDS_LABELS) == {"scanvi"})
cli = (ROOT / "scintegrate/cli.py").read_text()
check("the CLI re-inserts none when omitted", 'ok = ["none"] + ok' in cli)
check("and forces it to the front when present",
      '["none"] + [m for m in ok if m != "none"]' in cli)
check("an unknown method is REPORTED, not skipped", "not a known method" in
      (ROOT / "scintegrate/methods.py").read_text())
_, miss = available(["none", "not_a_method"])
check("an unknown name comes back named", "not_a_method" in miss, miss.get("not_a_method", ""))

section("D. a sentinel is not a cell type")
from scintegrate.inputs import (label_view, coarse_labels, check_counts, DEFAULT_SENTINELS,  # noqa
                               resolve_keys, Refuse, read_design)
import pandas as pd                                                                        # noqa
obs = pd.DataFrame({
    "sample": ["s1"] * 6 + ["s2"] * 6,
    "path": (["Immune/Myeloid/Macrophage"] * 4 + ["EXCLUDED"] + ["UNRESOLVED"]) * 2,
    "L1": (["Immune"] * 4 + ["EXCLUDED"] + ["UNRESOLVED"]) * 2,
})
lab, real, found = label_view(obs, "path")
check("both default sentinels are found and counted",
      found == {"EXCLUDED": 2, "UNRESOLVED": 2}, str(found))
check("the real-label mask excludes exactly those", int(real.sum()) == 8)
check("sentinel cells are NOT dropped from the array - they stay in the object",
      len(lab) == 12)
check("a declared-but-absent sentinel is not an error",
      label_view(obs, "path", ("NOPE",))[2] == {})
check("clearing the sentinels is possible", int(label_view(obs, "path", ())[1].sum()) == 12)
# The default has to survive the CLI, not merely exist in inputs.py. `a.label_sentinel or []`
# collapsed argparse's None into an empty tuple and passed "no sentinels" down, overriding the
# default rather than falling back to it - so EXCLUDED and UNRESOLVED were measured as cell types.
check("the CLI falls back to the default rather than overriding it with empty",
      "if a.label_sentinel is None:" in cli and "sent = inputs.DEFAULT_SENTINELS" in cli)


def _code(src):
    """Source with comment-only lines dropped.

    A grep for a bad pattern matches the COMMENT that explains why the pattern is not used, so
    the test fails on its own documentation. That happened twice while writing these checks,
    which is twice more than it needed to.
    """
    return "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))


check("and does NOT collapse None with `or`",
      "a.label_sentinel or []" not in _code(cli))

section("E. a coarse level is measured, or it is labelled a truncation")
v, note = coarse_labels(obs, "path", l1_key="L1")
check("a measured column is used when given", (v == obs["L1"].values).all())
check("and the provenance names it", "measured column" in note, note)
v2, note2 = coarse_labels(obs, "path")
check("without one, the FULL label is used - the path is not silently split",
      (v2 == obs["path"].values).all())
check("and the note says no coarse column was supplied", "no coarse column" in note2)
v3, note3 = coarse_labels(obs, "path", from_path=True)
check("truncation happens only on explicit request", v3[0] == "Immune")
check("and is labelled a truncation wherever it appears",
      "TRUNCATION" in note3 or "truncation" in note3, note3)

section("F. normalised values are never accepted as counts")


class _Fake:
    def __init__(self, layers):
        self.layers = layers


ok_, why = check_counts(_Fake({"counts": np.array([[1, 2], [3, 4]])}))
check("integral counts pass", ok_, why)
bad, why2 = check_counts(_Fake({"counts": np.array([[0.5, 2.1], [3.0, 4.0]])}))
check("log-normalised values are REFUSED", not bad)
check("and the refusal explains the consequence", "trains anyway" in why2, why2[:80])
absent, why3 = check_counts(_Fake({}))
check("a missing layer is named, not crashed", not absent and "no layers" in why3)

section("G. keys and design join refuse by NAME, listing what is actually there")
try:
    resolve_keys(obs, "sample", "nope")
    check("a missing key refuses", False)
except Refuse as e:
    check("a missing key refuses", True)
    check("and lists the columns that DO exist", "'path'" in str(e), str(e)[:90])

section("H. confounding classifies; it does not veto")
from scintegrate.metrics import confounding                                                # noqa
import scintegrate.metrics as _M                                                           # noqa
check("there is no ranking function in metrics - ranking happens once, in benchmark",
      not hasattr(_M, "composite") and not hasattr(_M, "recommend"))
o_nested = pd.DataFrame({"sample": ["s1"] * 5 + ["s2"] * 5 + ["s3"] * 5 + ["s4"] * 5,
                         "age": ["young"] * 10 + ["aged"] * 10,
                         "diet": (["chow"] * 3 + ["HFD"] * 2) * 4})
c = confounding(o_nested, "sample", ["age", "diet"])
check("a factor constant within batch is NESTED", c["age"]["status"] == "nested")
check("a factor varying within batch is SEPARABLE", c["diet"]["status"] == "separable")
o_alias = pd.DataFrame({"sample": ["s1"] * 10 + ["s2"] * 10, "age": ["young"] * 10 + ["aged"] * 10})
check("one-to-one with the batch key is ALIASED, the stronger case",
      confounding(o_alias, "sample", ["age"])["age"]["status"] == "aliased")
check("an absent factor is reported, not crashed",
      confounding(o_nested, "sample", ["nope"])["nope"]["status"] == "absent")
check("nesting is stated as a CONSTRAINT ON USE, not as a refusal to choose",
      "may carry visualisation" in cli and "must NOT carry" in cli)
check("and it explicitly exempts per-sample counts, which no correction here touches",
      "layers['counts'], which no correction here touches" in cli)

section("I. a rare population is not reported as perfectly mixed")
emb = np.vstack([rng.normal(0, 1, (300, 2)), rng.normal(0, 1, (20, 2))])
lab2 = np.array(["big"] * 300 + ["tiny"] * 20)
bt = np.array((["x"] * 150 + ["y"] * 150) + (["x"] * 10 + ["y"] * 10))
rows = per_celltype(emb, lab2, bt, {}, k=30, min_per_k=3)
by = {r["label"]: r for r in rows}
check("a type below min_per_k*k is NOT measured", by["tiny"]["measured"] is False)
check("its n is still reported", by["tiny"]["n"] == 20)
check("and the reason names the smallness trap",
      "approaches" in by["tiny"]["why"], by["tiny"]["why"][:70])
check("a type above the floor IS measured", by["big"]["measured"] is True)
check("single-level factors report why rather than a ratio",
      mixing_within(emb[:10], ["x"] * 10, k=5)["why"] is not None)

section("J. the assessment indicates, and never returns a verdict")
s_clean = summarise([{"label": "a", "measured": True, "n": 999,
                      "batch": {"ratio": 1.0}, "factors": {}}], 0.8)
check("all-mixed says integration is not indicated",
      "not indicated" in s_clean["indication"], s_clean["indication"][:60])
s_conf = summarise([{"label": "a", "measured": True, "n": 999,
                     "batch": {"ratio": 0.2},
                     "factors": {"age": {"ratio": 0.2}}}], 0.8)
check("batch AND factor both unmixed is flagged as the same structure",
      s_conf["types_below_and_factor_also_unmixed"] == ["a"])
check("and the wording says the contrast is removed with the batch",
      "same structure" in s_conf["indication"])
s_only = summarise([{"label": "a", "measured": True, "n": 999,
                     "batch": {"ratio": 0.2}, "factors": {"age": {"ratio": 1.0}}}], 0.8)
check("batch-only unmixed is reported as separable",
      s_only["types_below_batch_only"] == ["a"] and not
      s_only["types_below_and_factor_also_unmixed"])
check("the threshold travels with the summary", s_only["threshold"] == 0.8)
d = dominance(lab2, bt)
check("dominance is reported per cell type", {x["label"] for x in d} == {"big", "tiny"})

section("K. an absent metric is never a number")
from scintegrate.benchmark import aggregate, choose_default, BIO, BATCH, MEANING            # noqa
m = {k: {"value": 0.5, "why": None} for k in BIO + BATCH}
full = aggregate(m)
check("a complete set aggregates", abs(full["total"] - 0.5) < 1e-9)
check("and records how many metrics each mean covered",
      full["n_bio"] == len(BIO) and full["n_batch"] == len(BATCH))
m2 = dict(m)
m2["kbet"] = {"value": None, "why": "needs rpy2 + the kBET R package"}
part = aggregate(m2)
check("an absent metric lowers the COUNT, not the score",
      part["n_batch"] == len(BATCH) - 1 and abs(part["batch"] - 0.5) < 1e-9)
check("and it is named with its reason", "kbet" in part["absent"])
m3 = {k: {"value": float("nan"), "why": None} for k in BIO + BATCH}
check("NaN is treated as an ABSENCE, never averaged in",
      aggregate(m3)["total"] is None)
check("every metric has a stated meaning for the report",
      set(MEANING) >= set(BIO + BATCH))
# A graph-correcting method has no corrected coordinates. Computing a silhouette or a PCR on the
# uncorrected space it was handed would report the BASELINE's score as that method's own.
bench_src = (ROOT / "scintegrate/benchmark.py").read_text()
check("embedding metrics are absent by KIND for a graph method",
      'if kind == "graph":' in bench_src and 'if kind != "graph":' in bench_src)
check("and the absence says why, rather than being a blank",
      "would\n                             \"report the baseline's score as this method's\"" in bench_src
      or "report the baseline's score as this method's" in bench_src)
check("isolated_labels_f1 is NOT excluded - it clusters, so it is defined either way",
      "isolated_labels_f1 CLUSTERS" in bench_src)
rows_ = [{"method": "none", "aggregate": aggregate(m)},
         {"method": "harmony", "aggregate": aggregate({**m, "nmi": {"value": 0.9, "why": None}})}]
ch = choose_default(rows_)
check("the highest total wins", ch["default"] == "harmony", str(ch["default"]))
check("the margin is reported", ch["margin"] is not None)
check("totals resting on different metric counts are flagged NOT comparable",
      choose_default([{"method": "a", "aggregate": aggregate(m)},
                      {"method": "b", "aggregate": aggregate(m2)}])["comparable"] is False)
check("no complete total means NO default is chosen",
      choose_default([{"method": "a", "aggregate": aggregate(m3)}])["default"] is None)

section("L. the deliverable is one object, and it names what X is")
emitsrc = (ROOT / "scintegrate/emit.py").read_text()
check("there is exactly one write path", emitsrc.count("def write_h5ad") == 1)
check("and it holds the classic string encoding",
      "with classic_string_encoding():" in emitsrc)
check("X's identity is recorded rather than left implicit", '"X_is"' in emitsrc)
check("raw counts are carried for the next stage", '"counts"' in emitsrc)
check("the chosen embedding is named IN the object", '"default_embedding"' in emitsrc)
check("the uncorrected baseline is kept permanently", '"uncorrected_baseline"' in emitsrc)
check("obs is narrowed, and a missing column refuses by name",
      "cannot keep obs column(s) that do not exist" in emitsrc)
check("uns is reduced to plain types so other readers can hold it", "def _plain" in emitsrc)

section("L2. uns must never carry a list of dicts, and score must exist")
from scintegrate.emit import _plain                                                        # noqa
import json as _json                                                                       # noqa
_rows = [{"label": "A", "n": 1, "batch": {"ratio": 0.5}}, {"label": "B", "n": 2}]
_got = _plain({"celltypes": _rows, "flat": [1, 2, 3], "s": "x"})
check("a list of DICTS becomes a string, because HDF5 cannot hold one",
      isinstance(_got["celltypes"], str), type(_got["celltypes"]).__name__)
check("and it is valid JSON, so nothing is lost", len(_json.loads(_got["celltypes"])) == 2)
check("a list of scalars is left alone", _got["flat"] == [1, 2, 3])
check("this is what killed a 12-hour run at its final write",
      "LIST OF DICTS" in emitsrc)
check("`score` re-scores stored embeddings without retraining",
      'sub.add_parser("score"' in cli and "nothing is retrained" in cli)
check("and REFUSES an object holding no embeddings rather than inventing them",
      "does not create one" in cli)

section("L3. the expensive half is written before the fragile half")
check("the object is written BEFORE scoring",
      "BEFORE SCORING" in cli and "writing the embeddings BEFORE scoring" in cli)
check("and rewritten in place once the benchmark exists",
      "rewritten in place, now WITH the benchmark" in cli)
check("the scIB clustering grid is tunable, and says what that costs",
      '"--scib-resolutions"' in cli and "COARSER SEARCH" in cli)
check("the leiden flavour can be restored to scIB's literal behaviour",
      '"--no-fast-leiden"' in cli)

check("a None key in .layers cannot kill a finished run",
      "_names(obj.layers)" in cli and "ENUMERATE A `None` KEY" in cli)

section("N2. the scIB compatibility shims are scoped, not global")
bsrc = (ROOT / "scintegrate/benchmark.py").read_text()
check("pd.value_counts is restored for scib, which calls a pandas-1 API",
      "class scib_pandas_compat" in bsrc)
check("and REMOVED again afterwards, not left on the module", "del pd.value_counts" in bsrc)
check("the leiden shim is scoped and restored too",
      "class scanpy_fast_leiden" in bsrc and "sc.tl.leiden = self._orig" in bsrc)
check("it explains that the flavour changes the implementation, not the question",
      "IMPLEMENTATION rather than the" in bsrc)
check("and falls back rather than failing on an older scanpy",
      "Being slow is a cost" in bsrc)
esrc = (ROOT / "scintegrate/env.py").read_text()
check("doctor EXECUTES the LISI helper rather than checking it exists",
      "def lisi_binary" in esrc and "subprocess.run" in esrc)
check("it names what a broken helper costs: one metric from each side",
      "each side of the scIB" in esrc)
check("and prints the command that repairs it", "LISI_FIX" in esrc and "g++" in esrc)

section("M. no gene class may leave the study by way of HVG selection")
msrc = (ROOT / "scintegrate/methods.py").read_text()
check("the mask is reused verbatim when present", "reused verbatim" in msrc)
check("the fallback computes over ALL genes with no class excluded",
      "no class excluded" in msrc)
check("and the docstring says a mask must never be that route",
      "never become the route" in msrc)
import re as _re                                                                           # noqa
check("neither path clones the full object for a light read",
      _re.search(r"^\s*\w+\s*=\s*adata\.copy\(\)", msrc, _re.M) is None)
check("and each says what it builds instead", "A light object" in msrc)

section("N. an absence is named, never silently substituted")
from scintegrate.env import CAPABILITIES, probe, SCIB_OPTIONAL                             # noqa
check("every capability states what its absence loses",
      all(c["loses"] for c in CAPABILITIES.values()))
check("every capability states how to fix it",
      all(c["hint"] for c in CAPABILITIES.values()))
check("only reading is fatal; methods and the benchmark degrade",
      {k for k, v in CAPABILITIES.items() if v["fatal"]} == {"core", "read"})
check("scib's absence is documented as costing the DEFAULT CHOICE",
      "NO DEFAULT EMBEDDING IS CHOSEN" in CAPABILITIES["scib"]["loses"])
check("kBET's R dependency is declared up front", "kBET" in SCIB_OPTIONAL)
p = probe()
check("probe reports on this interpreter without importing the heavy packages",
      set(p) == set(CAPABILITIES))
check("the doctor is stdlib-only so it runs in a broken env",
      "import numpy" not in (ROOT / "scintegrate/env.py").read_text())

section("O. the report will not print a blank cell for an absent metric")
rep = (ROOT / "scintegrate/report.py").read_text()
check("an absence is rendered as the word, not an empty cell", "absent</td>" in rep)
check("the metric count travels with every aggregate", "n_bio" in rep and "of_bio" in rep)
check("a ranking is labelled a ranking, not a verdict", "not a verdict" in rep)
check("the weight is named as the value judgement", "value judgement" in rep)
check("the constraint on use is its own section", "Constraint on use" in rep)
check("graph and embed kinds are kept distinct in the table", "<b>kind</b>" in rep)
check("figures are referenced relatively, so the document travels with them",
      '"../figures/' in rep)
check("every table names its source file on disk", rep.count("tables/") >= 6)

section("P. every panel is drawn at one scale")
figsrc = (ROOT / "scintegrate/figures.py").read_text()
check("one UMAP min_dist for every view, declared in one place",
      "MIN_DIST = 0.2" in figsrc)
check("no UMAP is drawn without it",
      figsrc.count("sc.tl.umap(") == figsrc.count("min_dist=min_dist"))
check("the graph method's view uses the same value, or it is not comparable",
      "sc.tl.umap(r[\"adata\"], min_dist=a.min_dist" in cli)
check("it is exposed, because it is a layout choice and not a constant",
      '"--min-dist"' in cli and "LAYOUT parameter" in cli)
check("and recorded, so a figure can be traced to the value that drew it",
      '"umap_min_dist"' in cli)
fig = (ROOT / "scintegrate/figures.py").read_text()
check("a shared limit is computed", "_same_scale" in fig)
check("it is applied to every axis", fig.count("set_xlim(xlo, xhi)") >= 2)
check("the assessment figure pairs batch against each factor", "def mixing_bars" in fig)
check("and returns None rather than drawing an empty panel", "return None" in fig)

section("Q. the README is written by inspecting the directory")
rd = (ROOT / "scintegrate/readme.py").read_text()
check("it says so, so nobody trusts it as a template", "by INSPECTING" in rd)
check("all six questions are answered", all(f"## {i}." in rd for i in range(1, 7)))
check("question 6 is what is MISSING", "cannot be done with this output" in rd)
check("it states which file must not be used, and why",
      "Do not** use a `X_umap_*`" in rd or "**Do not**" in rd)

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    raise SystemExit(1)
print("scIntegrate OK — none is never optional, chance is computed, a sentinel is not a cell "
      "type,\n              an absence is named, and nothing is ranked twice.")
