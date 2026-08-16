"""scIntegrate: the baseline is not optional, mixing is measured against chance, and the tool
does not choose.

Every check here is a property the tool must keep, not an implementation detail.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

fails = []
def check(n, c, d=""):
    print(f"  {'ok  ' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c: fails.append(n)

from scintegrate.metrics import mixing, structure_retained, label_coherence, assess  # noqa
from scintegrate.methods import METHODS, available  # noqa

rng = np.random.default_rng(0)

print("A. mixing is measured against CHANCE, not against 100%")
# Two batches, perfectly separated: foreign share ~0
n = 200
emb = np.vstack([rng.normal(0, .3, (n, 2)), rng.normal(20, .3, (n, 2))])
bat = np.array(["a"] * n + ["b"] * n)
sep = mixing(emb, bat, k=15)
check("separated batches score near zero", sep["foreign_mean"] < 0.05,
      f"{sep['foreign_mean']:.3f}")
# Fully intermixed: foreign share approaches the chance level for these sizes
emb2 = rng.normal(0, 1, (2 * n, 2))
mix = mixing(emb2, bat, k=15)
check("chance level is computed from the batch sizes, not assumed 1.0",
      abs(sep["foreign_expected"] - 0.5) < 1e-9, f"{sep['foreign_expected']:.3f}")
check("intermixed lands near chance", abs(mix["ratio_to_chance"] - 1.0) < 0.15,
      f"{mix['ratio_to_chance']:.2f}x")
# unequal batches: chance is NOT 0.5
bat3 = np.array(["a"] * 360 + ["b"] * 40)
ch = mixing(rng.normal(0, 1, (400, 2)), bat3, k=15)["foreign_expected"]
check("unequal batches have a different chance level", abs(ch - 0.18) < 0.01, f"{ch:.3f}")

print("\nB. structure retention is the COST side and is measured against the baseline")
same = structure_retained(emb, emb, k=15)
check("an unchanged embedding retains everything", same["knn_retained_mean"] > 0.99,
      f"{same['knn_retained_mean']:.3f}")
shuf = structure_retained(emb, rng.normal(0, 1, emb.shape), k=15)
check("a destroyed embedding retains almost nothing", shuf["knn_retained_mean"] < 0.15,
      f"{shuf['knn_retained_mean']:.3f}")
check("mixing alone cannot distinguish them — which is why both are reported",
      mixing(rng.normal(0, 1, emb.shape), bat, k=15)["ratio_to_chance"] > 0.85)

print("\nC. `none` is a method and is never optional")
check("none is declared", "none" in METHODS)
ok, missing = available(["harmony"])
check("requesting only harmony still yields a comparison", isinstance(ok, list))
cli = (Path(__file__).resolve().parents[1] / "scintegrate/cli.py").read_text()
check("the CLI re-inserts none if omitted", 'ok = ["none"] + ok' in cli)
check("a missing method is REPORTED, not skipped", "NOT compared" in cli)

print("\nD. the composite score, and the confound that blocks it")
from scintegrate.metrics import composite, confounding, recommend  # noqa
import pandas as pd  # noqa
base = [
    {"method": "none",    "ratio_to_chance": 0.30, "knn_retained_mean": 1.00, "label_coherence_mean": 0.90},
    {"method": "harmony", "ratio_to_chance": 0.95, "knn_retained_mean": 0.70, "label_coherence_mean": 0.85},
    {"method": "wrecker", "ratio_to_chance": 1.40, "knn_retained_mean": 0.05, "label_coherence_mean": 0.20},
]
sc_ = composite(base)
by = {r["method"]: r for r in sc_}
check("over-correction is not rewarded", by["wrecker"]["batch_removal"] == 1.0,
      "ratio 1.40 clipped to 1.00")
check("a structure-destroying method ranks LAST",
      sorted(sc_, key=lambda r: -r["score"])[-1]["method"] == "wrecker")
check("`none` is scored with full retention by definition", by["none"]["bio_conservation"] > 0.9)
check("the weight is carried on every row", all(r["w_bio"] == 0.6 for r in sc_))

# nested: every batch carries exactly one age -> correcting batch removes age
# FOUR samples, TWO ages: age is constant within each sample (nested) but the two do not
# partition the cells identically (not aliased). Both block a recommendation; they are
# distinguished because "aliased" says the batch key IS the factor, which is worse.
obs_nested = pd.DataFrame({"sample": ["s1"]*5 + ["s2"]*5 + ["s3"]*5 + ["s4"]*5,
                           "age": ["young"]*10 + ["aged"]*10,
                           "diet": (["chow"]*3 + ["HFD"]*2) * 4})
c = confounding(obs_nested, "sample", ["age", "diet"])
check("a factor constant within batch is NESTED", c["age"]["status"] == "nested", c["age"]["detail"])
# two samples, two ages: the batch key and the factor are the same partition
obs_alias = pd.DataFrame({"sample": ["s1"]*10 + ["s2"]*10, "age": ["young"]*10 + ["aged"]*10})
check("a one-to-one factor is ALIASED, the stronger case",
      confounding(obs_alias, "sample", ["age"])["age"]["status"] == "aliased")
check("a factor varying within batch is SEPARABLE", c["diet"]["status"] == "separable")
check("an absent factor is reported, not crashed", confounding(obs_nested, "sample", ["nope"])["nope"]["status"] == "absent")

r_blocked = recommend(sc_, c)
check("a nested factor BLOCKS the recommendation", r_blocked["recommended"] is None)
check("the refusal names the blocking factor", "age" in r_blocked["refused_because"])
check("the ranking is still computed and returned", len(r_blocked["ranked"]) == 3)
r_ok = recommend(sc_, {"diet": c["diet"]})
check("with no nesting it DOES recommend", r_ok["recommended"] is not None, str(r_ok["recommended"]))
check("and reports the margin over the runner-up", r_ok["margin"] is not None)

print("\nE. the tool still does not pretend a ranking is a verdict")
rep = (Path(__file__).resolve().parents[1] / "scintegrate/report.py").read_text()
check("a ranking is labelled a ranking, not a verdict", "not a verdict" in rep)
check("the weight is named as the value judgement", "value judgement" in rep)
check("the refusal path exists and states its reason", "NO RECOMMENDATION IS MADE" in rep)
check("it explains why a score is wrong precisely when confounded",
      "most thoroughly\n            removed the contrast" in rep or
      "removed the contrast you intend to measure" in rep)
check("report.json carries the recommendation and the confounding",
      '"recommendation"' in rep and '"confounding"' in rep)

print("\nF. every panel is drawn at one scale")
fig = (Path(__file__).resolve().parents[1] / "scintegrate/figures.py").read_text()
check("a shared limit is computed", "_same_scale" in fig)
check("it is applied to every axis", fig.count("set_xlim(xlo, xhi)") >= 2)

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}"); raise SystemExit(1)
print("scIntegrate OK - none is never optional, mixing is against chance, and nothing is chosen")
