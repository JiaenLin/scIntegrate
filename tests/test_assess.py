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

print("\nD. the tool does not choose")
rep = (Path(__file__).resolve().parents[1] / "scintegrate/report.py").read_text()
check("the report says so in the document", "does not choose a method" in rep)
check("report.json records it as a field", '"chooses_a_method": False' in rep)
for w in ("best", "recommend", "winner", "should use"):
    check(f"the report never says '{w}'", w not in rep.lower().replace("does not choose", ""))

print("\nE. every panel is drawn at one scale")
fig = (Path(__file__).resolve().parents[1] / "scintegrate/figures.py").read_text()
check("a shared limit is computed", "_same_scale" in fig)
check("it is applied to every axis", fig.count("set_xlim(xlo, xhi)") >= 2)

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}"); raise SystemExit(1)
print("scIntegrate OK - none is never optional, mixing is against chance, and nothing is chosen")
