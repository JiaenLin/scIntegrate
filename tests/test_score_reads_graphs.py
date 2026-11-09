#!/usr/bin/env python3
"""`score` re-scores what `integrate` wrote - graphs included - and gives back the same numbers.

WHY THIS EXISTS (GAP-037). `score` read method results out of `obsm` only. A graph method's
result is its graph, in `obsp`, so BBKNN was dropped from every re-scored ranking without a word,
and report.json's winner was then rewritten over a ranking it was missing from. Its docstring
also promised to rewrite "the report and the object's own uns"; it rewrote neither the document
nor the object. And it computed kNN metrics it never wrote, on arrays that still held the NaN rows
of withheld cells - which sklearn refuses.

What is held here, through the real CLI on the human cohort of tests/_cohort.py:
  - every method `integrate` scored, BBKNN included, is re-scored to the same numbers;
  - the ranking and the default are the same; nothing is retrained;
  - report.json is rewritten and the HTML and README are rebuilt from it;
  - the object - the command's input - is byte-for-byte untouched, and the run says so.

Needs bbknn as well as the analysis stack; prints `SKIP:` without it.
"""
from __future__ import annotations

import hashlib
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
    import anndata  # noqa: F401
    import bbknn  # noqa: F401
    import harmonypy  # noqa: F401
    import scib  # noqa: F401
    from _cohort import cohort, design
except ImportError as e:
    print(f"SKIP: needs anndata, scanpy, harmonypy, bbknn and scib ({e})")
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, args)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


COMMON = ["--batch-key", "library", "--label-key", "cell_type", "--species", "homo_sapiens",
          "--k", "10", "--n-pcs", "15", "--scib-resolutions", "3"]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src, des = td / "human_scrna.h5ad", td / "design.csv"
    cohort(src)
    design(des)
    out = td / "run"
    rc, log = cli("integrate", "--h5ad", src, "--out", out, "--design", des,
                  "--bio-factor", "condition", "--drop-labels", "EXCLUDED",
                  "--methods", "none,harmony,bbknn", *COMMON)
    print("\nintegrate none,harmony,bbknn, with cells withheld from the fit")
    check("integrate exits 0", rc == 0, log[-2000:])
    if rc != 0:
        print(f"\nFAIL: {len(FAILED)} failing")
        sys.exit(1)
    before = json.loads((out / "report.json").read_text())
    obj = out / "objects" / "cohort_integrated.h5ad"
    obj_sha = sha(obj)
    (out / "README.md").unlink()
    (out / "reports" / "integration.html").unlink()

    rc, log = cli("score", "--h5ad", obj, "--out", out, *COMMON)
    print("\nscore on the object integrate wrote")
    check("score exits 0 on an object holding withheld (NaN) cells", rc == 0, log[-2000:])
    after = json.loads((out / "report.json").read_text())
    st = json.loads((out / "STATUS.json").read_text())
    check("STATUS.json ok", st.get("status") == "ok", st.get("headline"))
    check("BBKNN is re-scored, from its graph in obsp", "bbknn (graph, obsp)" in log, log[-1500:])
    bm = {m["method"]: m for m in before["methods"]}
    am = {m["method"]: m for m in after["methods"]}
    for m in ("none", "harmony", "bbknn"):
        check(f"{m}: every scIB metric re-scored to the value integrate computed",
              bm[m]["metrics"] == am[m]["metrics"], {"integrate": bm[m]["metrics"],
                                                      "score": am[m]["metrics"]})
        check(f"{m}: and the same total", bm[m]["aggregate"]["total"] == am[m]["aggregate"]["total"])
    check("the ranking is the same, and still includes the graph method",
          before["chosen"]["ranked"] == after["chosen"]["ranked"]
          and "bbknn" in after["chosen"]["ranked"], (before["chosen"]["ranked"], after["chosen"]["ranked"]))
    agg = (out / "tables" / "scib_aggregate.csv").read_text()
    check("tables/scib_aggregate.csv has a row for the graph method", "\nbbknn,graph," in agg, agg[:400])
    check("report.json says it was re-scored, and that the object was not",
          after.get("rescored") is True and after.get("object_rescored") is False)
    check("the document and the README are rebuilt from the rewritten report.json",
          (out / "README.md").exists() and (out / "reports" / "integration.html").exists())
    check("the object, which is the input here, is byte-for-byte untouched", sha(obj) == obj_sha)
    check("and the run says so", "the object itself is NOT rewritten" in log, log[-600:])

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
