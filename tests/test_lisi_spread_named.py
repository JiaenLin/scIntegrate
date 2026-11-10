#!/usr/bin/env python3
"""The LISI metrics' measured spread is named beside the totals it moves (GAP-C14).

WHY THIS EXISTS. Under the `--lisi-subsample` that versions before 0.5.0 accepted, scIB's LISI helper
drew its subsample from std::random_device (scib/knn_graph/knn_graph.cpp: `random_device rd; mt19937
gen(rd())`), so no seed reached it, two executions scored different cells, and iLISI and cLISI - one
on each side of the total - moved between two runs of one command. The movement was stated nowhere a
reader of the report would see it. From 0.5.0 they are computed on every cell
(tests/test_lisi_deterministic.py holds that), so the spread now describes only a RECORD written with
a subsample, and the report of such a record must still say so.

What is held here:
  1. the measured spread is declared once, in the package, and KNOWN_ISSUES.md states the same
     numbers;
  2. the report says, under the totals, which case this run is: estimated on a subsample (with the
     measured spread), computed on every cell (no subsample is drawn), or not recorded (a report.json
     from before the record);
  3. through the real CLI: integrate records `lisi_subsample: null` in report.json, the page says
     every cell, and score's rewrite says null too.
Parts 1 and 2 are stdlib; part 3 needs the analysis stack and prints `SKIP:` without it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scintegrate import benchmark as BM  # noqa: E402
from scintegrate.report import write_integrate  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def sci(x):
    """1.5e-4, as the measurement is written in KNOWN_ISSUES and on the page."""
    return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e+")


# --------------------------------------------------------------------------------------- 1
print("\n1. the measured spread is declared once, and KNOWN_ISSUES states the same numbers")
SP = getattr(BM, "LISI_SUBSAMPLE_SPREAD", None)
check("benchmark.LISI_SUBSAMPLE_SPREAD declares the measured spread", isinstance(SP, dict), SP)
SP = SP if isinstance(SP, dict) else {}
check("for iLISI and cLISI, at a stated subsample percentage, with the date it was measured",
      all(isinstance(SP.get(k), float) and SP[k] > 0 for k in ("ilisi", "clisi"))
      and isinstance(SP.get("at_percent"), int) and bool(SP.get("measured")), SP)
issues = (ROOT / "KNOWN_ISSUES.md").read_text(encoding="utf-8")
entry = next((s for s in re.split(r"\n(?=## )", issues) if "--lisi-subsample" in s
              and "iLISI" in s and "cLISI" in s and "random_device" in s), "")
check("KNOWN_ISSUES.md has an entry on the subsample's spread", bool(entry))
check("and it states the declared numbers", bool(SP) and all(
    s in entry for s in (sci(SP.get("ilisi", 0)), sci(SP.get("clisi", 0)),
                         f"--lisi-subsample {SP.get('at_percent')}")), entry[:300])

# --------------------------------------------------------------------------------------- 2
print("\n2. the report says which case this run is, under the totals")


def payload(**over):
    m = {"nmi": 0.5, "clisi": 0.9, "ilisi": 0.2}
    p = {"command": "integrate", "n_cells": 100, "n_genes": 10, "w_bio": 0.6,
         "chosen": {"default": "none", "total": 0.5, "reason": "highest scIB total"},
         "methods": [{"method": "none", "kind": "embed", "metrics": m, "absent": {},
                      "aggregate": {"bio": 0.7, "batch": 0.2, "total": 0.5, "n_bio": 2, "of_bio": 6,
                                    "n_batch": 1, "of_batch": 5}, "knn": {}}]}
    p.update(over)
    return p


def beside_totals(p):
    """The page between the benchmark table and the metric glossary: where the totals are read."""
    with tempfile.TemporaryDirectory() as td:
        html = write_integrate(td, p, []).read_text(encoding="utf-8")
    sec = html.split("<h2>The scIB benchmark</h2>", 1)[-1]
    return sec.split("</table></div>", 1)[-1].split("<h3>What each metric answers</h3>", 1)[0]


sub = beside_totals(payload(lisi_subsample=50))
check("a subsampled run: says the two were estimated on a 50% subsample", "50% subsample" in sub, sub)
check("and names the measured spread, where it was measured, and KNOWN_ISSUES",
      bool(SP) and sci(SP["ilisi"]) in sub and sci(SP["clisi"]) in sub
      and f"--lisi-subsample {SP['at_percent']}" in sub and "KNOWN_ISSUES" in sub, sub)
every = beside_totals(payload(lisi_subsample=None))
check("a run on every cell: says no subsample was drawn, and claims no spread for this page",
      "all the cells" in every and "no subsample" in every and "move between" not in every, every)
nuclei = beside_totals(payload(lisi_subsample=50, unit="nuclei"))
check("and it names the run's own unit: a nucleus run's subsample is of the nuclei",
      "50% subsample of the nuclei" in nuclei, nuclei)
old = beside_totals(payload())
check("a report.json from before the record: says it does not record which, and names the spread",
      "does not record" in old and bool(SP) and sci(SP["ilisi"]) in old, old)
none_there = beside_totals(payload(lisi_subsample=50, methods=[
    {**payload()["methods"][0], "metrics": {"nmi": 0.5, "clisi": None, "ilisi": None},
     "absent": {"clisi": "x", "ilisi": "x"}}]))
check("a run where neither LISI metric exists says nothing about their subsample",
      "subsample" not in none_there, none_there)

# --------------------------------------------------------------------------------------- 3
print("\n3. through the real CLI: integrate records every cell, score's rewrite says the same")
try:
    import anndata  # noqa: F401
    import scib  # noqa: F401
    from _cohort import cohort
except ImportError as e:
    print(f"SKIP: part 3 needs anndata, scanpy and scib ({e})")
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
    sys.exit(1 if FAILED else 0)


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, args)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


COMMON = ["--batch-key", "library", "--label-key", "cell_type", "--species", "homo_sapiens",
          "--k", "10", "--n-pcs", "15", "--scib-resolutions", "3"]
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src, out = td / "human_scrna.h5ad", td / "run"
    cohort(src)
    rc, log = cli("integrate", "--h5ad", src, "--out", out, "--methods", "none", *COMMON)
    check("integrate exits 0", rc == 0, log[-1500:])
    rep = json.loads((out / "report.json").read_text()) if rc == 0 else {}
    check("report.json records lisi_subsample, and it is null: every cell",
          "lisi_subsample" in rep and rep["lisi_subsample"] is None,
          {k: rep.get(k, "absent") for k in ("lisi_subsample", "version")})
    page = (out / "reports" / "integration.html").read_text() if rc == 0 else ""
    have = any(m.get("metrics", {}).get(k) is not None for m in rep.get("methods", [])
               for k in ("ilisi", "clisi"))
    # Where scIB's compiled helper cannot run (a prebuilt binary on another platform), both LISI
    # metrics are absent, and a sentence about them would describe numbers that are not on the
    # page. Each case is held to what it should say.
    check("the page says this run's LISI numbers are on every cell, and names no subsample" if have
          else "LISI is absent here, and the page says nothing about how it was scored",
          ("computed on all the cells" in page and "% subsample" not in page) if have
          else ("computed on all the cells" not in page and "ilisi" in page),
          f"LISI present: {have}")
    rc, log = cli("score", "--h5ad", out / "objects" / "cohort_integrated.h5ad", "--out", out, *COMMON)
    rep2 = json.loads((out / "report.json").read_text()) if rc == 0 else {}
    check("score's rewrite of the record says the same: lisi_subsample is null, every cell",
          rc == 0 and "lisi_subsample" in rep2 and rep2["lisi_subsample"] is None,
          {"rc": rc, "lisi_subsample": rep2.get("lisi_subsample", "absent")})

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
