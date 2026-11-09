#!/usr/bin/env python3
"""`score` rewrites only the record `integrate` wrote, never another command's (GAP-C22).

WHY THIS EXISTS. The README and the quickstart ran `assess` and `integrate` into ONE --out. Both
write report.json there. `integrate` writes its object before it scores and its report.json after,
so an integrate killed during scoring - a walltime, the case `score` exists for - left the object
beside ASSESS's report.json. `score` then found "the run's report.json", wrote the integration's
`chosen`, `rescored` and `object_rescored` into it, and rebuilt assessment.html from the mixture:
a record that says it was an assessment and carries a ranking no assessment computes.

What is held here, through the real CLI on the human cohort of tests/_cohort.py:
  - `assess` writes into a directory; an `integrate` object is then placed there exactly as a run
    killed during scoring leaves it (its object, no report.json of its own);
  - `score` on that object re-scores into tables/ as always;
  - and leaves assess's report.json byte-for-byte as it was, does not rebuild assessment.html,
    and says whose record it found and why it left it.
The other half - an integrate report.json IS rewritten - is tests/test_score_reads_graphs.py.

Needs the analysis stack and scib; prints `SKIP:` without them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import anndata  # noqa: F401
    import scib  # noqa: F401
    from _cohort import cohort, design
except ImportError as e:
    print(f"SKIP: needs anndata, scanpy and scib ({e})")
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


COMMON = ["--batch-key", "library", "--label-key", "cell_type", "--species", "homo_sapiens",
          "--k", "10", "--n-pcs", "15"]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src, des = td / "human_scrna.h5ad", td / "design.csv"
    cohort(src)
    design(des)

    shared = td / "shared"
    rc, log = cli("assess", "--h5ad", src, "--out", shared, "--design", des,
                  "--bio-factor", "condition", *COMMON)
    print("\nassess into a directory")
    check("assess exits 0", rc == 0, log[-1500:])
    rp, page = shared / "report.json", shared / "reports" / "assessment.html"
    check("and leaves its own report.json and assessment.html there",
          rp.is_file() and page.is_file()
          and json.loads(rp.read_text()).get("command") == "assess")

    # The object an integrate killed during scoring leaves: written before the benchmark, with no
    # report.json of its own beside it. Produced by a real integrate elsewhere and placed here.
    other = td / "integrate_elsewhere"
    rc, log = cli("integrate", "--h5ad", src, "--out", other, "--methods", "none",
                  "--scib-resolutions", "3", *COMMON)
    check("integrate, run elsewhere, exits 0", rc == 0, log[-1500:])
    (shared / "objects").mkdir()
    obj = shared / "objects" / "cohort_integrated.h5ad"
    shutil.copy2(other / "objects" / "cohort_integrated.h5ad", obj)

    before = rp.read_bytes()
    page_bytes, page_mtime = page.read_bytes(), page.stat().st_mtime_ns
    time.sleep(0.05)                              # a rewrite would carry a later mtime
    rc, log = cli("score", "--h5ad", obj, "--out", shared, "--scib-resolutions", "3", *COMMON)
    print("\nscore on that object, with assess's record beside it")
    check("score exits 0", rc == 0, log[-1500:])
    check("it re-scores into tables/ as always", (shared / "tables" / "scib_aggregate.csv").is_file())
    after = json.loads(rp.read_text())
    check("assess's report.json is byte-for-byte what assess wrote", rp.read_bytes() == before,
          {k: after.get(k) for k in ("command", "rescored", "object_rescored")})
    check("so it carries no ranking, and claims no re-scoring",
          after.get("command") == "assess"
          and not {"chosen", "rescored", "object_rescored"} & set(after), sorted(after))
    check("assessment.html is not rebuilt from it",
          page.read_bytes() == page_bytes and page.stat().st_mtime_ns == page_mtime)
    check("and the run says whose record it found, and that it left it",
          "written by `assess`" in log and "NOT rewritten" in log, log[-900:])

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
