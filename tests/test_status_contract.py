#!/usr/bin/env python3
"""What a run leaves behind: STATUS.json and a self-written seal; `describe`; sees declared once.

WHAT IS CHECKED (stdlib only)

  1. begin() writes STATUS.json `partial` and RUNNING.txt with the commit read from .git by file
  2. finish(ok) seals only when every expected product exists; a missing one is `failed`
  3. finish(refused) carries a fix; the seal is FAILED (a refusal is not a success)
  4. `scintegrate describe` prints JSON with needs/provides/sees/cannot_show/state_version, and
     sees says exactly which methods see the labels — read from one declaration
  5. benchmark.LABEL_SUPERVISED is derived from methods.SEES, never restated
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from scintegrate import status as ST  # noqa: E402

print("\nstatus writers")
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "run"
    ST.begin(out, command="integrate", version="0.0.0", state_version=1, sees=["labels"], cannot_show=["x"])
    rec = json.loads((out / "STATUS.json").read_text())
    check("1 partial first", rec["status"] == "partial" and (out / "RUNNING.txt").exists())
    check("1 commit read by file", rec["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", rec["commit"]) is not None, rec["commit"])
    (out / "report.json").write_text("{}")
    rec = ST.finish(out, status="ok", headline="done", exit_code=0, expected=["report.json", "objects/x.h5ad"])
    check("2 a missing expected product is failed, named", rec["status"] == "failed" and rec["missing"] == ["objects/x.h5ad"]
          and (out / "FAILED.txt").exists())
    (out / "objects").mkdir(); (out / "objects" / "x.h5ad").write_text("h")
    rec = ST.finish(out, status="ok", headline="done", exit_code=0, expected=["report.json", "objects/x.h5ad"])
    check("2 every product present seals", rec["status"] == "ok" and (out / "SEALED.txt").exists()
          and not (out / "FAILED.txt").exists() and not (out / "RUNNING.txt").exists())
    check("2 products are relative", all(not p["path"].startswith("/") for p in rec["products"]))
    rec = ST.finish(out, status="refused", headline="no label column", exit_code=2)
    check("3 refused carries a fix and FAILED", rec["refusal"]["fix"] and (out / "FAILED.txt").exists())

print("\ndescribe")
from scintegrate import cli  # noqa: E402
buf = io.StringIO()
with redirect_stdout(buf):
    rc = cli.main(["describe"])
d = json.loads(buf.getvalue())
check("4 exits 0 with JSON", rc == 0 and isinstance(d, dict))
for f in ("needs", "provides", "sees", "cannot_show", "state_version", "gates", "escapes", "methods", "baseline_first"):
    check(f"4 field {f}", f in d)
check("4 only scANVI sees the labels", [m for m, s in d["sees"].items() if "labels" in s] == ["scanvi"], d["sees"])
check("4 cannot_show has at least three sentences", len(d["cannot_show"]) >= 3)
check("4 state_version is an integer", isinstance(d["state_version"], int))

print("\none declaration")
from scintegrate import methods as ME, benchmark as BM  # noqa: E402
check("5 LABEL_SUPERVISED is derived from SEES", tuple(BM.LABEL_SUPERVISED) == tuple(m for m, s in ME.SEES.items() if "labels" in s))
src = (ROOT / "scintegrate" / "benchmark.py").read_text()
check("5 benchmark does not restate the tuple", re.search(r'LABEL_SUPERVISED\s*=\s*\(', src) is None)
check("5 NEEDS_LABELS is derived too", ME.NEEDS_LABELS == tuple(m for m, s in ME.SEES.items() if "labels" in s))

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
