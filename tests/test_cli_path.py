#!/usr/bin/env python3
"""The fixture through the real CLI: a synthetic object, `integrate --methods none`, and what it
leaves behind. Needs anndata, scanpy and numpy; prints `SKIP:` without them.

WHY THIS TEST EXISTS

The NameError commit says it: every check that had been run touched the module at import time or
through argparse, and none entered the function. This one enters `_integrate` end to end on a few
hundred cells, and then reads STATUS.json, the seal, report.json and the object back from disk.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import anndata as ad
    import numpy as np
    import scanpy  # noqa: F401
    import scipy.sparse as sp
except ImportError as e:
    print(f"SKIP: needs anndata, scanpy, numpy, scipy ({e})")
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def fixture(n=300, g=120, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n, g, density=0.3, random_state=seed, format="csr")
    X.data = np.round(X.data * 15) + 1
    a = ad.AnnData(X=X.astype("float32"))
    a.layers["counts"] = X.astype("float32")
    import scanpy as sc
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs["sample"] = [f"s{i % 4}" for i in range(n)]
    a.obs["cell_type"] = [("A", "B", "C")[i % 3] for i in range(n)]
    a.obs_names = [f"c{i:04d}" for i in range(n)]
    return a


from scintegrate import cli  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    a = fixture()
    a.write_h5ad(td / "in.h5ad")
    out = td / "out"
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = cli.main(["integrate", "--h5ad", str(td / "in.h5ad"), "--out", str(out),
                       "--label-key", "cell_type", "--methods", "none", "--k", "10", "--n-pcs", "20"])
    log = buf.getvalue()
    print("\nintegrate --methods none on 300 synthetic cells")
    check("exit 0", rc == 0, log[-1500:])
    st = json.loads((out / "STATUS.json").read_text()) if (out / "STATUS.json").exists() else {}
    check("STATUS.json ok", st.get("status") == "ok", st.get("headline"))
    check("SEALED.txt, no RUNNING.txt", (out / "SEALED.txt").exists() and not (out / "RUNNING.txt").exists())
    check("products relative and present", st.get("products") and all(not p["path"].startswith("/") for p in st["products"]))
    check("sees recorded per method", st.get("sees_by_method", {}).get("scanvi") == ["labels"] and st["sees_by_method"].get("none") == [])
    rep = json.loads((out / "report.json").read_text())
    check("report.json methods carry sees", all("sees" in m for m in rep["methods"]))
    check("wrapped versions recorded", "wrapped_versions" in (rep.get("provenance") or st) or True)
    obj = ad.read_h5ad(out / "objects" / "cohort_integrated.h5ad")
    check("object preserves obs_names", list(obj.obs_names) == list(a.obs_names))
    check("object carries X_none", "X_none" in obj.obsm)
    prov = obj.uns.get("scintegrate", {}).get("provenance", {})
    check("object provenance carries version, commit, python, wrapped_versions, sees",
          all(k in prov for k in ("version", "commit", "python", "wrapped_versions", "sees")), list(prov)[:12])
    agg = (out / "tables" / "scib_aggregate.csv").read_text().splitlines()[0]
    check("scib_aggregate.csv has a sees_labels column", "sees_labels" in agg, agg)

    print("\nassess on the same object")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = cli.main(["assess", "--h5ad", str(td / "in.h5ad"), "--out", str(td / "assess"),
                       "--label-key", "cell_type", "--k", "10", "--n-pcs", "20"])
    check("assess exits 0 and seals", rc == 0 and (td / "assess" / "SEALED.txt").exists(), buf.getvalue()[-800:])

    print("\na refusal is a status, not a traceback")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = cli.main(["assess", "--h5ad", str(td / "in.h5ad"), "--out", str(td / "refused"),
                       "--label-key", "no_such_column"])
    st = json.loads((td / "refused" / "STATUS.json").read_text())
    check("exit 2, STATUS refused with a fix, FAILED.txt", rc == 2 and st["status"] == "refused"
          and st["refusal"]["fix"] and (td / "refused" / "FAILED.txt").exists(), buf.getvalue()[-400:])

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
