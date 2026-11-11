#!/usr/bin/env python3
"""A batch column with an unused category, and a requested method that fails (GAP-C65).

WHY THIS EXISTS. Harmony failed on a batch column holding a category no cell carries:
harmonypy counts the levels it OBSERVES for theta and builds its design matrix from EVERY
category, so the two disagree by one (`operands could not be broadcast together with shapes
(17,6) (17,7)` on this file's cohort at 0.5.0). An object filtered upstream keeps the categories of
what it lost, so this is the ordinary case, and the harness's own fixture is one. Worse, the run
then reported `ok`, "every method compared and the object written", and sealed: a method the run
was asked for was absent and nothing on disk said so but a line in report.json. `sch dev check`
on the harmony point therefore never ran harmony, and its baseline holds `none` only.

WHAT IT HOLDS THE TOOL TO
  1. harmony runs, through the real CLI, on a batch column holding an unused category, and gives
     the same embedding, to the last bit, as on the same column without it
  2. a requested method that fails leaves STATUS.json `partial`, naming the method in the headline
     and in `absent`, and SEALED.txt - the harness's status contract: it ran, it produced some
     of what it declared, and it says which. The failure is forced in-process, through
     `cli.main`, because no input makes a method fail on purpose once (1) is fixed

Needs anndata, scanpy, pandas, harmonypy and scib; prints `SKIP:` without them.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import harmonypy  # noqa: F401
    import scib  # noqa: F401
except ImportError as e:
    print(f"SKIP: needs anndata, numpy, pandas, harmonypy and scib ({e})")
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from _cohort import cohort  # noqa: E402

ARGS = ["--batch-key", "library", "--label-key", "cell_type", "--species", "homo_sapiens",
        "--methods", "none,harmony", "--k", "10", "--n-pcs", "15", "--scib-resolutions", "3"]


def integrate(src, out):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", "integrate", "--h5ad", str(src),
                        "--out", str(out), *ARGS],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def harmony_of(out):
    p = out / "objects" / "cohort_integrated.h5ad"
    if not p.exists():
        return None
    o = ad.read_h5ad(p)
    return np.asarray(o.obsm["X_harmony"]) if "X_harmony" in o.obsm else None


def status(out):
    p = out / "STATUS.json"
    return json.loads(p.read_text()) if p.exists() else {}


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    A = cohort(td / "base.h5ad")
    libs = sorted(set(A.obs["library"].astype(str)))
    plain = A.copy()
    plain.obs["library"] = pd.Categorical(plain.obs["library"].astype(str), categories=libs)
    plain.write_h5ad(td / "plain.h5ad")
    extra = A.copy()
    extra.obs["library"] = pd.Categorical(extra.obs["library"].astype(str), categories=libs + ["L99"])
    extra.write_h5ad(td / "unused.h5ad")

    # ------------------------------------------------------------------------------------- 1
    print("\n1. harmony on a batch column holding a category no cell carries")
    rc, log = integrate(td / "unused.h5ad", td / "unused")
    rep = json.loads((td / "unused" / "report.json").read_text()) if rc == 0 else {}
    ran = [m["method"] for m in rep.get("methods") or []]
    check("integrate exits 0", rc == 0, log[-1500:])
    check("harmony is compared (0.5.0: 'failed: ValueError: operands could not be broadcast')",
          "harmony" in ran and "harmony" not in (rep.get("not_compared") or {}),
          f"compared {ran}, not compared {rep.get('not_compared')}")
    check("STATUS.json ok", status(td / "unused").get("status") == "ok", status(td / "unused").get("headline"))
    rc2, log2 = integrate(td / "plain.h5ad", td / "plain")
    h_extra, h_plain = harmony_of(td / "unused"), harmony_of(td / "plain")
    check("the same embedding to the last bit as without the unused category",
          rc2 == 0 and h_extra is not None and h_plain is not None and np.array_equal(h_extra, h_plain),
          log2[-800:] if rc2 else "")

    # ------------------------------------------------------------------------------------- 2
    print("\n2. a requested method that fails: STATUS partial, naming it, and sealed")
    from scintegrate import cli, methods as ME  # noqa: E402
    real_run = ME.run

    def failing(adata, method, *a, **kw):
        if method == "harmony":
            raise RuntimeError("forced by the test: this method fails")
        return real_run(adata, method, *a, **kw)

    out = td / "partial"
    ME.run = failing
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = cli.main(["integrate", "--h5ad", str(td / "plain.h5ad"), "--out", str(out), *ARGS])
    finally:
        ME.run = real_run
    st = status(out)
    absent = st.get("absent") or []
    check("integrate returns 0: the run completed, without one method", rc == 0, buf.getvalue()[-1500:])
    check("STATUS.json is partial (0.5.0: ok, 'every method compared')", st.get("status") == "partial",
          f"{st.get('status')!r}: {st.get('headline')!r}")
    check("its headline names the method and the failure",
          "harmony" in (st.get("headline") or "") and "forced by the test" in (st.get("headline") or ""),
          st.get("headline"))
    check("absent[] names the method, with why",
          any(x.get("what") == "harmony" and "forced by the test" in (x.get("why") or "") for x in absent),
          absent)
    check("sealed SEALED.txt, as the contract says for partial, and no RUNNING.txt or FAILED.txt",
          (out / "SEALED.txt").exists() and not (out / "FAILED.txt").exists()
          and not (out / "RUNNING.txt").exists())
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    check("the object and report.json are written, with harmony among the not compared",
          (out / "objects" / "cohort_integrated.h5ad").exists()
          and "harmony" in (rep.get("not_compared") or {}), rep.get("not_compared"))

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
