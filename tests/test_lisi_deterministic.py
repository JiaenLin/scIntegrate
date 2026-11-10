#!/usr/bin/env python3
"""iLISI and cLISI give the same value every time a command computes them (GAP-C67, GAP-C14's class).

WHY THIS EXISTS. Under `--lisi-subsample`, scIB's compiled LISI helper scored a random share of the
cells, and it draws that share from std::random_device (scib/knn_graph/knn_graph.cpp: `random_device
rd; mt19937 gen(rd())`). It takes no seed, so no seed this tool passes reaches it: two runs of one
command scored different cells, and iLISI moved by more than 1e-3 between two executions on a 10^5-cell
object, over the tolerance a reproduction allowed it. One metric sits on each side of the scIB total,
so every total moved with them.

THE CHOICE: every cell, not a seed. The helper skips a cell only when its draw from 1 to 100 exceeds
the percentage, so at 100 - what scIB's functions use when they are passed no subsample - no draw can
skip one and the result does not depend on the draw. A seeded subsample would need the helper run on
every cell anyway (it cannot be told which) and scIB's own glue replaced to pick the half, saving only
half of the Python step. Measured on 10^5 synthetic cells: every cell costs about 9 to 11 seconds per
metric more than a 50% subsample; the seeded route would have saved about 6 of them.

What is held, in order:
  1. THE CAUSE (passes on any commit; reads the installed scIB). The helper still seeds from
     std::random_device, takes exactly five arguments (no seed), and skips a cell only when its draw
     exceeds the percentage.
  2. THE BENCHMARK. Every call scintegrate.benchmark makes to scIB's LISI passes `subsample=None`,
     and `benchmark.score` takes no subsample. Fails on f3edb66, which passed the flag's value through.
     Control (any commit): every-cell LISI through `benchmark.score`, twice on one object, is identical.
  3. THE COMMANDS, through the real CLI. For every command line the parser accepts that scores LISI,
     two executions give the same iLISI and cLISI to the last bit, or the line stops at argparse with
     exit 2 naming `--lisi-subsample` and why. The two executions split the Simpson step over
     different numbers of processes, which must not matter either. Fails on f3edb66: `integrate
     --lisi-subsample 50` and `score --lisi-subsample 50` ran, and moved.
  4. THE RECORD. A run records `lisi_subsample: null` (every cell), and its page says the two were
     computed on all the cells and are identical in every execution on the same embedding.

Synthetic throughout: nothing here is biology and nothing may be quoted. Needs anndata, scanpy and scib
with a LISI helper that runs here; prints `SKIP:` without them.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:900]}" if detail and not cond
                                                      else ""))
    if not cond:
        FAILED.append(name)


def finish():
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
    sys.exit(1 if FAILED else 0)


# ---------------------------------------------------------------- 2. the benchmark, read from source
print("\n2. the benchmark passes scIB no subsample")
LISI_FUNCS = {"ilisi_graph", "clisi_graph", "lisi_graph", "lisi_graph_py"}
tree = ast.parse((ROOT / "scintegrate" / "benchmark.py").read_text(encoding="utf-8"))
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
         and getattr(n.func, "attr", getattr(n.func, "id", None)) in LISI_FUNCS]


def _subsample_is_none(call):
    kw = {k.arg: k.value for k in call.keywords}
    return "subsample" in kw and isinstance(kw["subsample"], ast.Constant) and kw["subsample"].value is None


check("benchmark.py calls scIB's iLISI and cLISI", {getattr(c.func, "attr", None) for c in calls}
      >= {"ilisi_graph", "clisi_graph"}, [ast.dump(c.func) for c in calls])
check("and every such call passes subsample=None, so scIB scores every cell",
      bool(calls) and all(_subsample_is_none(c) for c in calls),
      [ast.unparse(c)[:200] for c in calls if not _subsample_is_none(c)])

from scintegrate import benchmark as BM  # noqa: E402

check("benchmark.score takes no subsample, so no caller can believe it passes one",
      not [p for p in inspect.signature(BM.score).parameters if "subsample" in p],
      list(inspect.signature(BM.score).parameters))

try:
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy  # noqa: F401
    import scib
    import scib.metrics as M  # noqa: F401
    import scipy.sparse as sp
except ImportError as e:
    print(f"SKIP: parts 1, 3 and 4 need numpy, pandas, anndata, scanpy and scib ({e})")
    finish()

# ---------------------------------------------------------------- 1. the cause, in the installed scIB
print("\n1. the cause: scIB's LISI helper draws its subsample from an unseeded source")
cpp = Path(scib.__file__).resolve().parent / "knn_graph" / "knn_graph.cpp"
if not cpp.exists():
    print(f"  SKIP: the installed scIB ships no knn_graph.cpp to read ({cpp})")
else:
    src = cpp.read_text(encoding="utf-8", errors="replace")
    check("the helper seeds its generator from std::random_device",
          "random_device rd" in src and re.search(r"mt19937\s+gen\(\s*rd\(\)\s*\)", src) is not None)
    check("it takes exactly five arguments, none of them a seed", "argc != 6" in src)
    check("and skips a cell only when its draw from 1 to 100 exceeds the percentage, so at 100 none",
          re.search(r"uniform_int_distribution<>\s*dis\(\s*1\s*,\s*100\s*\)", src) is not None
          and re.search(r"if\s*\(\s*rand_res\s*>\s*sub\s*\)", src) is not None)


def synthetic(n=3000, d=20, n_lib=4, seed=5):
    """Four libraries with a modest shift, so iLISI and cLISI both sit away from their floors and a
    draw of half the cells moves them (measured: iLISI 0.316 on every cell, 5e-3 between two halves)."""
    rng = np.random.default_rng(seed)
    latent, k = 6, 6
    centres = rng.normal(0, 2.2, size=(k, latent))
    lab = rng.integers(0, k, size=n)
    lib = rng.integers(0, n_lib, size=n)
    Z = centres[lab] + rng.normal(0, 1.0, size=(n, latent))
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    X = Z @ (Q[:, :latent] * np.linspace(1.4, 0.7, latent)).T
    X += rng.normal(0, 0.8, size=(n, d))
    X += rng.normal(0, 0.6, size=(n_lib, d))[lib]
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    obs = pd.DataFrame({"library": [f"L{b + 1}" for b in lib],
                        "cell_type": [f"type_{t}" for t in lab]},
                       index=[f"cell{i:05d}" for i in range(n)])
    return (U * S).astype("float32"), obs


pcs, obs = synthetic()

# ---------------------------------------------------------------- 2b. the control, in process
print("\n2b. every-cell LISI through benchmark.score, twice on one object (any commit)")
RES = [0.5, 1.0]


def lisi_in_process():
    import contextlib
    import io
    A = BM.prepare(pcs, None, obs, "library", "cell_type", kind="embed", seed=0)
    with contextlib.redirect_stdout(io.StringIO()):          # scIB prints every clustering step
        m = BM.score(A, A, "library", "cell_type", kind="embed", n_cores=1, resolutions=RES)
    return {k: m[k]["value"] for k in ("ilisi", "clisi")}, {k: m[k]["why"] for k in ("ilisi", "clisi")}


(v1, why1), (v2, _) = lisi_in_process(), lisi_in_process()
if any(v is None for v in v1.values()):
    print(f"SKIP: scIB's LISI helper does not run here, so parts 2b, 3 and 4 cannot measure ({why1}); "
          f"`scintegrate doctor` says how to build it")
    finish()
check("iLISI and cLISI are identical twice", v1 == v2, (v1, v2))


# ---------------------------------------------------------------- 3. the commands, real CLI
print("\n3. every command line that scores LISI gives the same iLISI and cLISI twice, or is refused")


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, args)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout, r.stderr


COMMON = ["--batch-key", "library", "--label-key", "cell_type", "--species", "homo_sapiens",
          "--n-pcs", "20", "--k", "15", "--scib-resolutions", "2"]
SUBSAMPLED = ["--lisi-subsample", "50"]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n = len(obs)
    rng = np.random.default_rng(3)
    counts = rng.poisson(1.0, size=(n, 40)).astype("float32")
    B = ad.AnnData(X=sp.csr_matrix(np.log1p(counts)), obs=obs.copy())
    B.layers["counts"] = sp.csr_matrix(counts)
    B.var_names = [f"HSG{i:04d}" for i in range(40)]
    B.var["gene_ids"] = [f"ENSG{100000 + i:011d}" for i in range(40)]
    B.obsm["X_pca"] = pcs
    src = td / "synthetic.h5ad"
    B.write_h5ad(src)

    def twice(cmd, extra, tag):
        """Run one command line twice; the two report.json files, or None if it was refused.

        `score` runs over a copy of the first `integrate` run, so that it rewrites that run's record
        at full precision (its tables round to 1e-6), as a killed run's scoring is finished."""
        said = " ".join(extra) or "(no LISI flag)"
        reps = []
        for i, cores in ((1, 1), (2, 3)):
            out = td / f"{tag}_{i}"
            if cmd == "score":
                shutil.copytree(td / "every_1", out)
                rc, so, se = cli("score", "--h5ad", out / "objects" / "cohort_integrated.h5ad",
                                 "--out", out, *COMMON, "--n-cores", cores, *extra)
            else:
                rc, so, se = cli("integrate", "--h5ad", src, "--out", out, "--methods", "none",
                                 *COMMON, "--n-cores", cores, *extra)
            if rc == 2 and i == 1:
                last = (se.strip().splitlines() or [""])[-1]
                check(f"`{cmd} {said}` stops at argparse (exit 2), naming --lisi-subsample and why",
                      "--lisi-subsample" in last and "random_device" in last and "every cell" in last,
                      last)
                return None
            check(f"`{cmd} {said}` run {i} (--n-cores {cores}) exits 0", rc == 0, (so + se)[-1500:])
            if rc != 0:
                return None
            reps.append(json.loads((out / "report.json").read_text(encoding="utf-8")))
        a, b = ({m["method"]: {k: m["metrics"].get(k) for k in ("ilisi", "clisi")} for m in r["methods"]}
                for r in reps)
        check(f"`{cmd} {said}`: iLISI and cLISI were scored",
              bool(a) and all(v is not None for m in a.values() for v in m.values()), a)
        check(f"`{cmd} {said}`: identical to the last bit in the two executions", a == b,
              {m: {k: (a[m][k], (b.get(m) or {}).get(k)) for k in a[m]} for m in a})
        return reps

    every = twice("integrate", [], "every")
    twice("integrate", SUBSAMPLED, "sub")
    rescore = twice("score", [], "rescore") if every else None
    if every:
        twice("score", SUBSAMPLED, "rescore_sub")

    # ------------------------------------------------------------ 4. the record
    print("\n4. the record says which case the run is")
    if every:
        rep = every[0]
        check("integrate records lisi_subsample, and it is null: every cell",
              "lisi_subsample" in rep and rep["lisi_subsample"] is None,
              rep.get("lisi_subsample", "absent"))
        page = (td / "every_1" / "reports" / "integration.html").read_text(encoding="utf-8")
        sec = page.split("<h2>The scIB benchmark</h2>", 1)[-1]
        under = sec.split("</table></div>", 1)[-1].split("<h3>What each metric answers</h3>", 1)[0]
        check("the page, under the totals, says they were computed on all the cells and are identical "
              "in every execution", "computed on all the cells" in under
              and "identical in every execution" in under and "% subsample of the" not in under, under)
    if rescore:
        check("score's rewrite of integrate's record keeps lisi_subsample null",
              all("lisi_subsample" in r and r["lisi_subsample"] is None for r in rescore),
              [r.get("lisi_subsample", "absent") for r in rescore])

finish()
