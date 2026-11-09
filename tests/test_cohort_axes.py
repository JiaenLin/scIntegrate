#!/usr/bin/env python3
"""A human single-cell cohort through the real CLI, and the two axes it declares.

WHY THIS EXISTS. scIntegrate was built against one cohort, and nothing in it had ever been run as
anything else. It took no species and no assay, and it told every cohort that its design was "the
ordinary structure of a design with one library per animal" - a sentence about one study, printed
for all of them. The family's next cohort is not that study (single-cell-harness ADR-0027).

So this builds a small HUMAN SINGLE-CELL cohort - Ensembl ENSG identifiers in var, upper-case
symbols, six libraries from six donors under two conditions, a population found in one library
only, cells withheld from the fit - and runs it through `python -m scintegrate.cli`, the same entry
point a job script and the harness's fixture tier call. Then it reads what the run left on disk.

WHAT IT HOLDS THE TOOL TO
  1. the run completes on that cohort, and nothing it writes says "animal"; the constraint on use
     is worded from THIS design - its factor, its batch key, its counts - and tells a nested
     factor from an aliased one
  2. a declared species the gene identifiers contradict is REFUSED, naming both, before anything
     is computed; a missing --species is refused by argparse, naming it
  3. the assay is recorded - "scrna", "snrna", or "not declared" - and changes no number: the same
     object integrated as cells, as nuclei and undeclared gives identical embeddings and metrics
  4. no computation in the package can read the assay (a source check), so 3 stays true
  5. withheld cells carry NaN in every embedding, never zero; a population in one library is
     reported as not measurable rather than crashing the tables

Needs anndata, scanpy, scipy and harmonypy; prints `SKIP:` without them.
"""
from __future__ import annotations

import ast
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
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp
    import harmonypy  # noqa: F401
    import scib  # noqa: F401
except ImportError as e:
    print(f"SKIP: needs anndata, scanpy, scipy, harmonypy and scib ({e})")
    sys.exit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from _cohort import LIBRARIES, cohort, design  # noqa: E402,F401


def cli(*args):
    """The real entry point, in its own process, as a job script calls it."""
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, args)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout, r.stderr


def integrate(src, out, des, *extra, methods="none,harmony"):
    return cli("integrate", "--h5ad", src, "--out", out, "--batch-key", "library",
               "--label-key", "cell_type", "--design", des, "--bio-factor", "condition",
               "--bio-factor", "donor", "--drop-labels", "EXCLUDED", "--methods", methods,
               "--k", "10", "--n-pcs", "15", "--scib-resolutions", "3", *extra)


def text_of(out):
    """Every human-readable thing the run wrote, and the object's own record."""
    parts = []
    for f in ("report.json", "README.md", "reports/integration.html"):
        p = out / f
        parts.append(p.read_text(encoding="utf-8") if p.exists() else "")
    return "\n".join(parts)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src, des = td / "human_scrna.h5ad", td / "design.csv"
    A = cohort(src)
    design(des)

    # ------------------------------------------------------------------------------------- 1
    print("\n1. a human single-cell cohort runs, and is described in its own terms")
    out = td / "scrna"
    rc, so, se = integrate(src, out, des, "--species", "homo_sapiens", "--assay", "scrna")
    check("integrate exits 0 on a human scRNA cohort", rc == 0, (so + se)[-2500:])
    st = json.loads((out / "STATUS.json").read_text()) if (out / "STATUS.json").exists() else {}
    check("STATUS.json ok and the run sealed", st.get("status") == "ok"
          and (out / "SEALED.txt").exists(), st.get("headline"))
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    con = rep.get("constraint_on_use") or ""
    words = text_of(out)
    obj = ad.read_h5ad(out / "objects" / "cohort_integrated.h5ad") if rc == 0 else None
    uns = obj.uns.get("scintegrate", {}) if obj is not None else {}
    check("no first-cohort sentence: 'one library per animal' is nowhere in what the run wrote",
          rc == 0 and "one library per animal" not in words
          and "one library per animal" not in (uns.get("constraint_on_use") or ""), con[:400])
    check("and the run says nothing about animals at all - the design table cannot know them",
          rc == 0 and "animal" not in words.lower()
          and "animal" not in (uns.get("constraint_on_use") or "").lower())
    check("the constraint is worded from THIS design: its batch key, its factors and its counts",
          "library" in con and "condition takes one value in every level of library" in con
          and "2 values over 6 levels (stimulated on 3, unstimulated on 3)" in con, con[:500])
    check("an aliased factor is told apart from a nested one",
          "donor and library split the cells identically" in con
          and "can remove it entirely for donor" in con, con[:600])
    conf = rep.get("confounding") or {}
    check("report.json carries the classification it rests on",
          (conf.get("condition") or {}).get("status") == "nested"
          and (conf.get("donor") or {}).get("status") == "aliased", conf)
    check("the constraint keeps what it allows and what it forbids",
          "may carry visualisation" in con and "must NOT carry" in con
          and "layers['counts'], which no correction here touches" in con)
    sp_ = rep.get("species") or {}
    check("the species is recorded as declared, and as the gene identifiers measure it",
          sp_.get("as") == "homo_sapiens" and sp_.get("measured") == "homo_sapiens"
          and "agrees" in (sp_.get("verdict") or "") and "gene_ids" in (sp_.get("basis") or ""), sp_)
    check("the assay is recorded as declared", (rep.get("assay") or {}).get("declared") == "scrna"
          and rep.get("unit") == "cells", rep.get("assay"))
    prov = uns.get("provenance") or {}
    check("the object's own provenance carries both axes",
          (prov.get("species") or {}).get("as") == "homo_sapiens"
          and (prov.get("assay") or {}).get("declared") == "scrna", list(prov)[:30])
    readme = (out / "README.md").read_text() if (out / "README.md").exists() else ""
    html = (out / "reports" / "integration.html").read_text() if (out / "reports" / "integration.html").exists() else ""
    check("the README and the report name the species and the assay",
          "homo_sapiens" in readme and "**assay**: `scrna`" in readme
          and "homo_sapiens" in html and "cells ×" in html, readme[:600])
    from scintegrate import __version__
    check("the README and the report say which version wrote them - not 'None on None'",
          f"Written by scintegrate {__version__} on 20" in readme and "None on None" not in readme
          and f"scintegrate {__version__} · generated 20" in html, readme[:200])
    check("the withheld-cells sentence is a sentence", "every embedding. Their per-arm rate" in con,
          con[-400:])

    # ------------------------------------------------------------------------------------- 5
    print("\n5. withheld cells are NaN, and a one-library population is not a crash")
    if obj is not None:
        held = np.asarray(obj.obs["cell_type"].astype(str)) == "EXCLUDED"
        for key in ("X_none", "X_harmony"):
            E = np.asarray(obj.obsm[key], dtype="float64")
            check(f"{key}: every withheld cell is NaN, and no other is",
                  bool(np.isnan(E[held]).all()) and not np.isnan(E[~held]).any(),
                  f"{int(np.isnan(E).any(1).sum())} NaN rows, {int(held.sum())} withheld")
            check(f"{key}: no withheld cell was padded with zero", not (E[held] == 0).any())
    mix = (out / "tables" / "celltype_batch_mixing.csv")
    rows = mix.read_text().splitlines() if mix.exists() else []
    plate = [r for r in rows if r.startswith("Platelet,")]
    check("a population in one library is reported as not measurable, with the reason",
          bool(plate) and "not measured" in plate[0] and "only one level" in plate[0], plate or rows[:3])

    # ------------------------------------------------------------------------------------- 2
    print("\n2. a species the gene identifiers contradict is refused, by name")
    for given in ("mus_musculus", "Mouse"):
        o2 = td / f"refused_{given}"
        rc2, so2, se2 = integrate(src, o2, des, "--species", given, "--assay", "scrna")
        st2 = json.loads((o2 / "STATUS.json").read_text()) if (o2 / "STATUS.json").exists() else {}
        reason = (st2.get("refusal") or {}).get("reason") or ""
        check(f"--species {given} on ENSG identifiers exits 2 and is refused",
              rc2 == 2 and st2.get("status") == "refused" and (o2 / "FAILED.txt").exists(),
              (so2 + se2)[-800:])
        check(f"the refusal names the flag, the declared species and the measured one ({given})",
              all(w in se2 for w in ("--species", given, "homo_sapiens"))
              and all(w in reason for w in (given, "homo_sapiens")), se2[-600:])
        check(f"nothing was computed before the refusal ({given})",
              not (o2 / "objects").exists() and not (o2 / "report.json").exists())
    rc3, so3, se3 = cli("integrate", "--h5ad", src, "--out", td / "nospecies",
                        "--batch-key", "library", "--label-key", "cell_type")
    check("a run without --species is refused by argparse, naming it",
          rc3 == 2 and "--species" in se3 and "required" in se3, se3[-400:])

    # ------------------------------------------------------------------------------------- 3
    print("\n3. the assay is recorded, and changes no number")
    o4 = td / "snrna"
    rc4, so4, se4 = integrate(src, o4, des, "--species", "homo_sapiens", "--assay", "snrna")
    o5 = td / "undeclared"
    rc5, so5, se5 = integrate(src, o5, des, "--species", "homo_sapiens")
    check("the same object runs as nuclei, and undeclared", rc4 == 0 and rc5 == 0,
          (se4 + se5)[-1200:])
    r4 = json.loads((o4 / "report.json").read_text()) if rc4 == 0 else {}
    r5 = json.loads((o5 / "report.json").read_text()) if rc5 == 0 else {}
    check("declared snrna is recorded as snrna, and worded as nuclei",
          (r4.get("assay") or {}).get("declared") == "snrna" and r4.get("unit") == "nuclei"
          and "nuclei ×" in ((o4 / "reports" / "integration.html").read_text() if rc4 == 0 else ""))
    check("absent, it is recorded as 'not declared' - never guessed, never defaulted",
          (r5.get("assay") or {}).get("declared") == "not declared"
          and "assay not declared" in (r5.get("unit") or ""), r5.get("assay"))
    if rc == rc4 == rc5 == 0:
        objs = {k: ad.read_h5ad(d / "objects" / "cohort_integrated.h5ad")
                for k, d in (("scrna", out), ("snrna", o4), ("undeclared", o5))}
        for key in ("X_none", "X_harmony", "X_umap_none", "X_umap_harmony"):
            ref = np.asarray(objs["scrna"].obsm[key], dtype="float64")
            same = all(np.allclose(np.asarray(objs[k].obsm[key], dtype="float64"), ref,
                                   atol=1e-9, rtol=0, equal_nan=True) for k in ("snrna", "undeclared"))
            check(f"{key} is identical as cells, as nuclei and undeclared", same)

        def numbers(r):
            return {m["method"]: (m["metrics"], m["aggregate"], m["knn"]) for m in r["methods"]}
        check("every scIB metric, aggregate and kNN measurement is identical across the three",
              numbers(rep) == numbers(r4) == numbers(r5))
        check("and so is the default, the ranking and the constraint",
              rep["chosen"]["ranked"] == r4["chosen"]["ranked"] == r5["chosen"]["ranked"]
              and rep["constraint_on_use"] == r4["constraint_on_use"] == r5["constraint_on_use"])
        for t in ("scib_aggregate.csv", "knn_metrics.csv", "celltype_batch_mixing.csv"):
            check(f"tables/{t} is byte-identical across the three",
                  (out / "tables" / t).read_bytes() == (o4 / "tables" / t).read_bytes()
                  == (o5 / "tables" / t).read_bytes())

# ----------------------------------------------------------------------------------------- 4
print("\n4. no computation can read the assay")
from scintegrate import axes  # noqa: E402

COMPUTE = ("methods.py", "benchmark.py", "metrics.py", "assess.py", "inputs.py", "emit.py",
           "figures.py", "env.py", "status.py")
leaks = [f for f in COMPUTE if "assay" in (ROOT / "scintegrate" / f).read_text(encoding="utf-8").lower()]
check("the modules that compute never see the assay", not leaks, leaks)


def _tests_assay(node):
    return any((isinstance(n, ast.Name) and "assay" in n.id.lower())
               or (isinstance(n, ast.Attribute) and "assay" in n.attr.lower())
               or (isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "assay")
               for n in ast.walk(node))


branches = []
for f in sorted((ROOT / "scintegrate").glob("*.py")):
    if f.name == "axes.py":
        continue
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        tests = []
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            tests = [node.test]
        elif isinstance(node, ast.comprehension):
            tests = list(node.ifs)
        elif isinstance(node, ast.Match):
            tests = [node.subject]
        branches += [f"{f.name}:{node.lineno}" for t in tests if _tests_assay(t)]
check("no branch anywhere outside axes.py is taken on the assay", not branches, branches)
src = (ROOT / "scintegrate" / "cli.py").read_text(encoding="utf-8")
parser = ast.parse(src)
assay_arg = [kw for n in ast.walk(parser) if isinstance(n, ast.Call)
             and any(isinstance(x, ast.Constant) and x.value == "--assay" for x in n.args)
             for kw in n.keywords if kw.arg == "required"]
required = bool(assay_arg) and isinstance(assay_arg[0].value, ast.Constant) and assay_arg[0].value.value is True
check("--assay is required exactly when a rule is declared to differ by assay",
      required == bool(axes.ASSAY_RULES), f"required={required}, rules={axes.ASSAY_RULES}")
check("the accepted assays are the harness's names", tuple(axes.ASSAYS) == ("scrna", "snrna"))

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
