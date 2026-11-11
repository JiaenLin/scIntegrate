#!/usr/bin/env python3
"""The design's KEY column travels in the deliverable, under the name the design gave it.

THE DEFECT (single-cell-harness GAP-C102, measured with the real tools). A design table is keyed on
the study's own name for its library column; scIntegrate matches its rows on `--batch-key` and
carried every design column but that one into the object. A later stage is handed the study's
name, finds no such column in the object scIntegrate wrote, and stops. Every study whose library
column is not called what `--batch-key` calls it stopped there.

WHAT IT HOLDS THE TOOL TO, through `python -m scintegrate.cli integrate`, on tests/_cohort.py's
cohort (six libraries in `library`, cells withheld with `--drop-labels EXCLUDED`):
  1. a design keyed on a name the object lacks: that column is in the object, LAST, equal to the
     batch column on every cell, the withheld cells included; report.json and the object's
     provenance say it was added;
  2. a design keyed on the batch key itself: nothing is added, the columns are the ones written
     before this change, and the record says the key is the batch column;
  3. an object that already carries a column of the key's name, equal to the batch: it is kept as
     it is, and the record says so;
  4. one that differs from the batch on any cell: refused, by name, before anything is computed
     (exit 2, no object), because two columns would claim one name;
  5. runs 1 and 2 differ in that one column only: every other obs column, var, X, layers, every
     obsm entry and every obsp entry are equal, bit for bit.

    python tests/test_design_key_kept.py
"""
from __future__ import annotations

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
except ImportError as e:
    print(f"SKIP: needs anndata and numpy ({e}). A skip is not a pass.")
    raise SystemExit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from _cohort import DESIGN_COLUMNS, cohort, design  # noqa: E402

KEY = "library_name"      # the design's own word for the library column; not the batch key's


def integrate(src, out, des, *extra):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", "integrate", "--h5ad", str(src),
                        "--out", str(out), "--batch-key", "library", "--label-key", "cell_type",
                        "--species", "homo_sapiens", "--methods", "none,harmony", "--k", "10",
                        "--n-pcs", "15", "--scib-resolutions", "3", "--drop-labels", "EXCLUDED",
                        "--design", str(des), *map(str, extra)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def read(out):
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    p = out / "objects" / "cohort_integrated.h5ad"
    return rep, (ad.read_h5ad(p) if p.exists() else None)


def provenance(obj):
    return ((obj.uns.get("scintegrate") or {}).get("provenance") or {}) if obj is not None else {}


def renamed_design(path):
    """The cohort's design table with its key column under the study's own word, rows unchanged."""
    design(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    head[DESIGN_COLUMNS.index("library")] = KEY
    path.write_text("\n".join([",".join(head)] + lines[1:]) + "\n", encoding="utf-8")


def keys(m):
    """A mapping's names, sorted; anndata can list a None among a container's keys."""
    return sorted(str(k) for k in m.keys() if k is not None)


def same_matrix(a, b):
    if hasattr(a, "toarray"):
        a = a.toarray()
    if hasattr(b, "toarray"):
        b = b.toarray()
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src = td / "joint.h5ad"
    A0 = cohort(src)
    lib = np.asarray(A0.obs["library"].astype(str))
    withheld = np.asarray(A0.obs["cell_type"].astype(str)) == "EXCLUDED"
    plain, renamed = td / "design.csv", td / "design_renamed.csv"
    design(plain)
    renamed_design(renamed)

    print("\n1 - a design keyed on a name the object lacks: the column is added, last")
    rc1, log1 = integrate(src, td / "r1", renamed, "--design-sample-col", KEY)
    check("integrate exits 0", rc1 == 0, log1[-1500:])
    rep1, o1 = read(td / "r1")
    cols1 = list(o1.obs.columns) if o1 is not None else []
    check(f"the object carries {KEY!r}", KEY in cols1, f"obs columns {cols1}")
    check(f"{KEY!r} is the last column, after the design's factors", cols1[-1:] == [KEY], f"{cols1}")
    if o1 is not None and KEY in cols1:
        got = np.asarray(o1.obs[KEY].astype(str))
        check(f"{KEY!r} equals the batch column on every one of the {len(got)} cells",
              np.array_equal(got, lib), f"{int((got != lib).sum())} differ")
        check(f"... the {int(withheld.sum())} withheld cells included (their own library, not a sentinel)",
              bool(withheld.any()) and np.array_equal(got[withheld], lib[withheld]),
              f"{sorted(set(got[withheld]))}")
    dk1 = rep1.get("design_key") or {}
    check("report.json design_key: added, under the design's name",
          dk1.get("column") == KEY and dk1.get("added") is True, f"{rep1.get('design_key', 'absent')!r}")
    check("the object's provenance carries the same record", provenance(o1).get("design_key") == dk1 and bool(dk1),
          f"{provenance(o1).get('design_key', 'absent')!r}")

    print("\n2 - a design keyed on the batch key itself: nothing is added")
    rc2, log2 = integrate(src, td / "r2", plain)
    check("integrate exits 0", rc2 == 0, log2[-1500:])
    rep2, o2 = read(td / "r2")
    cols2 = list(o2.obs.columns) if o2 is not None else []
    want2 = ["library", "cell_type"] + [c for c in DESIGN_COLUMNS if c != "library"]
    check("the columns are the batch, the label and the design's factors, as before", cols2 == want2,
          f"{cols2} vs {want2}")
    dk2 = rep2.get("design_key") or {}
    check("report.json design_key: the batch column itself, nothing added",
          dk2.get("column") == "library" and dk2.get("added") is False and dk2.get("how") == "the batch column itself",
          f"{rep2.get('design_key', 'absent')!r}")

    print("\n5 - runs 1 and 2 differ in that one column only")
    if o1 is not None and o2 is not None:
        check("obs: run 1's columns are run 2's plus the key", cols1 == cols2 + [KEY], f"{cols1} vs {cols2}")
        check("obs_names equal, in order", list(o1.obs_names) == list(o2.obs_names))
        diff = [c for c in cols2 if not o1.obs[c].equals(o2.obs[c])]
        check("every other obs column equal, values and dtype", not diff, f"differ: {diff}")
        check("var equal", o1.var.equals(o2.var) and list(o1.var_names) == list(o2.var_names))
        check("X equal", same_matrix(o1.X, o2.X))
        check("layers equal", keys(o1.layers) == keys(o2.layers)
              and all(same_matrix(o1.layers[k], o2.layers[k]) for k in keys(o1.layers)), f"{keys(o1.layers)}")
        check("obsm keys equal", keys(o1.obsm) == keys(o2.obsm), f"{keys(o1.obsm)} vs {keys(o2.obsm)}")
        bad = [k for k in keys(o1.obsm) if k in keys(o2.obsm) and not same_matrix(o1.obsm[k], o2.obsm[k])]
        check(f"every obsm entry bit-identical ({', '.join(keys(o1.obsm))})", bad == [] and bool(keys(o1.obsm)),
              f"differ: {bad}")
        check("obsp keys and entries equal", keys(o1.obsp) == keys(o2.obsp)
              and all(same_matrix(o1.obsp[k], o2.obsp[k]) for k in keys(o1.obsp)), f"{keys(o1.obsp)}")
    else:
        check("both runs wrote an object", False)

    print("\n3 - the object already carries the key's name, equal to the batch: kept as it is")
    A3 = ad.read_h5ad(src)
    A3.obs[KEY] = A3.obs["library"].astype(str).values
    src3 = td / "with_key.h5ad"
    A3.write_h5ad(src3)
    rc3, log3 = integrate(src3, td / "r3", renamed, "--design-sample-col", KEY)
    check("integrate exits 0", rc3 == 0, log3[-1500:])
    rep3, o3 = read(td / "r3")
    if o3 is not None and KEY in o3.obs:
        back = ad.read_h5ad(src3).obs[KEY]
        check(f"the object's own {KEY!r} is kept, its values as given", o3.obs[KEY].astype(str).equals(back.astype(str)))
    else:
        check(f"the object's own {KEY!r} is kept", False, f"obs {list(o3.obs.columns) if o3 is not None else None}")
    dk3 = rep3.get("design_key") or {}
    check("report.json design_key: the object's own column, nothing added",
          dk3.get("column") == KEY and dk3.get("added") is False and "object's own" in str(dk3.get("how")),
          f"{rep3.get('design_key', 'absent')!r}")

    print("\n4 - a column of the key's name that differs from the batch: refused before anything runs")
    A4 = ad.read_h5ad(src)
    v = A4.obs["library"].astype(str).values.copy()
    v[:5] = "L2" if v[0] != "L2" else "L1"
    A4.obs[KEY] = v
    src4 = td / "conflict.h5ad"
    A4.write_h5ad(src4)
    rc4, log4 = integrate(src4, td / "r4", renamed, "--design-sample-col", KEY)
    check("integrate refuses: exit 2", rc4 == 2, f"exit {rc4}; {log4[-600:]}")
    check("the refusal names the column and the batch key, and the count",
          f"{KEY!r}" in log4 and "'library'" in log4 and "on 5 of" in log4, log4[-600:])
    check("nothing was computed: no object written", not (td / "r4" / "objects").exists()
          or not any((td / "r4" / "objects").iterdir()))

print()
print("=" * 64)
if FAILED:
    print(f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
    raise SystemExit(1)
print("the design's key column travels in the object, under the design's name")
