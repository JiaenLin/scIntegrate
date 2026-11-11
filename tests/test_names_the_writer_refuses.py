#!/usr/bin/env python3
"""A name the object's writer refuses is refused BEFORE anything is computed; a name it takes is carried.

THE DEFECT (W046's verifier, with the real CLI). The design's key, every design column, and the
columns the other options name all become columns of the object's obs. A name the file format
forbids ('/' anywhere, '_index') was accepted at input and found by the writer at the write, after
every method had trained: exit 1, STATUS failed, and a half-written object (cells, no genes, no
embeddings) listed among the run's products. A factor named like the label key was appended twice,
which the writer refuses as a repeated name, at the same moment.

WHAT IT HOLDS THE TOOL TO, through `python -m scintegrate.cli integrate`, on tests/_cohort.py's
cohort (cells withheld with `--drop-labels EXCLUDED`):
  1. each forbidden name - the design's key `lib/run` and `_index`, a factor `donor/x` and `_index`,
     an empty column from a header ending in a comma, a factor named like `--label-key` holding other
     values, and `--object-name sub/x.h5ad` - exits 2 with the name in the refusal, BEFORE any
     method runs, with STATUS `refused`, nothing under objects/ and no .h5ad among its products;
  2. names the writer takes are carried exactly, not refused: a key with a leading underscore,
     factors with a space, a dot and non-ASCII letters, a design column named like `--batch-key`
     equal to it on every cell (carried once), an object name with a space; and a sentinel label
     and an asked method holding '/', which key dicts of uns, are kept as JSON text there;
  3. a failed write leaves no partial file under the object's name and leaves an earlier object
     as it was;
  4. the run records state_version 2 in describe, STATUS and the object's provenance.

    python tests/test_names_the_writer_refuses.py
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
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs anndata, numpy and pandas ({e}). A skip is not a pass.")
    raise SystemExit(0)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


from _cohort import DESIGN_COLUMNS, LIBRARIES, cohort, design  # noqa: E402


def run(*argv):
    r = subprocess.run([sys.executable, "-m", "scintegrate.cli", *map(str, argv)],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT), "TMPDIR": str(TMP)},
                       capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def integrate(src, out, des, *extra, methods="none,harmony"):
    return run("integrate", "--h5ad", src, "--out", out, "--batch-key", "library",
               "--label-key", "cell_type", "--species", "homo_sapiens", "--methods", methods,
               "--k", "10", "--n-pcs", "15", "--scib-resolutions", "3", "--drop-labels", "EXCLUDED",
               "--design", des, *extra)


def write_design(path, header, extra=None):
    """The cohort's design rows under another header; `extra` adds columns, one value per library."""
    design(path)
    rows = [ln.split(",") for ln in path.read_text(encoding="utf-8").splitlines()[1:]]
    head = list(header)
    for name, value in (extra or {}).items():
        head.append(name)
        for r in rows:
            r.append(value(r[0]))
    path.write_text("\n".join([",".join(head)] + [",".join(r) for r in rows]) + "\n", encoding="utf-8")


def status(out):
    p = out / "STATUS.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


TMP = Path(tempfile.mkdtemp(prefix="names-tmpdir-"))     # the runs' TMPDIR: the probe must not land here


def probes_left(out):
    return [p.name for p in list(out.glob(".scintegrate-names-*")) + list(TMP.glob("*scintegrate-names-*"))]


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src = td / "joint.h5ad"
    A0 = cohort(src)
    H = list(DESIGN_COLUMNS)                              # library, donor, condition, chemistry

    print("1. a name the writer refuses: exit 2 before any method runs, no object, none listed")
    cases = [
        ("the design's key 'lib/run'", ["lib/run"] + H[1:], None, ["--design-sample-col", "lib/run"],
         "lib/run"),
        ("the design's key '_index'", ["_index"] + H[1:], None, ["--design-sample-col", "_index"],
         "_index"),
        ("a factor 'donor/x'", [H[0], "donor/x"] + H[2:], None, [], "donor/x"),
        ("a factor '_index'", [H[0], "_index"] + H[2:], None, [], "_index"),
        ("an empty column (a header ending in a comma)", H, {"": lambda lib: ""}, [], "''"),
        ("a factor named like --label-key, other values", H, {"cell_type": lambda lib: "x" + lib},
         [], "cell_type"),
        ("--object-name 'sub/x.h5ad'", H, None, ["--object-name", "sub/x.h5ad"], "sub/x.h5ad"),
    ]
    for i, (what, header, extra, flags, shown) in enumerate(cases):
        des, out = td / f"design_{i}.csv", td / f"refused_{i}"
        write_design(des, header, extra)
        rc, log = integrate(src, out, des, *flags)
        st = status(out)
        listed = [p["path"] for p in st.get("products") or [] if p["path"].endswith(".h5ad")]
        check(f"{what}: exit 2, refused", rc == 2 and st.get("status") == "refused",
              f"exit {rc}, status {st.get('status')!r}: {log.strip().splitlines()[-1][:160]}")
        check(f"{what}: before any method ran", "=== none ===" not in log and "\nmethods:" not in log)
        check(f"{what}: the refusal names it", shown in log, log.strip().splitlines()[-1][:160])
        check(f"{what}: nothing under objects/, no .h5ad listed",
              not any((out / "objects").glob("*")) and not listed, f"listed {listed}")
        check(f"{what}: no probe left in the run or in TMPDIR", not probes_left(out), f"{probes_left(out)}")

    print("2. names the writer takes are carried, not refused")
    lab = A0.obs["cell_type"].astype(str).replace({"UNRESOLVED": "no/call"})
    A0.obs["cell_type"] = lab.values
    src2 = td / "joint_slash_sentinel.h5ad"
    A0.write_h5ad(src2)
    names = {"donor": "donor id", "condition": "condition.v2", "chemistry": "chémistry 库"}
    # 2b first: a design column named like --batch-key and equal to it is one column, not two.
    des, out = td / "design_batch_named.csv", td / "batch_named"
    write_design(des, H, {"library_copy": lambda lib: lib})
    des.write_text(des.read_text(encoding="utf-8").replace("library,", "lib0,", 1)
                   .replace("library_copy", "library"), encoding="utf-8")
    rc, log = integrate(src, out, des, "--design-sample-col", "lib0")
    check("a design column named like --batch-key, equal on every cell: exit 0", rc == 0,
          log.strip().splitlines()[-1][:200] if log.strip() else "")
    p = out / "objects" / "cohort_integrated.h5ad"
    cols = list(ad.read_h5ad(p).obs.columns) if p.exists() else []
    check("it is carried once", cols.count("library") == 1 and cols[:2] == ["library", "cell_type"],
          f"{cols}")
    # 2a: the rest, with no column named twice, so each case here fails on its own reason.
    des, out = td / "design_ok.csv", td / "accepted"
    write_design(des, ["_lib"] + [names[c] for c in H[1:]])
    rc, log = integrate(src2, out, des, "--design-sample-col", "_lib", "--object-name", "the object.h5ad",
                        "--label-sentinel", "no/call", "--label-sentinel", "EXCLUDED",
                        methods="none,harmony,a/b")
    check("accepted names: exit 0", rc == 0, log.strip().splitlines()[-1][:200] if log.strip() else "")
    p = out / "objects" / "the object.h5ad"
    obj = ad.read_h5ad(p) if p.exists() else None
    cols = list(obj.obs.columns) if obj is not None else []
    want = ["library", "cell_type", "donor id", "condition.v2", "chémistry 库", "_lib"]
    check("obs carries every name exactly, in order", cols == want, f"{cols}")
    if obj is not None and "_lib" in cols:
        check("the key '_lib' equals the batch on every cell, the withheld included",
              bool((obj.obs["_lib"].astype(str) == obj.obs["library"].astype(str)).all()))
    prov = ((obj.uns.get("scintegrate") or {}).get("provenance") or {}) if obj is not None else {}
    sent = prov.get("sentinels")
    check("a sentinel holding '/' is kept in provenance as JSON text",
          isinstance(sent, str) and "no/call" in json.loads(sent), f"{sent!r}")
    absent = prov.get("methods_absent")
    check("an asked method holding '/' is kept as JSON text", isinstance(absent, str) and "a/b" in
          json.loads(absent), f"{absent!r}")
    st = status(out)
    check("no probe left in the run or in TMPDIR", not probes_left(out), f"{probes_left(out)}")
    check("STATUS lists the object under its name", "objects/the object.h5ad" in
          [x["path"] for x in st.get("products") or []])

    print("3. a failed write leaves no partial file and the earlier object as it was")
    from scintegrate import emit  # noqa: E402
    d = td / "atomic"
    d.mkdir()
    good = ad.AnnData(X=np.zeros((2, 1), dtype="float32"), obs=pd.DataFrame(index=["c0", "c1"]))
    target = d / "o.h5ad"
    emit.write_h5ad(good, target)
    before = target.read_bytes()
    bad = ad.AnnData(X=np.zeros((2, 1), dtype="float32"),
                     obs=pd.DataFrame({"x/y": ["a", "b"]}, index=["c0", "c1"]))
    try:
        emit.write_h5ad(bad, target)
        raised = False
    except Exception:                                                     # noqa: BLE001
        raised = True
    check("the writer refused it", raised)
    check("the earlier object is byte for byte as it was", target.read_bytes() == before)
    check("no partial file is left beside it", sorted(x.name for x in d.iterdir()) == ["o.h5ad"],
          f"{sorted(x.name for x in d.iterdir())}")

    print("4. state_version 2, said the same way everywhere")
    rc, dj = run("describe")
    try:
        sv = json.loads(dj[dj.index("{"):]).get("state_version")
    except ValueError:
        sv = None
    check("describe: state_version 2", sv == 2, f"{sv!r}")
    check("STATUS: state_version 2", st.get("state_version") == 2, f"{st.get('state_version')!r}")
    check("the object's provenance: state_version 2", prov.get("state_version") == 2,
          f"{prov.get('state_version')!r}")

import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
print("\n" + ("PASS" if not FAILED else f"FAIL ({len(FAILED)}): " + "; ".join(FAILED)))
raise SystemExit(1 if FAILED else 0)
