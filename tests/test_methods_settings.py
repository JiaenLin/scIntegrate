#!/usr/bin/env python3
"""The settings each method is run at, held to what docs/METHODS.md says.

A benchmark whose parameters live only in source code is one nobody can reproduce or challenge,
and a document describing parameters that have since changed is worse than none. These checks read
the SOURCE rather than run the methods, so they need none of the heavy packages installed - which
is the only way they can run anywhere.

    python3 tests/test_methods_settings.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond
                                                      else ""))
    if not cond:
        FAILED.append(name)


def _code(path):
    """Source with comment-only lines stripped, so a check cannot pass on its own explanation."""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        out.append(ln)
    return "\n".join(out)


def test_no_project_data():
    """The repo is public. Nothing from any particular dataset belongs in it.

    Checked because it happened: a worked example in QUICKSTART.md carried a real cell type and
    a real cell count from the study the tool was being built against. Example output has to be
    the SHAPE of a table, never a result.
    """
    import re
    print("\nno dataset-specific content")
    bad = []
    pat = re.compile(r"cardiomyo|matrifibro|endocardial|pericyte|celescope|cellbender"
                     r"|\bsambo\b|wangyb|duke-nus", re.I)
    for f in list(ROOT.glob("docs/*.md")) + list(ROOT.glob("*.md")) \
            + list(ROOT.glob("scintegrate/*.py")):
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(ln):
                bad.append(f"{f.name}:{i}")
    check("no cell types or tools from a particular dataset", not bad, ", ".join(bad[:5]))


def main():
    m = _code(ROOT / "scintegrate" / "methods.py")
    f = _code(ROOT / "scintegrate" / "figures.py")
    b = _code(ROOT / "scintegrate" / "benchmark.py")
    doc = (ROOT / "docs" / "METHODS.md").read_text(encoding="utf-8")

    print("\nharmony")
    check("random_state is the run's seed, not harmonypy's fixed default",
          "random_state=seed" in m,
          "unpassed, --seed changed every method EXCEPT harmony and nothing said so")
    check("max_iter_harmony is the documented 20", "max_iter_harmony=20" in m)
    check("the deviation from harmonypy's 10 is written down",
          "max_iter_harmony" in doc and "10" in doc)

    print("\nshared inputs")
    check("a PC shortfall is reported, not silently sliced", "used_pcs" in m and
          "pcs_available" in m)
    check("every method slices the SAME used_pcs", m.count("[:, :used_pcs]") >= 3,
          f"{m.count('[:, :used_pcs]')} sites")
    check("no method still slices the requested n_pcs", "[:, :n_pcs]" not in m)

    print("\nscanvi: trained on what it is scored against")
    check("methods.py records that it uses labels", "USES_LABELS" in m)
    check("benchmark declares which methods are supervised", "LABEL_SUPERVISED" in b)
    check("a caveat is produced for the report", "def supervision_caveat" in b)
    check("scanvi is in the supervised list", re.search(r'LABEL_SUPERVISED\s*=\s*\("scanvi"', b)
          is not None)
    check("scvi is NOT in it (it never sees labels)",
          not re.search(r'LABEL_SUPERVISED\s*=\s*\([^)]*"scvi"[,)]', b))
    check("the document says so too", "trained on the label column" in doc.lower())
    check("scanvi fine-tuning epochs are exposed", "scanvi_max_epochs" in m)
    check("n_samples_per_label matches the tutorial", "n_samples_per_label=100" in m)

    print("\nscvi")
    check("counts come from a layer, never X", "layer=counts_layer" in m)
    check("batch is the only covariate", "batch_key=batch_key" in m)

    print("\nbbknn")
    check("scored as a graph, not as an embedding", '"kind": "graph"' in m)
    check("neighbors_within_batch is passed explicitly", "neighbors_within_batch=nwb" in m)
    check("its edge count is reported", "edges per cell" in m)

    print("\nUMAP is scanpy's, with stated parameters")
    check("neighbours built on the embedding as X", 'use_rep="X"' in f)
    check("n_neighbors is explicit, not inherited", "n_neighbors=n_neighbors" in f
          and "N_NEIGHBORS = 15" in f)
    check("min_dist is explicit", "min_dist=min_dist" in f)
    check("no dummy zero matrix hiding the real coordinates", "np.zeros" not in
          f.split("def _umap")[1].split("def ")[1] if "def _umap" in f else True)
    check("every sc.tl.umap call sets min_dist",
          f.count("sc.tl.umap(") == f.count("min_dist=min_dist"),
          f"{f.count('sc.tl.umap(')} calls, {f.count('min_dist=min_dist')} set it")

    test_no_project_data()

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("every documented setting matches the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
