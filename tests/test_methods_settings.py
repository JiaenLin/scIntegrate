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


def test_report_fields_exist_in_payload():
    """Anything the report renders must be in the dict report.json is built from.

    Caught in a real run: colour_by and umap_n_neighbors were added to the provenance dict that
    goes into the OBJECT's uns, while report.json is built from a different dict a few lines
    below. The figures were drawn correctly; the record of how they were drawn rendered as None.
    A report field reading `None` looks like a value that was not set, not like a plumbing bug,
    which is why it survived a full run.
    """
    import re
    print("\nreport fields reach report.json")
    src = (ROOT / "scintegrate" / "cli.py").read_text()
    rep = (ROOT / "scintegrate" / "report.py").read_text()

    # every payload.get("x") the report reads
    read = set(re.findall(r'payload\.get\(["\'](\w+)["\']', rep))
    # the integrate payload literal
    start = src.index('payload = {"command": "integrate"')
    body = src[start:src.index("_write_json(out, payload)", start)]
    written = set(re.findall(r'"(\w+)":', body))
    # A key may legitimately be unwritten if it is only ever read as a FALLBACK -
    # `payload.get("input") or payload.get("h5ad", "")`. Treat such a chain as satisfied when any
    # member of it is written, so the check tests reachability rather than spelling.
    for chain in re.findall(r'payload\.get\([^)]*\)(?:\s*or\s*payload\.get\([^)]*\))+', rep):
        keys = set(re.findall(r'payload\.get\(["\'](\w+)["\']', chain))
        if keys & written:
            read -= keys
    missing = sorted(read - written - {"command", "generated", "tool", "version"})
    check("every field the report reads is written", not missing, f"missing: {missing}")
    for k in ("umap_n_neighbors", "colour_by", "method_settings"):
        check(f"{k} is in the integrate payload", k in written)


def test_figure_defects():
    """The five defects the stage-3 panels shipped with, each asserted against the source.

    They were all found and fixed once in a sibling tool and never carried back here, which is why
    they are pinned rather than trusted: a fix that lives in one repository is a fix that will be
    re-lost in the next.
    """
    print("\nfigure defects")
    f = _code(ROOT / "scintegrate" / "figures.py")
    c = _code(ROOT / "scintegrate" / "cli.py")

    # subplots_adjust sets ONLY what it is given; bottom/top alone leaves 22.5% of the width blank
    import re
    for call in re.findall(r"subplots_adjust\(([^)]*)\)", f):
        check("subplots_adjust sets left and right", "left=" in call and "right=" in call, call)

    check("sentinels are held out of the palette", "def palette" in f and "sentinels" in f)
    check("a sentinel is labelled, not just greyed", "not a cell type" in f)
    check("no tab20 anywhere", "tab20" not in f and "tab20" not in c,
          "tab20 is neither colourblind-safe nor long enough for a real cohort")
    check("colourblind-safe palette", "OKABE_ITO" in f)
    check("embedding axes are named", "UMAP 1" in f)
    check("vector output with live text", 'with_suffix(".pdf")' in f and '"pdf.fonttype": 42' in f)


def test_withheld_padding_is_a_copy():
    """The NaN padding must not mutate the results the metrics still have to read.

    The object is written BEFORE scoring on purpose, so anything mutated at that point reaches the
    kNN metrics, scIB, the second scoring pass and the figures. Padding in place handed them
    full-length NaN arrays while the label and batch vectors were still at fit length, and sklearn
    refused the NaN three hours into a run - after the object had already been written.
    """
    print("\nwithheld cells")
    src = (ROOT / "scintegrate" / "cli.py").read_text()
    fn = src[src.index("def _restore_withheld"):src.index("def _integrate")]
    check("padding builds a new list", "wide_results" in fn and "dict(r)" in fn)
    check("nothing is written back into the caller's results", "        r[k] = " not in fn)
    check("callers keep the fit-length results", "A_out, results_wide = _restore_withheld" in src)
    check("emit gets the padded copy", "emit.build(A_out, results_wide" in src)


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
    test_figure_defects()
    test_withheld_padding_is_a_copy()
    test_report_fields_exist_in_payload()

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("every documented setting matches the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
