#!/usr/bin/env python3
"""Every example design in this repository is written in names the repository declares itself.

WHY THIS EXISTS (GAP-C21). The tool was built against one cohort, and its examples drifted into that
cohort's design: the example factor in `metrics.confounding`'s docstring, in docs/DESIGN.md and in
the design rows test_design.py builds was that cohort's own factor, and a content check in
test_methods_settings.py spelled that cohort's cell types in order to forbid them. None of it was on
a word list - the lists hold sample names and site strings, not a study's vocabulary - so the leak
guard passed over all of it.

A word list cannot be the fix from inside the repository: the guard would have to spell what it
guards against. So this holds the positive rule. The repository's examples use the names it
DECLARES: the columns of the design tables its own documents show, and of the synthetic fixture
(tests/_cohort.py, DESIGN_COLUMNS). A study's factor can then only become an example by being put in
a table on the page, where a reviewer reads it.

Three places carry a design factor in a form a parser can find, and each is held:
  (a) the `factors` of a design row a test hands to the assessment (`summarise` and friends);
  (b) a name in backticks in the docstring of a function that takes design factors - it must be one
      of the function's own parameters, or a declared name;
  (c) each `--bio-factor` in a documented command - it must be a column of the design table that
      same document shows (or, where the document shows none, a declared name). Otherwise the
      command is refused as written: `--bio-factor ... not in design.csv`.

WHAT IT CANNOT SEE: a factor named in running prose. docs/DESIGN.md and the package's prose were
read by hand for GAP-C21; the word lists generated from each cohort's design (single-cell-harness
GAP-008) are the check for prose. Stdlib only.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import _docs  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# ------------------------------------------------------------------------- the declared names
def fixture_columns():
    """DESIGN_COLUMNS, read from tests/_cohort.py without importing it (it imports anndata)."""
    tree = ast.parse((HERE / "_cohort.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DESIGN_COLUMNS"
                                                for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


DOCS = _docs.documents()
fixture = fixture_columns()
tables = {name: [c for t in tabs for c in t] for name, _, tabs in DOCS}
DECLARED = set(fixture) | {c for cols in tables.values() for c in cols}

print("\nthe names this repository declares for its examples")
check("the synthetic fixture declares its design columns", bool(fixture), fixture)
check("and the documents show at least one design table", any(tables.values()), tables)
print(f"       declared: {sorted(DECLARED)}")

# --------------------------------------------------------------------- (a) design rows in tests
print("\n(a) the factors of every design row a test builds are declared names")
undeclared = []
for f in sorted(HERE.glob("*.py")):
    if f.name == Path(__file__).name:
        continue
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == "factors" and isinstance(v, ast.Dict):
                for fk in v.keys:
                    if isinstance(fk, ast.Constant) and fk.value not in DECLARED:
                        undeclared.append(f"{f.name}:{fk.lineno} {fk.value!r}")
check("every factor a design row names is a declared column", not undeclared, undeclared)

# -------------------------------------------------- (b) docstrings of functions that take factors
print("\n(b) a function that takes design factors explains itself in its own or declared names")
FACTOR_PARAMS = {"factors", "bio_factors"}
loose, seen = [], 0
for f in sorted((ROOT / "scintegrate").glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        if not params & FACTOR_PARAMS:
            continue
        seen += 1
        for name in re.findall(r"`([A-Za-z_]\w*)`", ast.get_docstring(node) or ""):
            if name not in params and name not in DECLARED:
                loose.append(f"{f.name}:{node.lineno} {node.name}: `{name}`")
check("the package has functions that take design factors", seen > 0, seen)
check("every name their docstrings put in backticks is a parameter or a declared column",
      not loose, loose)

# ------------------------------------------------ (c) documented --bio-factor against its table
print("\n(c) every documented --bio-factor is a column of the design table it is run against")
refused = []
for name, cmds, _ in DOCS:
    own = set(tables.get(name) or [])
    for cmd, argv in cmds:
        for fac in _docs.option(argv, "--bio-factor"):
            if fac not in (own or DECLARED):
                refused.append(f"{name}: {cmd} --bio-factor {fac} (the table there offers "
                               f"{sorted(own) or 'nothing'})")
check("no documented command names a factor its design table does not have", not refused, refused)

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
