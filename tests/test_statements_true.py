#!/usr/bin/env python3
"""What this repository says about itself is true, and nothing in it is dead (GAP-037).

Every check here fails on a statement that WAS false or dead in this repository, read by an audit
(single-cell-harness autodev pass 0001) rather than found by a run:

  1. DEVPOINTS.yaml named tests/test_methods.py, which never existed
  2. a test ended its condition in `or True`, so it could not fail
  3. benchmark.score took a `cluster_seed` it never read - a parameter that looks like control
     and controls nothing; three more were found the same way
  4. `describe` declared confounding a gate that "vetoes any claim", while metrics.confounding
     classifies and vetoes nothing
  5. tests/run_all.py named an undefined variable on its red path, so the first red suite killed
     the runner before it said which suite was red
  6. (a guard, not a past defect) every documented command line still parses, now that
     --species is required - the README, docs/ and the usage text `--help` prints
  7. the README, the quickstart and `--help` ran `assess` and `integrate` into one --out, so a
     killed integrate left assess's report.json where `score` would rewrite it (GAP-C22)

Needs numpy, which `describe` imports; nothing heavier.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {str(detail)[:600]}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# ----------------------------------------------------------------------------------------- 1
print("\n1. DEVPOINTS names only tests that exist")
# Comments are dropped first: a comment may mention a file that is gone, and say so.
dev = "\n".join(ln.split("#", 1)[0] for ln in
                (ROOT / "DEVPOINTS.yaml").read_text(encoding="utf-8").splitlines())
named = sorted({t for block in re.findall(r"\btests:\s*\[(.*?)\]", dev, re.S)
                for t in re.findall(r"tests/[\w./-]+\.py", block)})
missing = [t for t in named if not (ROOT / t).is_file()]
check("DEVPOINTS.yaml names at least one test", bool(named))
check("every test DEVPOINTS.yaml names exists", not missing, missing)

# ----------------------------------------------------------------------------------------- 2
print("\n2. no test condition is short-circuited to true")


def _truthy_constant(n):
    return isinstance(n, ast.Constant) and bool(n.value) and not isinstance(n.value, str)


vacuous = []
for f in sorted((ROOT / "tests").glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) \
                and any(_truthy_constant(v) for v in node.values):
            vacuous.append(f"{f.name}:{node.lineno}")
check("no `... or True` (or any truthy constant) in a test", not vacuous, vacuous)

# ----------------------------------------------------------------------------------------- 3
print("\n3. no function takes a parameter it never reads")
cli_tree = ast.parse((ROOT / "scintegrate" / "cli.py").read_text(encoding="utf-8"))
# A command handler is called by argparse with the parsed arguments whether it reads them or not.
handlers = {kw.value.id for n in ast.walk(cli_tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "set_defaults"
            for kw in n.keywords if kw.arg == "fn" and isinstance(kw.value, ast.Name)}
dead = []
for f in sorted((ROOT / "scintegrate").glob("*.py")):
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name in handlers:
            continue
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.arg not in ("self", "cls") and not arg.arg.startswith("_") and arg.arg not in used:
                dead.append(f"{f.name}:{node.lineno} {node.name}({arg.arg})")
check("every parameter of every function is read", not dead, dead)
check("benchmark.score in particular takes no cluster_seed",
      "cluster_seed" not in (ROOT / "scintegrate" / "benchmark.py").read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------------------- 4
print("\n4. `describe` declares as a gate only what refuses")
import io  # noqa: E402
import json  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402

from scintegrate import cli  # noqa: E402

buf = io.StringIO()
with redirect_stdout(buf):
    cli.main(["describe"])
d = json.loads(buf.getvalue())
gates = d.get("gates") or {}
check("confounding is not declared a gate - it refuses nothing", "confounding" not in gates, gates)
check("no gate is described as a veto", not any("veto" in str(v).lower() for v in gates.values()), gates)
conf = (d.get("constraints") or {}).get("confounding") or ""
check("confounding is declared as what it is: classified, never vetoed",
      "CLASSIFIED" in conf and "never vetoed" in conf, conf)
mtext = (ROOT / "scintegrate" / "metrics.py").read_text(encoding="utf-8")
check("and metrics.py says the same", "`confounding` classifies; it does not veto" in mtext)
check("the species refusal is declared as the gate it is", "species" in gates, gates)
ax = d.get("axes") or {}
check("the axes are declared: species required, assay optional and recorded",
      (ax.get("species") or {}).get("required") is True
      and (ax.get("assay") or {}).get("required") is False
      and (ax.get("assay") or {}).get("absent_is_recorded_as") == "not declared", ax)

# ----------------------------------------------------------------------------------------- 5
print("\n5. the test runner reports a red suite, and every suite after it")
with tempfile.TemporaryDirectory() as td:
    t = Path(td) / "tests"
    t.mkdir()
    (t / "run_all.py").write_text((ROOT / "tests" / "run_all.py").read_text(encoding="utf-8"))
    (t / "test_a_red.py").write_text("import sys\nprint('deliberately red')\nsys.exit(1)\n")
    (t / "test_b_green.py").write_text("print('green')\n")
    r = subprocess.run([sys.executable, str(t / "run_all.py")], capture_output=True, text=True,
                       cwd=td, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    out = r.stdout + r.stderr
    check("a red suite exits 1, is named with its exit code, and the rest still run",
          r.returncode == 1 and "RED   test_a_red.py   exit=1" in out
          and "green test_b_green.py" in out and "2 suites   1 red" in out, out[-800:])
    check("and the runner does not die on its own red path", "Traceback" not in out, out[-800:])

# ----------------------------------------------------------------------------------------- 6
print("\n6. every documented command line parses")
ap = cli.parser()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _docs  # noqa: E402

DOCS = _docs.documents()
examples = [(name, [cmd, *argv]) for name, cmds, _ in DOCS for cmd, argv in cmds]
bad = []
for where, argv in examples:
    try:
        with redirect_stdout(io.StringIO()):
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()):
                ap.parse_args(argv)
    except SystemExit:
        bad.append(f"{where}: scintegrate {' '.join(argv)[:120]}")
check("the documentation carries command lines to check", len(examples) >= 5, len(examples))
check("each of them is accepted by the parser as written", not bad, bad)

# ----------------------------------------------------------------------------------------- 7
print("\n7. each command is documented into its own run directory (GAP-C22)")
# `assess` and `integrate` both write report.json, STATUS.json and the seal into --out. Documented
# into ONE directory, an integrate killed during scoring left assess's report.json there, and
# `score` then wrote the integration's ranking into it and rebuilt assessment.html from the mix.
shared, aimed = [], []
for name, cmds, _ in DOCS:
    outs = {c: {o.rstrip("/") for c_, argv in cmds if c_ == c for o in _docs.option(argv, "--out")}
            for c in ("assess", "integrate")}
    if outs["assess"] & outs["integrate"]:
        shared.append(f"{name}: {sorted(outs['assess'] & outs['integrate'])}")
    for c_, argv in cmds:
        if c_ != "score" or not outs["integrate"]:
            continue
        o = [x.rstrip("/") for x in _docs.option(argv, "--out")]
        h = _docs.option(argv, "--h5ad")
        if not (o and o[0] in outs["integrate"] and h and h[0].startswith(o[0] + "/")):
            aimed.append(f"{name}: score {' '.join(argv)[:110]}")
check("no document runs assess and integrate into the same --out", not shared, shared)
check("where a document runs integrate, its score command re-scores that run: --out is integrate's "
      "directory and --h5ad the object inside it", not aimed, aimed)

print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failing")
sys.exit(1 if FAILED else 0)
