"""The version is written in three places and they must agree.

A sibling tool's pyproject said 0.1.0 while its VERSION, its __init__ and its citation file said
0.3.0, and nothing noticed because each file is read by a different consumer: pip reads one,
`--version` reads another, a reader cites the third. Drift between them is invisible until
somebody tries to reproduce a result from a version that never existed.

And the number must move when the interface breaks (GAP-C20): a command line an earlier release
documented, and today's parser refuses, means the version moved past that release and the README
and KNOWN_ISSUES say which flag did it.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(n, c, d=""):
    print(f"  {'ok  ' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


from scintegrate import __version__                                             # noqa: E402

vfile = (ROOT / "VERSION").read_text().strip()
pyproj = (ROOT / "pyproject.toml").read_text()

check("VERSION matches __version__", vfile == __version__, f"{vfile} vs {__version__}")
check("pyproject reads the version from the file rather than restating it",
      'version = { file = "VERSION" }' in pyproj)
check("pyproject does not ALSO hardcode one",
      not re.search(r'^version\s*=\s*"', pyproj, re.M))
check("the version is a plain three-part number", re.fullmatch(r"\d+\.\d+\.\d+", vfile) is not None,
      vfile)
cff = re.search(r"^version:\s*([\d.]+)", (ROOT / "CITATION.cff").read_text(), re.M)
check("CITATION.cff carries the same version", cff is not None and cff.group(1) == vfile,
      f"{cff.group(1) if cff else None} vs {vfile}")

# every extra named in the install docs must exist as an extra, or the documented command fails
block = pyproj.split("[project.optional-dependencies]")[1].split("\n[")[0]
extras = set(re.findall(r"^(\w+) = \[", block, re.M))
readme = (ROOT / "README.md").read_text()
documented = set(re.findall(r"pip install -e '\.\[([a-z,]+)\]'", readme))
named = {e for group in documented for e in group.split(",")}
check("every extra the README tells you to install exists",
      named <= extras, f"documented {sorted(named)} vs declared {sorted(extras)}")
check("scvi and scib are separate extras, since one pulls torch and the other does not",
      {"scvi", "scib"} <= extras)

# ------------------------------------------------- a command line that stops parsing moves the version
# GAP-C20. `--species` became required while VERSION stayed 0.4.0, so every command line written for
# 0.4.0 stopped at argparse - exit 2 - under a version that said nothing had changed, and a reader had
# no number to tell the two interfaces apart by. What each release told a user to type is held here,
# from that release's own README, and parsed by today's parser.
import contextlib  # noqa: E402
import io  # noqa: E402
import shlex  # noqa: E402

from scintegrate import cli  # noqa: E402

SHIPPED = {
    # `git show 722bc79:README.md`, its "## Run" block: the last commit at 0.4.0 before --species.
    "0.4.0": [
        "assess --h5ad joint.h5ad --out results/ --batch-key sample --label-key cell_type "
        "--design design.csv",
        "integrate --h5ad joint.h5ad --out results/ --batch-key sample --label-key cell_type "
        "--design design.csv --colour-by cell_type,cell_type_forced,cell_compartment "
        "--methods none,harmony,bbknn,scvi,scanvi",
        "score --h5ad results/objects/cohort_integrated.h5ad --out results/ --batch-key sample "
        "--label-key cell_type",
        # `git show 722bc79:docs/PBSPRO.md`, its lever block: the integrate line above with the two
        # levers that block told a user to add. 0.5.0 removed `--lisi-subsample` (GAP-C67).
        "integrate --h5ad joint.h5ad --out results/ --batch-key sample --label-key cell_type "
        "--design design.csv --colour-by cell_type,cell_type_forced,cell_compartment "
        "--methods none,harmony,bbknn,scvi,scanvi --scib-resolutions 7 --lisi-subsample 50",
    ],
}


def parts(v):
    return tuple(int(x) for x in v.split("."))


def breaks_past(v, release):
    """Semantic versioning: below 1.0.0 the MINOR number is the one a breaking change moves, from
    1.0.0 the major."""
    i = 0 if parts(release)[0] >= 1 else 1
    return parts(v)[:i + 1] > parts(release)[:i + 1]


def stops(line):
    """(exit code, the flags argparse's error names) when today's parser refuses a line, else None."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            cli.parser().parse_args(shlex.split(line))
    except SystemExit as e:
        said = (err.getvalue().strip().splitlines() or [""])[-1]
        return e.code, re.findall(r"--[a-z][\w-]*", said.split("error:", 1)[-1])
    return None


for release, lines in SHIPPED.items():
    stopped = {ln: s for ln in lines if (s := stops(ln))}
    flags = sorted({f for _, fl in stopped.values() for f in fl})
    if not stopped:
        check(f"every command line {release} documented still parses", True)
        continue
    check(f"a {release} command line that no longer parses stops at argparse with exit 2, naming "
          f"what it lacks", all(code == 2 for code, _ in stopped.values()) and bool(flags),
          stopped)
    check(f"so VERSION has moved past {release} in the number a breaking change moves",
          breaks_past(vfile, release), f"VERSION {vfile}; refused: {flags}")
    for doc in ("README.md", "KNOWN_ISSUES.md"):
        paras = re.split(r"\n\s*\n", (ROOT / doc).read_text(encoding="utf-8"))
        said = [f for f in flags if any(
            f in p and any(breaks_past(v, release) and parts(v) <= parts(vfile)
                           for v in re.findall(r"\b\d+\.\d+\.\d+\b", p)) for p in paras)]
        check(f"{doc} names the version from which {', '.join(flags)} changed",
              said == flags, f"named: {said} of {flags}")

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    raise SystemExit(1)
print("version OK — one number, three files, no drift")
