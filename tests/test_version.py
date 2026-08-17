"""The version is written in three places and they must agree.

A sibling tool's pyproject said 0.1.0 while its VERSION, its __init__ and its citation file said
0.3.0, and nothing noticed because each file is read by a different consumer: pip reads one,
`--version` reads another, a reader cites the third. Drift between them is invisible until
somebody tries to reproduce a result from a version that never existed.
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

# every extra named in the install docs must exist as an extra, or the documented command fails
extras = set(re.findall(r"^(\w+) = \[", pyproj.split("[project.optional-dependencies]")[1], re.M))
readme = (ROOT / "README.md").read_text()
documented = set(re.findall(r"pip install -e '\.\[([a-z,]+)\]'", readme))
named = {e for group in documented for e in group.split(",")}
check("every extra the README tells you to install exists",
      named <= extras, f"documented {sorted(named)} vs declared {sorted(extras)}")
check("scvi and scib are separate extras, since one pulls torch and the other does not",
      {"scvi", "scib"} <= extras)

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    raise SystemExit(1)
print("version OK — one number, three files, no drift")
