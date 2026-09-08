"""The gate. One subprocess per suite, and the exit code decides.

WHY THIS FILE EXISTS. Until it did, running these suites meant a shell loop typed at a prompt,
or `pytest`, which is not installed in the environment this tool actually runs in. A gate that
depends on what the last person happened to type is not a gate, and a gate that reports green
because a runner found no tests to run is worse than none. Both were true here until 2026-09-06,
when `sch dev check` asked this repository how to run its tests and there was no answer.

WHY A SUBPROCESS EACH, AND NOT AN IMPORT. Every suite here is a module-scope script that ends in
a non-zero exit when it fails. A runner that IMPORTS them has to catch `SystemExit`, which
inherits from `BaseException` and escapes a bare `except Exception`; a runner that stops on the
first one exits with THAT suite's code and hides every file sorted after it. A subprocess cannot
do this - its exit code is a fact about that file and nothing else.

WHY `unittest discover` IS NOT USED. These suites define no TestCase, so discovery would find
zero tests and exit 0. A check that never ran, reported as a check that succeeded, is the exact
defect this project exists to catch.

    python tests/run_all.py             every suite
    python tests/run_all.py -k design   only files whose name contains 'design'

VERIFY IT BY MAKING IT FAIL. Drop a file that exits non-zero into tests/ and confirm this reports
RED. A gate that has never been seen to fail proves nothing about the runs it passed.

WHY `--jobs` EXISTS, AND WHY IT DEFAULTS TO 1

One subprocess per suite is the point of this runner and is not negotiable: an exit code is then
a fact about one file. But ISOLATION AND SERIALISATION ARE INDEPENDENT, and this ran them one at
a time. Each subprocess re-imports the whole stack, so the wall clock is dominated by the same
imports repeated once per suite.

`--jobs N` runs N of those subprocesses at once. Each is still its own process with its own exit
code, and results are collected and reported in FILE ORDER rather than completion order, so the
report is identical to the serial one. It defaults to 1 because suites sharing a temporary path
would collide, and that is a property of the suites rather than of the runner - the default may
only be raised for a suite set MEASURED to give the same result both ways.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pat = argv[argv.index("-k") + 1] if "-k" in argv else None
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1

    files = sorted(p for p in HERE.glob("test_*.py") if pat is None or pat in p.name)
    if not files:
        print(f"run_all: no suites matched {pat!r}", file=sys.stderr)
        return 2

    # The tool is imported from the checkout, not from whatever is installed.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    failed, skipped = [], []

    def one(f):
        r = subprocess.run([sys.executable, str(f)], cwd=str(ROOT), env=env,
                           capture_output=True, text=True)
        return f, r.returncode, (r.stdout or "") + (r.stderr or "")

    if jobs > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(one, files))     # file order, not completion order
    else:
        outcomes = [one(f) for f in files]
    for f, code, out in outcomes:
        # A SKIP is not a PASS, and it is reported on its own line so that a missing dependency
        # cannot be read as a suite that ran.
        if "SKIP" in out:
            skipped.append(f.name)
        if code != 0:
            failed.append(f.name)
            print(f"RED   {f.name}   exit={r.returncode}")
            print("".join(f"      {ln}\n" for ln in out.strip().splitlines()[-12:]))
        else:
            print(f"green {f.name}" + ("   (contains a SKIP)" if "SKIP" in out else ""))

    print("=" * 64)
    print(f"{len(files)} suites   {len(failed)} red   {len(skipped)} contain a skip")
    if skipped:
        print("a skip is NOT a pass: " + ", ".join(skipped))
    if failed:
        print("RED: " + ", ".join(failed))
        return 1
    print("all suites green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
