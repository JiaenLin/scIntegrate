#!/usr/bin/env python3
"""Exercise the colour-column path, because parsing it is not running it.

This exists because of a specific failure. `--colour-by` was added, the file parsed, `--help`
rendered the new flag, the whole suite passed - and the run died 33 seconds in with a NameError
on a helper that does not exist in this package. Every check that had been run touched the module
at IMPORT time or through argparse; none of them entered the function.

Needs numpy and anndata, so it does not run everywhere. That is the point: the checks that run
everywhere could not have caught this.

    python3 tests/test_colour_columns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail and not cond
                                                      else ""))
    if not cond:
        FAILED.append(name)


def main():
    import numpy as np
    import pandas as pd
    from scintegrate import inputs

    obs = pd.DataFrame({
        "sample": pd.Categorical(["a", "b"] * 6),
        "fine": pd.Categorical(["X", "Y", "Z"] * 4),
        "fine_forced": pd.Categorical(["X", "Y", "Z"] * 4),
        "coarse": pd.Categorical(["P", "Q"] * 6),
        "coarse_forced": pd.Categorical(["P", "Q"] * 6),
    })

    print("\nexplicit --colour-by")
    cols, why = inputs.colour_columns(
        obs, ["fine", "fine_forced", "coarse", "coarse_forced"], "sample", "fine", "coarse")
    check("all four columns kept, in order", cols == ["fine", "fine_forced", "coarse",
                                                      "coarse_forced"], str(cols))
    check("each records why it was chosen", all(c in why for c in cols))

    print("\ndefault when --colour-by is absent")
    cols2, why2 = inputs.colour_columns(obs, [], "sample", "fine", "coarse")
    check("falls back to the l1 and label keys", cols2 == ["coarse", "fine"], str(cols2))
    check("the fallback says how to draw more",
          any("--colour-by" in v for v in why2.values()), str(why2))

    print("\na column that is not there")
    try:
        inputs.colour_columns(obs, ["nope"], "sample", "fine", "coarse")
        check("refuses a column that does not exist", False, "it was accepted")
    except inputs.Refuse as e:
        check("refuses a column that does not exist", True)
        check("names the offender and lists what IS present",
              "nope" in str(e) and "fine" in str(e), str(e)[:90])

    print("\nthe CLI's own splitting, as _load does it")
    # Reproduced verbatim from cli._load. A helper borrowed from a sibling package is exactly how
    # the NameError got in, so this asserts the expression rather than importing one.
    raw = "fine, fine_forced ,coarse,"
    got = [c.strip() for c in (raw or "").split(",") if c.strip()]
    check("splits, strips and drops empties", got == ["fine", "fine_forced", "coarse"], str(got))
    src = (Path(__file__).resolve().parents[1] / "scintegrate" / "cli.py").read_text()
    check("cli.py calls no undefined _split helper", "_split(" not in src)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("the colour-column path runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
