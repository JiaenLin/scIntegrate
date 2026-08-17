"""What is installed, and what each absence costs you.

STDLIB ONLY, ON PURPOSE. `scintegrate doctor` has to run in the interpreter you are worried
about, which is by definition the one where something is missing. A diagnostic that needs numpy
to tell you numpy is missing is not a diagnostic.

The design principle this module serves: **a capability that cannot run is NAMED, never silently
skipped.** A method missing from a comparison changes what the comparison means, and a reader
cannot see the absence of a panel nobody drew. So every absence here carries the sentence that
belongs in the report.
"""
from __future__ import annotations
import importlib.util
import sys

# ---------------------------------------------------------------------------------------------
#: capability -> (packages required, what you lose without it, install hint)
#:
#: `core` and `read` are the only entries that stop the tool. Everything else degrades: the run
#: continues, the capability is named as absent in the report, and the numbers that depended on
#: it are absent rather than approximated. Substituting a weaker proxy silently is the failure
#: mode this table exists to prevent.
CAPABILITIES = {
    "core": {
        "packages": ["numpy", "sklearn"],
        "loses": "nothing runs: the kNN metrics are the tool's own floor",
        "hint": "pip install -e .",
        "fatal": True,
    },
    "read": {
        "packages": ["anndata", "scanpy"],
        "loses": "cannot read .h5ad at all",
        "hint": "pip install -e '.[run]'",
        "fatal": True,
    },
    "figures": {
        "packages": ["matplotlib"],
        "loses": "every figure becomes a named absence; the report is still written",
        "hint": "pip install -e '.[run]'",
        "fatal": False,
    },
    "harmony": {
        "packages": ["harmonypy"],
        "loses": "Harmony is not compared",
        "hint": "pip install -e '.[harmony]'",
        "fatal": False,
    },
    "bbknn": {
        "packages": ["bbknn"],
        "loses": "BBKNN is not compared",
        "hint": "pip install -e '.[bbknn]'",
        "fatal": False,
    },
    "scvi": {
        "packages": ["scvi", "torch"],
        "loses": "scVI and scANVI are not compared (both are scvi-tools models)",
        "hint": "pip install -e '.[scvi]'   # pulls torch, ~2.5 GB",
        "fatal": False,
    },
    "scib": {
        "packages": ["scib"],
        "loses": "the scIB benchmark is not computed and NO DEFAULT EMBEDDING IS CHOSEN; "
                 "the tool's own kNN metrics are still reported",
        "hint": "pip install -e '.[scib]'",
        "fatal": False,
    },
}

#: scIB metrics that need something beyond python, and are therefore expected to be absent.
#: Named here so the report can say "not computed, and why" instead of leaving a blank cell.
SCIB_OPTIONAL = {
    "kBET": ("rpy2 + the kBET R package", "batch"),
    "trajectory": ("a trajectory/pseudotime key, which this stage does not define", "bio"),
}


def _version(pkg):
    """Measured version, or None if the package is not importable.

    Deliberately does NOT import the module: importing torch costs seconds and importing scvi
    emits warnings, and `doctor` should be instant. importlib.metadata reads the installed
    distribution instead.
    """
    if importlib.util.find_spec(pkg) is None:
        return None
    import importlib.metadata as md
    # the import name and the distribution name differ for several of these
    for dist in (_DIST.get(pkg, pkg), pkg):
        try:
            return md.version(dist)
        except Exception:
            continue
    return "present"


#: import name -> distribution name, where they differ. Measured, not guessed: `import sklearn`
#: comes from `scikit-learn`, `import scvi` from `scvi-tools`.
_DIST = {"sklearn": "scikit-learn", "scvi": "scvi-tools"}


def probe():
    """{capability: {"ok": bool, "have": {pkg: version|None}, ...}} for this interpreter."""
    out = {}
    for name, spec in CAPABILITIES.items():
        have = {p: _version(p) for p in spec["packages"]}
        out[name] = {
            "ok": all(v is not None for v in have.values()),
            "have": have,
            "missing": [p for p, v in have.items() if v is None],
            "loses": spec["loses"],
            "hint": spec["hint"],
            "fatal": spec["fatal"],
        }
    return out


def gpu():
    """Is a GPU visible to torch? Reported because scVI on CPU is a different proposition.

    Returns a short human string, never raises: torch may be absent, or present and unable to
    talk to a driver, and both are things the user needs told rather than crashed on.
    """
    if importlib.util.find_spec("torch") is None:
        return "torch absent"
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return f"{n} CUDA device(s): " + ", ".join(
                torch.cuda.get_device_name(i) for i in range(n))
        return "torch present, no CUDA device visible (scVI/scANVI will run on CPU)"
    except Exception as e:                                  # a broken driver is not our crash
        return f"torch present, CUDA probe failed: {type(e).__name__}"


def doctor(argv_out=print):
    """Print the audit. Returns 0 if the tool can run at all, 2 if a fatal capability is absent."""
    p = probe()
    argv_out(f"python {sys.version.split()[0]}   {sys.executable}")
    argv_out("")
    width = max(len(k) for k in p)
    broken = []
    for name, r in p.items():
        mark = "ok  " if r["ok"] else ("MISS" if not r["fatal"] else "STOP")
        vers = "  ".join(f"{k} {v}" for k, v in r["have"].items() if v) or "-"
        argv_out(f"  {mark}  {name:<{width}}  {vers}")
        if not r["ok"]:
            argv_out(f"        missing: {', '.join(r['missing'])}")
            argv_out(f"        without it: {r['loses']}")
            argv_out(f"        fix: {r['hint']}")
            if r["fatal"]:
                broken.append(name)
    argv_out("")
    argv_out(f"  gpu: {gpu()}")
    argv_out("")
    argv_out("  scIB metrics that need more than python, and are reported as absent rather")
    argv_out("  than approximated:")
    for m, (needs, side) in SCIB_OPTIONAL.items():
        argv_out(f"    {m:12s} needs {needs}  [{side}]")
    argv_out("")
    if broken:
        argv_out(f"CANNOT RUN: {', '.join(broken)}. Everything else degrades to a named absence.")
        return 2
    degraded = [k for k, r in p.items() if not r["ok"]]
    if degraded:
        argv_out(f"Can run, with {len(degraded)} capability(ies) absent: {', '.join(degraded)}.")
        argv_out("Each will be named in the report. Nothing is silently substituted.")
    else:
        argv_out("Everything is present.")
    return 0
