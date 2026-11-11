#!/usr/bin/env bash
# Create the environment scIntegrate runs in, or report what is missing from the one you have.
#
#   setup/install_env.sh --check                        # audit the CURRENT interpreter, change nothing
#   setup/install_env.sh --check --python /path/to/python
#   setup/install_env.sh --check --strict --python P    # also refuse what the lock does not pin
#   setup/install_env.sh --prefix ~/envs/scintegrate    # create it from the lock, then check it strictly
#   setup/install_env.sh --prefix P --no-editable       # and write nothing into this checkout
#   setup/install_env.sh --add-to /path/to/python       # add the heavy methods to an EXISTING env
#
# scIntegrate's measurement layer is numpy + scikit-learn. This script exists for the two halves
# that `pip install -e '.[run]'` assumes you already have somewhere: the analysis stack, and the
# method packages - one of which brings torch.
#
# --add-to is here because the usual situation is not a fresh environment. It is an environment
# that already produced upstream results, into which you now need scvi-tools and scib WITHOUT
# disturbing anything those results depend on. It resolves first, prints exactly which existing
# packages a real install would change, and stops if any would.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
LOCK="$HERE/environment.lock.yml"
PREFIX=""; MODE="create"; PY=""; ADD=""; STRICT=0; EDITABLE=1

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do case "$1" in
  --prefix) PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
  --check)  MODE="check"; shift ;;
  --add-to) MODE="add"; ADD="${2:?--add-to needs a python path}"; shift 2 ;;
  --python) PY="${2:?--python needs a path}"; shift 2 ;;
  --strict) STRICT=1; shift ;;
  --no-editable) EDITABLE=0; shift ;;
  -h|--help) usage 0 ;;
  *) echo "unknown option: $1" >&2; usage 2 ;;
esac; done

# ------------------------------------------------------------------ check: report, never fix
# --strict is what a build is held to: the locked interpreter, every pin at its version, and no
# distribution the lock does not pin except the manager's pip, setuptools and wheel (when the lock
# does not pin them) and scIntegrate itself. Every build is checked this way before it says done.
check() {
  "$1" - "$LOCK" "$2" <<'PYEOF'
import sys, re, pathlib
import importlib.metadata as md
lock = pathlib.Path(sys.argv[1]).read_text()
strict = sys.argv[2] == "1"
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
want = dict(re.findall(r'^\s+- ([A-Za-z0-9_.-]+)==([0-9][^\s]*)', lock, re.M))
pyv = re.search(r'python=([0-9.]+)', lock)
have_py = sys.version.split()[0]
print(f"  python {have_py}" + (f"   (locked {pyv.group(1)})" if pyv else ""))
missing, differ, ok = [], [], 0
for p, v in want.items():
    try:
        got = md.version(p)
    except Exception:
        missing.append(p); continue
    if got == v: ok += 1
    else: differ.append((p, got, v))
print(f"  {ok} package(s) at the locked version, of {len(want)} locked")
for p, got, v in differ:
    print(f"  DIFFERS  {p:<18} have {got:<12} locked {v}")
for p in missing:
    print(f"  MISSING  {p}")
locked = {norm(p) for p in want}
manager = {"pip", "setuptools", "wheel"} - locked
present = {norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
stray = sorted(f"{n}=={v}" for n, v in present.items() if n not in locked | manager | {"scintegrate"})
if strict:
    print("  outside the lock: " + (", ".join(stray) if stray else "nothing") + "   (the manager's: "
          + (", ".join(f"{n}=={present[n]}" for n in sorted(manager & set(present))) or "none") + ")")

# grouped by what the absence actually costs, which is the only useful grouping
core   = [p for p in missing if p in ("numpy", "scikit-learn")]
read   = [p for p in missing if p in ("anndata", "scanpy", "scipy", "pandas", "h5py")]
figs   = [p for p in missing if p in ("matplotlib", "seaborn")]
bench  = [p for p in missing if p in ("scib", "scikit-misc", "igraph", "leidenalg")]
models = [p for p in missing if p in ("scvi-tools", "torch", "lightning", "pytorch-lightning",
                                      "torchmetrics", "pyro-ppl", "harmonypy", "bbknn")]
print()
if core:
    print("  NOTHING RUNS without: " + ", ".join(core))
    print("  -> pip install -e .")
if read:
    print("  Cannot READ .h5ad without: " + ", ".join(read))
    print("  -> pip install -e '.[run]'")
if figs:
    print("  Figures become NAMED ABSENCES without: " + ", ".join(figs))
    print("     The report is still written. A degradation, not a failure.")
if bench:
    print("  NO DEFAULT EMBEDDING IS CHOSEN without: " + ", ".join(bench))
    print("     The choice is defined as the scIB total, so without scib there is no total.")
    print("     The tool's own kNN metrics are still reported.  -> pip install -e '.[scib]'")
if models:
    print("  Method(s) will be named NOT COMPARED without: " + ", ".join(models))
    print("     -> pip install -e '.[harmony]' / '.[bbknn]' / '.[scvi]'")
if differ and not core and not read:
    print("  Importable, but a published number may not reproduce: neither UMAP nor Leiden is")
    print("  bit-reproducible across versions, and a latent space is not reproducible across a")
    print("  change of torch. Use --prefix if you are comparing against one.")
py_ok = not pyv or have_py == pyv.group(1)
if not (core or read or figs or bench or models or differ or missing) and py_ok and not (strict and stray):
    print("  environment matches the lock exactly")
print()
print("  For the capability view - what is present and what each absence costs - run:")
print("      scintegrate doctor")
if strict:
    sys.exit(0 if (not missing and not differ and not stray and py_ok) else 1)
sys.exit(1 if (core or read) else 0)
PYEOF
}

if [ "$MODE" = check ]; then
  PY="${PY:-$(command -v python3 || command -v python)}"
  [ -x "$PY" ] || { echo "no python found. Pass --python /path/to/python"; exit 2; }
  echo "checking: $PY"
  check "$PY" "$STRICT"
  exit $?
fi

# --------------------------------------- add-to: resolve first, refuse to disturb what is there
if [ "$MODE" = add ]; then
  [ -x "$ADD" ] || { echo "not an executable python: $ADD" >&2; exit 2; }
  echo "target : $ADD"
  echo "resolving scib + scvi-tools + bbknn + harmonypy against it, WITHOUT installing"
  RPT="$(mktemp -t scint_resolve.XXXXXX.json)"
  if ! "$ADD" -m pip install --dry-run --quiet --report "$RPT" \
        scib scvi-tools bbknn harmonypy; then
    echo
    echo "RESOLUTION FAILED. Nothing was installed. The message above names the conflict; that"
    echo "is the answer, and forcing past it with --no-deps would produce an environment whose"
    echo "failures appear later and elsewhere."
    rm -f "$RPT"; exit 1
  fi
  "$ADD" - "$RPT" <<'PYEOF'
import json, sys
import importlib.metadata as md
r = json.load(open(sys.argv[1]))
rows = []
for it in r.get("install", []):
    m = it["metadata"]; n, v = m["name"], m["version"]
    try: cur = md.version(n)
    except Exception: cur = None
    rows.append((n, cur, v))
changed = [x for x in rows if x[1] is not None and x[1] != x[2]]
new = [x for x in rows if x[1] is None]
print(f"  {len(rows)} package(s) resolved: {len(new)} new, {len(changed)} would CHANGE")
if changed:
    print()
    print("  THESE ALREADY-INSTALLED PACKAGES WOULD CHANGE VERSION:")
    for n, cur, v in sorted(changed):
        print(f"    {n:24s} {cur}  ->  {v}")
    print()
    print("  STOPPING. Anything this environment has already produced was produced on the")
    print("  versions on the left. Changing them does not invalidate the files, but it does mean")
    print("  the environment can no longer reproduce them - and that is a decision to take")
    print("  deliberately, not a side effect of adding a method.")
    print()
    print("  Either build a separate environment (--prefix), or re-run this having decided the")
    print("  upstream results do not need to be reproducible from here.")
    sys.exit(3)
print("  no already-installed package would change: this addition is additive and safe")
sys.exit(0)
PYEOF
  rc=$?; rm -f "$RPT"
  [ "$rc" -eq 0 ] || exit "$rc"
  echo
  echo "installing"
  "$ADD" -m pip install scib scvi-tools bbknn harmonypy || exit 1
  echo
  echo "done. Verify with:  $ADD -m scintegrate.cli doctor"
  exit 0
fi

# ---------------------------------------------------------------------------------- create
[ -n "$PREFIX" ] || { echo "--prefix is required to create an environment" >&2; usage 2; }
MAMBA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$MAMBA" ] || {
  cat >&2 <<'MSG'
no conda, mamba or micromamba on PATH.

scIntegrate does not bundle one. Either install micromamba
  https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
or, if you already have python 3.10+ and only want part of the comparison:
  pip install -e '.[run]'          # read .h5ad, assess, figures
  pip install -e '.[scib]'         # the benchmark and therefore the default choice
  pip install -e '.[scvi]'         # scVI and scANVI, pulls torch
and then check it with:  setup/install_env.sh --check
MSG
  exit 2; }

# `--yes` is ASKED of the manager, not assumed: micromamba's `env create` takes it, and conda's did
# not until long after 4.10 (its `env create` never prompts there, and refuses the flag as an
# unrecognised argument). A cluster's module conda can be that old.
YES=()
if "$MAMBA" env create --help 2>&1 | grep -q -- '--yes'; then YES=(--yes); fi
# The manager reads A COPY of the lock: conda writes its pip requirements file beside the file it
# is given, which would put it inside this checkout.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/scintegrate-env.XXXXXX")" || { echo "cannot make a temporary directory" >&2; exit 1; }
trap 'rm -rf "$WORK"' EXIT
cp "$LOCK" "$WORK/environment.lock.yml"

echo "creating with: $MAMBA ($("$MAMBA" --version 2>&1 | head -1))${YES[*]:+, ${YES[*]}}"
echo "prefix       : $PREFIX"
echo "note         : this lock includes torch and its CUDA wheels on linux. Expect a few GB and a few minutes."
"$MAMBA" env create ${YES[@]+"${YES[@]}"} --prefix "$PREFIX" --file "$WORK/environment.lock.yml" || {
  echo; echo "environment creation FAILED."
  echo "Check what is there before retrying: ls '$PREFIX'"; exit 1; }
if [ "$EDITABLE" = 1 ]; then
  "$PREFIX/bin/pip" install --no-deps -e "$ROOT" >/dev/null 2>&1 \
    || echo "  note: editable install of scIntegrate itself did not run; do it by hand"
else
  echo "  --no-editable: scIntegrate is not installed into the environment; it runs from this checkout by PYTHONPATH"
fi
echo
echo "checking the environment against the lock (strict: nothing the lock does not pin):"
check "$PREFIX/bin/python" 1 || {
  echo; echo "the environment does NOT match the lock: a version above differs, or the resolver added a"
  echo "package the lock does not pin. It was built, and is not the locked environment."; exit 1; }
echo
echo "done. Verify, then use it by PATH - do not rely on activation:"
echo "  $PREFIX/bin/python -m scintegrate.cli doctor"
echo
echo "On a cluster, the interpreter is the PROJECT'S and the code is the TOOL'S:"
echo "  PYTHONPATH=$ROOT  <project>/env/bin/python -m scintegrate.cli integrate \\"
echo "      --out <inside your project>   # never the tool's own directory"
