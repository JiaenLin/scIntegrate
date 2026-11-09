"""The two cohort axes this tool is told - species and assay - and what the object can say back.

WHY THIS EXISTS. scIntegrate took neither. A tool that is never told the species cannot notice
that it was handed another cohort's object, and a run record that does not say what the cells
were cannot be read by the stage after it. Two cohorts can differ on both axes at once, and every
tool that took an axis without checking it would have run one as the other and said nothing
(single-cell-harness ADR-0027).

SPECIES IS IN THE DATA, AND IS CHECKED. Ensembl gene identifiers name it outright (ENSG human,
ENSMUSG mouse, ...). The rule is the harness's `axes` probe, restated here rather than imported,
because this package must run without the harness present:

  - the column of `var` (or the index) holding the most Ensembl gene identifiers is read;
  - prefixes are matched LONGEST FIRST, because ENSGALG is chicken and starts with ENSG;
  - 90% agreement is decisive, and a declared species the identifiers contradict is REFUSED,
    naming both;
  - identifiers of several species (a barnyard reference) settle nothing, and say so;
  - without identifiers only the symbol convention is left - mouse and rat write Title-case,
    human and macaque UPPER - which is a convention, not evidence. A disagreement there is a note
    and never a refusal.

THE ASSAY IS NOT IN A COUNT MATRIX, AND NOTHING HERE DEPENDS ON IT. Whether the counts came from
cells or nuclei lives in the aligner's intronic fraction, not in the matrix this tool reads, so
it cannot be checked here. And no parameter, threshold, method setting or metric in this package
reads it: the methods are run, scored and ranked identically for cells and nuclei (docs/METHODS.md;
tests/test_cohort_axes.py runs one object under both and finds every number identical). So
`--assay` is OPTIONAL. A required flag that changes nothing would only teach a reader that the
assay mattered here. It is recorded in the run's provenance and wording when given, and recorded
as "not declared" when it is not - never guessed, and never defaulted. If a rule here ever comes
to depend on the assay, it is declared per assay in ASSAY_RULES and the flag becomes required:
tests/test_cohort_axes.py fails on any branch taken on the assay outside this module, on any
mention of it in the modules that compute, and on a required flag without a declared rule (or the
reverse).
"""
from __future__ import annotations

import re

#: Ensembl gene-identifier prefixes and the species each names, LONGEST FIRST: `ENSGALG` is
#: chicken and starts with `ENSG`. The names are Ensembl's own, which is the vocabulary the
#: harness declares axes in (`sch init --context species=homo_sapiens`).
ENSEMBL = sorted({"ENSG": "homo_sapiens", "ENSMUSG": "mus_musculus", "ENSRNOG": "rattus_norvegicus",
                  "ENSDARG": "danio_rerio", "ENSMMUG": "macaca_mulatta", "ENSGALG": "gallus_gallus",
                  "ENSSSCG": "sus_scrofa", "ENSBTAG": "bos_taurus"}.items(),
                 key=lambda kv: -len(kv[0]))

#: The symbol case each species writes in. A HINT, never a measurement: mouse and rat share one,
#: human and macaque another.
CONVENTION = {"homo_sapiens": "upper", "macaca_mulatta": "upper", "sus_scrofa": "upper",
              "bos_taurus": "upper", "gallus_gallus": "upper", "mus_musculus": "title",
              "rattus_norvegicus": "title", "danio_rerio": "lower"}

#: The common names the family's other tools are given, read as Ensembl's. `--species Mouse` is
#: how a sibling tool is called; it means mus_musculus here too, and is recorded as given.
COMMON = {"human": "homo_sapiens", "mouse": "mus_musculus", "rat": "rattus_norvegicus",
          "zebrafish": "danio_rerio", "macaque": "macaca_mulatta", "rhesus": "macaca_mulatta",
          "chicken": "gallus_gallus", "pig": "sus_scrofa", "cow": "bos_taurus"}

_ID = re.compile(r"^(ENS[A-Z]*G)\d{6,}")
DECISIVE = 0.9

#: The assays a run may declare. The names are the harness's (`--context assay=scrna`).
ASSAYS = ("scrna", "snrna")

#: EVERY RULE IN THIS TOOL THAT DIFFERS BY ASSAY. Empty, and that is a measured statement, not an
#: omission: see the module docstring. A rule that comes to differ is declared here, once per
#: assay, and `--assay` then becomes required - an assay a rule depends on is never defaulted.
ASSAY_RULES: dict = {}

#: The word for one observation, per declared assay. WORDING ONLY - nothing is computed from it.
UNIT = {"scrna": "cells", "snrna": "nuclei", None: "cells or nuclei (assay not declared)"}


class SpeciesContradicted(Exception):
    """The declared species is not the one the object's gene identifiers name."""


def canonical(species: str) -> str:
    """Ensembl's spelling: 'Mouse' -> mus_musculus, 'Homo sapiens' -> homo_sapiens.

    Anything else is kept as written, lower-cased and joined with underscores, so a species the
    table does not know can still be declared and recorded - and still contradicted, because the
    comparison is with what the identifiers measure, not with this table.
    """
    s = re.sub(r"[\s_]+", "_", str(species or "").strip().lower())
    return COMMON.get(s, s)


def _columns(var, var_names) -> dict:
    """{name: [str]} for the index and every var column that holds strings."""
    cols = {"var_names": [str(v) for v in var_names]}
    for c in getattr(var, "columns", []):
        s = var[c]
        if getattr(getattr(s, "dtype", None), "kind", "O") in "biufcmM":
            continue                            # a number or a flag cannot be an identifier
        cols[f"var[{c!r}]"] = [str(v) for v in s]
    return cols


def measure(var, var_names) -> dict:
    """What the object says its species is: {measured, basis, column, convention}.

    `measured` is a species only when the identifiers are decisive; `basis` is the sentence that
    says why, in either case, and travels into the run record.
    """
    cols = _columns(var, var_names)
    best = None
    for name, vals in cols.items():
        hits = {}
        for v in vals:
            m = _ID.match(v)
            if m:
                sp = next((s for p, s in ENSEMBL if m.group(1).startswith(p)), None)
                hits[sp] = hits.get(sp, 0) + 1
        n = sum(hits.values())
        if n and (best is None or n > best[1]):
            best = (name, n, hits, len(vals))
    out = {"measured": None, "column": None,
           "basis": "the object carries no Ensembl gene identifiers, in var or its index",
           "convention": None}
    if best:
        name, n, hits, total = best
        top, k = max(hits.items(), key=lambda kv: kv[1])
        out["column"] = name
        if top and k / n >= DECISIVE:
            out.update(measured=top, basis=f"{k:,} of {n:,} Ensembl gene identifiers in {name} "
                                           f"are {top}")
        elif top is None:
            out["basis"] = (f"the Ensembl gene identifiers in {name} carry a prefix this check "
                            f"does not know, so they name no species it can compare")
        else:
            out["basis"] = (f"the Ensembl gene identifiers in {name} are mixed ({k / n:.0%} {top}): "
                            f"a multi-species reference, or no single species")
    case = {"upper": 0, "title": 0, "lower": 0}
    for s in cols["var_names"]:
        if _ID.match(s):
            continue
        core = re.sub(r"[^A-Za-z]", "", s)
        if len(core) < 3:
            continue
        if core.isupper():
            case["upper"] += 1
        elif core[0].isupper() and core[1:].islower():
            case["title"] += 1
        elif core.islower():
            case["lower"] += 1
    tot = sum(case.values())
    if tot and max(case.values()) / tot >= 0.6:
        out["convention"] = max(case, key=case.get)
    return out


def check_species(declared, var, var_names) -> dict:
    """The declared species against the object. Returns the record the run carries; raises
    SpeciesContradicted when the gene identifiers name a different species.

    The record holds strings only - it is read by people and compared by name, and a count in it
    would be a number about the gene table masquerading as a result.
    """
    given = str(declared or "").strip()
    if not given:
        raise SpeciesContradicted("--species is empty. Declare the species the cells came from, "
                                  "as Ensembl writes it: homo_sapiens, mus_musculus, ...")
    want = canonical(given)
    m = measure(var, var_names)
    rec = {"declared": given, "as": want, "measured": m["measured"] or "not measured",
           "basis": m["basis"], "verdict": None, "note": ""}
    if m["measured"] and m["measured"] != want:
        raise SpeciesContradicted(
            f"--species {given!r} contradicts the object: {m['basis']}. A species carried over from "
            f"another cohort is exactly this, and nothing downstream would notice - declare "
            f"--species {m['measured']}, or check that this is the object you meant.")
    if m["measured"]:
        rec["verdict"] = "agrees with the gene identifiers"
        return rec
    rec["verdict"] = "not checkable from identifiers"
    expect = CONVENTION.get(want)
    if m["convention"] and expect and m["convention"] != expect:
        rec["note"] = (f"{want} writes its gene symbols {expect}-case and this object's read "
                       f"{m['convention']}-case - a convention, not proof, and there are no gene "
                       f"identifiers to settle it")
    elif m["convention"] and expect:
        rec["note"] = (f"the gene symbols read {m['convention']}-case, as {want} writes them - a "
                       f"convention, not proof")
    else:
        rec["note"] = "neither identifiers nor a symbol convention could test the declaration"
    return rec


def assay_record(assay) -> dict:
    """What the run records about the assay. Never a guess: absent is "not declared"."""
    return {"declared": assay or "not declared",
            "rules_that_differ": sorted(ASSAY_RULES) or "none",
            "meaning": ("recorded for the reader and the next stage. No parameter, threshold or "
                        "method in scIntegrate depends on the assay, so cells and nuclei are "
                        "integrated and scored by the same rules; a count matrix cannot show which "
                        "it holds, so it is not checked")}


def unit(assay) -> str:
    """The word for one observation in the run's wording."""
    return UNIT.get(assay or None, UNIT[None])
