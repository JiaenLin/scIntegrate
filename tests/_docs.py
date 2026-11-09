"""The command lines and design tables this repository's documents show, read the same way by
every suite that holds them to something.

Not a suite: run_all.py runs `test_*.py` only. Stdlib only. Imported as a sibling module.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("assess", "integrate", "score", "report")


def documents():
    """(name, the command lines it shows, the design tables it shows) for every document a user
    reads: the README, docs/*.md, and the usage text `scintegrate --help` prints, which is the CLI
    module's docstring and carries its commands unfenced."""
    out = []
    for p in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = p.read_text(encoding="utf-8")
        out.append((str(p.relative_to(ROOT)), commands(text), design_tables(text)))
    import ast
    tree = ast.parse((ROOT / "scintegrate" / "cli.py").read_text(encoding="utf-8"))
    usage = ast.get_docstring(tree) or ""
    out.append(("scintegrate --help", commands(usage, fenced=False), design_tables(usage)))
    return out


def blocks(text, lang=None):
    """The bodies of fenced code blocks, paired by walking the fences in order - a regex over the
    whole text pairs the closing fence of one block with the opening of the next. With `lang`,
    only blocks opened as ```<lang>."""
    out, cur, keep = [], None, False
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            if cur is None:
                cur, keep = [], (lang is None or ln.strip()[3:].strip() == lang)
            else:
                if keep:
                    out.append("\n".join(cur))
                cur = None
        elif cur is not None:
            cur.append(ln)
    return out


def commands(text, fenced=True):
    """Every `scintegrate <command> ...` in the text, as (command, argv after it). Shell variables
    are replaced by placeholders so a recipe parses; continuation lines are joined first."""
    bodies = blocks(text) if fenced else [text]
    found = []
    for body in bodies:
        joined = re.sub(r"\\\n\s*", " ", body)
        for ln in joined.splitlines():
            m = re.search(r"(?:^|\s)(?:scintegrate|-m scintegrate\.cli)\s+(" + "|".join(COMMANDS)
                          + r")\b(.*)", ln)
            if not m:
                continue
            argv = m.group(2).split("#")[0]
            argv = re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", argv)          # ${NCPUS:-1} -> 1
            argv = re.sub(r"\$\{?\w+\}?", "X", argv)                    # $RUNDIR -> X
            argv = argv.split("2>&1")[0].split("|")[0]
            found.append((m.group(1), shlex.split(argv)))
    return found


def option(argv, flag):
    """Every value given to `flag` in an argv, in order."""
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def design_tables(text):
    """The header of every ```csv block - the design tables a document shows."""
    return [[c.strip() for c in b.strip().splitlines()[0].split(",")]
            for b in blocks(text, lang="csv") if b.strip()]
