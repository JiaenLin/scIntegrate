"""What a run leaves behind so that a reader who did not watch it can tell four states apart:
ok, partial, refused, failed — from the filesystem alone.

    STATUS.json   written FIRST as `partial`, rewritten LAST with the outcome and the products
    RUNNING.txt   written at start; replaced at exit by
    SEALED.txt    exit 0 and every expected product present, or
    FAILED.txt    anything else, naming what is missing

A run that dies leaves STATUS.json saying `partial` and RUNNING.txt standing — a different fact
from `refused` and from `ok`. The job script's own EXIT trap is the second witness; this is the
first, and the one that exists when there is no job script. Same shape as the status contract
shared with the tools this one is orchestrated beside. Stdlib only.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

CONTRACT = "1.0"
STATUSES = ("ok", "partial", "refused", "failed")
ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def commit() -> str | None:
    """The checkout's commit, read from .git by FILE — compute nodes have no git binary."""
    git = ROOT / ".git"
    try:
        if git.is_file():                              # a worktree: gitdir: <path>
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            f = git / ref
            if f.exists():
                return f.read_text().strip()
            packed = git / "packed-refs"
            if packed.exists():
                for ln in packed.read_text().splitlines():
                    if ln.endswith(" " + ref):
                        return ln.split()[0]
            return None
        return head
    except OSError:
        return None


def job() -> dict:
    for var, name in (("PBS_JOBID", "pbs"), ("SLURM_JOB_ID", "slurm")):
        if os.environ.get(var):
            return {"scheduler": name, "id": os.environ[var], "host": socket.gethostname()}
    return {"scheduler": None, "id": None, "host": socket.gethostname()}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def products_of(out: Path) -> list:
    res = []
    for sub in ("", "tables", "objects", "figures", "reports"):
        d = out / sub if sub else out
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix in (".json", ".csv", ".html", ".h5ad", ".md", ".png", ".pdf"):
                res.append({"path": _rel(p, out), "bytes": p.stat().st_size})
    return res


def begin(out: Path, *, command: str, version: str, state_version: int, sees: list,
          cannot_show: list, argv: list | None = None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"contract": CONTRACT, "tool": "scintegrate", "command": command, "version": version,
           "commit": commit(), "state_version": state_version, "status": "partial",
           "headline": "started", "started": _now(), "finished": None, "job": job(),
           "python": sys.version.split()[0], "argv": list(argv if argv is not None else sys.argv),
           "inputs": [], "products": [], "absent": [], "refusal": None,
           "sees": list(sees), "escapes": [], "cannot_show": list(cannot_show),
           "wrapped_versions": {}}
    (out / "STATUS.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "RUNNING.txt").write_text(
        f"started={rec['started']}\njobid={rec['job']['id'] or 'none'}\nhost={rec['job']['host']}\n"
        f"commit={rec['commit'] or 'unidentified'}\ncommand={command}\n", encoding="utf-8")
    return out / "STATUS.json"


def finish(out: Path, *, status: str, headline: str, exit_code: int, refusal: dict | None = None,
           expected: list | None = None, wrapped_versions: dict | None = None,
           absent: list | None = None, inputs: list | None = None, sees: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    out = Path(out)
    p = out / "STATUS.json"
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {"contract": CONTRACT, "tool": "scintegrate", "started": None, "job": job()}
    products = products_of(out)
    present = {x["path"] for x in products if x["bytes"] > 0}
    missing = [e for e in (expected or []) if e not in present]
    if status == "ok" and missing:
        status = "failed"
        headline = f"{headline}; missing products: {', '.join(missing)}"
    if status == "refused":
        refusal = dict(refusal or {})
        refusal.setdefault("reason", headline)
        refusal.setdefault("fix", "the refusal names what to change; see the message above it")
    rec.update({"status": status, "headline": headline, "finished": _now(), "exit": exit_code,
                "products": products, "missing": missing,
                "refusal": refusal if status == "refused" else None,
                "absent": list(absent or []), "inputs": list(inputs or rec.get("inputs") or []),
                "wrapped_versions": dict(wrapped_versions or rec.get("wrapped_versions") or {})})
    if sees is not None:
        rec["sees_by_method"] = sees
    p.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    sealed = status == "ok" and exit_code == 0 and not missing
    seal = out / ("SEALED.txt" if sealed else "FAILED.txt")
    lines = [f"exit={exit_code}", f"status={status}", f"jobid={(rec.get('job') or {}).get('id') or 'none'}",
             f"commit={rec.get('commit') or 'unidentified'}", f"started={rec.get('started')}",
             f"finished={rec['finished']}", f"host={socket.gethostname()}",
             "products=" + " ".join(x["path"] for x in products)]
    if missing:
        lines.append("missing=" + " ".join(missing))
    seal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    other = out / ("FAILED.txt" if sealed else "SEALED.txt")
    if other.exists():
        other.unlink()
    if (out / "RUNNING.txt").exists():
        (out / "RUNNING.txt").unlink()
    return rec
