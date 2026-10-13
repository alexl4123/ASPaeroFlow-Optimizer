"""Answers to the questions a replay is asked, computed once (K35).

    python -m src.aspaeroflow.xai.answer_cache <trace folder> [--data-dir <instance folder>] [--check]
                                                                                   (from 01_ASPaeroFlow)

A replay used to solve every question when it was asked. The step's clingo control is reused across questions, so
when two answers are equally good the one returned could depend on which questions came before (same costs, other
routes). The cache computes every question the interface asks of a kept step in one fixed order - the what-if menu,
the flights in ascending order, the hotspot, the sectors, then each requirement the menu offers on its own, sorted -
and the replay serves those answers, so every participant gets the same answer to the same question. Questions
outside the cache (combinations of requirements) are computed on demand in a fresh explainer of the step.

The file `answers.json.gz` sits in the trace folder and carries a fingerprint of everything an answer depends on:
the trace files, the keep contrasts it loaded, the clingo version, the explaining code and the navpoint names of the
instance. A replay whose fingerprint differs ignores the file. `--check` builds twice and compares the bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import clingo

from .contrastive import Lock

FILE = "answers.json.gz"
VERSION = 1
HERE = Path(__file__).resolve().parent
#: the code that shapes an answer (a change to any of them makes old caches unusable)
CODE = ["contrastive.py", "reasons.py", "subproblem.py", "keep.py", "time_words.py", "trace.py", "session.py",
        "answer_cache.py", "reasons_text.lp", "reasons.lp"]


def explain_key(iteration: int, question: str, flight: Optional[int] = None) -> str:
    return f"explain|{iteration}|{question}|{int(flight) if question == 'flight' and flight is not None else ''}"


def what_if_key(iteration: int, locks: Iterable[str]) -> str:
    return f"what_if|{iteration}|" + ",".join(sorted({str(Lock.parse(t)) for t in locks}))


def fingerprint(folder: Path, keep: Dict[int, Any], names: Optional[Dict[int, str]]) -> str:
    h = hashlib.sha256()

    def add(label: str, data: bytes) -> None:
        h.update(label.encode() + b"\0" + hashlib.sha256(data).digest())

    folder = Path(folder)
    for name in ["trace.jsonl", "run.json", "encoding.lp"] + sorted(p.name for p in (folder / "lp").glob("*.lp")):
        path = folder / name if (folder / name).exists() else folder / "lp" / name
        if path.exists():
            add(name, path.read_bytes())
    add("keep", json.dumps({str(k): v for k, v in sorted(keep.items())}, sort_keys=True).encode())
    add("clingo", clingo.__version__.encode())
    for name in CODE:
        path = HERE / name
        if path.exists():
            add(name, path.read_bytes())
    add("names", json.dumps(sorted((names or {}).items())).encode())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    """The answer as the service sends it (JSON: dict keys become strings)."""
    return json.loads(json.dumps(value))


def build(folder: Path, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Every cached answer of the trace, in the fixed order; one explainer per step, used for that step only."""
    from .session import ReplaySession
    session = ReplaySession(Path(folder), data_dir)
    options = " ".join(session.trace().run.get("solver_options") or [])
    if "parallel" in options or "-t " in f"{options} " or "--threads" in options:
        raise ValueError(f"solver options {options!r}: parallel solving makes answers depend on timing; not cached")
    entries: Dict[str, Any] = {}
    for record in session.records:
        if not record.get("accepted"):
            continue
        n = int(record["iteration"])
        ex = session.explainer(n)
        menu = ex.what_if_menu()
        entries[explain_key(n, "menu")] = menu
        flights = sorted({int(f) for f in record.get("flight_changes") or {}} | set(ex.sub.decision_flights))
        for f in flights:
            try:
                entries[explain_key(n, "flight", f)] = ex.why_flight(f)
            except (KeyError, ValueError):
                pass                      # a flight the explainer cannot answer stays on demand (an error as today)
        entries[explain_key(n, "hotspot")] = ex.why_hotspot()
        entries[explain_key(n, "sectors")] = ex.why_sectors()
        locks = {e["lock"] for e in menu.get("menu", []) if e.get("state") == "open"}
        locks |= {v["lock"] for v in menu.get("navpoints", []) if v.get("state") == "open"}
        for lock in sorted(locks):
            entries[what_if_key(n, [lock])] = ex.what_if([Lock.parse(lock)])
    return {"version": VERSION, "fingerprint": fingerprint(Path(folder), session._keep, session.display_names()),
            "entries": _plain(entries)}


def to_bytes(cache: Dict[str, Any]) -> bytes:
    raw = json.dumps(cache, sort_keys=True, separators=(",", ":")).encode()
    return gzip.compress(raw, mtime=0)


def load(folder: Path) -> Optional[Dict[str, Any]]:
    path = Path(folder) / FILE
    if not path.exists():
        return None
    return json.loads(gzip.decompress(path.read_bytes()))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--data-dir", default=None, help="the instance folder the service uses (display names)")
    ap.add_argument("--check", action="store_true", help="build twice and compare the bytes")
    a = ap.parse_args(argv)
    data_dir = Path(a.data_dir) if a.data_dir else None
    data = to_bytes(build(Path(a.trace), data_dir))
    if a.check and to_bytes(build(Path(a.trace), data_dir)) != data:
        raise SystemExit("two builds differ: not written")
    (Path(a.trace) / FILE).write_bytes(data)
    cache = json.loads(gzip.decompress(data))
    print(f"wrote {Path(a.trace) / FILE}: {len(cache['entries'])} answers, {len(data) // 1024} KB")


if __name__ == "__main__":
    main()
