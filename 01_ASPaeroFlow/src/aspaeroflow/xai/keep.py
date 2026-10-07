"""The keep contrasts of a trace (IterationExplainer.keep_contrasts), one JSON line per kept step.

    keep_contrasts.jsonl   in the trace folder, next to trace.jsonl

The per-measure reasons of the change list (xai/reasons.py explain_rows) are derived from these numbers. A replay
reads the file, so its lines come with each step; it is written once after recording:

    cd 01_ASPaeroFlow && python -m src.aspaeroflow.xai.keep <trace folder> [--force]

Only this module writes the file into a trace folder (a live session appends to its own folder through
append_keep; a replay never writes). The file must be made again whenever encoding.lp or the contrast code changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

from .contrastive import IterationExplainer
from .trace import TraceReader

FILE_NAME = "keep_contrasts.jsonl"
log = logging.getLogger(__name__)
_lock = threading.Lock()


def read_keep(folder: Path) -> Dict[int, Dict[str, Any]]:
    """Lines of the file by iteration; an iteration with two different lines is left out (logged)."""
    path = Path(folder) / FILE_NAME
    if not path.exists():
        return {}
    found: Dict[int, Dict[str, Any]] = {}
    bad = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n") or not line.strip():
                continue                     # a half-written last line
            try:
                entry = json.loads(line)
                n = int(entry["iteration"])
            except (ValueError, KeyError, TypeError):
                log.warning("%s: a line is not a keep-contrast entry", path)
                continue
            if n in found and found[n] != entry:
                bad.add(n)
            found.setdefault(n, entry)
    for n in bad:
        log.warning("%s: iteration %s has two different lines; it is not used", path, n)
        del found[n]
    return found


def append_keep(folder: Path, entry: Dict[str, Any]) -> None:
    """One whole line per call (one write under a lock)."""
    line = json.dumps(entry, sort_keys=True) + "\n"
    with _lock:
        with open(Path(folder) / FILE_NAME, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def precompute(folder: Path, force: bool = False) -> Dict[int, Dict[str, Any]]:
    """Writes the line of every kept step that has none yet (all of them with force); returns all lines."""
    folder = Path(folder)
    trace = TraceReader(folder)
    existing = {} if force else read_keep(folder)
    entries: Dict[int, Dict[str, Any]] = {}
    for n in trace.accepted():
        if not trace.iterations[n].get("subproblems"):
            continue
        entries[n] = existing[n] if n in existing else IterationExplainer(trace, n).keep_contrasts()
    text = "".join(json.dumps(entries[n], sort_keys=True) + "\n" for n in sorted(entries))
    tmp = folder / (FILE_NAME + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, folder / FILE_NAME)
    return entries


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="trace folder (trace.jsonl, run.json, encoding.lp, lp/)")
    p.add_argument("--force", action="store_true", help="compute every line again")
    args = p.parse_args(argv)
    entries = precompute(args.folder, force=args.force)
    seconds = sum(float(e.get("seconds") or 0) for e in entries.values())
    print(f"{len(entries)} kept steps in {args.folder / FILE_NAME} ({seconds:.1f} s of solving)")


if __name__ == "__main__":
    main()
