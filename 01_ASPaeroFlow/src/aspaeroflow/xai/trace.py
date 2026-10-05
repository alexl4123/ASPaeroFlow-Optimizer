"""Iteration trace of an ASPaeroFlow run: what each iteration looked at, chose and achieved.

The trace is a folder:

    encoding.lp      the encoding exactly as the sub-problems were solved with (incl. the metric fact)
    run.json         instance folder, seed, parameters, initial objective values
    trace.jsonl      one line per iteration (see IterationTrace.record)
    lp/iter_<n>_<k>.lp   the ASP instance of sub-problem k of iteration n

It holds everything an explanation needs (the instance text, the chosen answer, the hotspot, the
current trajectories of the candidate flights) and nothing heavy: a 60-flight run writes well under
a megabyte. The XAI front end and the explanation engine read it instead of receiving numpy backups.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


def _plain(value: Any) -> Any:
    """JSON-safe copy (numpy scalars and int keys)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def trajectory_of(navpoint_row: np.ndarray, fill_value: int = -1) -> Dict[int, int]:
    """time step -> navpoint of one row of the navpoint matrix."""
    times = np.flatnonzero(navpoint_row != fill_value)
    return {int(t): int(navpoint_row[t]) for t in times}


class IterationTrace:
    """Writes the trace folder while the optimizer runs."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        (self.folder / "lp").mkdir(parents=True, exist_ok=True)
        self._lines = open(self.folder / "trace.jsonl", "w", encoding="utf-8")
        self._encoding_written = False

    def write_run(self, info: Dict[str, Any]) -> None:
        with open(self.folder / "run.json", "w", encoding="utf-8") as fh:
            json.dump(_plain(info), fh, indent=2)

    def write_encoding(self, encoding: str) -> None:
        if not self._encoding_written:
            (self.folder / "encoding.lp").write_text(encoding, encoding="utf-8")
            self._encoding_written = True

    def record(self, iteration: int, hotspot: Dict[str, Any], accepted: bool,
               objectives: Dict[str, Any], subproblems: List[Dict[str, Any]],
               flight_changes: Dict[int, Any], sector_changes: Dict[str, Any],
               parameters: Dict[str, Any]) -> None:
        """One iteration. `subproblems` items carry the instance text under "instance"; it goes to lp/."""
        stored = []
        for k, sub in enumerate(subproblems):
            sub = dict(sub)
            name = f"lp/iter_{iteration:05d}_{k}.lp"
            (self.folder / name).write_text(sub.pop("instance"), encoding="utf-8")
            sub["instance_file"] = name
            stored.append(sub)
        line = {
            "iteration": iteration,
            "hotspot": hotspot,
            "accepted": accepted,
            "objectives": objectives,
            "subproblems": stored,
            "flight_changes": flight_changes,
            "sector_changes": sector_changes,
            "parameters": parameters,
        }
        self._lines.write(json.dumps(_plain(line)) + "\n")
        self._lines.flush()

    def close(self) -> None:
        self._lines.close()


class TraceReader:
    """Reads a trace folder back."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.encoding = (self.folder / "encoding.lp").read_text(encoding="utf-8")
        run = self.folder / "run.json"
        self.run = json.loads(run.read_text(encoding="utf-8")) if run.exists() else {}
        self.iterations: Dict[int, Dict[str, Any]] = {}
        with open(self.folder / "trace.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    self.iterations[int(record["iteration"])] = record

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for n in sorted(self.iterations):
            yield self.iterations[n]

    def instance(self, iteration: int, k: int = 0) -> str:
        sub = self.iterations[iteration]["subproblems"][k]
        return (self.folder / sub["instance_file"]).read_text(encoding="utf-8")

    def accepted(self) -> List[int]:
        return [n for n, r in sorted(self.iterations.items()) if r["accepted"]]

    def current_trajectory(self, iteration: int, flight: int, k: int = 0) -> Optional[Dict[int, int]]:
        sub = self.iterations[iteration]["subproblems"][k]
        raw = sub.get("current_trajectories", {}).get(str(flight))
        return None if raw is None else {int(t): int(n) for t, n in raw.items()}
