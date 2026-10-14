"""Demand, capacity and overload per sector and time step (K08), taken from the optimizer's own matrices.

The optimizer's definitions (optimize_flights.py), which this module only reads, never re-derives:
    demand[s, t]    flights present in sector s at time step t (system_loads, bucket_histogram); between two navpoints
                    a flight is counted half in the previous sector and half in the next
    capacity[s, t]  capacity of sector s at t (capacity_time_matrix): the composite rule (max) over the navpoints that
                    form s at t; 0 where no navpoint forms s
    overload        the sum over every (s, t) of max(0, demand - capacity), the trace's OVERLOAD

A sector is named by a navpoint id; which navpoints form it at a time step is the optimizer's navpoint-to-sector matrix
(navaid_sector_time_assignment). A split or a combination changes that matrix over a range of time steps, and the
record states the composition of every sector over the whole horizon, so a split's full time span is known.

The optimizer writes the file itself (xai/hooks.py) when the trace is on:

    <trace folder>/load.jsonl     one JSON object per line
        {"header": {...}}         first line of a replay side file only (see rerun below)
        {"iteration": 0, ...}     the filed plan, written by Main.begin
        {"iteration": n, ...}     the plan after step n, written when the step's state is final

A record (v 1):
    iteration   0 = filed plan, n = after step n
    kept        null for the filed plan, else whether step n was kept
    state_of    the iteration whose plan this is (a rejected step leaves the plan of the step before; its line then
                holds only iteration, kept, state_of and check, and read() fills in the rest)
    horizon     time steps of the optimizer's matrices
    from, to    first and last time step with demand in any listed sector (null when there is none)
    overload    sum of max(0, demand - capacity) over the optimizer's whole grid
    sectors     one row per sector that exists at some time step, airport-only sectors left out:
                {"sector": s,
                 "demand":   [demand[s, t] for t in from..to],
                 "capacity": [capacity[s, t] or null where no navpoint forms s at t, for t in from..to],
                 "parts":    [{"from", "to", "navpoints", "filed_sectors", "kind", "capacity"}, ...]}
                parts: maximal runs of time steps (over the whole horizon) with the same navpoints; kind "filed"
                (exactly the navpoints of one filed sector), "split" (some of one filed sector's navpoints) or
                "combined" (navpoints of several filed sectors)
    check       {"ok", "overload": {"grid", "rows", "recorded", "ok"}, "demand_without_sector",
                 "hotspot": {"sector", "time", "recorded", "grid", "ok"} (iteration >= 1: the trace's hotspot cell of
                 step n against the plan before step n)}

The service (07_heuristic_controller/session_service.py) serves a record with GET /sessions/{id}/load?upto=n, only for
delivered steps (load_for). A replay serves `load.jsonl` of its trace folder; a study trace recorded before this hook
gets the file from a re-run that reproduces the trace exactly:

    python -m src.aspaeroflow.xai.load <trace folder> --out <file> [--data-dir <instance>] [--work <folder>]
                                                                      (from 01_ASPaeroFlow)

The re-run compares every record of its trace with the stored trace.jsonl (all fields but solve_seconds and
TOTAL-TIME-TO-THIS-POINT), and run.json, encoding.lp and lp/*.lp byte for byte; on any difference nothing is written.
The header carries the sha256 of the trace.jsonl it belongs to; a replay ignores a file whose hash differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

FILE = "load.jsonl"
VERSION = 1
log = logging.getLogger(__name__)

#: trace fields that differ between two runs of the same plan (wall-clock times)
VOLATILE_OBJECTIVES = ("TOTAL-TIME-TO-THIS-POINT",)
VOLATILE_SUBPROBLEM = ("solve_seconds",)


# ---------------------------------------------------------------------------------------------------------------------
# One record from the optimizer's matrices
# ---------------------------------------------------------------------------------------------------------------------

def _widen(filed: np.ndarray, horizon: int) -> np.ndarray:
    """The filed navpoint-to-sector matrix on the current horizon: the optimizer widens its matrices by repeating the
    last column (evaluate_solution.py), never the filed one, so the same is done here."""
    if filed.shape[1] >= horizon:
        return filed[:, :horizon]
    extra = np.repeat(filed[:, [-1]], horizon - filed.shape[1], axis=1)
    return np.concatenate([filed, extra], axis=1)


def _parts(assignment: np.ndarray, filed: np.ndarray, capacity: np.ndarray,
           sectors: Iterable[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Composition of each sector over the whole horizon, as maximal runs of time steps with the same navpoints (and
    the same filed sectors, kind and capacity)."""
    wanted = set(int(s) for s in sectors)
    horizon = assignment.shape[1]
    runs: Dict[int, List[Dict[str, Any]]] = {s: [] for s in wanted}
    open_key: Dict[int, Optional[tuple]] = {s: None for s in wanted}
    for t in range(horizon):
        column, filed_column = assignment[:, t], filed[:, t]
        order = np.argsort(column, kind="stable")
        bounds = np.flatnonzero(np.diff(column[order])) + 1
        groups = {int(column[g[0]]): g for g in np.split(order, bounds)}     # sector -> its navpoints at t
        filed_members: Dict[int, frozenset] = {}
        for s in wanted:
            navpoints = groups.get(s)
            if navpoints is None:
                open_key[s] = None
                continue
            navpoints = tuple(sorted(int(v) for v in navpoints))
            filed_sectors = tuple(sorted({int(filed_column[v]) for v in navpoints}))
            if len(filed_sectors) > 1:
                kind = "combined"
            else:
                f = filed_sectors[0]
                if f not in filed_members:
                    filed_members[f] = frozenset(int(v) for v in np.flatnonzero(filed_column == f))
                kind = "filed" if frozenset(navpoints) == filed_members[f] else "split"
            key = (navpoints, filed_sectors, kind, int(capacity[s, t]))
            if open_key[s] == key:
                runs[s][-1]["to"] = t
            else:
                runs[s].append({"from": t, "to": t, "navpoints": list(navpoints), "filed_sectors": list(filed_sectors),
                                "kind": kind, "capacity": int(capacity[s, t])})
                open_key[s] = key
    return runs


def grid(assignment, loads, capacity, filed, airports) -> Dict[str, Any]:
    """The record's plan part (horizon, window, overload, sectors) for the optimizer's matrices of one state."""
    a = np.asarray(assignment, dtype=np.int64)
    d = np.asarray(loads, dtype=np.int64)
    c = np.asarray(capacity, dtype=np.int64)
    n, horizon = a.shape
    if d.shape != (n, horizon) or c.shape != (n, horizon):
        raise ValueError(f"matrix shapes differ: assignment {a.shape}, demand {d.shape}, capacity {c.shape}")
    f = _widen(np.asarray(filed, dtype=np.int64), horizon)

    exists = np.zeros((n, horizon), dtype=bool)
    exists[a, np.broadcast_to(np.arange(horizon), a.shape)] = True
    airport_set = {int(v) for v in np.asarray(airports if airports is not None else [], dtype=np.int64).ravel()}
    is_airport = np.array([v in airport_set for v in range(n)], dtype=bool)
    # a sector is left out when every navpoint that ever forms it is an airport
    non_airport_sectors = set(int(s) for s in np.unique(a[~is_airport, :])) if (~is_airport).any() else set()
    sectors = sorted(set(int(s) for s in np.unique(a)) & non_airport_sectors)

    over = np.maximum(d - c, 0)
    grid_overload = int(over.sum())
    rows_overload = int(over[sectors, :].sum()) if sectors else 0
    listed = d[sectors, :].sum(axis=0) if sectors else np.zeros(horizon, dtype=np.int64)
    times = np.flatnonzero(listed > 0)
    lo, hi = (int(times[0]), int(times[-1])) if times.size else (None, None)

    parts = _parts(a, f, c, sectors)
    rows = []
    for s in sectors:
        row: Dict[str, Any] = {"sector": s}
        if lo is not None:
            row["demand"] = [int(x) for x in d[s, lo:hi + 1]]
            row["capacity"] = [int(c[s, t]) if exists[s, t] else None for t in range(lo, hi + 1)]
        else:
            row["demand"], row["capacity"] = [], []
        row["parts"] = parts[s]
        rows.append(row)
    return {"horizon": int(horizon), "from": lo, "to": hi, "overload": grid_overload, "sectors": rows,
            "_rows_overload": rows_overload,
            "_demand_without_sector": int(((d > 0) & ~exists).sum())}


class LoadWriter:
    """Writes load.jsonl of a trace folder while the optimizer runs (xai/hooks.py holds one per trace)."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / FILE
        self._fh = open(self.path, "w", encoding="utf-8")
        self._previous: Optional[Tuple[np.ndarray, np.ndarray]] = None   # demand, capacity of the plan before
        self._plan: Optional[str] = None                                 # the last written plan part (JSON)
        self._plan_of: Optional[int] = None

    def write(self, iteration: int, kept: Optional[bool], recorded_overload: Optional[int],
              hotspot: Optional[Dict[str, Any]], assignment, loads, capacity, filed, airports) -> Dict[str, Any]:
        plan = grid(assignment, loads, capacity, filed, airports)
        rows_overload = plan.pop("_rows_overload")
        demand_without_sector = plan.pop("_demand_without_sector")
        overload_ok = plan["overload"] == rows_overload and (recorded_overload is None
                                                           or plan["overload"] == int(recorded_overload))
        check: Dict[str, Any] = {
            "overload": {"grid": plan["overload"], "rows": rows_overload,
                         "recorded": None if recorded_overload is None else int(recorded_overload), "ok": overload_ok},
            "demand_without_sector": demand_without_sector,
        }
        ok = overload_ok and demand_without_sector == 0
        if iteration > 0:
            check["hotspot"] = self._hotspot_check(hotspot)
            ok = ok and check["hotspot"]["ok"]
        check["ok"] = bool(ok)

        text = json.dumps(plan, sort_keys=True)
        record: Dict[str, Any] = {"v": VERSION, "iteration": int(iteration), "kept": kept}
        if self._plan is not None and text == self._plan and kept is not True:
            record["state_of"] = self._plan_of          # the same plan as before: written once
            line = record | {"check": check}
        else:
            record["state_of"] = int(iteration)
            self._plan, self._plan_of = text, int(iteration)
            line = record | plan | {"check": check}
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self._previous = (np.array(loads, dtype=np.int64, copy=True), np.array(capacity, dtype=np.int64, copy=True))
        return line

    def write_error(self, iteration: int, kept: Optional[bool], exc: BaseException) -> None:
        self._fh.write(json.dumps({"v": VERSION, "iteration": int(iteration), "kept": kept,
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "check": {"ok": False}}) + "\n")
        self._fh.flush()
        self._previous = None

    def _hotspot_check(self, hotspot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        hotspot = hotspot or {}
        s, t = hotspot.get("sector"), hotspot.get("time")
        out: Dict[str, Any] = {"sector": s, "time": t,
                               "recorded": [hotspot.get("demand"), hotspot.get("capacity")], "grid": None, "ok": False}
        if self._previous is None or s is None or t is None:
            return out
        d, c = self._previous
        if 0 <= s < d.shape[0] and 0 <= t < d.shape[1]:
            out["grid"] = [int(d[s, t]), int(c[s, t])]
            out["ok"] = out["grid"] == out["recorded"]
        return out

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------------------------------------------------
# Reading and serving
# ---------------------------------------------------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """(header or None, iteration -> full record). A rejected step's short line gets the plan part of state_of; a
    half-written last line (a live run writing) is skipped."""
    header, records = None, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n") or not line.strip():
                continue
            item = json.loads(line)
            if "header" in item:
                header = item["header"]
                continue
            n = int(item["iteration"])
            if "sectors" not in item and "error" not in item and item.get("state_of") in records:
                base = records[item["state_of"]]
                plan = {k: v for k, v in base.items() if k not in ("v", "iteration", "kept", "state_of", "check")}
                item = {**item, **plan}
            records[n] = item
    return header, records


_CACHE: Dict[str, Tuple[tuple, Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], Optional[bool]]] = {}
_CACHE_LOCK = threading.Lock()


def _cached(folder: Path):
    """read() of a folder's load.jsonl, again only when the file changed; with whether its header matches the trace."""
    path = Path(folder) / FILE
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    with _CACHE_LOCK:
        hit = _CACHE.get(str(path))
        if hit is not None and hit[0] == key:
            return hit[1], hit[2], hit[3]
    header, records = read(path)
    matches = None
    if header is not None:
        matches = header.get("trace_sha256") == sha256_of(Path(folder) / "trace.jsonl")
        if not matches:
            log.warning("%s: load.jsonl was made for another trace.jsonl; not served", folder)
    with _CACHE_LOCK:
        _CACHE[str(path)] = (key, header, records, matches)
    return header, records, matches


def load_for(session, upto: int) -> Dict[str, Any]:
    """The load record of step `upto` (0 = filed plan) of a live or replayed session. Like the plan, only a step the
    session has delivered can be asked for. KeyError (the service answers 404) when the step has not been delivered,
    the trace folder has no load.jsonl, the file belongs to another trace, or it has no record for the step."""
    delivered = session.delivered()
    last = max((int(r["iteration"]) for r in delivered), default=0)
    if upto < 0 or upto > last:
        raise KeyError(f"step {upto} has not been delivered (last delivered: {last})")
    folder = Path(session.folder)
    if not (folder / FILE).is_file():
        raise KeyError(f"no load data for this trace ({FILE} missing)")
    _header, records, matches = _cached(folder)
    if matches is False:
        raise KeyError(f"{FILE} does not belong to this trace")
    if upto not in records:
        raise KeyError(f"no load record for step {upto}")
    return records[upto]


# ---------------------------------------------------------------------------------------------------------------------
# Side file of a recorded trace: a re-run that reproduces it exactly
# ---------------------------------------------------------------------------------------------------------------------

def _stable(record: Dict[str, Any]) -> Dict[str, Any]:
    """A trace record without its wall-clock fields."""
    record = json.loads(json.dumps(record))
    for key in VOLATILE_OBJECTIVES:
        record.get("objectives", {}).pop(key, None)
    for sub in record.get("subproblems", []):
        for key in VOLATILE_SUBPROBLEM:
            sub.pop(key, None)
    return record


def compare_traces(stored: Path, rerun: Path) -> List[str]:
    """Every difference between two trace folders (empty: the re-run reproduces the stored trace)."""
    stored, rerun = Path(stored), Path(rerun)
    diffs: List[str] = []
    a = [json.loads(x) for x in (stored / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    b = [json.loads(x) for x in (rerun / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(a) != len(b):
        diffs.append(f"trace.jsonl: {len(a)} records stored, {len(b)} in the re-run")
    for x, y in zip(a, b):
        if _stable(x) != _stable(y):
            fields = sorted(k for k in set(x) | set(y) if _stable(x).get(k) != _stable(y).get(k))
            diffs.append(f"trace.jsonl iteration {x.get('iteration')}: differs in {', '.join(fields)}")
    run_a = json.loads((stored / "run.json").read_text(encoding="utf-8"))
    run_b = json.loads((rerun / "run.json").read_text(encoding="utf-8"))
    for key in sorted(set(run_a) | set(run_b)):
        if key != "data_dir" and run_a.get(key) != run_b.get(key):
            diffs.append(f"run.json: {key} {run_a.get(key)!r} stored, {run_b.get(key)!r} in the re-run")
    if (stored / "encoding.lp").read_bytes() != (rerun / "encoding.lp").read_bytes():
        diffs.append("encoding.lp differs")
    names_a = sorted(p.name for p in (stored / "lp").glob("*.lp"))
    names_b = sorted(p.name for p in (rerun / "lp").glob("*.lp"))
    if names_a != names_b:
        diffs.append(f"lp/: {len(names_a)} files stored, {len(names_b)} in the re-run")
    for name in sorted(set(names_a) & set(names_b)):
        if (stored / "lp" / name).read_bytes() != (rerun / "lp" / name).read_bytes():
            diffs.append(f"lp/{name} differs")
    return diffs


def rerun_options(run: Dict[str, Any], number_threads: int) -> Dict[str, Any]:
    """main.py options (long names without dashes) that repeat a recorded run; run.json is compared afterwards."""
    options = {
        "seed": run["seed"],
        "timestep_granularity": run["timestep_granularity"],
        "max_explored_vertices": run["max_explored_vertices"],
        "max_delay_per_iteration": run["max_delay_per_iteration"],
        "max_considered_aircraft": run["max_considered_aircraft"],
        "number_capacity_management_configs": run["number_capacity_management_configs"],
        "capacity_management_enabled": str(bool(run["capacity_management_enabled"])).lower(),
        "composite_sector_function": run["composite_sector_function"],
        "sector_capacity_factor": run["sector_capacity_factor"],
        "arrival_delay_metric": run["arrival_delay_metric"],
        "sequential_execution": str(bool(run["sequential_execution"])).lower(),
        "minimize_number_sectors_enabled": str(bool(run["minimize_number_sectors"])).lower(),
        "max_number_sectors": run["max_number_sectors"],
        "convex_sectors": run["convex_sectors"],
        "number_threads": number_threads,
        "verbosity": 0,
    }
    if run.get("solver_options"):
        raise ValueError("the recorded run used solver options; repeating them is not supported")
    return options


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rerun(trace: Path, out: Path, data_dir: Optional[Path] = None, work: Optional[Path] = None,
          number_threads: int = 4) -> Dict[str, Any]:
    """Run the recorded trace again; write `out` (header + the re-run's load.jsonl) only if the re-run reproduces the
    trace. Returns a summary (differences, checks per step, size, seconds)."""
    from .session import OptimizerSession

    trace = Path(trace)
    run = json.loads((trace / "run.json").read_text(encoding="utf-8"))
    data_dir = Path(data_dir) if data_dir is not None else Path(run["data_dir"])
    own_work = work is None
    work = Path(tempfile.mkdtemp(prefix="load-rerun-")) if own_work else Path(work)
    folder = work / "trace"
    if folder.exists():
        shutil.rmtree(folder)
    started = time.time()
    try:
        session = OptimizerSession(data_dir, folder, rerun_options(run, number_threads))
        session.begin()
        while session.status != "finished":
            if session.step() is None:
                break
        seconds = round(time.time() - started, 1)
        diffs = compare_traces(trace, folder)
        _header, records = read(folder / FILE)
        summary: Dict[str, Any] = {"trace": str(trace), "seconds": seconds, "differences": diffs,
                                   "steps": [{"iteration": n, "kept": r.get("kept"), "state_of": r.get("state_of"),
                                              "check": r.get("check")} for n, r in sorted(records.items())]}
        if diffs:
            summary["written"] = None
            return summary
        header = {"v": VERSION, "trace_sha256": sha256_of(trace / "trace.jsonl"), "records": len(records),
                  "made_by": "re-run reproducing the trace", "optimizer_commit": _commit(),
                  "number_threads": number_threads,
                  "compared": "every trace.jsonl record without solve_seconds and TOTAL-TIME-TO-THIS-POINT; run.json "
                              "without data_dir; encoding.lp and lp/*.lp byte for byte"}
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = (folder / FILE).read_text(encoding="utf-8")
        out.write_text(json.dumps({"header": header}) + "\n" + body, encoding="utf-8")
        summary["written"] = str(out)
        summary["bytes"] = out.stat().st_size
        return summary
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="load.jsonl of a recorded trace from a re-run that reproduces it")
    p.add_argument("trace", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--work", type=Path, default=None, help="keep the re-run's trace folder here")
    p.add_argument("--number-threads", type=int, default=4)
    p.add_argument("--summary", type=Path, default=None, help="write the summary (JSON) here; default: stderr")
    args = p.parse_args(argv)
    summary = rerun(args.trace, args.out, args.data_dir, args.work, args.number_threads)
    text = json.dumps(summary, indent=1)
    if args.summary is not None:
        args.summary.write_text(text + "\n", encoding="utf-8")
    else:
        print(text, file=sys.stderr)        # the optimizer prints its objective lines to stdout
    return 0 if summary.get("written") else 1


if __name__ == "__main__":
    sys.exit(main())
