"""Next-step preview (S8 item 4c, K27 step strip): after step n, the hotspot cell step n+1 works on and the flights it
passes to the solver, from the optimizer's state after step n.

The preview never re-implements the rule. It calls the optimizer's own code on the state a step leaves:
IterationStep.step_scope (main_loop_components/iteration_step.py: the matrices widened in time when needed, then
first_overloads, then select_flights), which optimization_step calls on the same state at the start of step n+1.
The rule, as that code has it:
    hotspot     the earliest overloaded time step, at it the lowest sector index (in sequential mode cells already
                tried are skipped)
    flights     the cell's flights sorted by their stored duration, shortest first (ties: the lower flight id)
    solver      the first min(max_aircraft, max(2, overload), cell size) of them; each brings the flights of its
                aircraft that start no earlier ("later" flights)
    max_aircraft  run.json max_considered_aircraft after every kept step; after REJECTIONS_BEFORE_ONE_AIRCRAFT
                rejected steps in a row 1
A rejected step restores the plan before it, so the next step works on the same cell again ("retry").

The optimizer writes the file itself (xai/hooks.py) when the trace is on:

    <trace folder>/preview.jsonl    one JSON object per line
        {"header": {...}}           first line of a replay side file only (see rerun below)
        {"iteration": 0, ...}       the preview made from the filed plan (Main.begin): step 1
        {"iteration": n, ...}       the preview made after step n (Main.step): step n+1

A record (v 1):
    iteration   n: the preview was made after step n (0 = filed plan)
    kept        null for the filed plan, else whether step n was kept
    next        n + 1, or null when the run is finished
    finished    true when no step n+1 follows; "reason" then says why: "no_overload" (the plan has no overload),
                "no_progress" (the optimizer stops: no further progress possible), "no_cell" (sequential mode: every
                overloaded cell was tried)
    hotspot     {"sector", "time", "clock" ('07:15', '+1d 01:15'), "demand", "capacity", "overload"}
    flights     the cell's flights in the optimizer's order: [{"id", "duration"}, ...]
    solver      the flights passed to the solver (the first "taken" of flights), in that order
    later       the later flights of their aircraft that join them (ascending ids)
    solver_flights  solver + later (ascending ids): every flight step n+1's sub-problem holds
    aircraft    [{"aircraft", "flights"}]: per aircraft of a solver flight its flights from that flight on, by start
    rule        {"max_aircraft", "overload", "taken", "cells_per_step"}
    streak      {"rejections": rejected steps in a row before step n+1 (sequential mode does not count them),
                "retry": step n was rejected and step n+1 works on its cell again (false in sequential mode, which
                moves on to the next untried cell), "until_one_flight": further rejections in a row after which a
                step passes only one flight (0: already), "additional_time_increase"}
    check       iteration >= 1: {"ok", "made_after": n-1, "differences": [...]}, the hook's own comparison of the
                preview made after step n-1 with what step n then selected (cell, solver flights in order, every
                sub-problem flight, max_aircraft)

The service (07_heuristic_controller/session_service.py) serves a record with GET /sessions/{id}/preview?upto=n, only
for delivered steps (preview_for). A replay serves `preview.jsonl` of its trace folder; a study trace recorded before
this hook gets the file from a re-run that reproduces the trace exactly:

    python -m src.aspaeroflow.xai.preview <trace folder> --out <file> [--data-dir <instance>] [--work <folder>]
                                                                      (from 01_ASPaeroFlow)

The re-run compares its trace with the stored one (xai/load.py compare_traces) and proves every preview against the
stored trace (prove: hotspot, cell flights, taken flights, the sub-problem's flights in the record and in its lp file);
on any difference nothing is written. The header carries the sha256 of the trace.jsonl it belongs to.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FILE = "preview.jsonl"
VERSION = 1
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------------
# The preview from the optimizer's state
# ---------------------------------------------------------------------------------------------------------------------

def compute(main, outcome: Optional[str] = None) -> Dict[str, Any]:
    """The preview of the step that follows the optimizer's current state (`main`: the Main object after begin() or
    after a step; `outcome`: what Main.step returned, "terminate" when the optimizer stops). Reads the state only."""
    from ..auxiliaries.dto_helpers import convert_global_vars_to_dto
    from ..main_loop_components.evaluate_solution import REJECTIONS_BEFORE_ONE_AIRCRAFT
    from ..main_loop_components.iteration_step import CELLS_PER_STEP, IterationStep
    from .time_words import clock_text

    if outcome == "terminate":
        return {"next": None, "finished": True, "reason": "no_progress"}
    if not main.has_overload():
        return {"next": None, "finished": True, "reason": "no_overload"}
    dto = main.optimization_dto
    stepper = IterationStep(convert_global_vars_to_dto(main))
    scope = stepper.step_scope(dto["navaid_sector_time_assignment"], dto["converted_instance_matrix"],
                               dto["converted_navpoint_matrix"], dto["system_loads"], dto["capacity_time_matrix"],
                               dto["capacity_demand_diff_matrix"], dto["capacity_overload_mask"],
                               dto["additional_time_increase"], dto["max_number_airplanes_considered_in_ASP"],
                               dto["flight_durations"], dto["fill_value"])
    if not scope["cells"]:
        return {"next": None, "finished": True, "reason": "no_cell"}
    t, s = (int(x) for x in scope["cells"][0])
    selection = scope["selections"][0]
    durations = dto["flight_durations"]
    solver = [int(f) for f in selection["rows_pool"]]
    solver_flights = sorted(int(f) for f in selection["all_potentially_problematic_flight_indices"])
    max_aircraft = int(dto["max_number_airplanes_considered_in_ASP"])
    rejections = int(dto["counter_equal_solutions"] or 0)
    return {
        "next": None,               # set by the writer (iteration + 1)
        "finished": False,
        "reason": None,
        "hotspot": {"sector": s, "time": t, "clock": clock_text(t, int(main._timestep_granularity)),
                    "demand": int(scope["system_loads"][s, t]), "capacity": int(scope["capacity_time_matrix"][s, t]),
                    "overload": int(-scope["capacity_demand_diff_matrix"][s, t])},
        "flights": [{"id": int(f), "duration": int(durations[f])} for f in selection["cell"]],
        "solver": solver,
        "later": [f for f in solver_flights if f not in set(solver)],
        "solver_flights": solver_flights,
        "aircraft": [{"aircraft": int(a), "flights": [int(f) for f in fs]}
                     for a, fs in selection["airplane_flight_map"].items()],
        "rule": {"max_aircraft": max_aircraft, "overload": int(-scope["capacity_demand_diff_matrix"][s, t]),
                 "taken": len(solver), "cells_per_step": int(CELLS_PER_STEP)},
        "streak": {"rejections": rejections,
                   "retry": None,      # set by the writer (step n rejected and step n+1 on its cell)
                   "until_one_flight": max(0, int(REJECTIONS_BEFORE_ONE_AIRCRAFT) - rejections),
                   "additional_time_increase": int(dto["additional_time_increase"])},
    }


def selected(optimization_dto: Dict[str, Any]) -> Dict[str, Any]:
    """What a step actually selected (set by optimization_step before solving)."""
    cell = optimization_dto.get("xai_cell")
    return {"cell": None if cell is None else [int(cell[0]), int(cell[1])],
            "solver": [int(f) for f in optimization_dto.get("xai_taken") or []],
            "solver_flights": sorted(int(f) for f in optimization_dto.get("xai_solver_flights") or []),
            "max_aircraft": int(optimization_dto["max_number_airplanes_considered_in_ASP"])}


def compare(preview: Optional[Dict[str, Any]], actual: Optional[Dict[str, Any]]) -> List[str]:
    """Differences between a preview and what the step then selected (empty: equal)."""
    if preview is None:
        return ["no preview"]
    if actual is None:
        return ["the step selected nothing"]
    if preview.get("finished"):
        return [f"preview said finished ({preview.get('reason')}), the step selected cell {actual['cell']}"]
    diffs = []
    h = preview["hotspot"]
    if [h["time"], h["sector"]] != actual["cell"]:
        diffs.append(f"cell: preview (time {h['time']}, sector {h['sector']}), step {actual['cell']}")
    if preview["solver"] != actual["solver"]:
        diffs.append(f"solver flights: preview {preview['solver']}, step {actual['solver']}")
    if preview["solver_flights"] != actual["solver_flights"]:
        diffs.append(f"sub-problem flights: preview {preview['solver_flights']}, step {actual['solver_flights']}")
    if preview["rule"]["max_aircraft"] != actual["max_aircraft"]:
        diffs.append(f"max_aircraft: preview {preview['rule']['max_aircraft']}, step {actual['max_aircraft']}")
    return diffs


class PreviewWriter:
    """Writes preview.jsonl of a trace folder while the optimizer runs (xai/hooks.py holds one per trace)."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / FILE
        self._fh = open(self.path, "w", encoding="utf-8")
        self.pending: Optional[Dict[str, Any]] = None       # the last preview written, for the next step's check
        self.actual: Optional[Dict[str, Any]] = None        # what the step being evaluated selected

    def write(self, iteration: int, kept: Optional[bool], body: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {"v": VERSION, "iteration": int(iteration), "kept": kept} | body
        if not record["finished"]:
            record["next"] = int(iteration) + 1
            h = record["hotspot"]
            record["streak"]["retry"] = bool(kept is False and self.actual is not None
                                             and self.actual["cell"] == [h["time"], h["sector"]])
        if iteration > 0:
            diffs = compare(self.pending, self.actual)
            made_after = None if self.pending is None else self.pending["iteration"]
            record["check"] = {"ok": not diffs, "made_after": made_after, "differences": diffs}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.pending, self.actual = record, None
        return record

    def write_error(self, iteration: int, kept: Optional[bool], exc: BaseException) -> None:
        self._fh.write(json.dumps({"v": VERSION, "iteration": int(iteration), "kept": kept,
                                   "error": f"{type(exc).__name__}: {exc}", "check": {"ok": False}}) + "\n")
        self._fh.flush()
        self.pending, self.actual = None, None

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------------------------------------------------
# Reading and serving
# ---------------------------------------------------------------------------------------------------------------------

def read(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """(header or None, iteration -> record); a half-written last line (a live run writing) is skipped."""
    header, records = None, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n") or not line.strip():
                continue
            item = json.loads(line)
            if "header" in item:
                header = item["header"]
                continue
            records[int(item["iteration"])] = item
    return header, records


_CACHE: Dict[str, Tuple[tuple, Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], Optional[bool]]] = {}
_CACHE_LOCK = threading.Lock()


def _cached(folder: Path):
    """read() of a folder's preview.jsonl, again only when the file changed; with whether its header matches."""
    from .load import sha256_of
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
            log.warning("%s: preview.jsonl was made for another trace.jsonl; not served", folder)
    with _CACHE_LOCK:
        _CACHE[str(path)] = (key, header, records, matches)
    return header, records, matches


def preview_for(session, upto: int) -> Dict[str, Any]:
    """The preview made after step `upto` (0 = filed plan) of a live or replayed session; only for a step the
    session has delivered. KeyError (the service answers 404) when the step has not been delivered, the trace folder
    has no preview.jsonl, the file belongs to another trace, or it has no record for the step."""
    delivered = session.delivered()
    last = max((int(r["iteration"]) for r in delivered), default=0)
    if upto < 0 or upto > last:
        raise KeyError(f"step {upto} has not been delivered (last delivered: {last})")
    folder = Path(session.folder)
    if not (folder / FILE).is_file():
        raise KeyError(f"no preview data for this trace ({FILE} missing)")
    _header, records, matches = _cached(folder)
    if matches is False:
        raise KeyError(f"{FILE} does not belong to this trace")
    if upto not in records:
        raise KeyError(f"no preview record for step {upto}")
    return records[upto]


# ---------------------------------------------------------------------------------------------------------------------
# Proof against a stored trace, and the side file of a recorded trace
# ---------------------------------------------------------------------------------------------------------------------

def _lp_flights(path: Path) -> Tuple[List[int], List[int]]:
    """(flights with paths(F, ...): the solver flights; flights with planned_arrival_time(F, _): every flight of the
    sub-problem) of a sub-problem's lp file."""
    text = Path(path).read_text(encoding="utf-8")
    taken = sorted({int(x) for x in re.findall(r"^paths\((\d+),", text, flags=re.M)})
    every = sorted({int(x) for x in re.findall(r"^planned_arrival_time\((\d+),", text, flags=re.M)})
    return taken, every


def prove(trace: Path, records: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every preview against the stored trace: the preview made after step n-1 against record n (hotspot cell,
    demand, capacity, overload, the cell's flights in order, the taken flights, max_aircraft, rejections, and the
    sub-problem's flights in record n's chosen paths and in its lp file); the preview after the last step must say
    finished. One row per preview."""
    trace = Path(trace)
    stored = [json.loads(x) for x in (trace / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by_n = {int(r["iteration"]): r for r in stored}
    last = max(by_n, default=0)
    rows = []
    for n in range(0, last + 1):
        p = records.get(n)
        row: Dict[str, Any] = {"after": n, "differences": []}
        d = row["differences"]
        if p is None or "error" in p:
            d.append("no preview record" if p is None else f"error: {p['error']}")
        elif n == last:
            if not p.get("finished"):
                d.append(f"the trace ends after step {n}, the preview names step {p.get('next')}")
            row["finished"] = p.get("finished")
            row["reason"] = p.get("reason")
        else:
            r = by_n.get(n + 1)
            h, rh = p.get("hotspot") or {}, (r or {}).get("hotspot") or {}
            if r is None:
                d.append(f"no record {n + 1}")
            elif p.get("finished"):
                d.append(f"preview says finished, the trace has step {n + 1}")
            else:
                for key in ("sector", "time", "demand", "capacity", "overload"):
                    if h.get(key) != rh.get(key):
                        d.append(f"hotspot {key}: preview {h.get(key)}, trace {rh.get(key)}")
                if p["flights"] != rh.get("flights"):
                    d.append(f"cell flights: preview {p['flights']}, trace {rh.get('flights')}")
                taken = [f["id"] for f in (rh.get("flights") or [])[:rh.get("taken", 0)]]
                if p["solver"] != taken or p["rule"]["taken"] != rh.get("taken"):
                    d.append(f"solver flights: preview {p['solver']}, trace {taken}")
                params = r.get("parameters") or {}
                if p["rule"]["max_aircraft"] != params.get("max_aircraft"):
                    d.append(f"max_aircraft: preview {p['rule']['max_aircraft']}, trace {params.get('max_aircraft')}")
                if p["streak"]["rejections"] != params.get("failed_attempts_before"):
                    d.append(f"rejections: preview {p['streak']['rejections']}, "
                             f"trace {params.get('failed_attempts_before')}")
                subs = r.get("subproblems") or []
                chosen = sorted({int(f) for sub in subs for f in sub.get("chosen_paths", {})})
                if p["solver_flights"] != chosen:
                    d.append(f"sub-problem flights: preview {p['solver_flights']}, trace chosen paths {chosen}")
                files = [trace / sub["instance_file"] for sub in subs if sub.get("instance_file")]
                if not files or not all(f.is_file() for f in files):
                    d.append(f"lp file of step {n + 1} missing")
                else:
                    lp_taken, lp_every = set(), set()
                    for f in files:
                        a, b = _lp_flights(f)
                        lp_taken |= set(a)
                        lp_every |= set(b)
                    if sorted(lp_taken) != sorted(p["solver"]):
                        d.append(f"lp paths flights {sorted(lp_taken)}, preview solver {sorted(p['solver'])}")
                    if sorted(lp_every) != p["solver_flights"]:
                        d.append(f"lp flights {sorted(lp_every)}, preview {p['solver_flights']}")
                row.update({"next": n + 1, "sector": h.get("sector"), "time": h.get("time"),
                            "solver": p["solver"], "later": p["later"],
                            "changed": sorted(int(f) for f in (r.get("flight_changes") or {})
                                              if int(f) in set(p["solver_flights"])),
                            "kept": r.get("accepted")})
            if p is not None and "check" in p and not p["check"].get("ok"):
                d.append(f"hook check: {p['check']}")
        row["ok"] = not d
        rows.append(row)
    extra = sorted(k for k in records if k > last)
    if extra:
        rows.append({"after": None, "ok": False, "differences": [f"preview records past the trace: {extra}"]})
    return rows


def rerun(trace: Path, out: Path, data_dir: Optional[Path] = None, work: Optional[Path] = None,
          number_threads: int = 4) -> Dict[str, Any]:
    """Run the recorded trace again; write `out` (header + the re-run's preview.jsonl) only if the re-run reproduces
    the trace and every preview is proven against the stored trace. Returns a summary."""
    from .load import compare_traces, rerun_options, sha256_of, _commit
    from .session import OptimizerSession

    trace = Path(trace)
    run = json.loads((trace / "run.json").read_text(encoding="utf-8"))
    data_dir = Path(data_dir) if data_dir is not None else Path(run["data_dir"])
    own_work = work is None
    work = Path(tempfile.mkdtemp(prefix="preview-rerun-")) if own_work else Path(work)
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
        rows = prove(trace, records)
        summary: Dict[str, Any] = {"trace": str(trace), "seconds": seconds, "differences": diffs, "proof": rows}
        stored_load = trace / "load.jsonl"
        if stored_load.is_file() and (folder / "load.jsonl").is_file():
            body = [x for x in stored_load.read_text(encoding="utf-8").splitlines() if not x.startswith('{"header"')]
            summary["load_jsonl_equal"] = body == (folder / "load.jsonl").read_text(encoding="utf-8").splitlines()
        if diffs or not all(r["ok"] for r in rows):
            summary["written"] = None
            return summary
        header = {"v": VERSION, "trace_sha256": sha256_of(trace / "trace.jsonl"), "records": len(records),
                  "made_by": "re-run reproducing the trace", "optimizer_commit": _commit(),
                  "number_threads": number_threads,
                  "compared": "every trace.jsonl record without solve_seconds and TOTAL-TIME-TO-THIS-POINT; run.json "
                              "without data_dir; encoding.lp and lp/*.lp byte for byte",
                  "proven": "preview after step n equals record n+1 (hotspot, cell flights, solver flights, "
                            "sub-problem flights in the record and its lp file); the last preview says finished"}
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
    p = argparse.ArgumentParser(description="preview.jsonl of a recorded trace from a re-run that reproduces it")
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
