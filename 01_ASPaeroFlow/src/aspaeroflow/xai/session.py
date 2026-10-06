"""An ASPaeroFlow run that a caller drives one iteration at a time, and its replay.

OptimizerSession wraps Main (begin / step / finish) with the trace switched on, so every step
yields a small record (xai/trace.py) and every recorded iteration can be explained or questioned
with locks (xai/contrastive.py). ReplaySession serves a finished trace the same way without an
optimizer, e.g. for a study session in which every participant must see the same run.

Neither class knows about transport; 07_heuristic_controller/session_service.py puts HTTP + SSE
in front of them.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contrastive import IterationExplainer, Lock
from .trace import TraceReader

OPTIMIZER_DIR = Path(__file__).resolve().parents[3]          # 01_ASPaeroFlow/


def _load_cli():
    """01_ASPaeroFlow/main.py as a module (parse_cli, make_app)."""
    spec = importlib.util.spec_from_file_location("aspaeroflow_cli", OPTIMIZER_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """What an event carries about an iteration: small, no instance text, no trajectories of unchanged flights."""
    return {
        "iteration": record["iteration"],
        "accepted": record["accepted"],
        "hotspot": record["hotspot"],
        "objectives": record["objectives"],
        "decision_flights": sorted(int(f) for f in record["subproblems"][0]["chosen_paths"]) if record["subproblems"] else [],
        "flight_changes": record.get("flight_changes", {}),
        "sector_changes": record.get("sector_changes", {}),
        "parameters": record.get("parameters", {}),
    }


def instance_graph(data_dir: Path) -> Dict[str, Any]:
    """Vertices (with coordinates if the instance has them), edges, airports and the initial sectors."""
    data_dir = Path(data_dir)

    def rows(name):
        path = data_dir / name
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    names = {int(r["VERTEX_ID"]): r["IDENTIFIER"] for r in rows("mappings/vertex_map.csv")}
    coords = {r["IDENTIFIER"]: (float(r["LAT"]), float(r["LON"])) for r in rows("navgraph/vertices.csv")}
    airports = {int(r["Airport_Vertex"]) for r in rows("airports.csv")}
    sectors = {int(r["Navaid_ID"]): int(r["Sector_ID"]) for r in rows("navaid_sector_assignment.csv")}
    capacity = {int(r["Sector_ID"]): int(float(r["Capacity"])) for r in rows("sectors.csv")}
    edges = [(int(float(r["source"])), int(float(r["target"]))) for r in rows("graph_edges.csv")]
    # flights per edge of the filed trajectories (either direction), for line widths
    edge_flights: Dict[tuple, int] = {}
    previous = None
    for r in rows("flights.csv"):
        current = (r["Flight_ID"], int(float(r["Position"])))
        if previous is not None and previous[0] == current[0] and previous[1] != current[1]:
            key = tuple(sorted((previous[1], current[1])))
            edge_flights[key] = edge_flights.get(key, 0) + 1
        previous = current

    ids = sorted(set(names) | set(sectors) | {v for e in edges for v in e})
    vertices = []
    for v in ids:
        name = names.get(v, str(v))
        lat, lon = coords.get(name, (None, None))
        if lat is None:
            m = re.search(r"Y(\d+)X(\d+)", name)          # grid instances: position from the name
            if m:
                lat, lon = -float(m.group(1)), float(m.group(2))
        vertices.append({"id": v, "name": name, "lat": lat, "lon": lon, "airport": v in airports,
                         "sector": sectors.get(v), "capacity": capacity.get(v)})
    return {"vertices": vertices,
            "edges": [{"source": a, "target": b, "flights": edge_flights.get(tuple(sorted((a, b))), 0)} for a, b in edges],
            "coordinates": "geographic" if coords else "grid"}


class _ExplainingSession:
    """What live and replayed sessions share: the trace and the questions about it."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._trace: Optional[TraceReader] = None
        self._explainers: "OrderedDict[int, IterationExplainer]" = OrderedDict()
        self._explain_lock = threading.Lock()

    def trace(self) -> TraceReader:
        if self._trace is None:
            self._trace = TraceReader(self.folder)
        else:
            self._trace.refresh()
        return self._trace

    def explainer(self, iteration: int) -> IterationExplainer:
        with self._explain_lock:
            if iteration in self._explainers:
                self._explainers.move_to_end(iteration)
                return self._explainers[iteration]
            trace = self.trace()
            if iteration not in trace.iterations:
                raise KeyError(f"iteration {iteration} is not (yet) in the trace")
            ex = IterationExplainer(trace, iteration)
            self._explainers[iteration] = ex
            while len(self._explainers) > 16:
                self._explainers.popitem(last=False)
            return ex

    def _run_extras(self) -> Dict[str, Any]:
        run = self.trace().run
        return {"sector_overload": run.get("initial_sector_overload", {}),
                "initial_objectives": run.get("initial_objectives", {})}

    def iteration(self, iteration: int) -> Dict[str, Any]:
        trace = self.trace()
        if iteration not in trace.iterations:
            raise KeyError(f"iteration {iteration} is not (yet) in the trace")
        record = dict(trace.iterations[iteration])
        record["subproblems"] = [{k: v for k, v in sub.items() if k != "instance_file"} for sub in record["subproblems"]]
        return record

    def explain(self, iteration: int, question: str, flight: Optional[int] = None) -> Dict[str, Any]:
        ex = self.explainer(iteration)
        with self._explain_lock:      # one clingo control per iteration; queries on it run one at a time
            if question == "hotspot":
                return ex.why_hotspot()
            if question == "flight":
                if flight is None:
                    raise ValueError("question 'flight' needs a flight id")
                return ex.why_flight(int(flight))
            if question == "sectors":
                return ex.why_sectors()
            if question == "tie":
                return ex.tie_check()
            if question == "alternatives":
                return ex.alternatives()
        raise ValueError(f"unknown question: {question}")

    def what_if(self, iteration: int, locks: List[str]) -> Dict[str, Any]:
        ex = self.explainer(iteration)
        with self._explain_lock:
            return ex.what_if([Lock.parse(text) for text in locks])


class OptimizerSession(_ExplainingSession):
    """A live run with its trace in `folder`; options are main.py's long options without the dashes."""

    def __init__(self, data_dir: Path, folder: Path, options: Optional[Dict[str, Any]] = None):
        super().__init__(folder)
        cli = _load_cli()
        argv = [f"--data-dir={data_dir}", f"--encoding-path={OPTIMIZER_DIR / 'encoding.lp'}",
                "--save-results=false", f"--xai-trace-dir={folder}"]
        argv += [f"--{key.replace('_', '-')}={value}" for key, value in (options or {}).items()]
        self.args = cli.parse_cli(argv)
        self.app = cli.make_app(self.args, xai_trace_dir=folder)
        self.data_dir = Path(data_dir)
        self.status = "created"
        self.records: List[Dict[str, Any]] = []
        self.final: Optional[Dict[str, Any]] = None
        self._step_lock = threading.Lock()

    def begin(self) -> Dict[str, Any]:
        with self._step_lock:
            dto = self.app.begin()
            self.status = "ready" if self.app.has_overload() else "finished"
            self.initial = self.trace().run.get("initial_objectives") or {"OVERLOAD": int(dto["number_of_conflicts"]), "ITERATION": 0}
            return {"status": self.status, "objectives": self.initial}

    def step(self) -> Optional[Dict[str, Any]]:
        """One iteration; None once the plan has no overload left (or no progress is possible)."""
        with self._step_lock:
            if self.status in ("created", "finished"):
                return None
            outcome = self.app.step()
            record = self.app._xai_trace.last
            if record is not None and (not self.records or self.records[-1]["iteration"] != record["iteration"]):
                self.records.append(record)
            if outcome == "terminate" or not self.app.has_overload():
                self._finish()
            return record_summary(record) if record is not None else None

    def _finish(self) -> None:
        self.app.finish()
        self.status = "finished"
        last = self.records[-1]["objectives"] if self.records else self.initial
        self.final = {"objectives": last, "iterations": len(self.records),
                      "accepted": sum(1 for r in self.records if r["accepted"])}

    def graph(self) -> Dict[str, Any]:
        return instance_graph(self.data_dir) | self._run_extras()


class ReplaySession(_ExplainingSession):
    """A finished trace served like a live run: step() hands out the next recorded iteration."""

    def __init__(self, folder: Path, data_dir: Optional[Path] = None):
        """`data_dir` overrides the instance folder recorded in the trace (needed when the trace was
        written on another machine or outside the container that replays it)."""
        super().__init__(folder)
        self.records = list(self.trace())
        self.cursor = 0
        self.status = "ready" if self.records else "finished"
        run = self.trace().run
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(run["data_dir"]) if run.get("data_dir") else None
        self.initial = run.get("initial_objectives") or {"OVERLOAD": run.get("initial_overload"), "ITERATION": 0}
        self.final: Optional[Dict[str, Any]] = None

    def begin(self) -> Dict[str, Any]:
        return {"status": self.status, "objectives": self.initial}

    def step(self) -> Optional[Dict[str, Any]]:
        if self.cursor >= len(self.records):
            return None
        record = self.records[self.cursor]
        self.cursor += 1
        if self.cursor == len(self.records):
            self.status = "finished"
            self.final = {"objectives": record["objectives"], "iterations": len(self.records),
                          "accepted": sum(1 for r in self.records if r["accepted"])}
        return record_summary(record)

    def graph(self) -> Dict[str, Any]:
        base = instance_graph(self.data_dir) if self.data_dir else {"vertices": [], "edges": []}
        return base | self._run_extras()


def precompute(folder: Path, out: Path) -> None:
    """Answers to the standard questions of every accepted iteration, for a replay without solving."""
    session = ReplaySession(folder)
    answers: Dict[str, Any] = {}
    for record in session.records:
        n = record["iteration"]
        if not record["accepted"]:
            continue
        ex = session.explainer(n)
        answers[str(n)] = {
            "hotspot": ex.why_hotspot(),
            "flights": {str(f): ex.why_flight(f) for f in ex.sub.decision_flights},
            "sectors": ex.why_sectors(),
            "tie": ex.tie_check(),
        }
    Path(out).write_text(json.dumps(answers, indent=1), encoding="utf-8")
