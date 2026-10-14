"""K08 demand vs capacity per sector and time step (xai/load.py and its hook in the optimizer).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

The filed grid is rebuilt here from the instance files alone (flights.csv, navaid_sector_assignment.csv, sectors.csv,
airports.csv) and compared with the optimizer's record; the sum check is recomputed from every record; the split of
the practice run is compared with the optimizer's own navpoint-to-sector matrix.
"""
import atexit
import csv
import json
import math
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai import load as load_mod  # noqa: E402
from src.aspaeroflow.xai.session import OptimizerSession  # noqa: E402

FIXTURES = HERE / "fixtures"
EAST_ASIA = FIXTURES / "EAST-ASIA-3x3-V2_0000010_SEED150699"
PRACTICE = FIXTURES / "CE-NAVPT-PRACTICE"
PRACTICE_TRACE = FIXTURES / "NAVPT_PRACTICE_TRACE"
PRACTICE_OPTIONS = {"timestep_granularity": 4, "max_number_sectors": 100000, "seed": 11904657,
                    "max_delay_per_iteration": 9, "number_threads": 4}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def instance_grid(instance):
    """(demand {(sector, t): flights}, capacity {sector: c}, airport-only sectors) of the filed plan, from the
    instance files only: a flight is in its navpoint's sector at the navpoint's time step, and between two navpoints
    the first half of the gap (rounded down) in the previous navpoint's sector, the rest in the next one's; a sector's
    capacity is the maximum over its navpoints' capacities."""
    sector_of = {int(r["Navaid_ID"]): int(r["Sector_ID"]) for r in _rows(instance / "navaid_sector_assignment.csv")}
    navpoint_capacity = {int(r["Sector_ID"]): int(float(r["Capacity"])) for r in _rows(instance / "sectors.csv")}
    airports = {int(float(r["Airport_Vertex"])) for r in _rows(instance / "airports.csv")}
    members = defaultdict(set)
    for v, s in sector_of.items():
        members[s].add(v)
    capacity = {s: max(navpoint_capacity[v] for v in vs) for s, vs in members.items()}
    airport_only = {s for s, vs in members.items() if vs <= airports}
    hops = defaultdict(list)
    for r in _rows(instance / "flights.csv"):
        hops[int(float(r["Flight_ID"]))].append((int(float(r["Time"])), int(float(r["Position"]))))
    demand = defaultdict(int)
    for route in hops.values():
        route.sort()
        cells = {route[0][0]: sector_of[route[0][1]]}
        for (tp, p), (tn, n) in zip(route, route[1:]):
            for k in range(1, tn - tp + 1):
                cells[tp + k] = sector_of[p] if k <= math.floor((tn - tp) / 2) else sector_of[n]
        for t, s in cells.items():
            demand[(s, t)] += 1
    return dict(demand), capacity, airport_only


def records_of(folder):
    _header, records = load_mod.read(Path(folder) / load_mod.FILE)
    return records


def overload_of(record):
    """Sum of max(0, demand - capacity) over the record's rows (a time step where no navpoint forms the sector has
    capacity 0 in the optimizer)."""
    return sum(max(0, d - (c or 0)) for row in record["sectors"] for d, c in zip(row["demand"], row["capacity"]))


def trace_records(folder):
    return [json.loads(x) for x in (Path(folder) / "trace.jsonl").read_text().splitlines() if x.strip()]


_practice = None


def practice_run():
    """(trace folder, load records, session) of the practice run, run once per interpreter; the session also holds
    the optimizer's navpoint-to-sector matrix after each step (assignments)."""
    global _practice
    if _practice is None:
        tmp = Path(tempfile.mkdtemp(prefix="aspaeroflow-load-test-"))
        atexit.register(shutil.rmtree, tmp, True)
        session = OptimizerSession(PRACTICE, tmp / "trace", PRACTICE_OPTIONS)
        session.begin()
        assignments = {0: session.app.optimization_dto["navaid_sector_time_assignment"].copy()}
        while session.status != "finished":
            summary = session.step()
            if summary is None:
                break
            assignments[summary["iteration"]] = session.app.optimization_dto["navaid_sector_time_assignment"].copy()
        session.assignments = assignments
        _practice = (tmp / "trace", records_of(tmp / "trace"), session)
    return _practice


class FiledGrid(unittest.TestCase):
    """The record of the filed plan equals the grid rebuilt from the instance files."""

    def check(self, instance, record, initial_overload):
        demand, capacity, airport_only = instance_grid(instance)
        rows = {row["sector"]: row for row in record["sectors"]}
        self.assertEqual(set(rows), set(capacity) - airport_only)
        lo, hi = record["from"], record["to"]
        for (s, t), d in demand.items():
            if s in airport_only:
                continue
            self.assertTrue(lo <= t <= hi, (s, t))
            self.assertEqual(rows[s]["demand"][t - lo], d, (s, t))
        for s, row in rows.items():
            self.assertEqual(sum(row["demand"]), sum(d for (x, _t), d in demand.items() if x == s), s)
            self.assertEqual(set(row["capacity"]), {capacity[s]}, s)
            self.assertEqual([(p["from"], p["kind"], p["capacity"]) for p in row["parts"]], [(0, "filed", capacity[s])])
        expected = sum(max(0, d - capacity[s]) for (s, _t), d in demand.items())
        self.assertEqual(record["overload"], expected)
        self.assertEqual(record["overload"], initial_overload)
        self.assertTrue(record["check"]["ok"], record["check"])

    def test_east_asia(self):
        _plain, _traced, folder = shared_run.get()
        run = json.loads((folder / "run.json").read_text())
        self.check(EAST_ASIA, records_of(folder)[0], run["initial_overload"])

    def test_practice(self):
        run = json.loads((PRACTICE_TRACE / "run.json").read_text())
        self.check(PRACTICE, practice_run()[1][0], run["initial_overload"])


class SumCheck(unittest.TestCase):
    """At every step the sum of max(0, demand - capacity) over the grid is the trace's OVERLOAD."""

    def check(self, folder, records):
        trace = {r["iteration"]: r for r in trace_records(folder)}
        run = json.loads((Path(folder) / "run.json").read_text())
        self.assertEqual(sorted(records), [0] + sorted(trace))
        for n, record in sorted(records.items()):
            recorded = run["initial_overload"] if n == 0 else trace[n]["objectives"]["OVERLOAD"]
            self.assertTrue(record["check"]["ok"], (n, record["check"]))
            self.assertEqual(record["overload"], recorded, n)
            self.assertEqual(overload_of(record), recorded, n)
            if n > 0:   # the step's hotspot cell, as the trace recorded it, in the plan before the step
                hot, before = trace[n]["hotspot"], records[n - 1]
                row = next(r for r in before["sectors"] if r["sector"] == hot["sector"])
                t = hot["time"] - before["from"]
                self.assertEqual([row["demand"][t], row["capacity"][t]], [hot["demand"], hot["capacity"]], n)

    def test_east_asia(self):
        _plain, _traced, folder = shared_run.get()
        self.check(folder, records_of(folder))

    def test_practice(self):
        folder, records, _session = practice_run()
        self.check(folder, records)


class Practice(unittest.TestCase):
    """The practice run of the study, run again: it reproduces the study trace, and step 2's split spans the time
    steps the optimizer's matrix gives it (from the hotspot time to the end of the horizon), not only the hotspot."""

    def test_reproduces_the_study_trace(self):
        folder, _records, _session = practice_run()
        stable = [load_mod._stable(r) for r in trace_records(folder)]
        self.assertEqual(stable, [load_mod._stable(r) for r in trace_records(PRACTICE_TRACE)])

    def test_split_spans_its_time_steps(self):
        folder, records, session = practice_run()
        trace = {r["iteration"]: r for r in trace_records(folder)}
        changes = trace[2]["sector_changes"]
        self.assertEqual(changes["time_index"], 35)                     # the trace names only the hotspot time
        self.assertEqual(sorted(changes["post_sector_config"]), ["10", "25"])
        record, matrix = records[2], session.assignments[2]
        horizon = record["horizon"]
        self.assertEqual(matrix.shape[1], horizon)
        rows = {row["sector"]: row for row in record["sectors"]}
        self.assertEqual([(p["from"], p["to"], p["kind"]) for p in rows[10]["parts"]],
                         [(0, 34, "filed"), (35, horizon - 1, "split")])
        self.assertEqual([(p["from"], p["to"], p["kind"], p["navpoints"]) for p in rows[25]["parts"]],
                         [(35, horizon - 1, "split", changes["post_sector_config"]["25"]["vertices"])])
        self.assertGreater(horizon - 1 - 35, 40)                         # many hours, not one time step
        # every part of every sector is the optimizer's matrix: exactly these navpoints form the sector at each step
        for n, rec in records.items():
            a = session.assignments[n]
            for row in rec["sectors"]:
                covered = set()
                for p in row["parts"]:
                    for t in range(p["from"], p["to"] + 1):
                        self.assertEqual(sorted(int(v) for v in (a[:, t] == row["sector"]).nonzero()[0]), p["navpoints"])
                        covered.add(t)
                self.assertEqual(covered, {t for t in range(a.shape[1]) if (a[:, t] == row["sector"]).any()})
        # before the step the sector was whole over the same span
        self.assertEqual([(p["from"], p["to"], p["kind"]) for p in
                          next(r for r in records[1]["sectors"] if r["sector"] == 10)["parts"]],
                         [(0, records[1]["horizon"] - 1, "filed")])


class Writer(unittest.TestCase):
    """LoadWriter on small hand-made matrices: a rejected step's line refers to the plan before, and the check fails
    when the grid does not give the recorded overload."""

    def test_rejected_step_and_failing_check(self):
        import numpy as np
        a = np.array([[0] * 5, [0] * 5, [2, 2, 1, 1, 1], [3] * 5])      # navpoint 3 is an airport, alone in sector 3
        d = np.array([[0, 2, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [5, 0, 0, 0, 0]])
        c = np.array([[1] * 5, [0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [9] * 5])
        filed = np.array([[0] * 5, [0] * 5, [2] * 5, [3] * 5])
        with tempfile.TemporaryDirectory() as tmp:
            w = load_mod.LoadWriter(Path(tmp))
            w.write(0, None, 1, None, a, d, c, filed, [3])
            w.write(1, False, 1, {"sector": 0, "time": 1, "demand": 2, "capacity": 1}, a, d, c, filed, [3])
            w.write(2, True, 0, {"sector": 0, "time": 1, "demand": 2, "capacity": 1}, a, d, c, filed, [3])
            w.close()
            lines = [json.loads(x) for x in (Path(tmp) / load_mod.FILE).read_text().splitlines()]
            _header, records = load_mod.read(Path(tmp) / load_mod.FILE)
        self.assertNotIn("sectors", lines[1])
        self.assertEqual(lines[1]["state_of"], 0)
        self.assertEqual({k: v for k, v in records[1].items() if k not in ("iteration", "kept", "state_of", "check")},
                         {k: v for k, v in records[0].items() if k not in ("iteration", "kept", "state_of", "check")})
        self.assertTrue(records[1]["check"]["ok"], records[1]["check"])
        self.assertFalse(records[2]["check"]["ok"])                      # recorded 0, the grid gives 1
        self.assertEqual(records[0]["overload"], 1)
        self.assertEqual([r["sector"] for r in records[0]["sectors"]], [0, 1, 2])
        self.assertEqual((records[0]["from"], records[0]["to"]), (0, 3))
        self.assertEqual(records[0]["sectors"][1]["capacity"], [None, None, 1, 1])
        self.assertEqual([(p["from"], p["to"], p["navpoints"], p["filed_sectors"], p["kind"])
                          for p in records[0]["sectors"][1]["parts"]], [(2, 4, [2], [2], "filed")])
        self.assertEqual([(p["from"], p["to"], p["navpoints"], p["kind"]) for p in records[0]["sectors"][2]["parts"]],
                         [(0, 1, [2], "filed")])


class SideFile(unittest.TestCase):
    """The side file of a recorded trace is written only when a re-run reproduces the trace."""

    def test_rerun_writes_only_when_identical(self):
        _plain, _traced, folder = shared_run.get()
        with tempfile.TemporaryDirectory() as tmp:
            stored = Path(tmp) / "stored"
            shutil.copytree(folder, stored)
            (stored / load_mod.FILE).unlink()
            out = Path(tmp) / "load.jsonl"
            summary = load_mod.rerun(stored, out, data_dir=EAST_ASIA, number_threads=20)
            self.assertEqual(summary["differences"], [])
            self.assertTrue(out.exists())
            header, records = load_mod.read(out)
            self.assertEqual(header["trace_sha256"], load_mod.sha256_of(stored / "trace.jsonl"))
            self.assertEqual(records, records_of(folder))
            # one changed value in the stored trace: nothing is written
            lines = (stored / "trace.jsonl").read_text().splitlines()
            first = json.loads(lines[0])
            first["objectives"]["ARRIVAL-DELAY"] += 1
            (stored / "trace.jsonl").write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
            out2 = Path(tmp) / "load2.jsonl"
            summary = load_mod.rerun(stored, out2, data_dir=EAST_ASIA, number_threads=20)
            self.assertEqual(summary["written"], None)
            self.assertIn("objectives", summary["differences"][0])
            self.assertFalse(out2.exists())


if __name__ == "__main__":
    unittest.main()
