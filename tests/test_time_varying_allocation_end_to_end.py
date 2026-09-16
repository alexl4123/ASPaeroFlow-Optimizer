"""A navpoint that changes sector at t=8 changes what the solvers compute; a schedule that changes
nothing changes nothing.

Fixture: the 10-flight EAST-ASIA-3x3 instance (ASPaeroFlow-DataGenerator,
experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2/0000010_SEED42), timestep granularity 1,
embedded below. Sectors 0 = {0,1,2}, 4 = {3,4,5}, 6 = {6,7,8}, each with capacity 1; airports
9..16 are sectors of their own. Three bundles are written:

  static   no navaid_sector_schedule.csv
  t0       a schedule with every navpoint once, at From_Time 0 (what the generator writes today)
  moved    t0 plus "1,4,8": navpoint 1 leaves sector 0 for sector 4 at t=8
  opened   moved plus "2,2,10": navpoint 2 leaves sector 0 at t=10 and opens sector 2

Every expectation about a moved navpoint is computed from the allocation array and then checked
against the solver, and also checked to differ from the static instance, so that a solver which
ignored the schedule would fail.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "02_ASP")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common import navpoint_sector_allocation as nsa  # noqa: E402

FILES = {
    "graph_edges.csv": "source,target,dist_m\n" + "\n".join(
        "0,1,1351188.334 0,2,993085.935 0,3,1635045.387 1,4,1351188.334 1,2,1635045.387 "
        "1,3,993085.935 1,5,1635045.387 4,3,1635045.387 4,5,993085.935 2,3,1243589.281 "
        "2,6,993085.935 2,7,1538525.388 3,5,1243589.281 3,6,1538525.388 3,7,993085.935 "
        "3,8,1538525.388 5,7,1538525.388 5,8,993085.935 6,7,1105884.217 7,8,1105884.217 "
        "9,1,634161.212 10,1,562044.302 11,5,474658.817 12,3,395248.691 13,5,354800.083 "
        "14,7,424654.855 15,0,143053.731 16,0,105938.504".split()) + "\n",
    "flights.csv": "Flight_ID,Position,Time\n" + "\n".join(
        "0,15,2 0,0,3 0,1,5 0,10,6 1,11,4 1,5,5 1,3,7 1,12,8 2,16,4 2,0,5 2,1,7 2,5,9 2,11,10 "
        "3,14,5 3,7,6 3,5,8 3,11,9 4,15,7 4,0,8 4,1,10 4,5,12 4,11,13 5,9,8 5,1,9 5,2,11 5,7,13 "
        "5,14,14 6,15,9 6,0,10 6,3,12 6,12,13 7,14,15 7,7,16 7,5,18 7,11,19 8,11,16 8,5,17 "
        "8,7,19 8,14,20 9,11,19 9,5,20 9,7,22 9,14,23".split()) + "\n",
    "airplanes.csv": "Airplane_ID,Speed_kts\n" + "\n".join(
        "0,450.0 1,430.0 2,480.0 3,450.0 4,480.0 5,480.0 6,480.0".split()) + "\n",
    "airplane_flight_assignment.csv": "Airplane_ID,Flight_ID\n" + "\n".join(
        "0,0 1,1 2,2 2,9 3,3 3,8 4,4 5,5 5,7 6,6".split()) + "\n",
    "airports.csv": "Airport_Vertex\n" + "\n".join(str(a) for a in range(9, 17)) + "\n",
    "sectors.csv": "Sector_ID,Capacity\n" + "\n".join(
        [f"{s},1" for s in range(9)] + [f"{s},60000" for s in range(9, 17)]) + "\n",
    "navaid_sector_assignment.csv": "Navaid_ID,Sector_ID\n" + "\n".join(
        f"{n},{s}" for n, s in [(0, 0), (1, 0), (2, 0), (3, 4), (4, 4), (5, 4), (6, 6), (7, 6),
                                (8, 6)] + [(a, a) for a in range(9, 17)]) + "\n",
}
T0_ROWS = "Navaid_ID,Sector_ID,From_Time\n" + "".join(
    f"{row.split(',')[0]},{row.split(',')[1]},0\n"
    for row in FILES["navaid_sector_assignment.csv"].splitlines()[1:])
SCHEDULES = {"static": None, "t0": T0_ROWS, "moved": T0_ROWS + "1,4,8\n",
             "opened": T0_ROWS + "1,4,8\n2,2,10\n"}
MAX_TIME, TG = 24, 1
MOVED_CELLS = {(2, 8), (4, 10), (4, 11), (5, 9), (5, 10)}      # (flight, t): sector 0 -> 4

TMP = None
BUNDLES = {}


def setUpModule():
    global TMP
    TMP = tempfile.TemporaryDirectory()
    for name, schedule in SCHEDULES.items():
        d = Path(TMP.name) / name
        d.mkdir()
        for fname, text in FILES.items():
            (d / fname).write_text(text)
        if schedule is not None:
            (d / nsa.SCHEDULE_FILENAME).write_text(schedule)
        BUNDLES[name] = d


def tearDownModule():
    TMP.cleanup()


def load(bundle, fname):
    return np.loadtxt(bundle / fname, delimiter=",", skiprows=1, dtype=float, ndmin=2).astype(int)


def allocation(bundle):
    fl, af, ns = (load(bundle, f) for f in ("flights.csv", "airplane_flight_assignment.csv",
                                            "navaid_sector_assignment.csv"))
    return nsa.build_assignment(fl, af, ns, MAX_TIME, TG,
                                schedule=nsa.load_schedule_for(bundle / "navaid_sector_assignment.csv"),
                                airports=load(bundle, "airports.csv").ravel())


def predicted_sectors_and_overload(bundle):
    """Flight -> sector per timestep, and total overload, from the allocation array alone."""
    translate = importlib.import_module("translate").TranslateCSVtoLogicProgram
    A = allocation(bundle)
    fl, af = load(bundle, "flights.csv"), load(bundle, "airplane_flight_assignment.csv")
    matrix, _ = translate.instance_to_matrix(fl, af, A.shape[1], TG, A)
    loads = np.zeros_like(A)
    for f, t in zip(*np.nonzero(matrix >= 0)):
        loads[matrix[f, t], t] += 1
    capacity = translate.capacity_time_matrix(load(bundle, "sectors.csv"), A.shape[1], TG, A)
    return matrix, int(np.maximum(loads - capacity, 0).sum())


class TestAsp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import clingo  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("clingo not available")
        cls.translate = importlib.import_module("translate").TranslateCSVtoLogicProgram
        cls.encoding = (REPO / "02_ASP/encoding.lp").read_text() + "arrival_delay_metric(signed).\n"

    def facts(self, name, ds, gd=0, rr=0):
        d = BUNDLES[name]
        tr = self.translate()
        atoms = tr.main(d / "graph_edges.csv", d / "flights.csv", d / "sectors.csv", d / "airports.csv",
                        d / "airplanes.csv", d / "airplane_flight_assignment.csv",
                        d / "navaid_sector_assignment.csv", REPO / "02_ASP/encoding.lp",
                        TG, MAX_TIME, 6, gd, rr, ds)
        return atoms, tr

    def solve(self, program, options=()):
        import clingo
        ctl = clingo.Control(list(options))
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        found = {}

        def on_model(model):
            found["atoms"] = model.symbols(atoms=True)
        result = ctl.solve(on_model=on_model)
        return result, found.get("atoms"), ctl

    @staticmethod
    def given(tr):
        """A constraint that the result be exactly the allocation array, at every timestep."""
        A, listed = tr.navaid_sector_time_assignment, tr.navaid_sector[:, 0]
        facts = [f"given({n},{A[n, t]},{t})." for n in listed for t in range(MAX_TIME * TG + 1)]
        return "\n".join(facts) + "\n:- given(N,S,T), time(T), not navpoint_sector(N,S,T).\n"

    def test_a_from_time_0_schedule_emits_the_static_facts(self):
        for ds in (0, 1, 2):
            with self.subTest(ds=ds):
                self.assertEqual(self.facts("t0", ds)[0], self.facts("static", ds)[0])

    def test_no_schedule_rule_grounds_on_a_static_instance(self):
        new_predicates = {"navpoint_sector_scheduled", "navpoint_sector_given",
                          "navpoint_sector_time_varying", "navpoint_sector_candidate",
                          "restricted_epoch_at", "navpoint_sector_restricted_sector_allocation_from"}
        for ds in (0, 1, 2):
            with self.subTest(ds=ds):
                atoms, tr = self.facts("static", ds)
                result, model, ctl = self.solve(self.encoding + "\n".join(atoms) + self.given(tr),
                                                ["--opt-mode=ignore"])
                self.assertTrue(result.satisfiable)
                grounded = [(name, arity) for name, arity, positive in ctl.symbolic_atoms.signatures
                            if name in new_predicates
                            and any(True for _ in ctl.symbolic_atoms.by_signature(name, arity, positive))]
                self.assertEqual(grounded, [])

    def test_a_moved_navpoint_changes_occupancy_and_overload(self):
        results = {}
        for name in ("static", "moved", "opened"):
            atoms, _ = self.facts(name, ds=0)
            result, model, _ = self.solve(self.encoding + "\n".join(atoms))
            self.assertTrue(result.satisfiable, name)
            matrix = np.full((10, MAX_TIME * TG + 1), -1)
            per_cell = defaultdict(set)
            overload = 0
            for s in model:
                if s.name == "flight":
                    f, sec, t = (a.number for a in s.arguments)
                    matrix[f, t] = sec
                elif s.name == "overload":
                    overload += s.arguments[2].number
                elif s.name == "navpoint_sector":
                    n, sec, t = (a.number for a in s.arguments)
                    per_cell[(n, t)].add(sec)
            expected_matrix, expected_overload = predicted_sectors_and_overload(BUNDLES[name])
            w = matrix.shape[1]
            with self.subTest(instance=name):
                self.assertTrue(np.array_equal(matrix, expected_matrix[:, :w]))
                self.assertEqual(overload, expected_overload)
                self.assertTrue(all(len(v) == 1 for v in per_cell.values()))
                nav1 = [next(iter(per_cell[(1, t)])) for t in range(w)]
                self.assertEqual(nav1, [0] * 8 + [4] * (w - 8) if name != "static" else [0] * w)
            results[name] = (matrix, overload)
        static, moved = results["static"], results["moved"]
        self.assertEqual((static[1], moved[1]), (9, 7))
        self.assertEqual({(int(f), int(t)) for f, t in zip(*np.nonzero(static[0] != moved[0]))}, MOVED_CELLS)

    def test_the_schedule_is_a_solution_under_every_sector_allocation_mode(self):
        for name in ("moved", "opened"):
            for ds in (0, 1, 2):
                with self.subTest(instance=name, ds=ds):
                    atoms, tr = self.facts(name, ds)
                    result, model, _ = self.solve(self.encoding + "\n".join(atoms) + self.given(tr),
                                                  ["--opt-mode=ignore"])
                    self.assertTrue(result.satisfiable)
                    self.assertEqual([s for s in model if s.name == "reconfig"], [])


class TestPythonSolvers(unittest.TestCase):
    """ASPaeroFlow, the MIP and the capacity analysis, run as the benchmark runs them."""

    def run_solver(self, script, bundle, *extra):
        out = Path(TMP.name) / "results" / script.replace("/", "_") / bundle.name
        cmd = [sys.executable, "-u", str(REPO / script),
               *(f"--{flag}={bundle / fname}" for flag, fname in [
                   ("graph-path", "graph_edges.csv"), ("sectors-path", "sectors.csv"),
                   ("flights-path", "flights.csv"), ("airports-path", "airports.csv"),
                   ("airplanes-path", "airplanes.csv"), ("airplane-flight-path", "airplane_flight_assignment.csv"),
                   ("navaid-sector-path", "navaid_sector_assignment.csv")]),
               f"--timestep-granularity={TG}", "--verbosity=0", "--seed=11904657",
               f"--encoding-path={REPO / '01_ASPaeroFlow/encoding.lp'}", *extra]
        if "ANALYZE" not in script:
            cmd += ["--wandb-enabled=false", "--results-format=csv", f"--results-root={out.parent}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=TMP.name, timeout=600)
        lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
        return proc, lines, out

    def assert_follows_allocation(self, results_dir):
        """Each cell of the final flight->sector matrix is its navpoint's sector at that timestep."""
        inst = np.loadtxt(results_dir / "converted_instance_matrix.csv", delimiter=",", ndmin=2).astype(int)
        navs = np.loadtxt(results_dir / "converted_navpoint_matrix.csv", delimiter=",", ndmin=2).astype(int)
        A = np.loadtxt(results_dir / "navaid_sector_time_assignment.csv", delimiter=",", ndmin=2).astype(int)
        for f in range(navs.shape[0]):
            ts = np.flatnonzero(navs[f] >= 0)
            for a, b in zip(ts[:-1], ts[1:]):
                for t in range(a + 1, b + 1):
                    nav = navs[f, a] if t - a <= (b - a) // 2 else navs[f, b]
                    self.assertEqual(inst[f, t], A[nav, min(t, A.shape[1] - 1)], (f, t))

    def test_aspaeroflow(self):
        try:
            import zmq, joblib  # noqa: F401,E401
        except ImportError:
            self.skipTest("ASPaeroFlow dependencies not available")
        first = {}
        for name in ("static", "t0", "moved"):
            proc, lines, out = self.run_solver("01_ASPaeroFlow/main.py", BUNDLES[name],
                                               "--capacity-management-enabled=False",
                                               "--max-explored-vertices=3", "--max-delay-per-iteration=5")
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            first[name] = lines[0]["OVERLOAD"]
            self.assertEqual(first[name], predicted_sectors_and_overload(BUNDLES[name])[1])
            A = np.loadtxt(out / "navaid_sector_time_assignment.csv", delimiter=",").astype(int)
            self.assertEqual(A[1, :24].tolist(), [0] * 24 if name != "moved" else [0] * 8 + [4] * 16)
            self.assert_follows_allocation(out)
            if name == "t0":
                static_out = out.parent / "static"
                for f in ("navaid_sector_time_assignment.csv", "converted_instance_matrix.csv"):
                    self.assertEqual((out / f).read_text(), (static_out / f).read_text())
        self.assertEqual(first, {"static": 9, "t0": 9, "moved": 7})

    def test_mip(self):
        try:
            import gurobipy
            gurobipy.Env().dispose()
        except Exception as exc:                                  # not installed or not licensed
            self.skipTest(f"Gurobi unavailable: {exc}")
        first = {}
        for name in ("static", "moved"):
            proc, lines, out = self.run_solver("04_MIP/main.py", BUNDLES[name])
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            first[name] = lines[0]["OVERLOAD"]
            self.assert_follows_allocation(out)
        self.assertEqual(first, {"static": 9, "moved": 7})

    def test_capacity_analysis(self):
        peaks = {}
        for name in ("static", "t0", "moved"):
            proc, _, _ = self.run_solver("10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py", BUNDLES[name])
            if proc.returncode != 0 and "ModuleNotFoundError" in proc.stderr:
                self.skipTest("analysis dependencies not available")
            peaks[name] = proc.stdout.strip()
        self.assertEqual(peaks, {"static": "3", "t0": "3", "moved": "2"})

    def test_a_malformed_schedule_stops_the_run_with_the_reason(self):
        bad = Path(TMP.name) / "bad"
        bad.mkdir(exist_ok=True)
        for fname, text in FILES.items():
            (bad / fname).write_text(text)
        (bad / nsa.SCHEDULE_FILENAME).write_text(T0_ROWS.replace("\n1,0,0\n", "\n1,4,0\n"))
        proc, lines, _ = self.run_solver("10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py", bad)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ScheduleError", proc.stderr)
        self.assertIn("at From_Time 0, but navaid_sector_assignment.csv puts it into sector 0", proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
