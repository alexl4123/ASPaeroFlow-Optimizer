"""Solvers charge an edge the generator's number of timesteps, computed from the float dist_m.

The data generator times each filed hop from the float ``dist_m`` in ``graph_edges.csv``. The
solvers used to read ``dist_m`` with ``int(ceil(float(x)))`` and so charged one timestep more
wherever a timestep boundary lies in ``[dist_m, ceil(dist_m))``; the filed plan was then infeasible.
The 22 edge-speed pairs below are every such pair in the V2 data (e.g. DACH TG60, edge 347-1202,
13272.548 m at 430 kts: generator 1 timestep, old solver cost 2).

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import importlib
import importlib.util
import math
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.edge_cost import edge_duration_timesteps, load_graph_edges  # noqa: E402

# (timestep granularity, dist_m as written in graph_edges.csv, Speed_kts) for every V2 edge-speed
# pair a filed flight uses whose generator cost differs from the ceiled-distance cost.
BOUNDARY_PAIRS = [
    (15, "55559.368", 450),                                   # EUROPE TG15 3440-6184
    (60, "13272.548", 430),                                   # DACH TG60 347-1202
    (60, "13272.548", 430), (60, "55559.368", 450),           # EUROPE TG60
    (60, "27779.633", 450), (60, "14815.476", 480),
    (60, "14815.661", 480), (60, "13889.479", 450),           # USA-MAINLAND TG60
    (60, "13889.839", 450), (60, "13272.005", 430),
    (60, "13272.293", 430), (60, "14815.568", 480),
    (60, "13889.208", 450), (60, "13889.276", 450),
    (60, "13272.156", 430), (60, "13272.541", 430),
    (60, "14815.65", 480), (60, "13889.806", 450),
    (60, "13889.327", 450), (60, "14815.577", 480),
    (60, "13889.635", 450), (60, "13889.346", 450),
]


def generator_edge_duration_slots(distance_m, speed_kts, time_granularity):
    """ASPaeroFlow-DataGenerator, 04_simplified_filed_flight_plan_generator.py, _edge_duration_slots."""
    speed_ms = float(speed_kts) * 0.51444
    if speed_ms <= 0:
        return 1  # defensive
    duration_seconds = float(distance_m) / speed_ms
    slot_sec = 3600.0 / float(time_granularity)
    slots = int(math.ceil(duration_seconds / slot_sec))
    return max(slots, 1)


def ceiled_distance_cost(distance_m, speed_kts, time_granularity):
    """What the solvers computed before: dist_m rounded up to whole metres first."""
    return generator_edge_duration_slots(int(math.ceil(float(distance_m))), speed_kts, time_granularity)


class TestEdgeDuration(unittest.TestCase):

    def test_boundary_pairs_match_the_generator(self):
        self.assertEqual(len(BOUNDARY_PAIRS), 22)
        for tg, dist, speed in BOUNDARY_PAIRS:
            with self.subTest(tg=tg, dist=dist, speed=speed):
                expected = generator_edge_duration_slots(float(dist), float(speed), tg)
                self.assertEqual(edge_duration_timesteps(float(dist), speed, tg), expected)
                # The pair really is a boundary pair: the old rounding charged one timestep more.
                self.assertEqual(ceiled_distance_cost(dist, speed, tg), expected + 1)

    def test_reference_edge_costs_one_timestep(self):
        self.assertEqual(edge_duration_timesteps(13272.548, 430, 60), 1)
        self.assertEqual(edge_duration_timesteps(np.float64(13272.548), np.int64(430), 60), 1)

    def test_matches_the_generator_on_random_edges(self):
        rng = random.Random(7)
        for _ in range(20000):
            dist = round(rng.uniform(0.0, 2_000_000.0), rng.choice((0, 1, 3)))
            speed = rng.choice((430, 450, 480, 250, 1))
            tg = rng.choice((1, 4, 15, 60))
            self.assertEqual(edge_duration_timesteps(dist, speed, tg),
                             generator_edge_duration_slots(dist, float(speed), tg), (dist, speed, tg))


GRAPH_CSV = "source,target,dist_m\n0,1,13272.548\n1,2,55559.368\n2,3,1000\n"


class TestGraphLoaders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        files = {
            "graph_edges.csv": GRAPH_CSV,
            "sectors.csv": "Sector_ID,Capacity\n0,60000\n1,1200\n2,1200\n3,60000\n",
            "flights.csv": "Flight_ID,Position,Time\n0,0,2\n0,1,3\n1,3,5\n1,2,6\n",
            "airports.csv": "Airport_Vertex\n0\n3\n",
            "airplanes.csv": "Airplane_ID,Speed_kts\n0,430.0\n1,450.0\n",
            "airplane_flight_assignment.csv": "Airplane_ID,Flight_ID\n0,0\n1,1\n",
            "navaid_sector_assignment.csv": "Navaid_ID,Sector_ID\n0,0\n1,1\n2,2\n3,3\n",
        }
        for name, text in files.items():
            (cls.dir / name).write_text(text)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, graph, dist_m):
        self.assertEqual(graph.shape, (3, 2))
        self.assertTrue(np.issubdtype(graph.dtype, np.integer))
        self.assertEqual(graph.tolist(), [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(dist_m.tolist(), [13272.548, 55559.368, 1000.0])

    def test_shared_loader_keeps_the_fractional_part(self):
        self.check(*load_graph_edges(self.dir / "graph_edges.csv"))

    def test_every_solver_loader_keeps_the_fractional_part(self):
        paths = types.SimpleNamespace(
            _graph_path=self.dir / "graph_edges.csv", _sectors_path=self.dir / "sectors.csv",
            _flights_path=self.dir / "flights.csv", _airports_path=self.dir / "airports.csv",
            _airplanes_path=self.dir / "airplanes.csv",
            _airplane_flight_path=self.dir / "airplane_flight_assignment.csv",
            _navaid_sector_path=self.dir / "navaid_sector_assignment.csv", _encoding_path=None)
        loaders = {
            "04_MIP": ("04_MIP/main.py", "Main"),
            "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS":
                ("10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py", "Main"),
        }
        for folder, (file, cls_name) in loaders.items():
            with self.subTest(folder=folder):
                sys.path.insert(0, str(REPO / folder))
                spec = importlib.util.spec_from_file_location(f"edge_cost_test_{folder}", REPO / file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ns = types.SimpleNamespace(**vars(paths))
                getattr(module, cls_name).load_data(ns)
                self.check(ns.graph, ns.graph_dist_m)
        with self.subTest(folder="01_ASPaeroFlow"):
            sys.path.insert(0, str(REPO / "01_ASPaeroFlow"))
            setup = importlib.import_module("src.aspaeroflow.main_loop_components.setup_before_optimization")
            ns = types.SimpleNamespace(**vars(paths))
            setup.SetupBeforeOptimization.load_data(ns)
            self.check(ns.graph, ns.graph_dist_m)
        with self.subTest(folder="02_ASP"):
            sys.path.insert(0, str(REPO / "02_ASP"))
            tr = importlib.import_module("translate").TranslateCSVtoLogicProgram()
            d = self.dir
            tr.load_data(d / "graph_edges.csv", d / "sectors.csv", d / "flights.csv", d / "airports.csv",
                         d / "airplanes.csv", d / "airplane_flight_assignment.csv",
                         d / "navaid_sector_assignment.csv", REPO / "02_ASP/encoding.lp")
            self.check(tr.graph, tr.graph_dist_m)


class TestAspFiledPlanOnBoundaryEdge(unittest.TestCase):
    """The filed plan over the reference edge is a solution of the exact ASP program."""

    def test_filed_plan_is_satisfiable_with_regulations_off(self):
        try:
            import clingo
        except ImportError:
            raise unittest.SkipTest("clingo not available")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            files = {
                "graph_edges.csv": "source,target,dist_m\n0,1,13272.548\n1,2,13272.548\n",
                "sectors.csv": "Sector_ID,Capacity\n0,60000\n1,1200\n2,60000\n",
                # 13272.548 m at 430 kts is 1 timestep at TG=60, so every hop takes exactly one.
                "flights.csv": "Flight_ID,Position,Time\n0,0,2\n0,1,3\n0,2,4\n1,2,10\n1,1,11\n1,0,12\n",
                "airports.csv": "Airport_Vertex\n0\n2\n",
                "airplanes.csv": "Airplane_ID,Speed_kts\n0,430.0\n1,430.0\n",
                "airplane_flight_assignment.csv": "Airplane_ID,Flight_ID\n0,0\n1,1\n",
                "navaid_sector_assignment.csv": "Navaid_ID,Sector_ID\n0,0\n1,1\n2,2\n",
            }
            for name, text in files.items():
                (d / name).write_text(text)
            sys.path.insert(0, str(REPO / "02_ASP"))
            tr = importlib.import_module("translate").TranslateCSVtoLogicProgram()
            atoms = tr.main(d / "graph_edges.csv", d / "flights.csv", d / "sectors.csv", d / "airports.csv",
                            d / "airplanes.csv", d / "airplane_flight_assignment.csv",
                            d / "navaid_sector_assignment.csv", REPO / "02_ASP/encoding.lp",
                            60, 1, 6, 0, 0, 0)
        self.assertIn("navpoint_edge(0,1,430,1).", atoms)
        self.assertIn("navpoint_edge(1,2,430,1).", atoms)
        program = (REPO / "02_ASP/encoding.lp").read_text() + "arrival_delay_metric(signed).\n" + "\n".join(atoms)
        ctl = clingo.Control(["--opt-mode=ignore", "--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        self.assertTrue(ctl.solve().satisfiable)


if __name__ == "__main__":
    unittest.main()
