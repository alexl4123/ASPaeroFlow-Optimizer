"""Every position lies inside the time domain, and full sectorization contains the restricted one.

Two defects of 02_ASP/encoding.lp found on 2026-10-05 (proven optima of the usc27 run, V2 small
family, compared across variants):

  time guard  The planned route and the precomputed paths were shifted by the ground delay without
              a bound, so a delayed flight could fly past the last timestep, where there is no
              sector and no capacity: its load went uncounted and the claimed overload was too
              low (here, rp_dp_ns claimed 0 and had 1 at t=25). Full rerouting already kept to
              time(T), so it looked worse than restricted rerouting.
  sector domain  Full sectorization could only use the sectors open at t=0, so it could not form
              the partitions of the restricted options. Option 3 (initial full sectorization) keeps
              that behaviour for comparison.

Fixtures, embedded below: EAST-ASIA-3x3, 10 flights, seeds 11904657 and 13 (ASPaeroFlow-DataGenerator,
experiment_data_V2_small_scaling/30-0-EAST-ASIA-3x3-V2), timestep granularity 1, max_time 24.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import importlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "02_ASP")):
    if path not in sys.path:
        sys.path.insert(0, path)

SHARED = {
    "graph_edges.csv": ("source,target,dist_m", "0,1,1351188.334 0,2,993085.935 0,3,1635045.387 1,4,1351188.334 1,2,1635045.387 1,3,993085.935 1,5,1635045.387 4,3,1635045.387 4,5,993085.935 2,3,1243589.281 2,6,993085.935 2,7,1538525.388 3,5,1243589.281 3,6,1538525.388 3,7,993085.935 3,8,1538525.388 5,7,1538525.388 5,8,993085.935 6,7,1105884.217 7,8,1105884.217 9,1,634161.212 10,1,562044.302 11,5,474658.817 12,3,395248.691 13,5,354800.083 14,7,424654.855 15,0,143053.731 16,0,105938.504"),
    "airports.csv": ("Airport_Vertex", "9 10 11 12 13 14 15 16"),
    "sectors.csv": ("Sector_ID,Capacity", "0,1 1,1 2,1 3,1 4,1 5,1 6,1 7,1 8,1 9,60000 10,60000 11,60000 12,60000 13,60000 14,60000 15,60000 16,60000"),
    "navaid_sector_assignment.csv": ("Navaid_ID,Sector_ID", "0,0 1,0 2,0 3,4 4,4 5,4 6,6 7,6 8,6 9,9 10,10 11,11 12,12 13,13 14,14 15,15 16,16"),
}
FLIGHTS_SEED11904657 = {
    "flights.csv": ("Flight_ID,Position,Time", "0,15,6 0,0,7 0,2,9 0,7,11 0,14,12 1,13,10 1,5,11 1,3,13 1,12,14 2,11,12 2,5,13 2,1,15 2,10,16 3,11,14 3,5,15 3,1,17 3,10,18 4,15,16 4,0,17 4,1,19 4,10,20 5,16,17 5,0,18 5,1,20 5,5,22 5,11,23 6,12,17 6,3,18 6,7,20 6,14,21 7,11,18 7,5,19 7,1,21 7,0,23 7,15,24 8,14,20 8,7,21 8,5,23 8,11,24 9,15,21 9,0,22 9,16,23"),
    "airplanes.csv": ("Airplane_ID,Speed_kts", "0,480.0 1,430.0 2,450.0 3,450.0 4,450.0 5,480.0 6,480.0 7,480.0"),
    "airplane_flight_assignment.csv": ("Airplane_ID,Flight_ID", "0,0 0,8 1,1 1,6 2,2 3,3 4,4 5,5 6,7 7,9"),
}
FLIGHTS_SEED13 = {
    "flights.csv": ("Flight_ID,Position,Time", "0,15,2 0,0,3 0,1,5 0,10,6 1,13,7 1,5,8 1,3,10 1,12,11 2,15,7 2,0,8 2,3,10 2,12,11 3,11,10 3,5,11 3,1,13 3,10,14 4,15,10 4,0,11 4,1,13 4,10,14 5,13,11 5,5,12 5,3,14 5,12,15 6,14,11 6,7,12 6,5,14 6,11,15 7,13,15 7,5,16 7,11,17 8,15,15 8,0,16 8,1,18 8,10,19 9,12,17 9,3,18 9,7,20 9,14,21"),
    "airplanes.csv": ("Airplane_ID,Speed_kts", "0,450.0 1,450.0 2,480.0 3,480.0 4,430.0 5,450.0 6,450.0 7,430.0 8,430.0"),
    "airplane_flight_assignment.csv": ("Airplane_ID,Flight_ID", "0,0 1,1 1,9 2,2 3,3 4,4 5,5 6,6 7,7 8,8"),
}
MAX_TIME, TG = 24, 1
GUARD = ":- navpoint_flight(_,_,T), not time(T)."

TMP = None
BUNDLES = {}


def setUpModule():
    global TMP
    TMP = tempfile.TemporaryDirectory()
    for name, own in (("seed11904657", FLIGHTS_SEED11904657), ("seed13", FLIGHTS_SEED13)):
        d = Path(TMP.name) / name
        d.mkdir()
        for fname, (head, body) in {**SHARED, **own}.items():
            (d / fname).write_text(head + "\n" + "\n".join(body.split()) + "\n")
        BUNDLES[name] = d


def tearDownModule():
    TMP.cleanup()


class TestTimeGuardAndSectorDomain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import clingo  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("clingo not available")
        cls.translate = importlib.import_module("translate").TranslateCSVtoLogicProgram
        cls.encoding = (REPO / "02_ASP/encoding.lp").read_text() + "arrival_delay_metric(signed).\n"

    def facts(self, name, gd, rr, ds):
        d = BUNDLES[name]
        return self.translate().main(d / "graph_edges.csv", d / "flights.csv", d / "sectors.csv",
                                     d / "airports.csv", d / "airplanes.csv",
                                     d / "airplane_flight_assignment.csv", d / "navaid_sector_assignment.csv",
                                     REPO / "02_ASP/encoding.lp", TG, MAX_TIME, 6, gd, rr, ds)

    def optimum(self, program, extra=""):
        """(cost vector, shown atoms as strings) of the proven optimum, or (None, None) if unsatisfiable."""
        import clingo
        ctl = clingo.Control(["--opt-strategy=usc,oll", "--opt-usc-shrink=min", "--warn=none"])
        ctl.add("base", [], program + "\n#show navpoint_flight/3.\n#show navpoint_sector/3.\n" + extra)
        ctl.ground([("base", [])])
        found = {}

        def on_model(model):
            found["cost"] = list(model.cost)
            found["atoms"] = [str(s) for s in model.symbols(shown=True)]
        result = ctl.solve(on_model=on_model)
        if not result.satisfiable:
            return None, None
        self.assertTrue(result.exhausted, "the optimum was not proven")
        return found["cost"], found["atoms"]

    @staticmethod
    def times(atoms, predicate="navpoint_flight"):
        return [int(re.match(rf"{predicate}\(\d+,\d+,(\d+)\)", a).group(1)) for a in atoms if a.startswith(predicate + "(")]

    def test_the_guard_is_in_the_encoding(self):
        self.assertIn(GUARD, self.encoding)

    def test_without_the_guard_a_delayed_flight_flies_past_the_time_domain(self):
        # The defect itself: rp_dp_ns places positions after t=24 when the guard is taken out.
        program = self.encoding.replace(GUARD, "") + "\n".join(self.facts("seed11904657", 1, 1, 0))
        _, atoms = self.optimum(program)
        self.assertGreater(max(self.times(atoms)), MAX_TIME)

    def test_every_position_lies_inside_the_time_domain(self):
        for rr in (0, 1, 2):
            with self.subTest(rr=rr):
                _, atoms = self.optimum(self.encoding + "\n".join(self.facts("seed11904657", 1, rr, 0)))
                self.assertLessEqual(max(self.times(atoms)), MAX_TIME)

    def test_more_rerouting_freedom_is_never_worse(self):
        costs = [self.optimum(self.encoding + "\n".join(self.facts("seed11904657", 1, rr, 0)))[0]
                 for rr in (0, 1, 2)]                       # nr, rp, r with bounded delay
        self.assertLessEqual(costs[1], costs[0])
        self.assertLessEqual(costs[2], costs[1])

    def test_option_3_emits_the_initial_domain_and_other_values_are_refused(self):
        self.assertIn("regulation_initial_sector_domain.", self.facts("seed13", 0, 0, 3))
        self.assertNotIn("regulation_initial_sector_domain.", self.facts("seed13", 0, 0, 2))
        with self.assertRaises(ValueError):
            self.facts("seed13", 0, 0, 4)

    def test_full_sectorization_contains_the_restricted_one(self):
        restricted, atoms = self.optimum(self.encoding + "\n".join(self.facts("seed13", 2, 0, 1)))
        forced = "\n".join(f":- not {a}." for a in atoms if a.startswith("navpoint_sector("))
        full_program = self.encoding + "\n".join(self.facts("seed13", 2, 0, 2))
        # the restricted optimum is a solution of full sectorization ...
        cost, _ = self.optimum(full_program, forced)
        self.assertIsNotNone(cost)
        # ... so the full optimum is at least as good
        full, full_atoms = self.optimum(full_program)
        self.assertLessEqual(full, restricted)
        # and airports stay singleton sectors
        airports = {int(a) for a in SHARED["airports.csv"][1].split()}
        for a in full_atoms:
            if a.startswith("navpoint_sector("):
                nav, sec, _ = map(int, a[len("navpoint_sector("):-1].split(","))
                self.assertEqual(nav in airports, sec in airports, a)
                if nav in airports:
                    self.assertEqual(nav, sec, a)

    def test_initial_full_sectorization_cannot_form_the_restricted_partition(self):
        _, atoms = self.optimum(self.encoding + "\n".join(self.facts("seed13", 2, 0, 1)))
        forced = "\n".join(f":- not {a}." for a in atoms if a.startswith("navpoint_sector("))
        cost, _ = self.optimum(self.encoding + "\n".join(self.facts("seed13", 2, 0, 3)), forced)
        self.assertIsNone(cost)


if __name__ == "__main__":
    unittest.main()
