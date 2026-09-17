"""Task 2: the arrival-delay metric is selectable and means the same thing everywhere.

Fixture: one flight arrives 5 EARLY (delta = -5) and one arrives 3 LATE
(delta = +3), so the totals must be
    signed -> -2,  floored -> 3,  absolute -> 8.

The same fixture is run through the Python helper and through both ASP encodings, because a
metric that disagrees between the objective and the report is the failure this guards against.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common import arrival_delay as ad  # noqa: E402

# delta = -5 (early) and delta = +3 (late)
EXPECTED = {"signed": -2, "floored": 3, "absolute": 8}


class TestPythonMetric(unittest.TestCase):

    def test_fixture_totals(self):
        delta = np.array([-5, 3])
        for metric, expected in EXPECTED.items():
            with self.subTest(metric=metric):
                self.assertEqual(int(ad.apply(delta, metric).sum()), expected)

    def test_default_is_signed(self):
        """The default is the published definition: delta, with earliness offsetting lateness."""
        self.assertEqual(ad.DEFAULT_ARRIVAL_DELAY_METRIC, "signed")
        self.assertEqual(int(ad.apply(np.array([-5, 3])).sum()), EXPECTED["signed"])

    def test_floored_is_still_reachable(self):
        """The pre-change behaviour must remain available explicitly, not be deleted."""
        self.assertIn("floored", ad.ARRIVAL_DELAY_METRICS)
        self.assertEqual(int(ad.apply(np.array([-5, 3]), "floored").sum()), EXPECTED["floored"])

    def test_delay_matrix_matches_apply(self):
        """delay_matrix is what the reporting path calls; it must agree with apply()."""
        t_init = np.array([20, 20])
        t_final = np.array([15, 23])          # -5 and +3
        for metric, expected in EXPECTED.items():
            with self.subTest(metric=metric):
                self.assertEqual(int(ad.delay_matrix(t_init, t_final, metric).sum()), expected)

    def test_absent_flights_contribute_zero(self):
        """A flight missing from the initial schedule has no planned arrival to measure."""
        t_init = np.array([-1, 20])
        t_final = np.array([99, 23])
        for metric in ad.ARRIVAL_DELAY_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(int(ad.delay_matrix(t_init, t_final, metric).sum()), 3)

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            ad.normalise("late")

    def test_scalar_inputs(self):
        self.assertEqual(ad.apply(-5, "signed"), -5)
        self.assertEqual(ad.apply(-5, "floored"), 0)
        self.assertEqual(ad.apply(-5, "absolute"), 5)


class TestAspEncodings(unittest.TestCase):
    """Both encodings must score the fixture the same way the Python helper does."""

    # The arrival-delay block is copied out of each encoding so the test exercises the real
    # rules; the surrounding encoding needs a whole instance to ground, which this does not.
    @staticmethod
    def _extract_block(encoding_path, body):
        text = Path(encoding_path).read_text()
        keep = [ln for ln in text.splitlines()
                if ln.startswith("arrival_delay") or ln.startswith("#defined arrival_delay")]
        assert keep, f"no arrival-delay rules found in {encoding_path}"
        return "\n".join(keep).replace(body, "planned_arrival_time(ID,T), actual_arrival_time(ID,Y)")

    FIXTURE = """
planned_arrival_time(1,20). actual_arrival_time(1,15).
planned_arrival_time(2,20). actual_arrival_time(2,23).
"""

    ENCODINGS = [
        ("01_ASPaeroFlow/encoding.lp",
         "chosen_path(ID,P), planned_arrival_time(ID,T), actual_arrival_time(ID,Y,P)"),
        ("02_ASP/encoding.lp",
         "planned_arrival_time(ID,T), actual_arrival_time(ID,Y)"),
    ]

    def test_encodings_score_the_fixture(self):
        try:
            import clingo
        except ImportError:
            self.skipTest("clingo not available")

        for rel, body in self.ENCODINGS:
            block = self._extract_block(REPO / rel, body)
            for metric, expected in EXPECTED.items():
                with self.subTest(encoding=rel, metric=metric):
                    ctl = clingo.Control(["--opt-mode=optN"])
                    ctl.add("base", [], block + self.FIXTURE
                            + ":~ arrival_delay(ID,DIFF). [DIFF@9,ID]\n"
                            + ad.asp_metric_fact(metric))
                    ctl.ground([("base", [])])
                    costs = []
                    ctl.solve(on_model=lambda m: costs.append(list(m.cost)))
                    self.assertTrue(costs, f"{rel}/{metric}: no model")
                    self.assertEqual(costs[-1][0], expected,
                                     f"{rel} under {metric}: got {costs[-1][0]}, want {expected}")

    def test_encodings_default_matches_the_python_default(self):
        """An encoding run by hand, with no metric fact, must agree with common/arrival_delay.py.

        This is what stops the .lp fallback and the Python default drifting apart.
        """
        try:
            import clingo
        except ImportError:
            self.skipTest("clingo not available")

        for rel, body in self.ENCODINGS:
            with self.subTest(encoding=rel):
                block = self._extract_block(REPO / rel, body)
                ctl = clingo.Control(["--opt-mode=optN"])
                ctl.add("base", [], block + self.FIXTURE
                        + ":~ arrival_delay(ID,DIFF). [DIFF@9,ID]\n")
                ctl.ground([("base", [])])
                costs = []
                ctl.solve(on_model=lambda m: costs.append(list(m.cost)))
                self.assertEqual(costs[-1][0],
                                 EXPECTED[ad.DEFAULT_ARRIVAL_DELAY_METRIC])


class TestEveryFolderExposesTheOption(unittest.TestCase):
    """The check that catches a folder someone forgot to wire up."""

    ENTRY_POINTS = [
        "01_ASPaeroFlow/main.py",
        "02_ASP/main.py",
        "04_MIP/main.py",
        "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py",
    ]

    def test_cli_option_present_everywhere(self):
        for rel in self.ENTRY_POINTS:
            with self.subTest(entry_point=rel):
                script = REPO / rel
                out = subprocess.run(
                    [sys.executable, script.name, "--help"],
                    cwd=script.parent, capture_output=True, text=True, timeout=120,
                )
                self.assertIn("--arrival-delay-metric", out.stdout,
                              f"{rel} does not expose --arrival-delay-metric")
                for m in ad.ARRIVAL_DELAY_METRICS:
                    self.assertIn(m, out.stdout, f"{rel} help does not mention {m}")


class TestTheBenchmarkCallerNamesTheMetric(unittest.TestCase):
    """A campaign's ARRIVAL-DELAY column must be one definition by construction.

    The caller used to pass no metric at all, so every system fell back to its own default. They
    all default to the same reading today, which made the column comparable BY COINCIDENCE: one
    folder changing its default would have produced a silently mixed column, with nothing in the
    CSVs to say so.
    """

    PATHS = {key: Path(f"/instance/{key}.csv") for key in
             ("graph-edges", "sectors", "flights", "airports", "airplanes",
              "airplane-flight", "navaid-sector")}

    def _caller(self):
        import importlib.util
        folder = REPO / "06_benchmark_start_script"
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location(
            "benchmark_caller", folder / "start_benchmark_caller.py")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:                    # psutil is the runner's own dependency
            self.skipTest(f"benchmark caller not importable: {exc}")
        return module

    def test_every_built_command_names_the_metric(self):
        caller = self._caller()
        system = {"script": REPO / "02_ASP" / "main.py", "encoding": None, "verbosity": None}
        for metric in ad.ARRIVAL_DELAY_METRICS:
            with self.subTest(metric=metric):
                cmd = caller.build_command(system, self.PATHS, "python", 1,
                                           arrival_delay_metric=metric)
                self.assertIn(f"--arrival-delay-metric={metric}", cmd)

    def test_the_default_is_the_repository_default(self):
        caller = self._caller()
        system = {"script": REPO / "04_MIP" / "main.py", "encoding": None, "verbosity": None}
        cmd = caller.build_command(system, self.PATHS, "python", 1)
        self.assertIn(f"--arrival-delay-metric={ad.DEFAULT_ARRIVAL_DELAY_METRIC}", cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
