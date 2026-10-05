"""Fixed-plan exposure measures (xai/robustness.py) on the 10-flight fixture.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai.robustness import Plan, analyse, attribution, coordinates_of, overload_cells  # noqa: E402

INSTANCE = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"


class Robustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        subprocess.run([sys.executable, str(OPT / "main.py"), f"--data-dir={INSTANCE}",
                        f"--encoding-path={OPT / 'encoding.lp'}", "--save-results=true",
                        f"--results-root={cls.tmp.name}"], cwd=OPT.parent, capture_output=True, check=True, timeout=600)
        cls.results = Path(cls.tmp.name) / INSTANCE.name
        cls.plan = Plan.load(cls.results, INSTANCE)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_final_plan_has_no_overload(self):
        self.assertEqual(int(overload_cells(self.plan.demand(), self.plan.capacity()).sum()), 0)

    def test_capacity_matches_the_optimizer_on_used_cells(self):
        c = self.plan.capacity()
        saved = np.loadtxt(self.results / "capacity_time_matrix.csv", delimiter=",", dtype=np.int64, ndmin=2)[:, :c.shape[1]]
        for t in range(c.shape[1]):
            used = np.unique(self.plan.allocation[:, t])
            np.testing.assert_array_equal(c[used, t], saved[used, t])

    def test_attribution_adds_up_to_the_overload(self):
        q = self.plan.demand()
        degraded = np.zeros_like(self.plan.atomic)            # every navpoint at capacity 0
        c = self.plan.capacity(np.repeat(degraded[:, None], q.shape[1], axis=1))
        over = overload_cells(q, c)
        self.assertGreater(over.sum(), 0)
        self.assertAlmostEqual(attribution(self.plan, over, q).sum(), float(over.sum()))

    def test_scenarios_run(self):
        result = analyse(self.plan, scenarios=20, model="storms", seed=3, coords=coordinates_of(INSTANCE),
                         cells=2, radius=1.5, loss=1.0, life=4)
        self.assertEqual(result["nominal"]["overload"], 0)
        self.assertEqual(len(result["overload"]["histogram"]) - 1, int(result["overload"]["max"]))


if __name__ == "__main__":
    unittest.main()
