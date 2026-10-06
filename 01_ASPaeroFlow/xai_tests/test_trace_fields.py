"""The trace fields that explain a step: the flights in the hotspot cell, how many went to the solver, and the
demand and capacity of every sector part (10-flight run shared with test_xai_trace_and_explanations.py).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai.contrastive import IterationExplainer  # noqa: E402
from src.aspaeroflow.xai.subproblem import Subproblem  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402


class TraceFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, _, trace_dir = shared_run.get()
        cls.trace = TraceReader(trace_dir)
        cls.records = list(cls.trace)

    def test_records_exist(self):
        self.assertGreater(len(self.records), 0)

    def test_hotspot_flights_and_taken(self):
        for r in self.records:
            n, h, p = r["iteration"], r["hotspot"], r["parameters"]
            with self.subTest(iteration=n):
                flights = h["flights"]
                self.assertEqual(flights, sorted(flights, key=lambda x: (x["duration"], x["id"])))
                self.assertEqual(len(flights), h["demand"])
                self.assertEqual(len({f["id"] for f in flights}), len(flights))
                self.assertEqual(h["taken"], min(p["max_aircraft"], max(2, h["overload"]), h["demand"]))
                sub = Subproblem.from_instance(self.trace.instance(n))
                self.assertEqual(sorted(f["id"] for f in flights[:h["taken"]]), sorted(sub.decision_flights))

    def test_part_demand_and_capacity(self):
        for r in self.records:
            h = r["hotspot"]
            self.assertEqual(h["overload"], h["demand"] - h["capacity"])
            sc = r["sector_changes"]
            for config in ("prev_sector_config", "post_sector_config"):
                for s, part in (sc.get(config) or {}).items():
                    with self.subTest(iteration=r["iteration"], config=config, sector=s):
                        self.assertEqual(part["overload"], part["demand"] - part["capacity"])
            if r["accepted"]:
                prev = sc["prev_sector_config"][str(h["sector"])]
                self.assertEqual((prev["demand"], prev["capacity"]), (h["demand"], h["capacity"]))

    def test_why_hotspot_words(self):
        for n in self.trace.accepted():
            answer = IterationExplainer(self.trace, n).why_hotspot()["answer"]
            with self.subTest(iteration=n):
                self.assertIn("time ", answer)
                for bad in ("time step", "widened", "with the shortest flight time"):
                    self.assertNotIn(bad, answer)


if __name__ == "__main__":
    unittest.main()
