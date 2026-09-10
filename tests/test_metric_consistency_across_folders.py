"""Task 2: every folder must score arrival delay through the same implementation.

A cross-solver agreement check is meant to catch a folder someone forgot to wire up.
Comparing the delay *numbers* three solvers report does not do that: on a common instance the
heuristic, the ASP encoding and the MIP reach different feasible schedules (the MIP resolves
overload by rerouting where the heuristic delays), so their totals differ for reasons that have
nothing to do with the metric -- and they differed the same way before this change.

What does catch a forgotten folder is checking that each folder's delay path resolves to the one
shared implementation, and that no folder still carries a hard-coded max(0, delta).

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common import arrival_delay as canonical  # noqa: E402

SHIMS = [
    "01_ASPaeroFlow/src/aspaeroflow/arrival_delay_bootstrap.py",
    "02_ASP/arrival_delay_bootstrap.py",
    "04_MIP/arrival_delay_bootstrap.py",
    "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/arrival_delay_bootstrap.py",
]

# Every file in the repository that computes or reports an arrival delay.
DELAY_SITES = [
    "01_ASPaeroFlow/src/aspaeroflow/main_loop_components/after_optimization.py",
    "01_ASPaeroFlow/src/aspaeroflow/main_loop_components/evaluate_solution.py",
    "01_ASPaeroFlow/src/aspaeroflow/optimize_flights.py",
    "04_MIP/main.py",
    "04_MIP/mip_model.py",
    "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py",
    "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/optimize_flights.py",
]

ENCODINGS = ["01_ASPaeroFlow/encoding.lp", "02_ASP/encoding.lp"]


def load_shim(rel):
    import importlib.util
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(f"shim_{rel.replace('/', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAllFoldersShareOneImplementation(unittest.TestCase):

    def test_every_shim_resolves_to_the_canonical_module(self):
        for rel in SHIMS:
            with self.subTest(shim=rel):
                mod = load_shim(rel)
                self.assertIs(mod.apply, canonical.apply)
                self.assertIs(mod.delay_matrix, canonical.delay_matrix)
                self.assertIs(mod.normalise, canonical.normalise)
                self.assertEqual(mod.DEFAULT_ARRIVAL_DELAY_METRIC,
                                 canonical.DEFAULT_ARRIVAL_DELAY_METRIC)

    def test_every_shim_scores_the_fixture_identically(self):
        """delta = -5 and +3 -> signed -2, floored 3, absolute 8, in every folder."""
        delta = np.array([-5, 3])
        expected = {"signed": -2, "floored": 3, "absolute": 8}
        for rel in SHIMS:
            mod = load_shim(rel)
            for metric, want in expected.items():
                with self.subTest(shim=rel, metric=metric):
                    self.assertEqual(int(mod.apply(delta, metric).sum()), want)


class TestNoFolderStillHardCodesTheMetric(unittest.TestCase):

    FLOORED_PATTERN = re.compile(r"np\.maximum\(\s*0\s*,\s*t_final\s*-\s*t_init\s*\)")

    def test_no_hard_coded_floored_reporting(self):
        for rel in DELAY_SITES:
            with self.subTest(file=rel):
                text = (REPO / rel).read_text()
                self.assertIsNone(
                    self.FLOORED_PATTERN.search(text),
                    f"{rel} still floors the delay directly instead of using the metric",
                )

    def test_no_encoding_hard_codes_the_guard(self):
        """A bare `Y > T` guard on an unconditional arrival_delay rule is the floored metric."""
        for rel in ENCODINGS:
            with self.subTest(encoding=rel):
                for line in (REPO / rel).read_text().splitlines():
                    line = line.strip()
                    if line.startswith("arrival_delay(") and "arrival_delay_scoring" not in line:
                        self.fail(f"{rel}: unguarded arrival_delay rule: {line}")

    def test_both_encodings_cover_all_three_metrics(self):
        for rel in ENCODINGS:
            text = (REPO / rel).read_text()
            for metric in canonical.ARRIVAL_DELAY_METRICS:
                with self.subTest(encoding=rel, metric=metric):
                    self.assertIn(f"arrival_delay_scoring({metric})", text,
                                  f"{rel} has no rule for the {metric} metric")


if __name__ == "__main__":
    unittest.main(verbosity=2)
