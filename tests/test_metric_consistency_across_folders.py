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


#: Every file that reports SECTOR-NUMBER or RECONFIG on a result line.
SECTOR_METRIC_SITES = [
    "01_ASPaeroFlow/src/aspaeroflow/main_loop_components/setup_before_optimization.py",
    "01_ASPaeroFlow/src/aspaeroflow/main_loop_components/evaluate_solution.py",
    "01_ASPaeroFlow/src/aspaeroflow/main_loop_components/after_optimization.py",
    "04_MIP/main.py",
    "04_MIP/mip_model.py",
    "10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS/main.py",
]


class TestEveryFolderScoresOnTheInstanceWindow(unittest.TestCase):
    """SECTOR-NUMBER and RECONFIG are sums over time, so they need one agreed axis.

    Summed over each solver's own matrix width they are not comparable at all: on one 10-flight
    instance with an unchanged sectorisation the four systems reported 1364, 484, 367 and 275 for
    one physical fact, purely because their axes were 124, 44, 31 and 25 columns wide.
    """

    #: `compute_total_number_sectors(<the matrix itself>)` -- the un-normalised call. A
    #: normalised one passes `scored_navaid_sector_time_assignment`, which this does not match.
    RAW_SECTOR_NUMBER = re.compile(
        r"compute_total_number_sectors\(\s*(?:self\.)?(?:old_|original_)?"
        r"navaid_sector_time_assignment\s*\)")

    #: The hand-rolled widening the shared to_window() replaces; it also raised on a FINAL
    #: matrix narrower than the initial one, because np.repeat rejects a negative count.
    HAND_ROLLED_WIDENING = re.compile(r"np\.repeat\(\s*(?:self\.)?\w*original_navaid_sector_time"
                                      r"_assignment\[:,\s*\[-1\]\]")

    def test_no_folder_sums_sector_number_on_its_own_axis(self):
        for rel in SECTOR_METRIC_SITES:
            with self.subTest(file=rel):
                text = (REPO / rel).read_text()
                self.assertIsNone(
                    self.RAW_SECTOR_NUMBER.search(text),
                    f"{rel} sums SECTOR-NUMBER over its own matrix width; it must go through "
                    "to_window(..., the instance's evaluation window) first",
                )

    def test_every_site_reaches_the_shared_normalisation(self):
        for rel in SECTOR_METRIC_SITES:
            with self.subTest(file=rel):
                text = (REPO / rel).read_text()
                self.assertIn("to_evaluation_window", text,
                              f"{rel} reports the sector metrics without normalising the axis")
                self.assertIsNone(self.HAND_ROLLED_WIDENING.search(text),
                                  f"{rel} still widens the reference allocation by hand")

    def test_the_asp_scores_the_same_window(self):
        text = (REPO / "02_ASP" / "solver.py").read_text()
        self.assertIn("series_to_window", text,
                      "02_ASP sums its per-timestep metrics over its own ASP time domain, which "
                      "is max_time*T_gran + 1 and not the instance's window")


class TestTheAspCollectsAtomsByExactName(unittest.TestCase):
    """`symbol.name in NAVPOINT_FLIGHT` is a SUBSTRING test, and "flight" is a substring."""

    SUBSTRING_TEST = re.compile(
        r"symbol\.name\s+in\s+(FLIGHT|NAVPOINT_FLIGHT|NAVPOINT_SECTOR|REROUTE|RECONFIG|"
        r"OVERLOAD|ARRIVAL_DELAY|SECTOR_NUMBER|SECTOR_DIFF)\b")

    @staticmethod
    def _code_only(text):
        """Comments are dropped: this file NAMES the old substring tests to explain them."""
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    def test_no_name_is_matched_as_a_substring(self):
        text = self._code_only((REPO / "02_ASP" / "solver.py").read_text())
        found = self.SUBSTRING_TEST.findall(text)
        self.assertEqual(found, [],
                         f"02_ASP/solver.py tests atom names as substrings: {found}. "
                         "Use == against the constant, or `in SIGNATURES`.")

    def test_the_allocation_constant_names_an_atom_the_encoding_derives(self):
        solver = (REPO / "02_ASP" / "solver.py").read_text()
        encoding = (REPO / "02_ASP" / "encoding.lp").read_text()
        match = re.search(r'NAVPOINT_SECTOR:\s*Final\[str\]\s*=\s*"([^"]+)"', solver)
        self.assertIsNotNone(match, "02_ASP/solver.py no longer names the allocation atom")
        name = match.group(1)
        self.assertIn(f"{name}(", encoding,
                      f"02_ASP/solver.py looks for {name}/3, which encoding.lp never derives")


if __name__ == "__main__":
    unittest.main(verbosity=2)
