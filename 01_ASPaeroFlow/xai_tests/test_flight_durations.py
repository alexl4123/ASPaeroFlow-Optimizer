"""The flight duration the candidate choice sorts by: the number of time steps a flight occupies, at setup and
after every step (kept or rejected), equal to the span of its navpoint times; the run records the rule.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.auxiliaries.computation_helpers import (  # noqa: E402
    FLIGHT_DURATION_RULE, flight_spans_contiguous)
from src.aspaeroflow.main_loop_components import evaluate_solution  # noqa: E402
from src.aspaeroflow.xai.session import OptimizerSession  # noqa: E402

TEN = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"


class Helper(unittest.TestCase):

    def triple(self, row):
        start, stop, duration = flight_spans_contiguous(np.array([row]), fill_value=-1)
        return int(start[0]), int(stop[0]), int(duration[0])

    def test_one_block(self):
        self.assertEqual(self.triple([-1, -1, 5, 5, 7, -1]), (2, 5, 3))

    def test_empty_row(self):
        start, stop, duration = flight_spans_contiguous(np.array([[-1, -1, -1]]), fill_value=-1)
        self.assertEqual((start.tolist(), stop.tolist(), duration.tolist()), ([-1], [-1], [0]))

    def test_two_blocks_give_the_first(self):
        self.assertEqual(self.triple([5, -1, 6, 6, -1]), (0, 1, 1))


def navpoint_spans(navpoint_matrix):
    """Last minus first navpoint time + 1 per flight (0 for a flight without navpoints)."""
    spans = []
    for row in navpoint_matrix:
        times = np.flatnonzero(row != -1)
        spans.append(int(times[-1] - times[0] + 1) if times.size else 0)
    return np.array(spans)


class DurationsAfterEveryStep(unittest.TestCase):
    """10-flight fixture, three option sets; after setup and after every step, for every flight:
    recorded duration == occupied time steps == navpoint span."""

    def check_run(self, options):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trace"
            session = OptimizerSession(TEN, folder, options)
            session.begin()
            self.check_dto(session.app.optimization_dto, "setup")
            seen = []
            for _ in range(60):
                summary = session.step()
                # sequential mode ends with "SEQUENTIAL END", after which step() repeats the last record
                if summary is None or summary["iteration"] in seen:
                    break
                seen.append(summary["iteration"])
                record = session.records[-1]
                dto = session.app.optimization_dto
                self.check_dto(dto, f"iteration {record['iteration']}")
                if record["accepted"]:
                    # the step's solver flights and later legs (every flight with a recorded trajectory)
                    flights = {int(f) for sub in record["subproblems"] for f in sub["current_trajectories"]}
                    self.assertTrue(flights)
                    occupied = (dto["converted_instance_matrix"] != -1).sum(axis=1)
                    for f in flights:
                        self.assertEqual(int(dto["flight_durations"][f]), int(occupied[f]))
                if session.status == "finished":
                    break
            self.assertGreater(len(session.records), 0)
            session.app._xai_trace.close()
            run = json.loads((folder / "run.json").read_text())
            self.assertEqual(run["flight_duration_rule"], FLIGHT_DURATION_RULE)
            self.assertEqual(FLIGHT_DURATION_RULE, "occupied_steps")
            return session.records, run

    def check_dto(self, dto, where):
        durations = np.asarray(dto["flight_durations"])
        occupied = (dto["converted_instance_matrix"] != -1).sum(axis=1)
        spans = navpoint_spans(dto["converted_navpoint_matrix"])
        bad = [(int(f), int(durations[f]), int(occupied[f]), int(spans[f]))
               for f in range(len(durations)) if not durations[f] == occupied[f] == spans[f]]
        self.assertEqual(bad, [], f"{where}: (flight, recorded, occupied, navpoint span)")

    def test_default_options(self):
        records, _ = self.check_run({})
        self.assertTrue(any(r["accepted"] for r in records))

    def test_sequential_execution(self):
        """On this instance every sequential step is rejected, so the rejected path is checked."""
        records, run = self.check_run({"sequential_execution": "true"})
        self.assertIs(run["sequential_execution"], True)
        self.assertTrue(any(not r["accepted"] for r in records))

    def test_minimize_number_sectors(self):
        """Sector merging after the answer (evaluate_solution.py) rewrites rows of flights outside the answer."""
        calls = []
        original = evaluate_solution.minimize_number_of_sectors_new

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(evaluate_solution, "minimize_number_of_sectors_new", counting):
            records, run = self.check_run({"minimize_number_sectors_enabled": "true"})
        self.assertIs(run["minimize_number_sectors"], True)
        self.assertGreater(len(calls), 0)
        self.assertTrue(any(r["accepted"] for r in records))


if __name__ == "__main__":
    unittest.main()
