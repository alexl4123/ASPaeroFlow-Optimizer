"""End to end: run ASPaeroFlow with the XAI trace on a 10-flight V2 instance, then explain it.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

Checks that the trace does not change the run, that the recorded choice of every iteration is
reproduced and optimal for its sub-problem, that no foil beats the recorded choice, and that
contradictory locks come back as a minimal core.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai.contrastive import IterationExplainer, Lock  # noqa: E402
from src.aspaeroflow.xai.subproblem import compare  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402

INSTANCE = shared_run.INSTANCE


class TraceAndExplanations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # the two runs are shared with test_trace_fields.py (xai_tests/shared_run.py removes them at exit)
        cls.plain, cls.traced, cls.trace_dir = shared_run.get()
        cls.trace = TraceReader(cls.trace_dir)

    def test_trace_does_not_change_the_run(self):
        strip = lambda rows: [{k: v for k, v in r.items() if k != "TOTAL-TIME-TO-THIS-POINT"} for r in rows]
        self.assertEqual(strip(self.plain), strip(self.traced))

    def test_one_record_per_iteration(self):
        iterations = [r["ITERATION"] for r in self.traced if not r["COMPUTATION-FINISHED"] and "ITERATION" in r]
        self.assertGreater(len(self.trace.iterations), 0)
        self.assertEqual(sorted(self.trace.iterations), sorted(set(iterations)))

    def test_recorded_choice_is_reproduced_and_optimal(self):
        for n in self.trace.iterations:
            ex = IterationExplainer(self.trace, n)
            recorded = ex.sub_record["clingo_cost"]
            if len(recorded) == 5:
                self.assertEqual(list(ex.factual.vector()), recorded, f"iteration {n}")
            free, _ = ex.solver.solve()
            self.assertIsNone(compare(ex.factual, free), f"iteration {n}: a better answer than the recorded one")

    def test_no_foil_beats_the_recorded_choice(self):
        for n in self.trace.accepted():
            ex = IterationExplainer(self.trace, n)
            answers = [ex.why_flight(f) for f in ex.sub.decision_flights] + [ex.why_sectors()]
            for answer in answers:
                for c in answer["contrasts"]:
                    if c["feasible"] and c["deciding_level"]:
                        level = c["deciding_level"]
                        self.assertGreater(c["costs"][level], answer["factual"]["costs"][level], c["text"])

    def test_changed_row_counts_decision_flights_off_path_0(self):
        """The priority-7 row counts decision flights not on path 0, so its label must not claim all changes."""
        for n in self.trace.accepted():
            ex = IterationExplainer(self.trace, n)
            off = sum(1 for f in ex.sub.decision_flights if ex.chosen_paths.get(f, 0) != 0)
            self.assertEqual(ex.factual.costs["changed"], off, f"iteration {n}")
            for c in ex.why_sectors()["contrasts"]:
                for row in c["ladder"] or []:
                    self.assertNotIn("changed flights", row["text"], f"iteration {n}")

    def test_no_delay_card_says_what_its_foil_allows(self):
        """The no-delay foil allows every path of F that departs no later, the unchanged one included, so its card
        says "not delaying flight F" and never claims a reroute; its answer departs no later than now."""
        seen = 0
        for n in self.trace.accepted():
            ex = IterationExplainer(self.trace, n)
            for f in ex.sub.decision_flights:
                answer = ex.why_flight(f)
                for c in answer["contrasts"]:
                    self.assertNotIn("rerouting", c["label"])
                    if c["label"] == f"not delaying flight {f}" and c["feasible"]:
                        seen += 1
                        foil, _ = ex.solver.solve(ban_paths={(f, p) for p in ex.sub.paths[f]
                                                             if ex.describe_path(f, p).get("departure_shift", 0) > 0})
                        self.assertLessEqual(ex.describe_path(f, foil.chosen_paths[f]).get("departure_shift", 0), 0)
                        self.assertTrue(c["text"].startswith(f"Not delaying flight {f} "), c["text"])
        self.assertGreater(seen, 0)

    def test_contradictory_locks_give_a_minimal_core(self):
        n = self.trace.accepted()[0]
        ex = IterationExplainer(self.trace, n)
        f = ex.sub.decision_flights[0]
        first_path = sorted(ex.sub.paths[f])[0]
        other = [p for p in sorted(ex.sub.paths[f]) if p != first_path][0]
        result = ex.what_if([Lock("path", (f, first_path)), Lock("path", (f, other)), Lock("keep_sectors")])
        self.assertFalse(result["feasible"])
        self.assertEqual(sorted(result["core"]), sorted([f"path {f} {first_path}", f"path {f} {other}"]))


class PathTextWords(unittest.TestCase):
    """Durations in the contrastive texts are time periods ("step N" is the optimizer step)."""

    def text(self, description):
        explainer = IterationExplainer.__new__(IterationExplainer)
        explainer.describe_path = lambda flight, index: description
        return explainer.path_text(18, 0)

    def test_time_periods(self):
        self.assertEqual(self.text({"departure_shift": 1, "arrival_shift": 1, "rerouted": False}),
                         "flight 18 departs 1 time period later, on its current route")
        self.assertEqual(self.text({"departure_shift": 3, "arrival_shift": 5, "rerouted": False}),
                         "flight 18 departs 3 time periods later, on its current route, arrives 5 time periods later")
        self.assertEqual(self.text({"departure_shift": -2, "arrival_shift": -2, "rerouted": False}),
                         "flight 18 departs 2 time periods earlier, on its current route")
        self.assertNotIn("step", self.text({"departure_shift": 0, "arrival_shift": -1, "rerouted": True,
                                            "route": [1, 2], "current_route": [1, 3]}))

    def test_contrast_amounts(self):
        from src.aspaeroflow.xai.subproblem import amount
        self.assertEqual(amount("delay", 3), "3 more time periods of arrival delay")
        self.assertEqual(amount("delay", -1), "1 more time period of arrival delay")

    def test_criterion_words(self):
        """One word per criterion in the cards, the step header and the row lines: the header says "total overload",
        and level 8 counts every open sector of the network at the hotspot time (config_number_sectors)."""
        from src.aspaeroflow.xai.contrastive import SCOPE
        from src.aspaeroflow.xai.subproblem import LEVEL_TEXT
        self.assertEqual(LEVEL_TEXT["overload"], "total overload")
        self.assertEqual(LEVEL_TEXT["sectors"], "open sectors at the hotspot time")
        self.assertNotIn("network overload", SCOPE)
        self.assertIn("total overload", SCOPE)


@unittest.skipUnless(os.environ.get("CE7_TRACE"), "set CE7_TRACE to the full CE-7x7 study trace (with lp/)")
class StudyStepSixLabel(unittest.TestCase):
    """Study step 6: the header lists 3 new trajectories (18, 19, 26); the dialog's priority-7 row counts 2."""

    def test_row_label_does_not_claim_changed_flights(self):
        trace = TraceReader(os.environ["CE7_TRACE"])
        ex = IterationExplainer(trace, 6)
        self.assertEqual(sorted(int(f) for f in ex.record["flight_changes"]), [18, 19, 26])
        rows = [row for c in ex.why_sectors()["contrasts"] for row in c["ladder"] or [] if row["level"] == "changed"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["chosen"], 2)
            self.assertNotIn("changed flights", row["text"])
            self.assertIn("filed route", row["text"])


if __name__ == "__main__":
    unittest.main()
