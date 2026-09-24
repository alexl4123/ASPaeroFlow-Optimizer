"""One epsilon-constraint step of a Pareto front (02_ASP/pareto_step.py) is opt-in and exact.

  * without the options the encoding is returned unchanged (the byte-identical result lines of the
    existing command lines are checked end to end in the campaign's dev log, not here);
  * sectors-first swaps exactly the two weak constraints and nothing else, and refuses an encoding
    in which it cannot find each of them exactly once;
  * the delay bound excludes exactly the answer sets whose delay sum exceeds K, negative delays
    (the `signed` metric) included.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "02_ASP"))

import clingo  # noqa: E402

import pareto_step  # noqa: E402

ENCODING = (REPO / "02_ASP" / "encoding.lp").read_text()


class TestObjectiveOrder(unittest.TestCase):

    def test_default_changes_nothing(self):
        self.assertEqual(pareto_step.apply(ENCODING, None, None), ENCODING)
        self.assertEqual(pareto_step.apply(ENCODING, "delay-first", None), ENCODING)

    def test_sectors_first_swaps_exactly_two_lines(self):
        swapped = pareto_step.apply(ENCODING, "sectors-first", None)
        before, after = ENCODING.splitlines(), swapped.splitlines()
        self.assertEqual(len(before), len(after))
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        self.assertEqual(changed, [
            (":~ arrival_delay(ID,DIFF). [DIFF@9,ID]", ":~ arrival_delay(ID,DIFF). [DIFF@8,ID]"),
            (":~ sector_number(T,NUM). [NUM@8,T]", ":~ sector_number(T,NUM). [NUM@9,T]"),
        ])

    def test_refuses_a_second_swap(self):
        swapped = pareto_step.swap_to_sectors_first(ENCODING)
        with self.assertRaises(ValueError):
            pareto_step.swap_to_sectors_first(swapped)

    def test_refuses_a_third_constraint_at_the_same_level(self):
        with self.assertRaises(ValueError):
            pareto_step.swap_to_sectors_first(ENCODING + "\n:~ reroute(ID). [1@9,ID]\n")

    def test_commented_constraints_do_not_count(self):
        swapped = pareto_step.swap_to_sectors_first(ENCODING + "\n% :~ reroute(ID). [1@9,ID]\n")
        self.assertIn(":~ sector_number(T,NUM). [NUM@9,T]", swapped)


class TestDelayBound(unittest.TestCase):

    @staticmethod
    def models(program):
        ctl = clingo.Control(["0"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        found = []
        ctl.solve(on_model=lambda m: found.append(sorted(str(s) for s in m.symbols(shown=True))))
        return found

    # Two flights; flight 1 may arrive 2 late or 1 early, flight 2 is 3 late: sums 5 or 2.
    PROGRAM = ("1 { arrival_delay(1,2); arrival_delay(1,-1) } 1. arrival_delay(2,3). "
               "#show arrival_delay/2.")

    def test_bound_cuts_exactly_above_k(self):
        for k, expected in ((5, 2), (4, 1), (2, 1), (1, 0)):
            with self.subTest(k=k):
                program = self.PROGRAM + pareto_step.delay_bound_constraint(k)
                self.assertEqual(len(self.models(program)), expected)

    def test_negative_bound(self):
        program = "arrival_delay(1,-3). arrival_delay(2,1)." + pareto_step.delay_bound_constraint(-2)
        self.assertEqual(len(self.models(program)), 1)            # sum -2 <= -2
        program = "arrival_delay(1,-3). arrival_delay(2,2)." + pareto_step.delay_bound_constraint(-2)
        self.assertEqual(len(self.models(program)), 0)            # sum -1 > -2


class TestResultFields(unittest.TestCase):

    def test_inactive_without_options(self):
        from types import SimpleNamespace
        args = SimpleNamespace(objective_order=None, delay_bound=None, fixed_horizon=False)
        self.assertFalse(pareto_step.active(args))
        self.assertEqual(pareto_step.export_header(args), "")

    def test_fields_name_every_option(self):
        from types import SimpleNamespace
        args = SimpleNamespace(objective_order="sectors-first", delay_bound=7, fixed_horizon=True)
        self.assertEqual(pareto_step.result_fields(args), {
            "PARETO-OBJECTIVE-ORDER": "sectors-first", "PARETO-DELAY-BOUND": 7,
            "PARETO-FIXED-HORIZON": True})


if __name__ == "__main__":
    unittest.main()
