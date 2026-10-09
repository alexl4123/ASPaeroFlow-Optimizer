"""Certification of a (delay, sectors) front from its steps (06_benchmark_start_script/pareto_lib).

The steps are built by hand, so each rule of pareto_lib.certify() is pinned on its own:

  * a front swept to the end with proven steps is exact, and its points are the laptop's
    (central-europe seed 42, rp_d_sp: (1,307) (2,304) (3,303) (6,302) (9,301) (12,300));
  * a bounded step that repeats the last point does NOT pin the right end -- only the unbounded
    sectors-first step does (the corrected rule of plot_fronts.py);
  * a step stopped at its deadline certifies a stretch when its lower bound is high enough, and
    does not when it is not;
  * the left end: the delay-first end, a bound that forces overload up, or `floored` with D_min 0;
  * a no-model step contributes its bound, an unsatisfiable bound certifies everything below it.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "06_benchmark_start_script"))

import pareto_lib as pl  # noqa: E402

LABEL = "CENTRAL-EUROPE-5x5_20_42__rp_d_sp__floored"
SIGNED = "CENTRAL-EUROPE-5x5_20_42__rp_d_sp"


def cost(o, s, d):
    return {"o": o, "s": s, "d": d, "sd": 0, "rr": 0, "rc": 0}


def step(key, inc=None, lower=None, exhausted=False, stopped=False):
    order = "delay-first" if key == "DF" else "sectors-first"
    bound = int(key[1:]) if key.startswith("K") else None
    s = pl.Step(label=LABEL, round="G", key=key, order=order, bound=bound,
                path=Path(f"G_{key}.jsonl"))
    s.final = {"SOLVER-STOPPED-AT-DEADLINE": stopped}
    s.exhausted = exhausted
    s.stopped = stopped
    if inc is not None:
        s.incumbent = cost(*inc)
        s.models = [cost(*inc)]
    s.lower = dict(s.incumbent) if (exhausted and inc is not None) else (
        cost(*lower) if lower is not None else None)
    return s


def proven(key, o, s, d):
    return step(key, inc=(o, s, d), exhausted=True)


# The laptop sweep of ce42 (floored), as proven steps: unbounded, then K = d - 1 down to K = 0.
LAPTOP = [proven("SF", 0, 300, 12), proven("K11", 0, 301, 9), proven("K8", 0, 302, 6),
          proven("K5", 0, 303, 3), proven("K2", 0, 304, 2), proven("K1", 0, 307, 1),
          proven("K0", 1, 307, 0)]


class TestExactFront(unittest.TestCase):

    def test_laptop_sweep_is_exact(self):
        front = pl.certify(LABEL, LAPTOP)
        self.assertEqual(front.corners, [(1, 307), (2, 304), (3, 303), (6, 302), (9, 301),
                                         (12, 300)])
        self.assertTrue(all(front.point_proven))
        self.assertTrue(all(front.stretch_ok))
        self.assertEqual(front.cls, "exact")
        self.assertEqual(front.certified_share, 1.0)

    def test_delay_first_end_replaces_the_overload_step(self):
        # Without K0 the left end needs another certificate: here the proven delay-first end.
        steps = LAPTOP[:-1] + [proven("DF", 0, 307, 1)]
        self.assertEqual(pl.certify(SIGNED, steps).cls, "exact")
        self.assertFalse(pl.certify(SIGNED, LAPTOP[:-1]).left_ok)

    def test_floored_needs_no_left_certificate_at_zero(self):
        steps = [proven("SF", 0, 300, 2), proven("K1", 0, 301, 0)]
        self.assertEqual(pl.certify(LABEL, steps).cls, "exact")
        self.assertEqual(pl.certify(SIGNED, steps).cls, "partial")   # signed: delay may go < 0


class TestRightEnd(unittest.TestCase):

    def test_bounded_repeat_does_not_pin_the_right_end(self):
        # The unbounded step timed out; K = 14 returned (12, 300) proven. f may still drop at
        # K > 14, so the right end stays open.
        steps = [s for s in LAPTOP if s.key != "SF"] + [proven("K14", 0, 300, 12),
                                                        step("SF", inc=(0, 305, 20),
                                                             lower=(0, 290, 0), stopped=True)]
        front = pl.certify(LABEL, steps)
        self.assertFalse(front.right_ok)
        self.assertEqual(front.cls, "partial")

    def test_unbounded_lower_bound_equal_to_last_point_pins_it(self):
        steps = [s for s in LAPTOP if s.key != "SF"] + [proven("K14", 0, 300, 12),
                                                        step("SF", inc=(0, 305, 20),
                                                             lower=(0, 300, 0), stopped=True)]
        front = pl.certify(LABEL, steps)
        self.assertTrue(front.right_ok)
        self.assertEqual(front.cls, "exact")


class TestLowerBoundsCertifyStretches(unittest.TestCase):

    # Corners (1,307) (2,304) (3,303) (9,301) (12,300). The K8 step is the only one that can say
    # anything about f on [3, 8]: the exhausted K11 step (301 sectors at delay 9) gives f >= 302
    # there, not 303.
    def base(self):
        return [proven("SF", 0, 300, 12), proven("K11", 0, 301, 9), proven("K2", 0, 304, 2),
                proven("K1", 0, 307, 1), proven("K0", 1, 307, 0)]

    def test_timed_out_step_with_high_bound_certifies(self):
        # K8 stopped with incumbent (0,303,3) and bound 303 = its incumbent: (3,303) is on the
        # front and the stretches on both sides of it are certified, although K8 never finished.
        steps = self.base() + [step("K8", inc=(0, 303, 3), lower=(0, 303, 0), stopped=True)]
        front = pl.certify(LABEL, steps)
        self.assertEqual(front.corners, [(1, 307), (2, 304), (3, 303), (9, 301), (12, 300)])
        self.assertTrue(all(front.stretch_ok))
        self.assertEqual(front.cls, "exact")

    def test_timed_out_step_with_low_bound_does_not(self):
        steps = self.base() + [step("K8", inc=(0, 303, 3), lower=(0, 298, 0), stopped=True)]
        front = pl.certify(LABEL, steps)
        self.assertEqual(front.stretch_ok, [True, False, False, True])
        self.assertFalse(front.point_proven[2])
        self.assertEqual(front.cls, "partial")
        self.assertEqual(front.L(5), 302)

    def test_no_model_step_contributes_its_bound(self):
        steps = self.base() + [step("K8", lower=(0, 303, 0), stopped=True),
                               proven("K3", 0, 303, 3)]
        self.assertEqual(pl.certify(LABEL, steps).cls, "exact")

    def test_bound_on_unfinished_overload_level_is_not_used(self):
        # lower overload 0 < o* would be meaningless here; o* is 0, so use o* = 1 instead: all
        # points at overload 1, a step whose overload bound is still 0 contributes nothing.
        steps = [proven("SF", 1, 300, 5), proven("DF", 1, 305, 1),
                 step("K3", inc=(1, 303, 3), lower=(0, 0, 0), stopped=True)]
        front = pl.certify(SIGNED, steps)
        self.assertEqual(front.o_star, 1)
        self.assertFalse(any(front.stretch_ok))

    def test_unsatisfiable_bound(self):
        steps = [proven("SF", 0, 300, 2), step("K-1", exhausted=True)]
        front = pl.certify(SIGNED, steps)
        self.assertEqual(front.L(-1), pl.INF)


class TestFrontCsv(unittest.TestCase):

    def test_five_columns_in_sectors_first_order(self):
        rows = pl.front_csv_rows(LABEL, [proven("SF", 0, 300, 12), proven("DF", 0, 307, 1),
                                         step("K5", inc=(0, 303, 3), stopped=True),
                                         step("K4", stopped=True)])
        self.assertEqual([r.split(",")[:3] for r in rows], [
            [LABEL, "none", "OPTIMUM FOUND"], [LABEL, "5", "SATISFIABLE"],
            [LABEL, "4", "UNKNOWN"], [LABEL, "1", "OPTIMUM FOUND"]])
        self.assertTrue(all(len(r.split(",")) == 5 for r in rows))
        self.assertEqual(rows[0].split(",")[4], "0 300 12 0 0 0")
        self.assertEqual(rows[2].split(",")[4], "none")


class TestPlace(unittest.TestCase):

    def test_missing_levels_by_priority(self):
        # no reroute level (@6) in an nr program
        self.assertEqual(pl.place([0, 7, 300, 2, 1], [10, 9, 8, 7, 5], "delay-first"),
                         {"o": 0, "d": 7, "s": 300, "sd": 2, "rr": 0, "rc": 1})
        self.assertEqual(pl.place([0, 300, 7, 2, 0, 1], None, "sectors-first"),
                         {"o": 0, "s": 300, "d": 7, "sd": 2, "rr": 0, "rc": 1})
        self.assertIsNone(pl.place([], None, "sectors-first"))


if __name__ == "__main__":
    unittest.main()
