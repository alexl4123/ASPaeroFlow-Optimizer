"""The navpoint-to-sector allocation can vary over time, and without a schedule nothing changes.

Fixture (a miniature of 30-0-EAST-ASIA-3x3): navpoints 0..8 form three sectors,
    sector 0 = {0,1,2},  sector 4 = {3,4,5},  sector 6 = {6,7,8},
and navpoints 9 and 10 are airports, each its own sector. A sector is open while its own
navpoint is assigned to it.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common import navpoint_sector_allocation as nsa  # noqa: E402

STATIC = np.array([[0, 0], [1, 0], [2, 0],
                   [3, 4], [4, 4], [5, 4],
                   [6, 6], [7, 6], [8, 6],
                   [9, 9], [10, 10]])
# One flight, landing at t=11, so with max_time=2 and granularity 4 the horizon is 12 timesteps.
FLIGHTS = np.array([[0, 9, 2], [0, 1, 5], [0, 10, 11]])
AIRPLANE_FLIGHT = np.array([[0, 0]])
MAX_TIME, GRANULARITY = 2, 4


AIRPORTS = np.array([9, 10])


def build(schedule=None, airports=AIRPORTS):
    return nsa.build_assignment(FLIGHTS, AIRPLANE_FLIGHT, STATIC, MAX_TIME, GRANULARITY,
                                schedule=schedule, airports=airports)


def write(tmp, text, name=nsa.SCHEDULE_FILENAME):
    path = Path(tmp) / name
    path.write_text(text)
    return path


class TestStaticAllocationUnchanged(unittest.TestCase):

    def test_no_schedule_broadcasts_the_static_file(self):
        a = build()
        self.assertEqual(a.shape, (11, 12))
        for navaid, sector in STATIC:
            self.assertTrue((a[navaid] == sector).all())

    def test_empty_schedule_is_no_schedule(self):
        self.assertTrue(np.array_equal(build([]), build()))

    def test_schedule_at_t0_only_equals_static(self):
        """What the generator emits for now: every navpoint once, at From_Time 0."""
        schedule = [(int(n), int(s), 0) for n, s in STATIC]
        self.assertTrue(np.array_equal(build(schedule), build()))

    def test_static_allocation_emits_no_asp_facts(self):
        self.assertEqual(nsa.asp_change_point_facts(build()), [])
        self.assertEqual(list(nsa.change_points(build())), [])

    def test_static_allocation_has_one_epoch(self):
        self.assertEqual(nsa.epoch_starts(build()), [0])
        self.assertFalse(nsa.is_time_varying(build()))


class TestChangePointsTakeEffect(unittest.TestCase):

    def test_navpoint_moves_at_k_and_stays_moved(self):
        a = build([(1, 4, 6)])
        self.assertTrue((a[1, :6] == 0).all())
        self.assertTrue((a[1, 6:] == 4).all())
        # nothing else moved
        others = np.delete(np.arange(a.shape[0]), 1)
        self.assertTrue(np.array_equal(a[others], build()[others]))

    def test_rows_hold_until_the_next_change_point_in_time_order(self):
        # given out of order on purpose
        a = build([(1, 0, 9), (1, 4, 3)])
        self.assertEqual(a[1].tolist(), [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0])

    def test_change_point_facts_describe_the_whole_row(self):
        facts = nsa.asp_change_point_facts(build([(1, 4, 6)]))
        self.assertEqual(facts, ["navpoint_sector_from(1,0,0).",
                                 "navpoint_sector_from(1,4,6)."])

    def test_epochs_are_the_union_of_every_navpoints_change_points(self):
        a = build([(1, 4, 6), (1, 0, 9), (5, 6, 3)])
        self.assertEqual(nsa.epoch_starts(a), [0, 3, 6, 9])
        self.assertTrue(nsa.is_time_varying(a))
        for start, stop in [(0, 3), (3, 6), (6, 9), (9, 12)]:
            block = a[:, start:stop]
            self.assertTrue((block == block[:, :1]).all())

    def test_change_point_past_horizon_has_no_effect(self):
        stderr = io.StringIO()
        old, sys.stderr = sys.stderr, stderr
        try:
            a = build([(1, 4, 500)])
        finally:
            sys.stderr = old
        self.assertTrue(np.array_equal(a, build()))
        self.assertIn("past the end of the horizon", stderr.getvalue())


class TestInconsistentSchedulesAreRejected(unittest.TestCase):

    def test_unknown_navpoint(self):
        with self.assertRaisesRegex(nsa.ScheduleError, "navpoint 99"):
            build([(99, 0, 3)])

    def test_navpoint_missing_from_the_static_file(self):
        gap = STATIC[STATIC[:, 0] != 8]       # 8 is inside the id range but not listed
        with self.assertRaisesRegex(nsa.ScheduleError, "navpoint 8, which navaid_sector_assignment"):
            nsa.build_assignment(FLIGHTS, AIRPLANE_FLIGHT, gap, MAX_TIME, GRANULARITY,
                                 schedule=[(8, 6, 3)])

    def test_from_time_0_must_agree_with_the_static_file(self):
        with self.assertRaisesRegex(nsa.ScheduleError, "at From_Time 0, but navaid_sector_assignment"):
            build([(1, 4, 0)])

    def test_moving_an_airport(self):
        with self.assertRaisesRegex(nsa.ScheduleError, "moves airport 9"):
            build([(9, 0, 3)])

    def test_joining_an_airport_sector(self):
        with self.assertRaisesRegex(nsa.ScheduleError, "into airport sector 10"):
            build([(1, 10, 3)])

    def test_airport_checks_need_the_airports(self):
        # without the airport list the builder cannot tell; the solvers always pass it
        a = build([(1, 10, 3)], airports=None)
        self.assertEqual(int(a[1, 3]), 10)

    def test_unknown_sector(self):
        with self.assertRaisesRegex(nsa.ScheduleError, "sector 99"):
            build([(1, 99, 3)])

    def test_moving_into_a_closed_sector(self):
        # navpoint 2 is a member of sector 0, never a sector of its own
        with self.assertRaisesRegex(nsa.ScheduleError, "sector 2 is not open"):
            build([(1, 2, 3)])

    def test_closing_a_sector_that_still_has_members(self):
        # moving sector 0's own navpoint away strands 1 and 2, which did not move
        with self.assertRaisesRegex(nsa.ScheduleError, "sector 0 is not open"):
            build([(0, 4, 3)])

    def test_closing_a_sector_is_fine_once_its_members_have_left(self):
        a = build([(0, 4, 3), (1, 4, 3), (2, 4, 3)])
        self.assertTrue((a[:3, 3:] == 4).all())


class TestScheduleFile(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "Navaid_ID,Sector_ID,From_Time\n1,4,6\n1,0,9\n3,4,0\n")
            self.assertEqual(nsa.load_schedule(path), [(1, 4, 6), (1, 0, 9), (3, 4, 0)])

    def test_header_only_is_an_empty_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(nsa.load_schedule(write(tmp, "Navaid_ID,Sector_ID,From_Time\n")), [])

    def test_absent_sibling_means_static(self):
        with tempfile.TemporaryDirectory() as tmp:
            static = write(tmp, "Navaid_ID,Sector_ID\n0,0\n", "navaid_sector_assignment.csv")
            self.assertIsNone(nsa.load_schedule_for(static))

    def test_present_sibling_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            static = write(tmp, "Navaid_ID,Sector_ID\n0,0\n", "navaid_sector_assignment.csv")
            write(tmp, "Navaid_ID,Sector_ID,From_Time\n1,4,6\n")
            self.assertEqual(nsa.load_schedule_for(static), [(1, 4, 6)])

    def test_a_schedule_elsewhere_is_not_picked_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            static = write(tmp, "Navaid_ID,Sector_ID\n0,0\n", "navaid_sector_assignment.csv")
            write(tmp, "Navaid_ID,Sector_ID,From_Time\n1,4,6\n", "other_schedule.csv")
            self.assertIsNone(nsa.load_schedule_for(static))

    def test_malformed_files_fail_with_the_line_named(self):
        cases = {
            "wrong header":      ("Navaid_ID,Sector_ID\n1,4\n",                     ":1: header"),
            "two columns":       ("Navaid_ID,Sector_ID,From_Time\n1,4\n",            ":2: has 2 field"),
            "four columns":      ("Navaid_ID,Sector_ID,From_Time\n1,4,6,7\n",        ":2: has 4 field"),
            "non-integer":       ("Navaid_ID,Sector_ID,From_Time\n1,4,six\n",        ":2: has a non-integer"),
            "fractional time":   ("Navaid_ID,Sector_ID,From_Time\n1,4,6.5\n",        ":2: has a non-integer"),
            "negative time":     ("Navaid_ID,Sector_ID,From_Time\n1,4,-1\n",         ":2: From_Time is negative"),
            "duplicate":         ("Navaid_ID,Sector_ID,From_Time\n1,4,6\n1,0,6\n",   ":3: navpoint 1 already"),
            "empty file":        ("",                                                "file is empty"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for label, (text, fragment) in cases.items():
                with self.subTest(case=label):
                    path = write(tmp, text)
                    with self.assertRaises(nsa.ScheduleError) as ctx:
                        nsa.load_schedule(path)
                    self.assertIn(fragment, str(ctx.exception))
                    self.assertIn("Navaid_ID,Sector_ID,From_Time", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
