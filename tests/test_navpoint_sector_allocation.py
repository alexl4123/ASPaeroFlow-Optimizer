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

    def test_static_allocation_has_no_change_points(self):
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

    def test_change_points_describe_the_whole_row(self):
        self.assertEqual(list(nsa.change_points(build([(1, 4, 6)]))), [(1, [(0, 0), (6, 4)])])

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


def sector_number(assignment):
    """SECTOR-NUMBER, as every folder computes it: distinct sectors per column, summed."""
    return int(sum(len(set(assignment[:, t].tolist())) for t in range(assignment.shape[1])))


class TestTheEvaluationWindow(unittest.TestCase):
    """SECTOR-NUMBER and RECONFIG are sums over time, so the axis has to be the instance's.

    Without this, a solver that widened its matrices reported a larger number for the very same
    sectorisation: 1364 / 484 / 367 / 275 from the four systems on one 10-flight instance where
    nothing in the airspace differed.
    """

    def test_it_is_the_width_build_assignment_uses(self):
        self.assertEqual(nsa.evaluation_window(FLIGHTS, MAX_TIME, GRANULARITY),
                         build().shape[1])

    def test_a_flight_past_max_time_widens_it_to_a_whole_number_of_buckets(self):
        # The fixture's flight lands at t=11, past (MAX_TIME + 1) * GRANULARITY would otherwise
        # be the only term. The window stays a multiple of the granularity either way.
        self.assertEqual(nsa.evaluation_window(FLIGHTS, MAX_TIME, GRANULARITY), 12)
        late = np.array([[0, 9, 2], [0, 1, 5], [0, 10, 13]])
        self.assertEqual(nsa.evaluation_window(late, MAX_TIME, GRANULARITY), 16)
        self.assertEqual(nsa.evaluation_window(late, MAX_TIME, GRANULARITY) % GRANULARITY, 0)

    def test_padding_holds_the_last_column(self):
        widened = nsa.to_window(build(), 20)
        self.assertEqual(widened.shape[1], 20)
        for t in range(12, 20):
            np.testing.assert_array_equal(widened[:, t], build()[:, -1])

    def test_truncation_drops_the_extra_columns(self):
        np.testing.assert_array_equal(nsa.to_window(build(), 5), build()[:, :5])

    def test_a_widened_matrix_scores_what_the_narrow_one_scored(self):
        """The actual defect: the same allocation on a longer axis must not score higher."""
        base = build()
        on_its_own_axis = sector_number(base)
        widened = np.hstack([base, np.repeat(base[:, [-1]], 100, axis=1)])
        self.assertGreater(sector_number(widened), on_its_own_axis)          # the old behaviour
        self.assertEqual(sector_number(nsa.to_window(widened, base.shape[1])),
                         on_its_own_axis)

    def test_reconfig_counts_the_same_cells_on_either_axis(self):
        """A navpoint that moves and stays moved contributes once per remaining column."""
        moved = build().copy()
        moved[3, 6:] = 0
        window = moved.shape[1]
        expected = int(np.count_nonzero(moved != build()))
        widened = np.hstack([moved, np.repeat(moved[:, [-1]], 40, axis=1)])
        reference = np.hstack([build(), np.repeat(build()[:, [-1]], 40, axis=1)])
        self.assertGreater(int(np.count_nonzero(widened != reference)), expected)
        self.assertEqual(
            int(np.count_nonzero(nsa.to_window(widened, window)
                                 != nsa.to_window(reference, window))),
            expected)

    def test_none_leaves_the_matrix_alone(self):
        base = build()
        self.assertIs(nsa.to_window(base, None), base)

    def test_a_series_is_padded_and_cut_the_same_way(self):
        """02_ASP holds these metrics as one value per timestep, not as a matrix."""
        counts = {0: 11, 1: 11, 2: 11}
        self.assertEqual(nsa.series_to_window(counts, 5), [11, 11, 11, 11, 11])
        self.assertEqual(nsa.series_to_window(counts, 2), [11, 11])

    def test_a_silent_series_is_padded_from_its_axis_and_not_from_its_last_entry(self):
        """A count series says nothing where the count is zero, so it cannot state its own axis.

        reconfig(NAV,T) exists only where something moved. Padding from the highest atom would
        repeat that count across the rest of the window; padding from the time domain repeats
        the zero that is actually there.
        """
        reconfig = {3: 2, 4: 2}
        self.assertEqual(sum(nsa.series_to_window(reconfig, 10, own_width=8)), 4)
        self.assertEqual(sum(nsa.series_to_window(reconfig, 10)), 4 + 2 * 5)   # without the axis

    def test_an_empty_series_is_zero_over_the_whole_window(self):
        self.assertEqual(nsa.series_to_window({}, 4), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
