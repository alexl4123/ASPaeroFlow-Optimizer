"""The Pareto campaign's planner (06_benchmark_start_script/build_pareto_manifest.py).

  * round G's grid: every interior K up to 24 of them, a stride beyond, K = D_min - 1 first, and
    D_min + 1 .. D_min + 24 when the sectors-first end is open; nothing when the delay-first end is;
  * the campaign on the small family is the draft's 780 fronts (580 full, 200 ends only);
  * labels round-trip.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "06_benchmark_start_script"
sys.path.insert(0, str(SCRIPTS))

import pareto_lib as pl  # noqa: E402

_spec = importlib.util.spec_from_file_location("_bpm", SCRIPTS / "build_pareto_manifest.py")
bpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpm)


def ends(d_min=None, d_s=None):
    return {"df_proven": d_min is not None, "sf_proven": d_s is not None,
            "d_min": d_min, "d_s": d_s}


class TestGrid(unittest.TestCase):

    def test_every_interior_k_up_to_the_cap(self):
        self.assertEqual(bpm.grid_bounds(ends(3, 11), "signed"), [2, 10, 9, 8, 7, 6, 5, 4])

    def test_stride_beyond_the_cap(self):
        ks = bpm.grid_bounds(ends(0, 60), "signed")          # 59 interior values, stride 3
        self.assertEqual(ks[0], -1)
        self.assertEqual(ks[1:4], [59, 56, 53])
        self.assertLessEqual(len(ks) - 1, pl.GRID_CAP)
        self.assertTrue(all(0 < k < 60 for k in ks[1:]))

    def test_sectors_first_end_open(self):
        self.assertEqual(bpm.grid_bounds(ends(30), "signed"), [29] + list(range(31, 55)))

    def test_delay_first_end_open(self):
        self.assertEqual(bpm.grid_bounds(ends(None, 12), "signed"), [])

    def test_floored_zero_skips_the_unsatisfiable_step(self):
        self.assertEqual(bpm.grid_bounds(ends(0, 3), "floored"), [2, 1])
        self.assertEqual(bpm.grid_bounds(ends(0, 3), "signed"), [-1, 2, 1])

    def test_single_point_front(self):
        self.assertEqual(bpm.grid_bounds(ends(5, 5), "signed"), [4])


class TestCampaign(unittest.TestCase):

    def test_fronts_per_group(self):
        full = sum(len(g["variants"]) * len(g["full"]) for g in pl.GROUPS.values()) - 2 * 2
        ends_only = sum(len(g["variants"]) * len(g["ends"]) for g in pl.GROUPS.values())
        # x 5 regions x 4 seeds; r_dp_sp and r_nd_sp run at 20 flights only (-2 sizes each)
        self.assertEqual(full * 20, 580)
        self.assertEqual(ends_only * 20, 200)

    def test_memory_classes(self):
        self.assertEqual({pl.mem_class_of(v) for v in pl.GROUPS["B"]["variants"]}, {"40G"})
        self.assertEqual({pl.mem_class_of(v) for g in "AC" for v in pl.GROUPS[g]["variants"]},
                         {"8G"})

    def test_labels_round_trip(self):
        label = pl.make_label("30-1-CENTRAL-EUROPE-5x5-V2", "0000020_SEED42", "rp_d_sp")
        self.assertEqual(label, "CENTRAL-EUROPE-5x5_20_42__rp_d_sp")
        self.assertEqual(pl.parse_label(label)["metric"], "signed")
        label = pl.make_label("V1-MAJOR-EUROPE-10x10", "0000020_SEED150699", "rp_dp_sp", "floored")
        self.assertEqual(pl.parse_label(label),
                         {"region": "V1-MAJOR-EUROPE-10x10", "flights": 20, "seed": 150699,
                          "variant": "rp_dp_sp", "metric": "floored"})

    def test_regulation_flags(self):
        self.assertEqual(pl.regulation_flags("rp_d_sp"), (2, 1, 1))
        self.assertEqual(pl.regulation_flags("nr_dp_sp"), (1, 0, 1))
        self.assertEqual(pl.regulation_flags("r_nd_sp"), (0, 2, 1))
        self.assertEqual(pl.regulation_flags("rp_dp_s"), (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
