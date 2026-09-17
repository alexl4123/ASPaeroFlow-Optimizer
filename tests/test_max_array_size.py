"""No printed sbatch line may use an array index of MaxArraySize or more.

SLURM's MaxArraySize bounds the array task INDEX, not the number of tasks: the highest index it
accepts is MaxArraySize - 1. On the cluster (MaxArraySize=50000) `sbatch --array=1-50000` was
rejected with "Invalid job array specification", and build_worklist.py had printed exactly that
line for a full wave. The same wave arithmetic was in build_ablation_manifest.py.

These tests pin the boundary on the pure wave planners and on the lines both scripts actually
print, and check that the waves still cover every unit (or manifest row) exactly once, in order.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "06_benchmark_start_script"
for path in (str(REPO), str(BENCH)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_ablation_manifest as manifest                 # noqa: E402
import build_worklist as worklist                          # noqa: E402

ARRAY = re.compile(r"--array=1-(\d+)")


def covered(waves, chunk, first=0):
    """The units (or rows) the waves run, in order, as the slurm scripts compute them."""
    units = []
    for offset, n in waves:
        for task in range(1, n + 1):
            units.extend(range(offset + (task - 1) * chunk + 1, offset + task * chunk + 1))
    return units


class TestWavePlanners(unittest.TestCase):

    def test_the_rejected_submission(self):
        # 50,000 tasks at MaxArraySize=50000 used to be ONE --array=1-50000.
        self.assertEqual(worklist.plan_waves(50000, 1, 50000), [(0, 49999), (49999, 1)])
        self.assertEqual(manifest.plan_waves(1, 50000, 1, 50000), [(0, 49999), (49999, 1)])

    def test_one_below_the_limit_is_one_wave(self):
        self.assertEqual(worklist.plan_waves(49999, 1, 50000), [(0, 49999)])
        self.assertEqual(manifest.plan_waves(1, 49999, 1, 50000), [(0, 49999)])

    def test_no_wave_reaches_max_array_size_and_all_units_are_covered_once(self):
        for tasks in (1, 2, 999, 1000, 1001, 2000, 2001, 59000, 100000):
            for chunk in (1, 3):
                for size in (2, 3, 1001, 50000):
                    with self.subTest(tasks=tasks, chunk=chunk, size=size):
                        waves = worklist.plan_waves(tasks, chunk, size)
                        self.assertTrue(all(n <= size - 1 for _, n in waves))
                        self.assertEqual(covered(waves, chunk),
                                         list(range(1, tasks * chunk + 1)))
                        # The manifest planner, for a tier starting at row 101.
                        rows = manifest.plan_waves(101, tasks, chunk, size)
                        self.assertTrue(all(n <= size - 1 for _, n in rows))
                        self.assertEqual(covered(rows, chunk),
                                         list(range(101, 101 + tasks * chunk)))


class TestPrintedLines(unittest.TestCase):

    def test_build_worklist_prints_no_index_at_or_above_max_array_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "instances"
            problem = root / "P1"
            names = [f"{n:07d}_SEED42" for n in range(1, 8)]          # 7 instances
            for name in names:
                (problem / name).mkdir(parents=True)
            (root / "problems.tsv").write_text(
                "problem_dir\ttime_granularity\tregion\tcapacity_level\tn_instances\n"
                "P1\t1\tR\tPCAP100\t7\n", encoding="utf-8")
            out = Path(tmp) / "worklist.tsv"
            # --run-mip only: one system per instance, so 7 units, 7 tasks at CHUNK=1.
            printed = subprocess.run(
                [sys.executable, str(BENCH / "build_worklist.py"), "--instance-root", str(root),
                 "--out", str(out), "--run-mip", "only", "--max-array-size", "4"],
                capture_output=True, text=True, check=True, cwd=BENCH).stdout
            sizes = [int(n) for n in ARRAY.findall(printed)]
            offsets = [int(o) for o in re.findall(r"UNIT_OFFSET=(\d+)", printed)]
            self.assertEqual(sizes, [3, 3, 1], printed)
            self.assertEqual(offsets, [0, 3, 6], printed)
            # The worklist itself is untouched by this: one row per unit, numbered from 1.
            ids = [line.split("\t")[0] for line in out.read_text().splitlines()[1:]]
            self.assertEqual(ids, [str(i) for i in range(1, 8)])

    def test_build_ablation_manifest_prints_no_index_at_or_above_max_array_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tasks.tsv"
            printed = subprocess.run(
                [sys.executable, str(BENCH / "build_ablation_manifest.py"), "--assume-grid",
                 "--problems", "A,B", "--per-run-systems", "--tiers", "P,A",
                 "--max-array-size", "500", "--out", str(out)],
                capture_output=True, text=True, check=True, cwd=BENCH).stdout
            lines = [l for l in printed.splitlines() if "sbatch" in l]
            waves = [(int(re.search(r"ROW_OFFSET=(\d+)", l).group(1)),
                      int(ARRAY.search(l).group(1))) for l in lines]
            self.assertTrue(all(n <= 499 for _, n in waves), printed)
            rows = len(out.read_text().splitlines()) - 1
            self.assertEqual(covered(waves, 1), list(range(1, rows + 1)))


if __name__ == "__main__":
    unittest.main()
