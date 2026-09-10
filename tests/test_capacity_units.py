"""Task 1: sectors.csv::Capacity is the capacity of ONE timestep.

capacity_time_matrix must broadcast that value across time for every granularity T.
Before the fix it returned roughly c/T, which under-stated capacity by a factor of T.
All four copies of the function must agree.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# (folder, module, class) for each copy of capacity_time_matrix in the repository.
IMPLEMENTATIONS = [
    ("01_ASPaeroFlow", "src.aspaeroflow.optimize_flights", "OptimizeFlights"),
    ("04_MIP", "mip_model", "MIPModel"),
    ("10_ANALYZE_NOMINAL_CAPACITY_REQUIREMENTS", "optimize_flights", "OptimizeFlights"),
    ("02_ASP", "translate", "TranslateCSVtoLogicProgram"),
]


def load(module_dir, module_name, cls_name):
    """Import one folder's copy the way its own entry point does (script dir on sys.path)."""
    d = str(REPO / module_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    return getattr(importlib.import_module(module_name), cls_name)


class TestCapacityIsPerTimestep(unittest.TestCase):

    def test_single_sector_capacity_is_per_timestep(self):
        """A sector of capacity c must have budget c in every slot, at every granularity."""
        for module_dir, module_name, cls_name in IMPLEMENTATIONS:
            cls = load(module_dir, module_name, cls_name)
            for T in (1, 4, 15, 60):
                for c in (1, 28, 45, 101, 1200):
                    with self.subTest(impl=module_dir, T=T, c=c):
                        n_times = 24 * T
                        cap = np.array([[0, c]], dtype=np.int64)
                        assignment = np.zeros((1, n_times), dtype=np.int64)
                        out = cls.capacity_time_matrix(cap, n_times, T, assignment)
                        self.assertEqual(out.shape, (1, n_times))
                        self.assertTrue(
                            np.all(out == c),
                            f"{module_dir}: expected {c} in every slot at T={T}, "
                            f"got min={out.min()} max={out.max()}",
                        )

    def test_capacity_is_independent_of_granularity(self):
        """The same capacity must yield the same per-slot budget at any T."""
        cls = load(*IMPLEMENTATIONS[0])
        c = 45
        for T in (1, 4, 15, 60):
            with self.subTest(T=T):
                out = cls.capacity_time_matrix(
                    np.array([[0, c]], dtype=np.int64),
                    24 * T, T,
                    np.zeros((1, 24 * T), dtype=np.int64),
                )
                self.assertEqual(set(np.unique(out).tolist()), {c})

    def test_all_implementations_agree(self):
        """The four copies must not drift apart on a non-trivial composite instance."""
        T, n_times, n = 4, 96, 12
        rng = np.random.default_rng(0)
        cap = np.stack([np.arange(n), rng.integers(1, 200, size=n)], axis=1).astype(np.int64)
        assignment = rng.integers(0, n, size=(n, n_times)).astype(np.int64)

        outs = [load(*impl).capacity_time_matrix(cap, n_times, T, assignment)
                for impl in IMPLEMENTATIONS]
        for impl, out in zip(IMPLEMENTATIONS[1:], outs[1:]):
            self.assertTrue(np.array_equal(outs[0], out),
                            f"{impl[0]} disagrees with 01_ASPaeroFlow")

    def test_empty_sectors_stay_zero(self):
        """A sector with no navpoints assigned keeps capacity 0; MAX rule still applies."""
        cls = load(*IMPLEMENTATIONS[0])
        T, n_times = 4, 16
        cap = np.array([[0, 50], [1, 70]], dtype=np.int64)
        assignment = np.zeros((2, n_times), dtype=np.int64)  # both navpoints -> sector 0
        out = cls.capacity_time_matrix(cap, n_times, T, assignment)
        self.assertTrue(np.all(out[1] == 0), "empty sector must have capacity 0")
        self.assertTrue(np.all(out[0] == 70), "MAX rule over {50,70} must give 70")


if __name__ == "__main__":
    unittest.main(verbosity=2)
