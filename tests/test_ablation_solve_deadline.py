"""The opt-in solve deadline, and the ablation pieces that depend on it.

The claim that matters -- that stopping from inside works on REAL instances, how fast, and what the
benchmark caller records -- cannot be made by a unit test and is not made here; it was measured on
generated instances. These tests pin the contracts that measurement relies on:

  * Solver(deadline=None) is the blocking solve it always was: no observer, no extra keys;
  * with a deadline, a search that cannot close is stopped close to it, and the statistics
    (lower bound, the priorities of the cost vector) are read afterwards, with or without a model;
  * the caller passes --solve-deadline to 02_ASP systems only, and only when asked;
  * the analyser places 4-6 level vectors on the six objective levels and ignores bounds from a
    horizon the re-solve loop would have left;
  * tier B of the ablation is the ten named variants, one row per run, and there is no tier C.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import contextlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from importlib import util
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "06_benchmark_start_script"
for path in (str(REPO), str(BENCH)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load(name, path):
    # The module's own folder goes on sys.path first: 02_ASP/solver.py imports its sibling
    # bootstrap shims (navpoint_sector_allocation_bootstrap, ...) by plain name.
    folder = str(Path(path).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


solver_module = _load("asp02_solver_deadline", REPO / "02_ASP" / "solver.py")
import analyze_asp_ablation as analyser                    # noqa: E402
import build_ablation_manifest as manifest                 # noqa: E402
import start_benchmark_caller as caller                    # noqa: E402


@contextlib.contextmanager
def _quiet():
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


#: Closes immediately.
EASY = """
sector(1..4).
1 { pick(S) : sector(S) } 1.
overload(a,1,N) :- pick(N).
arrival_delay(f,N) :- pick(N).
sector_number(1,N) :- pick(N).
sector_diff(1,0).
:~ pick(S). [S@10,S]
"""

#: Does not close within seconds under branch-and-bound (measured: still open after 5 s).
HARD = """
n(1..70).
{ p(I) : n(I) }.
:- #count{ I : p(I) } != 35.
overload(a,1,C)    :- C = #sum{ I*I \\ 13, I : p(I) }.
arrival_delay(f,C) :- C = #count{ I,J : p(I), p(J), I < J, (I+J) \\ 11 = 0 }.
sector_number(1,C) :- C = #count{ I : p(I) }.
sector_diff(1,0).
:~ p(I). [I*I\\13@10,I]
:~ p(I), p(J), I < J, (I+J) \\ 11 = 0. [1@9,I,J]
"""


def _solve(program, deadline=None, stats=True):
    solver = solver_module.Solver(program, "", seed=1, report_solver_stats=stats,
                                  deadline=deadline)
    with _quiet():
        model = solver.solve()
    return solver, model


class TestSolverDeadline(unittest.TestCase):

    def test_without_a_deadline_nothing_is_added(self):
        solver, model = _solve(EASY)
        self.assertTrue(model.computation_finished)
        self.assertIsNone(solver.solve_summary)
        self.assertIsNone(solver.search_started_at)
        self.assertNotIn("SOLVER-COST-PRIORITIES", model.get_model_optimization_string())

    def test_a_search_that_ends_by_itself_is_not_stopped(self):
        solver, model = _solve(EASY, deadline=time.monotonic() + 60)
        self.assertFalse(solver.stopped_at_deadline)
        self.assertTrue(model.computation_finished)
        self.assertEqual(solver.solve_summary["SOLVER-COST-PRIORITIES"], [10])

    def test_a_search_that_cannot_close_is_stopped_near_the_deadline(self):
        deadline = time.monotonic() + 1.0
        solver, model = _solve(HARD, deadline=deadline)
        self.assertTrue(solver.stopped_at_deadline)
        self.assertIsNotNone(model)
        self.assertFalse(model.computation_finished)
        self.assertLess(solver.solve_ended_at - deadline, 0.5, "stopping took too long")
        summary = solver.solve_summary
        self.assertEqual(summary["SOLVER-COST-PRIORITIES"], [10, 9])
        self.assertEqual(len(summary["SOLVER-LOWER-BOUND"]), 2)
        self.assertFalse(summary["SOLVER-EXHAUSTED"])

    def test_no_model_still_reports_statistics_without_infinity(self):
        solver, model = _solve(HARD, deadline=time.monotonic() - 1.0)
        self.assertIsNone(model)
        self.assertTrue(solver.stopped_at_deadline)
        self.assertIsNone(solver.solve_summary["SOLVER-COSTS"])     # clingo says [inf, inf]
        self.assertIsNotNone(solver.solve_summary["SOLVER-LOWER-BOUND"])

    def test_finite_or_none(self):
        self.assertIsNone(solver_module._finite_or_none([float("inf"), 1.0]))
        self.assertIsNone(solver_module._finite_or_none(None))
        self.assertEqual(solver_module._finite_or_none([1.0, -2.0]), [1.0, -2.0])


class TestMainDeadlineOption(unittest.TestCase):

    def test_rejects_non_positive(self):
        for bad in ("0", "-5", "inf"):
            with self.subTest(value=bad), tempfile.TemporaryDirectory() as tmp:
                for name in ("graph_edges", "sectors", "flights", "airports", "airplanes",
                             "airplane_flight_assignment", "navaid_sector_assignment"):
                    Path(tmp, f"{name}.csv").write_text("")
                result = subprocess.run(
                    [sys.executable, str(REPO / "02_ASP" / "main.py"), f"--data-dir={tmp}",
                     f"--encoding-path={REPO / '02_ASP' / 'encoding.lp'}",
                     f"--solve-deadline={bad}"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--solve-deadline", result.stderr)


class TestCallerMargin(unittest.TestCase):

    def systems(self, extra):
        args = caller.build_arg_parser().parse_args(["problem", "--time-limit=120", *extra])
        return args, caller.build_system_config(BENCH, Path("out"), "t", args)

    def test_absent_margin_changes_no_command_line(self):
        args, systems = self.systems([])
        self.assertTrue(all(not any(f.startswith("--solve-deadline") for f in
                                    caller.solver_option_cli(s, args)) for s in systems))

    def test_margin_reaches_02_asp_only(self):
        args, systems = self.systems(["--solve-deadline-margin=30"])
        for system in systems:
            with self.subTest(system=system["key"]):
                flags = caller.solver_option_cli(system, args)
                if "02_ASP" in str(system["script"]):
                    self.assertIn("--solve-deadline=90", flags)
                else:
                    self.assertFalse(any(f.startswith("--solve-deadline") for f in flags))

    def test_margin_must_fit_the_limit(self):
        for margin in (0, 120, 200):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                caller.solve_deadline_for(120, margin)


class TestAnalyserVectors(unittest.TestCase):

    def test_short_vectors_are_placed_by_priority(self):
        self.assertEqual(analyser.six_levels([16.0, 300.0, 0.0, 0.0], [10, 8, 7, 6]),
                         [16, 0, 300, 0, 0, 0])
        self.assertEqual(analyser.six_levels([1, 2, 3, 4, 5, 6]), [1, 2, 3, 4, 5, 6])
        self.assertIsNone(analyser.six_levels([16.0, 300.0, 0.0, 0.0]))
        self.assertIsNone(analyser.six_levels([float("inf")] * 6))
        self.assertIsNone(analyser.six_levels(None))

    def test_bound_from_a_non_final_horizon_is_ignored(self):
        line = {"SOLVER-LOWER-BOUND": [0, 1, 2, 3, 4, 5],
                "SOLVER-COST-PRIORITIES": [10, 9, 8, 7, 6, 5]}
        self.assertEqual(analyser.lower_bound(line), [0, 1, 2, 3, 4, 5])
        self.assertIsNone(analyser.lower_bound({**line, "SOLVER-HORIZON-FINAL": False}))


class TestAblationTiers(unittest.TestCase):

    def test_tier_b_is_the_ten_named_variants(self):
        keys = manifest.breadth10_systems()
        self.assertEqual([k.split("_ASP_", 1)[1] for k in keys],
                         ["rp_dp_sp", "rp_d_sp", "rp_nd_sp", "rp_dp_ns", "rp_dp_s",
                          "nr_dp_sp", "r_dp_sp", "nr_nd_ns", "r_d_s", "r_d_ns"])
        self.assertTrue(set(keys) <= set(manifest.all27_systems()))
        self.assertEqual(sum(1 for k in keys if not k.endswith("_sp")), 5)

    def run_builder(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "m.tsv")
            result = subprocess.run(
                [sys.executable, str(BENCH / "build_ablation_manifest.py"), "--assume-grid",
                 "--problems", "A,B,C,D,E", "--out", str(out), *extra],
                capture_output=True, text=True, cwd=BENCH)
            rows = out.read_text().splitlines()[1:] if out.exists() else []
            return result, [r.split("\t") for r in rows]

    def test_default_tiers_and_counts(self):
        result, rows = self.run_builder("--per-run-systems")
        self.assertEqual(result.returncode, 0, result.stderr)
        tiers = [r[1] for r in rows]
        self.assertEqual((tiers.count("P"), tiers.count("A"), tiers.count("B")), (8, 1600, 800))
        self.assertNotIn("C", tiers)
        self.assertTrue(all(r[3] == "1" for r in rows))
        self.assertNotIn("tier C", result.stdout)

    def test_tier_b_without_per_run_rows_is_refused(self):
        result, rows = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--per-run-systems", result.stderr + result.stdout)

    def test_tier_c_is_gone(self):
        result, _ = self.run_builder("--per-run-systems", "--tiers", "C")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
