"""COMPUTATION-FINISHED must mean "the search was closed", not "we reached the end of main()".

02_ASP/main.py used to assign `model.computation_finished = True` unconditionally just before
printing the result line, so the field said "finished" for every run that reached that line,
proved optimum or not -- while every line a timeout could see said False, because it came from
`Model.optimality_proven`, which clingo leaves False under the default --opt-mode=opt even for a
run that exhausted the search space.

It now comes from clingo's SolveResult.exhausted, which is the signal that the search space was
closed and the optimum proven.

LIMITATION these tests cannot cover, and which the field does not claim to fix: the benchmark's
time limit arrives as an external SIGKILL, so a run that overruns never reaches the final print at
all. This makes the field trustworthy for runs that COMPLETE within the limit; a killed run's last
visible line is still an intermediate model carrying False.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import contextlib
import os
import sys
import unittest
from importlib import util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_02_asp_solver():
    """02_ASP runs as a top-level script, so its solver is loaded by path."""
    spec = util.spec_from_file_location("asp02_solver", REPO / "02_ASP" / "solver.py")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


solver_module = _load_02_asp_solver()


@contextlib.contextmanager
def _quiet():
    """Silence the result lines Solver.on_model writes straight to file descriptor 1."""
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


#: Small, and closed by clasp immediately.
TRIVIAL = """
sector(1..4).
1 { pick(S) : sector(S) } 1.
overload(a,1,N) :- pick(N).
arrival_delay(f,N) :- pick(N).
sector_number(1,N) :- pick(N).
sector_diff(1,0).
:~ pick(S). [S@10,S]
"""

#: Enough search that a one-conflict budget stops it with a model in hand but nothing proven.
SEARCH = """
n(1..14).
{ p(I) : n(I) }.
:- #count{ I : p(I) } < 7.
:- #count{ I : p(I) } > 7.
overload(a,1,C)    :- C = #count{ I : p(I) }.
arrival_delay(f,C) :- C = #count{ I : p(I), I > 7 }.
sector_number(1,C) :- C = #count{ I : p(I) }.
sector_diff(1,0).
:~ p(I). [I@10,I]
:~ p(I). [1@9,I]
"""


def _solve(program, options=None):
    solver = solver_module.Solver(program, "", seed=1, solver_options=options or [])
    with _quiet():
        model = solver.solve()
    return solver, model


class TestComputationFinished(unittest.TestCase):

    def test_true_when_the_search_is_exhausted(self):
        for label, program in (("trivial", TRIVIAL), ("with search", SEARCH)):
            with self.subTest(program=label):
                solver, model = _solve(program)
                self.assertTrue(solver.search_exhausted)
                self.assertIsNotNone(model)
                self.assertTrue(model.computation_finished)

    def test_false_when_the_search_was_stopped_at_a_limit(self):
        for limit in ("--solve-limit=1", "--solve-limit=3"):
            with self.subTest(limit=limit):
                solver, model = _solve(SEARCH, [limit])
                self.assertFalse(solver.search_exhausted)
                self.assertIsNotNone(model, "expected a model despite the limit")
                self.assertFalse(model.computation_finished,
                                 "a run cut short must not claim to have finished")

    def test_a_stopped_run_can_report_the_optimal_cost_without_claiming_to_be_finished(self):
        """The distinction the field exists for: same numbers, one proven and one not."""
        _, proved = _solve(SEARCH)
        solver, stopped = _solve(SEARCH, ["--solve-limit=3"])
        self.assertTrue(proved.computation_finished)
        self.assertFalse(stopped.computation_finished)

    def test_no_model_at_all_returns_none_rather_than_a_false_claim(self):
        solver, model = _solve(TRIVIAL, ["--solve-limit=0"])
        self.assertIsNone(model)
        self.assertFalse(solver.search_exhausted)

    def test_optimality_proven_is_not_used_as_the_signal(self):
        """It is False on every model under --opt-mode=opt, including this proved-optimal one."""
        solver, model = _solve(SEARCH)
        self.assertTrue(model.computation_finished)
        self.assertNotIn("model.optimality_proven",
                         (REPO / "02_ASP" / "main.py").read_text())

    def test_main_no_longer_forces_the_field_true(self):
        source = (REPO / "02_ASP" / "main.py").read_text()
        self.assertNotIn("model.computation_finished = True", source,
                         "the unconditional assignment is back; it makes the field meaningless")


if __name__ == "__main__":
    unittest.main()
