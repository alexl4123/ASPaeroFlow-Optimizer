"""The clingo solver profiles are opt-in, and 'default' means "change nothing".

A benchmark campaign and the published LPNMR/ATMOS numbers were produced with the solver
configuration these two call sites hard-coded. Making that configuration selectable is only safe
if selecting nothing reproduces it exactly, so that is what these tests pin:

  * the default profile is the EMPTY flag list;
  * clingo.Control([]) is configured identically to clingo.Control() -- the 02_ASP call site;
  * a Solver built with no seed and no options passes --seed=11904657 and nothing else -- the
    01_ASPaeroFlow call site, which used to hard-code that seed;
  * the non-default profiles genuinely reach clasp, which is checked through clingo's own
    configuration rather than by inspecting our argument list.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import clingo  # noqa: E402

from common.clingo_options import (  # noqa: E402
    DEFAULT_SOLVER_PROFILE,
    SOLVER_PROFILES,
    build_options,
    normalise_profile,
    options_from_args,
    profile_options,
)

#: The flags the 01_ASPaeroFlow Solver hard-coded before it took a seed.
LEGACY_01_ARGS = ["--seed=11904657"]

#: Configuration keys that say what search clasp will actually run.
WATCHED = [("solver", "opt_strategy"), ("solver", "opt_usc_shrink"),
           ("solver", "heuristic"), ("solver", "seed"),
           ("solve", "parallel_mode"), ("solve", "opt_mode")]


def configuration(arguments):
    ctl = clingo.Control(list(arguments))
    return {f"{a}.{b}": getattr(getattr(ctl.configuration, a), b) for a, b in WATCHED}


class TestDefaultChangesNothing(unittest.TestCase):

    def test_default_profile_is_no_flags(self):
        self.assertEqual(profile_options(DEFAULT_SOLVER_PROFILE), [])
        self.assertEqual(build_options(), [])
        self.assertEqual(build_options(None, None), [])
        self.assertEqual(build_options(None, []), [])

    def test_empty_argument_list_matches_a_bare_control(self):
        """02_ASP built clingo.Control(); it now builds clingo.Control([])."""
        self.assertEqual(configuration([]), configuration(None or []))
        bare = clingo.Control()
        bare_cfg = {f"{a}.{b}": getattr(getattr(bare.configuration, a), b) for a, b in WATCHED}
        self.assertEqual(bare_cfg, configuration(build_options()))

    def test_legacy_01_call_site_is_reproduced(self):
        """01_ASPaeroFlow hard-coded --seed=11904657; the default must still be exactly that."""
        from importlib import util
        spec = util.spec_from_file_location(
            "aspaeroflow_solver", REPO / "01_ASPaeroFlow" / "src" / "aspaeroflow" / "solver.py")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solver = module.Solver("", "")
        self.assertEqual(solver.seed, 11904657)
        self.assertEqual(solver.solver_options, [])
        self.assertEqual(configuration([f"--seed={solver.seed}"] + solver.solver_options),
                         configuration(LEGACY_01_ARGS))

    def test_namespace_without_the_options_still_yields_nothing(self):
        class Bare:
            pass
        self.assertEqual(options_from_args(Bare()), [])


class TestProfilesReachClasp(unittest.TestCase):

    def test_usc_sets_the_optimisation_strategy(self):
        cfg = configuration(build_options("usc"))
        self.assertTrue(str(cfg["solver.opt_strategy"]).startswith("usc"),
                        f"opt_strategy is {cfg['solver.opt_strategy']}")
        self.assertTrue(str(cfg["solver.opt_usc_shrink"]).startswith("min"),
                        f"opt_usc_shrink is {cfg['solver.opt_usc_shrink']}")

    def test_domain_switches_the_decision_heuristic(self):
        """#heuristic directives in an encoding are inert without this."""
        self.assertTrue(str(configuration(build_options("domain"))["solver.heuristic"])
                        .startswith("domain"))
        self.assertFalse(str(configuration(build_options())["solver.heuristic"])
                         .startswith("domain"))

    def test_usc_domain_is_both(self):
        cfg = configuration(build_options("usc-domain"))
        self.assertTrue(str(cfg["solver.opt_strategy"]).startswith("usc"))
        self.assertTrue(str(cfg["solver.heuristic"]).startswith("domain"))

    def test_raw_escape_hatch_is_appended_and_wins(self):
        cfg = configuration(build_options("usc-domain", ["--parallel-mode=4"]))
        self.assertTrue(str(cfg["solve.parallel_mode"]).startswith("4"))
        self.assertTrue(str(cfg["solver.opt_strategy"]).startswith("usc"))

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            normalise_profile("no-such-profile")

    def test_every_profile_is_accepted_by_clingo(self):
        for name in SOLVER_PROFILES:
            with self.subTest(profile=name):
                clingo.Control(build_options(name))


class TestBothMainsExposeTheOption(unittest.TestCase):
    """A benchmark configuration must be able to name the strategy on the command line."""

    MAINS = ["01_ASPaeroFlow/main.py", "02_ASP/main.py"]

    def test_help_advertises_solver_profile(self):
        import subprocess
        for rel in self.MAINS:
            with self.subTest(main=rel):
                out = subprocess.run([sys.executable, str(REPO / rel), "--help"],
                                     capture_output=True, text=True, cwd=REPO / Path(rel).parent)
                self.assertIn("--solver-profile", out.stdout, f"{rel} does not expose it")
                self.assertIn("--solver-arg", out.stdout, f"{rel} does not expose it")


if __name__ == "__main__":
    unittest.main()
