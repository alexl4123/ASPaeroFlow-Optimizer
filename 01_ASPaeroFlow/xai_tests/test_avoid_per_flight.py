"""Per-flight "avoid navpoint" of the what-if (S8 4a, D1: only flight M): `avoid N M` bans exactly the candidates of
M's own trajectory that pass N, for a solver flight and for a later flight alike; its text names M; a solved answer
never routes M through N; the step-wide `avoid N` is unchanged; an impossible per-flight requirement comes back as the
minimal core with readable texts; the menu lists per flight the navpoints of its candidate routes.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

Runs on steps 2 and 3 of the practice study trace (fixtures/NAVPT_PRACTICE_TRACE, solvable, with later flights) and on
the shared 10-flight run (xai_tests/shared_run.py, no later flights).
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai.contrastive import IterationExplainer, Lock  # noqa: E402
from src.aspaeroflow.xai.session import display_names  # noqa: E402
from src.aspaeroflow.xai.subproblem import SubproblemSolver  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402

PRACTICE = HERE / "fixtures" / "NAVPT_PRACTICE_TRACE"
SOLVER_SYNTAX = ("keep_", "no_delay", "avoid ", "keep ", "same_route", "max_delay")


def members(ex):
    """(flight, its solver flight) for every flight of the step; a solver flight is its own solver flight."""
    out = []
    for f in ex.sub.decision_flights:
        out.append((f, f))
        out += [(m, f) for m, par in sorted(ex.sub.parent.items()) if par == f]
    return out


def own_route_bans(ex, nav, m, f):
    """Brute force: the path indices of the solver flight f at which flight m's own candidate passes nav."""
    return {(f, p) for p in sorted(ex.sub.paths[f])
            if p in ex.sub.paths.get(m, {}) and nav in ex.sub.paths[m][p].route}


def step_wide_bans(ex, nav):
    """Brute force of the step-wide lock: a solver flight's index is banned when its own candidate or the candidate of
    one of its later flights at that index passes nav."""
    return {(f, p) for m, f in members(ex) for p in sorted(ex.sub.paths[f])
            if p in ex.sub.paths.get(m, {}) and nav in ex.sub.paths[m][p].route}


def navpoints_of(ex):
    out = set()
    for m, _ in members(ex):
        for cp in ex.sub.paths.get(m, {}).values():
            out.update(cp.route)
    return out


class AvoidPerFlight(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.names = display_names(PRACTICE / "CE-NAVPT-PRACTICE")
        cls.practice = TraceReader(PRACTICE)
        cls.shared = TraceReader(shared_run.get()[2])

    def explainers(self):
        """Fresh explainers of the practice steps 2 and 3 (with fix names) and of every kept step of the shared run."""
        out = []
        for n in (2, 3):
            ex = IterationExplainer(self.practice, n)
            ex.names = self.names
            out.append((f"practice {n}", ex))
        for n in self.shared.accepted():
            out.append((f"shared {n}", IterationExplainer(self.shared, n)))
        return out

    def test_fixture_has_later_flights(self):
        ex = IterationExplainer(self.practice, 2)
        self.assertEqual(ex.sub.parent, {10: 3, 8: 4})
        self.assertEqual(IterationExplainer(self.practice, 3).sub.parent, {14: 11})

    def test_bans_are_the_flights_own_route(self):
        """For every flight M of the step and every navpoint N of the step (and one outside it): the bans of `avoid N
        M` are exactly the indices at which M's own candidate passes N; no configuration is banned."""
        for label, ex in self.explainers():
            navs = sorted(navpoints_of(ex)) + [max(navpoints_of(ex)) + 1000]
            for m, f in members(ex):
                for nav in navs:
                    paths, configs = ex._lock_bans(Lock("avoid", (nav, m)))
                    self.assertEqual(paths, own_route_bans(ex, nav, m, f), f"{label} avoid {nav} {m}")
                    self.assertEqual(configs, set())

    def test_later_flight_bans_ignore_the_solver_flight(self):
        """A later flight's lock never bans because of its solver flight's route, and a solver flight's lock never
        because of its later flight's: on practice step 2, EHAM (departure of 3) bans every path of 3 for `avoid N 3`
        and nothing for `avoid N 10`; ABAMI (on every route of 8) bans every path of 4 step-wide and for 8, but only
        4's own routes through it for `avoid N 4`."""
        ex = IterationExplainer(self.practice, 2)
        ex.names = self.names
        ids = {name: v for v, name in self.names.items()}
        eham, abami = ids["EHAM"], ids["ABAMI"]
        all3 = {(3, p) for p in ex.sub.paths[3]}
        all4 = {(4, p) for p in ex.sub.paths[4]}
        self.assertEqual(ex._lock_bans(Lock("avoid", (eham, 3)))[0], all3)
        self.assertEqual(ex._lock_bans(Lock("avoid", (eham, 10)))[0], set())
        self.assertEqual(ex._lock_bans(Lock("avoid", (abami, 8)))[0], all4)
        self.assertEqual(ex._lock_bans(Lock("avoid", (abami,)))[0] & all4, all4)
        own4 = ex._lock_bans(Lock("avoid", (abami, 4)))[0]
        self.assertTrue(own4 and own4 < all4)
        self.assertEqual(own4, {(4, p) for p in ex.sub.paths[4] if abami in ex.sub.paths[4][p].route})

    def test_bans_equal_the_constraint_on_the_flight(self):
        """The ASP check of the semantics: for every flight M and every navpoint on M's candidate routes, the bans of
        `avoid N M` solve to the same feasibility and costs as the literal constraints ":- chosen_path(M,p)" for M's
        candidates p that pass N (a later flight M has no choice of its own; its chosen_path follows the solver
        flight's)."""
        checked = 0
        for n in (2, 3):
            ex = IterationExplainer(self.practice, n)
            literal = SubproblemSolver(self.practice.encoding, ex.sub, seed=int(self.practice.run.get("seed", 11904657)))
            for m, _ in members(ex):
                for nav in sorted({v for cp in ex.sub.paths[m].values() for v in cp.route}):
                    paths, _ = ex._lock_bans(Lock("avoid", (nav, m)))
                    a, _ = ex.solver.solve(ban_paths=paths)
                    bodies = [f"chosen_path({m},{p})" for p, cp in sorted(ex.sub.paths[m].items()) if nav in cp.route]
                    name = f"literal {nav} {m}"
                    literal.add_group(name, bodies=bodies)
                    b, _ = literal.solve(groups=[name])
                    self.assertEqual(a.satisfiable, b.satisfiable, f"step {n} avoid {nav} {m}")
                    if a.satisfiable:
                        self.assertEqual(a.costs, b.costs, f"step {n} avoid {nav} {m}")
                        index = a.chosen_paths[ex.sub.decision_flight_of(m)]
                        self.assertNotIn(nav, ex.sub.paths[m][index].route, f"step {n} avoid {nav} {m}")
                    checked += 1
        self.assertGreater(checked, 30)

    def test_text_names_the_flight(self):
        """The requirement and the question name M (a later flight by its own number, not its solver flight's) and the
        navpoint by its fix name; no solver syntax."""
        ex = IterationExplainer(self.practice, 2)
        ex.names = self.names
        ids = {name: v for v, name in self.names.items()}
        self.assertEqual(ex.lock_text(Lock.parse(f"avoid {ids['AALEN']} 10")), "flight 10 does not pass navpoint AALEN")
        self.assertEqual(ex.lock_text(Lock.parse(f"avoid {ids['ADUSU']} 3")), "flight 3 does not pass navpoint ADUSU")
        self.assertEqual(ex.lock_text(Lock.parse(f"avoid {ids['ADUSU']}")),
                         "no flight of the step passes navpoint ADUSU")
        result = ex.what_if([Lock.parse(f"avoid {ids['AALEN']} 10")])
        self.assertEqual(result["question"], "What if, in step 2, flight 10 does not pass navpoint AALEN?")
        self.assertEqual(result["requirements"][0]["text"], "flight 10 does not pass navpoint AALEN")
        # a navpoint on no candidate route of flight 10: already met, said for flight 10
        result = ex.what_if([Lock.parse(f"avoid {ids['EHAM']} 10")])
        self.assertTrue(result["already_met"])
        self.assertEqual(result["answer"], "No candidate route of flight 10 in step 2 passes navpoint EHAM; step 2's "
                                           "choice already meets this requirement.")
        for word in SOLVER_SYNTAX:
            self.assertNotIn(word, result["answer"])

    def test_solved_answer_never_routes_the_flight_through_the_navpoint(self):
        """Every open per-flight navpoint of the menu: the decoded what-if answer gives M a route and a trajectory
        without N; a later flight's met lock solved together with an open one: likewise for the later flight."""
        solved = 0
        for label, ex in self.explainers():
            groups = ex.what_if_menu()["flight_navpoints"]
            for g in groups:
                for e in g["navpoints"]:
                    if e["state"] != "open":
                        continue
                    lock = Lock.parse(e["lock"])
                    nav, m = lock.args
                    result = ex.what_if([lock])
                    self.assertTrue(result["feasible"], f"{label} {lock}")
                    flights = {x["flight"]: x for x in result["answers"]["what_if"]["flights"]}
                    self.assertNotIn(nav, flights[m]["route"], f"{label} {lock}")
                    self.assertNotIn(nav, flights[m]["trajectory"].values(), f"{label} {lock}")
                    recorded = {x["flight"]: x for x in result["answers"]["recorded"]["flights"]}
                    self.assertIn(nav, recorded[m]["route"], f"{label} {lock}")       # open: the recorded one passes it
                    solved += 1
        self.assertGreater(solved, 3)
        # practice step 2: flight 10 (after 3) has one route; a navpoint not on it, with the open "flight 3 avoids ADUSU"
        ex = IterationExplainer(self.practice, 2)
        ex.names = self.names
        ids = {name: v for v, name in self.names.items()}
        result = ex.what_if([Lock("avoid", (ids["ADUSU"], 3)), Lock("avoid", (ids["EHAM"], 10))])
        self.assertTrue(result["feasible"])
        self.assertEqual([r["met"] for r in result["requirements"]], [False, True])
        flights = {x["flight"]: x for x in result["answers"]["what_if"]["flights"]}
        self.assertNotIn(ids["EHAM"], flights[10]["route"])
        self.assertNotIn(ids["ADUSU"], flights[3]["route"])

    def test_step_wide_avoid_unchanged(self):
        """`avoid N` bans a solver flight's index when its own or a later flight's candidate there passes N; the
        navpoint list keeps its entries (lock `avoid N`, same states)."""
        for label, ex in self.explainers():
            navs = navpoints_of(ex)
            for nav in sorted(navs):
                self.assertEqual(ex._lock_bans(Lock("avoid", (nav,)))[0], step_wide_bans(ex, nav), f"{label} {nav}")
            entries = ex.what_if_menu()["navpoints"]
            self.assertEqual({e["navpoint"] for e in entries}, navs)
            for e in entries:
                self.assertEqual(e["lock"], f"avoid {e['navpoint']}")
                bans = step_wide_bans(ex, e["navpoint"])
                blocked = any(all((f, p) in bans for p in ex.sub.paths[f]) for f in ex.sub.decision_flights)
                met = not any((f, p) in bans for f, p in ex.chosen_paths.items())
                self.assertEqual(e["state"], "not_offered" if blocked else ("met" if met else "open"), f"{label} {e}")

    def test_impossible_per_flight_requirement(self):
        """A per-flight avoid no candidate meets is not offered, with the airport reason, and its what-if is impossible
        with a core of that one lock; with a path lock through the navpoint and a third lock that fits both, the core
        is the minimal pair, in readable words."""
        ex = IterationExplainer(self.practice, 2)
        ex.names = self.names
        ids = {name: v for v, name in self.names.items()}
        groups = {g["flight"]: {e["name"]: e for e in g["navpoints"]} for g in ex.what_if_menu()["flight_navpoints"]}
        entry = groups[3]["EHAM"]
        self.assertEqual((entry["state"], entry["reason"]), ("not_offered", "EHAM is the departure airport of flight 3"))
        result = ex.what_if([Lock.parse(entry["lock"])])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["outcome"], "impossible")
        self.assertEqual(result["core"], [entry["lock"]])
        self.assertEqual(result["core_texts"], ["flight 3 does not pass navpoint EHAM"])
        self.assertEqual(result["answer"], "No combination of step 2's candidate routes, delays and sector options "
                                           "meets the requirement: flight 3 does not pass navpoint EHAM.")
        self.assertEqual(groups[10]["AALEN"]["reason"], "every candidate route of flight 10 passes AALEN")
        self.assertEqual(groups[10]["LSZH"]["reason"], "LSZH is the arrival airport of flight 10")
        # a pair that cannot hold together, plus the sectors lock, which fits each of them
        adusu = ids["ADUSU"]
        through = min(p for p, cp in ex.sub.paths[4].items() if adusu in cp.route)
        avoid, path = Lock("avoid", (adusu, 4)), Lock("path", (4, through))
        for alone in ([avoid, Lock("keep_sectors")], [path, Lock("keep_sectors")]):
            self.assertTrue(IterationExplainer(self.practice, 2).what_if(alone)["feasible"], alone)
        result = ex.what_if([avoid, path, Lock("keep_sectors")])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["core"], [str(avoid), str(path)])
        self.assertEqual(result["core_texts"], ["flight 4 does not pass navpoint ADUSU",
                                                f"flight 4 takes candidate path {through}"])
        self.assertEqual(result["answer"], "No combination of step 2's candidate routes, delays and sector options "
                                           "meets these requirements together: flight 4 does not pass navpoint ADUSU "
                                           f"and flight 4 takes candidate path {through}.")

    def test_menu_flight_navpoints(self):
        """`flight_navpoints`: one group per flight of the step (solver flight, then its later flights), the navpoints of
        that flight's own candidate routes, lock `avoid N M`, state from the bans (not offered: every index banned; met:
        the recorded index not banned), a reason exactly for not offered ones; without per_flight the menu is the same
        without the field."""
        reasons = set()
        for label, ex in self.explainers():
            menu = ex.what_if_menu()
            plain = ex.what_if_menu(per_flight=False)
            self.assertNotIn("flight_navpoints", plain)
            self.assertEqual({k: v for k, v in menu.items() if k != "flight_navpoints"}, plain)
            groups = menu["flight_navpoints"]
            self.assertEqual([(g["flight"], g["parent"]) for g in groups],
                             [(m, None if m == f else f) for m, f in members(ex)])
            for g in groups:
                m = g["flight"]
                f = g["parent"] if g["parent"] is not None else m
                own = {v for cp in ex.sub.paths.get(m, {}).values() for v in cp.route}
                self.assertEqual({e["navpoint"] for e in g["navpoints"]}, own, f"{label} flight {m}")
                for e in g["navpoints"]:
                    self.assertEqual(e["lock"], f"avoid {e['navpoint']} {m}")
                    bans = own_route_bans(ex, e["navpoint"], m, f)
                    if all((f, p) in bans for p in ex.sub.paths[f]):
                        state = "not_offered"
                    else:
                        state = "open" if (f, ex.chosen_paths[f]) in bans else "met"
                    self.assertEqual(e["state"], state, f"{label} {e['lock']}")
                    self.assertEqual(e["reason"] is not None, state == "not_offered")
                    if e["reason"]:
                        forms = {"departure": f"{e['name']} is the departure airport of flight {m}",
                                 "arrival": f"{e['name']} is the arrival airport of flight {m}",
                                 "route": f"every candidate route of flight {m} passes {e['name']}"}
                        kind = [k for k, text in forms.items() if text == e["reason"]]
                        self.assertEqual(len(kind), 1, e["reason"])
                        reasons.add(kind[0])
                if m != f:       # the later flights of these runs have one route: each of its navpoints is unavoidable
                    self.assertTrue(all(e["state"] == "not_offered" for e in g["navpoints"]), f"{label} flight {m}")
        self.assertEqual(reasons, {"departure", "arrival", "route"})

    def test_lock_outside_the_step_or_malformed(self):
        """A flight that is not part of the step, or a lock with too many arguments, raises ValueError (400)."""
        ex = IterationExplainer(self.practice, 2)
        for text in ("avoid 9 999", "avoid 9 11", "avoid 9 3 1"):
            with self.assertRaises(ValueError, msg=text):
                ex.what_if([Lock.parse(text)])
            with self.assertRaises(ValueError, msg=text):
                ex._lock_bans(Lock.parse(text))


if __name__ == "__main__":
    unittest.main()
