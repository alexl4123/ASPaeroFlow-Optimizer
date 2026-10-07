"""The what-if of one step as the explanation panel shows it: the menu of requirements, the baseline (only the step's
sub-problem, compared with its recorded answer), the same comparison as a card, and the texts.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

Runs on the shared 10-flight run (xai_tests/shared_run.py).
"""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai.contrastive import IterationExplainer, Lock  # noqa: E402
from src.aspaeroflow.xai.subproblem import Solution, compare  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402

SOLVER_SYNTAX = ("keep_", "no_delay", "avoid ", "keep ")


def chosen_meets(ex, lock):
    """Direct check: no path ban of the lock hits a recorded path, and the recorded configuration is not banned."""
    paths, configs = ex._lock_bans(lock)
    return all((f, p) not in paths for f, p in ex.chosen_paths.items()) and ex.chosen_config not in configs


class WhatIfPanel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, _, cls.trace_dir = shared_run.get()
        cls.trace = TraceReader(cls.trace_dir)
        cls.explainers = {n: IterationExplainer(cls.trace, n) for n in cls.trace.accepted()}

    def test_menu_entries_and_states(self):
        """U1: one keep and one no_delay entry per solver flight with a recorded trajectory, then keep_sectors; the
        state equals a direct check of the recorded answer and of the candidate set."""
        for n, ex in self.explainers.items():
            menu = ex.what_if_menu()["menu"]
            expected = []
            for f in ex.sub.decision_flights:
                if f in ex.current:
                    expected += [f"keep {f}", f"no_delay {f}"]
            self.assertEqual([e["lock"] for e in menu], expected + ["keep_sectors"], f"step {n}")
            for e in menu:
                lock = Lock.parse(e["lock"])
                paths, configs = ex._lock_bans(lock)
                if lock.kind == "keep_sectors":
                    offered = 0 in ex.sub.configs
                else:
                    offered = any((lock.args[0], p) not in paths for p in ex.sub.paths[lock.args[0]])
                state = "not_offered" if not offered else ("met" if chosen_meets(ex, lock) else "open")
                self.assertEqual(e["state"], state, f"step {n} {e['lock']}")
                self.assertTrue(e["text"][0].islower() or e["text"].startswith("the "), e["text"])
                for word in SOLVER_SYNTAX:
                    self.assertNotIn(word, e["text"])

    def test_met_states(self):
        """U2: keep_sectors met iff the recorded configuration is 0; keep F met iff the recorded path keeps F's
        trajectory; a flight without a recorded trajectory gets no entries."""
        for n, ex in self.explainers.items():
            menu = {e["lock"]: e for e in ex.what_if_menu()["menu"]}
            self.assertEqual(menu["keep_sectors"]["state"] == "met", ex.chosen_config == 0, f"step {n}")
            for f in ex.sub.decision_flights:
                if f in ex.current:
                    self.assertEqual(menu[f"keep {f}"]["state"] == "met", ex.chosen_current(f), f"step {n} flight {f}")
        n, ex = next(iter(self.explainers.items()))
        f = ex.sub.decision_flights[0]
        bare = IterationExplainer(self.trace, n)
        del bare.current[f]
        self.assertNotIn(f, [e["flight"] for e in bare.what_if_menu()["menu"]])

    def test_all_met_solves_nothing(self):
        """U3: locks the recorded answer already meets are not solved; a mixed what-if solves all locks together."""
        for n in self.explainers:
            ex = IterationExplainer(self.trace, n)
            met = [Lock.parse(e["lock"]) for e in ex.what_if_menu()["menu"] if e["state"] == "met"]
            opened = [Lock.parse(e["lock"]) for e in ex.what_if_menu()["menu"] if e["state"] == "open"]
            if not met:
                continue
            calls = []
            solve = ex.solver.solve
            ex.solver.solve = lambda *a, **k: calls.append(k) or solve(*a, **k)
            result = ex.what_if(met)
            self.assertEqual(calls, [], f"step {n}")
            self.assertTrue(result["already_met"])
            self.assertTrue(result["feasible"])
            self.assertNotIn("answers", result)
            self.assertIn("already meets", result["answer"])
            if opened:
                result = ex.what_if(met + opened[:1])
                self.assertFalse(result["already_met"])
                self.assertEqual(calls[0]["groups"], [str(x) for x in met + opened[:1]], f"step {n}")
                self.assertEqual([r["met"] for r in result["requirements"]], [True] * len(met) + [False])
            return
        self.skipTest("no step of the run has a met requirement")

    def test_lock_outside_the_step(self):
        """U4: a lock on a flight that is not part of the step raises ValueError (the service answers 400)."""
        for n, ex in self.explainers.items():
            outside = max(ex.sub.paths) + 1000
            for text in (f"keep {outside}", f"no_delay {outside}", f"path {outside} 0", f"avoid 1 {outside}"):
                with self.assertRaises(ValueError, msg=f"step {n} {text}"):
                    ex.what_if([Lock.parse(text)])

    def test_single_open_locks(self):
        """U5: every open single lock: the ladder compares the recorded answer with the what-if answer; every table
        row's flight differs between the two answers; every same clause's flight differs from before the step; the
        question has no solver syntax and names the step."""
        seen = 0
        for n, ex in self.explainers.items():
            for e in ex.what_if_menu()["menu"]:
                if e["state"] != "open":
                    continue
                result = ex.what_if([Lock.parse(e["lock"])])
                self.assertTrue(result["question"].startswith(f"What if, in step {n}, "), result["question"])
                for word in SOLVER_SYNTAX:
                    self.assertNotIn(word, result["question"])
                if not result["feasible"]:
                    continue
                seen += 1
                sol = Solution(True, True, result["costs"], [], result["chosen_config"],
                               {int(f): p for f, p in result["chosen_paths"].items()}, [], {}, 0.0)
                self.assertEqual(result["ladder"], ex._ladder(ex.factual, sol))
                recorded = {x["flight"]: x for x in result["answers"]["recorded"]["flights"]}
                other = {x["flight"]: x for x in result["answers"]["what_if"]["flights"]}
                for row in result["table"]["rows"]:
                    self.assertNotEqual(recorded[row["flight"]]["trajectory"], other[row["flight"]]["trajectory"])
                differing = {f for f in recorded if recorded[f]["trajectory"] != other[f]["trajectory"]}
                self.assertEqual({r["flight"] for r in result["table"]["rows"]}, differing)
                for clause in result["table"]["same"]:
                    if clause.startswith("flight "):
                        f = int(clause.split()[1])
                        self.assertTrue(recorded[f]["changed"] or other[f]["changed"], clause)
                self.assertEqual(result["table"]["config"] is None,
                                 result["answers"]["recorded"]["config"] == result["answers"]["what_if"]["config"])
                if "accepts" in result["verdict"]:
                    self.assertIn(f"Step {n}'s choice accepts", result["verdict"])
                self.assertIn("The answer with your requirement", result["verdict"])
        self.assertGreater(seen, 0)

    def test_on_time_label(self):
        """U6: a flight is "on time, as in the filed plan" iff no earlier kept step changed it; a copy of the run with
        the flight added to an earlier record says "no later than before the step"."""
        later = [n for n in self.explainers if any(m < n for m in self.explainers)]
        self.assertTrue(later)
        for n in later:
            ex = self.explainers[n]
            for f in ex.sub.decision_flights:
                earlier = any(str(f) in (self.trace.iterations[m].get("flight_changes") or {})
                              for m in self.trace.accepted() if m < n)
                text = ex.lock_text(Lock("no_delay", (f,)))
                self.assertEqual(text.endswith("departs on time, as in the filed plan"), not earlier, text)
                self.assertEqual(text.endswith("departs no later than before the step"), earlier, text)
        n = later[0]
        f = self.explainers[n].sub.decision_flights[0]
        first = min(self.explainers)
        tmp = Path(tempfile.mkdtemp(prefix="aspaeroflow-xai-ontime-"))
        try:
            copy_dir = tmp / "trace"
            shutil.copytree(self.trace_dir, copy_dir)
            lines = (copy_dir / "trace.jsonl").read_text().splitlines()
            records = [json.loads(line) for line in lines]
            for r in records:
                if r["iteration"] == first:
                    r.setdefault("flight_changes", {})[str(f)] = {"id": f}
            (copy_dir / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            forged = IterationExplainer(TraceReader(copy_dir), n)
            self.assertEqual(forged.lock_text(Lock("no_delay", (f,))),
                             f"flight {f} departs no later than before the step")
            del records[[r["iteration"] for r in records].index(n)]["flight_changes"]
            (copy_dir / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            old = IterationExplainer(TraceReader(copy_dir), n)
            self.assertEqual(old.lock_text(Lock("no_delay", (f,))), f"flight {f} departs no later than before the step")
        finally:
            shutil.rmtree(tmp, True)

    def test_same_comparison_as_a_card(self):
        """U7: on every changed solver flight, the what-if "keep F" is the card "Keeping flight F as it was", with the
        card's numbers (keep contrasts); "no_delay F" likewise with "Not delaying flight F" where that card exists; the
        sectors lock with the sectors card."""
        seen = set()
        for n in self.explainers:
            ex = IterationExplainer(self.trace, n)
            keep = ex.keep_contrasts()
            for f in ex.sub.decision_flights:
                cards = {c["label"]: c for c in ex.why_flight(f)["contrasts"]}
                for lock, label in ((f"keep {f}", f"keeping flight {f} as it was"),
                                    (f"no_delay {f}", f"not delaying flight {f}")):
                    if label not in cards:
                        continue
                    result = ex.what_if([Lock.parse(lock)])
                    self.assertEqual(result["same_as"], label[0].upper() + label[1:], f"step {n} {lock}")
                    self.assertEqual(result["costs"], cards[label]["costs"], f"step {n} {lock}")
                    self.assertEqual(result["ladder"], cards[label]["ladder"], f"step {n} {lock}")
                    if lock.startswith("keep"):
                        self.assertEqual(result["costs"], keep["flights"][str(f)]["keep"]["costs"])
                    seen.add(lock.split()[0])
            sectors = {c["label"]: c for c in ex.why_sectors()["contrasts"]}
            if "keeping the hotspot's sectors as they are" in sectors:
                result = ex.what_if([Lock("keep_sectors")])
                self.assertEqual(result["same_as"], "Keeping the hotspot's sectors as they are")
                self.assertEqual(result["costs"], sectors["keeping the hotspot's sectors as they are"]["costs"])
            for lock in ("keep_sectors",):
                if ex.chosen_config == 0:
                    self.assertIsNone(ex.what_if([Lock.parse(lock)]).get("same_as"))
        self.assertIn("keep", seen)

    def test_verdict_branches_and_impossible_texts(self):
        """U8: a what-if answer better than the recorded one uses the "would be better on" branch; impossible texts
        for a core of one and of two requirements."""
        n, ex = next(iter(self.explainers.items()))
        better = copy.deepcopy(ex.factual)
        better.costs = dict(better.costs, overload=ex.factual.costs["overload"] - 1)
        deciding, text = ex._contrast_text("the answer with your requirement", better, chosen_name=f"step {n}'s choice")
        self.assertEqual(deciding, "overload")
        self.assertTrue(text.startswith("The answer with your requirement would be better on total overload"), text)
        self.assertIn("the recorded choice is not optimal for this step", text)
        tie = ex._contrast_text("the answer with your requirement", ex.factual, chosen_name=f"step {n}'s choice")[1]
        self.assertEqual(tie, "The answer with your requirement is exactly as good on every criterion. The step's "
                              "choice between the two is a tie broken by the solver, not a reason.")
        f = ex.sub.decision_flights[0]
        paths = sorted(ex.sub.paths[f])
        first, other = paths[0], [p for p in paths if p != paths[0]][0]
        result = ex.what_if([Lock("path", (f, first)), Lock("path", (f, other))])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["answer"],
                         f"No combination of step {n}'s candidate routes, delays and sector options meets these "
                         f"requirements together: flight {f} takes candidate path {first} and flight {f} takes "
                         f"candidate path {other}.")
        missing = max(paths) + 1
        result = ex.what_if([Lock("path", (f, missing))])
        self.assertEqual(result["answer"], f"No combination of step {n}'s candidate routes, delays and sector "
                                           f"options meets the requirement: flight {f} takes candidate path {missing}.")

    def test_cards_and_what_if_share_one_answer(self):
        """A card and a what-if with the same bans are solved once per explainer."""
        for n in self.explainers:
            ex = IterationExplainer(self.trace, n)
            for f in ex.sub.decision_flights:
                if any(c["label"] == f"not delaying flight {f}" for c in ex.why_flight(f)["contrasts"]):
                    before = len(ex._memo)
                    ex.what_if([Lock("no_delay", (f,))])
                    self.assertEqual(len(ex._memo), before)
                    return
        self.skipTest("no not-delaying card in the run")


if __name__ == "__main__":
    unittest.main()
