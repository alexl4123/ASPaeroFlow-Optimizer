"""Study items with keys from a recorded run (xai/study_items.py, K32b) on the June CE-7x7 trace.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai.study_items import (  # noqa: E402
    Run, build_items, claim_items, counterfactual_items, forward_items, reason_items, review_table)

TRACE = HERE / "fixtures" / "CE7_JUNE_TRACE"


class StudyItems(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trace = Run(TRACE)
        cls.result = build_items(TRACE, "A")

    def test_forward_keys_need_no_tie_break(self):
        items = forward_items(self.trace)
        self.assertTrue(items)
        by_step = {r["iteration"]: r for r in self.trace.records}
        for i in items:
            r = by_step[i["step"]]
            decision = self.trace.decision(i["step"])
            changed = [int(f) for f in r["flight_changes"] if int(f) in decision]
            self.assertEqual(changed, [int(i["key"][1:])])
            self.assertNotEqual(self.trace.rows(i["step"]).get(f"flight:{changed[0]}", {}).get("class"), "tie")
            self.assertEqual(i["cap"], i["step"] - 1)

    def test_counterfactual_keys_follow_the_keep_contrast(self):
        for i in counterfactual_items(self.trace):
            factual = self.trace.factual(i["step"])
            keep = self.trace.keep_of(i["step"], i["flight"])["keep"]["costs"]["overload"]
            self.assertEqual(i["key"], "higher" if keep > factual["overload"] else "same")
            self.assertNotEqual(i["key"], "lower")

    def test_claims_are_dominated_plans(self):
        for i in claim_items(self.trace):
            factual = self.trace.factual(i["step"])
            keep = self.trace.keep_of(i["step"], i["flight"])["keep"]["costs"]
            diffs = {k: keep[k] - factual[k] for k in ("overload", "delay", "sectors", "changed")}
            self.assertEqual(sum(1 for v in diffs.values() if v > 0), 1, diffs)
            self.assertTrue(all(v >= 0 for v in diffs.values()), diffs)
            self.assertEqual(i["key"], "false")

    def test_reason_keys_are_among_the_options(self):
        for i in reason_items(self.trace):
            self.assertIn(i["key"], [o["id"] for o in i["options"]])

    def test_selection_mix_order_and_determinism(self):
        items = self.result["items"]
        kinds = sorted(i["kind"] for i in items)
        self.assertEqual(kinds, sorted(["forward", "counterfactual", "claim", "reason", "reason"]), self.result["pool"])
        caps = [i["cap"] for i in items]
        self.assertEqual(caps, sorted(caps))
        flights = [i.get("flight") for i in items if i.get("flight") is not None]
        self.assertEqual(len(flights), len(set(flights)))
        self.assertEqual([i["id"] for i in build_items(TRACE, "A")["items"]], [i["id"] for i in items])
        self.assertEqual([(i["kind"], i["step"]) for i in build_items(TRACE, "A")["items"]],
                         [(i["kind"], i["step"]) for i in items])
        self.assertIn("(key)", review_table(self.result))


    def test_practice_mode_one_item_about_the_given_step(self):
        result = build_items(TRACE, "P", practice_step=6)
        self.assertLessEqual(len(result["items"]), 1)
        for i in result["items"]:
            self.assertEqual((i["id"], i["step"]), ("P1", 6))
        self.assertEqual(result["missing"], [] if result["items"] else ["practice"])


if __name__ == "__main__":
    unittest.main()
