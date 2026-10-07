"""The step header texts of xai/reasons.py and reasons.lp on the June study run (CE-7x7, 13 kept steps).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

Fixture fixtures/CE7_JUNE_TRACE (reduced June trace, see its README). Records marked "synthetic" are the
June records with fields changed by the test; the comment next to each says which.
"""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai import reasons  # noqa: E402
from src.aspaeroflow.xai.session import OptimizerSession, ReplaySession  # noqa: E402

FIX = HERE / "fixtures" / "CE7_JUNE_TRACE"
TEN = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"
XAI_DIR = OPT / "src" / "aspaeroflow" / "xai"
CLINGO_58 = Path.home() / "anaconda3" / "envs" / "aspaeroflow-xai" / "bin" / "clingo"


def june():
    run = json.loads((FIX / "run.json").read_text())
    records = {}
    for line in (FIX / "trace.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["iteration"]] = r
    return run, records


def explain(records, n, run, folder=FIX, **kw):
    previous, rejected = reasons.step_context(records, n)
    return reasons.explain_step(records[n], previous=previous, run=run, trace_folder=folder,
                                rejected_before=rejected, **kw)


def text_of(out, kind_prefix):
    found = [line["text"] for line in out["lines"] if line["kind"].startswith(kind_prefix)]
    return found[0] if found else None


def all_text(out):
    return " ".join(line["text"] for line in out["lines"])


def codes(out):
    return [e["code"] for e in out["errors"]]


def with_occupants(record, durations, taken=2):
    """synthetic: hotspot.flights {flight: stored duration} sorted as the code sorts, and hotspot.taken"""
    r = copy.deepcopy(record)
    r["hotspot"]["flights"] = sorted(({"id": f, "duration": d} for f, d in durations.items()),
                                     key=lambda x: (x["duration"], x["id"]))
    r["hotspot"]["taken"] = taken
    return r


def rejected_copy(record, iteration, objectives_of, params=None):
    """synthetic: `record` as a rejected attempt with number `iteration` and the objectives of the restored plan"""
    r = copy.deepcopy(record)
    r["iteration"] = iteration
    r["accepted"] = False
    r["objectives"] = dict(objectives_of["objectives"], ITERATION=iteration,
                           **{"OVERLOAD-BEFORE": objectives_of["objectives"]["OVERLOAD"]})
    r["flight_changes"] = {}
    r["sector_changes"] = {}
    if params:
        r["parameters"] = dict(r["parameters"], **params)
    return r


def renumbered(record, iteration, params=None):
    r = copy.deepcopy(record)
    r["iteration"] = iteration
    r["objectives"] = dict(r["objectives"], ITERATION=iteration)
    if params:
        r["parameters"] = dict(r["parameters"], **params)
    return r


class Base(unittest.TestCase):

    def setUp(self):
        self.run_info, self.records = june()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def folder_with_lp(self, n, text):
        """synthetic instance: a trace folder whose lp file of step n is `text`"""
        folder = self.tmp / f"lp{n}_{abs(hash(text))}"
        (folder / "lp").mkdir(parents=True, exist_ok=True)
        (folder / "lp" / f"iter_{n:05d}_0.lp").write_text(text)
        return folder


class Step6June(Base):
    """1. Step 6 on the June data (no hotspot.flights): the fallback texts."""

    def test_texts_and_deltas(self):
        out = explain(self.records, 6, self.run_info, run_length=13, run_kept=13)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["lines"][0]["level"], "visible")
        self.assertEqual(out["lines"][0]["text"], "Sector 21 at time 11 had 3 flights for a capacity of 1: the earliest "
                                                  "overload, lowest sector number first.")
        details = all_text(out)
        for part in ("18 and 19 went to the solver (rule: at most 2", "26 (after 19), 47 and 58 (after 18)",
                     "0 to 19 time periods", "among 7 layouts of sector 21: the current one and 6 splits",
                     "New trajectory in this step: 18, 19 and 26.", "The solver left 47 and 58 unchanged.",
                     "split into 2 sectors (21 and 35) from time 11 on"):
            self.assertIn(part, details)
        got = [(d["key"], d["before"], d["after"], d["direction"]) for d in out["deltas"]]
        self.assertEqual(got, [("OVERLOAD", 28, 22, "lower"), ("ARRIVAL-DELAY", 241, 268, "higher"),
                               ("SECTOR-NUMBER", 630, 644, "higher"), ("REROUTE", 22, 25, "higher"),
                               ("SECTOR-DIFF", 19, 25, "higher"), ("RECONFIG", 330, 414, "higher")])
        self.assertEqual(out["run_length"], 13)
        self.assertEqual(out["arrival_delay_metric"], "signed")
        # never says that 47 or 58 got a new trajectory
        for line in out["lines"]:
            if line["kind"].startswith("changed"):
                self.assertNotIn("47", line["text"])
                self.assertNotIn("58", line["text"])
        refs = [r["ref"] for line in out["lines"] for r in line["refs"]]
        self.assertIn("sector:35@6:after", refs)
        legs = next(line for line in out["lines"] if line["kind"] == "legs")
        self.assertEqual([r["ref"] for r in legs["refs"]],
                         ["flight:26", "flight:19", "flight:47", "flight:18", "flight:58"])


class Occupants(Base):
    """2. Synthetic hotspot.flights (hotspots and limits from the June records, max_aircraft 2)."""

    def subproblem(self, n, durations, taken=2):
        self.records[n] = with_occupants(self.records[n], durations, taken)
        out = explain(self.records, n, self.run_info)
        return out, text_of(out, "subproblem")

    def test_tie_step3(self):
        out, line = self.subproblem(3, {6: 5, 8: 5, 11: 6, 12: 5})
        self.assertEqual(out["errors"], [])
        self.assertEqual(line, "Of the 4 flights in sector 6 at time 9, 2 went to the solver: 6 and 8; 12 has the same "
                               "stored duration and was left out by its higher flight number; 11 has a longer one.")
        refs = [r["ref"] for r in next(x for x in out["lines"] if x["kind"].startswith("subproblem"))["refs"]]
        self.assertEqual(refs, ["sector:6@3:before", "flight:6", "flight:8", "flight:12", "flight:11"])

    def test_shortest_step6(self):
        out, line = self.subproblem(6, {17: 6, 18: 4, 19: 3})
        self.assertEqual(line, "Of the 3 flights in sector 21 at time 11, the 2 with the shortest stored flight duration "
                               "went to the solver: 19 (3 time periods) and 18 (4); 17 has 6.")

    def test_both_step9(self):
        out, line = self.subproblem(9, {28: 7, 30: 5})
        self.assertEqual(line, "Both flights in sector 35 at time 15 went to the solver: 28 and 30.")

    def test_tie_step12(self):
        out, line = self.subproblem(12, {55: 4, 56: 4, 57: 4})
        self.assertEqual(line, "Of the 3 flights in sector 5 at time 20, 2 went to the solver: 55 and 56; 57 has the "
                               "same stored duration and was left out by its higher flight number.")

    def test_forged_candidate(self):
        out, _ = self.subproblem(6, {17: 3, 18: 4, 19: 6})       # 17 would be first, but 18 and 19 were passed
        self.assertIn("candidate_rule", codes(out))
        self.assertEqual([line["kind"] for line in out["lines"]], ["record_error"])

    def test_forged_taken(self):
        out, _ = self.subproblem(6, {17: 6, 18: 4, 19: 3}, taken=3)
        self.assertIn("taken_rule", codes(out))


class DurationRule(Base):
    """2b. Runs that record flight_duration_rule = occupied_steps say "flight duration"; traces without the field
    (the June trace) keep "stored flight duration". Synthetic occupants as in Occupants; their durations equal the
    time periods of the June trajectories (step 3: 6 and 8 have 5; step 6: 19 has 3, 18 has 4; step 12: 55, 56: 4)."""

    def setUp(self):
        super().setUp()
        self.run_info = dict(self.run_info, flight_duration_rule="occupied_steps")

    def subproblem_line(self, n, durations, taken=2):
        self.records[n] = with_occupants(self.records[n], durations, taken)
        out = explain(self.records, n, self.run_info)
        return out, next((line for line in out["lines"] if line["kind"].startswith("subproblem")), None)

    def test_shortest_step6(self):
        out, line = self.subproblem_line(6, {17: 6, 18: 4, 19: 3})
        self.assertEqual(out["errors"], [])
        self.assertEqual(line["kind"], "subproblem_shortest")
        self.assertEqual(line["text"], "Of the 3 flights in sector 21 at time 11, the 2 with the shortest flight "
                                       "duration went to the solver: 19 (3 time periods) and 18 (4); 17 has 6.")
        self.assertIn("run_option(flight_duration_rule,occupied_steps)", line["source"])

    def test_tie_step3(self):
        out, line = self.subproblem_line(3, {6: 5, 8: 5, 11: 6, 12: 5})
        self.assertEqual(out["errors"], [])
        self.assertEqual(line["kind"], "subproblem_tie")
        self.assertEqual(line["text"], "Of the 4 flights in sector 6 at time 9, 2 went to the solver: 6 and 8; 12 has "
                                       "the same flight duration and was left out by its higher flight number; 11 has "
                                       "a longer one.")

    def test_tie_step12(self):
        out, line = self.subproblem_line(12, {55: 4, 56: 4, 57: 4})
        self.assertEqual(out["errors"], [])
        self.assertEqual(line["text"], "Of the 3 flights in sector 5 at time 20, 2 went to the solver: 55 and 56; 57 "
                                       "has the same flight duration and was left out by its higher flight number.")

    def test_rule_only_without_occupants(self):
        out = explain(self.records, 6, self.run_info)
        self.assertEqual(out["errors"], [])
        line = next(line for line in out["lines"] if line["kind"] == "subproblem_rule")
        self.assertEqual(line["text"], "Of the 3 flights in sector 21 at time 11, 18 and 19 went to the solver (rule: "
                                       "at most 2, shortest flight duration first, lower flight number on ties).")
        self.assertIn("run_option(flight_duration_rule,occupied_steps)", line["source"])

    def test_other_rule_keeps_stored_wording(self):
        self.run_info["flight_duration_rule"] = "something_else"
        out, line = self.subproblem_line(6, {17: 6, 18: 4, 19: 3})
        self.assertEqual(out["errors"], [])
        self.assertEqual(line["kind"], "subproblem_shortest")
        self.assertIn("the 2 with the shortest stored flight duration went to the solver", line["text"])
        self.assertIn("run_option(flight_duration_rule,other)", line["source"])

    def test_forged_duration(self):
        # synthetic: 18 recorded with 5, its trajectory has 4 time periods (same order, so no candidate_rule)
        out, _ = self.subproblem_line(6, {17: 6, 18: 5, 19: 3})
        self.assertEqual(codes(out), ["duration_rule"])
        self.assertEqual([line["kind"] for line in out["lines"]], ["record_error"])
        self.assertIn("a recorded flight duration differs from the time periods of its trajectory",
                      out["lines"][0]["text"])

    def test_forged_duration_without_rule_is_no_error(self):
        # traces without the field may hold a stored duration one period short: not checked
        del self.run_info["flight_duration_rule"]
        out, line = self.subproblem_line(6, {17: 6, 18: 5, 19: 3})
        self.assertEqual(out["errors"], [])
        self.assertIn("shortest stored flight duration", line["text"])

    def test_old_trace_source(self):
        del self.run_info["flight_duration_rule"]
        out, line = self.subproblem_line(6, {17: 6, 18: 4, 19: 3})
        self.assertIn("option_assumed(flight_duration_rule)", line["source"])
        self.assertEqual(line["kind"], "subproblem_shortest")


class AllSteps(Base):
    """3. All 13 steps; templates; the sector count of a split."""

    def setUp(self):
        super().setUp()
        self.run_info["evaluation_window"] = 25        # 350 = 14 x 25; the filed CE-7x7 layout is static

    def test_no_error_sources_templates(self):
        _, templates, error_texts = reasons.solve("")
        for n in self.records:
            out = explain(self.records, n, self.run_info)
            self.assertEqual(out["errors"], [], f"step {n}")
            for line in out["lines"]:
                self.assertTrue(line["source"], f"step {n} {line['id']}")
                self.assertIn(line["kind"], templates)
        for text in list(templates.values()) + list(error_texts.values()):
            self.assertTrue(text.isascii(), text)
            for bad in ('"', "\\", "'"):
                self.assertNotIn(bad, text)

    def test_ce7_split_arithmetic(self):
        """A CE-7x7 fixture fact (not a rule of reasons.lp): delta = (parts-1) x (25 - time)."""
        expected = {1: 76, 2: 108, 3: 96, 6: 14, 9: 50}
        for n, change in expected.items():
            out = explain(self.records, n, self.run_info)
            delta = next(d for d in out["deltas"] if d["key"] == "SECTOR-NUMBER")
            self.assertEqual(delta["change"], change, f"step {n}")
            sc = self.records[n]["sector_changes"]
            parts = len(sc["post_sector_config"])
            self.assertEqual(change, (parts - 1) * (25 - sc["time_index"]))

    def with_sector_number(self, after):
        """synthetic: step 6 with another SECTOR-NUMBER after the step"""
        self.records[6]["objectives"]["SECTOR-NUMBER"] = after

    def test_smaller_delta_is_no_error(self):
        self.with_sector_number(640)                     # delta 10 < 14: a later split overwritten
        self.assertEqual(codes(explain(self.records, 6, self.run_info)), [])

    def test_larger_delta_is_split_extent(self):
        self.with_sector_number(650)                     # delta 20 > 14
        self.assertIn("split_extent", codes(explain(self.records, 6, self.run_info)))

    def test_no_window_no_split_check(self):
        self.with_sector_number(650)
        del self.run_info["evaluation_window"]
        self.assertEqual(codes(explain(self.records, 6, self.run_info)), [])

    def test_merging_on(self):
        # synthetic: merging on, and an after-part holding navpoint 40 from outside sector 21
        self.run_info["minimize_number_sectors"] = True
        self.records[6]["sector_changes"]["post_sector_config"]["35"]["vertices"].append(40)
        out = explain(self.records, 6, self.run_info)
        self.assertNotIn("parts_mismatch", codes(out))
        self.assertEqual(out["errors"], [])
        self.assertIn("Sector merging was on in this run; layout changes outside sector 21 are not described.",
                      [line["text"] for line in out["lines"]])
        self.assertNotIn("layout_split", [line["kind"] for line in out["lines"]])

    def test_parts_mismatch_without_merging(self):
        self.records[6]["sector_changes"]["post_sector_config"]["35"]["vertices"].append(40)
        self.assertIn("parts_mismatch", codes(explain(self.records, 6, self.run_info)))


class Outcomes(Base):
    """4. Layout and flight results on single steps; a record without its lp file."""

    def lines(self, n):
        out = explain(self.records, n, self.run_info)
        self.assertEqual(out["errors"], [])
        return [line["text"] for line in out["lines"]]

    def test_step4_layout_same(self):
        self.assertIn("The layout of sector 10 did not change.", self.lines(4))

    def test_step9_one_changed(self):
        self.assertIn("New trajectory in this step: 30.", self.lines(9))

    def test_step11_unchanged_and_one_split(self):
        lines = self.lines(11)
        self.assertIn("The solver left 33 and 45 unchanged.", lines)
        self.assertIn("The solver could choose among 2 layouts of sector 21: the current one and 1 split.", lines)

    def test_step12_current_layout_only(self):
        self.assertIn("Only the current layout of sector 5 was offered.", self.lines(12))

    def test_without_lp_file(self):
        out = explain(self.records, 6, self.run_info, folder=self.tmp)
        self.assertEqual(out["errors"], [])
        kinds = [line["kind"] for line in out["lines"]]
        for prefix in ("subproblem", "legs", "delays", "layouts", "unchanged"):
            self.assertFalse([k for k in kinds if k.startswith(prefix)], prefix)
        self.assertIn("changed", kinds)


class Rejected(Base):
    """5. Synthetic rejected attempt: step 7 rejected with step 6's objectives, later steps renumbered."""

    def setUp(self):
        super().setUp()
        r = self.records
        records = {n: r[n] for n in range(1, 7)}
        records[7] = rejected_copy(r[7], 7, r[6])
        for n in range(7, 14):
            params = {"additional_time_increase": 1, "failed_attempts_before": 1} if n == 7 else None
            records[n + 1] = renumbered(r[n], n + 1, params)
        self.records = records

    def test_not_kept_and_retry(self):
        out = explain(self.records, 7, self.run_info)
        self.assertEqual(out["errors"], [])
        self.assertIs(out["kept"], False)
        self.assertEqual([line["kind"] for line in out["lines"]], ["hotspot_rule", "not_kept"])
        not_kept = text_of(out, "not_kept")
        self.assertIn("stays at 22", not_kept)
        self.assertIn("20 to 39", not_kept)
        # the line names the attempt's own spot (16 at 12), not the spot of step 6 under whose title it is shown
        self.assertEqual(not_kept, "Step 7 (sector 16 at time 12) was not kept: the total overload did not fall "
                                   "(stays at 22). The plan is unchanged; a next attempt at this spot would offer "
                                   "departure delays of 20 to 39 time periods.")
        self.assertIn("hotspot(7,16,12)", next(l for l in out["lines"] if l["kind"] == "not_kept")["source"])
        nxt = explain(self.records, 8, self.run_info)
        self.assertEqual(nxt["errors"], [])
        self.assertEqual(text_of(nxt, "retry"), "Step 7 at this spot was not kept, so this step offered departure "
                                                "delays of 20 to 39 time periods (no shorter ones).")
        self.assertIn("20 to 39 time periods later", text_of(nxt, "delays"))

    @staticmethod
    def lp_with_starts(first, plan_start=10, flights=(18, 19), width=20):
        """synthetic lp: each of `flights` has `width` paths starting at first..first+width-1, current start plan_start"""
        text = "config(0,14).\n"
        for f in flights:
            text += f"paths({f},0..{width - 1}).\nflightPlan({f},{plan_start},51).\n"
            text += "".join(f"actual_flight_operations_start_time({f},{first + p},{p}).\n" for p in range(width))
        return text

    def test_offered_delays_checked_against_the_delay_range(self):
        # step 6 (k = 0, W = 20): starts 10..29 after a current start at 10 are the delays 0..19
        # (the reduced lp has no later legs, so other checks may fire; only the delay check is looked at here)
        out = explain(self.records, 6, self.run_info, folder=self.folder_with_lp(6, self.lp_with_starts(10)))
        self.assertNotIn("delay_options", codes(out))
        # step 8 after one rejection (k = 1; its record names the lp file of June step 7): the lp must offer 20..39;
        # one that offers 0..19 is a record error
        self.assertEqual(self.records[8]["subproblems"][0]["instance_file"], "lp/iter_00007_0.lp")
        wrong = explain(self.records, 8, self.run_info, folder=self.folder_with_lp(7, self.lp_with_starts(10)))
        self.assertIn("delay_options", codes(wrong))
        right = explain(self.records, 8, self.run_info, folder=self.folder_with_lp(7, self.lp_with_starts(30)))
        self.assertNotIn("delay_options", codes(right))

    def test_forged_overload_before(self):
        self.records[6]["objectives"]["OVERLOAD-BEFORE"] = 27
        out = explain(self.records, 6, self.run_info)
        self.assertIn("before_mismatch", codes(out))
        self.assertTrue(next(d for d in out["deltas"] if d["key"] == "OVERLOAD")["hidden"])
        self.assertFalse(next(d for d in out["deltas"] if d["key"] == "REROUTE")["hidden"])
        self.assertEqual([line["text"] for line in out["lines"]],
                         ["The recorded numbers of this step do not match the documented rules: the overload before "
                          "the step differs from the value of the previous step. No explanation is shown."])

    def test_missing_key(self):
        del self.records[6]["objectives"]["ARRIVAL-DELAY"]
        self.assertIn("missing_objective", codes(explain(self.records, 6, self.run_info)))

    def test_missing_previous(self):
        del self.records[5]
        out = explain(self.records, 6, self.run_info)
        self.assertIn("missing_previous", codes(out))
        self.assertTrue(all(d["hidden"] for d in out["deltas"]))

    def test_template_with_wrong_key(self):
        real = reasons._texts

        def broken(ctl):
            templates, errors = real(ctl)
            templates = dict(templates, hotspot_rule=templates["hotspot_rule"].replace("{sector}", "{wrong}"))
            return templates, errors

        with mock.patch.object(reasons, "_texts", broken):
            out = explain(self.records, 6, self.run_info)
        self.assertEqual(codes(out), ["render_error"])
        self.assertEqual([line["text"] for line in out["lines"]],
                         ["The explanation text of this step could not be produced."])


class Start(Base):
    """6. The filed plan (step 0)."""

    def test_june_start(self):
        out = reasons.explain_start(self.run_info, run_length=13, run_kept=13)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["iteration"], 0)
        first = out["lines"][0]["text"]
        self.assertIn("total overload of 110", first)
        self.assertIn("6 (44), 0 (29), 24 (19), 21 (14) and 12 (4)", first)
        self.assertEqual(out["lines"][1]["text"], "Each step works on one overloaded sector at a single time and is kept "
                                                  "only if the total overload falls.")
        self.assertEqual([r["ref"] for r in out["lines"][0]["refs"]][:2], ["sector:6@0:initial", "sector:0@0:initial"])
        overload = out["deltas"][0]
        self.assertEqual((overload["key"], overload["before"], overload["after"]), ("OVERLOAD", None, 110))


class AspLayer(unittest.TestCase):
    """7. Shown atoms of reasons.lp + reasons_text.lp on the hand-reviewed step-6 facts."""

    @staticmethod
    def expected():
        return {line.strip().rstrip(".") for line in (FIX / "step06_expected.lp").read_text().splitlines()
                if line.strip() and not line.startswith("%")}

    def test_clingo_562(self):
        atoms, _, _ = reasons.solve((FIX / "step06_facts.lp").read_text())
        self.assertEqual({str(a) for a in atoms}, self.expected())

    def test_clingo_580(self):
        if not CLINGO_58.exists():
            self.skipTest("clingo 5.8.0 (aspaeroflow-xai env) not installed")
        out = subprocess.run([str(CLINGO_58), "--warn=none", "--outf=2", "0", str(XAI_DIR / "reasons.lp"),
                              str(XAI_DIR / "reasons_text.lp"), str(FIX / "step06_facts.lp")],
                             capture_output=True, text=True, timeout=60)
        result = json.loads(out.stdout)
        self.assertIn("5.8", result["Solver"])
        self.assertEqual(result["Models"]["Number"], 1)
        witnesses = result["Call"][0]["Witnesses"]
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(set(witnesses[0]["Value"]), self.expected())


class Branches(Base):
    """9. One synthetic record (or run.json) per branch, each with its exact line (numbers of step 6 unless stated)."""

    LP_ONE = "paths(18,0..59).\nconfig(0,14).\nconfig(1,14).\n"

    def lines(self, out):
        self.assertEqual(out["errors"], [])
        return [line["text"] for line in out["lines"]]

    # sequential mode: optimize_flights.py writes one path per solver flight, delay k*W with k = 0
    LP_SEQUENTIAL = ("paths(18,0..0).\npaths(19,0..0).\nactual_flight_operations_start_time(18,9,0).\n"
                     "actual_flight_operations_start_time(19,8,0).\nconfig(0,14).\n"
                     "chosen_path(26,0) :- chosen_path(19,0).\n")

    def test_sequential(self):
        self.run_info["sequential_execution"] = True
        folder = self.folder_with_lp(6, self.LP_SEQUENTIAL)
        out = explain(self.records, 6, self.run_info, folder=folder)
        lines = self.lines(out)
        self.assertEqual(lines[0], "Sector 21 at time 11 had 3 flights for a capacity of 1.")
        self.assertNotIn("rule", [line["kind"] for line in out["lines"]])
        self.assertIn("18 and 19 could each take one candidate route only, departing as in the current plan "
                      "(sequential mode).", lines)
        self.assertFalse([t for t in lines if "0 to 19" in t or "delay range" in t], lines)
        self.assertIn("This describes this step only. The spot and the flights follow fixed rules; the new "
                      "trajectories and the layout are the answer of the solver to the sub-problem of this step.", lines)
        # one solver flight
        one = "paths(18,0..0).\nactual_flight_operations_start_time(18,9,0).\nconfig(0,14).\n"
        self.records[6] = self.one_flight_record()
        self.assertIn("18 could take one candidate route only, departing as in the current plan (sequential mode).",
                      self.lines(explain(self.records, 6, self.run_info, folder=self.folder_with_lp(6, one))))
        self.records[7] = rejected_copy(self.records[7], 7, self.records[6])
        out = explain(self.records, 7, self.run_info, folder=self.tmp)      # (step 7's lp is not sequential)
        self.assertEqual(text_of(out, "not_kept"), "Step 7 (sector 16 at time 12) was not kept: the total overload "
                                                   "did not fall (stays at 22). The plan is unchanged.")

    def one_flight_record(self, n=6):
        """synthetic: step 6 with one flight in the cell (18), capacity 0, only 18 changed, an lp with 18 only"""
        r = copy.deepcopy(self.records[n])
        r["hotspot"].update({"demand": 1, "capacity": 0, "overload": 1})
        r = with_occupants(r, {18: 4}, taken=1)
        r["flight_changes"] = {"18": r["flight_changes"]["18"]}
        return r

    def test_sequential_without_start_times(self):
        # an lp without start times is no evidence of one departure option: no delays line in sequential mode
        self.run_info["sequential_execution"] = True
        out = explain(self.records, 6, self.run_info,
                      folder=self.folder_with_lp(6, "paths(18,0..0).\npaths(19,0..0).\nconfig(0,14).\n"
                                                    "chosen_path(26,0) :- chosen_path(19,0).\n"))
        self.assertEqual(out["errors"], [])
        self.assertFalse([line for line in out["lines"] if line["kind"].startswith("delays")])

    def test_delay_options_mismatch(self):
        # the June lp (60 paths for 18) does not fit sequential mode; a non-sequential lp with one start does not
        # fit W = 20
        self.run_info["sequential_execution"] = True
        self.assertEqual(codes(explain(self.records, 6, self.run_info)), ["delay_options"])
        self.run_info["sequential_execution"] = False
        self.assertEqual(codes(explain(self.records, 6, self.run_info,
                                       folder=self.folder_with_lp(6, self.LP_SEQUENTIAL))), ["delay_options"])

    def test_one_flight_capacity_zero(self):
        self.records[6] = self.one_flight_record()
        lines = self.lines(explain(self.records, 6, self.run_info, folder=self.folder_with_lp(6, self.LP_ONE)))
        self.assertEqual(lines[0], "Sector 21 at time 11 had 1 flight for a capacity of 0: the earliest overload, "
                                   "lowest sector number first.")
        self.assertIn("The only flight in sector 21 at time 11 went to the solver: 18.", lines)
        self.assertIn("18 could depart 0 to 19 time periods later than in the current plan.", lines)

    def test_composite_sum(self):
        self.run_info["composite_sector_function"] = "sum"
        kinds = [line["kind"] for line in explain(self.records, 6, self.run_info)["lines"]]
        self.assertFalse([k for k in kinds if k.startswith("capacity")])

    def test_merging_on(self):
        self.run_info["minimize_number_sectors"] = True
        self.assertIn("Sector merging was on in this run; layout changes outside sector 21 are not described.",
                      self.lines(explain(self.records, 6, self.run_info)))

    def test_start_without_overload(self):
        run = dict(self.run_info, initial_overload=0, initial_sector_overload={})
        out = reasons.explain_start(run)
        self.assertEqual(out["lines"][0]["text"], "The filed plan has no overload.")

    def test_one_leg(self):
        lp = "paths(18,0..59).\npaths(19,0..19).\nconfig(0,14).\nchosen_path(26,0) :- chosen_path(19,0).\n"
        self.assertIn("Later flights of the same aircraft also went to the solver: 26 (after 19).",
                      self.lines(explain(self.records, 6, self.run_info, folder=self.folder_with_lp(6, lp))))

    def test_shortest_one(self):
        # synthetic: occupants {18:4, 19:3}, demand 2, capacity 1, max_aircraft 1; only 19 passed and changed
        r = copy.deepcopy(self.records[6])
        r["hotspot"].update({"demand": 2, "capacity": 1, "overload": 1})
        r["parameters"]["max_aircraft"] = 1
        r = with_occupants(r, {18: 4, 19: 3}, taken=1)
        r["flight_changes"] = {"19": r["flight_changes"]["19"]}
        self.records[6] = r
        lines = self.lines(explain(self.records, 6, self.run_info,
                                   folder=self.folder_with_lp(6, "paths(19,0..19).\nconfig(0,14).\n")))
        self.assertIn("Of the 2 flights in sector 21 at time 11, the one with the shortest stored flight duration went "
                      "to the solver: 19 (3 time periods); 18 has 4.", lines)

    def test_tie_many(self):
        self.records[3] = with_occupants(self.records[3], {6: 5, 8: 5, 11: 5, 12: 5})
        self.assertIn("Of the 4 flights in sector 6 at time 9, 2 went to the solver: 6 and 8; 11 and 12 have the "
                      "same stored duration and were left out by their higher flight numbers.",
                      self.lines(explain(self.records, 3, self.run_info)))

    def test_left_out_two(self):
        r = copy.deepcopy(self.records[6])
        r["hotspot"].update({"demand": 4, "overload": 3})
        self.records[6] = with_occupants(r, {17: 6, 18: 4, 19: 3, 20: 7})
        line = text_of(explain(self.records, 6, self.run_info), "subproblem")
        self.assertTrue(line.endswith("; 17 has 6 and 20 has 7."), line)

    def retries(self, count, max_aircraft=2):
        """synthetic: `count` rejected attempts at the spot of June step 7, then step 7 kept"""
        r = self.records
        records = {n: r[n] for n in range(1, 7)}
        for k in range(count):
            records[7 + k] = rejected_copy(r[7], 7 + k, r[6], {"additional_time_increase": k, "failed_attempts_before": k})
        records[7 + count] = renumbered(r[7], 7 + count, {"additional_time_increase": count,
                                                           "failed_attempts_before": count, "max_aircraft": max_aircraft})
        return records, 7 + count

    def test_retry_five(self):
        records, n = self.retries(5)
        out = explain(records, n, self.run_info)
        self.assertEqual(out["errors"], [])
        self.assertEqual(text_of(out, "retry"), "Steps 7, 8, 9, 10 and 11 at this spot were not kept, so this step "
                                                "offered departure delays of 100 to 119 time periods (no shorter ones) "
                                                "and only the current layout.")

    def test_retry_ten(self):
        records, n = self.retries(10, max_aircraft=1)
        out = explain(records, n, self.run_info)
        self.assertEqual(out["errors"], [])
        self.assertTrue(text_of(out, "retry").endswith(
            "offered departure delays of 200 to 219 time periods (no shorter ones) and only the current layout and "
            "at most 1 flight."), text_of(out, "retry"))

    def test_run_ends_on_rejection(self):
        records = dict(self.records)
        records[14] = rejected_copy(self.records[13], 14, self.records[13])
        out = explain(records, 14, self.run_info, run_length=14, run_kept=13)
        self.assertEqual(out["errors"], [])
        self.assertIn("a next attempt at this spot would offer departure delays of 20 to 39 time periods.",
                      text_of(out, "not_kept"))

    def test_retry_count_mismatch(self):
        records, n = self.retries(2)
        records[n]["parameters"]["failed_attempts_before"] = 3
        self.assertIn("retry_count", codes(explain(records, n, self.run_info)))


class Sessions(unittest.TestCase):
    """8. Step headers in replayed and live sessions."""

    def test_replay(self):
        session = ReplaySession(FIX)
        begun = session.begin()
        self.assertEqual(begun["step"]["run_length"], 13)
        self.assertEqual(begun["step"]["errors"], [])
        count = 0
        while True:
            summary = session.step()
            if summary is None:
                break
            count += 1
            self.assertEqual(summary["step"]["errors"], [], summary["iteration"])
            self.assertEqual(summary["step"]["iteration"], summary["iteration"])
            self.assertEqual(summary["step"]["run_length"], 13)
        self.assertEqual(count, 13)

    def test_live(self):
        with tempfile.TemporaryDirectory() as tmp:   # (the optimizer prints its JSON lines)
            session = OptimizerSession(TEN, Path(tmp) / "trace")
            begun = session.begin()
            self.assertEqual(begun["step"]["errors"], [])
            run = json.loads((Path(tmp) / "trace" / "run.json").read_text())
            for key in ("sequential_execution", "minimize_number_sectors", "max_number_sectors", "convex_sectors",
                        "evaluation_window"):
                self.assertIn(key, run)
            summaries = []
            while True:
                summary = session.step()
                if summary is None:
                    break
                summaries.append(summary)
                if session.status == "finished":
                    break
            self.assertGreater(len(summaries), 0)
            for s in summaries:
                self.assertEqual(s["step"]["errors"], [], s["iteration"])
                self.assertIsNone(s["step"]["run_length"])
                self.assertIn("flights", s["hotspot"])
                self.assertIn("taken", s["hotspot"])
                for f in s["flight_changes"]:
                    self.assertIn(str(f), s["aircraft"])

    def test_live_sequential(self):
        """A live run in sequential mode: one path per solver flight, so no delay range in any header. On this
        instance no sequential step is kept and the run ends with "SEQUENTIAL END", which OptimizerSession does
        not treat as the end (step() then returns the last record again), so the loop stops at a repeat. Its
        first record, made kept (synthetic: accepted, overload one lower, no flight changes), shows the
        sequential delays line read from the real lp file."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trace"
            session = OptimizerSession(TEN, folder, {"sequential_execution": "true"})
            session.begin()
            seen = []
            for _ in range(10):
                summary = session.step()
                if summary is None or summary["iteration"] in seen:
                    break
                seen.append(summary["iteration"])
                self.assertEqual(summary["step"]["errors"], [], summary["iteration"])
                texts = [line["text"] for line in summary["step"]["lines"]]
                self.assertFalse([t for t in texts if "time periods later" in t or "delay range" in t], texts)
                if session.status == "finished":
                    break
            self.assertGreater(len(seen), 0)
            session.app._xai_trace.close()
            run = json.loads((folder / "run.json").read_text())
            self.assertIs(run["sequential_execution"], True)
            record = json.loads((folder / "trace.jsonl").read_text().splitlines()[0])
            self.assertEqual(record["iteration"], 1)
            lp = (folder / record["subproblems"][0]["instance_file"]).read_text()
            record = dict(record, accepted=True, flight_changes={},
                          objectives=dict(record["objectives"], OVERLOAD=run["initial_objectives"]["OVERLOAD"] - 1))
            out = reasons.explain_step(record, previous=None, run=run, trace_folder=folder)
            self.assertEqual(out["errors"], [])
            kinds = [line["kind"] for line in out["lines"]]
            texts = [line["text"] for line in out["lines"]]
            self.assertTrue([k for k in kinds if k.startswith("delays_sequential")], (kinds, lp[:300]))
            self.assertIn("scope_sequential", kinds)
            self.assertFalse([t for t in texts if "time periods later" in t or "delay range" in t], texts)

if __name__ == "__main__":
    unittest.main()
