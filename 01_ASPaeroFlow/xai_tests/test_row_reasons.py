"""Per-measure reasons (K12): one line under each row of the change list, from the keep contrasts of the step.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

Fixtures: fixtures/CE7_JUNE_TRACE (June study run, 13 kept steps) and fixtures/K30_TG4 (study trace at 15-minute
time steps, 10 kept steps), each with keep_contrasts.jsonl computed by xai/keep.py from the full lp files (the lp
files in the fixtures are reduced and cannot be solved). Entries marked "synthetic" are fixture entries with fields
changed by the test; the comment next to each says which. Live checks use the 10-flight run of shared_run.py.
With XAI_CE7_FULL=<folder with ce7_trace and K30_trace_ce7_tg4> the two files are also recomputed and compared.
"""
import copy
import json
import os
import re
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
REPO = OPT.parent
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai import reasons  # noqa: E402
from src.aspaeroflow.xai.contrastive import IterationExplainer  # noqa: E402
from src.aspaeroflow.xai.keep import FILE_NAME, append_keep, keep_matches, precompute, read_keep  # noqa: E402
from src.aspaeroflow.xai.session import ReplaySession  # noqa: E402
from src.aspaeroflow.xai.subproblem import compare  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402

FIX = HERE / "fixtures" / "CE7_JUNE_TRACE"
TG4 = HERE / "fixtures" / "K30_TG4"
XAI_DIR = OPT / "src" / "aspaeroflow" / "xai"
CLINGO_58 = Path.home() / "anaconda3" / "envs" / "aspaeroflow-xai" / "bin" / "clingo"
FOLDERS = {"JUNE": FIX, "TG4": TG4}
LEVELS = ("overload", "delay", "sectors", "changed", "config")
KIND_LEVEL = {"keep_overload": "overload", "keep_delay": "delay", "keep_sectors": "sectors",
              "keep_changed": "changed", "keep_config": "config"}

EXPECTED = {
    ('JUNE', 3): [
        ('sector:6@3:before', 'layout_keep_overload', 'capacity',
         'If sector 6 had not been split, the best answer of this step would leave a total overload of 62 '
         'instead of 42, but it would have 24 instead of 30 open sectors at time 9.'),
        ('flight:6', 'keep_overload', 'capacity',
         'With flight 6 kept as it was, the best answer of this step would leave a total overload of 46 '
         'instead of 42, but the net arrival delay of the flights of this step '
         'would be 0 instead of 31 time periods. Another new '
         'trajectory was exactly as good.'),
        ('flight:8', 'keep_overload', 'capacity',
         'With flight 8 kept as it was, the best answer of this step would leave a total overload of 43 '
         'instead of 42, but it would have 1 instead of 2 flights sent to the solver off the filed route or '
         'the earliest offered departure. Another new trajectory was exactly as good.'),
        ('flight:32', 'leg', 'rule',
         'Flies after flight 6 on the same aircraft and moved with it.'),
        ('flight:53', 'leg', 'rule',
         'Flies after flight 6 on the same aircraft and moved with it.'),
    ],
    ('JUNE', 6): [
        ('sector:21@6:before', 'layout_keep_overload', 'capacity',
         'If sector 21 had not been split, the best answer of this step would leave a total overload of 25 '
         'instead of 22, but it would have 30 instead of 31 open sectors at time 11.'),
        ('flight:18', 'keep_overload', 'capacity',
         'With flight 18 kept as it was, the best answer of this step would leave a total overload of 23 '
         'instead of 22, but the net arrival delay of the flights of this step '
         'would be 26 instead of 27 time periods. Another new '
         'trajectory was exactly as good.'),
        ('flight:19', 'keep_overload', 'capacity',
         'With flight 19 kept as it was, the best answer of this step would leave a total overload of 24 '
         'instead of 22, but the net arrival delay of the flights of this step '
         'would be 1 instead of 27 time periods.'),
        ('flight:26', 'leg', 'rule',
         'Flies after flight 19 on the same aircraft and moved with it.'),
    ],
    ('JUNE', 9): [
        ('sector:35@9:before', 'layout_keep_delay', 'cost',
         'If sector 35 had not been split, the best answer of this step would have the same total overload '
         'and a net arrival delay of 1 instead of 0 time periods, summed over the flights of this step, but '
         'it would have 31 instead of 36 open sectors at time 15.'),
        ('flight:30', 'keep_delay', 'cost',
         'With flight 30 kept as it was, the best answer of this step would have the same total overload and '
         'a net arrival delay of 8 instead of 0 time periods, summed over the flights of this step, but it '
         'would have 31 instead of 36 open sectors at time 15.'),
    ],
    ('JUNE', 11): [
        ('flight:38', 'keep_tie_other', 'tie',
         'Flight 38 or flight 33 had to change: keeping both as they were would leave a total overload of 7 '
         'instead of 6. Keeping flight 38 as it was and changing flight 33 is exactly as good; the solver '
         'picked one of the equally good answers.'),
    ],
    ('JUNE', 12): [
        ('flight:55', 'keep_overload', 'capacity',
         'With flight 55 kept as it was, the best answer of this step would leave a total overload of 4 '
         'instead of 2, but the net arrival delay of the flights of this step '
         'would be 2 instead of 11 time periods. Another new '
         'trajectory was exactly as good.'),
        ('flight:56', 'keep_overload', 'capacity',
         'With flight 56 kept as it was, the best answer of this step would leave a total overload of 4 '
         'instead of 2, but the net arrival delay of the flights of this step '
         'would be 2 instead of 11 time periods. Another new '
         'trajectory was exactly as good.'),
    ],
    ('TG4', 1): [
        ('sector:6@1:before', 'layout_keep_overload', 'capacity',
         'If sector 6 had not been split, the best answer of this step would leave a total overload of 55 '
         'instead of 31, but it would have 14 instead of 18 open sectors at time 17.'),
        ('flight:4', 'keep_overload', 'capacity',
         'With flight 4 kept as it was, the best answer of this step would leave a total overload of 33 '
         'instead of 31, but the net arrival delay of the flights of this step '
         'would be 3 instead of 10 time periods. Another new '
         'trajectory was exactly as good.'),
        ('flight:29', 'leg', 'rule',
         'Flies after flight 4 on the same aircraft and moved with it.'),
    ],
    ('TG4', 2): [
        ('flight:11', 'keep_delay', 'cost',
         'With flight 11 kept as it was, the best answer of this step would have the same total overload and '
         'a net arrival delay of 1 instead of 0 time periods, summed over the flights of this step.'),
    ],
    ('TG4', 3): [
        ('sector:24@3:before', 'layout_keep_overload', 'capacity',
         'If sector 24 had not been split, the best answer of this step would leave a total overload of 29 '
         'instead of 13, but it would have 18 instead of 24 open sectors at time 23.'),
    ],
    ('TG4', 4): [
        ('sector:0@4:before', 'layout_keep_overload', 'capacity',
         'If sector 0 had not been split, the best answer of this step would leave a total overload of 11 '
         'instead of 8, but it would have 24 instead of 25 open sectors at time 27.'),
    ],
    ('TG4', 5): [
        ('sector:16@5:before', 'layout_keep_delay', 'cost',
         'If sector 16 had not been split, the best answer of this step would have the same total overload '
         'and a net arrival delay of 5 instead of 0 time periods, summed over the flights of this step, but '
         'it would have 25 instead of 26 open sectors at time 28.'),
        ('flight:27', 'keep_overload', 'capacity',
         'With flight 27 kept as it was, the best answer of this step would leave a total overload of 7 '
         'instead of 6, but it would have 1 instead of 2 flights sent to the solver off the filed route or '
         'the earliest offered departure.'),
        ('flight:28', 'keep_overload', 'capacity',
         'With flight 28 kept as it was, the best answer of this step would leave a total overload of 7 '
         'instead of 6, but it would have 1 instead of 2 flights sent to the solver off the filed route or '
         'the earliest offered departure.'),
    ],
    ('TG4', 6): [
        ('flight:25', 'keep_tie_other', 'tie',
         'Flight 25 or flight 23 had to change: keeping both as they were would leave a total overload of 6 '
         'instead of 5. Keeping flight 25 as it was and changing flight 23 is exactly as good; the solver '
         'picked one of the equally good answers.'),
    ],
    ('TG4', 7): [
        ('flight:44', 'keep_delay', 'cost',
         'With flight 44 kept as it was, the best answer of this step would have the same total overload and '
         'a net arrival delay of 1 instead of 0 time periods, summed over the flights of this step. Another '
         'new trajectory was exactly as good.'),
    ],
    ('TG4', 8): [
        ('flight:54', 'keep_tie_other', 'tie',
         'Flight 54 or flight 55 had to change: keeping both as they were would leave a total overload of 4 '
         'instead of 2. Keeping flight 54 as it was and changing flight 55 is exactly as good; the solver '
         'picked one of the equally good answers.'),
    ],
    ('TG4', 9): [
        ('flight:78', 'keep_tie_other', 'tie',
         'Flight 78 or flight 76 had to change: keeping both as they were would leave a total overload of 2 '
         'instead of 1. Keeping flight 78 as it was and changing flight 76 is exactly as good; the solver '
         'picked one of the equally good answers.'),
    ],
    ('TG4', 10): [
        ('flight:88', 'keep_tie_other', 'tie',
         'Flight 88 or flight 79 had to change: keeping both as they were would leave a total overload of 1 '
         'instead of 0. Keeping flight 88 as it was and changing flight 79 is exactly as good; the solver '
         'picked one of the equally good answers.'),
    ],
}

#: kept steps with a sector line (a split of the hotspot sector, chosen layout not 0)
SECTOR_ROWS = {"JUNE": {1, 2, 3, 6, 9}, "TG4": {1, 3, 4, 5}}


def load(folder):
    run = json.loads((folder / "run.json").read_text())
    records = {}
    for line in (folder / "trace.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["iteration"]] = r
    return run, records, read_keep(folder)


def rows_of(folder, records, run, keep, n, record=None):
    previous, rejected = reasons.step_context(records, n)
    return reasons.explain_rows(record or records[n], keep, previous=previous, run=run, trace_folder=folder,
                                rejected_before=rejected)


def header_of(folder, records, run, n, record=None):
    previous, rejected = reasons.step_context(records, n)
    return reasons.explain_step(record or records[n], previous=previous, run=run, trace_folder=folder,
                                rejected_before=rejected)


def set_costs(entry, **costs):
    """synthetic: costs of a stored answer changed; its clingo vector follows (all five levels)"""
    entry["costs"] = dict(entry["costs"], **costs)
    entry["clingo_cost"] = [entry["costs"][level] for level in LEVELS]


class Texts(unittest.TestCase):
    """1, 2. The rows of both traces: verbatim on the oracle steps, one per changed flight, sector rows."""

    @classmethod
    def setUpClass(cls):
        cls.data = {name: load(folder) for name, folder in FOLDERS.items()}

    def test_verbatim(self):
        for (name, n), expected in EXPECTED.items():
            run, records, keep = self.data[name]
            with self.subTest(trace=name, step=n):
                out = rows_of(FOLDERS[name], records, run, keep[n], n)
                self.assertEqual(out["errors"], [])
                self.assertIsNone(out["notice"])
                self.assertEqual([(r["ref"], r["kind"], r["class"], r["text"]) for r in out["rows"]], expected)

    def test_design_sentences(self):
        """The three sentences quoted in full by the design."""
        texts = {}
        for name in FOLDERS:
            run, records, keep = self.data[name]
            for n in keep:
                for r in rows_of(FOLDERS[name], records, run, keep[n], n)["rows"]:
                    texts[(name, n, r["ref"])] = r["text"]
        self.assertEqual(texts[("JUNE", 6, "flight:19")],
                         "With flight 19 kept as it was, the best answer of this step would leave a total overload "
                         "of 24 instead of 22, but the net arrival delay of the flights of this step would be 1 instead of "
                         "27 time periods.")
        self.assertEqual(texts[("JUNE", 9, "flight:30")],
                         "With flight 30 kept as it was, the best answer of this step would have the same total "
                         "overload and a net arrival delay of 8 instead of 0 time periods, summed over the flights of "
                         "this step, but it would have 31 instead of 36 open sectors at time 15.")
        self.assertEqual(texts[("TG4", 8, "flight:54")],
                         "Flight 54 or flight 55 had to change: keeping both as they were would leave a total "
                         "overload of 4 instead of 2. Keeping flight 54 as it was and changing flight 55 is exactly "
                         "as good; the solver picked one of the equally good answers.")

    def test_every_step(self):
        for name, folder in FOLDERS.items():
            run, records, keep = self.data[name]
            self.assertEqual(sorted(keep), [n for n, r in sorted(records.items()) if r["accepted"]])
            for n in keep:
                with self.subTest(trace=name, step=n):
                    out = rows_of(folder, records, run, keep[n], n)
                    self.assertEqual(out["errors"], [])
                    flights = sorted(int(f) for f in records[n]["flight_changes"])
                    self.assertEqual([r["ref"] for r in out["rows"] if r["ref"].startswith("flight:")],
                                     [f"flight:{f}" for f in flights])
                    sectors = [r for r in out["rows"] if r["ref"].startswith("sector:")]
                    self.assertEqual(len(sectors), 1 if n in SECTOR_ROWS[name] else 0)
                    if sectors:
                        self.assertEqual(sectors[0]["ref"], f"sector:{records[n]['hotspot']['sector']}@{n}:before")
                    for r in out["rows"]:
                        self.assertTrue(r["source"], r)
                        self.assertNotIn('"', r["text"])
                        self.assertNotIn("'", r["text"])
                        self.assertNotIn("\\", r["text"])
                    # the row of a flight links the flights it names
                    for r in out["rows"]:
                        named = set(re.findall(r"flight (\d+)", r["text"]))
                        self.assertEqual({x["text"] for x in r["refs"] if x["ref"].startswith("flight:")}, named,
                                         r["text"])

    def test_header_unchanged(self):
        """The header lines are the same with and without the keep facts (rows never enter the header)."""
        for name, folder in FOLDERS.items():
            run, records, keep = self.data[name]
            for n in keep:
                with self.subTest(trace=name, step=n):
                    previous, rejected = reasons.step_context(records, n)
                    facts = reasons.step_facts(records[n], previous=previous, run=run, trace_folder=folder,
                                               rejected_before=rejected)
                    atoms, templates, error_texts = reasons.solve(facts + reasons.keep_facts(keep[n], run, records[n]))
                    lines, errors = reasons._finish(n, atoms, templates, error_texts)
                    header = header_of(folder, records, run, n)
                    self.assertEqual(lines, header["lines"])
                    self.assertEqual(errors, header["errors"])
                    self.assertFalse([line for line in header["lines"] if line["level"] == "row"])

    def test_step_payload_has_no_rows(self):
        session = ReplaySession(FIX)
        session.begin()
        kept = 0
        while True:
            summary = session.step()
            if summary is None:
                break
            self.assertNotIn("rows", summary["step"])
            self.assertFalse([line for line in summary["step"]["lines"] if line["level"] == "row"])
            if summary["accepted"]:
                kept += 1
                self.assertEqual(summary["reasons"]["errors"], [], summary["iteration"])
                self.assertEqual(summary["reasons"]["iteration"], summary["iteration"])
        self.assertEqual(kept, 13)


class Synthetic(unittest.TestCase):
    """3. One synthetic change of the June step 6 entry (or record) per check."""

    def setUp(self):
        self.run, self.records, keep = load(FIX)
        self.keep = copy.deepcopy(keep[6])
        self.header = header_of(FIX, self.records, self.run, 6)

    def out(self, record=None):
        return rows_of(FIX, self.records, self.run, self.keep, 6, record=record)

    def row(self, out, ref):
        return next(r for r in out["rows"] if r["ref"] == ref)

    def assert_row_error(self, code):
        out = self.out()
        self.assertEqual(out["rows"], [])
        self.assertIn(code, [e["code"] for e in out["errors"]], out["errors"])
        self.assertEqual(out["notice"], "The reasons of this step could not be checked.")
        return out

    def test_tie_without_moves(self):
        # synthetic: keeping 18 costs what the recorded answer costs, and moves no other flight
        k = self.keep["flights"]["18"]["keep"]
        set_costs(k, **self.keep["factual"]["costs"])
        k["moves"], k["others"] = [], []
        r = self.row(self.out(), "flight:18")
        self.assertEqual((r["kind"], r["class"]), ("keep_tie", "tie"))
        self.assertEqual(r["text"], "Keeping flight 18 as it was is exactly as good on every criterion of this step; "
                                    "the solver picked one of the equally good answers.")

    def tie_with_move(self, both_costs):
        # synthetic: keeping 18 is a tie that moves flight 17 (and nothing else); keeping 18 and 17 costs both_costs
        k = self.keep["flights"]["18"]["keep"]
        set_costs(k, **self.keep["factual"]["costs"])
        k["moves"], k["others"] = [17], [17]
        both = copy.deepcopy(k)
        both.pop("moves"), both.pop("others"), both.pop("layout_differs")
        set_costs(both, **both_costs)
        self.keep["flights"]["18"]["keep_both"] = dict(both, flight=17)

    def test_tie_other(self):
        self.tie_with_move({"overload": 23})
        r = self.row(self.out(), "flight:18")
        self.assertEqual((r["kind"], r["class"]), ("keep_tie_other", "tie"))
        self.assertEqual(r["text"], "Flight 18 or flight 17 had to change: keeping both as they were would leave a "
                                    "total overload of 23 instead of 22. Keeping flight 18 as it was and changing "
                                    "flight 17 is exactly as good; the solver picked one of the equally good answers.")
        self.assertIn("keep_moves(6,18,17)", r["source"])
        self.assertEqual([x["ref"] for x in r["refs"]], ["flight:18", "flight:17"])

    def test_tie_other_by_delay(self):
        self.tie_with_move({"delay": 30})
        self.assertIn("would have the same total overload and a net arrival delay of 30 instead of 27 time periods, "
                      "summed over the flights of this step. Keeping flight 18",
                      self.row(self.out(), "flight:18")["text"])

    def test_tie_other_impossible(self):
        self.tie_with_move({})
        self.keep["flights"]["18"]["keep_both"] = {"feasible": False, "optimal": True, "costs": None,
                                                  "clingo_cost": [], "chosen_paths": None, "chosen_config": None,
                                                  "flight": 17}
        self.assertIn("keeping both as they were is not possible in this step",
                      self.row(self.out(), "flight:18")["text"])

    def test_tie_with_equal_keep_both_is_a_plain_tie(self):
        self.tie_with_move({})
        self.assertEqual(self.row(self.out(), "flight:18")["kind"], "keep_tie")

    def test_tie_that_moves_the_layout_too_is_a_plain_tie(self):
        self.tie_with_move({"overload": 23})
        self.keep["flights"]["18"]["keep"]["layout_differs"] = True
        self.assertEqual(self.row(self.out(), "flight:18")["kind"], "keep_tie")

    def test_tie_without_keep_both_is_incomplete(self):
        self.tie_with_move({"overload": 23})
        self.keep["flights"]["18"]["keep_both"] = None
        self.assert_row_error("incomplete")

    def test_better_foil(self):
        # synthetic: keeping 18 would leave a total overload of 21 (better than the recorded answer)
        set_costs(self.keep["flights"]["18"]["keep"], overload=21)
        self.assert_row_error("not_optimal")

    def record_with(self, **sub):
        """synthetic: record 6 with fields of its sub-problem 0 replaced"""
        record = copy.deepcopy(self.records[6])
        record["subproblems"][0].update(sub)
        return record

    def assert_record_row_error(self, record, code):
        out = self.out(record)
        self.assertEqual(out["rows"], [])
        self.assertIn(code, [e["code"] for e in out["errors"]], out["errors"])
        self.assertEqual(out["notice"], "The reasons of this step could not be checked.")

    def test_factual_mismatch(self):
        # synthetic: the record's clingo cost vector differs from the factual re-solve of the stored line
        self.assert_record_row_error(self.record_with(clingo_cost=[99, 99, 99, 99, 99]), "factual_mismatch")
        # the stored line's own copy of the recorded vector is not what is checked
        self.keep["recorded_cost"] = [99, 99, 99, 99, 99]
        self.assertEqual(self.out()["errors"], [])

    def test_keep_stale(self):
        # synthetic: the record chose other paths or another layout than the factual of the stored line (a line made
        # for an earlier run of the folder), with the same clingo cost vector
        paths = {f: 0 for f in self.records[6]["subproblems"][0]["chosen_paths"]}
        self.assert_record_row_error(self.record_with(chosen_paths=paths), "keep_stale")
        self.assert_record_row_error(self.record_with(chosen_config=0), "keep_stale")
        self.assertFalse(keep_matches(self.record_with(chosen_paths=paths), self.keep))
        self.assertFalse(keep_matches(self.record_with(chosen_config=0), self.keep))
        self.assertFalse(keep_matches(self.record_with(clingo_cost=[99, 99, 99, 99, 99]), self.keep))
        self.assertTrue(keep_matches(self.records[6], self.keep))
        # the review's probe: every path 0 and the clingo vector 99 at once
        out = self.out(self.record_with(chosen_paths=paths, clingo_cost=[99, 99, 99, 99, 99]))
        self.assertEqual(out["rows"], [])
        self.assertTrue({"factual_mismatch", "keep_stale"} <= {e["code"] for e in out["errors"]}, out["errors"])

    def test_missing_level_is_no_mismatch(self):
        # synthetic: a step that offers layout 0 only: clingo drops the level of the layout rank from both vectors
        self.keep["factual"]["clingo_cost"] = self.keep["factual"]["clingo_cost"][:4]
        out = self.out(self.record_with(clingo_cost=list(self.keep["factual"]["clingo_cost"])))
        self.assertEqual(out["errors"], [])
        self.assertEqual(len(out["rows"]), 4)

    def test_keep_scope(self):
        # synthetic: the trace says the total overload after step 6 is 23 (the factual says 22)
        record = copy.deepcopy(self.records[6])
        record["objectives"]["OVERLOAD"] = 23
        out = self.out(record)
        self.assertEqual(out["rows"], [])
        self.assertIn("keep_scope", [e["code"] for e in out["errors"]])

    def test_not_offered(self):
        # synthetic: no path of 18 equals its (recorded) current trajectory
        self.keep["flights"]["18"]["unchanged_paths"] = None
        self.keep["flights"]["18"]["keep"] = None
        r = self.row(self.out(), "flight:18")
        self.assertEqual((r["kind"], r["class"]), ("not_offered", "rule"))
        self.assertEqual(r["text"], "The current trajectory of flight 18 was not among the options of this step, so "
                                    "every answer changed it.")

    def test_unknown_current_is_incomplete(self):
        # synthetic: as above, and the current trajectory of 18 is not recorded
        item = self.keep["flights"]["18"]
        item.update(unchanged_paths=None, keep=None, current_recorded=False)
        self.assert_row_error("incomplete")

    def test_keep_infeasible(self):
        # synthetic: no answer of the step keeps 18 on its current trajectory
        self.keep["flights"]["18"]["keep"] = {"feasible": False, "optimal": True, "costs": None, "clingo_cost": [],
                                              "chosen_paths": None, "chosen_config": None}
        r = self.row(self.out(), "flight:18")
        self.assertEqual(r["kind"], "keep_infeasible")
        self.assertEqual(r["text"], "No answer of this step keeps flight 18 on its current trajectory.")

    def test_changed_current(self):
        # synthetic: 18 marked as getting its current trajectory although the trace lists it as changed
        self.keep["flights"]["18"]["chosen_current"] = True
        self.assert_row_error("changed_current")

    def test_row_missing(self):
        # synthetic: the entry of 18 is missing
        del self.keep["flights"]["18"]
        self.assert_row_error("row_missing")

    def test_unproven_foil(self):
        # synthetic: the answer that keeps 18 is not proven optimal
        self.keep["flights"]["18"]["keep"]["optimal"] = False
        self.assert_row_error("foil_unproven")

    def test_foil_cost_mismatch(self):
        # synthetic: the clingo vector of the answer that keeps 18 does not match its costs
        self.keep["flights"]["18"]["keep"]["clingo_cost"] = [24, 26, 31, 1, 1]
        self.assert_row_error("foil_cost_mismatch")

    def test_two_subproblems(self):
        # synthetic: the record of step 6 has two sub-problems. The step facts refuse such a record (explain_rows
        # raises, a session reports reasons_failed); reasons.lp checks it too, on the count read from the record
        record = copy.deepcopy(self.records[6])
        record["subproblems"].append(copy.deepcopy(record["subproblems"][0]))
        with self.assertRaises(ValueError):
            self.out(record)
        previous, rejected = reasons.step_context(self.records, 6)
        facts = reasons.step_facts(self.records[6], previous=previous, run=self.run, trace_folder=FIX,
                                   rejected_before=rejected)
        atoms, _, _ = reasons.solve(facts + reasons.keep_facts(self.keep, self.run, record))
        self.assertIn("multi_subproblem(6)", [str(a.arguments[0]) for a in atoms if a.name == "row_error"])
        self.assertFalse(keep_matches(record, self.keep))

    def test_layout_mismatch(self):
        # synthetic: the recorded split goes with layout 0
        self.keep["layout"] = {"chosen_config": 0, "keep": None, "alt": None}
        self.assert_row_error("layout_mismatch")

    def test_layout_decided_by_sectors(self):
        # synthetic: the current layout differs only in the open sectors (no template: a row check)
        set_costs(self.keep["layout"]["keep"], overload=22, delay=27, sectors=32)
        self.assert_row_error("layout_unexpected")

    def test_forged_overload_before(self):
        # synthetic: OVERLOAD-BEFORE 29 (step 5 left 28): a header error; no rows, no notice
        record = copy.deepcopy(self.records[6])
        record["objectives"]["OVERLOAD-BEFORE"] = 29
        out = self.out(record)
        self.assertEqual(out["rows"], [])
        self.assertEqual([e["code"] for e in out["errors"]], ["record_error"])
        self.assertIsNone(out["notice"])
        self.assertIn("before_mismatch", [e["code"] for e in header_of(FIX, self.records, self.run, 6, record)["errors"]])

    def test_negative_delay(self):
        # synthetic: the recorded answer has a net arrival delay of 0, the answer that keeps 18 one of -2
        set_costs(self.keep["factual"], delay=0)
        record = self.record_with(clingo_cost=list(self.keep["factual"]["clingo_cost"]))
        set_costs(self.keep["flights"]["18"]["keep"], delay=-2)
        self.assertIn("but the net arrival delay of the flights of this step would be -2 instead of 0 time periods",
                      self.row(self.out(record), "flight:18")["text"])

    def test_plain_delay_words(self):
        # synthetic: a run that scores late arrivals only
        self.run = dict(self.run, arrival_delay_metric="floored")
        self.assertIn("but the arrival delay of the flights of this step would be 26 instead of 27 time periods",
                      self.row(self.out(), "flight:18")["text"])

    def test_leg_after_a_kept_flight(self):
        # synthetic: 19 is not in the trace's flight changes, its later leg 26 is
        record = copy.deepcopy(self.records[6])
        del record["flight_changes"]["19"]
        self.keep["flights"]["19"]["changed"] = False
        self.keep["flights"]["19"]["chosen_current"] = True
        r = self.row(rows_of(FIX, self.records, self.run, self.keep, 6, record=record), "flight:26")
        self.assertEqual(r["text"], "Flies after flight 19 on the same aircraft; its trajectory follows the option "
                                    "chosen for flight 19.")

    def program_without(self, pattern):
        text = reasons.program()
        changed = re.sub(pattern, "", text, flags=re.S)
        self.assertNotEqual(changed, text)
        return changed

    def assert_rows_dropped_header_kept(self, program, code):
        with mock.patch.object(reasons, "program", lambda: program):
            out = self.out()
            header = header_of(FIX, self.records, self.run, 6)
        self.assertEqual(out["rows"], [])
        self.assertIn(code, [e["code"] for e in out["errors"]])
        self.assertEqual(header, self.header)

    def test_kind_without_template(self):
        self.assert_rows_dropped_header_kept(self.program_without(r"template\(keep_overload, developer, [^\n]*\n"),
                                             "no_template")

    def test_row_without_source(self):
        self.assert_rows_dropped_header_kept(
            self.program_without(r"% sources: the numbers each line states.*?(?=% -{10,} row checks)"), "no_source")

    def test_wrong_placeholder(self):
        program = reasons.program().replace("instead of {chosen}{gain}.{alt}", "instead of {chosen}{gian}.{alt}")
        self.assertIn("{gian}", program)
        self.assert_rows_dropped_header_kept(program, "render_error")


class Templates(unittest.TestCase):
    """4. Every row kind and clause has a template; all texts are ASCII without double quote, backslash, apostrophe."""

    def test_templates(self):
        _, templates, _ = reasons.solve("")
        for kind, text in templates.items():
            self.assertTrue(text.isascii(), kind)
            for ch in ('"', "\\", "'"):
                self.assertNotIn(ch, text, kind)
        lp = (XAI_DIR / "reasons.lp").read_text()
        kinds = set(re.findall(r"row_class\(([a-z]\w*),[a-z]\w*\)\.", lp))
        clauses = set(re.findall(r"(?:gain|both)_clause\([a-z]\w*,([a-z]\w*)\)\.", lp)) | {"alt_flight", "alt_layout",
                                                                                 "both_infeasible", "words_net",
                                                                                 "words_plain"}
        self.assertGreaterEqual(len(kinds), 15)
        for kind in kinds | clauses:
            self.assertIn(kind, templates, kind)


class AspLayer(unittest.TestCase):
    """5. Shown atoms of reasons.lp + reasons_text.lp on the step-6 facts with the keep facts (hand-reviewed)."""

    @staticmethod
    def expected():
        return {line.strip().rstrip(".") for line in (FIX / "step06_keep_expected.lp").read_text().splitlines()
                if line.strip() and not line.startswith("%")}

    def test_facts_are_current(self):
        run, records, keep = load(FIX)
        previous, rejected = reasons.step_context(records, 6)
        facts = reasons.step_facts(records[6], previous=previous, run=run, trace_folder=FIX, rejected_before=rejected)
        self.assertEqual(facts + reasons.keep_facts(keep[6], run, records[6]), (FIX / "step06_keep_facts.lp").read_text())

    def test_clingo_562(self):
        atoms, _, _ = reasons.solve((FIX / "step06_keep_facts.lp").read_text())
        self.assertEqual({str(a) for a in atoms}, self.expected())

    def test_clingo_580(self):
        if not CLINGO_58.exists():
            self.skipTest("clingo 5.8.0 (aspaeroflow-xai env) not installed")
        import subprocess
        out = subprocess.run([str(CLINGO_58), "--warn=none", "--outf=2", "0", str(XAI_DIR / "reasons.lp"),
                              str(XAI_DIR / "reasons_text.lp"), str(FIX / "step06_keep_facts.lp")],
                             capture_output=True, text=True, timeout=60)
        result = json.loads(out.stdout)
        self.assertIn("5.8", result["Solver"])
        self.assertEqual(result["Models"]["Number"], 1)
        self.assertEqual(set(result["Call"][0]["Witnesses"][0]["Value"]), self.expected())


class Live(unittest.TestCase):
    """6. The 10-flight run: keep contrasts of every kept step, the dialog's keep card equals the row line."""

    @classmethod
    def setUpClass(cls):
        _, _, cls.trace_dir = shared_run.get()
        cls.trace = TraceReader(cls.trace_dir)

    def test_rows_and_cards_agree(self):
        run = self.trace.run
        records = self.trace.iterations
        seen = set()
        for n in self.trace.accepted():
            ex = IterationExplainer(self.trace, n)
            entry = ex.keep_contrasts()
            self.assertEqual(entry["factual"]["clingo_cost"], entry["recorded_cost"], n)
            out = rows_of(self.trace_dir, records, run, entry, n)
            self.assertEqual(out["errors"], [], n)
            self.assertEqual(len([r for r in out["rows"] if r["ref"].startswith("flight:")]),
                             len(records[n]["flight_changes"]))
            for r in out["rows"]:
                seen.add(r["class"])
                if not r["ref"].startswith("flight:") or r["class"] in ("rule",):
                    continue
                f = int(r["ref"].split(":")[1])
                card = next(c for c in ex.why_flight(f)["contrasts"] if c["label"] == f"keeping flight {f} as it was")
                keep = entry["flights"][str(f)]["keep"]
                self.assertEqual(card["costs"], keep["costs"], (n, f))
                if r["class"] in ("capacity", "cost"):
                    self.assertEqual(card["deciding_level"], KIND_LEVEL[r["kind"]], (n, f))
                else:
                    self.assertIsNone(card["deciding_level"], (n, f))
                # the flights the card names as changed in that answer and unchanged in the recorded one = moves
                named = {int(g) for g in re.findall(r"flight (\d+)", card["text"].split(" In that answer ")[1])} \
                    if " In that answer " in card["text"] else set()
                unchanged = {g for g in named if ex.chosen_current(g)}
                self.assertEqual(unchanged, set(keep["moves"]), (n, f, card["text"]))
        self.assertTrue(seen)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = []
            for n in self.trace.accepted():
                entry = IterationExplainer(self.trace, n).keep_contrasts()
                append_keep(Path(tmp), entry)
                entries.append(json.loads(json.dumps(entry)))
            self.assertEqual(read_keep(Path(tmp)), {e["iteration"]: e for e in entries})
            # a stored line used by a fresh explainer gives the same dialog card
            n = self.trace.accepted()[0]
            ex = IterationExplainer(self.trace, n)
            ex.load_keep(entries[0])
            f = next((int(f) for f, item in entries[0]["flights"].items() if item["keep"]), None)
            if f is not None:
                card = next(c for c in ex.why_flight(f)["contrasts"] if c["label"] == f"keeping flight {f} as it was")
                self.assertEqual(card["costs"], entries[0]["flights"][str(f)]["keep"]["costs"])
            # a half-written last line is skipped, two different lines for one step are both left out
            path = Path(tmp) / FILE_NAME
            with open(path, "a") as fh:
                fh.write('{"iteration": 99')
            self.assertEqual(len(read_keep(Path(tmp))), len(entries))
            with open(path, "a") as fh:
                fh.write("\n" + json.dumps(dict(entries[0], seconds=-1)) + "\n")
            self.assertNotIn(entries[0]["iteration"], read_keep(Path(tmp)))


@unittest.skipUnless(os.environ.get("XAI_CE7_FULL"), "set XAI_CE7_FULL to a folder with ce7_trace and K30_trace_ce7_tg4")
class Recompute(unittest.TestCase):
    """7. The two fixture files equal a new computation on the full traces (copies in a temporary folder)."""

    def test_recompute(self):
        root = Path(os.environ["XAI_CE7_FULL"])
        for name, fixture in (("ce7_trace", FIX), ("K30_trace_ce7_tg4", TG4)):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / name
                shutil.copytree(root / name, folder)
                (folder / FILE_NAME).unlink(missing_ok=True)
                new = precompute(folder)
                old = read_keep(fixture)
                strip = lambda d: {n: {k: v for k, v in e.items() if k != "seconds"} for n, e in d.items()}
                self.assertEqual(strip(new), strip(old), name)


def _tree(folder):
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*")}


class Sessions(unittest.TestCase):
    """8. Replay with and without the file, and the reasons worker of the service, on the 10-flight trace."""

    @classmethod
    def setUpClass(cls):
        _, _, cls.trace_dir = shared_run.get()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.with_file = Path(cls.tmp.name) / "with_file"
        shutil.copytree(cls.trace_dir, cls.with_file)
        precompute(cls.with_file)
        cls.without = Path(cls.tmp.name) / "without"
        shutil.copytree(cls.trace_dir, cls.without)
        (cls.without / FILE_NAME).unlink(missing_ok=True)

    @classmethod
    def tearDownClass(cls):
        for p in [cls.without, *cls.without.rglob("*")]:
            os.chmod(p, stat.S_IRWXU)
        cls.tmp.cleanup()

    def summaries(self, session):
        session.begin()
        out = []
        while True:
            s = session.step()
            if s is None:
                return out
            out.append(s)

    def test_replay_with_file(self):
        summaries = self.summaries(ReplaySession(self.with_file))
        self.assertTrue([s for s in summaries if s["accepted"]])
        for s in summaries:
            if s["accepted"]:
                self.assertEqual(s["reasons"]["errors"], [], s["iteration"])
            else:
                self.assertNotIn("reasons", s)

    def test_stale_line_is_not_used(self):
        # synthetic: the line of the first kept step says the factual chose path 0 for every flight (a line left
        # from an earlier run of the folder); a replay drops it and computes it again, precompute replaces it
        folder = Path(self.tmp.name) / "stale"
        shutil.copytree(self.with_file, folder)
        lines = read_keep(folder)
        n = min(lines)
        good = copy.deepcopy(lines[n])
        lines[n]["factual"]["chosen_paths"] = {f: 0 for f in lines[n]["factual"]["chosen_paths"]}
        (folder / FILE_NAME).write_text("".join(json.dumps(lines[m], sort_keys=True) + "\n" for m in sorted(lines)))
        session = ReplaySession(folder)
        summaries = {s["iteration"]: s for s in self.summaries(session)}
        self.assertNotIn("reasons", summaries[n])
        expected = {s["iteration"]: s for s in self.summaries(ReplaySession(self.with_file))}
        self.assertEqual(session.reasons(n), expected[n]["reasons"])
        for m in lines:
            if m != n:
                self.assertEqual(summaries[m]["reasons"], expected[m]["reasons"])
        new = precompute(folder)
        strip = lambda e: {k: v for k, v in e.items() if k != "seconds"}
        self.assertEqual(strip(new[n]), strip(good))
        self.assertEqual(strip(read_keep(folder)[n]), strip(good))

    def test_replay_without_file_is_read_only(self):
        for p in [*self.without.rglob("*"), self.without]:
            os.chmod(p, stat.S_IRUSR | stat.S_IXUSR if p.is_dir() else stat.S_IRUSR)
        before = _tree(self.without)
        session = ReplaySession(self.without)
        summaries = self.summaries(session)
        self.assertFalse([s for s in summaries if "reasons" in s])
        expected = {s["iteration"]: s for s in self.summaries(ReplaySession(self.with_file))}
        for s in summaries:
            if s["accepted"]:
                got = session.reasons(s["iteration"])
                self.assertEqual(got, expected[s["iteration"]]["reasons"])
        with self.assertRaises(KeyError):
            session.reasons(10_000)
        self.assertEqual(_tree(self.without), before)


class Service(unittest.TestCase):
    """8. The service sends a reasons event after each kept iteration whose summary has none; the worker stops."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import uvicorn
        spec = importlib.util.spec_from_file_location("session_service", REPO / "07_heuristic_controller" / "session_service.py")
        cls.service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.service)
        cls.tmp = tempfile.TemporaryDirectory()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cls.app = cls.service.create_app(HERE / "fixtures", Path(cls.tmp.name))
        cls.server = uvicorn.Server(uvicorn.Config(cls.app, host="127.0.0.1", port=port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.base + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        _, _, trace_dir = shared_run.get()
        cls.replay = Path(cls.tmp.name) / "replay"
        shutil.copytree(trace_dir, cls.replay)
        (cls.replay / FILE_NAME).unlink(missing_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.tmp.cleanup()

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def events(self, sid, until, seconds=60):
        deadline = time.time() + seconds
        while time.time() < deadline:
            req = urllib.request.Request(f"{self.base}/sessions/{sid}/events")
            got = []
            with urllib.request.urlopen(req, timeout=30) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith(":"):
                        break                      # keep-alive: everything so far has been sent
                    if line.startswith("data: "):
                        got.append(json.loads(line[6:]))
            if until(got):
                return got
            time.sleep(0.2)
        return got

    def check(self, sid, events):
        iterations = [e["data"] for e in events if e["type"] == "iteration"]
        kept = [d["iteration"] for d in iterations if d["accepted"]]
        self.assertTrue(kept)
        got = [e["data"] for e in events if e["type"] == "reasons"]
        self.assertEqual(sorted(d["iteration"] for d in got), sorted(kept))
        for d in got:
            self.assertEqual(d["errors"], [], d)
            self.assertTrue(d["rows"])
        # each reasons event comes after the iteration event of its step
        order = [(e["type"], e["data"].get("iteration")) for e in events if e["type"] in ("iteration", "reasons")]
        for n in kept:
            self.assertLess(order.index(("iteration", n)), order.index(("reasons", n)))

    def test_replay_without_file(self):
        reply = self.call("POST", "/sessions", {"replay": str(self.replay)})
        self.assertIs(reply["reasons_worker"], True)
        sid = reply["id"]
        while self.call("POST", f"/sessions/{sid}/step")["status"] != "finished":
            pass
        kept = [n for n, r in TraceReader(self.replay).iterations.items() if r["accepted"]]
        events = self.events(sid, lambda ev: len([e for e in ev if e["type"] == "reasons"]) >= len(kept))
        self.check(sid, events)
        self.assertFalse((self.replay / FILE_NAME).exists())     # a replay never writes its folder
        handle = self.app.state.sessions[sid] if hasattr(self.app.state, "sessions") else None
        self.call("DELETE", f"/sessions/{sid}")
        if handle is not None:
            handle.reasons_worker.join(timeout=5)
            self.assertFalse(handle.reasons_worker.is_alive())

    def test_live(self):
        sid = self.call("POST", "/sessions", {"instance": shared_run.INSTANCE.name})["id"]
        self.call("POST", f"/sessions/{sid}/run", {"pace_ms": 0})
        for _ in range(600):
            if self.call("GET", f"/sessions/{sid}")["status"] == "finished":
                break
            time.sleep(0.1)
        records = TraceReader(Path(self.tmp.name) / sid).iterations
        kept = [n for n, r in records.items() if r["accepted"]]
        events = self.events(sid, lambda ev: len([e for e in ev if e["type"] == "reasons"]) >= len(kept))
        self.check(sid, events)
        self.assertEqual(sorted(read_keep(Path(self.tmp.name) / sid)), sorted(kept))   # the live folder has the file


if __name__ == "__main__":
    unittest.main()
