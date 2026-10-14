"""S8 4c next-step preview (xai/preview.py, its hook in the optimizer, and the selection code it shares with
iteration_step.py).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)

The preview made after step n must equal what step n+1 then does. Proven three ways: the hook's own check in every
record (the preview against the cell and flights optimization_step selected), prove() against the stored trace
(record n+1's hotspot, taken flights, chosen paths and lp file), and a re-run of the practice study trace (recorded
before the selection was moved into select_flights/step_scope) that must reproduce it exactly.
"""
import atexit
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai import preview as preview_mod  # noqa: E402
from src.aspaeroflow.xai.session import OptimizerSession  # noqa: E402
from src.aspaeroflow.main_loop_components.iteration_step import select_flights  # noqa: E402

FIXTURES = HERE / "fixtures"
EAST_ASIA = FIXTURES / "EAST-ASIA-3x3-V2_0000010_SEED150699"
PRACTICE = FIXTURES / "CE-NAVPT-PRACTICE"
PRACTICE_TRACE = FIXTURES / "NAVPT_PRACTICE_TRACE"          # the practice study trace CE-NAVPT-PRACTICE_S15_2
BASE = {"timestep_granularity": 4, "seed": 11904657, "number_threads": 4, "max_number_sectors": 100000}

_tmp = Path(tempfile.mkdtemp(prefix="aspaeroflow-preview-test-"))
atexit.register(shutil.rmtree, _tmp, True)
_runs = {}


def run(name, instance, options, steps=300):
    """A live run with the trace on (cached per name): (trace folder, preview records, session)."""
    if name not in _runs:
        folder = _tmp / name
        session = OptimizerSession(instance, folder, options)
        session.begin()
        for _ in range(steps):
            if session.status == "finished" or session.step() is None:
                break
        if session.status != "finished":
            session.app.finish()                 # closes the files of a run cut short
        _runs[name] = (folder, preview_mod.read(folder / preview_mod.FILE)[1], session)
    return _runs[name]


def trace_of(folder):
    return [json.loads(x) for x in (Path(folder) / "trace.jsonl").read_text().splitlines() if x.strip()]


class Proof(unittest.TestCase):
    """Every preview equals the next step, by the hook's check and by prove() against the trace."""

    def check_run(self, folder, records, finished_reason=None, cut_short=False):
        trace = trace_of(folder)
        self.assertEqual(sorted(records), list(range(len(trace) + 1)))
        for n, r in records.items():
            self.assertNotIn("error", r, r)
            if n > 0:
                self.assertTrue(r["check"]["ok"], (n, r["check"]))
                self.assertEqual(r["check"]["made_after"], n - 1)
        rows = preview_mod.prove(folder, records)
        bad = [row for row in rows if not row["ok"]]
        if cut_short:     # the run was stopped by the test: the last preview names a step that was never made
            self.assertEqual([row["after"] for row in bad], [len(trace)])
        else:
            self.assertEqual(bad, [])
            self.assertTrue(records[len(trace)]["finished"])
            self.assertEqual(records[len(trace)]["reason"], finished_reason)
        return trace, rows

    def test_east_asia(self):
        folder, records, _ = run("east_asia", EAST_ASIA, {})
        self.check_run(folder, records, "no_overload")

    def test_practice_with_its_study_settings(self):
        folder, records, _ = run("practice", PRACTICE, BASE | {"max_delay_per_iteration": 9})
        self.check_run(folder, records, "no_overload")

    def test_rejections_retry_the_cell_and_cut_to_one_flight(self):
        """Every step rejected: the same cell again ("retry"), the streak counts, and from 10 rejections on one
        flight goes to the solver."""
        folder, records, _ = run("rejections", PRACTICE, BASE | {"max_delay_per_iteration": 1,
                                                                 "max_explored_vertices": 1,
                                                                 "number_capacity_management_configs": 1}, steps=13)
        trace, _rows = self.check_run(folder, records, cut_short=True)
        self.assertTrue(all(not r["accepted"] for r in trace))
        for n in range(1, 14):
            r = records[n]
            self.assertTrue(r["streak"]["retry"])
            self.assertEqual(r["streak"]["rejections"], n)
            self.assertEqual(r["hotspot"], records[0]["hotspot"])
            self.assertEqual(r["rule"]["max_aircraft"], 1 if n >= 10 else 2)
            self.assertEqual(len(r["solver"]), 1 if n >= 10 else 2)
            self.assertEqual(r["streak"]["until_one_flight"], max(0, 10 - n))
        self.assertEqual([x["id"] for x in trace[12]["hotspot"]["flights"][:trace[12]["hotspot"]["taken"]]],
                         records[11]["solver"])

    def test_no_progress_ends_finished(self):
        """The optimizer gives up (no further progress possible): the last preview says so."""
        folder, records, session = run("no_progress", EAST_ASIA, BASE | {"max_delay_per_iteration": 3,
                                                                         "max_explored_vertices": 1,
                                                                         "number_capacity_management_configs": 1,
                                                                         "sector_capacity_factor": 1})
        self.assertEqual(session.status, "finished")
        self.assertTrue(session.app.has_overload())
        self.check_run(folder, records, "no_progress")

    def test_sequential_mode(self):
        """Sequential mode skips cells already tried; the preview reads the same record of tried cells."""
        folder, records, _ = run("sequential", PRACTICE, BASE | {"max_delay_per_iteration": 1,
                                                                 "max_explored_vertices": 1,
                                                                 "number_capacity_management_configs": 1,
                                                                 "sequential_execution": "true"}, steps=12)
        self.check_run(folder, records, "no_cell")     # every overloaded cell tried: the optimizer stops
        cells = [(r["hotspot"]["sector"], r["hotspot"]["time"]) for n, r in sorted(records.items()) if "hotspot" in r]
        self.assertEqual(len(set(cells)), len(cells))
        self.assertFalse(any(r["streak"]["retry"] for r in records.values() if "streak" in r))


class Record(unittest.TestCase):

    def test_fields(self):
        folder, records, _ = run("practice", PRACTICE, BASE | {"max_delay_per_iteration": 9})
        trace = trace_of(folder)
        p, r = records[0], trace[0]
        self.assertEqual((p["iteration"], p["kept"], p["next"], p["finished"]), (0, None, 1, False))
        self.assertEqual(p["flights"], r["hotspot"]["flights"])
        self.assertEqual(p["hotspot"]["clock"], f"{p['hotspot']['time'] * 15 // 60:02d}:{p['hotspot']['time'] * 15 % 60:02d}")
        self.assertEqual(p["solver"], [f["id"] for f in p["flights"]][:p["rule"]["taken"]])
        self.assertEqual(p["solver_flights"], sorted(set(p["solver"]) | set(p["later"])))
        self.assertEqual(sorted({f for a in p["aircraft"] for f in a["flights"]}), p["solver_flights"])
        self.assertEqual(p["rule"]["taken"], min(p["rule"]["max_aircraft"], max(2, p["rule"]["overload"]),
                                                 len(p["flights"])))
        # sorted by stored duration, ties to the lower id
        keys = [(f["duration"], f["id"]) for f in p["flights"]]
        self.assertEqual(keys, sorted(keys))
        self.assertIn("check", records[1])
        self.assertNotIn("check", p)

    def test_preview_leaves_the_state_unchanged(self):
        """compute() may run at any time: the optimizer's matrices and its record of tried cells stay as they were."""
        folder = _tmp / "unchanged"
        session = OptimizerSession(PRACTICE, folder, BASE | {"max_delay_per_iteration": 9})
        session.begin()
        session.step()
        dto = session.app.optimization_dto

        def digest():
            h = hashlib.sha256()
            for key in sorted(dto):
                value = dto[key]
                if isinstance(value, np.ndarray):
                    h.update(key.encode() + value.tobytes() + str(value.shape).encode())
                elif isinstance(value, (int, float, bool, str, type(None))):
                    h.update(f"{key}={value!r}".encode())
            h.update(repr(sorted(session.app._sequential_execution_candidate_dict.items())).encode())
            return h.hexdigest()

        before = digest()
        first = preview_mod.compute(session.app)
        second = preview_mod.compute(session.app)
        self.assertEqual(digest(), before)
        self.assertEqual(first, second)
        session.app.finish()

    def test_select_flights_rule(self):
        """The rule on a small hand-made cell: sort by duration (ties to the lower id), take
        min(n_proc * max_rows, max(2, overload)), later flights of the aircraft from the taken flight's start on."""
        fill = -1
        # flights 0..4; at time 2 flights 0, 1, 3, 4 are in sector 7
        instance = np.full((5, 8), fill)
        instance[0, 2:7] = 7       # duration 5
        instance[1, 1:4] = 7       # duration 3
        instance[2, 5:8] = 3       # flight 2: later flight of aircraft 1 (aircraft of flight 1)
        instance[3, 2:5] = 7       # duration 3 (ties with flight 1: lower id first)
        instance[4, 0:3] = 7       # duration 3
        durations = np.array([5, 3, 3, 3, 3])
        airplane_flight = np.array([[0, 0], [1, 1], [1, 2], [2, 3], [3, 4]])
        diff = np.zeros((8, 8), dtype=int)
        diff[7, 2] = -3            # overload 3
        sel = select_flights(2, 7, instance, diff, durations, 2, 1, airplane_flight, fill)
        self.assertEqual(list(sel["cell"]), [1, 3, 4, 0])
        self.assertEqual(list(sel["rows_pool"]), [1, 3])                    # max_rows 2 < overload 3
        self.assertEqual(sorted(int(f) for f in sel["all_potentially_problematic_flight_indices"]), [1, 2, 3])
        sel = select_flights(2, 7, instance, diff, durations, 5, 1, airplane_flight, fill)
        self.assertEqual(list(sel["rows_pool"]), [1, 3, 4])                 # overload 3 < max_rows 5
        diff[7, 2] = -1
        sel = select_flights(2, 7, instance, diff, durations, 5, 1, airplane_flight, fill)
        self.assertEqual(list(sel["rows_pool"]), [1, 3])                    # at least 2


class StudyTrace(unittest.TestCase):
    """Re-runs: the practice study trace, recorded before the selection was shared with the preview, and a trace of
    this code."""

    def test_rerun_reproduces_and_proves_the_practice_study_trace(self):
        """The fixture leaves step 1's lp file out (README), so that file and its lp check are the only gaps; the side
        file is then not written. The four study traces themselves are proven in full by the re-runs that made their
        side files (analysis/xai_study/18_sprint8_20261014/preview_data)."""
        out = _tmp / "side_practice" / "preview.jsonl"
        summary = preview_mod.rerun(PRACTICE_TRACE, out, data_dir=PRACTICE)
        self.assertEqual(summary["differences"], ["lp/: 2 files stored, 3 in the re-run"])
        self.assertEqual(summary["proof"][0]["differences"], ["lp file of step 1 missing"])
        self.assertTrue(all(row["ok"] for row in summary["proof"][1:]), summary["proof"])
        self.assertIsNone(summary["written"])
        trace = trace_of(PRACTICE_TRACE)
        self.assertEqual(len(summary["proof"]), len(trace) + 1)
        for row, r in zip(summary["proof"], trace):
            self.assertEqual((row["sector"], row["time"]), (r["hotspot"]["sector"], r["hotspot"]["time"]))
        self.assertTrue(summary["proof"][-1]["finished"])

    def test_rerun_writes_the_side_file(self):
        stored, _records, _ = run("east_asia", EAST_ASIA, {})
        out = _tmp / "side_east_asia" / "preview.jsonl"
        summary = preview_mod.rerun(stored, out, data_dir=EAST_ASIA)
        self.assertEqual(summary["differences"], [])
        self.assertTrue(all(row["ok"] for row in summary["proof"]))
        self.assertEqual(summary["written"], str(out))
        header, records = preview_mod.read(out)
        self.assertEqual(header["trace_sha256"], hashlib.sha256((stored / "trace.jsonl").read_bytes()).hexdigest())
        self.assertEqual(records, preview_mod.read(stored / preview_mod.FILE)[1])

    def test_rerun_refuses_a_trace_it_does_not_reproduce(self):
        stored = _tmp / "altered"
        shutil.copytree(PRACTICE_TRACE, stored)
        lines = (stored / "trace.jsonl").read_text().splitlines()
        first = json.loads(lines[0])
        first["hotspot"]["taken"] = 1
        lines[0] = json.dumps(first)
        (stored / "trace.jsonl").write_text("\n".join(lines) + "\n")
        out = _tmp / "altered_out" / "preview.jsonl"
        summary = preview_mod.rerun(stored, out, data_dir=PRACTICE)
        self.assertIsNone(summary["written"])
        self.assertFalse(out.exists())
        self.assertTrue(summary["differences"])
        self.assertFalse(summary["proof"][0]["ok"])


class Serving(unittest.TestCase):

    class _Session:
        def __init__(self, folder, delivered):
            self.folder, self._delivered = folder, delivered

        def delivered(self):
            return [{"iteration": n} for n in range(1, self._delivered + 1)]

    def test_only_delivered_steps_and_matching_side_file(self):
        folder, records, _ = run("practice", PRACTICE, BASE | {"max_delay_per_iteration": 9})
        s = self._Session(folder, 1)
        self.assertEqual(preview_mod.preview_for(s, 0), records[0])
        self.assertEqual(preview_mod.preview_for(s, 1), records[1])
        for n in (2, -1):
            with self.assertRaises(KeyError):
                preview_mod.preview_for(s, n)
        copy = _tmp / "practice_copy"
        shutil.copytree(folder, copy)
        body = (copy / preview_mod.FILE).read_text()
        (copy / preview_mod.FILE).write_text(json.dumps({"header": {"v": 1, "trace_sha256": "0" * 64}}) + "\n" + body)
        with self.assertRaises(KeyError):
            preview_mod.preview_for(self._Session(copy, 1), 0)
        (copy / preview_mod.FILE).unlink()
        with self.assertRaises(KeyError):
            preview_mod.preview_for(self._Session(copy, 1), 0)


if __name__ == "__main__":
    unittest.main()
