"""What the sector card of a step reads from the trace: the hotspot sector before the step and the parts its
navpoints form after it (10-flight run shared with test_xai_trace_and_explanations.py; the trace is only read).

For every kept step with a sector change: the parts after the step are pairwise disjoint and together hold exactly
the navpoints of the sector before; every part id is one of its own navpoints; the part with the hotspot navpoint
keeps the hotspot id; the numbers agree (overload = demand - capacity, for the hotspot and for every part that has
them); the open-sector count rises by at most (k - 1) * (T - t) for a split into k from time t on (window T).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402

# The June study trace (CENTRAL-EUROPE-7x7, 13 kept steps, 5 of them split the hotspot sector). Its run.json predates
# the evaluation_window field; the regenerated run of the same instance and options records 25.
CE7_JUNE = HERE / "fixtures" / "CE7_JUNE_TRACE"
CE7_WINDOW = 25


def sector_change_records(trace_dir):
    """(run info, [(record, objectives before the step)]) for the kept steps whose layout changed."""
    run = json.loads((Path(trace_dir) / "run.json").read_text())
    previous = run["initial_objectives"]
    out = []
    for line in (Path(trace_dir) / "trace.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record["accepted"]:
            continue
        sc = record.get("sector_changes") or {}
        prev, post = sc.get("prev_sector_config") or {}, sc.get("post_sector_config") or {}
        layout = lambda c: sorted((k, sorted(v["vertices"])) for k, v in c.items())  # noqa: E731
        if post and layout(prev) != layout(post):
            out.append((record, previous))
        previous = record["objectives"]
    return run, out


def check_record(test, run, record, before, window=None):
    """The invariants of one step (assertions of `test`); returns a note when the sector count rose less than a split.
    `window` replaces a missing run.json evaluation_window (traces written before it was recorded)."""
    sc, h = record["sector_changes"], record["hotspot"]
    s, t = int(sc["sector_index"]), int(sc["time_index"])
    prev, post = sc["prev_sector_config"], sc["post_sector_config"]
    old = set(prev[str(s)]["vertices"])
    seen = set()
    for key, part in post.items():
        vertices = set(part["vertices"])
        test.assertFalse(vertices & seen, f"part {key} overlaps another part")
        seen |= vertices
        test.assertIn(int(key), vertices, f"part id {key} is none of its navpoints")
    test.assertEqual(seen, old, "parts after the step != navpoints before")
    keeper = [k for k, part in post.items() if s in part["vertices"]]
    test.assertEqual(keeper, [str(s)], "the part with the hotspot navpoint keeps the hotspot id")
    test.assertEqual(h["overload"], h["demand"] - h["capacity"])
    for config in (prev, post):
        for key, part in config.items():
            if "demand" in part or "capacity" in part:
                test.assertEqual(part["overload"], part["demand"] - part["capacity"], f"part {key}")
    k, window = len(post), run.get("evaluation_window", window)
    rise = record["objectives"]["SECTOR-NUMBER"] - before["SECTOR-NUMBER"]
    if window is None:
        return None
    bound = (k - 1) * (window - t)
    test.assertLessEqual(rise, bound, "open sectors rose more than a split from the hotspot time on allows")
    if rise < bound:
        return f"step {record['iteration']}: open sectors +{rise} < (k-1)(T-t) = {bound} (a later split was overwritten)"
    return None


class SectorChanges(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, _, trace_dir = shared_run.get()
        cls.run_info, cls.records = sector_change_records(trace_dir)

    def test_sector_change_invariants(self):
        if not self.records:
            self.skipTest("no sector split in this run")
        for record, before in self.records:
            with self.subTest(iteration=record["iteration"]):
                note = check_record(self, self.run_info, record, before)
                if note:
                    print(note)


class SectorChangesJune(unittest.TestCase):
    """The same invariants on a run with splits, and the exact rise of the open-sector count per split."""

    def test_june_trace(self):
        run, records = sector_change_records(CE7_JUNE)
        self.assertEqual([r["iteration"] for r, _ in records], [1, 2, 3, 6, 9])
        rises = []
        for record, before in records:
            with self.subTest(iteration=record["iteration"]):
                self.assertIsNone(check_record(self, run, record, before, window=CE7_WINDOW))
                rises.append(record["objectives"]["SECTOR-NUMBER"] - before["SECTOR-NUMBER"])
        self.assertEqual(rises, [76, 108, 96, 14, 50])
        step6 = next(r for r, _ in records if r["iteration"] == 6)["sector_changes"]
        self.assertEqual({k: (sorted(v["vertices"]), v["overload"]) for k, v in step6["post_sector_config"].items()},
                         {"21": ([21, 22], -1), "35": ([23, 28, 29, 30, 35, 36], 0)})


if __name__ == "__main__":
    unittest.main()
