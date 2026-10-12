"""K24: times and durations in clock words (xai/time_words.py), the time scale of the STEP payload, and no old
words ("time 11", "N time periods") left in the explanation texts of a 15-minute trace."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai import reasons  # noqa: E402
from src.aspaeroflow.xai.keep import read_keep  # noqa: E402
from src.aspaeroflow.xai.plan import clock  # noqa: E402
from src.aspaeroflow.xai.time_words import (clock_text, duration_text, granularity, shift_text, time_steps,  # noqa: E402
                                            time_text)

TG4 = HERE / "fixtures" / "K30_TG4"
OLD_WORDS = re.compile(r"time periods?|\bat time \d|\bfrom time \d")


class Words(unittest.TestCase):
    def test_times(self):
        self.assertEqual(time_text(37, 4), "time step 37 (09:15)")
        self.assertEqual(time_text(11, 1), "time step 11 (11:00)")
        self.assertEqual(time_text(101, 4), "time step 101 (+1d 01:15)")
        self.assertEqual(clock_text(0, 4), "00:00")
        self.assertEqual(clock_text(3, 8), "00:22")            # whole minutes, never rounded up

    def test_same_clock_as_the_plan(self):
        for g in (1, 2, 4, 8):
            for t in (0, 1, 7, 24 * g - 1, 24 * g, 61):
                self.assertEqual(clock_text(t, g), clock(t, 60 / g, 24 * g), (g, t))

    def test_durations(self):
        self.assertEqual(duration_text(1, 4), "15 min")
        self.assertEqual(duration_text(4, 4), "1 h")
        self.assertEqual(duration_text(5, 4), "1 h 15 min")
        self.assertEqual(duration_text(0, 4), "0 min")
        self.assertEqual(duration_text(-2, 4), "-30 min")
        self.assertEqual(duration_text(27, 1), "27 h")
        self.assertEqual(shift_text(3, 4), "+45 min")
        self.assertEqual(shift_text(-1, 4), "-15 min")
        self.assertEqual(time_steps(1), "1 time step")
        self.assertEqual(time_steps(3), "3 time steps")

    def test_granularity(self):
        self.assertEqual(granularity({"timestep_granularity": 4}), 4)
        self.assertEqual(granularity({}), 1)
        self.assertEqual(granularity(None), 1)


class Texts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_info = json.loads((TG4 / "run.json").read_text())
        cls.records = {}
        for line in (TG4 / "trace.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cls.records[r["iteration"]] = r
        cls.keep = read_keep(TG4)

    def test_step_payload_has_the_scale(self):
        start = reasons.explain_start(self.run_info)
        self.assertEqual((start["minutes_per_step"], start["steps_per_day"]), (15, 96))

    def test_no_old_words(self):
        """Every header line and row line of the 15-minute fixture uses clock times and minutes."""
        seen_time = seen_minutes = False
        for n, record in sorted(self.records.items()):
            previous, rejected = reasons.step_context(self.records, n)
            step = reasons.explain_step(record, previous=previous, run=self.run_info, trace_folder=TG4,
                                        rejected_before=rejected)
            texts = [line["text"] for line in step["lines"]]
            if n in self.keep:
                rows = reasons.explain_rows(record, self.keep[n], previous=previous, run=self.run_info, trace_folder=TG4,
                                            rejected_before=rejected)
                texts += [row["text"] for row in rows["rows"]]
            for text in texts:
                with self.subTest(step=n, text=text[:60]):
                    self.assertIsNone(OLD_WORDS.search(text))
                seen_time |= bool(re.search(r"time step \d+ \(\d\d:\d\d\)", text))
                seen_minutes |= bool(re.search(r"\d+ (min|h)\b", text))
        self.assertTrue(seen_time and seen_minutes)


if __name__ == "__main__":
    unittest.main()
