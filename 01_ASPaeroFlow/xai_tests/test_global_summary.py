"""Run summary from local explanations (xai/global_summary.py) on the 10-flight fixture.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai.global_summary import summarize  # noqa: E402
from src.aspaeroflow.xai.trace import TraceReader  # noqa: E402

INSTANCE = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"


class GlobalSummary(unittest.TestCase):

    def test_summary_covers_every_step_and_is_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(OPT / "main.py"), f"--data-dir={INSTANCE}",
                            f"--encoding-path={OPT / 'encoding.lp'}", "--save-results=false",
                            f"--xai-trace-dir={tmp}/t"], cwd=OPT.parent, capture_output=True, check=True, timeout=600)
            trace = TraceReader(Path(tmp) / "t")
            s = summarize(trace)
        self.assertEqual(len(s["steps"]), len(trace.iterations))
        self.assertEqual(sum(s["actions"].values()), s["accepted_steps"])
        for st in s["steps"]:
            if st["accepted"]:
                # acceptance requires less overload, so doing nothing always loses on overload
                self.assertEqual(st["deciding_level_vs_nothing"], "overload")
            for lever, v in st["levers"].items():
                self.assertIn(lever, st["action"])
                self.assertEqual(v["needed"], v["level"] is not None)


if __name__ == "__main__":
    unittest.main()
