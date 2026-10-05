"""Fork-and-continue (xai/fork.py) on the 10-flight fixture.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.aspaeroflow.xai.fork import experiment  # noqa: E402

INSTANCE = HERE / "fixtures" / "EAST-ASIA-3x3-V2_0000010_SEED150699"


class Fork(unittest.TestCase):

    def test_every_fork_ends_without_overload_and_the_base_is_reproduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = experiment(INSTANCE, Path(tmp))
        self.assertEqual(result["base"]["final"]["OVERLOAD"], 0)
        self.assertEqual(len(result["forks"]), result["base"]["final"]["iterations"])
        for f in result["forks"]:
            self.assertEqual(f["final"]["OVERLOAD"], 0)
            if f["forked"]:
                # the alternative is never better than the recorded answer for its own step
                self.assertLessEqual(f["recorded_cost"], f["alternative_cost"])


if __name__ == "__main__":
    unittest.main()
