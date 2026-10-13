"""build_rerun_worklist.py --only-keys keeps exactly the listed units of the source worklist.

The 05_ASP rerun on the fixed exact encoding reruns only the large-family rows that reported a
solution (rerun_lists/05asp_large_with_incumbent.tsv). A listed unit missing from the source must
stop the build, so a typo in the list cannot quietly shrink the rerun.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "06_benchmark_start_script"
HEADER = "problem_dir\ttime_granularity\tregion\tcapacity_level\tn_instances\n"


class TestOnlyKeys(unittest.TestCase):

    def _source(self, tmp: Path) -> Path:
        """A source campaign: one large-family problem with two instances."""
        root = tmp / "instances"
        for name in ("0000010_SEED42", "0000100_SEED42"):
            (root / "P1-PCAP010" / name).mkdir(parents=True)
        (root / "problems.tsv").write_text(HEADER + "P1-PCAP010\t1\tR\tPCAP010\t2\n", encoding="utf-8")
        subprocess.run([sys.executable, str(BENCH / "build_worklist.py"), "--instance-root", str(root),
                        "--folder", "SRC", "--output-root", str(tmp / "output"), "--run-mip", "no",
                        "--max-array-size", "1001"], capture_output=True, text=True, check=True, cwd=BENCH)
        return tmp / "output"

    def _rerun(self, out: Path, keys: Path, folder: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(BENCH / "build_rerun_worklist.py"),
                               "--source-folder", "SRC", "--output-root", str(out), "--folder", folder,
                               "--only-keys", str(keys)], capture_output=True, text=True, cwd=BENCH)

    def test_keeps_exactly_the_listed_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._source(Path(tmp))
            keys = Path(tmp) / "keys.tsv"
            keys.write_text("problem_dir\tinstance\tsystem\n"
                            "P1-PCAP010\t0000100_SEED42\t05_ASP_rp_d_sp\n"
                            "P1-PCAP010\t0000010_SEED42\t05_ASP_rp_dp_sp\n", encoding="utf-8")
            done = self._rerun(out, keys, "RERUN")
            self.assertEqual(done.returncode, 0, done.stderr)
            rows = (out / "RERUN" / "units" / "worklist.tsv").read_text().splitlines()[1:]
            units = [tuple(r.split("\t")[i] for i in (0, 1, 6, 7)) for r in rows]
            # source order, renumbered from 1
            self.assertEqual(units, [("1", "P1-PCAP010", "0000010_SEED42", "05_ASP_rp_dp_sp"),
                                     ("2", "P1-PCAP010", "0000100_SEED42", "05_ASP_rp_d_sp")])

    def test_a_listed_unit_missing_from_the_source_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._source(Path(tmp))
            keys = Path(tmp) / "keys.tsv"
            keys.write_text("problem_dir\tinstance\tsystem\n"
                            "P1-PCAP010\t0000010_SEED42\t05_ASP_rp_dp_sp\n"
                            "P1-PCAP010\t0000999_SEED42\t05_ASP_rp_dp_sp\n", encoding="utf-8")
            done = self._rerun(out, keys, "RERUN")
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("not in the source worklist", done.stderr)
            self.assertFalse((out / "RERUN" / "units" / "worklist.tsv").exists())

    def test_a_wrong_header_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._source(Path(tmp))
            keys = Path(tmp) / "keys.tsv"
            keys.write_text("problem\tinstance\tmethod\nP1-PCAP010\t0000010_SEED42\t05_ASP_rp_dp_sp\n",
                            encoding="utf-8")
            done = self._rerun(out, keys, "RERUN")
            self.assertNotEqual(done.returncode, 0)

    def test_the_shipped_list_is_well_formed(self):
        path = BENCH / "rerun_lists" / "05asp_large_with_incumbent.tsv"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "problem_dir\tinstance\tsystem")
        keys = [tuple(line.split("\t")) for line in lines[1:]]
        self.assertEqual(len(keys), 1011)
        self.assertEqual(len(set(keys)), 1011)
        self.assertTrue(all(k[2] in ("05_ASP_rp_dp_sp", "05_ASP_rp_d_sp") for k in keys))
        self.assertTrue(all("-PCAP" in k[0] for k in keys))       # large family only


if __name__ == "__main__":
    unittest.main()
