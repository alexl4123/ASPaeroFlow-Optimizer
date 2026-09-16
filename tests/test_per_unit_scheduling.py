"""The per-unit path must ask the solvers for exactly what the monolithic path asks for.

run_benchmark_units.slurm schedules one (problem, instance, system) unit per job instead of one
problem per job, and merge_benchmark_shards.py glues the results back into the CSVs
run_all_benchmarks.slurm writes. The acceptance condition is that the output does not change, so
what these tests pin is the three places it could:

  * the family rules -- benchmark_families.py must say what run_all_benchmarks.slurm's own bash
    says, or the two paths would run different sets of solvers;
  * the command line -- a unit's argv must equal the monolithic argv for that system, except for
    --results-root, which necessarily points into the unit's shard;
  * the writer -- merge_benchmark_shards.py calls write_result_csvs(), so the order-dependent
    header discovery in the metric CSVs is pinned here to catch a "tidy-up" that would change
    merged and monolithic output together and look like agreement.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "06_benchmark_start_script"
for path in (str(REPO), str(BENCH)):
    if path not in sys.path:
        sys.path.insert(0, path)

import benchmark_families as families                      # noqa: E402
import start_benchmark_caller as caller                    # noqa: E402

SLURM = BENCH / "run_all_benchmarks.slurm"

#: The common flags both paths pass, from run_all_benchmarks.slurm's invocation.
COMMON = ["--output-root=output", "--timestep-granularity=1", "--memory-limit=35",
          "--time-limit=1800", "--experiment-name=P", "--scaling-experiments=0",
          "--wandb-enabled=False", "--results-format=csv.gz"]


def slurm_experiment_blocks():
    """The three EXPERIMENTS=( ... ) arrays of run_all_benchmarks.slurm, in file order.

    Read from the script rather than restated, so that editing one and not the other fails here
    instead of in a campaign.
    """
    text = SLURM.read_text(encoding="utf-8")
    blocks = []
    for raw in re.findall(r"EXPERIMENTS=\(\s*(.*?)\s*\)", text, re.DOTALL):
        flags = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            flags.extend(re.findall(r"--experiment-[a-z0-9-]+=\d", line))
        blocks.append(flags)
    return blocks


class TestFamilyRulesMatchTheSlurmScript(unittest.TestCase):
    def test_three_blocks(self):
        self.assertEqual(len(slurm_experiment_blocks()), 3,
                         "run_all_benchmarks.slurm should still build exactly three EXPERIMENTS "
                         "arrays: MIP only, small, large")

    def test_flags_are_the_same(self):
        only, small, large = slurm_experiment_blocks()
        # "$MIP_FLAG" is a variable in the bash and a computed flag here, so compare what follows
        # it; the MIP flag itself is covered by test_mip_modes.
        cases = [("PCAP100", "only", only), ("NONE", "yes", small), ("PCAP100", "yes", large)]
        for capacity, run_mip, expected in cases:
            with self.subTest(family=families.family_name(capacity, run_mip)):
                got = families.experiment_flags(capacity, run_mip)
                self.assertTrue(got[0].startswith("--experiment-mip="))
                self.assertEqual(got[1:], expected)

    def test_family_names_are_the_same(self):
        names = re.findall(r'FAMILY="([^"]*)"', SLURM.read_text(encoding="utf-8"))
        self.assertEqual(names, [families.FAMILY_MIP_ONLY, families.FAMILY_SMALL,
                                 families.FAMILY_LARGE])

    def test_mip_modes(self):
        # yes/no/only ignore the node; auto is the only mode that consults it.
        self.assertTrue(families.mip_enabled("yes", licence_found=False))
        self.assertFalse(families.mip_enabled("no", licence_found=True))
        self.assertTrue(families.mip_enabled("only", licence_found=False))
        self.assertTrue(families.mip_enabled("auto", licence_found=True))
        self.assertFalse(families.mip_enabled("auto", licence_found=False))
        with self.assertRaises(ValueError):
            families.mip_enabled("maybe")


class TestFamilySystemCounts(unittest.TestCase):
    """The counts run_all_benchmarks.slurm's comments state: 39 small, 12 large, 1 MIP-only."""

    def systems(self, capacity, run_mip):
        args = caller.build_arg_parser().parse_args(
            ["problem", *families.experiment_flags(capacity, run_mip), *COMMON])
        return [s["key"] for s in caller.build_system_config(BENCH, Path("out"), "P", args)]

    def test_small_is_39(self):
        self.assertEqual(len(self.systems("NONE", "yes")), 39)

    def test_large_is_12(self):
        keys = self.systems("PCAP100", "yes")
        self.assertEqual(len(keys), 12)
        self.assertIn("05_ASP_rp_dp_sp", keys)
        self.assertIn("05_ASP_rp_d_sp", keys)
        self.assertNotIn("18_ASP_rp_dp_sp", keys)

    def test_mip_only_is_one(self):
        self.assertEqual(self.systems("PCAP100", "only"), ["04_MIP"])

    def test_run_mip_no_drops_mip(self):
        self.assertNotIn("04_MIP", self.systems("PCAP100", "no"))


class TestUnitCommandLine(unittest.TestCase):
    """One unit's argv is the monolithic argv for that system, bar the results root.

    This is the claim the whole design rests on: if the two argv agree, the solver cannot tell
    which path invoked it, and the numbers cannot depend on the scheduling shape.
    """

    PATHS = {k: Path(f"/inst/{k}.csv") for k in
             ("graph-edges", "sectors", "flights", "airports", "airplanes", "airplane-flight",
              "navaid-sector")}

    def argv(self, output_path, extra):
        args = caller.build_arg_parser().parse_args(
            ["problem", *families.experiment_flags("PCAP100", "yes"), *COMMON, *extra])
        systems = caller.build_system_config(BENCH, output_path, "P", args)
        return args, systems

    def test_same_argv_except_results_root(self):
        mono_args, mono_systems = self.argv(Path("output/F/output_P"), [])
        for system in mono_systems:
            key = system["key"]
            with self.subTest(system=key):
                shard = Path(f"output/F/units/shards/P/INST/{key}")
                unit_args, unit_systems = self.argv(
                    shard, [f"--only-system={key}", "--only-instance=INST"])
                selected = caller.select_systems(unit_systems, [key])
                self.assertEqual([s["key"] for s in selected], [key])

                def full(sys_, args):
                    return caller.build_command(
                        sys_, self.PATHS, "/bin/python", 1,
                        solver_cli=caller.solver_option_cli(sys_, args)) + sys_["cmd"]

                mono_cmd = full(system, mono_args)
                unit_cmd = full(selected[0], unit_args)
                self.assertEqual(len(mono_cmd), len(unit_cmd))
                for a, b in zip(mono_cmd, unit_cmd):
                    if a.startswith("--results-root="):
                        self.assertTrue(b.startswith("--results-root="))
                        self.assertTrue(b.endswith(a.split("/solver_outputs/", 1)[1]),
                                        f"{b} should end in the same solver_outputs leaf as {a}")
                    else:
                        self.assertEqual(a, b)


class TestSelectors(unittest.TestCase):
    def test_systems_keep_build_order(self):
        systems = [{"key": k} for k in ("01_A", "04_MIP", "05_ASP_rp_d_sp")]
        got = caller.select_systems(systems, ["05_ASP_rp_d_sp", "01_A"])
        self.assertEqual([s["key"] for s in got], ["01_A", "05_ASP_rp_d_sp"],
                         "selection order must not leak into the CSV column order")

    def test_instances_keep_directory_order(self):
        instances = [Path("/p/a"), Path("/p/b"), Path("/p/c")]
        got = caller.select_instances(instances, ["c", "a"])
        self.assertEqual([p.name for p in got], ["a", "c"],
                         "selection order must not leak into the CSV row order")

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            caller.select_systems([{"key": "01_A"}], ["02_B"])
        with self.assertRaises(ValueError):
            caller.select_instances([Path("/p/a")], ["b"])

    def test_split_selection(self):
        self.assertIsNone(caller.split_selection(None))
        self.assertIsNone(caller.split_selection([]))
        self.assertEqual(caller.split_selection(["a,b", " c "]), ["a", "b", "c"])


class TestResultWriter(unittest.TestCase):
    """The metric CSVs discover their header while walking the instances. Pinned, warts and all.

    KNOWN DEFECT, PINNED DELIBERATELY AND NOT FIXED HERE. In the metric CSVs (overload.csv and
    its nine siblings) the HEADER is built in the order systems are first seen reporting that
    metric, while each ROW is built in system-list order. The two agree only when every system
    reports the metric on the first instance. When one does not -- a run that timed out before
    any model, say -- that system's column is added later, and from then on the row values sit
    under the wrong column names, and the first rows are short.

    Fixing it would change the numbers in every affected CSV, which is exactly what this branch
    promises not to do, so it is pinned instead: merged output and monolithic output must agree,
    and they do because merge_benchmark_shards.py calls this same function. execution_time.csv
    and ram_usage.csv are unaffected -- their header is the full system list up front.
    """

    def test_header_discovery_is_order_dependent(self):
        instances = ["i1", "i2"]
        systems = ["S_late", "S_always"]
        sol = {
            # S_late reports no OVERLOAD on i1 (it timed out with no model) and does on i2.
            "i1": {"S_late": [{"ERROR": "T"}], "S_always": [{"OVERLOAD": 1, "ERROR": ""}]},
            "i2": {"S_late": [{"OVERLOAD": 7, "ERROR": ""}], "S_always": [{"OVERLOAD": 2, "ERROR": ""}]},
        }
        exec_time = {i: {s: 1.5 for s in systems} for i in instances}
        ram = {i: {s: 42 for s in systems} for i in instances}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            caller.write_result_csvs(out, instances, systems, exec_time, ram, sol)
            overload = (out / "overload.csv").read_text().splitlines()
            # Header: S_always first (seen on i1), S_late appended when it turns up on i2.
            self.assertEqual(overload[0], "Instance,S_always,S_late")
            # i1 is SHORT: S_late contributed nothing.
            self.assertEqual(overload[1], "i1,1")
            # i2 is written in SYSTEM order (S_late=7 then S_always=2), so under that header the
            # two values are swapped. This is the defect; it is what the current code does.
            self.assertEqual(overload[2], "i2,7,2")
            # execution_time.csv is immune: its header is the whole system list, up front.
            self.assertEqual((out / "execution_time.csv").read_text().splitlines()[0],
                             "Instance,S_late,S_always")
            self.assertTrue((out / "individual_outputs" / "i1_S_late.json").exists())

    def test_every_metric_csv_is_written(self):
        instances, systems = ["i1"], ["S"]
        sol = {"i1": {"S": [{"OVERLOAD": 0, "ERROR": ""}]}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            caller.write_result_csvs(out, instances, systems, {"i1": {"S": 1.0}},
                                     {"i1": {"S": 2}}, sol)
            for metric in caller.RESULT_METRICS:
                self.assertTrue((out / f"{metric.lower()}.csv").exists(), metric)
            self.assertTrue((out / "execution_time.csv").exists())
            self.assertTrue((out / "ram_usage.csv").exists())


if __name__ == "__main__":
    unittest.main()
