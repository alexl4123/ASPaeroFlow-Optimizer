"""Packing the published solution matrices: which runs, which archive, and a verifiable zip.

pack_solution_matrices.py decides from the validator's CSVs alone which runs are published, cuts
each (family, licence) group into archives in sorted path order, and writes zips a reader can
validate with validate_solutions.py as they are. These tests pin

  * the selection -- finished, VALID, the matrix is the run's own, taken from the folder the
    system is taken from (03_DELAY / 03A_CASA from the rerun), nothing selected twice;
  * the split -- contiguous, every run once, no archive over the limit, the fewest archives;
  * a plan -> pack -> check round trip on a synthetic campaign, and that a changed source file,
    a result line that is not the validated one, or a corrupted member is caught;
  * which run files the plan requires (what the validator needs to call a run VALID, including
    every matrix a run's manifest.json lists) and that it counts the optional ones a run lacks;
  * that plan first deletes the plan of an earlier call, so a plan that fails leaves none behind;
  * that pack skips an archive already in the out folder only if it is the plan's, repacks it
    otherwise, and that check refuses an archive that is not the plan's or belongs to an earlier
    plan, and writes SHA256SUMS.archives ('<sha256>  ./<zip>') only when everything holds;
  * validate_solutions.py on the archive layout, with the instance found in the published
    <region>/PCAP<c>/ layout.

Run with:  python -m unittest discover -s tests -v      (from the repository root)
"""
import csv
import gzip
import io
import json
import random
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "06_benchmark_start_script"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import pack_solution_matrices as pack    # noqa: E402
import validate_solutions as vs          # noqa: E402

LARGE_CC = "00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1-V2-TG1-PCAP010"
LARGE_GPL = "08-0-CENTRAL-EUROPE-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-30-V2-TG60-PCAP070"
SMALL = "30-0-EAST-ASIA-3x3-V2"


def row(system, folder="20260918_V2", problem=SMALL, instance="0000010_SEED1", status="VALID",
        outcome="ok", owner=None, campaign_outcome="", matrix_dir=None, **claims):
    r = {c: "" for c in pack.ROW_COLUMNS}
    r.update(folder=folder, problem=problem, instance=instance, system=system, status=status,
             outcome=outcome, matrix_owner=system if owner is None else owner,
             campaign_outcome=campaign_outcome,
             matrix_dir=matrix_dir if matrix_dir is not None else f"solver_outputs/{system}/{instance}")
    for m in vs.METRICS:
        r[f"claimed_{m}"] = str(claims.get(m, 0))
    return r


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args)


def captured(fn, *args):
    """(return value, stdout + stderr)."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = fn(*args)
    return code, buf.getvalue()


class Classify(unittest.TestCase):
    def test_families_and_licences(self):
        self.assertEqual(pack.classify(SMALL), ("small", pack.CC_BY))
        self.assertEqual(pack.classify(LARGE_CC), ("large", pack.CC_BY))
        self.assertEqual(pack.classify(LARGE_GPL), ("large", pack.GPL))
        for region in ("04-0-DACH", "05-0-EUROPE", "06-0-USA-MAINLAND"):
            self.assertEqual(pack.classify(region + "-2019-06-01-X-TG4-PCAP020")[1], pack.GPL)
        for region in ("01-0-USA-EAST-COAST-20x10", "02-0-MAJOR-EUROPE-40x20", "03-0-EAST-ASIA-40x40"):
            self.assertEqual(pack.classify(region + "-2019-06-01-X-TG4-PCAP020")[1], pack.CC_BY)

    def test_unknown_region_is_an_error(self):
        with self.assertRaises(pack.PlanError):
            pack.classify("07-0-WORLD-2019-06-01-X-TG1-PCAP010")


class Selection(unittest.TestCase):
    rules = pack.parse_take(pack.DEFAULT_TAKE)
    systems = list(vs.PUBLISHED_SYSTEMS)

    def select(self, rows):
        return pack.select_runs(rows, self.rules, self.systems)

    def test_default_rule(self):
        self.assertEqual(pack.source_folders(self.rules, "03_DELAY"), ("20260930_V2_RERUN_DC",))
        self.assertEqual(pack.source_folders(self.rules, "03A_CASA"), ("20260930_V2_RERUN_DC",))
        self.assertEqual(pack.source_folders(self.rules, "0_Sequential"), ("20260930_V2_SEQ",))
        self.assertEqual(pack.source_folders(self.rules, "04_MIP"), ("20260918_V2", "20260918_V2_MIP"))
        with self.assertRaises(pack.PlanError):
            pack.parse_take(["01_ASPaeroFlow=A", "01_ASPaeroFlow=B"])

    def test_only_finished_valid_own_matrices_from_the_right_folder(self):
        rows = [
            row("01_ASPaeroFlow"),                                                     # yes
            row("04_MIP", folder="20260918_V2_MIP"),                                   # yes
            row("0B_Sector_NoReroute_NoDelay", outcome="TIMEOUT"),                     # VALID, not finished
            row("0C_Sector_NoReroute_Delay", status="NO_MATRIX", outcome="TIMEOUT"),
            row("0D_Sector_Reroute_NoDelay", status="INVALID", outcome="TIMEOUT"),
            row("03_DELAY", owner="ambiguous(03_DELAY+03A_CASA)"),                     # campaign: other folder
            row("03A_CASA"),                                                           # campaign: other folder
            row("03_DELAY", folder="20260930_V2_RERUN_DC", campaign_outcome="ok"),     # yes
            row("03A_CASA", folder="20260930_V2_RERUN_DC", campaign_outcome="ok",
                status="NO_MATRIX", outcome="TIMEOUT"),                                # rerun did not finish
            row("0_Sequential", folder="20260930_V2_SEQ"),                             # yes
            row("0A_ASPaeroFlow_NoConvex"),                                            # not published
        ]
        selected, excluded, unpublished = self.select(rows)
        self.assertEqual([(r["system"], r["folder"]) for r in selected],
                         [("01_ASPaeroFlow", "20260918_V2"), ("03_DELAY", "20260930_V2_RERUN_DC"),
                          ("04_MIP", "20260918_V2_MIP"), ("0_Sequential", "20260930_V2_SEQ")])
        reasons = {(s, f): r for (s, f, r) in excluded}
        self.assertEqual(reasons[("0B_Sector_NoReroute_NoDelay", "20260918_V2")], "not finished (TIMEOUT)")
        self.assertEqual(reasons[("03_DELAY", "20260918_V2")], "taken from 20260930_V2_RERUN_DC")
        self.assertEqual(reasons[("03A_CASA", "20260930_V2_RERUN_DC")], "not finished (TIMEOUT)")
        self.assertEqual(reasons[("0A_ASPaeroFlow_NoConvex", "20260918_V2")], "system not published")
        # 03A_CASA finished in the campaign and has no published matrix: reported
        self.assertEqual([key[2] for key, _ in unpublished], ["03A_CASA"])
        self.assertIn("20260930_V2_RERUN_DC: not finished (TIMEOUT)", unpublished[0][1])

    def test_rerun_row_needs_the_campaign_to_have_finished(self):
        selected, excluded, _ = self.select([
            row("03_DELAY", folder="20260930_V2_RERUN_DC", campaign_outcome="TIMEOUT")])
        self.assertEqual(selected, [])
        self.assertEqual(list(excluded)[0][2], "not finished in the campaign (TIMEOUT)")

    def test_matrix_of_another_system_is_not_published(self):
        rules = pack.parse_take(["*=20260918_V2"])
        selected, excluded, _ = pack.select_runs(
            [row("03A_CASA", owner="ambiguous(03A_CASA+03_DELAY)"), row("03_DELAY", owner="03A_CASA")],
            rules, self.systems)
        self.assertEqual(selected, [])
        self.assertEqual(sorted(r for (_, _, r) in excluded),
                         ["matrix owner 03A_CASA", "matrix owner ambiguous(03A_CASA+03_DELAY)"])

    def test_a_run_selected_twice_is_an_error(self):
        with self.assertRaises(pack.PlanError):
            self.select([row("04_MIP"), row("04_MIP", folder="20260918_V2_MIP")])
        with self.assertRaises(pack.PlanError):          # the same CSV passed twice
            self.select([row("01_ASPaeroFlow"), row("01_ASPaeroFlow")])

    def test_order_is_the_archive_path_order(self):
        rows = [row("2A_Reroute", instance="B"), row("01_ASPaeroFlow", instance="B"),
                row("01_ASPaeroFlow", instance="A", problem=LARGE_CC)]
        selected, _, _ = self.select(rows)
        self.assertEqual([(r["problem"][:2], r["system"], r["instance"]) for r in selected],
                         [("00", "01_ASPaeroFlow", "A"), ("30", "01_ASPaeroFlow", "B"), ("30", "2A_Reroute", "B")])

    def test_summary_csvs_are_not_read_as_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20260918_V2").mkdir()
            for name in ("validation_summary.csv", "validation_flagged_runs.csv",
                         "validation_rerun_deviations.csv", "20260918_V2/validation_P.csv"):
                (root / name).write_text("folder,problem,instance,system,status\n")
            self.assertEqual([p.name for p in pack.validation_csvs([root])], ["validation_P.csv"])


class Split(unittest.TestCase):
    def check_parts(self, sizes, cap, parts):
        self.assertEqual([i for p in parts for i in p], list(range(len(sizes))))    # contiguous, once each
        for p in parts:
            if len(p) > 1:
                self.assertLessEqual(sum(sizes[i] for i in p), cap)

    def test_fits_in_one(self):
        self.assertEqual(pack.split_sizes([3, 3, 3], 10), [[0, 1, 2]])
        self.assertEqual(pack.split_sizes([], 10), [])

    def test_fewest_parts_and_as_even_as_possible(self):
        parts = pack.split_sizes([4, 4, 4, 4, 4], 9)       # next-fit at 9: [4,4] [4,4] [4]
        self.assertEqual(len(parts), 3)
        self.assertEqual(max(sum(4 for _ in p) for p in parts), 8)
        parts = pack.split_sizes([5, 1, 1, 1, 1, 1], 6)    # next-fit [5,1] [1,1,1,1]; largest part 5
        self.assertEqual(parts, [[0], [1, 2, 3, 4, 5]])

    def test_an_item_over_the_limit_gets_its_own_part(self):
        self.assertEqual(pack.split_sizes([2, 20, 2], 10), [[0], [1], [2]])

    def test_random(self):
        rng = random.Random(7)
        for _ in range(300):
            sizes = [rng.randint(1, 50) for _ in range(rng.randint(1, 40))]
            cap = rng.randint(50, 300)
            parts = pack.split_sizes(sizes, cap)
            self.check_parts(sizes, cap, parts)
            # next-fit at the full limit needs no fewer parts
            n, filled = 1, 0
            for s in sizes:
                if filled and filled + s > cap:
                    n, filled = n + 1, 0
                filled += s
            self.assertEqual(len(parts), n)


class RunFiles(unittest.TestCase):
    def test_what_the_validator_reads_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("converted_navpoint_matrix.csv.gz", "converted_navpoint_matrix.csv",
                         "converted_instance_matrix.csv.gz", "navaid_sector_time_assignment.npz",
                         "capacity_time_matrix.csv.gz", "manifest.json", "core.1234", "tmp.lp"):
                (d / name).write_bytes(b"x" * 3)
            (d / "scratch").mkdir()
            files, left_out = pack.run_files(d)
            self.assertEqual([n for n, _ in files],
                             ["capacity_time_matrix.csv.gz", "converted_instance_matrix.csv.gz",
                              "converted_navpoint_matrix.csv.gz", "manifest.json",
                              "navaid_sector_time_assignment.npz"])
            self.assertEqual(left_out, ["converted_navpoint_matrix.csv", "core.1234", "scratch/", "tmp.lp"])


class RoundTrip(unittest.TestCase):
    """plan -> pack -> check on a synthetic campaign (the matrix files are opaque bytes here)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.out, self.val, self.plan, self.zips = root / "output", root / "val", root / "plan", root / "zips"
        rows = []
        for folder, system, problem, instance, seed in (
                ("20260918_V2", "01_ASPaeroFlow", SMALL, "0000010_SEED1", 1),
                ("20260918_V2", "01_ASPaeroFlow", SMALL, "0000010_SEED2", 2),
                ("20260918_V2", "01_ASPaeroFlow", LARGE_CC, "0001000_SEED1", 3),
                ("20260918_V2", "01_ASPaeroFlow", LARGE_GPL, "0001000_SEED1", 4),
                ("20260918_V2", "01_ASPaeroFlow", LARGE_GPL, "0001000_SEED2", 5),
                ("20260918_V2", "03_DELAY", LARGE_GPL, "0001000_SEED1", 6),
                ("20260930_V2_RERUN_DC", "03_DELAY", LARGE_GPL, "0001000_SEED1", 6)):
            run_dir = self.out / folder / f"output_{problem}" / "solver_outputs" / system / instance
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = bytes(random.Random(seed + len(folder)).getrandbits(8) for _ in range(3000))
            for name in vs.MATRIX_NAMES:
                (run_dir / f"{name}.csv.gz").write_bytes(payload)
            (run_dir / "manifest.json").write_text(json.dumps(
                {"folder": folder, "source": {"timestep_granularity": 1}}))
            (run_dir / "stray.tmp").write_text("left out")
            line = {"OVERLOAD": 0, "ARRIVAL-DELAY": 10 * seed, "SECTOR-NUMBER": 5, "SECTOR-DIFF": 1,
                    "REROUTE": seed, "RECONFIG": 2, "COMPUTATION-FINISHED": True, "ERROR": ""}
            io_dir = self.out / folder / f"output_{problem}" / "individual_outputs"
            io_dir.mkdir(parents=True, exist_ok=True)
            if folder == "20260918_V2":            # a rerun's claims are the campaign's
                (io_dir / f"{instance}_{system}.json").write_text(json.dumps({"object": [{"ITERATION": 0}, line]}))
            rerun = folder != "20260918_V2"
            rows.append(row(system, folder=folder, problem=problem, instance=instance,
                            campaign_outcome="ok" if rerun else "", **line))
        self.val.mkdir()
        with (self.val / "validation_all.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=pack.ROW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def plan_args(self, gb):
        args = ["plan", "--plan-dir", str(self.plan), "--validation", str(self.val),
                "--max-archive-gb", str(gb), "--claims-from", "20260930_V2_RERUN_DC=20260918_V2"]
        for folder in ("20260918_V2", "20260930_V2_RERUN_DC"):
            args += ["--results-folder", f"{folder}={self.out / folder}"]
        return args

    def pack_all(self, *extra):
        plan = json.loads((self.plan / "plan.json").read_text())
        return [quiet(pack.main, ["pack", "--plan-dir", str(self.plan), "--index", str(i),
                                  "--out-dir", str(self.zips), *extra])
                for i in range(1, len(plan["archives"]) + 1)], plan

    def test_round_trip(self):
        # runs of ~20 kB; 116 kB minus the 64 kB fixed share holds two: the GPL group (3 runs)
        # splits into two archives, the small group (2 runs) does not
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        codes, plan = self.pack_all()
        self.assertEqual(codes, [0] * len(codes))
        names = [a["name"] for a in plan["archives"]]
        self.assertEqual(names, ["solution_matrices_large_CC-BY-4.0_01",
                                 "solution_matrices_large_GPL-2.0-or-later_01",
                                 "solution_matrices_large_GPL-2.0-or-later_02",
                                 "solution_matrices_small_CC-BY-4.0_01"])
        self.assertEqual(plan["left_out_files"], {"stray.tmp": 6})
        self.assertEqual(quiet(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips),
                                           "--rehash"]), 0)
        # one line '<sha256>  ./<zip>' per archive, in name order, as in the other records' SHA256SUMS
        lines = (self.zips / pack.SUMS_FILE).read_text().splitlines()
        self.assertEqual([line.split("  ./")[1] for line in lines], [f"{n}.zip" for n in sorted(names)])
        for line in lines:
            self.assertRegex(line, r"^[0-9a-f]{64}  \./solution_matrices_[^/ ]+\.zip$")
            digest, path = line.split("  ", 1)
            self.assertEqual(pack.sha256_file(self.zips / path), digest)
        self.assertFalse((self.zips / "SHA256SUMS").exists())           # the record's own is merged at upload
        packed = {}
        for name in names:
            with zipfile.ZipFile(self.zips / f"{name}.zip") as zf:
                self.assertIsNone(zf.testzip())
                for member in zf.namelist():
                    parts = member.split("/")
                    self.assertEqual(parts[0], name)
                    if len(parts) == 5:
                        packed.setdefault(tuple(parts[1:4]), set()).add(name)
                    if parts[-1] == "manifest.json" and parts[2] == "03_DELAY":
                        self.assertEqual(json.loads(zf.read(member))["folder"], "20260930_V2_RERUN_DC")
                    if parts[-1] == pack.RESULT_LINE_FILE and parts[2] == "03_DELAY":
                        self.assertEqual(json.loads(zf.read(member))["ARRIVAL-DELAY"], 60)
        self.assertEqual(len(packed), 6)                               # the campaign's 03_DELAY is not packed
        self.assertTrue(all(len(v) == 1 for v in packed.values()))
        # plan and check count the same members: the zip's files, README.md and MANIFEST.csv included
        for meta in plan["archives"]:
            with zipfile.ZipFile(self.zips / f"{meta['name']}.zip") as zf:
                self.assertEqual(len(zf.namelist()), meta["members"])
            with (self.zips / f"{meta['name']}.MANIFEST.csv").open() as fh:
                self.assertEqual(len(list(csv.DictReader(fh))) + 2, meta["members"])
        # packing the same plan again gives the same bytes
        first = {n: (self.zips / f"{n}.zip.sha256").read_text() for n in names}
        codes, _ = self.pack_all("--force")
        self.assertEqual({n: (self.zips / f"{n}.zip.sha256").read_text() for n in names}, first)

    def test_a_file_changed_after_the_plan_is_refused(self):
        quiet(pack.main, self.plan_args(20))
        src = self.out / "20260918_V2" / f"output_{SMALL}" / "solver_outputs/01_ASPaeroFlow/0000010_SEED1"
        with (src / "manifest.json").open("a") as fh:
            fh.write(" ")
        codes, plan = self.pack_all()
        small = [a["index"] for a in plan["archives"] if a["family"] == "small"][0]
        self.assertEqual(codes[small - 1], 2)
        self.assertTrue(any(self.zips.glob("*small*.zip.partial")))
        self.assertFalse(any(self.zips.glob("*small*.zip")))

    def test_a_result_line_other_than_the_validated_one_is_refused(self):
        quiet(pack.main, self.plan_args(20))
        path = self.out / "20260918_V2" / f"output_{SMALL}" / "individual_outputs" / "0000010_SEED2_01_ASPaeroFlow.json"
        data = json.loads(path.read_text())
        data["object"][-1]["ARRIVAL-DELAY"] += 1
        path.write_text(json.dumps(data))
        codes, plan = self.pack_all()
        small = [a["index"] for a in plan["archives"] if a["family"] == "small"][0]
        self.assertEqual(codes[small - 1], 2)

    def test_verification_catches_a_corrupted_member(self):
        quiet(pack.main, self.plan_args(20))
        self.pack_all()
        name = "solution_matrices_small_CC-BY-4.0_01"
        with (self.zips / f"{name}.MANIFEST.csv").open() as fh:
            rows = [(r["problem"], r["instance"], r["system"], r["file"], int(r["bytes"]), r["sha256"])
                    for r in csv.DictReader(fh)]
        pack.verify_archive(self.zips / f"{name}.zip", name, rows)          # intact: no exception
        data = bytearray((self.zips / f"{name}.zip").read_bytes())
        with zipfile.ZipFile(self.zips / f"{name}.zip") as zf:
            info = zf.getinfo(f"{name}/{SMALL}/01_ASPaeroFlow/0000010_SEED1/converted_navpoint_matrix.csv.gz")
        data[info.header_offset + 30 + len(info.filename) + 100] ^= 0xFF     # a byte of stored data
        broken = self.zips / "broken.zip"
        broken.write_bytes(bytes(data))
        with self.assertRaises(pack.PackError):
            pack.verify_archive(broken, name, rows)

    # ---- files a run lacks ----------------------------------------------------------------------

    def small_run(self, instance="0000010_SEED1"):
        return self.out / "20260918_V2" / f"output_{SMALL}" / "solver_outputs" / "01_ASPaeroFlow" / instance

    def test_an_optional_file_missing_is_counted(self):
        (self.small_run() / "capacity_time_matrix.csv.gz").unlink()
        code, text = captured(pack.main, self.plan_args(20))
        self.assertEqual(code, 0, text)
        plan = json.loads((self.plan / "plan.json").read_text())
        lacking = plan["runs_without_file"]
        self.assertEqual(lacking["capacity_time_matrix"], {"required_for": "if manifest.json lists it",
                                                           "runs": 1, "by_system": {"01_ASPaeroFlow": 1}})
        self.assertEqual(lacking["converted_navpoint_matrix"]["runs"], 0)
        self.assertRegex(text, r"capacity_time_matrix\s+if manifest.json lists it\s+0\s+1\s+\(01_ASPaeroFlow 1\)")
        self.assertIn("[NOTE] 1 optional files missing", text)
        codes, plan = self.pack_all()
        self.assertEqual(codes, [0] * len(codes))
        self.assertEqual(quiet(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips)]), 0)

    def assert_plan_refused(self, what):
        (self.plan / "plan.json").parent.mkdir(parents=True, exist_ok=True)
        (self.plan / "plan.json").write_text("{}")                   # an earlier plan is not left behind
        code, text = captured(pack.main, self.plan_args(20))
        self.assertEqual(code, 2, text)
        self.assertFalse((self.plan / "plan.json").exists())
        self.assertIn(what, (self.plan / "plan_errors.txt").read_text())
        return text

    def test_a_required_matrix_missing_is_an_error(self):
        (self.small_run() / "navaid_sector_time_assignment.csv.gz").unlink()
        text = self.assert_plan_refused("no navaid_sector_time_assignment")
        self.assertRegex(text, r"navaid_sector_time_assignment\s+every run\s+1\s+-")

    def test_a_matrix_the_manifest_lists_is_required(self):
        # SEED1 lists all four matrices and lacks the capacity matrix: the validator would call it
        # INVALID ("listed in manifest.json but missing"). SEED2 does not list it (as 05_ASP_*):
        # optional there.
        for instance, names in (("0000010_SEED1", vs.MATRIX_NAMES),
                                ("0000010_SEED2", [n for n in vs.MATRIX_NAMES if n != "capacity_time_matrix"])):
            (self.small_run(instance) / "manifest.json").write_text(json.dumps(
                {"saved": {n: {"shape": [1, 1]} for n in names}, "source": {"timestep_granularity": 1}}))
            (self.small_run(instance) / "capacity_time_matrix.csv.gz").unlink()
        text = self.assert_plan_refused("no capacity_time_matrix (listed in its manifest.json)")
        self.assertRegex(text, r"capacity_time_matrix\s+if manifest.json lists it\s+1\s+1\s+\(01_ASPaeroFlow 2\)")
        self.assertEqual(len((self.plan / "plan_errors.txt").read_text().splitlines()), 1)
        self.assertIn("0000010_SEED1", (self.plan / "plan_errors.txt").read_text())
        # without the listing in SEED1 the plan goes through; both runs are counted as lacking it
        (self.small_run() / "manifest.json").write_text(json.dumps({"source": {"timestep_granularity": 1}}))
        code, text = captured(pack.main, self.plan_args(20))
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads((self.plan / "plan.json").read_text())["runs_without_file"]
                         ["capacity_time_matrix"]["runs"], 2)
        self.assertFalse((self.plan / "plan_errors.txt").exists())

    def test_a_missing_result_line_is_an_error(self):
        (self.out / "20260918_V2" / f"output_{SMALL}" / "individual_outputs" /
         "0000010_SEED2_01_ASPaeroFlow.json").unlink()
        text = self.assert_plan_refused("no result_line.json source")
        self.assertRegex(text, r"result_line.json source\s+every run\s+1\s+-")

    # ---- a plan that fails leaves no earlier plan behind ------------------------------------------

    def plan_files(self):
        return sorted(str(p.relative_to(self.plan)) for p in self.plan.rglob("*") if p.is_file())

    def test_plan_deletes_the_previous_plan_first(self):
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        full = self.plan_files()
        self.assertEqual(len(full), 3 + 4)       # plan.json, the two CSVs, archives/<4>.runs.json
        (self.plan / "notes.txt").write_text("mine")                   # not the plan's: kept
        # a check that fails before anything is read (--take), and one on the input (no CSV)
        for bad in (["--take", "nonsense"], ["--validation", str(self.val / "none.csv")]):
            code, text = captured(pack.main, self.plan_args(0.000116) + bad)
            self.assertEqual(code, 2, text)
            self.assertEqual(self.plan_files(), ["notes.txt"])
            self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        # an option argparse refuses
        with self.assertRaises(SystemExit) as ctx:
            quiet(pack.main, self.plan_args(0.000116) + ["--max-archive-gb", "abc"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(self.plan_files(), ["notes.txt"])
        # --help deletes nothing
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        with self.assertRaises(SystemExit) as ctx:
            quiet(pack.main, self.plan_args(0.000116) + ["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(len(self.plan_files()), 8)
        # --selection-only: no plan.json, no archive lists of the earlier plan
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116) + ["--selection-only"]), 0)
        self.assertEqual(self.plan_files(), ["notes.txt", "selected_runs.csv", "unpublished_finished_runs.csv"])

    def test_a_missing_run_folder_is_an_error(self):
        for f in self.small_run().iterdir():
            f.unlink()
        self.small_run().rmdir()
        self.assert_plan_refused("run folder missing")

    def test_a_problem_without_tg_needs_a_manifest_that_names_it(self):
        for instance in ("0000010_SEED1", "0000010_SEED2"):
            (self.small_run(instance) / "manifest.json").write_text("{}")
        self.assert_plan_refused(f"{SMALL}: no -TG<g> in the name")
        # a large problem carries it in its name: no manifest.json needed there
        for run_dir in (self.out / "20260918_V2" / f"output_{LARGE_CC}" / "solver_outputs").glob("*/*"):
            (run_dir / "manifest.json").unlink()
        (self.small_run() / "manifest.json").write_text(json.dumps({"source": {"timestep_granularity": 1}}))
        code, text = captured(pack.main, self.plan_args(20))
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads((self.plan / "plan.json").read_text())["runs_without_file"]
                         ["manifest.json"]["runs"], 1)

    # ---- an archive already in the out folder ---------------------------------------------------

    def pack_verbose(self):
        plan = json.loads((self.plan / "plan.json").read_text())
        out = {}
        for meta in plan["archives"]:
            code, text = captured(pack.main, ["pack", "--plan-dir", str(self.plan), "--index",
                                              str(meta["index"]), "--out-dir", str(self.zips)])
            self.assertEqual(code, 0, text)
            out[meta["name"]] = text
        return out

    def test_an_archive_of_the_same_plan_is_skipped(self):
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        self.pack_verbose()
        before = {p.name: p.stat().st_mtime_ns for p in self.zips.iterdir()}
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)       # the same plan again
        for name, text in self.pack_verbose().items():
            self.assertIn(f"[SKIP] {name}: already packed", text)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in self.zips.iterdir()}, before)

    def test_an_archive_of_another_plan_is_repacked(self):
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        first = self.pack_verbose()
        self.assertTrue(all("[OK]" in t for t in first.values()))
        # the selection changes: one small run is no longer VALID
        path = self.val / "validation_all.csv"
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if r["problem"] == SMALL and r["instance"] == "0000010_SEED2":
                r["status"] = "INVALID"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=pack.ROW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)
        second = self.pack_verbose()
        small = "solution_matrices_small_CC-BY-4.0_01"
        self.assertIn(f"[REPACK] {small}", second[small])
        self.assertIn("1 of its runs are not in the plan", second[small])
        self.assertIn("[OK]", second[small])
        self.assertTrue(all("[SKIP]" in t for n, t in second.items() if n != small))
        with zipfile.ZipFile(self.zips / f"{small}.zip") as zf:
            self.assertFalse(any("0000010_SEED2" in m for m in zf.namelist()))
        self.assertEqual(quiet(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips),
                                           "--rehash"]), 0)

    def test_same_runs_other_readme_is_repacked(self):
        self.assertEqual(quiet(pack.main, self.plan_args(20)), 0)
        self.pack_verbose()
        plan = json.loads((self.plan / "plan.json").read_text())
        plan["optimizer_commit"] = "0123abc"                   # the README names the commit
        (self.plan / "plan.json").write_text(json.dumps(plan))
        for name, text in self.pack_verbose().items():
            self.assertIn(f"[REPACK] {name}", text)
            self.assertIn("README.md differs", text)
            with zipfile.ZipFile(self.zips / f"{name}.zip") as zf:
                self.assertIn("0123abc", zf.read(f"{name}/README.md").decode())

    def test_a_changed_file_size_is_repacked(self):
        self.assertEqual(quiet(pack.main, self.plan_args(20)), 0)
        self.pack_verbose()
        with (self.small_run() / "manifest.json").open("a") as fh:
            fh.write(" ")
        self.assertEqual(quiet(pack.main, self.plan_args(20)), 0)
        texts = self.pack_verbose()
        small = "solution_matrices_small_CC-BY-4.0_01"
        self.assertIn("1 files differ in name or size", texts[small])
        self.assertTrue(all("[SKIP]" in t for n, t in texts.items() if n != small))

    def test_check_refuses_an_archive_of_an_earlier_plan(self):
        self.assertEqual(quiet(pack.main, self.plan_args(0.000116)), 0)       # GPL split in two
        self.pack_verbose()
        self.assertEqual(quiet(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips)]), 0)
        self.assertTrue((self.zips / pack.SUMS_FILE).exists())
        self.assertEqual(quiet(pack.main, self.plan_args(20)), 0)             # one GPL archive
        self.pack_verbose()
        self.assertFalse((self.zips / pack.SUMS_FILE).exists())                 # a repack makes it stale
        code, text = captured(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips)])
        self.assertEqual(code, 1)
        self.assertIn("belong to no archive of this plan", text)
        self.assertIn("solution_matrices_large_GPL-2.0-or-later_02.zip", text)
        self.assertFalse((self.zips / pack.SUMS_FILE).exists())
        for stale in self.zips.glob("solution_matrices_large_GPL-2.0-or-later_02.*"):
            stale.unlink()
        self.assertEqual(quiet(pack.main, ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips)]), 0)

    def test_check_refuses_an_archive_that_is_not_the_plans(self):
        self.assertEqual(quiet(pack.main, self.plan_args(20)), 0)
        self.pack_verbose()
        check = ["check", "--plan-dir", str(self.plan), "--out-dir", str(self.zips)]
        self.assertEqual(quiet(pack.main, check), 0)
        # the same runs and files, but the plan now names another optimizer commit: the README of
        # every archive is not the plan's (members and runs alone would pass)
        plan = json.loads((self.plan / "plan.json").read_text())
        plan["optimizer_commit"] = "0123abc"
        (self.plan / "plan.json").write_text(json.dumps(plan))
        code, text = captured(pack.main, check)
        self.assertEqual(code, 1, text)
        self.assertEqual(text.count("not the plan's archive (README.md differs"), len(plan["archives"]))
        self.assertFalse((self.zips / pack.SUMS_FILE).exists())
        texts = self.pack_verbose()                                       # pack repacks them
        self.assertTrue(all("[REPACK]" in t and "README.md differs" in t for t in texts.values()))
        self.assertEqual(quiet(pack.main, check), 0)
        # a MANIFEST.csv next to the zip that is not the one inside it
        small = "solution_matrices_small_CC-BY-4.0_01"
        side = self.zips / f"{small}.MANIFEST.csv"
        side.write_bytes(side.read_bytes() + b"x\n")
        code, text = captured(pack.main, check)
        self.assertEqual(code, 1, text)
        self.assertIn(f"{small}.MANIFEST.csv is not the MANIFEST.csv inside the zip", text)
        # a zip that cannot be read
        side.write_bytes(side.read_bytes()[:-2])
        self.assertEqual(quiet(pack.main, check), 0)
        (self.zips / f"{small}.zip").write_bytes(b"not a zip")
        code, text = captured(pack.main, check)
        self.assertEqual(code, 1, text)
        self.assertIn(f"{small}.zip unreadable", text)
        texts = self.pack_verbose()
        self.assertIn(f"[REPACK] {small}", texts[small])
        self.assertEqual(quiet(pack.main, check + ["--rehash"]), 0)

    def test_selection_only_needs_no_matrices(self):
        args = ["plan", "--plan-dir", str(self.plan), "--validation", str(self.val), "--selection-only"]
        self.assertEqual(quiet(pack.main, args), 0)
        self.assertFalse((self.plan / "plan.json").exists())
        with (self.plan / "selected_runs.csv").open() as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 6)


class ValidatorArchiveLayout(unittest.TestCase):
    """validate_solutions.py on <archive>/<problem>/<system>/<instance>/ with the published
    instance layout <root>/<region>/PCAP<c>/<instance>/."""

    problem = "00-0-CENTRAL-EUROPE-7x7-TEST-TG1-PCAP010"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        inst = root / "instances" / "00-0-CENTRAL-EUROPE-7x7-TEST-TG1" / "PCAP010" / "0000001_SEED1"
        inst.mkdir(parents=True)
        files = {   # airport 0 -> vertex 1 -> airport 2, one timestep per edge at T_gran = 1
            "sectors.csv": "Sector_ID,Capacity\n0,5\n1,5\n2,5\n",
            "airports.csv": "Airport_Vertex\n0\n2\n",
            "graph_edges.csv": "source,target,dist_m\n0,1,1000.0\n1,2,1000.0\n",
            "airplanes.csv": "Airplane_ID,Speed_kts\n0,400\n",
            "airplane_flight_assignment.csv": "Airplane_ID,Flight_ID\n0,0\n",
            "flights.csv": "Flight_ID,Position,Time\n0,0,0\n0,1,1\n0,2,2\n",
            "navaid_sector_assignment.csv": "Navaid_ID,Sector_ID\n0,0\n1,1\n2,2\n",
        }
        for name, text in files.items():
            (inst / name).write_text(text)
        self.instances = root / "instances"
        instance = vs.Instance(inst, 1)
        self.assertEqual(instance.self_check(), [])
        width = instance.window
        nav = np.full((1, width), -1, dtype=np.int64)
        nav[0, :3] = [0, 1, 2]
        alloc = np.repeat(np.arange(3)[:, None], width, axis=1)
        sec = nav.copy()
        capm = np.full((3, width), 5, dtype=np.int64)
        ev = vs.evaluate_solution(instance, nav, sec, alloc, capm, "signed")
        self.run_dir = root / "ARCHIVE" / self.problem / "01_ASPaeroFlow" / "0000001_SEED1"
        self.run_dir.mkdir(parents=True)
        for name, matrix in (("converted_navpoint_matrix", nav), ("converted_instance_matrix", sec),
                             ("navaid_sector_time_assignment", alloc), ("capacity_time_matrix", capm)):
            with gzip.open(self.run_dir / f"{name}.csv.gz", "wt") as fh:
                np.savetxt(fh, matrix, fmt="%d", delimiter=",")
        (self.run_dir / "manifest.json").write_text(json.dumps(
            {"source": {"timestep_granularity": 1, "arrival_delay_metric": "signed"}}))
        self.line = dict(ev.recomputed, **{"COMPUTATION-FINISHED": True, "ERROR": "",
                                           "ARRIVAL-DELAY-METRIC": "signed"})
        (self.run_dir / vs.RESULT_LINE_FILE).write_text(json.dumps(self.line))
        self.args = ["--problem-dir", str(root / "ARCHIVE" / self.problem), "--instance-root",
                     str(self.instances), "--out-dir", str(root / "check")]
        self.csv = root / "check" / "ARCHIVE" / f"validation_{self.problem}.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        with self.csv.open() as fh:
            return list(csv.DictReader(fh))

    def test_valid(self):
        self.assertTrue(vs.is_archive_layout(self.run_dir.parents[1]))
        self.assertEqual(quiet(vs.main, self.args), 0)
        (r,) = self.rows()
        self.assertEqual((r["status"], r["system"], r["instance"], r["folder"], r["matrix_dir"]),
                         ("VALID", "01_ASPaeroFlow", "0000001_SEED1", "ARCHIVE", "01_ASPaeroFlow/0000001_SEED1"))
        self.assertEqual(r["note"], "")

    def test_a_wrong_claim_is_a_mismatch(self):
        self.line["ARRIVAL-DELAY"] += 1
        (self.run_dir / vs.RESULT_LINE_FILE).write_text(json.dumps(self.line))
        self.assertEqual(quiet(vs.main, self.args), 1)
        self.assertEqual(self.rows()[0]["failed_checks"], "claim:ARRIVAL-DELAY")

    def test_a_matrix_the_manifest_lists_must_be_there(self):
        # what the plan's rule rests on: a missing matrix that manifest.json lists makes the run
        # INVALID; the same matrix missing without the listing only skips a check
        (self.run_dir / "capacity_time_matrix.csv.gz").unlink()
        self.assertEqual(quiet(vs.main, self.args), 0)
        self.assertEqual(self.rows()[0]["status"], "VALID")
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        manifest["saved"] = {"capacity_time_matrix": {"shape": [3, 1]}}
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(quiet(vs.main, self.args), 2)
        (r,) = self.rows()
        self.assertEqual((r["status"], r["failed_checks"]), ("INVALID", "matrix_unreadable"))
        self.assertIn("capacity_time_matrix listed in manifest.json but missing", r["note"])

    def test_instance_dir_prefers_the_campaign_layout(self):
        direct = self.instances / self.problem / "0000001_SEED1"
        nested = self.instances / "00-0-CENTRAL-EUROPE-7x7-TEST-TG1" / "PCAP010" / "0000001_SEED1"
        self.assertEqual(vs.instance_dir(self.instances, self.problem, "0000001_SEED1"), nested)
        direct.mkdir(parents=True)
        self.assertEqual(vs.instance_dir(self.instances, self.problem, "0000001_SEED1"), direct)

    def test_a_campaign_folder_is_not_an_archive(self):
        problem_dir = Path(self.tmp.name) / "output" / "F" / "output_P"
        (problem_dir / "solver_outputs" / "01_ASPaeroFlow" / "I").mkdir(parents=True)
        (problem_dir / "solver_outputs" / "01_ASPaeroFlow" / "I" / vs.RESULT_LINE_FILE).write_text("{}")
        self.assertFalse(vs.is_archive_layout(problem_dir))


if __name__ == "__main__":
    unittest.main()
