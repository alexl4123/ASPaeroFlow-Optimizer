"""Metric CSVs: every value sits under its own system's column (the V2 USA-EAST-COAST TG1 defect).

Run: python -m unittest tests/test_metric_csv_columns.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_benchmark_start_script"))
import merge_benchmark_shards as merge  # noqa: E402
from start_benchmark_caller import metric_rows  # noqa: E402

SYSTEMS = ["A", "B", "C"]


def container(values):
    """values[inst][system] -> final dict (or {} when the system reported nothing)."""
    return {i: {s: [values[i].get(s, {})] for s in SYSTEMS} for i in values}


class TestMetricCsvColumns(unittest.TestCase):

    def test_system_reporting_only_from_the_second_instance_keeps_its_column(self):
        # B reports nothing on the first instance (a timeout without a model), then a value.
        vals = {"i1": {"A": {"OVERLOAD": 1}, "C": {"OVERLOAD": 3}},
                "i2": {"A": {"OVERLOAD": 4}, "B": {"OVERLOAD": 5}, "C": {"OVERLOAD": 6}}}
        head, rows = metric_rows(["i1", "i2"], SYSTEMS, container(vals), "OVERLOAD")
        assert head == ["Instance", "A", "B", "C"]            # system order, not order of appearance
        assert rows == [["i1", 1, -1, 3], ["i2", 4, 5, 6]]     # full rows, -1 where nothing was reported


    def test_system_that_never_reports_has_no_column(self):
        vals = {"i1": {"A": {"OVERLOAD": 1}}, "i2": {"A": {"OVERLOAD": 2}}}
        head, rows = metric_rows(["i1", "i2"], SYSTEMS, container(vals), "OVERLOAD")
        assert head == ["Instance", "A"]
        assert rows == [["i1", 1], ["i2", 2]]


    def test_unchanged_where_every_reporting_system_appears_on_the_first_instance(self):
        vals = {"i1": {"A": {"OVERLOAD": 1}, "B": {"OVERLOAD": 2}},
                "i2": {"A": {"OVERLOAD": 3}}}
        head, rows = metric_rows(["i1", "i2"], SYSTEMS, container(vals), "OVERLOAD")
        assert head == ["Instance", "A", "B"]
        assert rows == [["i1", 1, 2], ["i2", 3, -1]]         # what the old writer produced here too


    def test_provenance_is_harvested_once_and_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            units = Path(tmp) / "units"
            (units / "provenance").mkdir(parents=True)
            (units / "provenance" / "task_1.txt").write_text("time_limit_s=1800\nrun_mip=no\n")
            (units / "provenance" / "task_2.txt").write_text("time_limit_s=3600\nrun_mip=no\n")
            first = merge.harvest_provenance(units)
            assert first == {"time_limit_s": ["1800", "3600"], "run_mip": ["no"]}
            (units / "provenance" / "task_3.txt").write_text("time_limit_s=60\n")
            assert merge.harvest_provenance(units) is first       # cached for the rest of the merge


if __name__ == "__main__":
    unittest.main()
