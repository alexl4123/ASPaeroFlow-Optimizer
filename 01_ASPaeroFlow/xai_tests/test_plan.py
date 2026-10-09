"""Plan data of a recorded run (xai/plan.py, K01) on the June CE-7x7 trace.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai.plan import build_plan, clock, plan_facts, route_of, write_plan  # noqa: E402

TRACE = HERE / "fixtures" / "CE7_JUNE_TRACE"


class Plan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(TRACE)
        cls.by = {x["flight"]: x for x in cls.plan["flights"]}

    def test_units_and_clock(self):
        self.assertEqual(self.plan["run"]["minutes_per_step"], 60)
        self.assertEqual(clock(11, 60, 24), "11:00")
        self.assertEqual(clock(43, 15, 96), "10:45")
        self.assertEqual(clock(97, 15, 96), "+1d 00:15")

    def test_step6_flights(self):
        f18, f19 = self.by[18], self.by[19]
        self.assertEqual(f18["filed"]["route"], [51, 22, 14, 53])
        self.assertEqual(f18["final"]["route"], [51, 22, 15, 53])
        self.assertTrue(f18["rerouted"])
        self.assertEqual([(m["iteration"], m["departure_shift"], m["arrival_shift"]) for m in f18["moved_by"]], [(6, 1, 1)])
        self.assertEqual(f18["largest_hotspot"], 6)
        self.assertEqual(f18["moved_by"][0]["hotspot"], {"sector": 21, "time": 11})
        self.assertIsNone(f18["moved_by"][0]["cause"])  # the June trace has no hotspot.flights
        self.assertFalse(f19["rerouted"])
        self.assertEqual(f19["arrival_delay"], {"steps": 13, "minutes": 780})

    def test_kpis_match_the_run(self):
        k = self.plan["kpis"]
        self.assertEqual(k["arrival_delay"], {"filed": 0, "final": 340})
        self.assertEqual(k["changed_flights"]["final"], 36)   # the optimizer's REROUTE counts delays too
        self.assertEqual(k["flights_rerouted"], 17)
        self.assertEqual(sum(x["arrival_delay"]["steps"] for x in self.plan["flights"]), 340)

    def test_checks_pass(self):
        self.assertEqual({c["name"]: c["ok"] for c in self.plan["checks"]},
                         {"chain": True, "rotation": True, "no_earlier_departure": True, "kpis": True})
        self.assertTrue(all(t["gap"] >= 1 for t in self.plan["turnarounds"]))

    def test_route_of_drops_repeats(self):
        self.assertEqual(route_of({3: 5, 1: 4, 2: 4, 4: 6}), [4, 5, 6])

    def _copy(self, tmp):
        dst = Path(tmp) / "t"
        shutil.copytree(TRACE, dst, ignore=shutil.ignore_patterns("lp"))
        return dst

    def test_broken_rotation_is_reported(self):
        """An edited copy: the second of two consecutive legs that no step moved departs from another navpoint."""
        pair = next(t for t in self.plan["turnarounds"]
                    if not self.by[t["previous"]]["changed"] and not self.by[t["next"]]["changed"])
        nxt = self.by[pair["next"]]
        dep, origin = nxt["filed"]["departure"], nxt["origin"]
        other = next(n for n in nxt["filed"]["route"] if n != origin)
        with tempfile.TemporaryDirectory() as tmp:
            dst = self._copy(tmp)
            path = dst / "CENTRAL-EUROPE-7x7" / "flights.csv"
            rows = path.read_text().splitlines()
            out = [rows[0]]
            for r in rows[1:]:
                f, n, t = r.split(",")
                if int(float(f)) == nxt["flight"] and int(float(t)) == dep:
                    n = str(other)
                out.append(",".join([f, n, t]))
            path.write_text("\n".join(out) + "\n")
            plan = build_plan(dst)
        rotation = {c["name"]: c for c in plan["checks"]}["rotation"]
        self.assertFalse(rotation["ok"])
        self.assertIn((pair["previous"], pair["next"], other),
                      [(v["previous"], v["next"], v["departs_from"]) for v in rotation["violations"]])

    def test_rejected_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = self._copy(tmp)
            lines = (dst / "trace.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            rejected = dict(rec, iteration=rec["iteration"], accepted=False,
                            flight_changes={"18": {"id": 18, "old_flight": {"1": 1}, "new_flight": {"2": 2}}})
            (dst / "trace.jsonl").write_text("\n".join([json.dumps(rejected)] + lines) + "\n")
            plan = build_plan(dst)
        self.assertEqual({c["name"]: c["ok"] for c in plan["checks"]}["chain"], True)
        self.assertEqual(plan["kpis"]["arrival_delay"]["final"], 340)

    def test_lp_file_is_valid_asp(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, lp = write_plan(TRACE, out_dir=Path(tmp))
            text = lp.read_text()
            self.assertEqual(sum(1 for l in text.splitlines() if l.startswith("plan_final(")), 60)
            import clingo
            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], text + "#show plan_final_at/3.")
            ctl.ground([("base", [])])
            with ctl.solve(yield_=True) as h:
                model = next(iter(h))
                self.assertIn("plan_final_at(18,13,15)", [str(s) for s in model.symbols(shown=True)])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, "-m", "src.aspaeroflow.xai.plan", str(TRACE), "--out", tmp],
                               cwd=OPT, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("checks: all ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
