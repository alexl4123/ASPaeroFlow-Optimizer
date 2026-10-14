"""The answer cache of a replay (xai/answer_cache.py, K35) on the shared 10-flight run: a build in the fixed order is
byte-identical when repeated; a replay with the cache serves cached questions without an explainer, gives every
cached answer exactly as built, computes a question outside the cache, and ignores a cache whose fingerprint does
not match.

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import shared_run  # noqa: E402
from src.aspaeroflow.xai import answer_cache  # noqa: E402
from src.aspaeroflow.xai.contrastive import Lock  # noqa: E402
from src.aspaeroflow.xai.session import ReplaySession  # noqa: E402


class AnswerCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="k35-"))
        cls.folder = cls.tmp / "trace"
        shutil.copytree(shared_run.get()[2], cls.folder)
        cls.cache = answer_cache.build(cls.folder)
        cls.data = answer_cache.to_bytes(cls.cache)
        (cls.folder / answer_cache.FILE).write_bytes(cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_build_is_repeatable_and_covers_the_questions(self):
        self.assertEqual(answer_cache.to_bytes(answer_cache.build(self.folder)), self.data)
        keys = list(self.cache["entries"])
        steps = {int(k.split("|")[1]) for k in keys}
        self.assertTrue(steps)
        for n in steps:
            for q in ("menu", "hotspot", "sectors"):
                self.assertIn(answer_cache.explain_key(n, q), keys)
        self.assertTrue(any(k.startswith("what_if|") for k in keys))
        self.assertTrue(any("|flight|" in k for k in keys))

    def test_replay_serves_the_cache_without_solving(self):
        s = ReplaySession(self.folder)
        self.assertIsNotNone(s._answers)

        def no_explainer(n):
            raise AssertionError("a cached question must not build an explainer")
        s.explainer = no_explainer
        for key, value in list(self.cache["entries"].items())[:40]:
            _, n, rest = key.split("|", 2)
            if key.startswith("explain|"):
                question, flight = rest.split("|")
                got = s.explain(int(n), question, int(flight) if flight else None)
            else:
                got = s.what_if(int(n), rest.split(","))
            self.assertEqual(json.loads(json.dumps(got)), value, key)

    def test_question_outside_the_cache_is_solved(self):
        s = ReplaySession(self.folder)
        n = min(int(k.split("|")[1]) for k in self.cache["entries"])
        menu = s.explain(n, "menu")
        open_locks = [e["lock"] for e in menu["menu"] if e["state"] == "open"]
        if len(open_locks) < 2:
            self.skipTest("no two open requirements in this step")
        pair = open_locks[:2]
        self.assertNotIn(answer_cache.what_if_key(n, pair), self.cache["entries"])
        result = s.what_if(n, pair)
        self.assertIn("question", result)
        self.assertEqual(s.what_if(n, list(reversed(pair))), result)      # order of the locks does not matter

    def test_default_build_has_no_per_flight_entries(self):
        """S8 4a: the default build (the study traces) holds no `flight_navpoints` in its menus and no per-flight
        what-if."""
        for key, value in self.cache["entries"].items():
            if key.startswith("explain|") and key.split("|")[2] == "menu":
                self.assertNotIn("flight_navpoints", value, key)
            if key.startswith("what_if|"):
                for lock in key.split("|")[2].split(","):
                    parsed = Lock.parse(lock)
                    self.assertFalse(parsed.kind == "avoid" and len(parsed.args) > 1, key)

    def test_per_flight_build_only_adds(self):
        """S8 4a: a --per-flight build has every entry of the default build unchanged (a menu gains only
        `flight_navpoints`), and its further entries are exactly the open per-flight navpoints of each step's menu,
        solved after the step's other entries; it is repeatable byte for byte; a replay serves them from the cache."""
        cache = answer_cache.build(self.folder, per_flight=True)
        data = answer_cache.to_bytes(cache)
        self.assertEqual(answer_cache.to_bytes(answer_cache.build(self.folder, per_flight=True)), data)
        old, new = self.cache["entries"], cache["entries"]
        self.assertEqual(cache["fingerprint"], self.cache["fingerprint"])
        for key, value in old.items():
            got = new[key]
            if key.startswith("explain|") and key.split("|")[2] == "menu":
                self.assertIn("flight_navpoints", got)
                got = {k: v for k, v in got.items() if k != "flight_navpoints"}
            self.assertEqual(json.dumps(got, sort_keys=True), json.dumps(value, sort_keys=True), key)
        expected = set()
        for key, value in new.items():
            if key.startswith("explain|") and key.split("|")[2] == "menu":
                n = int(key.split("|")[1])
                expected |= {answer_cache.what_if_key(n, [e["lock"]]) for g in value["flight_navpoints"]
                             for e in g["navpoints"] if e["state"] == "open"}
        self.assertEqual(set(new) - set(old), expected)
        if not expected:
            self.skipTest("no open per-flight navpoint in this run")
        other = self.tmp / "per_flight"
        shutil.copytree(self.folder, other)
        (other / answer_cache.FILE).write_bytes(data)
        s = ReplaySession(other)

        def no_explainer(n):
            raise AssertionError("a cached question must not build an explainer")
        s.explainer = no_explainer
        s._fresh_explainer = no_explainer
        for key in sorted(expected):
            _, n, rest = key.split("|", 2)
            self.assertEqual(json.loads(json.dumps(s.what_if(int(n), rest.split(",")))), new[key], key)

    def test_a_changed_fingerprint_is_ignored(self):
        other = self.tmp / "other"
        shutil.copytree(self.folder, other)
        cache = answer_cache.load(other)
        cache["fingerprint"] = "0" * 64
        (other / answer_cache.FILE).write_bytes(answer_cache.to_bytes(cache))
        self.assertIsNone(ReplaySession(other)._answers)
        (other / answer_cache.FILE).unlink()
        self.assertIsNone(ReplaySession(other)._answers)


if __name__ == "__main__":
    unittest.main()
