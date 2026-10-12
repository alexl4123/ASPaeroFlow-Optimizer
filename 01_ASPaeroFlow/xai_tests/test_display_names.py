"""Navpoint names in the explanation texts (fix names on real-navpoint instances, numbers on grid instances).

    python -m unittest discover -s 01_ASPaeroFlow/xai_tests      (from the repository root)
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
sys.path.insert(0, str(OPT))

from src.aspaeroflow.xai.session import ReplaySession, display_name, display_names  # noqa: E402

TRACE = HERE / "fixtures" / "CE7_JUNE_TRACE"


class DisplayNames(unittest.TestCase):

    def test_rule(self):
        self.assertEqual(display_name("ABGUS_FL300", False, 1), "ABGUS")
        self.assertEqual(display_name("BAKOL", False, 2), "BAKOL")
        self.assertEqual(display_name("EHAM", True, 51), "EHAM")
        self.assertEqual(display_name("GRID_Y3X4_FL300", False, 22), "22")      # grid instances keep numbers
        self.assertEqual(display_name("LFLX_FL300", False, 7), "7")              # not a 5-letter fix

    def test_names_from_an_instance_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "mappings").mkdir()
            (d / "mappings" / "vertex_map.csv").write_text("IDENTIFIER,VERTEX_ID\nEDDB,0\nABGUS_FL300,1\nGRID_Y1X1_FL300,2\n")
            (d / "airports.csv").write_text("Airport_Vertex\n0\n")
            self.assertEqual(display_names(d), {0: "EDDB", 1: "ABGUS", 2: "2"})
            (d / "mappings" / "vertex_map.csv").write_text("IDENTIFIER,VERTEX_ID\nEDDB,0\nGRID_Y1X1_FL300,2\n")
            self.assertIsNone(display_names(d))                  # a grid instance: numbers, airports included
        self.assertIsNone(display_names(None))

    def test_june_fixture_keeps_numbers_and_names_replace_them(self):
        s = ReplaySession(TRACE)
        self.assertIsNone(s.display_names())            # a grid instance: numbers as before
        ex = s.explainer(6)
        self.assertIsNone(ex.names)
        # the reduced fixture holds no candidate paths: describe one path by hand
        path = {"route": [51, 22, 15, 53], "current_route": [51, 22, 14, 53], "rerouted": True,
                "departure_shift": 1, "arrival_shift": 1, "start": 11}
        ex.describe_path = lambda f, i: path
        self.assertIn("flies 51-22-15-53 instead of 51-22-14-53", ex.change_clause(18, 0))
        ex.names = {51: "EHAM", 22: "ABADO", 15: "BAKOL", 14: "CODEX", 53: "LFPG"}
        self.assertIn("flies EHAM-ABADO-BAKOL-LFPG instead of EHAM-ABADO-CODEX-LFPG", ex.change_clause(18, 0))
        from src.aspaeroflow.xai.contrastive import Lock
        self.assertEqual(ex.lock_text(Lock("avoid", (22,))), "no flight of the step passes navpoint ABADO")
        ex.names = None
        del ex.describe_path


if __name__ == "__main__":
    unittest.main()
