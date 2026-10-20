"""The visible overview must preserve exact task, model, and execution identity."""
import hashlib
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "public"


class Cells(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.cells = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "button" and "overview-cell" in attrs.get("class", "").split():
            self.cells.append(attrs)


class OverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PUBLIC / "data/report/overview-provenance.json").read_text())
        cls.data = json.loads((PUBLIC / "data/case-comparisons.json").read_text())
        cls.html = (PUBLIC / "index.html").read_text()
        cls.cells = {c["data-scene"]: c for c in Cells(cls.html).cells}

    def test_every_column_is_one_native_task_across_all_models(self):
        m = self.manifest
        self.assertEqual(len(self.cells), 99)
        self.assertEqual(len(m["columns"]), 11)
        self.assertEqual(len({c["taskId"] for c in m["columns"]}), 11)
        for col, definition in enumerate(m["columns"], 1):
            tiles = [c for c in m["tiles"] if c["column"] == col]
            self.assertEqual([c["modelKey"] for c in tiles], m["rowOrder"])
            self.assertEqual({c["taskId"] for c in tiles}, {definition["taskId"]})
            self.assertEqual({c["identitySha256"] for c in tiles}, {definition["identitySha256"]})
            self.assertEqual(len({c["display_instruction"] for c in tiles}), 1)

    def test_each_click_and_recording_resolves_to_its_own_execution(self):
        for cell in self.manifest["tiles"]:
            with self.subTest(cell=cell["id"]):
                run = self.data["runs"]["record-" + cell["recordId"]]
                button = self.cells[cell["id"]]
                self.assertEqual(button["data-case"], cell["groupId"])
                self.assertEqual(int(button["data-model"]), cell["row"] - 1)
                self.assertEqual(button["data-record"], run["recordId"])
                self.assertEqual(cell["taskId"], run["taskId"])
                self.assertEqual(cell["modelKey"], run["modelKey"])
                self.assertEqual(cell["reportSha256"], run["sourceReportSha256"])
                self.assertEqual(cell["outcome"], run["outcome"])
                is_footage = run["mediaKind"] != "Animated execution log"
                self.assertEqual(cell["sourceVideo"], run["video"] if is_footage else None)
                self.assertEqual(cell["sourceAnimation"], run["animation"] if is_footage else None)
                media = cell["sourceVideo"] or cell["sourceAnimation"]
                if media:
                    self.assertEqual(cell["mediaReportSha256"], cell["reportSha256"])
                    self.assertEqual(hashlib.sha256((PUBLIC / media).read_bytes()).hexdigest(),
                                     cell["mediaSha256"])
                else:
                    self.assertIsNone(cell["poster"])
                    self.assertIn("No video", button["aria-label"])
                    self.assertIn("log-only", button["class"])

    def test_old_mixed_task_montage_is_no_longer_displayed(self):
        self.assertNotIn("reference-arena", self.html)
        self.assertNotIn("reference-arena.gif", (PUBLIC.parent / "README.md").read_text())
        self.assertNotIn('id="execution-video"', self.html)


if __name__ == "__main__":
    unittest.main()
