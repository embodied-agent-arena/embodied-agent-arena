"""Execution refreshes retain authored guidance while rebuilding current tables.

The real refresher runs only in a temporary copy of its inputs.
"""
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTRO = re.compile(r'<section id="reading-execution">.*?</section>', re.S)
UPDATED = re.compile(r'<p>Guide updated <time\b.*?</p>', re.S)


class ExecutionGuideRefreshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        for name in (
            'scripts/refresh_execution_views.py',
            'scripts/public_json.py',
            'scripts/execution-story-content.json',
            'public/execution-guide.html',
            'public/data/execution/index.json',
            'public/data/execution/complete-index.json',
        ):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        self.page = self.repo / 'public/execution-guide.html'
        self.index = json.loads((self.repo / 'public/data/execution/complete-index.json').read_text())

    def refresh(self):
        result = subprocess.run(
            [sys.executable, str(self.repo / 'scripts/refresh_execution_views.py')],
            cwd=self.repo, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.page.read_text()

    def assert_current_tables_and_scope(self, output):
        self.assertIn(f'{len(self.index["stories"])} tasks · {self.index["runCount"]} recorded episodes · 10 models', output)
        self.assertEqual(self.index['modelCount'], 10)
        for story in self.index['stories']:
            with self.subTest(story=story['id']):
                section = re.search(r'<section id="' + re.escape(story['id']) + r'">(.*?)</section>', output, re.S)
                self.assertIsNotNone(section)
                names = re.findall(r'<th scope="row">(.*?)</th>', section.group(1))
                self.assertEqual([html.unescape(name) for name in names], [run['model'] for run in story['runs']])
                self.assertEqual(len(names), 10)
                for run in story['runs']:
                    self.assertIn(html.escape(run['detailUrl']), section.group(1))
        self.assertIn('Supplementary RGB-expanded comparison', output)
        self.assertIn('not the adopted 100-episode CALVIN population', output)
        self.assertIn('findings.html#execution-recovery', output)

    def test_authored_intro_date_and_current_ten_model_tables_survive_refresh(self):
        original = self.page.read_text()
        match = INTRO.search(original)
        self.assertIsNotNone(match)
        # A later editorial addition must survive, not just today's known prose.
        intro = match.group(0).replace('</section>', '<p>Retain this later-reviewed population clarification.</p></section>')
        self.page.write_text(original[:match.start()] + intro + original[match.end():])
        updated = UPDATED.search(original)
        self.assertIsNotNone(updated)
        output = self.refresh()
        self.assertTrue(intro in output, 'The authored introduction changed or disappeared during refresh')
        self.assertIn(updated.group(0), output)
        self.assertEqual(output.count('id="reading-execution"'), 1)
        self.assertLess(output.index('id="reading-execution"'), output.index('id="recovery"'))
        self.assertEqual(output.split('<body>', 1)[0], original.split('<body>', 1)[0])
        self.assert_current_tables_and_scope(output)
        self.assertEqual(self.refresh(), output)

    def test_pages_without_authored_intro_still_refresh(self):
        original = self.page.read_text()
        self.page.write_text(UPDATED.sub('', INTRO.sub('', original)))
        output = self.refresh()
        self.assertNotIn('id="reading-execution"', output)
        self.assert_current_tables_and_scope(output)


if __name__ == '__main__':
    unittest.main()
