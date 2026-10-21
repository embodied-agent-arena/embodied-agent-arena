"""README-linked resources must reflect the current manuscript and exact case set."""
import csv
import hashlib
import json
import re
import unittest
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'public'


class Cards(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.cards = []
        self.active = None
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'button' and attrs.get('class') == 'study-open':
            self.active = {'case': attrs['data-case'], 'record': attrs.get('data-preview-record')}
            self.cards.append(self.active)
        if tag == 'img' and self.active is not None:
            self.active['image'] = attrs['src']
        if tag == 'video' and self.active is not None:
            self.active['image'] = attrs['poster']
            self.active['video'] = attrs['data-video']
            self.active['muted'] = 'muted' in attrs

    def handle_endtag(self, tag):
        if tag == 'button':
            self.active = None


class PublicResourceTests(unittest.TestCase):
    def test_atlas_exact_case_coverage_and_preview_identity(self):
        static = {c['id']: c for c in json.loads((PUB/'data/case-studies.json').read_text()) if c.get('predictions')}
        comp = json.loads((PUB/'data/case-comparisons.json').read_text())
        groups = {g['id']: g for g in comp['groups']}
        cards = Cards((PUB/'case-atlas.html').read_text()).cards
        self.assertEqual(len(cards), len(static) + len(groups))
        self.assertEqual({c['case'] for c in cards}, set(static) | set(groups))
        for card in cards:
            with self.subTest(case=card['case']):
                if card['case'] in static:
                    self.assertIn(card['image'], static[card['case']]['images'])
                    self.assertEqual(len(static[card['case']]['predictions']), 10)
                elif card['record']:
                    group = groups[card['case']]
                    self.assertIn(card['record'], group['runIds'])
                    run = comp['runs'][card['record']]
                    self.assertEqual(run['identitySha256'], group['identitySha256'])
                    self.assertEqual(card['image'], run['poster'])
                    if run.get('video'):
                        self.assertEqual(card['video'], run['video'])
                        self.assertTrue(card['muted'])
                    self.assertEqual(run['sourceReportSha256'], run['mediaReportSha256'])

    def test_current_figure_bundle_matches_manuscript_source(self):
        manifest = json.loads((PUB/'assets/analysis/figure-manifest.json').read_text())
        with zipfile.ZipFile(PUB/'assets/analysis/editable-figures.zip') as figures, zipfile.ZipFile(PUB/'author-source.zip') as source:
            for file in manifest['files']:
                data = figures.read(file['path'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), file['sha256'])
            for stem in manifest['scopes']:
                if not stem.startswith('figure'):
                    continue
                if re.match(r'figure[3-8]-', stem):
                    self.assertIn(b'Argon', figures.read(stem+'.drawio'))
                elif stem == 'figure-protocol-cases':
                    self.assertNotIn(b'Argon', figures.read(stem+'.drawio'))
                    self.assertIn(b'Fable', figures.read(stem+'.drawio'))
                names = [n for n in source.namelist() if n.endswith('/'+stem+'.pdf')]
                self.assertEqual(len(names), 1)
                self.assertEqual(figures.read(stem+'.pdf'), source.read(names[0]))

    def test_planning_csv_has_every_model_task_pair(self):
        data = json.loads((PUB/'data/report/planning-outcomes.json').read_text())
        with (PUB/'data/report/planning-outcomes.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1570)
        indexed = {(r['task_id'],r['model']): r for r in rows}
        self.assertEqual(len(indexed), 1570)
        for task in data['tasks']:
            for model in data['models']:
                self.assertEqual(indexed[task['taskId'],model]['outcome'], task['outcomes'][model])

    def test_readme_and_document_resource_links_exist(self):
        paths = [ROOT/'README.md', ROOT/'docs/RESULTS.md', ROOT/'docs/RECORDED_TASKS.md', ROOT/'docs/EXECUTION_ANALYSIS.md']
        for path in paths:
            for url in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
                if url.startswith(('https:', 'http:', '#')):
                    continue
                target = path.parent/url.split('#')[0]
                with self.subTest(document=path.name, link=url):
                    self.assertTrue(target.exists(), str(target))


if __name__ == '__main__':
    unittest.main()
