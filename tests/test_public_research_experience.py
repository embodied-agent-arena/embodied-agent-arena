"""Visual summaries must preserve source populations and static citation fallbacks."""
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


class ResearchPage(HTMLParser):
    def __init__(self, name):
        super().__init__()
        self.depth = 0
        self.measures = {}
        self.families = {}
        self.segments = []
        self.active_measure = None
        self.measure_depth = None
        self.bib_depth = None
        self.bib = []
        self.copy_controls = []
        self.scripts = []
        self.feed((PUBLIC / name).read_text())

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag not in ('meta', 'link', 'img', 'input', 'br', 'hr'):
            self.depth += 1
        if 'data-measure' in attrs:
            self.active_measure = attrs['data-measure']
            self.measure_depth = self.depth
            self.measures[self.active_measure] = {
                'value': int(attrs['data-value']), 'total': int(attrs['data-total'])
            }
        if self.active_measure and attrs.get('style', '').startswith('width:'):
            self.measures[self.active_measure]['width_percent'] = float(attrs['style'].split(':')[1].rstrip('%'))
        if 'data-query-family' in attrs:
            self.families[attrs['data-query-family']] = int(attrs['data-count'])
        if 'query-segment' in attrs.get('class', '').split():
            self.segments.append(int(re.search(r'--queries:(\d+)', attrs['style'])[1]))
        if attrs.get('id') == 'research-bibtex':
            self.bib_depth = self.depth
        if tag == 'button' and any(k in attrs for k in ('data-copy-citation', 'data-copy-link')):
            self.copy_controls.append(attrs)
        if tag == 'script' and 'src' in attrs:
            self.scripts.append(attrs['src'])

    def handle_endtag(self, tag):
        if self.depth == self.measure_depth:
            self.active_measure = self.measure_depth = None
        if self.depth == self.bib_depth:
            self.bib_depth = None
        self.depth -= 1

    def handle_data(self, data):
        if self.bib_depth is not None:
            self.bib.append(data)


class ResearchExperienceTests(unittest.TestCase):
    def test_query_visual_counts_every_family_once(self):
        page = ResearchPage('geoprobe.html')
        # The six families combine ten Table A2 protocols, without counting
        # relative pose again as displacement or dimensions again as scale.
        expected = {'displacement': 12 + 12 + 12 + 4, 'calibration': 24 + 12,
                    'pose': 24, 'depth': 12, 'dimensions': 12, 'scale': 44}
        self.assertEqual(page.families, expected)
        self.assertEqual(page.segments, list(expected.values()))
        self.assertEqual(sum(page.segments), 168)

    def test_findings_visuals_match_retained_episode_and_task_data(self):
        page = ResearchPage('findings.html')
        models = json.loads((PUBLIC / 'data/spatial-recovery/calvin-summary.json').read_text())['models']
        n = sum(m['n'] for m in models)
        errors = sum(m['error_episodes'] for m in models)
        expected = {
            'error-population': (errors, n),
            'resumed-execution': (sum(m['resumed'] for m in models), errors),
            'completed-after-error': (sum(m['completed_after_error'] for m in models), errors),
        }
        outcomes = json.loads((PUBLIC / 'data/report/planning-outcomes.json').read_text())
        tasks = outcomes['tasks']
        groups = [('astra-coverage', {'astra'}), ('pair-coverage', {'astra', 'gemini4argon'}),
                  ('all-model-coverage', set(outcomes['models']))]
        for name, selected in groups:
            expected[name] = (sum(bool(selected.intersection(t['successModels'])) for t in tasks), len(tasks))
        self.assertEqual(set(page.measures), set(expected))
        for name, (value, total) in expected.items():
            with self.subTest(measure=name):
                actual = page.measures[name]
                self.assertEqual((actual['value'], actual['total']), (value, total))
                if name != 'error-population':
                    self.assertAlmostEqual(actual['width_percent'], 100 * value / total, places=6)

    def test_bibtex_and_links_are_available_without_javascript(self):
        expected = (PUBLIC / 'citation.bib').read_text().strip()
        for name in ('geoprobe.html', 'findings.html', 'research-topics.html'):
            with self.subTest(page=name):
                page = ResearchPage(name)
                self.assertEqual(''.join(page.bib).strip(), expected)
                self.assertTrue(page.copy_controls)
                self.assertTrue(all('hidden' in c for c in page.copy_controls))
                self.assertTrue(any(s.startswith('research-pages.js') for s in page.scripts))
                self.assertFalse(any(s.startswith('app.js') for s in page.scripts))


if __name__ == '__main__':
    unittest.main()
