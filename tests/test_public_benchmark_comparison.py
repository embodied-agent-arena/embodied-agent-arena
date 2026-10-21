"""The homepage comparison must agree with the released paper's Table 1."""
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest
import zipfile

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


class Comparison(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.active = False
        self.rows = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'table' and attrs.get('id') == 'benchmark-coverage':
            self.active = True
        if self.active and tag == 'tr':
            self.rows.append([])
        if self.active and tag == 'td':
            self.rows[-1].append(attrs.get('data-scale', attrs.get('data-coverage')))

    def handle_endtag(self, tag):
        if tag == 'table':
            self.active = False


class BenchmarkComparisonTests(unittest.TestCase):
    def test_homepage_matches_released_manuscript_table(self):
        with zipfile.ZipFile(PUBLIC / 'author-source.zip') as archive:
            tex = archive.read('sections/table_benchmark_comparison.tex').decode()
        expected = []
        for line in tex.splitlines():
            if '&' not in line or r'\benchyes' not in line:
                continue
            cells = line.split('&')
            scale = re.sub(r'\\textbf\{([^}]+)\}', r'\1', cells[1]).strip()
            expected.append([scale] + [
                'yes*' if '$^{*}$' in cell else 'yes' if r'\benchyes' in cell else 'no'
                for cell in cells[2:]
            ])
        actual = [row for row in Comparison((PUBLIC / 'index.html').read_text()).rows if row]
        self.assertEqual(len(actual), 10)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
