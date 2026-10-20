"""Published manuscript, README and static website must expose identical measurements."""
import csv
import html
import json
import re
import statistics
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'public'


def numbers(text):
    return re.findall(r'\d+(?:\.\d+)?', text)


class PublicResultTests(unittest.TestCase):
    def test_all_eighty_results_match_manuscript_html_readme_and_csv(self):
        study = json.loads((PUBLIC / 'data/study.json').read_text())
        self.assertEqual(len(study['models']), 10)
        page = (PUBLIC / 'index.html').read_text().split('<!-- BEGIN RESULT_ROWS -->')[1].split('<!-- END RESULT_ROWS -->')[0]
        html_rows = re.findall(r'<tr>.*?</tr>', page, re.S)
        readme = (ROOT / 'README.md').read_text().split('<!-- BEGIN RESULT_TABLE -->')[1].split('<!-- END RESULT_TABLE -->')[0]
        md_rows = [line for line in readme.splitlines() if line.startswith('|')][2:]
        with zipfile.ZipFile(PUBLIC / 'author-source.zip') as archive:
            table = archive.read('sections/table_profile.tex').decode()
        tex_rows = [line for line in table.splitlines() if line.count(' & ') == 8 and line.split(' & ')[0] in ['Astra','Sol','Fable','Qwen-Max','Gemini 3.8','Qwen-397B','Qwen-27B','Opus~5.5','DeepSeek-1017','Gemini 4 Argon']]
        with (PUBLIC / 'data/model-comparison.csv').open() as source:
            csv_rows = list(csv.DictReader(source))
        self.assertEqual([len(html_rows), len(md_rows), len(tex_rows), len(csv_rows)], [10,10,10,80])
        for i, model in enumerate(study['models']):
            web_cells = re.findall(r'<td>(.*?)</td>', html_rows[i], re.S)
            md_cells = [html.unescape(s).strip() for s in md_rows[i].split('|')[2:-1]]
            tex_cells = tex_rows[i].split(' & ')[1:]
            for j, expected in enumerate(model['scores']):
                with self.subTest(model=model['name'], metric=j):
                    self.assertEqual(numbers(web_cells[j]), numbers(expected))
                    self.assertEqual(numbers(md_cells[j]), numbers(expected))
                    self.assertEqual(numbers(tex_cells[j]), numbers(expected))
                    row = csv_rows[i*8+j]
                    self.assertEqual(row['model'], model['name'])
                    self.assertEqual(row['model_identifier'], model['identifier'])
                    self.assertEqual(numbers(row['value']+' ± '+row['run_sd']), numbers(expected))
                    self.assertEqual(int(row['runs']), model['metric_runs'][j])
                    self.assertTrue(row['source_url'].endswith('#page=8'))

    def test_argon_repeated_statistics_are_run_sd_not_case_spread(self):
        audit = json.loads((PUBLIC / 'data/argon-results-audit.json').read_text())
        self.assertEqual(audit['api_model'], 'gemini-4-argon')
        for metric in audit['main']:
            self.assertEqual(metric['runs'], 5)
            self.assertAlmostEqual(statistics.mean(metric['run_means']), metric['mean'])
            self.assertAlmostEqual(statistics.stdev(metric['run_means']), metric['sd'])
        model = json.loads((PUBLIC / 'data/study.json').read_text())['models'][-1]
        self.assertEqual(model['scores'], audit['scores'])
        self.assertEqual(model['metric_runs'], [5,5,5,5,5,1,1,1])
        self.assertEqual(audit['case_count'],1000)
        self.assertEqual(audit['polygon_case_count'],70)
        self.assertEqual(audit['report_count'],3192)

    def test_aggregate_profiles_and_scoped_logs_are_distinct(self):
        report = json.loads((PUBLIC / 'data/report/metrics.json').read_text())
        study = json.loads((PUBLIC / 'data/study.json').read_text())
        self.assertEqual(len(report['profiles']), 10)
        self.assertEqual(len(report['models']), 9)
        for profile, model in zip(report['profiles'],study['models']):
            self.assertEqual(profile['scores'], model['scores'])
        with zipfile.ZipFile(PUBLIC / 'author-source.zip') as archive:
            intervention = archive.read('sections/table_review.tex').decode()
        for model in ['Opus', 'DeepSeek', 'DS-1017', 'Argon']:
            self.assertNotIn(model, intervention)


if __name__ == '__main__':
    unittest.main()
