"""Regenerating report tables must not restore superseded science or prose.

All writes occur in a temporary repository. Run with unittest (no pytest needed):
    python3 -m unittest discover -s tests -p test_build_technical_report.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('technical_report_builder', ROOT / 'scripts/build_technical_report.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReportRendererTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'repo'
        self.analysis = Path(self.temp.name) / 'analysis'
        self.analysis.mkdir()
        for name in ('public/data/study.json', 'public/data/report/metrics.json',
                     'public/data/report/planning-episodes.json',
                     'public/data/report/planning-outcomes.json',
                     'public/technical-report.html', 'scripts/templates/technical-report.html'):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        for source, target in [('metrics.json', 'report-metrics.json'),
                               ('planning-episodes.json', 'planning-episodes.json')]:
            shutil.copyfile(self.repo / 'public/data/report' / source, self.analysis / target)
        # Earlier versions silently overwrote these unrelated user-facing assets.
        for name in ('public/index.html', 'docs/EXECUTION_CLAIMS.md',
                     'public/data/report/planning-episodes.csv'):
            path = self.repo / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('Current reviewed content; preserve this byte-for-byte.\n')
        self.page = self.repo / 'public/technical-report.html'

    def snapshot(self):
        return {str(p.relative_to(self.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.repo.rglob('*') if p.is_file()}

    def edit_analysis(self, filename, change):
        path = self.analysis / filename
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def assert_rejected_without_write(self):
        before = self.snapshot()
        with self.assertRaises(builder.AuditError):
            builder.build(self.repo, self.analysis)
        self.assertEqual(self.snapshot(), before)

    def test_only_three_table_bodies_change_and_current_prose_survives(self):
        original = self.page.read_text().replace('</header>',
            '</header><p>Newly reviewed narrative that is absent from the template.</p>', 1)
        stale = builder.TABLE_PATTERN.sub(lambda m: m.group(1) +
            '<tr><td>Obsolete table values 35.7 and 17.8</td></tr>' + m.group(4), original)
        self.page.write_text(stale)
        before = self.snapshot()
        result = builder.build(self.repo, self.analysis)
        after = self.page.read_text()
        strip_rows = lambda text: builder.TABLE_PATTERN.sub(
            lambda m: m.group(1) + '[MEASURED ROWS]' + m.group(4), text)
        self.assertEqual(strip_rows(stale), strip_rows(after))
        self.assertNotIn('Obsolete table values', after)
        self.assertIn('<td>58/157</td>', after)
        self.assertIn('<td>94.3%</td>', after)
        self.assertEqual(result['tablesRefreshed'], 3)
        expected_changes = ['public/technical-report.html']
        self.assertEqual([name for name, digest in before.items()
                          if digest != self.snapshot()[name]], expected_changes)
        second = self.snapshot()
        builder.build(self.repo, self.analysis)
        self.assertEqual(self.snapshot(), second)

    def test_stale_aggregate_rejected_before_any_write(self):
        self.edit_analysis('report-metrics.json', lambda d:
            next(m for m in d['planning']['models'] if m['model'] == 'qwen').update(completed=56))
        self.assert_rejected_without_write()

    def test_stale_report_identity_or_usage_rejected_before_any_write(self):
        def stale(d):
            row = next(r for r in d if r['model'] == 'qwen' and r['benchmark'] == 'HumanCLAW')
            row['reportSha256'] = '0' * 64
            row['tokens'] = 0
        self.edit_analysis('planning-episodes.json', stale)
        self.assert_rejected_without_write()

    def test_old_nine_model_analysis_rejected_before_any_write(self):
        self.edit_analysis('report-metrics.json', lambda d: d['models'].pop())
        self.assert_rejected_without_write()

    def test_mislabeled_source_columns_rejected_before_any_write(self):
        self.page.write_text(self.page.read_text().replace('<th>HumanCLAW</th>', '<th>Another source</th>', 1))
        self.assert_rejected_without_write()

    def test_template_only_bootstraps_a_missing_report(self):
        self.page.unlink()
        before = self.snapshot()
        builder.build(self.repo, self.analysis)
        after = self.page.read_text()
        self.assertIn('Why Tasks Remain Unfinished', after)
        self.assertNotIn('{{SOURCE_RESULTS_ROWS}}', after)
        self.assertNotIn('{{RESOURCE_RESULTS_ROWS}}', after)
        self.assertNotIn('{{PROFILE_RESULTS_ROWS}}', after)
        self.assertEqual(before, {name: digest for name, digest in self.snapshot().items()
                                  if name != 'public/technical-report.html'})


if __name__ == '__main__':
    unittest.main()
