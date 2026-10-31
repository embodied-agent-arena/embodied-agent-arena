"""Keep the homepage diagnosis attached to the episode actually shown."""
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


class CompletionCards(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.cards = []
        self.card = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article' and 'data-completion-record' in attrs:
            self.card = {'record': attrs['data-completion-record'], 'meters': [], 'links': []}
            self.cards.append(self.card)
        if self.card is None:
            return
        if tag == 'video':
            self.card['video'] = attrs
        if tag == 'meter':
            self.card['meters'].append(attrs)
        if 'data-case' in attrs or 'data-execution-link' in attrs:
            self.card['links'].append(attrs)

    def handle_endtag(self, tag):
        if tag == 'article':
            self.card = None


class CompletionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = CompletionCards((PUBLIC / 'index.html').read_text()).cards
        cls.comparison = json.loads((PUBLIC / 'data/case-comparisons.json').read_text())
        cls.execution = json.loads((PUBLIC / 'data/execution/complete-index.json').read_text())

    def test_media_and_links_identify_the_task_and_separate_recordings(self):
        self.assertEqual(len(self.cards), 2)
        for card in self.cards:
            run = self.comparison['runs'][card['record']]
            self.assertEqual(run['sourceReportSha256'], run['mediaReportSha256'])
            self.assertFalse(run['success'])
            display = run
            if card['video'].get('data-recording-release'):
                display = run['updatedRecording']
                self.assertEqual(display['taskId'], run['taskId'])
                self.assertEqual(display['modelKey'], run['modelKey'])
                self.assertFalse(display['countInEvaluation'])
                self.assertEqual(card['video']['data-recording-release'], display['recordingDate'].replace('-', ''))
            for attr, field in [('data-video', 'video'), ('poster', 'poster')]:
                self.assertEqual(card['video'][attr], display[field])
                self.assertTrue((PUBLIC / display[field]).is_file())
            for link in card['links']:
                if 'data-case' in link:
                    group = next(g for g in self.comparison['groups'] if g['id'] == link['data-case'])
                    selected = self.comparison['runs'][group['runIds'][int(link['data-model'])]]
                    self.assertEqual(selected['id'], run['id'])
                else:
                    story = next(s for s in self.execution['stories'] if s['id'] == link['data-execution-link'])
                    self.assertIn(run['modelKey'], story['default'])
                    self.assertIn(run['recordId'], [r['recordId'] for r in story['runs']])

    def test_progress_and_stop_claims_follow_detailed_logs(self):
        sitting, toolbox = self.cards
        sit = json.loads((PUBLIC / 'data/case-records' / (sitting['record'] + '.json')).read_text())
        codes = [step.get('code', '') for step in sit['steps']]
        self.assertIn('p.sit(', codes[-2])
        self.assertIn('p.stop(', codes[-1])
        detail = json.loads((PUBLIC / 'data/case-records' / (toolbox['record'] + '.json')).read_text())
        self.assertIn('max_total_tokens', detail['stopLimits'])
        self.assertEqual(detail['stats']['modelCalls'], 17)
        time, tokens = toolbox['meters']
        self.assertEqual(float(time['value']), detail['stats']['elapsedSeconds'])
        self.assertEqual(float(time['max']), detail['budgets']['trial_timeout_seconds'])
        self.assertEqual(float(tokens['value']), detail['stats']['tokens'])
        self.assertEqual(float(tokens['max']), detail['budgets']['max_total_tokens'])


if __name__ == '__main__':
    unittest.main()
