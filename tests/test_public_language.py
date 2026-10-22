"""Public copy stays in English; original experimental evidence remains intact."""
import hashlib
import html
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'public'
HAN = re.compile(r'[\u3400-\u9fff]')


def strings(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, path + (str(index),))
    elif isinstance(value, str):
        yield path, value


class PublicLanguageTests(unittest.TestCase):
    def test_pages_assets_and_documentation_have_no_chinese_copy(self):
        paths = [ROOT / 'README.md', *ROOT.glob('docs/**/*.md')]
        paths += [p for p in PUB.rglob('*') if p.suffix in {'.html', '.js', '.css', '.svg', '.md', '.csv'}]
        for path in paths:
            with self.subTest(file=str(path.relative_to(ROOT))):
                text = html.unescape(path.read_text())
                # Include strings represented by Unicode escapes in generated JS.
                text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), text)
                self.assertIsNone(HAN.search(text))

    def test_public_data_has_no_untranslated_display_fields(self):
        for path in PUB.rglob('*.json'):
            if path.name == 'source-instructions.json':
                continue  # Linked source download; never rendered as page copy.
            for key, value in strings(json.loads(path.read_text())):
                if not HAN.search(value):
                    continue
                # Verbatim model code/errors are experimental evidence, opened
                # explicitly in the trace viewer or downloaded as raw records.
                raw_trace = path.parent.name in {'execution', 'case-records'}
                self.assertTrue(raw_trace and len(key) == 3 and key[0] == 'steps'
                                and key[-1] in {'code', 'error'}, (str(path), key))

    def test_translations_retain_task_identity_and_original_prompts(self):
        cases = {c['id']: c for c in json.loads((PUB / 'data/case-studies.json').read_text())}
        originals = json.loads((PUB / 'data/source-instructions.json').read_text())['cases']
        self.assertEqual(len(originals), 30)
        for key, original in originals.items():
            with self.subTest(case=key):
                case = cases[key]
                self.assertEqual(case['taskId'], original['taskId'])
                digest = hashlib.sha256(original['question'].encode()).hexdigest()
                self.assertEqual(digest, original['questionSha256'])
                self.assertEqual(digest, case['originalInstruction']['sha256'])
                self.assertEqual(case['originalInstruction']['id'], key)
                self.assertEqual(case['originalInstruction']['language'], 'zh-CN')
                self.assertTrue((PUB / case['originalInstruction']['url']).is_file())
                self.assertIn('English translation', case['instructionKind'])
                self.assertIsNone(HAN.search(case['question'] + case['shortTask']))
                if 'vlm_depth-' in key:
                    original_points = next(line for line in original['question'].splitlines() if line.startswith('[['))
                    self.assertIn(original_points, case['question'])


if __name__ == '__main__':
    unittest.main()
