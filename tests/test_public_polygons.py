"""Protect full-allocation scores and same-input polygon comparisons."""
import json
import math
import unittest
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / 'public'


def read(path):
    return json.loads((PUBLIC / path).read_text())


class PolygonComparisonTests(unittest.TestCase):
    def test_qwen_scores_all_seventy_assignments(self):
        audit = read('data/qwen-max-polygon-aggregation.json')
        rows = audit['records']
        self.assertEqual(len(rows), 70)
        self.assertEqual(len({r['taskId'] for r in rows}), 70)
        absent = [r for r in rows if r['materialStatus'] == 'api_unfinished']
        self.assertEqual(len(absent), 11)
        self.assertTrue(all(r['iou'] == 0 for r in absent))
        mean = math.fsum(r['iou'] for r in rows) / 70
        self.assertAlmostEqual(mean, audit['polygonIoU'])
        qwen = next(m for m in read('data/study.json')['models'] if m['name'] == audit['model'])
        self.assertEqual(qwen['scores'][5], f'{mean:.3f}')
        profile = next(m for m in read('data/report/metrics.json')['profiles'] if m['name'] == audit['model'])
        self.assertEqual(profile['scores'], qwen['scores'])
        self.assertAlmostEqual(profile['polygonMinusBox'], profile['means'][5] - profile['means'][4])

    def test_all_polygon_tiles_share_task_and_source_image(self):
        cases = [c for c in read('data/case-studies.json') if c['domainKey'] == 'affordance']
        self.assertEqual(len(cases), 9)
        for case in cases:
            with self.subTest(case=case['id']):
                self.assertEqual(case['representation'], 'polygon')
                self.assertEqual(len(case['predictions']), 10)
                self.assertEqual({p['taskId'] for p in case['predictions']}, {case['taskId']})
                self.assertEqual(len({p['model'] for p in case['predictions']}), 10)
                self.assertTrue((PUBLIC / case['images'][0]).is_file())
                self.assertTrue((PUBLIC / case['reference']['mask']).is_file())
                width, height = case['reference']['imageSize']
                x, y, w, h = case['polygonViewBox']
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x+w, width+.02)
                self.assertLessEqual(y+h, height+.02)
                for pred in case['predictions']:
                    score = pred['metrics']['polygon_mask_iou']
                    self.assertTrue(0 <= score <= 1)
                    if not pred['valid']:
                        self.assertEqual(score, 0)
                    else:
                        self.assertTrue(pred['answer']['polygons'])
                self.assertNotEqual(case['reference'], case['boxComparison']['reference'])
                self.assertNotEqual(case['representation'], case['boxComparison'].get('representation'))


if __name__ == '__main__':
    unittest.main()
