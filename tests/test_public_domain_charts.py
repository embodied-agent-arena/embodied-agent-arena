"""Data contracts for native web versions of the retained paper result figures."""
import json, math, unittest, xml.etree.ElementTree as ET
from pathlib import Path
U=Path(__file__).resolve().parents[1]/'public'
D=json.loads((U/'data/report/domain-charts.json').read_text())
class NativeChartDataTests(unittest.TestCase):
 def test_spatial_exact_means_sd_and_counts(self):
  source=json.loads((U/'data/spatial-recovery/spatial-summary.json').read_text())
  self.assertEqual(D['models'],source['models'])
  for panel in ['a','b']:
   for native,original in zip(D['spatial']['panels'][panel],source['panels'][panel]):
    self.assertEqual(native['results'],original['results'])
    self.assertEqual(native['category'],original['category'])
    self.assertGreater(len(native['definition']),30)
  self.assertEqual(D['spatial']['sources'],source['sources'])
 def test_original_heatmap_ramp(self):
  cells=ET.parse(U/'assets/analysis/figure5-spatial.drawio').findall('.//mxCell');ramp=[]
  for c in cells:
   g=c.find('mxGeometry');st=dict(x.split('=',1)for x in c.get('style','').split(';')if'='in x)
   # Identify the thin, unoutlined colorbar cells independently of canvas position.
   if g is not None and st.get('strokeColor')=='none' and 0<float(g.get('width',0))<4 and float(g.get('height',0))>2*float(g.get('width',0)):ramp.append((float(g.get('x')),st['fillColor']))
  self.assertEqual(len(ramp),100)
  self.assertEqual(D['heatRamp'],[color for _,color in sorted(ramp)])
 def test_planning_partitions_every_task_once(self):
  tasks=json.loads((U/'data/report/planning-outcomes.json').read_text())['tasks']
  self.assertEqual(D['planning']['total'],len(tasks))
  for r in D['planning']['models']:
   self.assertEqual(r['count'],sum(r['model'] in t['successModels'] for t in tasks))
   self.assertAlmostEqual(r['value'],r['count']/157*100)
  hist=[sum(len(t['successModels'])==i for t in tasks) for i in range(11)]
  self.assertEqual(D['planning']['histogram'],hist)
  self.assertEqual(sum(hist),157);self.assertEqual(hist[0],15);self.assertEqual(hist[10],8)
 def test_calvin_all_100_and_nested_counts(self):
  self.assertEqual(D['manipulation'],json.loads((U/'data/spatial-recovery/calvin-summary.json').read_text()))
  self.assertEqual(sum(r['n'] for r in D['manipulation']['models']),100)
  for r in D['manipulation']['models']:
   self.assertLessEqual(r['completed_after_error'],r['resumed']);self.assertLessEqual(r['resumed'],r['error_episodes'])
   self.assertEqual(r['reached'],sorted(r['reached'],reverse=True))
 def test_scatter_and_regions_match_published_precision(self):
  for r in D['geometry']['trace']:
   self.assertEqual(r['cases'],56);self.assertTrue(0<=r['success']<=100);self.assertGreater(r['rmse'],0)
  metrics=json.loads((U/'data/report/metrics.json').read_text())
  for r in D['affordance']['regions']:
   original=next(x for x in metrics['profiles'] if x['model']==r['model'])
   self.assertAlmostEqual(r['polygon']/100,original['means'][5]);self.assertEqual(r['cases'],70)
  self.assertEqual(sum(r['polygon']>r['box'] for r in D['affordance']['regions']),9)
 def test_step_traces_keep_recorded_rounds(self):
  for s in D['progress']:
   self.assertEqual([p['round'] for p in s['rounds']],list(range(1,len(s['rounds'])+1)))
  ends={r['model']:r['rounds'][-1]['score'] for r in D['progress']}
  self.assertEqual(ends,{'opus55':1,'gemini4argon':1,'deepseek':.5})
 def test_native_mounts_replace_result_images(self):
  app=(U/'app.js').read_text();self.assertIn('data-domain-chart="${d.id}"',app)
  self.assertNotIn('assets/${d.id}.webp',app)
  report=(U/'technical-report.html').read_text()
  self.assertIn('data-domain-chart="progress"',report);self.assertIn('data-domain-chart="manipulation"',report)
  self.assertNotIn('<img src="assets/manipulation.webp',report)
  for page in ['index.html','technical-report.html']:
   s=(U/page).read_text();self.assertIn('domain-charts.js',s);self.assertIn('domain-charts.css',s)
if __name__=='__main__':unittest.main()
