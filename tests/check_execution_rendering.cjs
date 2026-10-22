// Regression: round-summary records used to throw before rendering any decisions.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const pub = path.resolve(__dirname, '../public');
const source = fs.readFileSync(path.join(pub, 'execution-analysis.js'), 'utf8');
const context = {document: {getElementById: () => ({})}};
vm.createContext(context);
vm.runInContext(source.split('  function playbackLabel()')[0] +
  'globalThis.normalizeTrace = normalizeTrace;globalThis.hasSceneAnimation=hasSceneAnimation;globalThis.outcome=outcome;})();', context);
const index = JSON.parse(fs.readFileSync(path.join(pub, 'data/execution/complete-index.json')));
const health = JSON.parse(fs.readFileSync(path.join(pub, 'data/report/case-execution-health.json')));
let records = 0, summaries = 0;
for (const story of index.stories) {
  assert.equal(story.runs.length, 10);
  assert.equal(new Set(story.runs.map(r => r.taskId)).size, 1);
  for (const run of story.runs) {
    if(run.mediaKind==='Animated execution log') assert.equal(context.hasSceneAnimation(run),false);
    const issue=health.runs['record-'+run.recordId];
    if(issue) assert.equal(context.outcome({...run,executionReview:issue}),'Rerun required');
    const input = JSON.parse(fs.readFileSync(path.join(pub, run.detailUrl)));
    const detail = context.normalizeTrace(input, run);
    assert.equal(detail.steps.length, input.steps.length);
    assert.equal(detail.success, input.success);
    assert.deepEqual(JSON.parse(JSON.stringify(detail.progress)), input.progress || run.progress);
    if (input.steps.some(s => !s.event)) summaries++;
    for (let i = 0; i < detail.steps.length; i++) {
      const step = detail.steps[i], original = input.steps[i];
      assert(step.event.kind && step.event.label, run.id);
      assert(Array.isArray(step.calls));
      // Normalization adds labels, never measurements or invented tool calls.
      for (const field of ['score', 'goalsMet', 'goalsTotal', 'at', 'round']) {
        assert.equal(step[field], original[field], `${run.id}: ${field}`);
      }
      assert.equal(step.calls.length, (original.calls || []).length);
    }
    for (const m of detail.milestones || []) {
      assert(m.index >= 0 && m.index < detail.steps.length);
      assert.equal(m.at, detail.steps[m.index].at);
    }
    records++;
  }
}
assert(summaries > 0);
console.log(`Validated ${records} exact-task episodes, including ${summaries} round summaries. No inferred actions or scores.`);
