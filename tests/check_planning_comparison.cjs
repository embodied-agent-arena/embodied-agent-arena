const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const compare = require('../public/planning-comparison.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../public/data/report/planning-outcomes.json')));
const eps = JSON.parse(fs.readFileSync(path.join(__dirname, '../public/data/report/planning-episodes.json')));
const idx = new Map(eps.map(e => [`${e.model}:${e.taskId}`, e]));
assert.equal(new Set(data.tasks.map(t => t.taskId)).size, 157);
assert.equal(idx.size, 1570);
for (const t of data.tasks) for (const m of data.models) {
  const e = idx.get(`${m}:${t.taskId}`);
  assert.equal(e.recordId, t.recordIds[m]);
  assert.equal(e.outcome, t.outcomes[m]);
  assert.equal(e.outcome === 'success', compare.completes(t, m));
}
const astraOpus = compare.summarize(data.tasks, ['astra', 'opus55'], 10);
assert.deepEqual(astraOpus, {all:157,both:71,'a-only':55,'b-only':5,neither:26,shared:15,universal:8,solo:11});
const audit = JSON.parse(fs.readFileSync(path.join(__dirname, '../public/data/report/planning-comparison-audit.json')));
assert.equal(audit.verifiedEpisodes, eps.length);
assert.deepEqual(audit.models, data.models);
assert.equal(audit.counts.astraCompletedOpusIncomplete, astraOpus['a-only']);
assert.equal(audit.counts.astraAloneAmongTen, astraOpus.solo);
const execution = JSON.parse(fs.readFileSync(path.join(__dirname, '../public/data/execution/complete-index.json')));
let linkedEpisodes = 0;
for (const story of execution.stories) for (const run of story.runs) {
  const episode = idx.get(`${run.modelKey}:${run.taskId}`);
  if (!episode) continue;
  assert(compare.matchesEpisode(run, episode), run.id);
  assert(!compare.matchesEpisode({...run, reportSha256:'different-report'}, episode));
  assert.equal(run.outcome, episode.outcome);
  linkedEpisodes++;
}
assert.equal(linkedEpisodes, 160);
for (const source of [null, ...new Set(data.tasks.map(t => t.benchmark))]) {
  const tasks = data.tasks.filter(t => !source || t.benchmark === source);
  for (const a of data.models) for (const b of data.models.filter(m => m !== a)) {
    const pair = [a,b], counts = compare.summarize(tasks, pair, 10);
    assert.equal(counts.both + counts['a-only'] + counts['b-only'] + counts.neither, tasks.length);
    assert.equal(counts.both + counts['a-only'], tasks.filter(t => t.outcomes[a] === 'success').length);
    assert.equal(counts.both + counts['b-only'], tasks.filter(t => t.outcomes[b] === 'success').length);
    for (const filter of ['all','both','a-only','b-only','neither','shared','universal','solo']) {
      const selected = tasks.filter(t => compare.matches(t, filter, pair, 10));
      assert.equal(selected.length, counts[filter]);
      if (filter === 'solo') assert(selected.every(t => t.successModels.length === 1 && t.successModels[0] === a));
      for (const size of [6,24]) {
        const all = [];
        const first = compare.page(selected, 0, size);
        for (let i=0; i<first.pages; i++) all.push(...compare.page(selected,i,size).tasks);
        assert.deepEqual(all, selected, 'Pagination must neither omit nor duplicate a task');
        assert.equal(compare.page(selected,999,size).index, first.pages-1);
      }
    }
  }
}
console.log('Validated 1,570 outcomes, all 90 model pairs × 7 source selections, all filters, and complete desktop/mobile pagination. Pairwise Astra-only = 55; solo among ten = 11.');
