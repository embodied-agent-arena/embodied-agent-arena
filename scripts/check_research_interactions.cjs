// Exercise real citation handlers against allowed, denied and absent clipboard APIs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const sourceCode = fs.readFileSync(path.join(root, 'public/research-pages.js'), 'utf8');
const bibtex = fs.readFileSync(path.join(root, 'public/citation.bib'), 'utf8').trim();

function control(dataset) {
  const status = {textContent: ''};
  const fallback = {hidden: true, value: '', focus() { this.focused = true; }, select() { this.selected = true; }};
  const group = {querySelector(selector) { return selector === '.copy-status' ? status : fallback; }};
  return {dataset, hidden: true, disabled: false, status, fallback,
    closest() { return group; }, addEventListener(type, handler) { this[type] = handler; }};
}

function setup(clipboard, controls) {
  const details = {setAttribute(name) { this[name] = true; }};
  const citation = {textContent: `\n${bibtex}\n`, closest() { return details; }, focus() { this.focused = true; }};
  const selected = [];
  const document = {
    querySelector() { return {href: 'https://example.test/research/findings.html'}; },
    querySelectorAll() { return controls; },
    getElementById(id) { return id === 'research-bibtex' ? citation : null; },
    createRange() { return {selectNodeContents(node) { selected.push(node); }}; },
  };
  vm.runInNewContext(sourceCode, {document, navigator: {clipboard}, URL,
    window: {location: {href: 'https://example.test/preview?draft=1'},
      getSelection() { return {removeAllRanges() {}, addRange() {}}; }}});
  return {citation, details, selected};
}

(async () => {
  const copied = [];
  const link = control({copyLink: '#execution-recovery'});
  const bib = control({copyCitation: 'research-bibtex'});
  setup({writeText: async text => copied.push(text)}, [link, bib]);
  assert.equal(link.hidden, false);
  await link.click();
  await bib.click();
  assert.deepEqual(copied, ['https://example.test/research/findings.html#execution-recovery', bibtex]);
  assert.equal(link.status.textContent, 'Link copied.');
  assert.equal(bib.status.textContent, 'BibTeX copied.');

  const denied = control({copyLink: 'findings.html#planning-complementarity'});
  setup({writeText: async () => { throw new Error('Permission denied'); }}, [denied]);
  await denied.click();
  assert.equal(denied.disabled, false);
  assert.equal(denied.fallback.hidden, false);
  assert.equal(denied.fallback.value, 'https://example.test/research/findings.html#planning-complementarity');
  assert(denied.fallback.focused && denied.fallback.selected);
  assert.match(denied.status.textContent, /Copy unavailable/);

  const unavailable = control({copyCitation: 'research-bibtex'});
  const noClipboard = setup(undefined, [unavailable]);
  await unavailable.click();
  assert(noClipboard.details.open && noClipboard.citation.focused);
  assert.deepEqual(noClipboard.selected, [noClipboard.citation]);
  assert.equal(noClipboard.citation.textContent.trim(), bibtex);
  assert.match(unavailable.status.textContent, /download the \.bib file/);
  assert.equal(unavailable.disabled, false);

  let resolveWrite;
  const pending = control({copyLink: '#citation'});
  setup({writeText: () => new Promise(resolve => { resolveWrite = resolve; })}, [pending]);
  const operation = pending.click();
  assert.equal(pending.disabled, true);
  assert.equal(pending.status.textContent, 'Copying…');
  resolveWrite();
  await operation;
  assert.equal(pending.disabled, false);
  assert.equal(pending.status.textContent, 'Link copied.');

  const missing = control({copyCitation: 'absent'});
  const valid = control({copyLink: '#scope'});
  setup({writeText: async () => {}}, [missing, valid]);
  assert.equal(missing.hidden, true);
  assert.equal(missing.click, undefined);
  await valid.click();
  assert.equal(valid.status.textContent, 'Link copied.');
  console.log('Research citation interactions: exact canonical links/BibTeX, denied and absent clipboard fallbacks, pending state and missing-source isolation passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
