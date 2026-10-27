import assert from 'node:assert/strict';
import worker from '../scripts/legacy-redirect.mjs';

const origin = 'https://embodied-agent-arena.haojianhuang927.workers.dev';
const base = 'https://embodied-agent-arena.github.io/embodied-agent-arena/';
for (const [path, suffix] of [
  ['/', ''], ['/?analysis=sequence', '?analysis=sequence'], ['index.html', ''],
  ['/geoprobe.html', 'geoprobe.html'], ['/assets/logo-transparent.png', 'assets/logo-transparent.png'],
  ['/data/a%20b.json?download=1', 'data/a%20b.json?download=1'],
  ['//example.org/escape', 'example.org/escape'],
]) {
  for (const method of ['GET', 'HEAD']) {
    const response = worker.fetch(new Request(origin + '/' + path.replace(/^\//, ''), { method }));
    assert.equal(response.status, 301);
    assert.equal(response.headers.get('location'), base + suffix);
  }
}
console.log('Legacy redirects preserve paths and queries on the canonical host (14 checks).');
