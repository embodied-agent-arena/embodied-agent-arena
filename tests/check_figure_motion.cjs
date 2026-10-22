// Verify replay, accessibility and dynamic charts without a browser dependency.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('public/figure-motion.js', 'utf8');
class Element {
  constructor(selector, children = []) {
    this.selector = selector; this.children = children; this.nodeType = 1;
    this.isConnected = true; this.parentElement = null; this.calls = [];
    this.style = {width: '71.6%'}; this.textContent = '71.6 ± 0.7';
  }
  matches(selector) { return selector.split(', ').includes(this.selector); }
  closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector); }
  contains(element) { return element === this || this.children.some(child => child.contains(element)); }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  animate(keyframes, options) {
    const animation = {keyframes, options, cancelled: false, cancel() {this.cancelled = true; this.oncancel?.();}};
    this.calls.push(animation); return animation;
  }
}
const bars = [new Element('.ranking-fill'), new Element('.ranking-fill')];
const marks = [new Element('.ranking-sd'), new Element('.ranking-mean')];
const ranking = new Element('.ranking', [...bars, ...marks]);
const figure = new Element('.figure-button');
const rows = [new Element('tbody > tr'), new Element('tbody > tr')];
const table = new Element('.benchmark-scroll', rows);
const root = new Element('html', [ranking, figure, table]);
const doc = new EventTarget(); doc.documentElement = root; doc.body = root; doc.hidden = false;
const win = new EventTarget(), reduced = new EventTarget(); reduced.matches = false;
const observed = new Set(); let intersect, mutate;
win.IntersectionObserver = class {
  constructor(callback) {intersect = callback;}
  observe(el) {observed.add(el);} unobserve(el) {observed.delete(el);}
};
vm.runInNewContext(source, {window: win, document: doc, Element, matchMedia: () => reduced,
  IntersectionObserver: win.IntersectionObserver,
  MutationObserver: class {constructor(callback) {mutate = callback;} observe() {}}
});
const visit = (el, ratio) => intersect([{target: el, isIntersecting: ratio > 0, intersectionRatio: ratio}]);
const total = el => el.calls.length;
assert.equal(observed.size, 3);
assert.equal(total(bars[0]), 0, 'off-screen content must not animate');
visit(ranking, .02); assert.equal(total(bars[0]), 0);
visit(ranking, .2); assert.equal(total(bars[0]), 1);
assert.equal(total(ranking), 0, 'keep scores, labels and axes stationary and readable');
assert.equal(bars[0].style.width, '71.6%'); assert.equal(bars[0].textContent, '71.6 ± 0.7');
assert.equal(bars[1].calls[0].options.delay, 35);
visit(ranking, .02); visit(ranking, .2);
assert.equal(total(bars[0]), 1, 'crossing the reveal threshold must not replay in view');
visit(ranking, 0); assert(bars[0].calls[0].cancelled);
visit(ranking, .2); assert.equal(total(bars[0]), 2, 're-entering from either direction replays');
visit(figure, .2); assert.equal(total(figure), 1);
visit(table, .2); assert.equal(total(rows[0]), 1);
assert(table.calls[0].keyframes.every(frame => !('transform' in frame)), 'do not move sticky table columns');

reduced.matches = true; reduced.dispatchEvent(new Event('change'));
assert(figure.calls[0].cancelled); assert(bars[0].calls[1].cancelled);
visit(figure, 0); visit(figure, 1); assert.equal(total(figure), 1);
reduced.matches = false; reduced.dispatchEvent(new Event('change'));
visit(figure, 0); visit(figure, 1); assert.equal(total(figure), 2);
doc.hidden = true; doc.dispatchEvent(new Event('visibilitychange')); assert(figure.calls[1].cancelled);
visit(figure, 0); visit(figure, 1); assert.equal(total(figure), 2);
doc.hidden = false; win.dispatchEvent(new Event('beforeprint'));
visit(figure, 0); visit(figure, 1); assert.equal(total(figure), 2);
win.dispatchEvent(new Event('afterprint'));
doc.activeElement = figure; visit(figure, 0); visit(figure, 1); assert.equal(total(figure), 2, 'focused figure stays visible');
doc.activeElement = null;

const next = new Element('.ranking', [new Element('.ranking-fill')]);
ranking.isConnected = false; mutate([{addedNodes: [next]}]);
assert(observed.has(next)); assert(!observed.has(ranking));
visit(next, 1); assert.equal(total(next.children[0]), 1, 'replacement metric chart animates');
const focus = new Event('focusin'); Object.defineProperty(focus, 'target', {value: next});
doc.dispatchEvent(focus); assert(next.children[0].calls[0].cancelled);
vm.runInNewContext(source, {window: {}, Element}); // No animation API: original page remains usable.
console.log('Figure motion: viewport replay, stable scores/layout, dynamic tabs, reduced motion, focus, print and cleanup passed.');
