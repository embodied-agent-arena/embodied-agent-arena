// Viewport motion enhances native data marks without moving axes or changing values.
(() => {
  'use strict';
  if (!window.IntersectionObserver || !Element.prototype.animate) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const selector = '.figure-button, .report-figure img, .ranking, .benchmark-scroll, .table-scroll, .report-table-scroll, .report-matrix-scroll, .exec-chart, .dc-plot, .dc-heat-scroll';
  const states = new Map();
  let printing = false;
  const permitted = () => !reduced.matches && !document.hidden && !printing;

  function stop(state) {
    for (const animation of state.animations) animation.cancel();
    state.animations.clear();
  }

  function animate(state, element, keyframes, options) {
    const animation = element.animate(keyframes, {
      duration: 850, easing: 'cubic-bezier(.22, 1, .36, 1)', ...options
    });
    state.animations.add(animation);
    animation.onfinish = animation.oncancel = () => state.animations.delete(animation);
  }

  function enter(state) {
    if (!permitted() || state.el.contains(document.activeElement)) return;
    const el = state.el;
    const nativeBars = el.querySelectorAll('.dc-bar');
    const cells = el.querySelectorAll('.dc-heat-cell');
    const points = el.querySelectorAll('.dc-point');
    if (nativeBars.length || cells.length || points.length) {
      nativeBars.forEach((bar, i) => animate(state, bar,
        [{transform: 'scaleY(0)'}, {transform: 'scaleY(1)'}],
        {id: 'arena-native-bar-grow', delay: Math.min(i, 10) * 35, fill: 'backwards'}));
      cells.forEach((cell, i) => animate(state, cell,
        [{clipPath: 'inset(100% 0 0 0)'}, {clipPath: 'inset(0% 0 0 0)'}],
        {id: 'arena-heat-reveal', duration: 650, delay: Math.min(i % 11, 10) * 35, fill: 'backwards'}));
      points.forEach((point, i) => animate(state, point,
        [{opacity: 0}, {opacity: 1}],
        {id: 'arena-point-reveal', duration: 600, delay: i * 35, fill: 'backwards'}));
      el.querySelectorAll('.dc-whisker, .dc-comparison').forEach(mark => animate(state, mark,
        [{opacity: 0}, {opacity: 1}], {id: 'arena-uncertainty-reveal', duration: 300, delay: 550, fill: 'backwards'}));
      return;
    }
    const bars = el.querySelectorAll('.ranking-fill, .report-bar');
    if (bars.length) {
      // Keep axes, exact values and model labels readable while bars unfold.
      bars.forEach((bar, i) => animate(state, bar, [
        {transform: 'scaleX(0)', transformOrigin: 'left center'},
        {transform: 'scaleX(1)', transformOrigin: 'left center'}
      ], {id: 'arena-bar-grow', delay: Math.min(i, 9) * 35, fill: 'backwards'}));
      el.querySelectorAll('.ranking-sd, .ranking-mean').forEach(mark => animate(state, mark,
        [{opacity: 0}, {opacity: 1}], {id: 'arena-mark-reveal', duration: 300, delay: 650, fill: 'backwards'}));
      return;
    }
    // Tables only fade: sticky labels, horizontal scrolling and column alignment stay fixed.
    const figure = el.matches('.figure-button, .report-figure img, .exec-chart');
    animate(state, el, figure ? [
      {opacity: 0, transform: 'translateY(16px)'},
      {opacity: 1, transform: 'translateY(0)'}
    ] : [{opacity: .15}, {opacity: 1}], {id: 'arena-scroll-reveal'});
    el.querySelectorAll('tbody > tr').forEach((row, i) => animate(state, row,
      [{opacity: .25}, {opacity: 1}], {id: 'arena-row-reveal', duration: 550, delay: Math.min(i, 9) * 35, fill: 'backwards'}));
  }

  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const state = states.get(entry.target);
      if (!state) continue;
      if (!entry.isIntersecting) {
        // Re-arm only after leaving view completely; no flicker at the reveal threshold.
        state.entered = false;
        stop(state);
      } else if (entry.intersectionRatio >= .08 && !state.entered) {
        state.entered = true;
        enter(state);
      }
    }
  }, {threshold: [0, .08]});

  function register(el) {
    if (states.has(el) || el.closest('dialog') || el.parentElement?.closest(selector)) return;
    states.set(el, {el, entered: false, animations: new Set()});
    observer.observe(el);
  }

  function refresh(root) {
    if (root.nodeType !== 1) return;
    if (root.matches(selector)) register(root);
    root.querySelectorAll(selector).forEach(register);
  }
  refresh(document.documentElement);
  new MutationObserver(records => {
    // Metric tabs and analysis filters replace their charts after the initial page load.
    records.forEach(record => record.addedNodes.forEach(refresh));
    for (const [el, state] of states) {
      if (!el.isConnected) { stop(state); observer.unobserve(el); states.delete(el); }
    }
  }).observe(document.body, {childList: true, subtree: true});

  function stopAll() { states.forEach(stop); }
  reduced.addEventListener('change', stopAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopAll(); });
  document.addEventListener('focusin', event => {
    states.forEach(state => { if (state.el.contains(event.target)) stop(state); });
  });
  window.addEventListener('beforeprint', () => { printing = true; stopAll(); });
  window.addEventListener('afterprint', () => { printing = false; });
})();
