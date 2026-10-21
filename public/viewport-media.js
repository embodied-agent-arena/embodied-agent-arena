// Play visible evidence, without downloading off-screen recordings.
(() => {
  'use strict';
  const states = new Map();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const selector = 'video, img[data-animation], img[data-preview-src]';
  function allowed(s) {
    const modal = document.querySelector('dialog[open]');
    return s.visible && s.enabled && !s.paused && !document.hidden &&
      (!reduced.matches || s.explicit) && !s.el.closest('[inert]') &&
      (!s.el.closest('dialog') || s.el.closest('dialog').open) && (!modal || modal.contains(s.el));
  }
  function sync(s) {
    const el = s.el, active = allowed(s);
    if (el.tagName !== 'VIDEO') {
      const src = active ? (el.dataset.animation || el.dataset.previewSrc) : s.poster;
      if (src && el.getAttribute('src') !== src) el.src = src;
      return;
    }
    if (active) {
      const src = el.dataset.video || el.dataset.previewSrc || el.dataset.src;
      if (!el.getAttribute('src') && src) el.src = src;
      if (el.paused && s.starting) s.restart = true;
      if (el.paused && !s.starting) {
        s.starting = true;
        el.play().catch(() => {}).finally(() => {
          s.starting = false; const restart = s.restart; s.restart = false;
          if (!allowed(s)) pause(s); else if (restart && el.paused) sync(s);
        });
      }
    } else pause(s);
  }
  function pause(s) {
    if (!s.el.paused) {s.managedPauses++; s.el.pause();}
  }
  const observer = new IntersectionObserver(entries => {
    for (const e of entries) {const s = states.get(e.target); if (s) {s.visible = e.isIntersecting && e.intersectionRatio >= .2; sync(s);}}
  }, {threshold: [0, .2]});
  function register(el) {
    if (states.has(el)) return states.get(el);
    const s = {el, visible:false, enabled:true, paused:false, explicit:false, managedPauses:0, starting:false, poster:el.dataset.poster || el.getAttribute('src')};
    states.set(el,s);
    if (el.tagName === 'VIDEO') {
      el.muted = true; el.loop = true; el.playsInline = true;
      el.addEventListener('pause', () => {
        if (s.managedPauses) s.managedPauses--;
        else if (allowed(s) && !el.ended) s.paused = true;
      });
      el.addEventListener('play', () => {s.paused = false; if (reduced.matches) s.explicit = true;});
    }
    observer.observe(el); return s;
  }
  function scan(node) {if (node.nodeType !== 1) return; if (node.matches(selector)) register(node); node.querySelectorAll(selector).forEach(register);}
  function refresh() {states.forEach(sync);}
  window.ArenaMedia = {
    enable(el, enabled, explicit=false) {const s=register(el); s.enabled=enabled; if(explicit)s.explicit=true; sync(s);},
    toggle(elements, play) {for(const el of elements){const s=register(el);s.paused=!play;if(play)s.explicit=true;sync(s);}},
    refresh
  };
  scan(document.documentElement);
  new MutationObserver(records => {
    for (const record of records) for (const node of record.addedNodes) scan(node);
    for (const [el,s] of states) if (!el.isConnected) {if(el.tagName==='VIDEO')pause(s);observer.unobserve(el);states.delete(el);}
    refresh();
  }).observe(document.documentElement, {childList:true,subtree:true,attributes:true,attributeFilter:['open','inert']});
  document.addEventListener('visibilitychange',refresh);
  reduced.addEventListener('change',()=>{for(const s of states.values())s.explicit=false;refresh();});
})();
