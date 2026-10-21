/* Keep semantic tokens together in static and dynamically rendered text. */
(() => {
  const skip = 'script,style,pre,code,svg,canvas,textarea,select,option,[contenteditable],.sr-only';
  function protect(text) {
    return text
      .replace(/\b(?:GPT[ -]6(?: Astra| Sol)?|Gemini [34](?:\.8)?(?: Argon| Flash)?|(?:Claude )?(?:Opus 5\.5|Fable 5\.1)|Qwen 3\.[58](?: Max| 27B| 397B(?:-A17B)?)?)/g, s => s.replace(/ /g, '\u00a0').replace(/-/g, '\u2011'))
      .replace(/\b(?:DeepSeek|DS|Flash)-1017\b/g, s => s.replace('-', '\u2011'))
      .replace(/\b(?:Polygon|polygon|Box|box|Mask|mask) IoU\b/g, s => s.replace(' ', '\u00a0'))
      .replace(/(\d(?:[.,]\d+)?)\s+±\s+(\d+(?:\.\d+)?)/g, '$1\u00a0±\u00a0$2')
      .replace(/(\d) (cm|mm|ms|s|minutes?|seconds?|tokens?|rounds?|runs?)\b/g, '$1\u00a0$2')
      .replace(/(\d+(?:\.\d+)?%) \/ (\d+(?:\.\d+)?%)/g, '$1\u00a0/\u00a0$2')
      .replace(/ ([↗↓→])/g, '\u00a0$1');
  }
  function apply(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      if (root.parentElement?.closest(skip)) return;
      const next = protect(root.data); if (next !== root.data) root.data = next;
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || root.matches(skip)) return;
    const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node; (node = walk.nextNode());) apply(node);
  }
  function start() {
    apply(document.body);
    new MutationObserver(records => {
      for (const r of records) {
        if (r.type === 'characterData') apply(r.target);
        else for (const node of r.addedNodes) apply(node);
      }
    }).observe(document.body, {subtree: true, childList: true, characterData: true});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
})();
