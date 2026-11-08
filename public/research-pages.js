/* Citation actions enhance static links and visible BibTeX. No content fetches. */
(() => {
  'use strict';
  const canonical = document.querySelector('link[rel="canonical"]')?.href || window.location.href;

  function selectCitation(source) {
    source.closest('details')?.setAttribute('open', '');
    source.focus();
    try {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    } catch (_) {
      // The visible citation remains selectable with normal browser controls.
    }
  }

  for (const button of document.querySelectorAll('[data-copy-citation], [data-copy-link]')) {
    const citationId = button.dataset.copyCitation;
    const source = citationId ? document.getElementById(citationId) : null;
    const group = button.closest('.citation-actions, .copy-action');
    const status = group?.querySelector('.copy-status');
    const fallback = group?.querySelector('.copy-fallback');
    if (!group || !status || (citationId && !source)) continue;

    let text;
    try {
      text = source ? source.textContent.trim() : new URL(button.dataset.copyLink, canonical).href;
    } catch (_) {
      continue;
    }
    if (!text) continue;

    button.addEventListener('click', async () => {
      button.disabled = true;
      status.textContent = 'Copying…';
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(text);
        if (fallback) fallback.hidden = true;
        status.textContent = source ? 'BibTeX copied.' : 'Link copied.';
      } catch (_) {
        if (source) {
          selectCitation(source);
          status.textContent = 'Copy unavailable. Use your browser’s copy command on the citation below, or download the .bib file.';
        } else if (fallback) {
          fallback.value = text;
          fallback.hidden = false;
          fallback.focus();
          fallback.select();
          status.textContent = 'Copy unavailable. Copy the selected link below.';
        } else {
          status.textContent = 'Copy unavailable. Use the adjacent permanent link.';
        }
      } finally {
        button.disabled = false;
      }
    });
    button.hidden = false;
  }
})();
